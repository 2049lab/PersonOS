"""消费者/调度器单测:有序 + 单飞 + 崩溃回放 + 去重 + 公平上限 + 排空 + 池隔离。

用 MemoryMsgQueue + MemorySessionLock + 假 writer(记录 feed 顺序),不打真 LLM/Redis;
锁的 Redis 变体另有 test_session_lock 覆盖,这里聚焦消费编排逻辑。
"""

from __future__ import annotations

import threading
import time

from personos.app.ingest_worker import Dispatcher, SessionConsumer
from personos.storage.msg_queue import MemoryMsgQueue
from personos.storage.session_lock import MemorySessionLock


def _p(text: str) -> dict:
    """ingest 载荷:一条队列消息 = 一个原子批(这里每批单条,便于验证顺序)。"""
    return {"messages": [{"speaker": "user", "text": text}]}


class FakeWriter:
    """记录 feed_batch/end_session 的调用顺序;可注入 barrier 模拟慢消费(测单飞)。"""

    def __init__(self, log: list, key: tuple, barrier: threading.Event | None = None):
        self._log = log
        self._key = key
        self._barrier = barrier

    def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
        if self._barrier is not None:
            self._barrier.wait(2.0)
        for m in msgs:
            self._log.append((self._key, m.text))

    def end_session(self, *, task_type=None, scenario=""):
        self._log.append((self._key, "<END>"))
        return []


def _make(max_drain=20, barrier=None):
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []

    def writer_for(u, s):
        return FakeWriter(log, (u, s), barrier)

    consumer = SessionConsumer(mq, lock, writer_for, max_drain=max_drain)
    return mq, lock, consumer, log


def test_drain_preserves_fifo_order():
    mq, lock, consumer, log = _make()
    for i in range(5):
        mq.enqueue("u", "s", _p(f"m{i}"))
    rep = consumer.drain_session("u", "s")
    assert rep.locked and rep.applied == 5 and rep.skipped == 0
    assert [t for _, t in log] == ["m0", "m1", "m2", "m3", "m4"]
    assert mq.is_empty("u", "s")
    assert ("u", "s") not in mq.active_sessions()      # 排空后下看板


def test_cursor_advances():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    mq.enqueue("u", "s", _p("b"))
    consumer.drain_session("u", "s")
    assert mq.cursor_get("u", "s") == 2                # 游标推进到最大 seq
    assert [t for _, t in log] == ["a", "b"]


def test_crash_leftover_replayed_via_recover():
    """reserve 后未 ack(模拟 drain 中途崩)→ 下次 drain 先 recover 回放,不丢不乱。"""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    mq.enqueue("u", "s", _p("b"))
    mq.reserve("u", "s")                               # 取到 proc 但不处理(崩)
    mq.reserve("u", "s")
    assert mq.depth("u", "s") == 0 and not mq.is_empty("u", "s")   # 都在途未 ack
    rep = consumer.drain_session("u", "s")             # 应回放 proc 里的 a、b
    assert rep.applied == 2
    assert [t for _, t in log] == ["a", "b"]
    assert mq.is_empty("u", "s")


def test_recover_dedup_when_cursor_already_past():
    """在途消息 seq 已 <= 游标(上次已应用只是没 ack)→ recover 判重投跳过,不重复应用。"""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    env = mq.reserve("u", "s")
    mq.cursor_set("u", "s", env.seq)                   # 模拟"已应用但崩在 ack 前"
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 0 and rep.skipped == 1       # 判重投,跳过
    assert log == []                                   # 未重复 feed
    assert mq.is_empty("u", "s")


def test_max_drain_fairness_leaves_more():
    """单次抢锁至多 max_drain 条,剩余留队列(more=True),不霸占线程。"""
    mq, lock, consumer, log = _make(max_drain=3)
    for i in range(10):
        mq.enqueue("u", "s", _p(f"m{i}"))
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 3 and rep.more is True
    assert mq.depth("u", "s") == 7
    assert ("u", "s") in mq.active_sessions()          # 未排空,仍在看板
    consumer.drain_session("u", "s")                   # 接力,顺序继续
    assert [t for _, t in log][:6] == ["m0", "m1", "m2", "m3", "m4", "m5"]


def test_single_flight_second_drain_bounces():
    """一个线程持锁慢消费时,另一线程 drain 同会话应抢不到锁(locked=False),不并发不乱序。"""
    barrier = threading.Event()
    mq, lock, consumer, log = _make(barrier=barrier)
    mq.enqueue("u", "s", _p("slow"))

    reports: list = []
    t1 = threading.Thread(target=lambda: reports.append(consumer.drain_session("u", "s")))
    t1.start()
    time.sleep(0.1)                                    # 让 t1 抢到锁并卡在 barrier
    rep2 = consumer.drain_session("u", "s")            # t2:同会话,应抢不到
    assert rep2.locked is False and rep2.applied == 0
    barrier.set()
    t1.join(3.0)
    assert reports[0].locked and reports[0].applied == 1


def test_transient_failure_retries_not_lost():
    """瞬时失败(网络抖动)不丢消息:feed 首次抛异常 → 消息留在途,下轮 recover 重放成功。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []
    calls = {"n": 0}

    class FlakyWriter:
        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("模拟瞬时失败(如 MAAS 抖动)")
            for m in msgs:
                log.append(m.text)

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: FlakyWriter(), max_drain=5)
    mq.enqueue("u", "s", _p("重要消息"))
    rep1 = consumer.drain_session("u", "s")            # 首次:feed 抛异常
    assert rep1.applied == 0 and not mq.is_empty("u", "s")   # 未应用,消息仍在途(未丢)
    rep2 = consumer.drain_session("u", "s")            # 二次:recover 重放 → 成功
    assert rep2.applied == 1 and log == ["重要消息"]
    assert mq.is_empty("u", "s")                       # 最终恰好消费一次


def test_poison_message_skipped_after_max_retries():
    """毒消息(必然失败)连续失败达上限 → 跳过(推进游标)+ 解除队头阻塞,后续消息继续消费。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []

    class PoisonWriter:
        """第一条(毒)必崩;其余正常。"""
        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            if msgs[0].text == "毒":
                raise ValueError("必然失败的毒消息")
            for m in msgs:
                log.append(m.text)

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: PoisonWriter(), max_drain=10, max_retries=3)
    mq.enqueue("u", "s", _p("毒"))          # seq=1,毒
    mq.enqueue("u", "s", _p("正常1"))       # seq=2
    mq.enqueue("u", "s", _p("正常2"))       # seq=3

    # 前 max_retries-1 轮:毒消息卡队头,重试,后面消费不到(队头阻塞)
    for _ in range(consumer._max_retries - 1):
        rep = consumer.drain_session("u", "s")
        assert rep.applied == 0 and log == []          # 毒消息挡着,正常消息还没轮到
    # 第 max_retries 轮:判毒 → 跳过 → 后续正常消息被消费
    rep = consumer.drain_session("u", "s")
    assert rep.poisoned == 1
    assert log == ["正常1", "正常2"]                    # 队头解阻,后续消息恰好消费
    assert mq.is_empty("u", "s")
    assert mq.cursor_get("u", "s") == 3                # 游标推进过毒消息(seq=1)与后续


def test_poison_counter_resets_across_distinct_messages():
    """失败计数按 msg_id 独立:一条毒消息被跳过后,不影响另一条消息的重试预算。"""
    mq = MemoryMsgQueue()
    m1, _ = mq.enqueue("u", "s", _p("a"))
    m2, _ = mq.enqueue("u", "s", _p("b"))
    assert mq.mark_failed("u", "s", m1) == 1
    assert mq.mark_failed("u", "s", m1) == 2
    assert mq.mark_failed("u", "s", m2) == 1           # m2 独立计数,不受 m1 影响


def test_heartbeat_renews_lock_during_long_apply():
    """H2 修复:心跳在单条超长 apply 期间续锁(而非处理完才续),防锁中途过期被别副本抢入。"""
    mq = MemoryMsgQueue()

    class SpyLock(MemorySessionLock):
        def __init__(self):
            super().__init__()
            self.renews = 0

        def renew(self, u, s, h):
            self.renews += 1
            return super().renew(u, s, h)

    lock = SpyLock()

    def wf(u, s):
        class W:
            def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
                time.sleep(0.15)                           # 单条超长(> renew 间隔)
            def end_session(self, *, task_type=None, scenario=""):
                pass
        return W()

    consumer = SessionConsumer(mq, lock, wf, max_drain=5, renew_interval_s=0.05)
    mq.enqueue("u", "s", _p("x"))
    consumer.drain_session("u", "s")
    assert lock.renews >= 1                                # 处理期间心跳至少续过一次


def test_after_drain_hook_fires_only_when_applied():
    """关段钩子:有消息落库才回调(带 user/session);抢不到锁/空转不回调。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    calls: list = []
    consumer = SessionConsumer(mq, lock, lambda u, s: FakeWriter([], (u, s)),
                               after_drain=lambda u, s, sc="": calls.append((u, s)))
    # 空会话 drain:无消息应用 → 不回调
    consumer.drain_session("u", "empty")
    assert calls == []
    # 有消息 → 回调一次,带正确 user/session
    mq.enqueue("u", "s", _p("hi"))
    consumer.drain_session("u", "s")
    assert calls == [("u", "s")]


def test_after_drain_hook_exception_does_not_break_consume():
    """钩子抛异常不影响消费结果(画像触发失败绝不拖垮 ingest)。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []

    def boom(u, s, sc=""):
        raise RuntimeError("画像触发炸了")

    consumer = SessionConsumer(mq, lock, lambda u, s: FakeWriter(log, (u, s)), after_drain=boom)
    mq.enqueue("u", "s", _p("hi"))
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 1 and [t for _, t in log] == ["hi"]   # 消费照常成功


def test_session_end_kind():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", {}, kind="session_end")
    consumer.drain_session("u", "s")
    assert log == [(("u", "s"), "<END>")]


def test_different_sessions_independent():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s1", _p("s1a"))
    mq.enqueue("u", "s2", _p("s2a"))
    consumer.drain_session("u", "s1")
    consumer.drain_session("u", "s2")
    assert set(log) == {(("u", "s1"), "s1a"), (("u", "s2"), "s2a")}


def test_end_to_end_queue_to_evidence(db):
    """真链路:入队一批 → drain → 真 SessionWriter 落 evidence(W0)。验证消费管线接通写入侧。

    单批入首段:seg 为空,detect_boundary 不调 LLM(代码短路),build_cell 不触发 → 无需 LLM 响应。
    """
    from personos.online.write_path import SessionWriter
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.evidence_store import EvidenceStore
    from personos.storage.seg_store import MemorySegStore

    from .fakes import FakeEmbedder, FakeLLM

    ev = EvidenceStore(db, user_id="u_it")
    cells = CellStore(db, user_id="u_it")
    atoms = AtomStore(db, user_id="u_it")
    seg = MemorySegStore()
    llm, emb = FakeLLM([]), FakeEmbedder()

    def writer_for(u, s):
        return SessionWriter(llm, emb, ev, cells, atoms, session_id=s, user_id=u, seg_store=seg)

    mq, lock = MemoryMsgQueue(), MemorySessionLock()
    consumer = SessionConsumer(mq, lock, writer_for)
    mq.enqueue("u_it", "sess", _p("我明天要去打篮球"))
    rep = consumer.drain_session("u_it", "sess")

    assert rep.applied == 1
    recs = ev.by_session("sess")
    assert [r.content_inline for r in recs] == ["我明天要去打篮球"]        # W0 证据已落库
    assert [r.content_inline for r in seg.load("u_it", "sess")] == ["我明天要去打篮球"]  # 段已存


# —— Dispatcher ——

class _InlinePool:
    """同步执行的假池:submit 即刻跑完(测调度逻辑,不引入线程时序)。"""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        fn(*args)


class _HoldPool:
    """把作业存起来不立即执行(测 in-flight 去重):run_all 手动放行。"""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


def test_dispatcher_inflight_dedup_no_redundant_submit():
    """在派中的会话不重复派:job 未完成期间,再 run_once 不会对同会话二次提交(防饥饿根因)。"""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("x"))
    pool = _HoldPool()
    disp = Dispatcher(mq, consumer, pool, cap=4)
    assert disp.run_once() == 1                        # 派了 s(挂起,未执行)
    assert disp.run_once() == 0                        # s 在派中 → 不重复派
    pool.run_all()                                     # 放行:drain 完成,清 in-flight
    assert mq.is_empty("u", "s")
    mq.enqueue("u", "s", _p("y"))                       # 新消息 → 可再次派
    assert disp.run_once() == 1


def test_two_consumers_single_flight_cross_pod():
    """两个消费者共享同一队列+锁(模拟两个 pod):同会话并发消费仍严格有序、恰好一次。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()                         # 共享锁 = 跨 pod 的 Redis 锁语义
    log: list = []
    guard = threading.Lock()

    def wf(u, s):
        class W:
            def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
                time.sleep(0.002)                      # 放大并发窗口
                with guard:
                    for m in msgs:
                        log.append(int(m.text))

            def end_session(self, *, task_type=None, scenario=""):
                pass
        return W()

    c1 = SessionConsumer(mq, lock, wf, max_drain=3)
    c2 = SessionConsumer(mq, lock, wf, max_drain=3)
    for i in range(30):
        mq.enqueue("u", "s", _p(i))

    def worker(c):
        while not mq.is_empty("u", "s"):
            c.drain_session("u", "s")

    t1 = threading.Thread(target=worker, args=(c1,))
    t2 = threading.Thread(target=worker, args=(c2,))
    t1.start(); t2.start(); t1.join(10); t2.join(10)
    assert log == list(range(30))                      # 两 pod 抢消费,仍有序+无重+无丢


def test_dispatcher_run_once_drains_active_sessions():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s1", _p("a"))
    mq.enqueue("u", "s2", _p("b"))
    pool = _InlinePool()
    disp = Dispatcher(mq, consumer, pool, cap=10)
    n = disp.run_once()
    assert n == 2 and pool.submitted == 2
    assert {t for _, t in log} == {"a", "b"}
    assert disp.run_once() == 0                        # 都排空了,看板空,不再派


def test_dispatcher_backpressure_when_pool_full():
    """池容量=1 且作业不归还名额(模拟满池):一轮最多派 1 个,不无限堆积。"""
    mq, lock, consumer, log = _make()
    for i in range(5):
        mq.enqueue("u", f"s{i}", _p(f"m{i}"))

    class _NoLeavePool:
        def submit(self, fn, *args):
            pass                                       # 不执行 → gate 名额不归还

    disp = Dispatcher(mq, consumer, _NoLeavePool(), cap=1)
    assert disp.run_once() == 1                        # 只派出 1 个就满
    assert disp.run_once() == 0                        # 仍满,派不出
