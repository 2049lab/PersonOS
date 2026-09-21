"""B6 压测:突发乱序并发 + 崩溃恢复。用真 ThreadPoolExecutor + Dispatcher,调小池制造争用。

验收记忆服务的四条铁律(眼镜场景:短对话高频、可打断、多会话并发):
1. 每会话严格 FIFO(说话顺序不乱)
2. 每条恰好消费一次(无丢、无重)
3. 单飞:同一会话从不被两个线程同时消费(埋并发计数探针,违例即记录)
4. 崩溃恢复:reserve 后未 ack(模拟 pod 死在消费中)→ 重启消费不丢不重

不打真 LLM/Redis:MemoryMsgQueue(线程安全)+ MemorySessionLock + 记账假 writer。
Redis 变体的命令语义另有 test_msg_queue[redis] 覆盖,这里聚焦并发编排的正确性。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from personos.ingest_worker import Dispatcher, SessionConsumer
from personos.storage.msg_queue import MemoryMsgQueue
from personos.storage.session_lock import MemorySessionLock


def _p(session_local_seq: int) -> dict:
    """载荷带"会话内序号",消费侧据此校验 FIFO 与 exactly-once。"""
    return {"messages": [{"speaker": "user", "text": str(session_local_seq)}]}


class RecordingWriter:
    """记账假 writer:记录每会话消费到的序号;埋单飞探针(同会话并发消费即记违例)。"""

    def __init__(self, key, consumed, guard, inflight, inflight_guard, violations):
        self._key = key
        self._consumed = consumed
        self._guard = guard
        self._inflight = inflight
        self._inflight_guard = inflight_guard
        self._violations = violations

    def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
        with self._inflight_guard:                      # 单飞探针:进入即 +1,>1 说明并发消费
            self._inflight[self._key] += 1
            if self._inflight[self._key] > 1:
                self._violations.append(self._key)
        time.sleep(0.0005)                              # 放大并发窗口,让违例更易暴露
        with self._guard:
            for m in msgs:
                self._consumed[self._key].append(int(m.text))
        with self._inflight_guard:
            self._inflight[self._key] -= 1

    def end_session(self, *, task_type=None, scenario=""):
        pass


def _harness(max_drain=7, pool_size=4):
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    consumed: dict = defaultdict(list)
    guard = threading.Lock()
    inflight: dict = defaultdict(int)
    inflight_guard = threading.Lock()
    violations: list = []

    def writer_for(u, s):
        return RecordingWriter((u, s), consumed, guard, inflight, inflight_guard, violations)

    consumer = SessionConsumer(mq, lock, writer_for, max_drain=max_drain)
    pool = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="stress")
    disp = Dispatcher(mq, consumer, pool, cap=pool_size, tick_s=0.003, idle_tick_s=0.01)
    return mq, disp, pool, consumed, violations


def _wait_drained(mq, sessions, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(mq.is_empty("u", s) for s in sessions):
            return True
        time.sleep(0.02)
    return False


def test_stress_concurrent_ordered_exactly_once():
    """8 会话 × 每会话 60 条,多生产线程并发猛灌;小池(4)强制争用。

    断言:单飞无违例;每会话收到的序号 == [0..59](有序 + 无重 + 无丢)。
    """
    SESSIONS = [f"s{i}" for i in range(8)]
    PER = 60
    mq, disp, pool, consumed, violations = _harness(max_drain=7, pool_size=4)
    disp.start()

    def produce(s):
        for i in range(PER):
            mq.enqueue("u", s, _p(i))
            if i % 13 == 7:
                time.sleep(0.001)                       # 制造突发+间歇(眼镜打断节奏)

    producers = [threading.Thread(target=produce, args=(s,)) for s in SESSIONS]
    for t in producers:
        t.start()
    for t in producers:
        t.join()

    assert _wait_drained(mq, SESSIONS), "超时未排空(消费卡住?)"
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == [], f"单飞被破坏:同会话并发消费 {violations[:5]}"
    for s in SESSIONS:
        got = consumed[("u", s)]
        assert got == list(range(PER)), (
            f"会话 {s} 消费异常:len={len(got)} "
            f"dup={len(got) != len(set(got))} order_ok={got == sorted(got)}")


def test_stress_crash_recovery_no_loss_no_dup():
    """模拟 pod 死在消费中:先把前若干条 reserve 到 proc 但不 ack,再启消费 → 全量恰好一次、有序。"""
    mq, disp, pool, consumed, violations = _harness(max_drain=5, pool_size=2)
    N = 30
    for i in range(N):
        mq.enqueue("u", "s", _p(i))
    # 崩溃:前 5 条被取走(进 proc)但未处理未 ack
    for _ in range(5):
        mq.reserve("u", "s")
    assert mq.depth("u", "s") == N - 5

    disp.start()
    assert _wait_drained(mq, ["s"])
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == []
    got = consumed[("u", "s")]
    assert got == list(range(N)), f"崩溃恢复后异常:{got}"      # 回放的 5 条不丢不重,顺序完整


def test_stress_no_session_starvation():
    """池(2) < 会话数(6),每会话慢消费:验证所有会话【并发推进】,不出现"一个榨干才轮到下一个"。

    探针:记录每会话首次被消费的时间;若无饥饿,6 个会话的首次消费应挤在一个小窗口内
    (而非一个个串行等待)。回归 s2 被 s1 饿死的真实 bug。
    """
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    first_seen: dict = {}
    seen_guard = threading.Lock()
    t0 = time.monotonic()

    class SlowWriter:
        def __init__(self, key):
            self._key = key

        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            with seen_guard:
                first_seen.setdefault(self._key, time.monotonic() - t0)
            time.sleep(0.03)                            # 慢消费,放大饥饿

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: SlowWriter((u, s)), max_drain=3)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fair")
    disp = Dispatcher(mq, consumer, pool, cap=2, tick_s=0.003, idle_tick_s=0.01)

    SESSIONS = [f"s{i}" for i in range(6)]
    for s in SESSIONS:
        for i in range(12):
            mq.enqueue("u", s, _p(i))
    disp.start()
    assert _wait_drained(mq, SESSIONS, timeout=20)
    disp.stop()
    pool.shutdown(wait=True)

    # 所有会话都被消费过(无一被完全饿死)
    assert set(first_seen) == set(("u", s) for s in SESSIONS), f"有会话从未被消费:{first_seen}"
    # 首次消费时间跨度小(并发推进,不是串行):6 会话首触应在 ~数百 ms 内,而非线性累积到秒级
    span = max(first_seen.values()) - min(first_seen.values())
    assert span < 2.0, f"疑似饥饿:首次消费时间跨度 {span:.2f}s 过大 {first_seen}"


def test_stress_poison_under_concurrency():
    """并发下毒消息不拖垮:每会话第 3 条是毒(必崩),验证毒消息被跳过、其余全部有序消费。"""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    consumed: dict = defaultdict(list)
    guard = threading.Lock()

    class PoisonWriter:
        def __init__(self, key):
            self._key = key

        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            if msgs[0].text == "POISON":
                raise ValueError("毒消息")
            with guard:
                self._log_key = self._key
                consumed[self._key].append(msgs[0].text)

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: PoisonWriter((u, s)),
                               max_drain=5, max_retries=3)
    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="poison")
    disp = Dispatcher(mq, consumer, pool, cap=3, tick_s=0.003, idle_tick_s=0.01)

    SESSIONS = [f"s{i}" for i in range(4)]
    for s in SESSIONS:
        for i in range(6):
            mq.enqueue("u", s, _p("POISON" if i == 2 else f"m{i}"))
    disp.start()
    assert _wait_drained(mq, SESSIONS, timeout=20)
    disp.stop()
    pool.shutdown(wait=True)

    for s in SESSIONS:
        got = consumed[("u", s)]
        # 毒消息(位置 2)被跳过,其余 5 条按序消费(m0,m1,m3,m4,m5)
        assert got == ["m0", "m1", "m3", "m4", "m5"], f"{s}: {got}"


def test_stress_interleaved_late_arrivals():
    """消费追上后队列清空,过一会同会话又来新消息(消费快于生产)→ 仍从下一条按序续上。"""
    mq, disp, pool, consumed, violations = _harness(max_drain=10, pool_size=3)
    disp.start()
    # 第一波
    for i in range(10):
        mq.enqueue("u", "s", _p(i))
    assert _wait_drained(mq, ["s"])
    # 队列已空一段时间后第二波(seq 会从 11 继续,消费侧 payload 序号也接着给)
    time.sleep(0.05)
    for i in range(10, 20):
        mq.enqueue("u", "s", _p(i))
    assert _wait_drained(mq, ["s"])
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == []
    assert consumed[("u", "s")] == list(range(20))     # 两波拼接,有序无缝


# ── 公平阀门:话痨会话不独占池名额 ──────────────────────────────────────────

def test_max_drain_must_be_below_queue_depth_or_the_valve_never_fires():
    """**配置级不变量**:max_drain 必须显著小于 max_queue_depth,否则公平阀门形同虚设。

    这不是洁癖 —— 两个值分开看都合理,放一起才失效:队列最多堆 max_queue_depth 条,
    drain 会先在"队列空"处 break,永远走不到 max_drain 那个出口,实际行为退化成
    "占住名额直到该会话排空"。线上曾是 20 > 15 的失效态,视频场景下单个会话最坏能
    独占一个池名额十几小时(15 批 × 每批 20 clip × 约 216s)。
    """
    from personos.config import load_config

    s = load_config()
    assert s.max_drain_per_cycle < s.max_queue_depth, (
        f"max_drain={s.max_drain_per_cycle} 未小于 max_queue_depth={s.max_queue_depth},"
        "公平阀门不会触发,长会话将独占池名额")


def _race_long_vs_short(max_drain: int, long_n: int = 30, short_n: int = 3):
    """池只有 1 个名额:长会话**已在 drain 途中**时短会话才到,看它要等多久。

    "短会话后到"是本用例的关键——若两者同时在看板上,轮转可能先派短会话,
    那就根本没构成"名额被占"的竞争,断言会恒真(实测 max_drain=5/20/30 全绿,
    完全无判别力)。必须等长会话真的开始消费再投短会话。
    返回 (短会话首次被消费的位置, 长会话消费数, 短会话消费数)。
    """
    mq, lock = MemoryMsgQueue(), MemorySessionLock()
    order: list[str] = []
    guard = threading.Lock()

    class _W:
        def __init__(self, key):
            self._key = key

        def feed_batch(self, msgs, **kw):
            time.sleep(0.01)                    # 拉长单条处理,让占用差异显形
            with guard:
                order.extend(self._key for _ in msgs)

        def end_session(self, **kw):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: _W(s), max_drain=max_drain)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fair")
    disp = Dispatcher(mq, consumer, pool, cap=1, tick_s=0.003, idle_tick_s=0.01)
    for i in range(long_n):
        mq.enqueue("u", "long", _p(i))
    disp.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:      # 等长会话真的占住名额开始消费
            with guard:
                if order:
                    break
            time.sleep(0.002)
        for i in range(short_n):
            mq.enqueue("u", "short", _p(i))
        assert _wait_drained(mq, ["long", "short"])
    finally:
        disp.stop()
        pool.shutdown(wait=True)
    return order.index("short"), order.count("long"), order.count("short")


def test_long_session_yields_slot_so_late_comers_get_consumed():
    """话痨会话**不得**把后到的会话饿死 —— 这是 max_drain 存在的唯一理由。

    直接验效果而非实现:阀门生效则长会话消够 max_drain 就归还名额,短会话能插进来;
    失效则短会话一直等到长会话排空。判据只取"短会话在长会话消完前就被消费",
    不假设具体交错次序(线程调度本就不确定)。
    """
    first_short, n_long, n_short = _race_long_vs_short(max_drain=5)
    assert (n_long, n_short) == (30, 3)          # 先确认都没丢,否则下面的位置断言没意义
    assert first_short < 30, (
        f"短会话直到长会话 30 条消完才被消费(首次出现在第 {first_short} 位)"
        "—— 名额被独占,公平阀门没生效")


def test_valve_really_is_what_lets_late_comers_in():
    """**判别性对照**:把 max_drain 调到 ≥ 长会话长度(失效态),短会话必被压到最后。

    没有这一条,上面那个断言可能是恒真的 —— 绿了也证明不了阀门有用。
    这同时复现了线上曾经的失效态(max_drain=20 > max_queue_depth=15)。
    """
    first_short, _, _ = _race_long_vs_short(max_drain=30)
    assert first_short == 30, (
        f"阀门失效态下短会话本应等到最后(第 30 位),实测第 {first_short} 位"
        "—— 说明上面的公平断言没有判别力,用例需要重新设计")
