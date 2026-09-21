"""会话消息队列单测:Memory 与 Redis(FakeRedis)双实现走同一套断言(语义一致性)。

覆盖:FIFO 顺序 / 可靠出队(reserve→ack) / 崩溃回放(recover 未 ack 的) / 游标去重 /
看板增删 / 排空竞态(擦号瞬间来新消息补回) / 会话与用户隔离 / 背压深度。
"""

from __future__ import annotations

import pytest

from personos.storage.msg_queue import EnqueueBusy, Envelope, MemoryMsgQueue, RedisMsgQueue

from .fakes import FakeRedis


def _memory():
    return MemoryMsgQueue()


def _redis():
    return RedisMsgQueue(FakeRedis(), ttl_s=3600)


@pytest.fixture(params=[_memory, _redis], ids=["memory", "redis"])
def q(request):
    return request.param()


def _drain_texts(q, u, s):
    """把一个会话的消息按 reserve→ack 顺序全部取出,返回文本序列(验证 FIFO)。"""
    out = []
    while True:
        env = q.reserve(u, s)
        if env is None:
            break
        out.append(env.payload["text"])
        q.ack(u, s, env)
    return out


def test_enqueue_assigns_monotonic_seq_and_msgid(q):
    m1, s1 = q.enqueue("u", "s", {"text": "第一句"})
    m2, s2 = q.enqueue("u", "s", {"text": "第二句"})
    assert s1 == 1 and s2 == 2                      # 会话内单调
    assert m1 != m2                                 # msg_id 唯一
    assert q.depth("u", "s") == 2


def test_fifo_order(q):
    for i in range(5):
        q.enqueue("u", "s", {"text": f"m{i}"})
    assert _drain_texts(q, "u", "s") == ["m0", "m1", "m2", "m3", "m4"]


def test_reserve_moves_to_processing_ack_removes(q):
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")
    assert env.payload["text"] == "x"
    assert q.depth("u", "s") == 0                   # 已移出主队列
    assert not q.is_empty("u", "s")                 # 但在途未 ack → 非空
    q.ack("u", "s", env)
    assert q.is_empty("u", "s")                     # ack 后才空


def test_crash_replay_via_recover(q):
    """reserve 后未 ack(模拟崩溃)→ recover 能回放,且按 seq 升序。"""
    q.enqueue("u", "s", {"text": "a"})
    q.enqueue("u", "s", {"text": "b"})
    e1 = q.reserve("u", "s")                        # 取 a,不 ack(崩溃)
    e2 = q.reserve("u", "s")                        # 取 b,不 ack(崩溃)
    assert (e1.payload["text"], e2.payload["text"]) == ("a", "b")
    recovered = q.recover("u", "s")
    assert [e.payload["text"] for e in recovered] == ["a", "b"]   # 升序回放
    assert [e.seq for e in recovered] == [1, 2]


def test_cursor_dedup_watermark(q):
    assert q.cursor_get("u", "s") == 0
    q.cursor_set("u", "s", 3)
    assert q.cursor_get("u", "s") == 3
    # 消费侧用法:seq<=cursor 判重投跳过
    q.enqueue("u", "s", {"text": "old", "seq_hint": 2})
    env = q.reserve("u", "s")
    assert env.seq == 1                             # 新队列 seq 从 1 起(与 cursor 无关)
    # 语义演示:若某信封 seq<=cursor 应被消费方跳过(这里只验证 cursor 存取正确)


def test_active_board_add_and_remove(q):
    q.enqueue("u", "s", {"text": "x"})
    assert ("u", "s") in q.active_sessions()
    env = q.reserve("u", "s")
    q.ack("u", "s", env)
    assert q.deactivate_if_empty("u", "s") is True
    assert ("u", "s") not in q.active_sessions()


def test_deactivate_refuses_when_not_empty(q):
    q.enqueue("u", "s", {"text": "x"})
    assert q.deactivate_if_empty("u", "s") is False   # 还有待处理,不移除
    assert ("u", "s") in q.active_sessions()


def test_deactivate_race_new_message_re_adds(q):
    """排空后正要移出看板,恰好来了新消息 → 应留在看板(不制造孤儿)。

    构造:队列已空 + 手动把会话仍留在看板(模拟"上一轮排空判定与本次入队交错"),
    这里直接验证 deactivate 的复查分支:空则真移除,期间有新消息则补回。
    """
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")
    q.ack("u", "s", env)                            # 现在空了
    # 正常路径:空 → 移除成功
    assert q.deactivate_if_empty("u", "s") is True
    # 新消息到来 → 重新登记看板
    q.enqueue("u", "s", {"text": "y"})
    assert ("u", "s") in q.active_sessions()
    assert q.deactivate_if_empty("u", "s") is False


def test_session_and_user_isolation(q):
    q.enqueue("u1", "s", {"text": "u1s"})
    q.enqueue("u2", "s", {"text": "u2s"})
    q.enqueue("u1", "s2", {"text": "u1s2"})
    assert _drain_texts(q, "u1", "s") == ["u1s"]
    assert _drain_texts(q, "u2", "s") == ["u2s"]
    assert _drain_texts(q, "u1", "s2") == ["u1s2"]


def test_reserve_empty_returns_none(q):
    assert q.reserve("nobody", "s") is None
    assert q.is_empty("nobody", "s")


def test_kind_carried_through(q):
    q.enqueue("u", "s", {}, kind="session_end")
    env = q.reserve("u", "s")
    assert env.kind == "session_end"


def _expire_seq(q, u, s):
    """模拟 seq 键闲置过期(Redis)/纪元重置(Memory),不动其它键。"""
    if isinstance(q, RedisMsgQueue):
        q._c.delete(q._sk(u, s))
    else:
        q._seq.pop((u, s), None)


def test_seq_reset_clears_stale_cursor(q):
    """回归丢消息 bug:seq 闲置过期重置到 1,若陈旧高游标不清 → 新消息被误判重投跳过。

    修复:enqueue 时 seq==1(新纪元)清除残留游标 → 新消息 seq=1 > cursor=0,正常消费。
    """
    q.enqueue("u", "s", {"text": "旧纪元"})            # seq=1
    q.cursor_set("u", "s", 5)                          # 模拟旧纪元消费到 cursor=5
    _expire_seq(q, "u", "s")                           # seq 键过期(闲置>TTL)
    _mid, seq = q.enqueue("u", "s", {"text": "新纪元第一条"})
    assert seq == 1                                    # seq 重置到 1
    assert q.cursor_get("u", "s") == 0                 # 陈旧游标已被清 → 不会误判重投
    env = q.reserve("u", "s")
    assert env.seq == 1 and env.seq > q.cursor_get("u", "s")   # 会被正常消费(不跳过)


def test_deactivate_refuses_when_proc_nonempty(q):
    """M1 回归:消息 reserve 到 proc(main 空但 proc 非空)时,不能移出看板——否则孤儿会话。"""
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")                          # main→proc:main 空,proc 有
    assert q.depth("u", "s") == 0 and not q.is_empty("u", "s")
    assert q.deactivate_if_empty("u", "s") is False    # proc 非空 → 拒绝下看板
    assert ("u", "s") in q.active_sessions()
    q.ack("u", "s", env)
    assert q.deactivate_if_empty("u", "s") is True      # 真排空才下看板


def test_concurrent_enqueue_contiguous_seq_no_loss(q):
    """H1 不变量:多线程并发入队同会话 → seq 连续无缺无重、消费恰好一次(取号+入列原子)。"""
    import threading

    def prod():
        for _ in range(50):
            q.enqueue("u", "s", {"text": "x"})

    ts = [threading.Thread(target=prod) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    seqs = []
    while True:
        e = q.reserve("u", "s")
        if e is None:
            break
        seqs.append(e.seq)
        q.ack("u", "s", e)
    assert sorted(seqs) == list(range(1, 401))         # 400 条,seq 1..400 完整无缺无重


def test_redis_enqueue_releases_lock():
    """H1:入队锁用完即释放(不残留);微秒级临界区。"""
    c = FakeRedis()
    rq = RedisMsgQueue(c, ttl_s=3600)
    rq.enqueue("u", "s", {"text": "x"})
    assert rq._elk("u", "s") not in c.data             # enqlock 已释放


def test_redis_enqueue_busy_raises_not_silent_loss():
    """H-1 修复:入队锁被他人长期持有 → 超时抛 EnqueueBusy(拒收让上层重试),绝不硬上丢消息。"""
    c = FakeRedis()
    rq = RedisMsgQueue(c, ttl_s=3600)
    # 预占入队锁且不释放(FakeRedis TTL 不真过期,模拟"持锁者一直在")
    c.set(rq._elk("u", "s"), "someone-else", nx=True, px=999999)
    import personos.storage.msg_queue as mq_mod
    orig = mq_mod.time.monotonic
    calls = {"n": 0}

    def fake_mono():          # 快进时间,免真等 3s
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 100.0

    mq_mod.time.monotonic = fake_mono
    try:
        with pytest.raises(EnqueueBusy):
            rq.enqueue("u", "s", {"text": "会被拒"})
    finally:
        mq_mod.time.monotonic = orig
    # 关键:拒收后不得有半吊子写入(seq 未推进、主队列空)
    assert c.get(rq._sk("u", "s")) in (None, "0", 0)
    assert rq.depth("u", "s") == 0


def test_clear_failed_removes_counter(q):
    """毒计数键用完即清(不留垃圾)。"""
    assert q.mark_failed("u", "s", "m1") == 1
    assert q.mark_failed("u", "s", "m1") == 2
    q.clear_failed("u", "s", "m1")
    assert q.mark_failed("u", "s", "m1") == 1          # 清后重新从 1 计


def test_envelope_raw_stable_for_ack():
    """信封 raw 序列化稳定(sort_keys):同内容两次构造一致,ack 才能精确 LREM。"""
    r1 = Envelope.make("mid", 1, "ingest", {"text": "x", "a": 1})
    r2 = Envelope.make("mid", 1, "ingest", {"a": 1, "text": "x"})
    assert r1 == r2
    env = Envelope.from_raw(r1)
    assert env.msg_id == "mid" and env.seq == 1 and env.raw == r1
