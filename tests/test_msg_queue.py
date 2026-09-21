"""Session message queue unit tests: the Memory and Redis (FakeRedis) implementations run
against the same assertions, which is what proves their semantics agree.

Covers: FIFO order / reliable dequeue (reserve then ack) / crash replay (recovering anything
never acked) / the cursor watermark for deduplication / adding to and removing from the active
board / the drain race (a new message arriving at the instant the session is deregistered gets
it re-added) / session and user isolation / backpressure depth.
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
    """Pull every message of one session out in reserve-then-ack order and return the text
    sequence, which is what verifies FIFO."""
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
    assert s1 == 1 and s2 == 2                      # monotonic within a session
    assert m1 != m2                                 # msg_id is unique
    assert q.depth("u", "s") == 2


def test_fifo_order(q):
    for i in range(5):
        q.enqueue("u", "s", {"text": f"m{i}"})
    assert _drain_texts(q, "u", "s") == ["m0", "m1", "m2", "m3", "m4"]


def test_reserve_moves_to_processing_ack_removes(q):
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")
    assert env.payload["text"] == "x"
    assert q.depth("u", "s") == 0                   # already moved out of the main queue
    assert not q.is_empty("u", "s")                 # but in flight and unacked, so not empty
    q.ack("u", "s", env)
    assert q.is_empty("u", "s")                     # empty only after the ack


def test_crash_replay_via_recover(q):
    """Reserved but never acked (simulating a crash): recover replays them, in ascending seq
    order."""
    q.enqueue("u", "s", {"text": "a"})
    q.enqueue("u", "s", {"text": "b"})
    e1 = q.reserve("u", "s")                        # take a, do not ack (crash)
    e2 = q.reserve("u", "s")                        # take b, do not ack (crash)
    assert (e1.payload["text"], e2.payload["text"]) == ("a", "b")
    recovered = q.recover("u", "s")
    assert [e.payload["text"] for e in recovered] == ["a", "b"]   # replayed in ascending order
    assert [e.seq for e in recovered] == [1, 2]


def test_cursor_dedup_watermark(q):
    assert q.cursor_get("u", "s") == 0
    q.cursor_set("u", "s", 3)
    assert q.cursor_get("u", "s") == 3
    # How the consumer uses it: seq <= cursor means a redelivery and is skipped.
    q.enqueue("u", "s", {"text": "old", "seq_hint": 2})
    env = q.reserve("u", "s")
    assert env.seq == 1                             # a fresh queue starts seq at 1, independent of the cursor
    # Semantics illustrated: an envelope with seq <= cursor should be skipped by the consumer.
    # This test only verifies that cursor reads and writes are correct.


def test_active_board_add_and_remove(q):
    q.enqueue("u", "s", {"text": "x"})
    assert ("u", "s") in q.active_sessions()
    env = q.reserve("u", "s")
    q.ack("u", "s", env)
    assert q.deactivate_if_empty("u", "s") is True
    assert ("u", "s") not in q.active_sessions()


def test_deactivate_refuses_when_not_empty(q):
    q.enqueue("u", "s", {"text": "x"})
    assert q.deactivate_if_empty("u", "s") is False   # work is still pending, so do not remove
    assert ("u", "s") in q.active_sessions()


def test_deactivate_race_new_message_re_adds(q):
    """A new message arrives just as a drained session is about to be removed from the board,
    so it must stay on the board — otherwise the session is orphaned.

    Setup: an empty queue with the session still manually on the board (simulating the previous
    round's drain check interleaving with this enqueue). What is verified here is deactivate's
    re-check branch: genuinely empty means it really removes, and a new message in the meantime
    puts it back.
    """
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")
    q.ack("u", "s", env)                            # now empty
    # Normal path: empty, so removal succeeds.
    assert q.deactivate_if_empty("u", "s") is True
    # A new message arrives, so the session is registered on the board again.
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
    """Simulate the seq key expiring through inactivity (Redis) or an epoch reset (Memory),
    leaving every other key alone."""
    if isinstance(q, RedisMsgQueue):
        q._c.delete(q._sk(u, s))
    else:
        q._seq.pop((u, s), None)


def test_seq_reset_clears_stale_cursor(q):
    """Regression for a lost-message bug: when the seq key expires through inactivity and
    resets to 1, a stale high cursor that is not cleared makes new messages look like
    redeliveries and get skipped.

    The fix: on enqueue, seq == 1 means a new epoch, so the leftover cursor is cleared. The new
    message then has seq=1 > cursor=0 and is consumed normally.
    """
    q.enqueue("u", "s", {"text": "旧纪元"})            # seq=1
    q.cursor_set("u", "s", 5)                          # simulate the old epoch having consumed up to cursor=5
    _expire_seq(q, "u", "s")                           # the seq key expires (idle longer than the TTL)
    _mid, seq = q.enqueue("u", "s", {"text": "新纪元第一条"})
    assert seq == 1                                    # seq reset to 1
    assert q.cursor_get("u", "s") == 0                 # the stale cursor was cleared, so no false redelivery
    env = q.reserve("u", "s")
    # It gets consumed normally rather than skipped.
    assert env.seq == 1 and env.seq > q.cursor_get("u", "s")


def test_deactivate_refuses_when_proc_nonempty(q):
    """M1 regression: while a message is reserved into proc (main empty but proc non-empty) the
    session must not be taken off the board, or it becomes orphaned."""
    q.enqueue("u", "s", {"text": "x"})
    env = q.reserve("u", "s")                          # main -> proc: main is empty, proc has it
    assert q.depth("u", "s") == 0 and not q.is_empty("u", "s")
    assert q.deactivate_if_empty("u", "s") is False    # proc is non-empty, so refuse to deregister
    assert ("u", "s") in q.active_sessions()
    q.ack("u", "s", env)
    assert q.deactivate_if_empty("u", "s") is True      # only a genuine drain deregisters it


def test_concurrent_enqueue_contiguous_seq_no_loss(q):
    """H1 invariant: concurrent enqueues to the same session from many threads produce a
    contiguous seq range with no gaps and no duplicates, and each is consumed exactly once
    (taking a number and appending to the list are atomic together)."""
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
    # 400 messages, seq 1..400 complete with no gaps and no duplicates.
    assert sorted(seqs) == list(range(1, 401))


def test_redis_enqueue_releases_lock():
    """H1: the enqueue lock is released as soon as it is done with, leaving nothing behind; the
    critical section is microseconds long."""
    c = FakeRedis()
    rq = RedisMsgQueue(c, ttl_s=3600)
    rq.enqueue("u", "s", {"text": "x"})
    assert rq._elk("u", "s") not in c.data             # the enqueue lock was released


def test_redis_enqueue_busy_raises_not_silent_loss():
    """The H-1 fix: when someone else holds the enqueue lock for a long time, time out and
    raise EnqueueBusy (refuse the message so the caller retries) rather than forcing the write
    through and losing it."""
    c = FakeRedis()
    rq = RedisMsgQueue(c, ttl_s=3600)
    # Take the enqueue lock up front and never release it. FakeRedis does not really expire
    # TTLs, which simulates a holder that simply never goes away.
    c.set(rq._elk("u", "s"), "someone-else", nx=True, px=999999)
    import personos.storage.msg_queue as mq_mod
    orig = mq_mod.time.monotonic
    calls = {"n": 0}

    def fake_mono():          # fast-forward time so the test does not really wait 3s
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 100.0

    mq_mod.time.monotonic = fake_mono
    try:
        with pytest.raises(EnqueueBusy):
            rq.enqueue("u", "s", {"text": "会被拒"})
    finally:
        mq_mod.time.monotonic = orig
    # The crucial part: refusing must leave no half-finished write (seq did not advance, the
    # main queue is empty).
    assert c.get(rq._sk("u", "s")) in (None, "0", 0)
    assert rq.depth("u", "s") == 0


def test_clear_failed_removes_counter(q):
    """The poison counter key is cleared as soon as it is done with, leaving no garbage."""
    assert q.mark_failed("u", "s", "m1") == 1
    assert q.mark_failed("u", "s", "m1") == 2
    q.clear_failed("u", "s", "m1")
    assert q.mark_failed("u", "s", "m1") == 1          # after clearing, counting restarts at 1


def test_envelope_raw_stable_for_ack():
    """Envelope raw serialization is stable (sort_keys): building the same content twice gives
    an identical string, which is what lets ack do a precise LREM."""
    r1 = Envelope.make("mid", 1, "ingest", {"text": "x", "a": 1})
    r2 = Envelope.make("mid", 1, "ingest", {"a": 1, "text": "x"})
    assert r1 == r2
    env = Envelope.from_raw(r1)
    assert env.msg_id == "mid" and env.seq == 1 and env.raw == r1
