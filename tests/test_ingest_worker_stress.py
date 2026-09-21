"""B6 stress tests: bursty out-of-order concurrency plus crash recovery. Uses a real
ThreadPoolExecutor and Dispatcher with a deliberately small pool to force contention.

These check the four hard rules of the memory service, in a setting of short, frequent,
interruptible conversations across many concurrent sessions:
1. Strict FIFO per session, so utterance order is preserved.
2. Each message is consumed exactly once, with no loss and no duplication.
3. Single-flight: a session is never consumed by two threads at once. A concurrency counter
   probe records any violation.
4. Crash recovery: a message reserved but never acked (a worker that died mid-consumption)
   is replayed on restart without loss or duplication.

No real LLM or Redis is involved: a thread-safe MemoryMsgQueue, a MemorySessionLock and a
bookkeeping fake writer. The Redis variant's command semantics are covered separately by
test_msg_queue[redis]; this file focuses on the correctness of the concurrent orchestration.
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
    """The payload carries a per-session sequence number, which the consumer side uses to
    verify FIFO order and exactly-once delivery."""
    return {"messages": [{"speaker": "user", "text": str(session_local_seq)}]}


class RecordingWriter:
    """A bookkeeping fake writer: records the sequence numbers consumed per session and
    carries a single-flight probe that logs a violation if one session is consumed
    concurrently."""

    def __init__(self, key, consumed, guard, inflight, inflight_guard, violations):
        self._key = key
        self._consumed = consumed
        self._guard = guard
        self._inflight = inflight
        self._inflight_guard = inflight_guard
        self._violations = violations

    def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
        with self._inflight_guard:                      # single-flight probe: +1 on entry; a value >1 means concurrent consumption
            self._inflight[self._key] += 1
            if self._inflight[self._key] > 1:
                self._violations.append(self._key)
        time.sleep(0.0005)                              # widen the concurrency window so violations surface more easily
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
    """Eight sessions of 60 messages each, pushed in by concurrent producer threads, with a
    small pool of 4 to force contention.

    Asserts that single-flight is never violated and that each session receives exactly
    [0..59]: in order, with no duplicates and no losses.
    """
    SESSIONS = [f"s{i}" for i in range(8)]
    PER = 60
    mq, disp, pool, consumed, violations = _harness(max_drain=7, pool_size=4)
    disp.start()

    def produce(s):
        for i in range(PER):
            mq.enqueue("u", s, _p(i))
            if i % 13 == 7:
                time.sleep(0.001)                       # produce bursts and gaps, like an interruptible conversation

    producers = [threading.Thread(target=produce, args=(s,)) for s in SESSIONS]
    for t in producers:
        t.start()
    for t in producers:
        t.join()

    assert _wait_drained(mq, SESSIONS), "timed out without draining (consumption stuck?)"
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == [], f"single-flight broken: concurrent consumption of {violations[:5]}"
    for s in SESSIONS:
        got = consumed[("u", s)]
        assert got == list(range(PER)), (
            f"session {s} consumed wrongly: len={len(got)} "
            f"dup={len(got) != len(set(got))} order_ok={got == sorted(got)}")


def test_stress_crash_recovery_no_loss_no_dup():
    """Simulate a worker dying mid-consumption: reserve the first few messages into the
    processing list without acking them, then start consuming. Everything must still be
    consumed exactly once and in order."""
    mq, disp, pool, consumed, violations = _harness(max_drain=5, pool_size=2)
    N = 30
    for i in range(N):
        mq.enqueue("u", "s", _p(i))
    # the crash: the first 5 messages are taken into the processing list but never handled or acked
    for _ in range(5):
        mq.reserve("u", "s")
    assert mq.depth("u", "s") == N - 5

    disp.start()
    assert _wait_drained(mq, ["s"])
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == []
    got = consumed[("u", "s")]
    assert got == list(range(N)), f"wrong result after crash recovery: {got}"   # the 5 replayed messages are neither lost nor duplicated, and order holds


def test_stress_no_session_starvation():
    """With a pool of 2 against 6 sessions and slow consumption per session, all sessions must
    make progress concurrently instead of one being drained completely before the next is
    touched.

    The probe records when each session is first consumed. Without starvation the six first
    consumptions should cluster into a small window rather than queue up one after another.
    This is a regression test for a real bug where one session starved another.
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
            time.sleep(0.03)                            # slow consumption, to amplify any starvation

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

    # every session was consumed at some point, none was starved completely
    assert set(first_seen) == set(("u", s) for s in SESSIONS), f"some session was never consumed: {first_seen}"
    # the spread of first-consumption times stays small, showing concurrent rather than serial
    # progress: six sessions should start within a few hundred ms, not pile up into seconds
    span = max(first_seen.values()) - min(first_seen.values())
    assert span < 2.0, f"looks like starvation: first-consumption spread {span:.2f}s is too large {first_seen}"


def test_stress_poison_under_concurrency():
    """A poison message must not drag the system down under concurrency: the third message of
    every session always raises, and the rest must still be consumed in order."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    consumed: dict = defaultdict(list)
    guard = threading.Lock()

    class PoisonWriter:
        def __init__(self, key):
            self._key = key

        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            if msgs[0].text == "POISON":
                raise ValueError("poison message")
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
        # the poison message at index 2 is skipped and the other 5 are consumed in order
        assert got == ["m0", "m1", "m3", "m4", "m5"], f"{s}: {got}"


def test_stress_interleaved_late_arrivals():
    """When consumption outruns production the queue empties, and a later message for the same
    session must simply continue in order from the next one."""
    mq, disp, pool, consumed, violations = _harness(max_drain=10, pool_size=3)
    disp.start()
    # first wave
    for i in range(10):
        mq.enqueue("u", "s", _p(i))
    assert _wait_drained(mq, ["s"])
    # second wave, after the queue has been empty for a while; the queue seq continues from 11
    # and the payload numbering on the consumer side continues too
    time.sleep(0.05)
    for i in range(10, 20):
        mq.enqueue("u", "s", _p(i))
    assert _wait_drained(mq, ["s"])
    disp.stop()
    pool.shutdown(wait=True)

    assert violations == []
    assert consumed[("u", "s")] == list(range(20))     # the two waves join up in order, without a seam


# -- The fairness valve: a chatty session must not monopolise a pool slot --

def test_max_drain_must_be_below_queue_depth_or_the_valve_never_fires():
    """A configuration-level invariant: max_drain must be clearly smaller than
    max_queue_depth, otherwise the fairness valve is dead code.

    This is not fussiness. Each value looks reasonable on its own and only breaks in
    combination: the queue holds at most max_queue_depth messages, so drain always breaks out
    at the "queue is empty" exit and never reaches the max_drain exit. The real behaviour then
    degrades into "hold the slot until this session is fully drained". Production once ran in
    exactly that broken state with 20 > 15, where in a video workload a single session could
    hold one pool slot for more than ten hours (15 batches x 20 clips per batch x about 216s).
    """
    from personos.config import load_config

    s = load_config()
    assert s.max_drain_per_cycle < s.max_queue_depth, (
        f"max_drain={s.max_drain_per_cycle} is not smaller than max_queue_depth={s.max_queue_depth}, "
        "so the fairness valve never fires and a long session will monopolise a pool slot")


def _race_long_vs_short(max_drain: int, long_n: int = 30, short_n: int = 3):
    """With only one pool slot, let the short session arrive while the long one is already
    mid-drain, and measure how long it has to wait.

    The short session arriving late is the crux of this case. If both were visible at once the
    rotation might dispatch the short one first, so there would be no contention for the slot
    at all and the assertion would hold trivially: measured with max_drain of 5, 20 and 30 it
    passed every time, with no discriminating power whatsoever. The short session must only be
    enqueued once the long one has genuinely started consuming.

    Returns (position at which the short session was first consumed, number of long-session
    messages consumed, number of short-session messages consumed).
    """
    mq, lock = MemoryMsgQueue(), MemorySessionLock()
    order: list[str] = []
    guard = threading.Lock()

    class _W:
        def __init__(self, key):
            self._key = key

        def feed_batch(self, msgs, **kw):
            time.sleep(0.01)                    # stretch out per-message handling so the difference in slot occupancy shows
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
        while time.monotonic() < deadline:      # wait until the long session really holds the slot and is consuming
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
    """A chatty session must not starve a session that arrives later -- that is the entire
    reason max_drain exists.

    This checks the effect rather than the implementation: when the valve works the long
    session returns its slot after consuming max_drain messages and the short session can cut
    in; when it does not, the short session waits until the long one is fully drained. The
    criterion is only that the short session is consumed before the long one finishes; no
    particular interleaving is assumed, since thread scheduling is nondeterministic anyway.
    """
    first_short, n_long, n_short = _race_long_vs_short(max_drain=5)
    assert (n_long, n_short) == (30, 3)          # confirm nothing was lost first, or the position assertion below means nothing
    assert first_short < 30, (
        f"the short session was only consumed after all 30 long-session messages "
        f"(first at position {first_short}) -- the slot was monopolised and the fairness valve did not fire")


def test_valve_really_is_what_lets_late_comers_in():
    """A discriminating control: raise max_drain to at least the length of the long session
    (the broken state) and the short session must be pushed to the very end.

    Without this case the assertion above could be trivially true, so a green run would prove
    nothing about the valve. It also reproduces the state production once ran in, where
    max_drain of 20 exceeded a max_queue_depth of 15.
    """
    first_short, _, _ = _race_long_vs_short(max_drain=30)
    assert first_short == 30, (
        f"with the valve disabled the short session should have waited until the end (position 30), "
        f"but was consumed at position {first_short} -- the fairness assertion above has no "
        "discriminating power and this case needs redesigning")
