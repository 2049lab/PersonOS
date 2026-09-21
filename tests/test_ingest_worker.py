"""Consumer and dispatcher unit tests: ordering, single-flight, crash replay, dedup, the fairness
cap, draining, and pool isolation.

Uses MemoryMsgQueue, MemorySessionLock and a fake writer that records the feed order, so no real LLM
or Redis is involved; the Redis variant of the lock is covered separately in test_session_lock, and
these tests focus on the consumption orchestration logic.
"""

from __future__ import annotations

import threading
import time

from personos.ingest_worker import Dispatcher, SessionConsumer
from personos.storage.msg_queue import MemoryMsgQueue
from personos.storage.session_lock import MemorySessionLock


def _p(text: str) -> dict:
    """An ingest payload: one queue message is one atomic batch (one message per batch here, which
    makes the ordering easy to check)."""
    return {"messages": [{"speaker": "user", "text": text}]}


class FakeWriter:
    """Records the order of feed_batch and end_session calls; a barrier can be injected to simulate
    slow consumption for the single-flight test."""

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
    assert ("u", "s") not in mq.active_sessions()      # once drained it leaves the active board


def test_cursor_advances():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    mq.enqueue("u", "s", _p("b"))
    consumer.drain_session("u", "s")
    assert mq.cursor_get("u", "s") == 2                # the cursor advanced to the highest seq
    assert [t for _, t in log] == ["a", "b"]


def test_crash_leftover_replayed_via_recover():
    """Messages reserved but never acked (simulating a crash mid-drain) are replayed by recover at the
    start of the next drain, so nothing is lost or reordered."""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    mq.enqueue("u", "s", _p("b"))
    mq.reserve("u", "s")                               # moved into proc but never handled (crash)
    mq.reserve("u", "s")
    assert mq.depth("u", "s") == 0 and not mq.is_empty("u", "s")   # both in flight, unacked
    rep = consumer.drain_session("u", "s")             # should replay a and b out of proc
    assert rep.applied == 2
    assert [t for _, t in log] == ["a", "b"]
    assert mq.is_empty("u", "s")


def test_recover_dedup_when_cursor_already_past():
    """If an in-flight message's seq is already <= the cursor (it was applied last time and only the ack
    was lost), recover treats it as a redelivery and skips it instead of applying it twice."""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("a"))
    env = mq.reserve("u", "s")
    mq.cursor_set("u", "s", env.seq)                   # simulate "applied but crashed before the ack"
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 0 and rep.skipped == 1       # recognized as a redelivery and skipped
    assert log == []                                   # not fed a second time
    assert mq.is_empty("u", "s")


def test_max_drain_fairness_leaves_more():
    """A single lock acquisition consumes at most max_drain messages and leaves the rest in the queue
    (more=True), so one session cannot hog a thread."""
    mq, lock, consumer, log = _make(max_drain=3)
    for i in range(10):
        mq.enqueue("u", "s", _p(f"m{i}"))
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 3 and rep.more is True
    assert mq.depth("u", "s") == 7
    assert ("u", "s") in mq.active_sessions()          # not drained, so still on the active board
    consumer.drain_session("u", "s")                   # pick up where it left off, order continues
    assert [t for _, t in log][:6] == ["m0", "m1", "m2", "m3", "m4", "m5"]


def test_single_flight_second_drain_bounces():
    """While one thread holds the lock and consumes slowly, another thread draining the same session
    must fail to take the lock (locked=False), so there is no concurrency and no reordering."""
    barrier = threading.Event()
    mq, lock, consumer, log = _make(barrier=barrier)
    mq.enqueue("u", "s", _p("slow"))

    reports: list = []
    t1 = threading.Thread(target=lambda: reports.append(consumer.drain_session("u", "s")))
    t1.start()
    time.sleep(0.1)                                    # let t1 take the lock and block on the barrier
    rep2 = consumer.drain_session("u", "s")            # t2: same session, must not get the lock
    assert rep2.locked is False and rep2.applied == 0
    barrier.set()
    t1.join(3.0)
    assert reports[0].locked and reports[0].applied == 1


def test_transient_failure_retries_not_lost():
    """A transient failure such as a network blip must not lose a message: the first feed raises, the
    message stays in flight, and the next round's recover replays it successfully."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []
    calls = {"n": 0}

    class FlakyWriter:
        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated transient failure, e.g. a blip on the model gateway")
            for m in msgs:
                log.append(m.text)

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: FlakyWriter(), max_drain=5)
    mq.enqueue("u", "s", _p("重要消息"))
    rep1 = consumer.drain_session("u", "s")            # first round: feed raises
    assert rep1.applied == 0 and not mq.is_empty("u", "s")   # not applied, message still in flight (not lost)
    rep2 = consumer.drain_session("u", "s")            # second round: recover replays it and succeeds
    assert rep2.applied == 1 and log == ["重要消息"]
    assert mq.is_empty("u", "s")                       # consumed exactly once in the end


def test_poison_message_skipped_after_max_retries():
    """A poison message that always fails is skipped once it hits the retry limit: the cursor advances,
    the head-of-line block is cleared, and the messages behind it are consumed."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []

    class PoisonWriter:
        """The first (poison) message always blows up; the rest are handled normally."""
        def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
            if msgs[0].text == "毒":
                raise ValueError("a poison message that always fails")
            for m in msgs:
                log.append(m.text)

        def end_session(self, *, task_type=None, scenario=""):
            pass

    consumer = SessionConsumer(mq, lock, lambda u, s: PoisonWriter(), max_drain=10, max_retries=3)
    mq.enqueue("u", "s", _p("毒"))          # seq=1, the poison one
    mq.enqueue("u", "s", _p("正常1"))       # seq=2
    mq.enqueue("u", "s", _p("正常2"))       # seq=3

    # The first max_retries-1 rounds: the poison message sits at the head and is retried, so nothing
    # behind it can be consumed (head-of-line blocking)
    for _ in range(consumer._max_retries - 1):
        rep = consumer.drain_session("u", "s")
        assert rep.applied == 0 and log == []          # the poison message blocks; the good ones have not had a turn
    # Round max_retries: declared poison, skipped, and the good messages behind it are consumed
    rep = consumer.drain_session("u", "s")
    assert rep.poisoned == 1
    assert log == ["正常1", "正常2"]                    # head-of-line unblocked, the rest consumed exactly once
    assert mq.is_empty("u", "s")
    assert mq.cursor_get("u", "s") == 3                # the cursor moved past the poison message (seq=1) and the rest


def test_poison_counter_resets_across_distinct_messages():
    """Failure counts are per msg_id: skipping one poison message does not eat into another message's
    retry budget."""
    mq = MemoryMsgQueue()
    m1, _ = mq.enqueue("u", "s", _p("a"))
    m2, _ = mq.enqueue("u", "s", _p("b"))
    assert mq.mark_failed("u", "s", m1) == 1
    assert mq.mark_failed("u", "s", m1) == 2
    assert mq.mark_failed("u", "s", m2) == 1           # m2 counts independently, unaffected by m1


def test_heartbeat_renews_lock_during_long_apply():
    """The H2 fix: the heartbeat renews the lock *during* one very long apply, rather than only after
    it finishes, so the lock cannot expire mid-apply and let another replica take over."""
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
                time.sleep(0.15)                           # one very long message (longer than the renew interval)
            def end_session(self, *, task_type=None, scenario=""):
                pass
        return W()

    consumer = SessionConsumer(mq, lock, wf, max_drain=5, renew_interval_s=0.05)
    mq.enqueue("u", "s", _p("x"))
    consumer.drain_session("u", "s")
    assert lock.renews >= 1                                # the heartbeat renewed at least once while processing


def test_after_drain_hook_fires_only_when_applied():
    """The end-of-drain hook fires only when messages were actually stored, and carries the user and
    session; it does not fire when the lock was not acquired or there was nothing to do."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    calls: list = []
    consumer = SessionConsumer(mq, lock, lambda u, s: FakeWriter([], (u, s)),
                               after_drain=lambda u, s, sc="": calls.append((u, s)))
    # Draining an empty session applies no messages, so there is no callback
    consumer.drain_session("u", "empty")
    assert calls == []
    # With a message, the hook fires once with the right user and session
    mq.enqueue("u", "s", _p("hi"))
    consumer.drain_session("u", "s")
    assert calls == [("u", "s")]


def test_after_drain_hook_exception_does_not_break_consume():
    """An exception from the hook must not affect the consumption result -- a failed profile trigger
    can never drag down ingest."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()
    log: list = []

    def boom(u, s, sc=""):
        raise RuntimeError("the profile trigger blew up")

    consumer = SessionConsumer(mq, lock, lambda u, s: FakeWriter(log, (u, s)), after_drain=boom)
    mq.enqueue("u", "s", _p("hi"))
    rep = consumer.drain_session("u", "s")
    assert rep.applied == 1 and [t for _, t in log] == ["hi"]   # consumption still succeeds


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
    """The real pipeline: enqueue a batch, drain it, and let a real SessionWriter store the evidence
    (W0). This checks that the consumption pipeline is actually wired to the write side.

    A single batch into the first segment: seg is empty, so detect_boundary short-circuits without
    calling the LLM and build_cell never fires, which is why no LLM responses are needed.
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
    assert [r.content_inline for r in recs] == ["我明天要去打篮球"]        # the W0 evidence is stored
    assert [r.content_inline for r in seg.load("u_it", "sess")] == ["我明天要去打篮球"]  # the segment is persisted


# —— Dispatcher ——

class _InlinePool:
    """A fake pool that runs synchronously: submit finishes immediately, so the dispatch logic can be
    tested without thread timing."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        fn(*args)


class _HoldPool:
    """Stores jobs instead of running them right away, for the in-flight dedup test: run_all releases
    them manually."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


def test_dispatcher_inflight_dedup_no_redundant_submit():
    """A session already dispatched is not dispatched again: while its job is unfinished, another
    run_once will not submit the same session twice -- this was the root cause of starvation."""
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s", _p("x"))
    pool = _HoldPool()
    disp = Dispatcher(mq, consumer, pool, cap=4)
    assert disp.run_once() == 1                        # s was dispatched (pending, not yet executed)
    assert disp.run_once() == 0                        # s is in flight, so it is not dispatched again
    pool.run_all()                                     # release: the drain completes and clears in-flight
    assert mq.is_empty("u", "s")
    mq.enqueue("u", "s", _p("y"))                       # a new message makes it dispatchable again
    assert disp.run_once() == 1


def test_two_consumers_single_flight_cross_pod():
    """Two consumers share one queue and one lock (simulating two pods): concurrent consumption of the
    same session is still strictly ordered and exactly once."""
    mq = MemoryMsgQueue()
    lock = MemorySessionLock()                         # a shared lock stands in for the cross-pod Redis lock semantics
    log: list = []
    guard = threading.Lock()

    def wf(u, s):
        class W:
            def feed_batch(self, msgs, *, now_dt=None, source_extra=None, task_type=None, scenario=""):
                time.sleep(0.002)                      # widen the concurrency window
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
    assert log == list(range(30))                      # two pods competing, still ordered with no duplicates and no losses


def test_dispatcher_run_once_drains_active_sessions():
    mq, lock, consumer, log = _make()
    mq.enqueue("u", "s1", _p("a"))
    mq.enqueue("u", "s2", _p("b"))
    pool = _InlinePool()
    disp = Dispatcher(mq, consumer, pool, cap=10)
    n = disp.run_once()
    assert n == 2 and pool.submitted == 2
    assert {t for _, t in log} == {"a", "b"}
    assert disp.run_once() == 0                        # everything drained, the board is empty, nothing more to dispatch


def test_dispatcher_backpressure_when_pool_full():
    """Pool capacity 1 and jobs that never give their slot back (a full pool): at most one dispatch per
    round, so work does not pile up without bound."""
    mq, lock, consumer, log = _make()
    for i in range(5):
        mq.enqueue("u", f"s{i}", _p(f"m{i}"))

    class _NoLeavePool:
        def submit(self, fn, *args):
            pass                                       # never runs, so the gate slot is never returned

    disp = Dispatcher(mq, consumer, _NoLeavePool(), cap=1)
    assert disp.run_once() == 1                        # one dispatch fills the pool
    assert disp.run_once() == 0                        # still full, nothing can be dispatched
