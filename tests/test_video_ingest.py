"""Mechanism tests for the video consumer side (mocks and spies, no real models): queue
dispatch, ordering, cursor handling, mixed interleaving, and concurrency without crosstalk.

The real identity pipeline is not run — video_ingest.process_clip and finalize_video are
swapped for spies so only the consumer-side plumbing is checked:
- kind dispatch: video goes to process_clip (one clip at a time), session_end goes to
  writer.end_session followed by finalize_video, and ingest goes to feed_batch;
- Mixed session ordering, text then video then text then session_end, plus interleaved batches
  of text, clips, text, clips, end;
- The cursor ends up exactly equal to the message count (nothing lost, nothing repeated), and
  concurrent sessions stay independent (their cursors and calls never cross).
Correctness of the real identity and memory behaviour is covered by
scripts/video/verify_pipeline against real backends.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from personos.ingest_worker import SessionConsumer
from personos.online import video_ingest
from personos.storage.msg_queue import MemoryMsgQueue
from personos.storage.session_lock import MemorySessionLock


class _Writer:
    """A fake SessionWriter that only records feed_batch and end_session calls."""
    def __init__(self, log, sid):
        self.log, self.sid = log, sid

    def feed_batch(self, msgs, **k):
        self.log.append(("feed", self.sid, len(msgs)))

    def end_session(self, **k):
        self.log.append(("end", self.sid))


@pytest.fixture
def spy(monkeypatch):
    """Spies out the real identity pipeline (process_clip / finalize_video) so only dispatch is
    exercised."""
    log = []
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, *, session_id, clip_key="", clip_url="", **k:
                        log.append(("clip", session_id, clip_key or clip_url)))
    monkeypatch.setattr(video_ingest, "finalize_video",
                        lambda deps, *, session_id: log.append(("final", session_id)))
    return log


@pytest.fixture
def spy_deps(monkeypatch):
    """The same, but also records deps, which is what lets cross-user isolation be asserted:
    the deps must belong to the user that message came from."""
    log = []
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, *, session_id, clip_key="", clip_url="", **k:
                        log.append(("clip", deps, session_id, clip_key or clip_url)))
    monkeypatch.setattr(video_ingest, "finalize_video",
                        lambda deps, *, session_id: log.append(("final", deps, session_id)))
    return log


def _consumer(log):
    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer(log, s),
                         video_deps=lambda u: f"deps:{u}")
    return mq, sc


def _txt(speaker="user", text="hi"):
    return {"messages": [{"speaker": speaker, "text": text}]}


def _vid(*keys):
    """The shape for our own object-storage keys (the real endpoint sets an unfilled video_url
    to None, which is faithfully reproduced here)."""
    return {"messages": [{"speaker": "user", "video_oss_key": k, "video_url": None,
                          "clip_index": i} for i, k in enumerate(keys)]}


def _vid_url(*urls):
    """The external-link shape: video_oss_key is None and video_url has a value, matching what
    the endpoint produces.

    (Regression: dict.get('video_oss_key', '') returned None, which raised TypeError and got
    the whole message written off as poisonous.)"""
    return {"messages": [{"speaker": "user", "video_oss_key": None, "video_url": u,
                          "clip_index": i} for i, u in enumerate(urls)]}


def test_mixed_session_dispatch_and_cursor(spy):
    """Text, then video clips, then text, then session_end: dispatch is correct and the cursor
    equals the message count."""
    mq, sc = _consumer(spy)
    U, S = "u1", "s1"
    mq.enqueue(U, S, _txt(text="开场白"), kind="ingest")
    mq.enqueue(U, S, _vid("oss/c0.mp4", "oss/c1.mp4"), kind="video")
    mq.enqueue(U, S, _txt(text="拍完聊两句"), kind="ingest")
    mq.enqueue(U, S, {"task_type": None}, kind="session_end")
    rep = sc.drain_session(U, S)
    assert rep.applied == 4
    # Video is handled one clip at a time, and finalize follows end.
    assert spy == [("feed", S, 1), ("clip", S, "oss/c0.mp4"), ("clip", S, "oss/c1.mp4"),
                   ("feed", S, 1), ("end", S), ("final", S)]
    assert mq.cursor_get(U, S) == 4                            # exactly 4 messages consumed


def test_interleaved_video_bursts(spy):
    """A pathological caller sending text, clip batch 1, text, clip batch 2, session_end: both
    batches reach process_clip, and finalize runs once after end."""
    mq, sc = _consumer(spy)
    U, S = "u2", "s2"
    mq.enqueue(U, S, _txt(text="t1"), kind="ingest")
    mq.enqueue(U, S, _vid("k0.mp4"), kind="video")
    mq.enqueue(U, S, _txt(text="t2"), kind="ingest")
    mq.enqueue(U, S, _vid("k1.mp4"), kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    clips = [c for c in spy if c[0] == "clip"]
    assert clips == [("clip", S, "k0.mp4"), ("clip", S, "k1.mp4")]   # both batches consumed
    assert spy.count(("final", S)) == 1                              # finalized only once
    assert spy[-1] == ("final", S) and spy[-2] == ("end", S)
    assert mq.cursor_get(U, S) == 5


def test_concurrent_sessions_independent(spy):
    """Multiple sessions: each cursor independently equals its own message count, and calls do
    not cross."""
    mq, sc = _consumer(spy)
    U = "u3"
    for s in ("sa", "sb"):
        mq.enqueue(U, s, _vid(f"{s}-c0.mp4"), kind="video")
        mq.enqueue(U, s, {}, kind="session_end")
    sc.drain_session(U, "sa")
    sc.drain_session(U, "sb")
    assert mq.cursor_get(U, "sa") == 2 and mq.cursor_get(U, "sb") == 2
    assert ("clip", "sa", "sa-c0.mp4") in spy and ("clip", "sb", "sb-c0.mp4") in spy
    # sa's clip never carries sb's session_id.
    assert not any(c == ("clip", "sa", "sb-c0.mp4") for c in spy)


def test_video_idempotent_no_double_consume(spy):
    """Redelivery is idempotent: draining the same video message twice is blocked by the
    cursor, so process_clip does not run again."""
    mq, sc = _consumer(spy)
    U, S = "u4", "s4"
    mq.enqueue(U, S, _vid("once.mp4"), kind="video")
    sc.drain_session(U, S)
    sc.drain_session(U, S)   # drain again: the queue is empty and the cursor has advanced
    assert [c for c in spy if c[0] == "clip"] == [("clip", S, "once.mp4")]   # handled only once


def test_bulk_mixed_stress_no_loss_no_dup(spy):
    """Bulk stress (a producer-faster-than-consumer model: flood first, then consume): many
    sessions each with many mixed messages, after which every cursor equals exactly what was
    enqueued, the queues are drained, and every clip was processed exactly once."""
    mq, sc = _consumer(spy)
    U = "ustress"
    sessions = [f"s{i}" for i in range(6)]
    per_session_msgs = {}
    for s in sessions:                                   # flood everything first without consuming
        n = 0
        mq.enqueue(U, s, _txt(text="开"), kind="ingest"); n += 1
        mq.enqueue(U, s, _vid(f"{s}-a.mp4", f"{s}-b.mp4"), kind="video"); n += 1
        mq.enqueue(U, s, _txt(text="中"), kind="ingest"); n += 1
        mq.enqueue(U, s, _vid(f"{s}-c.mp4"), kind="video"); n += 1
        mq.enqueue(U, s, {}, kind="session_end"); n += 1
        per_session_msgs[s] = n
    # Consume: drain each session until empty. drain has a max_drain cap, so loop until
    # more is False.
    for s in sessions:
        while sc.drain_session(U, s).more:
            pass
    for s in sessions:
        assert mq.cursor_get(U, s) == per_session_msgs[s]        # consumed exactly its own, no more and no fewer
        assert mq.depth(U, s) == 0                               # the queue is drained
    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == len(sessions) * 3                       # 3 clips per session, each processed once
    assert len(clips) == len({(c[1], c[2]) for c in clips})      # no duplicates
    assert sum(1 for c in spy if c[0] == "final") == len(sessions)  # each session finalized once


def test_parallel_drain_many_sessions(spy):
    """Genuine concurrency: a thread pool drains many sessions in parallel under lock
    contention, and every cursor is still exact with no clip lost, duplicated, or crossed."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    mq, sc = _consumer(spy)
    U = "upar"
    sessions = [f"p{i}" for i in range(8)]
    for s in sessions:
        mq.enqueue(U, s, _txt(text="a"), kind="ingest")
        mq.enqueue(U, s, _vid(f"{s}-x.mp4", f"{s}-y.mp4"), kind="video")
        mq.enqueue(U, s, {}, kind="session_end")
    # Only guards the spy list itself; list.append is already atomic, so this is belt and braces.
    lock = threading.Lock()

    def work(s):
        with lock:
            pass
        while sc.drain_session(U, s).more:
            pass

    with ThreadPoolExecutor(max_workers=8) as ex:   # 8 threads running genuinely in parallel
        list(ex.map(work, sessions))

    for s in sessions:
        assert mq.cursor_get(U, s) == 3 and mq.depth(U, s) == 0      # each consumed exactly and drained
    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == len(sessions) * 2                            # nothing lost
    assert len(clips) == len({(c[1], c[2]) for c in clips})           # nothing duplicated
    for s in sessions:                                                # nothing crossed: the key prefix matches the session
        assert all(c[2].startswith(s) for c in clips if c[1] == s)
    assert sum(1 for c in spy if c[0] == "final") == len(sessions)


def test_same_session_lock_contention_no_double_consume(spy):
    """Lock contention on one session: several threads drain the same session at once, only
    the one holding the lock processes anything, and no message is consumed twice."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    mq, sc = _consumer(spy)
    U, S = "ulock", "s1"
    for i in range(6):
        mq.enqueue(U, S, _vid(f"c{i}.mp4"), kind="video")
    barrier = threading.Barrier(4)

    def work(_):
        barrier.wait()                       # 4 threads hit the same session simultaneously
        reps = []
        while True:
            r = sc.drain_session(U, S)
            reps.append(r)
            if not r.more:
                break
        return reps

    with ThreadPoolExecutor(max_workers=4) as ex:
        all_reps = list(ex.map(work, range(4)))

    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == 6                                   # all 6 were processed
    assert len(clips) == len({c[2] for c in clips})           # each exactly once, no double consumption
    assert mq.cursor_get(U, S) == 6 and mq.depth(U, S) == 0
    # Some thread really did fail to take the lock, which proves the contention happened.
    assert any(not r.locked for reps in all_reps for r in reps)


def test_multi_user_parallel_no_cross_contamination(spy_deps):
    """MULTI-USER CONCURRENCY (a hard production requirement): N different users each consume
    several sessions in parallel, and (1) the deps handed to every clip must belong to that
    clip's own user, with zero crossover, and (2) each user's per-session cursor is exact."""
    from concurrent.futures import ThreadPoolExecutor

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: f"deps:{u}")   # deps carry the user identity, for the isolation assertion
    users = [f"u{i}" for i in range(6)]
    sessions = ["a", "b"]
    for u in users:
        for s in sessions:
            mq.enqueue(u, s, _vid(f"{u}-{s}-1.mp4", f"{u}-{s}-2.mp4"), kind="video")
            mq.enqueue(u, s, {}, kind="session_end")

    jobs = [(u, s) for u in users for s in sessions]
    with ThreadPoolExecutor(max_workers=12) as ex:          # 12 threads running across users in parallel
        list(ex.map(lambda j: [None for _ in iter(lambda: sc.drain_session(*j).more, False)], jobs))

    clips = [c for c in spy_deps if c[0] == "clip"]
    assert len(clips) == len(users) * len(sessions) * 2                  # nothing lost
    assert len(clips) == len({c[3] for c in clips})                      # nothing duplicated
    for _tag, deps, _sid, keyname in clips:                             # CROSS-USER ISOLATION
        owner = keyname.split("-")[0]                                    # the key prefix identifies the owning user
        assert deps == f"deps:{owner}", f"users crossed: {deps} handled {keyname}"
    for _tag, deps, sid in [c for c in spy_deps if c[0] == "final"]:
        assert deps.startswith("deps:u")
    for u in users:
        for s in sessions:
            assert mq.cursor_get(u, s) == 2 and mq.depth(u, s) == 0      # each exact and drained


def test_video_url_payload_consumed(spy):
    """The external-link shape (video_oss_key=None plus video_url) is consumed normally — a
    regression test for the None[-16:] crash that got these written off as poisonous."""
    mq, sc = _consumer(spy)
    U, S = "uurl", "s1"
    mq.enqueue(U, S, _vid_url("https://caller-bucket/a.mp4", "https://caller-bucket/b.mp4"),
               kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    rep = sc.drain_session(U, S)
    assert rep.applied == 2 and rep.poisoned == 0          # no longer written off as poisonous
    assert [c for c in spy if c[0] == "clip"] == [
        ("clip", S, "https://caller-bucket/a.mp4"), ("clip", S, "https://caller-bucket/b.mp4")]
    assert mq.cursor_get(U, S) == 2


def test_mixed_url_and_key_bursts(spy):
    """Mixing the two sources (batch 1 external links, batch 2 our own keys) both consume fine
    and the cursor stays exact."""
    mq, sc = _consumer(spy)
    U, S = "umix", "s1"
    mq.enqueue(U, S, _vid_url("https://caller/x.mp4"), kind="video")
    mq.enqueue(U, S, _vid("ourkey/y.mp4"), kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    assert [c[2] for c in spy if c[0] == "clip"] == ["https://caller/x.mp4", "ourkey/y.mp4"]
    assert mq.cursor_get(U, S) == 3


def test_clip_rejected_is_recorded_not_silently_dropped(monkeypatch):
    """A permanently failing clip (a dead external link, oversized, or too long) is RECORDED
    and then skipped, with no retry and nothing silent; the rest of the batch carries on."""
    from personos.online import video_ingest

    recorded = []

    class _TS:   # a fake TaskStore that records the calls leaving a trace
        def create(self, tid, kind, u, s):
            recorded.append(("create", kind, u, s))

        def mark_error(self, tid, err):
            recorded.append(("error", err))

    def _proc(deps, *, session_id, clip_key="", clip_url="", **k):
        src = clip_key or clip_url
        if "bad" in src:
            raise video_ingest.ClipRejected(f"video address not reachable (HTTP 403): {src}")
        recorded.append(("ok", src))

    monkeypatch.setattr(video_ingest, "process_clip", _proc)
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", task_store=_TS())
    U, S = "urej", "s1"
    mq.enqueue(U, S, _vid_url("https://ok/a.mp4", "https://bad/b.mp4", "https://ok/c.mp4"),
               kind="video")
    rep = sc.drain_session(U, S)

    # The message as a whole counts as a success (no retry, not poisonous).
    assert rep.applied == 1 and rep.poisoned == 0
    # The rest carried on.
    assert ("ok", "https://ok/a.mp4") in recorded and ("ok", "https://ok/c.mp4") in recorded
    assert ("create", "video_clip_rejected", U, S) in recorded                                # a trace was left
    assert any(t == "error" and "403" in e for t, e in
               [(r[0], r[1]) for r in recorded if r[0] == "error"])                           # the reason is recoverable
    assert mq.cursor_get(U, S) == 1


def test_lock_renewed_during_long_clip_processing(monkeypatch):
    """LOCK RENEWAL: when a single clip takes far longer than the renewal interval (2-3 minutes
    is normal for video), the heartbeat has to keep renewing the lock, and no other consumer
    can take it during that time (so the same session is never consumed concurrently)."""
    import threading
    import time

    from personos.online import video_ingest

    renews: list[float] = []

    class _SpyLock(MemorySessionLock):
        def renew(self, u, s, handle):
            renews.append(time.monotonic())
            return super().renew(u, s, handle)

    PROC_S, RENEW_S = 0.9, 0.15          # processing time far exceeds the renewal interval, so it should renew repeatedly
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, **k: time.sleep(PROC_S))
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq, lock = MemoryMsgQueue(), _SpyLock()
    sc = SessionConsumer(mq, lock, lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", renew_interval_s=RENEW_S)
    U, S = "ulongclip", "s1"
    mq.enqueue(U, S, _vid("slow.mp4"), kind="video")

    stolen = []

    def intruder():                       # while processing is under way, another consumer tries to take the session
        time.sleep(PROC_S / 2)
        stolen.append(lock.try_acquire(U, S))

    t = threading.Thread(target=intruder)
    t.start()
    rep = sc.drain_session(U, S)
    t.join()

    assert rep.applied == 1                                   # completes normally despite the long runtime
    assert len(renews) >= 2, \
        f"not enough lock renewals: {len(renews)} (processing {PROC_S}s / interval {RENEW_S}s)"
    assert stolen == [None], \
        "the lock was stolen during processing, which would mean concurrent consumption of one session"


class _FakeResp:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status_code, self.headers, self._chunks = status, headers or {}, chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield from self._chunks


def test_fetch_rejects_http_error(monkeypatch):
    """A 4xx or 5xx on an external link becomes ClipRejected (a permanent failure, no retry)."""
    import httpx

    from personos.online import video_ingest

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(status=403))
    with pytest.raises(video_ingest.ClipRejected, match="not reachable"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://expired/a.mp4",
                                       owner="u")


def test_fetch_rejects_oversize_declared(monkeypatch):
    """A Content-Length declaring more than the limit is rejected immediately, without
    downloading."""
    import httpx

    from personos.online import video_ingest

    big = str(video_ingest.MAX_CLIP_BYTES + 1)
    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: _FakeResp(headers={"content-length": big}))
    with pytest.raises(video_ingest.ClipRejected, match="byte limit"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/big.mp4",
                                       owner="u")


def test_fetch_rejects_oversize_while_streaming(monkeypatch):
    """No Content-Length declared but the limit is exceeded mid-download: stop while streaming
    rather than reading an oversized file fully into memory."""
    import httpx

    from personos.online import video_ingest

    monkeypatch.setattr(video_ingest, "MAX_CLIP_BYTES", 10)
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(chunks=[b"x" * 6, b"x" * 6]))
    with pytest.raises(video_ingest.ClipRejected, match="exceeded mid-download"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/nolen.mp4",
                                       owner="u")


def test_temp_file_cleaned_on_reject(monkeypatch, tmp_path):
    """A rejection (oversized, too long, or a failed download) leaves no stray temp file — the
    failure path cleans up too."""
    import httpx

    from personos.online import video_ingest

    before = set(os.listdir(tempfile.gettempdir()))
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(status=404))
    with pytest.raises(video_ingest.ClipRejected):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/404.mp4",
                                       owner="u")
    leaked = [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith(".mp4")]
    assert not leaked, f"the failure path leaked temp files: {leaked}"


def test_video_concurrency_is_bounded_only_by_the_pool(monkeypatch):
    """Video concurrency is governed solely by the thread pool size — there must not be a
    second gate inside process_clip.

    Regression: the pool was once 50 while the function also held a BoundedSemaphore(4), so the
    stricter of the two always won and 46 threads sat waiting on the gate, each still holding a
    session lock. Two gates controlling the same thing means one of them should not exist.
    Here 8 clips run through an 8-wide pool; if a gate remained inside the function, peak
    parallelism would be squeezed down to its size.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from personos.online import video_ingest

    live, peak, lk = 0, 0, threading.Lock()

    def _slow(deps, **k):
        nonlocal live, peak
        with lk:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lk:
            live -= 1
        return 0

    monkeypatch.setattr(video_ingest, "_process_clip_locked", _slow)
    deps = object()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: video_ingest.process_clip(deps, session_id="s", clip_key=f"{i}.mp4"),
                    range(8)))
    assert peak == 8, \
        f"the pool offered 8 threads but only {peak} ran in parallel — a gate is still left inside the function"


def _fake_deps(monkeypatch, omni_calls):
    """Build a minimal VideoDeps (recording how many times omni was called), for the duration
    and ordering assertions."""
    from personos.identity.draft import MemoryDraftStore
    from personos.online import video_ingest

    class _Omni:
        def chat(self, *a, **k):
            omni_calls.append(1)
            return "{}"

    class _MS:
        def read_bytes(self, key):
            return b"fakebytes"

        def sign_url(self, key):
            return f"https://our-oss/{key}"

    return video_ingest.VideoDeps(
        store=type("S", (), {"user_id": "u"})(), cloud=None, draft=MemoryDraftStore("u"),
        backends={"mm_runner": _Omni()}, media_store=_MS(), llm=None, embedder=None,
        evidence=None, cells=None, atoms=None, chains=None)


def test_url_path_rejects_too_long_before_spending_mllm(monkeypatch):
    """The external-link path already has the file in hand, so CHECK THE DURATION BEFORE
    RUNNING THE SCREENPLAY: an over-long clip is rejected immediately, saving a 2-3 minute
    multimodal call."""
    from personos.online import video_ingest

    calls = []
    monkeypatch.setattr(video_ingest, "MAX_CLIP_DURATION_S", 5.0)
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 99.0)
    monkeypatch.setattr(video_ingest, "_materialize_clip",
                        lambda ms, *, clip_key, clip_url, owner: ("/tmp/fake.mp4", "our/k.mp4"))
    deps = _fake_deps(monkeypatch, calls)
    with pytest.raises(video_ingest.ClipRejected, match="video duration"):
        video_ingest.process_clip(deps, session_id="s", clip_url="https://caller/a.mp4")
    assert calls == [], "the external-link path should reject an over-long clip before running the screenplay"


def test_key_path_downloads_only_after_screenplay(monkeypatch):
    """Our own key path: nothing is on disk while the screenplay runs, and the download only
    happens when harvest needs it, keeping the temp file alive for the shortest possible time."""
    from personos.online import video_ingest

    calls, order = [], []
    monkeypatch.setattr(video_ingest, "MAX_CLIP_DURATION_S", 5.0)
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 99.0)

    def _mat(ms, *, clip_key, clip_url, owner):
        order.append(f"download(omni_calls={len(calls)})")
        return "/tmp/fake.mp4", clip_key or "our/k.mp4"

    monkeypatch.setattr(video_ingest, "_materialize_clip", _mat)
    deps = _fake_deps(monkeypatch, calls)
    with pytest.raises(video_ingest.ClipRejected, match="video duration"):
        video_ingest.process_clip(deps, session_id="s", clip_key="k.mp4")
    assert order == ["download(omni_calls=1)"], f"the download should happen after the screenplay: {order}"


def test_session_end_without_video_still_finalizes_noop(spy):
    """session_end on a text-only session still calls finalize_video, which sees no pending
    work and becomes a no-op rather than an error."""
    mq, sc = _consumer(spy)
    U, S = "u5", "s5"
    mq.enqueue(U, S, _txt(), kind="ingest")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    # finalize is always called (it is a no-op in the spy).
    assert ("end", S) in spy and ("final", S) in spy


def test_dispatcher_routes_video_to_dedicated_pool():
    """A DEDICATED VIDEO POOL: a session whose queue head is video goes to the video pool and a
    text session goes to the text pool, so a busy video workload does not occupy text workers
    and starve text consumption."""
    from personos.ingest_worker import Dispatcher

    submitted = {"text": [], "video": []}

    class _Pool:
        def __init__(self, tag):
            self.tag = tag

        def submit(self, fn, *a):
            submitted[self.tag].append(a[:2])

            class _F:
                pass
            return _F()

    mq = MemoryMsgQueue()
    mq.enqueue("u", "s_txt", _txt(), kind="ingest")
    mq.enqueue("u", "s_vid", _vid("a.mp4"), kind="video")
    d = Dispatcher(mq, object(), _Pool("text"), 10,
                   video_pool=_Pool("video"), video_cap=10)
    d.run_once()
    assert submitted["video"] == [("u", "s_vid")], f"the video session did not reach the video pool: {submitted}"
    assert submitted["text"] == [("u", "s_txt")], f"the text session did not reach the text pool: {submitted}"


def test_video_pool_full_does_not_block_text():
    """When the video pool is full, text sessions can still be dispatched — the whole point of
    the isolation."""
    from personos.ingest_worker import Dispatcher

    got = []

    class _Pool:
        def __init__(self, tag):
            self.tag = tag

        def submit(self, fn, *a):
            got.append((self.tag, a[1]))

            class _F:
                pass
            return _F()

    mq = MemoryMsgQueue()
    for i in range(3):
        mq.enqueue("u", f"v{i}", _vid(f"{i}.mp4"), kind="video")
    mq.enqueue("u", "t1", _txt(), kind="ingest")
    d = Dispatcher(mq, object(), _Pool("text"), 10,
                   video_pool=_Pool("video"), video_cap=1)   # the video pool has only one slot
    d.run_once()
    assert ("text", "t1") in got, f"a full video pool also blocked text: {got}"
    assert sum(1 for t, _ in got if t == "video") == 1        # only one video was dispatched, as capped


def test_poisoned_message_is_recorded(monkeypatch):
    """When retries for a transient fault run out and the message is written off as poisonous,
    a TRACE must still be left — otherwise the memory for that clip is gone and all that
    remains is a log line that will eventually be rotated away."""
    from personos.online import video_ingest

    recorded = []

    class _TS:
        def create(self, tid, kind, u, s):
            recorded.append(("create", kind))

        def mark_error(self, tid, err):
            recorded.append(("error", err[:40]))

    def _boom(deps, **k):
        raise RuntimeError("Omni read timeout")      # a transient fault, not a ClipRejected

    monkeypatch.setattr(video_ingest, "process_clip", _boom)
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", task_store=_TS(), max_retries=2)
    U, S = "upoison", "s1"
    mq.enqueue(U, S, _vid("boom.mp4"), kind="video")
    for _ in range(3):                                # retry until it is written off as poisonous
        try:
            sc.drain_session(U, S)
        except Exception:
            pass
    assert ("create", "poisoned_video") in recorded, f"being written off left no trace: {recorded}"
    assert any(t == "error" for t, _ in recorded)


def test_upstream_media_failures_become_clip_rejected(monkeypatch):
    """Upstream "cannot fetch the media" and "content review refused" become ClipRejected
    (recorded and skipped) rather than being retried 5 times as a transient fault.

    Regression for two real incidents:
    - A 2-minute clip at 7.3Mbps (106MB) made the model service time out downloading it and
      return 400;
    - The deployment image lacked PyAV, so `import av` raised ModuleNotFoundError.
    Both were treated as transient faults and retried 5 times, and because the failure point on
    our own key path comes AFTER the screenplay multimodal call, each round wasted a 2-minute
    screenplay call first. Five rounds burned ten minutes of upstream quota and the message was
    written off as poisonous anyway.
    """
    from personos.identity.backends.omni import ContentRejectedError, MediaUnfetchableError
    from personos.online import video_ingest

    for exc in (MediaUnfetchableError("Download multimodal file timed out"),
                ContentRejectedError("data_inspection_failed"),
                ModuleNotFoundError("No module named 'av'")):
        def _boom(*a, **k):
            raise exc

        monkeypatch.setattr(video_ingest, "_process_clip_locked", _boom)
        with pytest.raises(video_ingest.ClipRejected):
            video_ingest.process_clip(object(), session_id="s", clip_key="k.mp4")


def test_omni_maps_media_download_failure_to_its_own_error():
    """The model provider's `Download multimodal file timed out` must land on
    MediaUnfetchableError rather than being swallowed by the generic 4xx branch as an ordinary
    HTTPStatusError, because that would leave the layer above unable to tell whether to
    retry."""
    import httpx

    from personos.identity.backends import omni as omni_mod

    body = ('{"detail":"{\\"error\\":{\\"message\\":\\"<400> InternalError.Algo.'
            'InvalidParameter: Download multimodal file timed out\\"}}"}')

    class _Resp:
        status_code = 400
        text = body
        request = None

        def raise_for_status(self):
            raise httpx.HTTPStatusError("400", request=None, response=self)

    class _Client:
        def post(self, url, **kw):
            return _Resp()

    r = omni_mod.OmniRunner()
    r._client = _Client()
    r.cfg = type("C", (), {"mllm_api_key": "k", "mllm_endpoint": "http://x",
                           "mllm_timeout": 120.0})()
    with pytest.raises(omni_mod.MediaUnfetchableError):
        r.chat("p", video_url="https://oss/a.mp4")


def _idem_deps(monkeypatch, calls, fail_on=None):
    """Build the minimal deps needed to run _process_clip_locked end to end, recording how many
    times each clip was genuinely processed."""
    from personos.identity.draft import MemoryDraftStore
    from personos.identity.screenplay import ClipScript
    from personos.online import video_ingest

    class _Omni:
        def chat(self, *a, **k):
            return "{}"

    class _MS:
        def read_bytes(self, key):
            return b"x"

        def sign_url(self, key):
            return f"https://our/{key}"

    def _parse(raw, duration_sec=None):
        return ClipScript(cast_map={}, parsed_ok=True)

    monkeypatch.setattr(video_ingest, "parse_clip_output", _parse)
    monkeypatch.setattr(video_ingest, "build_clip_prompt", lambda **k: ("p", []))
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 10.0)
    monkeypatch.setattr(video_ingest, "_materialize_clip",
                        lambda ms, *, clip_key, clip_url, owner: ("", clip_key or "our/k.mp4"))

    def _harvest(path, script, backends):
        calls.append(path)
        if fail_on is not None and len(calls) == fail_on:
            # Not a ClipRejected, so the whole message should be redelivered.
            raise RuntimeError("瞬时故障")
        return {}

    monkeypatch.setattr(video_ingest, "harvest_clip", _harvest)
    return video_ingest.VideoDeps(
        store=type("S", (), {"user_id": "u"})(), cloud=None, draft=MemoryDraftStore("u"),
        backends={"mm_runner": _Omni()}, media_store=_MS(), llm=None, embedder=None,
        evidence=None, cells=None, atoms=None, chains=None)


def test_replayed_clip_is_skipped_not_double_counted(monkeypatch):
    """A replayed clip is genuinely processed only once and returns its original clip_index.

    Regression for a real defect: queue redelivery replays the WHOLE MESSAGE (up to 20 clips),
    so one transient fault on a clip near the end of the batch made every clip before it run
    again — presence counted twice, lines written into the memcell twice, and material uploaded
    twice.
    """
    from personos.online import video_ingest

    calls = []
    deps = _idem_deps(monkeypatch, calls)
    i1 = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")
    i2 = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")   # replay
    assert (i1, i2) == (0, 0), f"a replay should return the original index: {i1}, {i2}"
    assert len(calls) == 1, f"a replay should not run harvest again: {calls}"
    assert deps.draft.clip_keys("s") == {0: "a.mp4"}


def test_failed_clip_is_retried_not_marked_done(monkeypatch):
    """A clip that failed partway MUST be able to run again — deduplication hangs off
    "completed", not off "an index was allocated".

    If it hung off next_clip_seq (allocated before the screenplay multimodal call), a failed
    clip would be mistaken for already processed on redelivery and skipped, trading "duplicate
    bookkeeping" for "silently losing the memory", which is worse than the original defect.
    This test locks out that worse implementation.
    """
    from personos.online import video_ingest

    calls = []
    deps = _idem_deps(monkeypatch, calls, fail_on=1)      # the first attempt raises a transient fault
    with pytest.raises(RuntimeError, match="瞬时故障"):
        video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")
    assert deps.draft.clip_done_index("s", "a.mp4") is None, "a failed clip must not be marked complete"

    idx = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")   # redelivered and retried
    assert len(calls) == 2, f"a failed clip must genuinely run again: {calls}"
    assert idx == 1, "the retry takes a new index; the failed one is void and, having no lines, does not affect flush"
    assert deps.draft.clip_done_index("s", "a.mp4") == 1
