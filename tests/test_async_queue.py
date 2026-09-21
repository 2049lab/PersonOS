"""The library's write path is the same ordered-consumption queue the server uses.

add()/end_session() enqueue and return a receipt; a background dispatcher
drains per-session FIFO. These tests run the real queue (in-memory
implementation), the real dispatcher and the real consumer, with fake model
clients — so what is verified is the machinery, not the model.

Covered: receipt contract, read-your-own-writes via flush/wait, multi-session
concurrent writes without cross-contamination, reads while a write is in
flight, backpressure, poison-message handling, and entry-level warnings.

These tests need a real (throwaway) database file, because consumption happens
on dispatcher threads and the suite's shared rollback scope pins a single
connection — so the test guard is lifted here and everything lands in tmp_path.
"""

from __future__ import annotations

import threading
import time

import pytest

from personos import Memory, QueueBusy
from personos.config import Config, reset_config, set_config

from .fakes import FakeEmbedder, FakeLLM


def _pipeline_response(prompt: str) -> str:
    """One callable answering every write-path prompt with a valid shape."""
    if "Current segment turns:" in prompt:                    # W1 boundary
        return '{"should_end": false, "confidence": 0.1}'
    if "Full segment transcript" in prompt:                   # W2 step 1: weave
        return '{"topic": "test topic", "episode": "something happened", "domains": []}'
    if "Full original transcript" in prompt:                  # W2 step 2: atoms
        return ('{"atoms": [{"text": "a fact was said", "object_type": "fact", '
                '"holder": "user", "kind": "K04", "domains": [], "when": null, '
                '"quote": "a fact"}]}')
    return '{"ops":[]}'                                       # chain assignment etc.


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A Memory on a throwaway SQLite file with fake models, guard lifted."""
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    set_config(Config(
        data_dir=tmp_path,          # SQLite lives under here; db_url stays empty
        log_dir=tmp_path / "logs",
        dispatcher_tick_s=0.02, dispatcher_idle_tick_s=0.05,
    ))
    m = Memory()
    m.llm = FakeLLM([])
    # FakeLLM pops its queue; we want one callable answering every prompt.
    m.llm.chat = lambda messages, **kw: _pipeline_response(messages[-1]["content"])
    m.llm.available = True
    m.embedder = FakeEmbedder()
    m.embedder.available = True
    yield m
    m.close()
    reset_config()


# ── receipt contract ────────────────────────────────────────────────────

def test_add_returns_a_receipt_and_consumes_in_the_background(live):
    r = live.add([{"role": "user", "content": "I keep neon tetras."}],
                 user_id="u1", session_id="s1")
    assert r.accepted and r.kind == "ingest" and r.seq == 1

    live.flush(user_id="u1", session_id="s1", timeout_s=30)
    assert live.queue_status(user_id="u1", session_id="s1")["cursor"] >= r.seq
    evidence = live.for_user("u1").evidence.list(limit=10)
    assert any("neon tetras" in (e.content_inline or "") for e in evidence)


def test_end_session_wait_gives_read_your_own_writes(live):
    live.add("I am a backend engineer", user_id="u1", session_id="s1")
    live.end_session(user_id="u1", session_id="s1", sync=True, timeout_s=30)

    cells = live.for_user("u1").cells.iter_all()
    assert len(cells) == 1 and cells[0].episode


def test_queue_status_reports_depth_and_cursor(live):
    live.add("one", user_id="u1", session_id="s1")
    live.flush(user_id="u1", session_id="s1", timeout_s=30)
    status = live.queue_status(user_id="u1", session_id="s1")
    assert status == {"depth": 0, "cursor": 1}


# ── concurrency: many sessions at once, no cross-contamination ──────────

def test_concurrent_writes_across_sessions_and_users(live):
    """6 sessions across 3 users write at the same time; every batch must land
    exactly once, under the user and session it belonged to."""
    def work(user: str, session: str, tag: str):
        for i in range(3):
            live.add(f"{tag} fact number {i}", user_id=user, session_id=session)
        live.end_session(user_id=user, session_id=session, sync=True, timeout_s=60)

    jobs = [(f"user{n}", f"sess{k}", f"user{n}-sess{k}")
            for n in range(3) for k in range(2)]
    threads = [threading.Thread(target=work, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), "a writer thread hung"

    for user, session, tag in jobs:
        ctx = live.for_user(user)
        mine = [e for e in ctx.evidence.list(limit=100)
                if (e.content_inline or "").startswith(tag)]
        assert len(mine) == 3, f"{user}/{session}: expected 3, got {len(mine)}"
        # Nobody else's text may appear under this user.
        for e in ctx.evidence.list(limit=100):
            assert (e.content_inline or "").startswith(f"{user}-")


def test_search_works_while_another_session_is_writing(live):
    """Reads take no session lock: a recall on one user must return even while
    another user's queue is mid-drain."""
    blocker = threading.Event()

    def slow_response(prompt: str) -> str:
        if "Full segment transcript" in prompt:
            blocker.wait(5)                     # hold the write mid-weave
        return _pipeline_response(prompt)

    live.llm.chat = lambda messages, **kw: slow_response(messages[-1]["content"])
    live.add("a slow write", user_id="writer", session_id="s1")
    live.add("another", user_id="writer", session_id="s2")   # W1-free, but queued behind

    started = time.monotonic()
    out = live.search("anything?", user_id="reader", session_id="")
    elapsed = time.monotonic() - started
    blocker.set()
    live.flush(user_id="writer", session_id="s1", timeout_s=30)
    live.flush(user_id="writer", session_id="s2", timeout_s=30)

    assert out is not None and elapsed < 5, \
        f"recall was blocked by an unrelated write for {elapsed:.1f}s"


# ── backpressure and failure handling ───────────────────────────────────

def test_backpressure_raises_queue_busy(live, tmp_path, monkeypatch):
    set_config(Config(data_dir=tmp_path, log_dir=tmp_path / "logs",
                      max_queue_depth=2))
    monkeypatch.setattr(Memory, "_ensure_dispatcher", lambda self: None)

    live.add("one", user_id="u1", session_id="s1")
    live.add("two", user_id="u1", session_id="s1")
    with pytest.raises(QueueBusy, match="slow down"):
        live.add("three", user_id="u1", session_id="s1")
    # A different session is not throttled by this one's backlog.
    live.add("fine", user_id="u1", session_id="s2")


def test_a_poison_message_is_skipped_and_recorded(live, tmp_path):
    """A batch whose write keeps failing is retried, then skipped as poison —
    the queue keeps moving and the loss is queryable in the tasks table."""
    set_config(Config(data_dir=tmp_path, log_dir=tmp_path / "logs",
                      max_ingest_retries=1, dispatcher_tick_s=0.02,
                      dispatcher_idle_tick_s=0.05))

    def explode(messages, **kw):
        raise RuntimeError("the model gateway is down")
    live.llm.chat = explode

    live.add("first is doomed", user_id="u1", session_id="s1")
    live.end_session(user_id="u1", session_id="s1")
    # Poison handling happens across dispatcher retries; give it room, then the
    # queue must be empty — nothing blocks forever.
    deadline = time.monotonic() + 30
    while not live.msg_queue().is_empty("u1", "s1") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert live.msg_queue().is_empty("u1", "s1"), "poison message blocked the queue"

    rows = live.db.fetch_all(
        "SELECT kind FROM tasks WHERE user_id=%s ORDER BY created_at DESC LIMIT 20", ("u1",))
    kinds = {r["kind"] for r in rows}
    assert any(k.startswith("poisoned_") for k in kinds), f"no poison record in {kinds}"


# ── entry-level warnings ride on the receipt ────────────────────────────

def test_image_without_vision_model_warns_on_the_receipt(live):
    r = live.add([{"role": "user", "content": "look", "image": b"\x89PNG fake"}],
                 user_id="u1", session_id="s1")
    assert any("PERSONOS_MLLM_API_KEY" in w for w in r.warnings)
    live.flush(user_id="u1", session_id="s1", timeout_s=30)
    # The image still landed as evidence — degraded, not failed.
    assert live.for_user("u1").evidence.list(limit=10)
