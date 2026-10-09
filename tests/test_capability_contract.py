"""What happens when a feature is asked for and its configuration is absent.

The rule, one line: **cannot do it at all -> raise; did it partially -> return
and say so.**

The degradation that suits a server — never fail on a missing optional feature —
is the wrong default for a library. Someone who passes a video with no vision
model configured should get a sentence, not silence and not a ModuleNotFoundError
from inside a worker thread. Every row of the capability table gets a test here,
because this is the part of the library people meet when something is wrong.
"""

from __future__ import annotations

import pytest

from personos import MissingCapability
from personos.config import Config, reset_config, set_config
from personos.diagnostics import inspect, render
from personos.errors import (
    image_not_understood,
    no_deep_track,
    no_embedder,
    no_identity_backend,
    no_llm,
    no_public_media_url,
    no_vision,
    visual_recall_unavailable,
)


@pytest.fixture
def cfg_scope():
    """Swap in a configuration for one test, then restore."""
    def _use(**kw):
        set_config(Config(**kw))
    yield _use
    reset_config()


# ── every error carries a remedy you can actually run ───────────────────

@pytest.mark.parametrize("factory", [
    no_llm, no_embedder, no_vision, no_identity_backend, no_deep_track,
    no_public_media_url,
])
def test_errors_say_what_to_do(factory):
    err = factory()
    assert isinstance(err, MissingCapability)
    assert err.remedy, "an error without a remedy is just a complaint"
    text = str(err)
    assert err.capability in text and err.remedy in text
    # The remedy has to be actionable: a variable to set or a command to run.
    assert any(tok in err.remedy for tok in ("PERSONOS_", "pip install", "ANTHROPIC_")), \
        err.remedy


def test_warnings_also_explain_the_fix():
    for msg in (image_not_understood(), visual_recall_unavailable()):
        assert "PERSONOS_MLLM_API_KEY" in msg


# ── partial success is reported, never silent ───────────────────────────

def test_results_carry_a_warnings_field():
    """Additive on both result types: callers that ignore it are unaffected,
    callers that check it can tell degraded from complete."""
    from personos.online.recall_flow import RecallOutcome
    from personos.online.write_path import BatchStepResult

    assert RecallOutcome(query="q", mode="fast").warnings == []
    assert BatchStepResult(evidence_ids=[], records=[]).warnings == []


def test_to_public_omits_warnings_when_there_are_none():
    """An empty key on every response would train people to ignore it."""
    from personos.online.recall_flow import RecallOutcome

    clean = RecallOutcome(query="q", mode="fast").to_public()
    assert "warnings" not in clean

    noisy = RecallOutcome(query="q", mode="fast")
    noisy.warnings.append("something was skipped")
    assert noisy.to_public()["warnings"] == ["something was skipped"]


def test_to_public_hides_internals():
    """Scores, prompts and raw model output are implementation detail.
    Publishing them would turn them into contract."""
    from personos.online.recall_flow import RecallOutcome

    out = RecallOutcome(query="q", mode="fast").to_public()
    for leaked in ("hits", "ranked", "rw", "secs", "draft", "reviews"):
        assert leaked not in out, f"{leaked} is internal"
    assert {"query", "mode", "answer", "cited_cells", "memories"} <= set(out)


# ── the diagnostic tells the same story ─────────────────────────────────

def test_doctor_reports_required_capabilities_as_blocking(cfg_scope):
    cfg_scope()                                    # nothing configured
    caps = {c.name: c for c in inspect()}
    assert caps["chat model"].required and not caps["chat model"].available
    assert caps["embeddings"].required and not caps["embeddings"].available
    # Optional ones are reported as off, not as failures.
    assert not caps["image understanding"].required
    assert not caps["reranking"].required


def test_doctor_shows_the_zero_config_defaults_as_working(cfg_scope, tmp_path):
    cfg_scope(llm_api_key="sk-x", data_dir=tmp_path)
    caps = {c.name: c for c in inspect()}
    assert caps["chat model"].available and caps["embeddings"].available
    assert caps["storage"].available and "SQLite" in caps["storage"].detail
    assert caps["media storage"].available and "local" in caps["media storage"].detail


def test_doctor_explains_the_local_media_video_gap(cfg_scope, tmp_path):
    """Video plus local storage is the one combination that looks configured
    but cannot work: the model fetches the clip by URL, server-side."""
    pytest.importorskip("av")
    cfg_scope(llm_api_key="sk-x", mllm_api_key="sk-m", mllm_model="m",
              video_backend="real", data_dir=tmp_path)
    names = [c.name.strip() for c in inspect()]
    assert "video with local media" in names


def test_doctor_output_is_readable(cfg_scope):
    cfg_scope()
    text = render(inspect())
    assert "Not usable yet" in text
    assert "PERSONOS_LLM_API_KEY" in text


# ── required capabilities are checked at the entry point, not at build ──

def test_constructing_memory_does_not_require_configuration(cfg_scope, tmp_path):
    """Memory() is the first line of every quickstart. Failing there would make
    the library look broken before it has been asked to do anything."""
    cfg_scope(data_dir=tmp_path)
    from personos import Memory

    m = Memory()
    try:
        assert m.capabilities()
    finally:
        m.close()


def test_calls_that_need_a_model_raise_with_a_remedy(cfg_scope, tmp_path):
    cfg_scope(data_dir=tmp_path)
    from personos import Memory

    m = Memory()
    try:
        with pytest.raises(MissingCapability, match="PERSONOS_LLM_API_KEY"):
            m.add("hello", user_id="u", session_id="s")
        with pytest.raises(MissingCapability, match="PERSONOS_LLM_API_KEY"):
            m.search("hello?", user_id="u")
    finally:
        m.close()


# ── .env.example is the source of truth for variable names ──────────────

def test_documented_and_read_environment_variables_match():
    """Every variable the code reads must be in .env.example, and vice versa.

    Without this the two drift silently and in the worst direction: a rename
    leaves the old name in .env.example, someone sets it, nothing happens, and
    the library appears to ignore its own documentation. That is exactly what
    happened once here — a stale .env still carried the pre-rename names, so
    the model settings in it were dead, while an unrelated MYSQL_HOST left in
    the same file quietly switched the storage backend away from the
    documented SQLite default.

    Pairing the two directions is the point. Read-but-undocumented hides a
    knob; documented-but-unread advertises one that does nothing.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / "personos" / "config.py").read_text(encoding="utf-8")
    example = (root / ".env.example").read_text(encoding="utf-8")

    read = set(re.findall(r'_(?:env|flag)\("([A-Z_]+)"', source))
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, re.M))

    assert not (read - documented), (
        f"read by config.py but missing from .env.example: {sorted(read - documented)}")
    assert not (documented - read), (
        f"in .env.example but never read: {sorted(documented - read)}")


def test_configuration_variables_are_namespaced():
    """Names must be ours or a vendor's, never generic.

    A bare name like MYSQL_HOST is set on plenty of machines for unrelated
    reasons. Reading one means the library's behaviour depends on a variable
    the user never set for us — and the symptom is data appearing somewhere
    other than where the documentation says.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "personos" / "config.py").read_text()
    read = set(re.findall(r'_(?:env|flag)\("([A-Z_]+)"', source))
    # Vendor-standard names users already have set are fine, and expected.
    vendor = {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"}
    generic = {v for v in read
               if not v.startswith(("PERSONOS_", "LANGFUSE_")) and v not in vendor}
    assert not generic, f"un-namespaced configuration variables: {sorted(generic)}"


def test_local_video_without_public_url_fails_at_the_entry(cfg_scope, tmp_path, monkeypatch):
    """A local clip + local storage + no public prefix can never reach the
    remote model. Knowable synchronously, so it must raise at add() — not
    poison the queue after five retries in a worker thread."""
    cfg_scope(llm_api_key="sk-x", data_dir=tmp_path,
              media_backend="local", media_base_url="")
    from personos import Memory

    m = Memory.__new__(Memory)          # no pools/providers, like the facade fixture
    m.mllm = type("M", (), {"available": True})()
    monkeypatch.setattr(Memory, "video_deps", lambda self, uid: object())
    from personos.storage.media.local import LocalMediaStore
    monkeypatch.setattr(Memory, "_media", lambda self: LocalMediaStore(root=tmp_path))

    with pytest.raises(MissingCapability, match="PERSONOS_MEDIA_BASE_URL"):
        m._enqueue_videos([tmp_path / "clip.mp4"], user_id="u", session_id="s",
                          scenario="", sync=False, timeout_s=1)


def test_local_video_gate_ignores_remote_urls(cfg_scope, tmp_path, monkeypatch):
    """A clip already at a URL is fetched by the model service itself; local
    media storage is irrelevant, so the gate must not fire."""
    cfg_scope(llm_api_key="sk-x", data_dir=tmp_path,
              media_backend="local", media_base_url="")
    from personos import Memory

    m = Memory.__new__(Memory)
    m.mllm = type("M", (), {"available": True})()
    monkeypatch.setattr(Memory, "video_deps", lambda self, uid: object())
    monkeypatch.setattr(Memory, "_media", lambda self: None)
    sent = {}
    monkeypatch.setattr(Memory, "_enqueue",
                        lambda self, u, s, payload, kind, warnings=None: sent.update(payload) or
                        type("R", (), {"accepted": True})())

    m._enqueue_videos(["https://example.com/clip.mp4"], user_id="u", session_id="s",
                      scenario="", sync=False, timeout_s=1)
    assert sent["messages"][0]["video_url"] == "https://example.com/clip.mp4"


def test_insufficient_verdict_cites_nothing():
    """A refusal must not present anything as its basis: memories were always
    gated on insufficient_material; cited_cells now follows the same rule, so
    the honesty contract no longer depends on what the draft happened to cite."""
    from personos.online.arbitrate import ReviewResult
    from personos.online.recall_flow import RecallOutcome
    from personos.online.retrieval import MemoryAnswer

    out = RecallOutcome(query="q", mode="auto")
    out.ans = MemoryAnswer(answer="The materials say nothing about a cat.",
                           cited_cells=["cell_a", "cell_b"])
    out.reviews.append(ReviewResult(verdict="insufficient_material", critique=""))
    pub = out.to_public()
    assert pub["cited_cells"] == [] and pub["memories"] == []
    assert pub["verdict"] == "insufficient_material"

    ok = RecallOutcome(query="q", mode="fast")
    ok.ans = MemoryAnswer(answer="Jurong West.", cited_cells=["cell_a"])
    ok.reviews.append(ReviewResult(verdict="ok", critique=""))
    assert ok.to_public()["cited_cells"] == ["cell_a"]
