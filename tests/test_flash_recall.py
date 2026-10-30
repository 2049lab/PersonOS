"""Retrieval-only recall must never generate, rewrite, review, or mix users' sources."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from personos.models import EvidenceRecord, EvidenceRef, MemCell, MemoryAtom
from personos.online.recall_flow import run_recall, validate_recall
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.evidence_store import EvidenceStore

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


class NoChat:
    available = False

    def chat(self, *args, **kwargs):
        pytest.fail("flash invoked a chat model")


class Embedder:
    available = True

    def embed(self, texts):
        return np.array([[1., 0.] for _ in texts], dtype=np.float32)


def seed(db, user, suffix):
    evidence = EvidenceStore(db, user)
    ref = EvidenceRef(evidence_id=evidence.append(EvidenceRecord(
        id=f"ev_{suffix}", content_inline="I moved to Lisbon in July.",
        captured_at=NOW, source={"session_id": "s"})))
    cell = MemCell(id=f"cell_{suffix}", session_id="s", topic="Home",
                   episode="The user moved to Lisbon in July.", evidence_refs=[ref])
    CellStore(db, user).upsert(cell)
    AtomStore(db, user).upsert(MemoryAtom(
        id=f"atom_{suffix}", memcell_id=cell.id, text="The user lives in Lisbon.",
        evidence_refs=[ref]), np.array([1., 0.], dtype=np.float32))
    return cell


def recall(db, user="alice", **kwargs):
    return run_recall(NoChat(), Embedder(), AtomStore(db, user), CellStore(db, user),
                      EvidenceStore(db, user), query="Where do I live?", session_id="s",
                      now_dt=NOW, mode="flash", **kwargs)


def test_flash_returns_sources_without_answer_or_review(db, monkeypatch):
    import personos.online.recall_flow as flow

    monkeypatch.setattr(flow, "build_history", lambda *a, **kw: pytest.fail("history read"))
    seed(db, "alice", "a")
    seed(db, "bob", "b")
    result = recall(db)
    public = result.to_public()
    assert result.ans is None and result.reviews == [] and result.deep is None
    assert public["answer"] == "" and public["verdict"] is None
    assert public["cited_cells"] == []
    assert [x["cell_id"] for x in public["materials"]] == ["cell_a"]
    assert public["materials"][0]["evidence_refs"] == ["ev_a"]
    assert public["memories"][0]["evidence"][0]["content"] == "I moved to Lisbon in July."
    assert "cell_b" not in str(public) and "ev_b" not in str(public)


def test_flash_empty_store_has_no_fabricated_answer(db):
    result = recall(db).to_public()
    assert result["materials"] == result["memories"] == []
    assert result["answer"] == "" and result["verdict"] is None


def test_flash_honors_reranker_and_output_limit(db):
    seed(db, "alice", "a")
    seed(db, "alice", "c")

    class Reverse:
        def rerank(self, query, documents, **kwargs):
            return list(range(len(documents)))

    result = recall(db, reranker=Reverse())
    assert result.ranked[0].cell.id == "cell_c"
    assert len(result.to_public(max_memories=1)["materials"]) == 1


@pytest.mark.parametrize("mode,query,top_k,image", [
    ("unknown", "q", 10, False), ("flash", " ", 10, False),
    ("flash", "q", 0, False), ("flash", "q", 101, False),
    ("flash", "q", True, False), ("flash", "q", 10, True),
])
def test_invalid_request_fails_before_work(mode, query, top_k, image):
    with pytest.raises(ValueError):
        validate_recall(query, mode, top_k, has_image=image)


def test_facade_flash_needs_only_embedding_and_skips_profile(db, monkeypatch):
    from personos.memory import Memory

    seed(db, "alice", "a")
    memory = Memory.__new__(Memory)
    memory.llm, memory.embedder, memory.reranker = NoChat(), Embedder(), None
    memory.mllm = None
    monkeypatch.setattr(memory, "for_user", lambda user: SimpleNamespace(
        atoms=AtomStore(db, user), cells=CellStore(db, user), evidence=EvidenceStore(db, user)))
    monkeypatch.setattr(memory, "_media", lambda: None)
    monkeypatch.setattr(memory, "_profile_strings", lambda user: pytest.fail("profile read"))
    result = memory.search("Where do I live?", user_id="alice", mode="flash")
    assert result.to_public()["materials"][0]["cell_id"] == "cell_a"


def test_http_flash_uses_shared_renderer_and_validation(db, monkeypatch):
    from fastapi.testclient import TestClient

    from personos.memory import UserContext
    from server import api
    from server.app import app
    from server.signing import verify_signature

    seed(db, "alice", "a")
    monkeypatch.setattr(api.rt, "llm", NoChat())
    monkeypatch.setattr(api.rt, "embedder", Embedder())
    monkeypatch.setattr(api.rt, "_media", lambda: None)
    monkeypatch.setattr(api, "ProfileStore", lambda *a: pytest.fail("profile read"))
    app.dependency_overrides[api._ctx] = lambda: UserContext("alice", db)
    app.dependency_overrides[verify_signature] = lambda: None
    client = TestClient(app)
    try:
        response = client.post("/api/v1/recall", json={
            "session_id": "s", "query": "Where do I live?", "mode": "flash"})
        assert response.status_code == 200
        assert response.json()["data"]["materials"][0]["cell_id"] == "cell_a"
        invalid = client.post("/api/v1/recall", json={
            "session_id": "s", "query": "q", "mode": "flash", "top_k": 0})
        assert invalid.status_code == 400 and invalid.json()["code"] == 400
        boolean_count = client.post("/api/v1/recall", json={
            "session_id": "s", "query": "q", "mode": "flash", "top_k": True})
        assert boolean_count.status_code == 422
    finally:
        client.close()
        app.dependency_overrides.clear()
