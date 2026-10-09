"""Public service API contract: the memory view must be clean, traceable, and structurally
consistent. Only the pure renderer is tested here — no runtime import, no model provider calls.

The endpoints themselves (ingest/recall/trace) need a live model provider and the runtime
singleton, so they are covered by manual smoke tests rather than deterministic unit tests
(per the agreed convention).
"""

from datetime import datetime, timezone

from personos.models import EvidenceRecord, EvidenceRef, MemoryAtom
from personos.online.views import memory_view
from personos.storage.evidence_store import EvidenceStore


def test_memory_view_is_clean_and_traceable():
    a = MemoryAtom(
        object_type="fact", text="我对花生过敏", holder="user",
        memcell_id="cell_1", domains=["D05"], kind="K09",
        occurrence_time=datetime(2026, 7, 13, tzinfo=timezone.utc),
        evidence_refs=[EvidenceRef(evidence_id="ev_1"), EvidenceRef(evidence_id="ev_2")],
    )
    v = memory_view(a)   # no store passed

    # Traceable: atom_id and evidence_refs are always present. With no store, evidence is an
    # empty list but the structural slot still exists.
    assert v["atom_id"] == a.id and v["evidence_refs"] == ["ev_1", "ev_2"] and v["evidence"] == []
    # The semantic / ownership / time fields are complete, and the owning cell pointer is
    # exposed so the caller can look the episode narrative up again.
    assert v["type"] == "fact" and v["holder"] == "user" and v["cell_id"] == "cell_1"
    assert v["domains"] == ["D05"] and v["occurred_at"].startswith("2026-07-13")
    assert v["kind"] == "K09"
    # Leaks NOTHING internal: scoring, RRF score, rerank score and prompts stay out of the
    # public view.
    for leak in ("breakdown", "score", "similarity", "tier", "salience_score",
                 "provenance", "system", "raw"):
        assert leak not in v, f"public view must not contain internal field {leak}"


def test_memory_view_inlines_evidence_with_qa(db):
    # With an evidence_store supplied, every memory inlines its evidence (the original text plus
    # the Q and A from the same turn), so memories returned by recall carry their own provenance.
    ev = EvidenceStore(db)
    qid = ev.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    ev.append(EvidenceRecord(holder="assistant", content_inline="记住了，帮你避开花生",
                             source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏",
                   evidence_refs=[EvidenceRef(evidence_id=qid)])

    v = memory_view(a, ev)
    assert len(v["evidence"]) == 1
    e0 = v["evidence"][0]
    assert e0["content"] == "我对花生过敏" and e0["reply"]["content"] == "记住了，帮你避开花生"


def test_memory_view_falls_back_to_recorded_at():
    a = MemoryAtom(object_type="claim", text="随口一说")
    v = memory_view(a)
    assert v["occurred_at"] is not None and v["type"] == "claim"
    assert v["cell_id"] is None                            # not attached to a cell -> None, no crash
    assert v["evidence_refs"] == [] and v["evidence"] == []      # no evidence, still no crash
