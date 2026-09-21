import pytest
"""Trust-chain unit tests: from an adopted atom to its ownership / awareness state and then
down to the evidence — the chain must assemble correctly and degrade without crashing."""

from personos.models import EvidenceRecord, EvidenceRef, MemoryAtom
from personos.online.trust import build_trust_chain, trace_evidence
from personos.storage.atom_store import AtomStore
from personos.storage.evidence_store import EvidenceStore


def test_chain_drills_to_evidence(db):
    ev_store = EvidenceStore(db)
    at_store = AtomStore(db)
    eid = ev_store.append(EvidenceRecord(holder="user", content_inline="我上周打篮球扭了右腿"))
    a = MemoryAtom(object_type="event", text="用户上周打篮球扭了右腿", holder="user",
                   domains=["D05"], kind="K01", evidence_refs=[EvidenceRef(evidence_id=eid)])
    at_store.upsert(a)

    chain = build_trust_chain([a.id], at_store, ev_store)
    assert len(chain) == 1
    node = chain[0]
    assert node["text"] == "用户上周打篮球扭了右腿"
    assert node["object_type"] == "event" and node["kind"] == "K01"
    assert node["cell_id"] is None                            # an atom with no cell must not crash
    assert node["evidence_count"] == 1
    assert node["evidence"][0]["content"] == "我上周打篮球扭了右腿"


def test_chain_dedups_ids(db):
    at_store = AtomStore(db)
    a = MemoryAtom(text="无证据原子")
    at_store.upsert(a)
    # The same id passed twice is deduplicated; having no evidence must not crash either.
    chain = build_trust_chain([a.id, a.id], at_store, EvidenceStore(db))
    assert len(chain) == 1 and chain[0]["text"] == "无证据原子"
    assert chain[0]["evidence_count"] == 0


def test_evidence_pairs_assistant_reply(db):
    # Evidence in the trust chain is paired with the assistant reply from the same turn
    # (reply_to), so an external agent tracing provenance sees the complete Q and A rather than
    # just one side of it.
    ev_store = EvidenceStore(db)
    at_store = AtomStore(db)
    qid = ev_store.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    ev_store.append(EvidenceRecord(holder="assistant", content_inline="记住了，帮你避开花生",
                                   source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏", evidence_refs=[EvidenceRef(evidence_id=qid)])
    at_store.upsert(a)

    chain = build_trust_chain([a.id], at_store, ev_store)
    e0 = chain[0]["evidence"][0]
    assert e0["content"] == "我对花生过敏"
    assert e0["reply"]["content"] == "记住了，帮你避开花生"     # paired with the reply from that turn


def test_missing_atom_marked(db):
    chain = build_trust_chain(["atom_不存在"], AtomStore(db), EvidenceStore(db))
    assert chain[0]["missing"] is True


def test_trace_evidence_reverse(db):
    # Reverse tracing by evidence_id: reconstruct the whole turn (user then assistant) plus
    # cited_by. Querying either the user half or the assistant half fills in the full pair.
    ev_store, at_store = EvidenceStore(db), AtomStore(db)
    qid = ev_store.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    aid = ev_store.append(EvidenceRecord(holder="assistant", content_inline="记住了", source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏", evidence_refs=[EvidenceRef(evidence_id=qid)])
    at_store.upsert(a)

    # Query the USER half -> the full pair, ordered user then assistant.
    node = trace_evidence(qid, ev_store, at_store)
    assert node["node"] == "evidence" and node["evidence_id"] == qid
    assert [p["holder"] for p in node["pair"]] == ["user", "assistant"]        # order is fixed
    assert [p["content"] for p in node["pair"]] == ["我对花生过敏", "记住了"]
    assert node["cited_by"] == [a.id]                                          # the citation points at the user half

    # Query the ASSISTANT half -> the same pair is filled in, still user then assistant.
    node2 = trace_evidence(aid, ev_store, at_store)
    assert [p["holder"] for p in node2["pair"]] == ["user", "assistant"]
    assert node2["cited_by"] == [a.id]                                         # cited_by is based on the user half
    assert trace_evidence("ev_不存在", ev_store, at_store) is None
