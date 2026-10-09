"""Multi-tenant isolation: two users share one database, and the user_id binding at the store layer
must guarantee that none of A's data is visible to B.

This test doubles as the isolation guard for the MySQL migration -- it covers every store read path.
"""

from personos.models import EvidenceRecord, MemoryAtom
from personos.online.session_context import build_history
from personos.storage.atom_store import AtomStore
from personos.storage.evidence_store import EvidenceStore
from personos.storage.session_store import SessionContextStore
from personos.storage.user_store import UserStore
from tests.fakes import FakeEmbedder, FakeLLM


def test_cross_user_isolation(db):
    """None of the evidence, atoms or summaries written by A can be read through B's bound stores, and
    identical content with the same sha is not deduplicated across users."""
    ev_a, at_a = EvidenceStore(db, user_id="A"), AtomStore(db, user_id="A")
    ev_b, at_b = EvidenceStore(db, user_id="B"), AtomStore(db, user_id="B")

    emb = FakeEmbedder()
    eid = ev_a.append(EvidenceRecord(holder="user", content_inline="A用户的秘密",
                                     source={"session_id": "s1"}), embedding=emb.embed(["x"])[0])
    a_atom = MemoryAtom(object_type="fact", text="A用户的事实", lifecycle_status="active")
    at_a.upsert(a_atom, embedding=emb.embed(["x"])[0])
    SessionContextStore(db, user_id="A").save("s1", "A的摘要", 2)

    # From B's point of view everything is empty or not found
    assert ev_b.list() == [] and ev_b.all_with_embeddings() == []
    assert ev_b.get(eid) is None
    assert ev_b.by_session("s1") == [] and ev_b.in_session("s1") == []
    assert ev_b.session_stats() == {}
    assert ev_b.reply_for(eid) is None
    assert at_b.list() == [] and at_b.all_with_embeddings() == []
    assert at_b.get(a_atom.id) is None and at_b.get_embedding(a_atom.id) is None
    assert SessionContextStore(db, user_id="B").get("s1") == ("", 0)

    # Identical content is not deduplicated across users: when B writes evidence with the same sha it
    # must exist as a new piece of evidence belonging to B
    eid_b = ev_b.append(EvidenceRecord(holder="user", content_inline="A用户的秘密",
                                       source={"session_id": "s1"}))
    assert eid_b != eid and ev_b.get(eid_b) is not None

    # A still sees only its own rows; B's identical evidence never enters A's view
    assert len(ev_a.list()) == 1 and ev_a.get(eid_b) is None


def test_user_store_register_and_token(db):
    us = UserStore(db)
    base = us.count()  # the shared integration database holds real registrations, so assert on the delta rather than an absolute count
    r = us.register("linwan")
    assert r["user_id"] == "linwan" and r["token"]
    assert us.user_id_by_token(r["token"]) == "linwan"
    assert us.user_id_by_token("不存在") is None
    try:
        us.register("linwan")
        assert False, "a duplicate user_id should raise"
    except ValueError:
        pass
    auto = us.register()
    assert auto["user_id"].startswith("u_") and us.count() == base + 2


def test_build_history_per_user(db):
    """Session history is built per user: B cannot read A's session history, and summaries are isolated
    by the (user, session) pair."""
    ev_a = EvidenceStore(db, user_id="A")
    ev_a.append(EvidenceRecord(holder="user", content_inline="A说了一句话", source={"session_id": "s1"}))
    llm = FakeLLM([])   # compaction is not triggered
    h_a = build_history(ev_a, "s1", llm=llm)
    assert any("A说了" in t for _, t in h_a)
    h_b = build_history(EvidenceStore(db, user_id="B"), "s1", llm=llm)
    assert h_b == []
