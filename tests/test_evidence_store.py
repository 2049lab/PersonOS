from datetime import datetime, timezone

import pytest

from personos.models import EvidenceRecord
from personos.storage.evidence_store import EvidenceStore, sha256_of


def test_append_and_get(db):
    store = EvidenceStore(db)
    rec = EvidenceRecord(holder="user", content_inline="上周三打篮球扭了下右腿")
    eid = store.append(rec)
    got = store.get(eid)
    assert got is not None
    assert got.content_inline == "上周三打篮球扭了下右腿"
    assert got.sha256 == sha256_of("上周三打篮球扭了下右腿")  # the hash is filled in automatically


def test_content_dedup_is_explicit_opt_in(db):
    store = EvidenceStore(db)
    a = store.append(EvidenceRecord(content_inline="同样的话"), dedup_by_sha=True)
    b = store.append(EvidenceRecord(content_inline="同样的话"), dedup_by_sha=True)
    assert a == b  # identical content is deduplicated and returns the same id
    assert len(store.list()) == 1


def test_list_order_newest_first(db):
    store = EvidenceStore(db)
    store.append(EvidenceRecord(content_inline="第一句"))
    store.append(EvidenceRecord(content_inline="第二句"))
    items = store.list()
    assert len(items) == 2
    assert items[0].content_inline == "第二句"  # captured_at DESC


def test_search_keyword_and_semantics_and_order(db):
    """Keywords are ANDed within a single sentence and results come back in time order. This is
    the only retrieval path that does not go through the index — the fallback for when atom
    extraction missed something."""
    store = EvidenceStore(db)
    store.append(EvidenceRecord(content_inline="I broke my favourite bowl"))
    store.append(EvidenceRecord(content_inline="the bowl is blue"))
    store.append(EvidenceRecord(content_inline="broke a pen"))
    # Both words must appear in the same sentence to count as a hit.
    assert [r.content_inline for r in store.search_keyword(["bowl", "broke"])] == \
        ["I broke my favourite bowl"]
    assert len(store.search_keyword(["bowl"])) == 2        # one word matches two sentences
    assert store.search_keyword([]) == []                  # empty keywords issue no SQL


def test_search_keyword_holder_filter_and_like_escape(db):
    store = EvidenceStore(db)
    store.append(EvidenceRecord(holder="Melanie", content_inline="100% bowl"))
    store.append(EvidenceRecord(holder="user", content_inline="100% bowl too"))
    assert [r.holder for r in store.search_keyword(["bowl"], holder="Melanie")] == ["Melanie"]
    # % is not treated as a wildcard; it matches literally.
    assert len(store.search_keyword(["100%"])) == 2


def test_same_words_keep_message_session_speaker_and_time(db):
    store = EvidenceStore(db, "alice")
    first = EvidenceRecord(holder="user", content_inline="Yes", source={"session_id": "one"},
                           captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    second = EvidenceRecord(holder="assistant", content_inline="Yes", source={"session_id": "two"},
                            captured_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert store.append(first) != store.append(second)
    assert len(store.list()) == 2
    assert store.get(first.id).model_dump() == first.model_dump()
    assert store.get(second.id).model_dump() == second.model_dump()


def test_replay_same_id_preserves_immutable_record(db):
    store = EvidenceStore(db, "alice")
    original = EvidenceRecord(content_inline="I live in Oslo", source={"session_id": "one"})
    assert store.append(original) == original.id
    retry = original.model_copy(update={"content_inline": "Changed payload"})
    assert store.append(retry) == original.id
    assert len(store.list()) == 1
    assert store.get(original.id).content_inline == "I live in Oslo"


def test_same_id_cannot_claim_other_users_evidence(db):
    alice, bob = EvidenceStore(db, "alice"), EvidenceStore(db, "bob")
    original = EvidenceRecord(content_inline="Alice's original")
    alice.append(original)
    with pytest.raises(ValueError, match="unavailable"):
        bob.append(original.model_copy(update={"content_inline": "Bob's replacement"}))
    assert bob.list() == [] and bob.get(original.id) is None
    assert alice.get(original.id).content_inline == "Alice's original"
