from personos.models import EvidenceRecord
from personos.storage.evidence_store import EvidenceStore, sha256_of


def test_append_and_get(db):
    store = EvidenceStore(db)
    rec = EvidenceRecord(holder="user", content_inline="上周三打篮球扭了下右腿")
    eid = store.append(rec)
    got = store.get(eid)
    assert got is not None
    assert got.content_inline == "上周三打篮球扭了下右腿"
    assert got.sha256 == sha256_of("上周三打篮球扭了下右腿")  # 自动补哈希


def test_dedup_by_sha256(db):
    store = EvidenceStore(db)
    a = store.append(EvidenceRecord(content_inline="同样的话"))
    b = store.append(EvidenceRecord(content_inline="同样的话"))
    assert a == b  # 内容相同 → 去重,返回同一 id
    assert len(store.list()) == 1


def test_list_order_newest_first(db):
    store = EvidenceStore(db)
    store.append(EvidenceRecord(content_inline="第一句"))
    store.append(EvidenceRecord(content_inline="第二句"))
    items = store.list()
    assert len(items) == 2
    assert items[0].content_inline == "第二句"  # captured_at DESC


def test_search_keyword_and_semantics_and_order(db):
    """关键词同句 AND、时序返回;这是唯一不经索引的检索(atoms 漏抽时的兜底路)。"""
    store = EvidenceStore(db)
    store.append(EvidenceRecord(content_inline="I broke my favourite bowl"))
    store.append(EvidenceRecord(content_inline="the bowl is blue"))
    store.append(EvidenceRecord(content_inline="broke a pen"))
    assert [r.content_inline for r in store.search_keyword(["bowl", "broke"])] == \
        ["I broke my favourite bowl"]                     # 两词同句才算命中
    assert len(store.search_keyword(["bowl"])) == 2        # 单词两句命中
    assert store.search_keyword([]) == []                  # 空关键词不发 SQL


def test_search_keyword_holder_filter_and_like_escape(db):
    store = EvidenceStore(db)
    store.append(EvidenceRecord(holder="Melanie", content_inline="100% bowl"))
    store.append(EvidenceRecord(holder="user", content_inline="100% bowl too"))
    assert [r.holder for r in store.search_keyword(["bowl"], holder="Melanie")] == ["Melanie"]
    assert len(store.search_keyword(["100%"])) == 2        # % 不当通配符,按字面命中
