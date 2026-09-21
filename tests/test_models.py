"""Model layer unit tests: the controlled K and D vocabularies, plus the fallback behaviour of the
kind validator, which is what keeps the schema forward compatible."""

from personos.models import (
    DOMAIN_LABELS, DOMAIN_VOCAB, KIND_LABELS, KindLiteral, MemoryAtom, vocab_menu,
)


def test_vocab_shape():
    # 14 memory kinds and 16 life domains; each entry reads "name: examples" and the label can be
    # extracted (English wording, stable code points)
    assert len(KindLiteral) == 14 and list(KindLiteral)[:2] == ["K01", "K02"]
    assert len(DOMAIN_VOCAB) == 16 and list(DOMAIN_VOCAB)[-1] == "D16"
    assert KIND_LABELS["K01"] == "fact/attribute"
    assert DOMAIN_LABELS["D05"] == "body, health & functioning"


def test_vocab_menu_format():
    m = vocab_menu(KindLiteral)
    assert m.startswith("K01=fact/attribute(birth year, languages spoken, device model)")
    assert "\n  K14=" in m and "K14=assessment/feedback(evaluation of a piece of advice, a tool, an experience)" in m
    assert set(DOMAIN_VOCAB) == set(DOMAIN_LABELS)          # labels and vocabulary share the same keys


def test_domains_keep_only_valid_codes_and_dedupe():
    a = MemoryAtom(text="x", domains=["D05", "健康", "D99", "D05", {"code": "D01"}])
    assert a.domains == ["D05"]
    assert MemoryAtom(text="x", domains="D05").domains == []


def test_valid_kind_kept():
    for k in KindLiteral:
        assert MemoryAtom(text="x", kind=k).kind == k


def test_unknown_kind_becomes_none():
    # Values outside the vocabulary (an LLM making something up, or dirty rows from an old database,
    # including the previous English enum) become None rather than raising
    assert MemoryAtom(text="x", kind="不存在的类型").kind is None
    assert MemoryAtom(text="x", kind="").kind is None
    assert MemoryAtom(text="x", kind="fact").kind is None   # the old wording is no longer valid and is not migrated


def test_none_kind_ok():
    assert MemoryAtom(text="x").kind is None
