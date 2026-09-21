"""模型层单测:K/D 轴受控词表 + kind validator 的兜底行为(向前兼容关键)。"""

from personos.models import (
    DOMAIN_LABELS, DOMAIN_VOCAB, KIND_LABELS, KindLiteral, MemoryAtom, vocab_menu,
)


def test_vocab_shape():
    # 14 类记忆类型 / 16 个生活域;条目形如 "名称: 示例",标签可提取(英文说明,码位稳定)
    assert len(KindLiteral) == 14 and list(KindLiteral)[:2] == ["K01", "K02"]
    assert len(DOMAIN_VOCAB) == 16 and list(DOMAIN_VOCAB)[-1] == "D16"
    assert KIND_LABELS["K01"] == "fact/attribute"
    assert DOMAIN_LABELS["D05"] == "body, health & functioning"


def test_vocab_menu_format():
    m = vocab_menu(KindLiteral)
    assert m.startswith("K01=fact/attribute(birth year, languages spoken, device model)")
    assert "\n  K14=" in m and "K14=assessment/feedback(evaluation of a piece of advice, a tool, an experience)" in m
    assert set(DOMAIN_VOCAB) == set(DOMAIN_LABELS)          # 标签与词表同键


def test_domains_keep_only_valid_codes_and_dedupe():
    a = MemoryAtom(text="x", domains=["D05", "健康", "D99", "D05", {"code": "D01"}])
    assert a.domains == ["D05"]
    assert MemoryAtom(text="x", domains="D05").domains == []


def test_valid_kind_kept():
    for k in KindLiteral:
        assert MemoryAtom(text="x", kind=k).kind == k


def test_unknown_kind_becomes_none():
    # 词表外的值(LLM 瞎填 / 旧库脏数据,含旧版英文枚举)归 None,而非抛异常
    assert MemoryAtom(text="x", kind="不存在的类型").kind is None
    assert MemoryAtom(text="x", kind="").kind is None
    assert MemoryAtom(text="x", kind="fact").kind is None   # 旧词面不再合法(不做迁移)


def test_none_kind_ok():
    assert MemoryAtom(text="x").kind is None
