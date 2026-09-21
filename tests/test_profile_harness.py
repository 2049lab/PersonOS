"""补丁校验器单测(干净结构):合法通过 + 非法字段打回(未知域/枚举/长度/band/幻觉短标出处)。"""

from __future__ import annotations

from personos.online.profile_harness import validate_patch

VALID = {"c1", "c2"}          # 本轮 cell 的短标集


def _v(patch):
    return validate_patch(patch, valid_cell_ids=VALID)


def test_valid_patch_passes():
    patch = {
        "traits": {"occupation": {"text": "engineer", "status": "confirmed", "sources": ["c1"]},
                   "finance": None},
        "facts": {
            "add": [{"band": "today", "text": "ate ramen", "sources": ["c1"]}],
            "rewrite": [{"id": "f_1", "text": "merged", "sources": ["c1", "c2"]}],
            "drop": ["f_2"],
        },
    }
    assert _v(patch) == []


def test_unknown_trait_domain_rejected():
    errs = _v({"traits": {"favorite_color": {"text": "blue", "status": "confirmed", "sources": ["c1"]}}})
    assert any("未知域" in e for e in errs)


def test_trait_text_too_long_rejected():
    errs = _v({"traits": {"personality": {"text": "x" * 501, "status": "inferred", "sources": ["c1"]}}})
    assert any("超上限" in e for e in errs)          # 上限 500(c5880c1 由 250 提到 500)


def test_trait_text_within_limit_ok():
    """限界条件描述天然字多:120 字的一句话应通过(上限 500)。"""
    assert _v({"traits": {"personality": {"text": "x" * 120, "status": "inferred", "sources": ["c1"]}}}) == []


def test_bad_status_rejected():
    errs = _v({"traits": {"goals": {"text": "ok", "status": "maybe", "sources": ["c1"]}}})
    assert any("status" in e for e in errs)


def test_missing_sources_rejected():
    errs = _v({"traits": {"goals": {"text": "ok", "status": "confirmed"}}})
    assert any("sources" in e for e in errs)


def test_hallucinated_source_rejected():
    errs = _v({"facts": {"add": [{"band": "today", "text": "x", "sources": ["c_ghost"]}]}})
    assert any("幻觉出处" in e for e in errs)


def test_add_missing_band_rejected():
    errs = _v({"facts": {"add": [{"text": "x", "sources": ["c1"]}]}})
    assert any("band" in e for e in errs)


def test_add_missing_text_rejected():
    errs = _v({"facts": {"add": [{"band": "week", "sources": ["c1"]}]}})
    assert any("text" in e for e in errs)


def test_bad_band_rejected():
    errs = _v({"facts": {"add": [{"band": "yesterday", "text": "x", "sources": ["c1"]}]}})
    assert any("band" in e for e in errs)


def test_rewrite_missing_id_rejected():
    errs = _v({"facts": {"rewrite": [{"text": "new", "sources": ["c1"]}]}})
    assert any("缺 id" in e for e in errs)


def test_rewrite_partial_fields_ok():
    """rewrite 只改一个字段合法(未提字段保留,不强制全给)。"""
    assert _v({"facts": {"rewrite": [{"id": "f_1", "band": "long"}]}}) == []


def test_non_dict_patch_rejected():
    assert validate_patch([], valid_cell_ids=VALID) == ["补丁根须为 JSON 对象"]
