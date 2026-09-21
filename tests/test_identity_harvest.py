"""V2② AssetHarvest:横向三分位挑框 pick_face(对齐 mneme,含一致性校验)+ 质量取值。"""

from __future__ import annotations

import numpy as np

from personos.identity.backends.base import FaceDet
from personos.identity.harvest import _face_q, pick_face

# 帧宽 90:三分位中心 left=15 / center=45 / right=75


def _fd(x1: int, x2: int, *, quality: float = -1.0, blur: float = 0.5) -> FaceDet:
    return FaceDet(bbox=(x1, 0, x2, 10), crop_b64="", det_score=0.9,
                   blur_score=blur, embedding=np.zeros(4, dtype=np.float32), quality=quality)


def test_empty_returns_none():
    assert pick_face([], "left", 90) is None


def test_single_no_pos_taken():
    d = _fd(0, 10)
    assert pick_face([d], "", 90) is d          # 无 pos 单脸 → 收


def test_multi_no_pos_rejected():
    assert pick_face([_fd(0, 10), _fd(80, 90)], "", 90) is None   # 无 pos 多脸 → 拒


def test_single_pos_consistency_pass_and_reject():
    left_face = _fd(0, 12)                       # 中心≈6 → 属 left
    assert pick_face([left_face], "left", 90) is left_face        # 位置相符 → 收
    assert pick_face([left_face], "right", 90) is None            # 提名 right 但脸在 left → 拒(宁缺毋滥)


def test_multi_pos_picks_correct_side():
    left, right = _fd(0, 12), _fd(78, 90)
    assert pick_face([left, right], "left", 90) is left
    assert pick_face([left, right], "right", 90) is right


def test_collision_both_center_pos_left_rejected():
    """两人都在中间、提名 left → 挑到的中脸实际属 center≠left → 拒(正是修掉的撞车场景)。"""
    c1, c2 = _fd(38, 48), _fd(44, 54)            # 两张脸都在 center 附近
    assert pick_face([c1, c2], "left", 90) is None
    assert pick_face([c1, c2], "center", 90) in (c1, c2)          # 提名 center 才收


def test_face_quality_prefers_adaface_else_blur():
    assert _face_q(_fd(0, 10, quality=0.7, blur=0.3)) == 0.7
    assert _face_q(_fd(0, 10, quality=-1.0, blur=0.42)) == 0.42
