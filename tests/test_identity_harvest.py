"""V2 step 2, AssetHarvest: pick_face selects a box by horizontal thirds, matching the
reference implementation and including the consistency check, plus how the quality value is
chosen."""

from __future__ import annotations

import numpy as np

from personos.identity.backends.base import FaceDet
from personos.identity.harvest import _face_q, pick_face

# frame width 90, so the centres of the thirds are left=15, center=45, right=75


def _fd(x1: int, x2: int, *, quality: float = -1.0, blur: float = 0.5) -> FaceDet:
    return FaceDet(bbox=(x1, 0, x2, 10), crop_b64="", det_score=0.9,
                   blur_score=blur, embedding=np.zeros(4, dtype=np.float32), quality=quality)


def test_empty_returns_none():
    assert pick_face([], "left", 90) is None


def test_single_no_pos_taken():
    d = _fd(0, 10)
    assert pick_face([d], "", 90) is d          # no position given, a single face -> accept


def test_multi_no_pos_rejected():
    assert pick_face([_fd(0, 10), _fd(80, 90)], "", 90) is None   # no position given, several faces -> reject


def test_single_pos_consistency_pass_and_reject():
    left_face = _fd(0, 12)                       # centre about 6, so it belongs to left
    assert pick_face([left_face], "left", 90) is left_face        # position matches -> accept
    assert pick_face([left_face], "right", 90) is None            # right was named but the face is on the left -> reject, better nothing than a wrong asset


def test_multi_pos_picks_correct_side():
    left, right = _fd(0, 12), _fd(78, 90)
    assert pick_face([left, right], "left", 90) is left
    assert pick_face([left, right], "right", 90) is right


def test_collision_both_center_pos_left_rejected():
    """Both people stand in the middle and left is named: the face that gets picked actually
    belongs to center, not left, so it is rejected. This is exactly the collision case that
    was fixed."""
    c1, c2 = _fd(38, 48), _fd(44, 54)            # both faces sit near the centre
    assert pick_face([c1, c2], "left", 90) is None
    assert pick_face([c1, c2], "center", 90) in (c1, c2)          # only naming center accepts one


def test_face_quality_prefers_adaface_else_blur():
    assert _face_q(_fd(0, 10, quality=0.7, blur=0.3)) == 0.7
    assert _face_q(_fd(0, 10, quality=-1.0, blur=0.42)) == 0.42
