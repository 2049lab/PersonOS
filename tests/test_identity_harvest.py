"""V2 step 2, AssetHarvest: pick_face selects a box by horizontal thirds, matching the
reference implementation and including the consistency check, plus how the quality value is
chosen."""

from __future__ import annotations

import numpy as np

from personos.identity import harvest
from personos.identity.backends.base import FaceDet
from personos.identity.harvest import _face_q, pick_face, reject_reason

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
    # Naming center no longer accepts one either: two faces in the same third cannot be told apart,
    # and filing the wrong one under the nominated person is worse than harvesting nothing.
    assert pick_face([c1, c2], "center", 90) is None


def test_face_quality_prefers_adaface_else_blur():
    assert _face_q(_fd(0, 10, quality=0.7, blur=0.3)) == 0.7
    assert _face_q(_fd(0, 10, quality=-1.0, blur=0.42)) == 0.42


# -- guards: reject_reason, ambiguity, fallback sampling, cross-cast duplicates ------------------

def _kps(eye_dist: float, nose_dx: float = 0.0, cx: float = 100.0) -> np.ndarray:
    """left eye, right eye, nose, mouth corners around centre cx."""
    return np.array([[cx - eye_dist / 2, 50], [cx + eye_dist / 2, 50], [cx + nose_dx, 70],
                     [cx - 5, 90], [cx + 5, 90]], dtype=np.float32)


def _real(x: float, w: int = 60, *, det: float = 0.9, kps=None, emb=None) -> FaceDet:
    return FaceDet(bbox=(int(x), 0, int(x + w), w), crop_b64="", det_score=det, blur_score=0.5,
                   embedding=np.array([1, 0, 0, 0], dtype=np.float32) if emb is None else emb,
                   quality=0.8, kps=_kps(0.45 * w, cx=x + w / 2) if kps is None else kps)


def test_reject_reason_accepts_a_frontal_face():
    assert reject_reason(_real(600), 1280) is None


def test_reject_reason_small_box_low_score_and_missing_landmarks():
    assert "too_small" in reject_reason(_real(600, w=30), 1280)
    assert "low_det_score" in reject_reason(_real(600, det=0.55), 1280)
    d = _real(600)
    d.kps = None
    assert reject_reason(d, 1280) is None            # no landmarks: only the size / score guards apply


def test_reject_reason_back_of_head_or_profile():
    """The real contaminating detections: eye distance collapsed to <0.2 of the box, nose off the eyes."""
    ear = _real(600, w=66, kps=_kps(0.17 * 66, nose_dx=0.45 * 11, cx=633))
    assert "not_frontal" in reject_reason(ear, 1280)
    off = _real(600, w=60, kps=_kps(0.4 * 60, nose_dx=0.9 * 0.4 * 60, cx=630))     # nose well past the eye span
    assert reject_reason(off, 1280) == "not_frontal(nose_outside_eyes)"


def test_pick_face_ambiguity_runner_up_in_same_third():
    a, b = _fd(38, 48), _fd(44, 54)                  # 90px frame, thirds at 15/45/75: both centres in the middle third
    assert pick_face([a, b], "center", 90) is None
    c = _fd(60, 70)                                  # centre 65 -> the right third, and far from the winner's distance
    assert pick_face([a, c], "center", 90) is a
    left, far = _fd(0, 12), _fd(78, 90)
    assert pick_face([left, far], "left", 90) is left


def test_pick_face_refuses_near_tie_across_thirds():
    a, b = _fd(30, 40), _fd(50, 60)                  # centres 35 and 55 -> equally near the centre third (45)
    assert pick_face([a, b], "center", 90) is None


class _VP:
    def embed(self, wav):
        return np.ones(4, dtype=np.float32)


def _run(monkeypatch, by_time, noms, *, frames_until=100.0):
    from personos.identity.screenplay import ClipScript, Nomination

    cur = {}

    def fake_frame(path, t):
        if t > frames_until:
            return None
        cur["t"] = t                                   # the detector below keys its answers on this instant
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    class Det:
        def __init__(self):
            self.seen = []

        def detect(self, frame):
            self.seen.append(cur["t"])
            return by_time.get(round(cur["t"], 1), [])

    monkeypatch.setattr(harvest, "_frame_at", fake_frame)
    monkeypatch.setattr(harvest, "_body_crop_b64", lambda frame, bbox: "")
    det = Det()
    script = ClipScript(nominations=[Nomination(local_id=n[0], t=n[1], pos=n[2]) for n in noms])
    ev = harvest.harvest_clip("x.mp4", script, {"face_detector": det, "voiceprint": _VP()})
    return ev, det


def test_fallback_sampling_finds_face_at_neighbouring_instant(monkeypatch):
    good = _real(300)                                  # centre ~330/1280 = 0.26 -> left third
    ev, det = _run(monkeypatch, {10.5: [good]}, [("P1", 10.0, "left")])
    assert [f.t for f in ev["P1"].faces] == [10.5]
    assert det.seen == [10.0, 10.5], "must stop at the first accepted pick"


def test_fallback_gives_up_after_all_offsets_and_logs_reason(monkeypatch):
    from loguru import logger

    ear = _real(600, w=66, kps=_kps(0.17 * 66, nose_dx=5, cx=633))
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        ev, det = _run(monkeypatch, {t: [ear] for t in (10.0, 10.5, 9.5, 11.0, 9.0)}, [("P1", 10.0, "center")])
    finally:
        logger.remove(sink)
    assert not ev.get("P1") or not ev["P1"].faces
    assert det.seen == [10.0, 10.5, 9.5, 11.0, 9.0]
    assert any("face rejected local_id=P1" in ln and "not_frontal" in ln for ln in lines)


def test_fallback_clamps_negative_time(monkeypatch):
    ev, det = _run(monkeypatch, {}, [("P1", 0.2, "left")])
    assert min(det.seen) >= 0.0


def test_cross_cast_same_face_dropped_for_both(monkeypatch):
    same = np.array([1, 0, 0, 0], dtype=np.float32)
    a, b = _real(300, emb=same), _real(900, emb=same)  # left vs right third, identical embedding
    ev, _ = _run(monkeypatch, {5.0: [a, b]}, [("P1", 5.0, "left"), ("P2", 5.0, "right")])
    assert not ev.get("P1", None) or not ev["P1"].faces
    assert not ev.get("P2", None) or not ev["P2"].faces


def test_cross_cast_distinct_faces_kept(monkeypatch):
    a = _real(300, emb=np.array([1, 0, 0, 0], dtype=np.float32))
    b = _real(900, emb=np.array([0, 1, 0, 0], dtype=np.float32))
    ev, _ = _run(monkeypatch, {5.0: [a, b]}, [("P1", 5.0, "left"), ("P2", 5.0, "right")])
    assert len(ev["P1"].faces) == 1 and len(ev["P2"].faces) == 1
