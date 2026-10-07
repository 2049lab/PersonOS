"""AssetHarvest: pull assets out of a clip, guided by the screenplay's nominations
and voice ranges.

We detect on the **nominated frame** (plus a few instants right beside it when that one
yields nothing) and pick the box by **coarse position**; there is no full-video tracking:

- each nomination (t + a coarse horizontal position) -> grab that frame -> detect
  faces -> pick the box by position -> FacePick (emb + q + crop_b64);
- each voice range (t0..t1) -> extract that span as 16k mono audio -> voiceprint
  -> VoiceSample (emb + q).

The results are grouped by local_id (P# / SW) into CastEvidence, which feeds the
probability cloud's coarse recall and learning.

Frame and audio extraction need the clip's bytes locally (PyAV), so the caller
first downloads the clip from OSS to a local temp file. The small models run
locally; that does not conflict with handing the multimodal model a URL only.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from loguru import logger

from personos.identity.screenplay import ClipScript
from personos.identity.types import CastEvidence, FacePick, VoiceSample

_THIRDS = {"left": 1 / 6, "center": 1 / 2, "right": 5 / 6}   # centres of the horizontal thirds
_MIN_VOICE_SEC = 0.4          # skip segments under 0.4s: too short for a stable voiceprint
_VOICE_FULL_Q_SEC = 4.0       # q = min(1, dur/4), so 4s of clean speech scores full marks
# Face acceptance guards. All of them only subtract: a face we refuse is simply not harvested,
# which is always cheaper than filing a wrong person's face under a character (a bad template
# cannot be unlearned from the cloud).
# Calibrated on the M3-Bench living-room clips at 1280px: legitimate faces had boxes of 44-180px,
# det_score 0.73-0.90, eye-distance/box-width 0.39-0.44 and the nose between the eyes; the
# back-of-head / ear detections that polluted a character had boxes of 61-86px, eye/box 0.05-0.17
# and the nose far outside the eye span. 40px is the smallest box the recogniser embeds usefully.
_MIN_FACE_PX = 40
_MIN_FACE_FRAC = 0.025        # of frame width, so a very wide frame does not accept specks
_MIN_DET_SCORE = 0.6
_MIN_EYE_BOX_RATIO = 0.30     # eye distance / box width; side and rear views collapse it
_NOSE_SPAN_TOL = 0.10         # nose may overshoot the eye span by this fraction of the eye distance
_AMBIGUITY_MARGIN = 0.08      # normalised x distance within which a runner-up counts as a tie
_FALLBACK_OFFSETS = (0.5, -0.5, 1.0, -1.0)   # seconds around the nominated instant, tried in order
# The cloud is deliberately threshold-free (cloud.py), so there is no shared "same person" cutoff.
# This one is only used to drop a face that two different casts would both claim; it is
# conservative on purpose (ArcFace same-person cosines sit around 0.4-0.7, strangers near 0).
_CROSS_CAST_SAME_FACE_COS = 0.5
_VOICE_WAV_MAX_SEC = 10.0     # cap the audio sample attached to arbitration at 10s; the embedding still uses the whole segment


def reject_reason(det: Any, frame_width: int) -> str | None:
    """Why this detection must not be harvested, or None when it is acceptable.

    Landmark checks are skipped when the detector gives no landmarks.
    """
    x0, _, x1, _ = det.bbox
    width = x1 - x0
    if width < max(_MIN_FACE_PX, _MIN_FACE_FRAC * frame_width):
        return f"too_small({width}px)"
    score = float(getattr(det, "det_score", 1.0))
    if score < _MIN_DET_SCORE:
        return f"low_det_score({score:.2f})"
    kps = getattr(det, "kps", None)
    if kps is not None and len(kps) >= 3:
        left_eye, right_eye, nose = (np.asarray(kps[i], dtype=float) for i in range(3))
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist / max(1, width) < _MIN_EYE_BOX_RATIO:
            return f"not_frontal(eye/box={eye_dist / max(1, width):.2f})"
        lo, hi = sorted((left_eye[0], right_eye[0]))
        tol = _NOSE_SPAN_TOL * eye_dist
        if not (lo - tol <= nose[0] <= hi + tol):
            return "not_frontal(nose_outside_eyes)"
    return None


def pick_face(dets: list[Any], pos: str, frame_width: int) -> Any | None:
    """Pick the face at the nominated horizontal position, then check it agrees.

    We would rather return nothing than return the wrong person:

    - with a pos: take the face whose centre is closest to that third's centre,
      then verify that this face's own third really is pos, and refuse otherwise.
      This is what stops us from filing the neighbour's face under someone whose
      own face simply was not detected;
    - without a pos: accept only if there is exactly one face. With several we
      cannot tell them apart, so we refuse;
    - with a pos and several faces: refuse when the runner-up also sits in the nominated
      third, or is almost as close to its centre as the winner. Two people side by side
      in the same third are exactly how one person's face gets filed under the other.
    """
    if not dets:
        return None

    def _cr(d: Any) -> float:
        x0, _, x1, _ = d.bbox
        return (x0 + x1) / 2 / max(1, frame_width)

    if pos not in _THIRDS:
        return dets[0] if len(dets) == 1 else None
    best = min(dets, key=lambda d: abs(_cr(d) - _THIRDS[pos]))
    nearest = min(_THIRDS, key=lambda k: abs(_cr(best) - _THIRDS[k]))
    if nearest != pos:
        return None
    others = [d for d in dets if d is not best]
    if others:
        runner = min(others, key=lambda d: abs(_cr(d) - _THIRDS[pos]))
        runner_third = min(_THIRDS, key=lambda k: abs(_cr(runner) - _THIRDS[k]))
        gap = abs(_cr(runner) - _THIRDS[pos]) - abs(_cr(best) - _THIRDS[pos])
        if runner_third == pos or gap <= _AMBIGUITY_MARGIN:
            return None
    return best


def _body_crop_b64(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """The body shot: grow the face box out by human proportions (4x the width,
    1.2 face-heights above, 7 face-heights below) and encode as JPEG via PIL,
    which gets the RGB order right.

    Posture and clothing are one of the strongest recognition signals, especially
    when the face is small and blurry, so we produce this asset even when the face
    itself is unclear. It carries no embedding.
    """
    import base64
    import io

    from PIL import Image
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    cx = (x0 + x1) / 2
    bx0, bx1 = int(max(0, cx - fw * 2.0)), int(min(w, cx + fw * 2.0))
    by0, by1 = int(max(0, y0 - fh * 1.2)), int(min(h, y1 + fh * 7.0))
    crop = frame[by0:by1, bx0:bx1]
    if crop.size == 0:
        crop = frame
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _face_q(det: Any) -> float:
    """FaceDet -> quality q in [0,1].

    Use the AdaFace norm-quality when the detector supplies one, and fall back to
    the normalized blur (sharpness) score otherwise.
    """
    q = float(getattr(det, "quality", -1.0))
    return q if q >= 0 else float(getattr(det, "blur_score", 0.0))


def _frame_at(path: str, t: float) -> np.ndarray | None:
    import av
    c = av.open(path)
    try:
        stream = c.streams.video[0]
        try:
            c.seek(max(0, int(t / stream.time_base)), stream=stream, any_frame=False, backward=True)
        except Exception:  # noqa: BLE001  if seek fails, fall back to decoding from the start
            pass
        for frame in c.decode(video=0):
            if frame.time is not None and frame.time >= t - 0.05:
                return frame.to_ndarray(format="rgb24")
        return None
    finally:
        c.close()


def _wav_at(path: str, t0: float, t1: float) -> bytes:
    """Extract audio over [t0,t1] as 16k mono PCM16 wav bytes.

    On failure, or when there is no audio track, return b"" so the voiceprint step
    degrades gracefully by skipping this segment.
    """
    try:
        import av
        import soundfile as sf
    except ImportError:
        return b""
    c = av.open(path)
    try:
        if not c.streams.audio:
            return b""
        astream = c.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            c.seek(max(0, int(t0 / astream.time_base)), stream=astream, backward=True)
        except Exception:  # noqa: BLE001
            pass
        parts: list[np.ndarray] = []
        for frame in c.decode(audio=0):
            if frame.time is None:
                continue
            if frame.time > t1 + 0.1:
                break
            if frame.time + (frame.samples / max(1, frame.rate)) < t0:
                continue
            for rf in resampler.resample(frame):
                parts.append(rf.to_ndarray().reshape(-1))
        if not parts:
            return b""
        pcm = np.concatenate(parts).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001  flaky audio extraction must not block: this segment just gets no voiceprint
        logger.warning(f"audio extraction failed [{t0:.1f},{t1:.1f}]: {e}")
        return b""
    finally:
        c.close()


def _drop_cross_cast_duplicates(ev: dict[str, CastEvidence]) -> None:
    """One face cannot belong to two people: when faces harvested for two different casts in the
    same clip look like the same person, one of the nominations was wrong and we cannot tell
    which, so both are dropped (subtraction)."""
    doomed: set[tuple[str, int]] = set()
    casts = list(ev)
    for i, a in enumerate(casts):
        for b in casts[i + 1:]:
            for ia, fa in enumerate(ev[a].faces):
                for ib, fb in enumerate(ev[b].faces):
                    if fa.embedding is None or fb.embedding is None:
                        continue
                    cos = float(np.dot(fa.embedding, fb.embedding)
                                / (np.linalg.norm(fa.embedding) * np.linalg.norm(fb.embedding) + 1e-9))
                    if cos >= _CROSS_CAST_SAME_FACE_COS:
                        logger.info(f"face rejected local_id={a}/{b} t={fa.t:.1f}/{fb.t:.1f} "
                                    f"reasons=['same_face_in_two_casts(cos={cos:.2f})']")
                        doomed.update({(a, ia), (b, ib)})
    for cast, e in ev.items():
        e.faces = [f for i, f in enumerate(e.faces) if (cast, i) not in doomed]


def harvest_clip(clip_path: str, script: ClipScript,
                 backends: dict[str, Any]) -> dict[str, CastEvidence]:
    """Harvest assets per the screenplay, returning {local_id: CastEvidence}.

    SW gets no face, because the wearer cannot film themselves.
    """
    face_det = backends["face_detector"]
    voiceprint = backends["voiceprint"]
    ev: dict[str, CastEvidence] = {}

    def _ev(local_id: str) -> CastEvidence:
        return ev.setdefault(local_id, CastEvidence(cast_id=local_id))

    # Faces: detect on the nominated frame, drop unusable detections, pick the box by
    # horizontal third, then check it agrees. SW is skipped. A nomination often lands on a
    # turned-away moment, so when the nominated instant yields nothing we retry a second or
    # so either side before giving up.
    for nom in script.nominations:
        if nom.local_id == "SW":
            continue
        why: list[str] = []
        for off in (0.0, *_FALLBACK_OFFSETS):
            t = nom.t + off
            if t < 0:
                continue
            frame = _frame_at(clip_path, t)
            if frame is None:
                continue
            w = frame.shape[1]
            dets = []
            for d in face_det.detect(frame):
                reason = reject_reason(d, w)
                if reason:
                    why.append(f"{reason}@{t:.1f}")
                else:
                    dets.append(d)
            box = pick_face(dets, nom.pos, w)
            if box is None:
                if dets:
                    why.append(f"no_unambiguous_face_at_pos({nom.pos or '-'})@{t:.1f}")
                continue   # position disagrees / ambiguous / nobody usable -> try the next instant, else harvest nothing
            _ev(nom.local_id).faces.append(FacePick(
                t=t, embedding=np.asarray(box.embedding, dtype=np.float32),
                q=_face_q(box), crop_b64=getattr(box, "crop_b64", ""),
                body_crop_b64=_body_crop_b64(frame, tuple(box.bbox)), descriptor=nom.desc))
            break
        else:
            logger.info(f"face rejected local_id={nom.local_id} t={nom.t:.1f} pos={nom.pos or '-'} "
                        f"reasons={why or ['no_face_detected']}")
    _drop_cross_cast_duplicates(ev)

    # Voiceprints: extract each voice range and embed it. Segments under 0.4s are
    # skipped, q = min(1, dur/4), and the audio shown to arbitration is cut to the
    # first 10s.
    for vr in script.voice_ranges:
        dur = vr.t1 - vr.t0
        if dur < _MIN_VOICE_SEC:
            continue
        wav = _wav_at(clip_path, vr.t0, vr.t1)
        if not wav:
            continue
        try:
            emb = np.asarray(voiceprint.embed(wav), dtype=np.float32)   # the embedding uses the whole segment
        except Exception as e:  # noqa: BLE001
            logger.warning(f"voiceprint embed failed {vr.local_id} [{vr.t0:.1f},{vr.t1:.1f}]: {e}")
            continue
        # Arbitration only listens to the first _VOICE_WAV_MAX_SEC seconds; given a
        # longer sample the model attends to the opening anyway.
        wav_sample = wav if dur <= _VOICE_WAV_MAX_SEC else \
            (_wav_at(clip_path, vr.t0, vr.t0 + _VOICE_WAV_MAX_SEC) or wav)
        _ev(vr.local_id).voices.append(VoiceSample(
            t0=vr.t0, t1=vr.t1, embedding=emb,
            q=min(1.0, dur / _VOICE_FULL_Q_SEC), wav_bytes=wav_sample))
    return ev
