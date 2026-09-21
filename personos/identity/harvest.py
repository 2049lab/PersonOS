"""AssetHarvest: pull assets out of a clip, guided by the screenplay's nominations
and voice ranges.

We detect on the **nominated frame only** and pick the box by **coarse position**;
there is no full-video tracking:

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
import logging
from typing import Any

import numpy as np

from personos.identity.screenplay import ClipScript
from personos.identity.types import CastEvidence, FacePick, VoiceSample

logger = logging.getLogger(__name__)

_THIRDS = {"left": 1 / 6, "center": 1 / 2, "right": 5 / 6}   # centres of the horizontal thirds
_MIN_VOICE_SEC = 0.4          # skip segments under 0.4s: too short for a stable voiceprint
_VOICE_FULL_Q_SEC = 4.0       # q = min(1, dur/4), so 4s of clean speech scores full marks
_VOICE_WAV_MAX_SEC = 10.0     # cap the audio sample attached to arbitration at 10s; the embedding still uses the whole segment


def pick_face(dets: list[Any], pos: str, frame_width: int) -> Any | None:
    """Pick the face at the nominated horizontal position, then check it agrees.

    We would rather return nothing than return the wrong person:

    - with a pos: take the face whose centre is closest to that third's centre,
      then verify that this face's own third really is pos, and refuse otherwise.
      This is what stops us from filing the neighbour's face under someone whose
      own face simply was not detected;
    - without a pos: accept only if there is exactly one face. With several we
      cannot tell them apart, so we refuse.
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
    return best if nearest == pos else None


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

    # Faces: detect on the single nominated frame, pick the box by horizontal
    # third, then check it agrees. SW is skipped.
    for nom in script.nominations:
        if nom.local_id == "SW":
            continue
        frame = _frame_at(clip_path, nom.t)
        if frame is None:
            continue
        w = frame.shape[1]
        dets = face_det.detect(frame)
        box = pick_face(dets, nom.pos, w)
        if box is None:
            continue   # position disagrees, or several faces and no position to tell them apart -> harvest nothing this time
        _ev(nom.local_id).faces.append(FacePick(
            t=nom.t, embedding=np.asarray(box.embedding, dtype=np.float32),
            q=_face_q(box), crop_b64=getattr(box, "crop_b64", ""),
            body_crop_b64=_body_crop_b64(frame, tuple(box.bbox)), descriptor=nom.desc))

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
