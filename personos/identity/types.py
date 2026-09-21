"""The evidence value types CloudEngine depends on.

Within one clip, the machine evidence for a session cast is a set of face
nominations (FacePick) plus a set of clean speech segments (VoiceSample).
Probability-cloud scoring only reads embedding + q (quality); the OSS key of each
crop rides along separately in the payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FacePick:
    """What one nomination produced: a face, plus the body shot from the same frame.

    embedding is a normalized vector; it is None when no face was found, in which
    case the assets are still kept.
    """

    t: float
    embedding: Optional[np.ndarray]
    q: float                        # face quality in [0,1]; with no quality scale it is 0, which suppresses this pick on its own
    crop_b64: str = ""              # face crop (PNG b64); on persist it is uploaded to OSS and replaced by a key
    body_crop_b64: str = ""         # body shot (JPG b64), grown out of the face box by human proportions; posture and clothing are one of the strongest recognition signals
    face_oss_key: str = ""
    body_oss_key: str = ""
    descriptor: str = ""


@dataclass
class VoiceSample:
    t0: float
    t1: float
    embedding: Optional[np.ndarray]  # normalized voiceprint vector
    q: float                         # combined quality over the clean duration, in [0,1]
    wav_bytes: Optional[bytes] = None  # this segment as 16k mono wav, for OSS and for letting the arbiter recognize the speaker by ear
    wav_oss_key: str = ""


@dataclass
class CastEvidence:
    """All machine evidence for one session cast within one clip."""

    cast_id: str
    faces: list[FacePick] = field(default_factory=list)
    voices: list[VoiceSample] = field(default_factory=list)

    def best_face(self) -> Optional[FacePick]:
        picks = [p for p in self.faces if p.crop_b64]
        return max(picks, key=lambda p: p.q) if picks else None

    def best_body(self) -> Optional[FacePick]:
        picks = [p for p in self.faces if p.body_crop_b64]
        return max(picks, key=lambda p: p.q) if picks else None

    def best_voice_wav(self) -> Optional[VoiceSample]:
        """The best audible speech segment. The arbiter recognizes by ear, so wav bytes are required."""
        picks = [v for v in self.voices if v.wav_bytes]
        return max(picks, key=lambda v: v.q) if picks else None


@dataclass(frozen=True)
class CandidateCard:
    """One arbitration candidate: the human-readable assets handed to the multimodal model.

    We carry a single free-text ``desc`` field rather than a structured text
    profile, because the arbiter reads it as prose anyway.
    """

    character_id: str
    name: str = ""
    desc: str = ""
    face_b64: str = ""
    body_b64: str = ""
    voice_b64: str = ""       # voiceprint sample as wav b64, for recognition by ear
    last_seen_session: str = ""


def normalized(vector: Any) -> Optional[np.ndarray]:
    """Normalize onto the unit sphere. Invalid input returns None, since a missing modality is legal."""
    try:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return array / norm
