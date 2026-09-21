"""Assemble the real backends: face detection and recognition (InsightFace/ArcFace),
voiceprints (ECAPA), and the multimodal model.

These fail fast: if a configured backend is unavailable we raise instead of
quietly swapping in a stand-in model, because a stand-in produces results that
look plausible and are worthless.

Device selection goes through backends.device (GPU first, cpu as the fallback).
"""

from __future__ import annotations

import os
from typing import Any


def make_real_backends() -> dict[str, Any]:
    from personos.identity.backends.omni import OmniRunner
    from personos.identity.backends.voiceprint_ecapa import EcapaVoiceprint
    return {"mm_runner": OmniRunner(), "face_detector": _make_face_detector(),
            "voiceprint": EcapaVoiceprint()}


def _make_face_detector() -> Any:
    choice = os.getenv("PERSONOS_FACE_RECOGNIZER", "arcface").strip().lower()
    if choice == "arcface":
        from personos.identity.backends.face_insightface import InsightFaceDetector
        return InsightFaceDetector()
    # AdaFace (quality-adaptive, steadier on small faces) is the intended optional
    # replacement, but it needs its own weights and net; for now we ship arcface.
    raise ValueError(f"PERSONOS_FACE_RECOGNIZER currently supports arcface; got {choice!r}")
