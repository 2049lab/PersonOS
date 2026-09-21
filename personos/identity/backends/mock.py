"""Pure-numpy fake backends: no model downloads, no network, for unit tests and
smoke runs of the skeleton.

Two design choices make them useful for assertions. Output is deterministic for a
given input, because the seed is derived from the input content. And the vector
dimensions match the real backends (512 for faces, 192 for voice), so downstream
dimension checks behave exactly as they do on a real run.
"""

from __future__ import annotations

import base64
import hashlib

import numpy as np

from personos.identity.backends.base import FaceDet

_FACE_DIM = 512
_VOICE_DIM = 192


def _seed_vec(seed_bytes: bytes, dim: int) -> np.ndarray:
    """A deterministic normalized vector derived from the input content."""
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class MockFaceDetector:
    """Treat the whole frame as one face, derive the vector from the frame content, and fix quality at 0.8."""

    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        emb = _seed_vec(np.ascontiguousarray(frame).tobytes(), _FACE_DIM)
        return [FaceDet(bbox=(0, 0, int(w), int(h)), crop_b64="",
                        det_score=0.99, blur_score=100.0,
                        embedding=emb, quality=0.8, norm=25.0)]


class MockVoiceprint:
    def embed(self, wav_segment: bytes) -> np.ndarray:
        return _seed_vec(wav_segment or b"", _VOICE_DIM)


class MockOmni:
    """A fake multimodal model: chat returns a canned screenplay, injected by pipeline tests."""

    def __init__(self, canned: str = "END") -> None:
        self.canned = canned
        self.calls: list[dict] = []

    def chat(self, prompt: str, *, video_url: str | None = None,
             images_b64: list[str] | None = None, **kw) -> str:
        self.calls.append({"prompt": prompt, "video_url": video_url,
                           "images_b64": images_b64, **kw})
        return self.canned


def _blank_b64() -> str:
    return base64.b64encode(b"\x00").decode()
