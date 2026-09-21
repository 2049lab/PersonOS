"""纯 numpy 假后端:不下模型、不联网,供单测与骨架冒烟。

设计:输出对同一输入确定(seed 自输入内容),便于断言;向量维度与真后端一致
(face 512 / voice 192),下游 dim 校验行为与真跑一致。
"""

from __future__ import annotations

import base64
import hashlib

import numpy as np

from personos.identity.backends.base import FaceDet

_FACE_DIM = 512
_VOICE_DIM = 192


def _seed_vec(seed_bytes: bytes, dim: int) -> np.ndarray:
    """由输入内容派生的确定归一化向量。"""
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class MockFaceDetector:
    """整帧当一张脸,向量由帧内容派生;质量固定 0.8。"""

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
    """假 Omni:chat 返回可配的固定剧本(V2 mock 管线测试注入)。"""

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
