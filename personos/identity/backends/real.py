"""真后端装配:人脸检测识别(InsightFace/ArcFace)+ 声纹(ECAPA)+ Omni(V2 接线)。

fail fast:配置的后端不可用时直接抛,绝不用替身模型跑出"看似合理实则无效"的结果。
device 走 backends.device(gpu 优先,cpu 兜底)。
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
    # AdaFace(质量自适应,小脸更稳)为后续可选替换,需 adaface 权重 + net;V1 先 arcface。
    raise ValueError(f"PERSONOS_FACE_RECOGNIZER 目前支持 arcface;收到 {choice!r}")
