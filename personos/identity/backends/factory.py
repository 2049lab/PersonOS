"""按 profile 装配后端 dict:{mm_runner, face_detector, voiceprint}。

- "real"(**缺省**):真后端(InsightFace + ECAPA + xhs MAAS Omni),需模型权重与依赖就位;
- "mock":假后端(单测/冒烟,零依赖)——**只可用于测试**。

为什么缺省是 real:mock 会**编造剧本**(MockOmni)、**生成假人脸向量**(MockFaceDetector),
一旦在生产跑起来,写进库的是伪造的记忆内容和身份数据,而且外部看一切正常。
实测事故:SIT 首次部署没设这个 env → 缺省 mock,同时镜像又缺 PyAV,视频链路静默失效。
缺省 real 的代价只是"缺依赖时当场炸"——那是我们想要的失败方式。
"""

from __future__ import annotations

import os
from typing import Any


def make_backends(profile: str | None = None) -> dict[str, Any]:
    profile = (profile or os.getenv("PERSONOS_VIDEO_BACKEND", "real")).strip().lower()
    if profile == "mock":
        from personos.identity.backends.mock import (
            MockFaceDetector, MockOmni, MockVoiceprint,
        )
        return {"mm_runner": MockOmni(), "face_detector": MockFaceDetector(),
                "voiceprint": MockVoiceprint()}
    if profile in ("real", "wearable"):
        # 真后端在 V1 real 移植落地(AdaFace/InsightFace/ECAPA/Omni);缺依赖 fail fast。
        from personos.identity.backends.real import make_real_backends
        return make_real_backends()
    raise ValueError(f"PERSONOS_VIDEO_BACKEND 必须是 mock/real;收到 {profile!r}")


__all__ = ["make_backends"]
