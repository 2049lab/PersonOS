"""CloudEngine 依赖的证据值类型(移植自 mneme-release anchor/types.py 的最小子集)。

一个 clip 内、某会话 cast 的机器证据:若干人脸提名(FacePick)+ 若干干净语音段(VoiceSample)。
概率云打分只用到 embedding + q(质量),crop 的 OSS key 放 payload 里另行携带。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FacePick:
    """一次提名的机器产物:人脸(+同帧全身)。embedding 为归一化向量;无脸时 None(素材仍保留)。"""

    t: float
    embedding: Optional[np.ndarray]
    q: float                        # 人脸质量 [0,1];无质量刻度→0(数学上自抑制)
    crop_b64: str = ""              # 人脸 crop(PNG b64);持久化时上传 OSS 换 key
    body_crop_b64: str = ""         # 全身 crop(JPG b64,由人脸框按人体比例外扩;体态/服装,认人主力之一)
    face_oss_key: str = ""
    body_oss_key: str = ""
    descriptor: str = ""


@dataclass
class VoiceSample:
    t0: float
    t1: float
    embedding: Optional[np.ndarray]  # 归一化声纹向量
    q: float                         # 干净时长综合质量 [0,1]
    wav_bytes: Optional[bytes] = None  # 该段 16k 单声道 wav(供入 OSS + 喂仲裁听声辨人)
    wav_oss_key: str = ""


@dataclass
class CastEvidence:
    """一个 clip 内、某会话 cast 的全部机器证据。"""

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
        """最佳可听语音段(喂仲裁听声辨人:必须带 wav 字节)。"""
        picks = [v for v in self.voices if v.wav_bytes]
        return max(picks, key=lambda v: v.q) if picks else None


@dataclass(frozen=True)
class CandidateCard:
    """仲裁候选卡(喂 Omni 的人可读素材)。personos 保留 desc 字段(非 mneme text_profile dict)。"""

    character_id: str
    name: str = ""
    desc: str = ""
    face_b64: str = ""
    body_b64: str = ""
    voice_b64: str = ""       # 声纹样本 wav b64(听声辨人)
    last_seen_session: str = ""


def normalized(vector: Any) -> Optional[np.ndarray]:
    """归一化到单位球;非法输入返回 None(缺某模态是合法的)。"""
    try:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return array / norm
