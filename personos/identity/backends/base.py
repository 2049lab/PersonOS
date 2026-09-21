"""后端共享值类型(对齐 mneme backends/base.py)。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceDet:
    """一张检出的人脸及其识别向量。crop_b64 供 AssetHarvest 上传 OSS(库里只留 key)。"""

    bbox: tuple[int, int, int, int]
    crop_b64: str
    det_score: float
    blur_score: float
    embedding: np.ndarray        # 归一化向量(face 512d)
    quality: float = -1.0
    norm: float = -1.0
