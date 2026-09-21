"""Value types shared by the backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceDet:
    """One detected face and its recognition vector.

    crop_b64 is what AssetHarvest uploads to OSS; the database keeps only the key.
    """

    bbox: tuple[int, int, int, int]
    crop_b64: str
    det_score: float
    blur_score: float
    embedding: np.ndarray        # normalized vector (512-d for faces)
    quality: float = -1.0
    norm: float = -1.0
