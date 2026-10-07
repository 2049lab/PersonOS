"""Face detection plus ArcFace recognition.

RetinaFace/SCRFD detection (InsightFace buffalo_l) produces a 512-d normalized
ArcFace vector. blur_score is a normalized Laplacian variance, used to rank asset
quality. detect() takes an RGB frame and converts to BGR internally.

Env: PERSONOS_FACE_DET_SIZE (the detection canvas, default 1280) and
PERSONOS_ORT_EP (the onnx execution provider; CoreML on mac).

Note: AdaFace (quality-adaptive, steadier on small faces) is a possible future
replacement for the embedding. For now we use ArcFace's own weights, which
FaceAnalysis downloads on first use — self-contained, with no extra weight
management.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from personos.identity.backends.base import FaceDet

logger = logging.getLogger(__name__)

_face_app_lock = threading.Lock()
_face_app_cache: dict = {}


def _default_det_size() -> tuple[int, int]:
    """Input canvas for the detection network.

    We default to 1280 where the library defaults to 640, because a larger canvas
    both finds and aligns small faces better.
    """
    try:
        n = int(os.getenv("PERSONOS_FACE_DET_SIZE", "1280"))
    except ValueError:
        n = 1280
    return (n, n)


def _ort_providers() -> tuple[str, ...] | None:
    """Choose the onnxruntime execution provider (env PERSONOS_ORT_EP).

    On mac that means CoreML, to reach the ANE/GPU; on Linux/CPU we return None,
    which means plain CPU.
    """
    mode = os.getenv("PERSONOS_ORT_EP", "auto").lower()
    if mode in ("cpu", "off", "0"):
        return None
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return None
    if "CoreMLExecutionProvider" in avail:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    if mode == "coreml":
        logger.warning("PERSONOS_ORT_EP=coreml but the CoreML EP is unavailable, falling back to CPU")
    return None


def _get_face_app(model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640),
                  allowed_modules: tuple[str, ...] | None = None):
    """Lazily built singleton InsightFace FaceAnalysis.

    The first call downloads the weights into ~/.insightface.
    """
    modules_key = tuple(allowed_modules) if allowed_modules is not None else None
    providers = _ort_providers()
    cache_key = (model_name, det_size, modules_key, providers)
    with _face_app_lock:
        if cache_key in _face_app_cache:
            return _face_app_cache[cache_key]
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("insightface is required: pip install insightface onnxruntime "
                              "opencv-python-headless (use the headless build on a server, "
                              "not the GUI one)") from e
        kwargs = {"allowed_modules": list(allowed_modules)} if allowed_modules is not None else {}
        if providers is not None:
            kwargs["providers"] = list(providers)
        app = FaceAnalysis(name=model_name, **kwargs)
        app.prepare(ctx_id=-1, det_size=det_size)   # ctx_id=-1 defers to the execution provider (CPU/CoreML)
        _face_app_cache[cache_key] = app
        ep = providers[0] if providers else "CPU"
        logger.info(f"InsightFace {model_name} loaded (det_size={det_size}, ep={ep})")
        return app


@dataclass
class InsightFaceDetector:
    """RetinaFace detection plus ArcFace recognition. detect(RGB frame) -> list[FaceDet]."""

    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = field(default_factory=_default_det_size)
    min_det_score: float = 0.5
    # Load only the two modules we actually use: detect() reads nothing but bbox,
    # det_score and normed_embedding. We touch no field from the other three in the
    # buffalo_l bundle (1k3d68 3-d landmarks at 143MB, 2d106det at 5MB, genderage
    # at 1.3MB) — and without this restriction FaceAnalysis loads everything in the
    # directory into memory. The container image correspondingly keeps only
    # det_10g + w600k_r50 (see the model pre-seeding section of the Dockerfile), so
    # the two sides agree.
    allowed_modules: tuple[str, ...] = ("detection", "recognition")

    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        app = _get_face_app(self.model_name, self.det_size, self.allowed_modules)
        # buffalo_l was trained on BGR and PyAV hands us RGB, so swap channels before
        # detecting; otherwise det_score drops and the embeddings come out distorted.
        bgr = (np.ascontiguousarray(frame[:, :, ::-1])
               if frame.ndim == 3 and frame.shape[2] == 3 else frame)
        out: list[FaceDet] = []
        for f in app.get(bgr):
            if float(f.det_score) < self.min_det_score:
                continue
            x1, y1, x2, y2 = map(int, f.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            emb = np.asarray(getattr(f, "normed_embedding", None) if hasattr(f, "normed_embedding")
                             else f.embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            out.append(FaceDet(bbox=(x1, y1, x2, y2), crop_b64=_encode_crop(crop),
                               det_score=float(f.det_score), blur_score=_laplacian_blur(crop),
                               embedding=emb,
                               kps=(np.asarray(f.kps, dtype=np.float32) if getattr(f, "kps", None) is not None else None)))
        return out


def _encode_crop(crop: np.ndarray) -> str:
    """Turn an RGB face crop into a base64 PNG for AssetHarvest to upload to OSS."""
    from PIL import Image
    img = Image.fromarray(crop)
    min_side = min(img.size)
    if min_side <= 10:
        scale = 16 / max(1, min_side)
        img = img.resize((max(16, int(round(img.width * scale))),
                          max(16, int(round(img.height * scale)))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _laplacian_blur(crop: np.ndarray) -> float:
    """A sharpness score from Laplacian variance, normalized to [0,1].

    We first resize to a fixed height of 112 to remove the coupling with crop
    size, so small faces are scored fairly.
    """
    try:
        import cv2
    except ImportError:
        return _laplacian_blur_numpy(crop)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    if h != 112:
        scale = 112 / float(h)
        gray = cv2.resize(gray, (max(1, int(round(w * scale))), 112),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 100.0)


def _laplacian_blur_numpy(crop: np.ndarray) -> float:
    """An approximate fallback for when cv2 is missing: no resize, so slightly size-coupled."""
    if crop.size == 0:
        return 0.0
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float64)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return min(1.0, float(lap.var()) / 100.0) if lap.size else 0.0
