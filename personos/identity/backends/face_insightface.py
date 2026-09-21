"""人脸检测 + ArcFace 识别(移植 mneme face_insightface.py)。

RetinaFace/SCRFD 检测(InsightFace buffalo_l)→ 512d ArcFace 归一化向量;
blur_score = 归一化 Laplacian 方差(素材质量排序用)。检测输入为 RGB 帧(内部转 BGR)。
env:PERSONOS_FACE_DET_SIZE(检测画布,默认 1280)、PERSONOS_ORT_EP(onnx EP,mac 用 CoreML)。

注:AdaFace(质量自适应,小脸更稳)为后续可选的 embedding 替换项;V1 先用 ArcFace 自带权重
(FaceAnalysis 首次自动下载 buffalo_l),自包含、零额外权重管理。
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
    """检测网络输入画布(默认 1280;库默认 640——大画布利于检出/对齐小脸)。"""
    try:
        n = int(os.getenv("PERSONOS_FACE_DET_SIZE", "1280"))
    except ValueError:
        n = 1280
    return (n, n)


def _ort_providers() -> tuple[str, ...] | None:
    """onnxruntime EP 选择(env PERSONOS_ORT_EP):mac 用 CoreML(ANE/GPU),Linux/CPU 用 None=CPU。"""
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
        logger.warning("PERSONOS_ORT_EP=coreml 但 CoreML EP 不可用,退回 CPU")
    return None


def _get_face_app(model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640),
                  allowed_modules: tuple[str, ...] | None = None):
    """懒加载单例 InsightFace FaceAnalysis(首次自动下载权重到 ~/.insightface)。"""
    modules_key = tuple(allowed_modules) if allowed_modules is not None else None
    providers = _ort_providers()
    cache_key = (model_name, det_size, modules_key, providers)
    with _face_app_lock:
        if cache_key in _face_app_cache:
            return _face_app_cache[cache_key]
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError("需要 insightface:pip install insightface onnxruntime "
                              "opencv-python-headless(服务器用 headless,非 GUI 版)") from e
        kwargs = {"allowed_modules": list(allowed_modules)} if allowed_modules is not None else {}
        if providers is not None:
            kwargs["providers"] = list(providers)
        app = FaceAnalysis(name=model_name, **kwargs)
        app.prepare(ctx_id=-1, det_size=det_size)   # ctx_id=-1 走 EP(CPU/CoreML)
        _face_app_cache[cache_key] = app
        ep = providers[0] if providers else "CPU"
        logger.info(f"InsightFace {model_name} loaded (det_size={det_size}, ep={ep})")
        return app


@dataclass
class InsightFaceDetector:
    """RetinaFace 检测 + ArcFace 识别。detect(RGB 帧)→ list[FaceDet]。"""

    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = field(default_factory=_default_det_size)
    min_det_score: float = 0.5
    # 只加载真正用到的两个模块:detect() 只读 bbox / det_score / normed_embedding。
    # buffalo_l 包里另外三个(1k3d68 三维关键点 143MB、2d106det 5MB、genderage 1.3MB)
    # 我们一个字段都没用 —— 不限定的话 FaceAnalysis 会把目录里的全部加载进内存。
    # 镜像里对应地只保留 det_10g + w600k_r50(见 Dockerfile 的模型预置段),两边一致。
    allowed_modules: tuple[str, ...] = ("detection", "recognition")

    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        app = _get_face_app(self.model_name, self.det_size, self.allowed_modules)
        # buffalo_l 训练于 BGR,PyAV 出 RGB → 检测前交换通道(否则 det_score 掉、向量畸变)
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
                               embedding=emb))
        return out


def _encode_crop(crop: np.ndarray) -> str:
    """RGB 人脸 crop → base64 PNG(供 AssetHarvest 上传 OSS)。"""
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
    """Laplacian 方差清晰度分,归一化 [0,1];先缩到固定高 112 去除尺寸耦合(小脸也公平)。"""
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
    """cv2 缺失时的近似兜底(不缩放,略带尺寸耦合)。"""
    if crop.size == 0:
        return 0.0
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float64)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return min(1.0, float(lap.var()) / 100.0) if lap.size else 0.0
