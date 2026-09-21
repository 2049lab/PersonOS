"""推理设备选择:gpu 优先(cuda>mps)否则 cpu(移植 mneme device.py)。

优先级:后端专属 env > 全局 PERSONOS_VIDEO_DEVICE > auto(cuda>mps>cpu,探测失败静默退 cpu)。
现在 CPU 服务器 → cpu;换 GPU 服务器 → cuda 自动生效;本机 Mac → mps。
allow_mps:某后端在 MPS 未过数值等价校验(或依赖不支持)时置 False,auto 只在 cuda/cpu 间选。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def pick_device(env_key: str, *, allow_mps: bool = True) -> str:
    want = (os.getenv(env_key) or os.getenv("PERSONOS_VIDEO_DEVICE") or "auto").strip().lower()
    if want and want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001  torch 缺失/探测失败 → 一律退 CPU
        pass
    return "cpu"
