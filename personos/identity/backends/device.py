"""Pick the inference device: prefer a GPU (cuda > mps), otherwise cpu.

Precedence: the backend's own env var > the global PERSONOS_VIDEO_DEVICE > auto
(cuda > mps > cpu, silently falling back to cpu if probing fails). This way a
CPU-only server lands on cpu, moving to a GPU server switches to cuda with no
code change, and a local Mac gets mps.

Set allow_mps=False for a backend that has not passed numerical-equivalence
checks on MPS, or whose dependencies do not support it; auto then chooses only
between cuda and cpu.
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
    except Exception:  # noqa: BLE001  torch missing or probing failed -> always fall back to CPU
        pass
    return "cpu"
