"""SpeechBrain ECAPA-TDNN 声纹后端(移植 mneme voiceprint_ecapa.py)。

一段 16kHz 单声道 wav(bytes)→ 192d L2 归一化说话人向量。懒加载。
env:PERSONOS_ECAPA_MODEL(默认 speechbrain/spkrec-ecapa-voxceleb)、PERSONOS_ECAPA_DEVICE、PERSONOS_ECAPA_DIR。
device 排除 MPS(speechbrain 在 MPS 有兼容 bug);CPU ~37ms/段,非热点。
"""

from __future__ import annotations

import io
import logging
import os

import numpy as np

from personos.identity.backends.device import pick_device

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 8000  # ECAPA 在 <0.5s 不稳,短段零填充


class EcapaVoiceprint:
    def __init__(self) -> None:
        self._model = None
        self._source = os.getenv("PERSONOS_ECAPA_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        self._device = pick_device("PERSONOS_ECAPA_DEVICE", allow_mps=False)  # 见模块注
        self._savedir = os.getenv("PERSONOS_ECAPA_DIR")

    def _ensure(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            logger.info("loading ECAPA voiceprint %s (device=%s)", self._source, self._device)
            kwargs = {"source": self._source, "run_opts": {"device": self._device}}
            if self._savedir:
                kwargs["savedir"] = self._savedir
            try:
                self._model = EncoderClassifier.from_hparams(**kwargs)
            except Exception:  # noqa: BLE001  非 CPU 加载失败 → 退 CPU
                if self._device == "cpu":
                    raise
                logger.warning("ECAPA 于 %s 加载失败,退回 CPU", self._device)
                self._device = "cpu"
                kwargs["run_opts"] = {"device": "cpu"}
                self._model = EncoderClassifier.from_hparams(**kwargs)
        return self._model

    def embed(self, wav_segment: bytes) -> np.ndarray:
        import soundfile as sf
        import torch

        model = self._ensure()
        audio, sr = sf.read(io.BytesIO(wav_segment), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.shape[0] < _MIN_SAMPLES:
            audio = np.pad(audio, (0, _MIN_SAMPLES - audio.shape[0]))
        sig = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(sig).reshape(-1).cpu().numpy().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)
