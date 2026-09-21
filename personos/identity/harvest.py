"""AssetHarvest:按剧本的提名/语音段,从 clip 抽素材(移植 mneme AssetHarvest 思路 + 粗格子归属)。

只在**提名帧**单帧检测、按**粗格子**挑框(定案 §4,不跑全帧 track):
- 每条 nom(t + pos 粗格子)→ 抽该帧 → 人脸检测 → 按格挑框 → FacePick(emb + q + crop_b64);
- 每条 voice(t0..t1)→ 抽该段 16k 单声道音频 → 声纹 → VoiceSample(emb + q)。
按 local_id(P#/SW)聚合成 CastEvidence,供概率云粗召回 + 学习。

抽帧/抽音频需 clip 本地字节(PyAV),故调用方先把 OSS clip 下载到本地临时文件(小模型本地处理,
与"给 Omni 只传 URL"不冲突,定案 §5.3)。
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from personos.identity.screenplay import ClipScript
from personos.identity.types import CastEvidence, FacePick, VoiceSample

logger = logging.getLogger(__name__)

_THIRDS = {"left": 1 / 6, "center": 1 / 2, "right": 5 / 6}   # 横向三分位中心(对齐 mneme)
_MIN_VOICE_SEC = 0.4          # <0.4s 段跳过(对齐 mneme,太短声纹不稳)
_VOICE_FULL_Q_SEC = 4.0       # q=min(1, dur/4):4s 干净语音给满分(对齐 mneme)
_VOICE_WAV_MAX_SEC = 10.0     # 仲裁附的音频样本上限 10s(对齐 mneme;embedding 用全段)


def pick_face(dets: list[Any], pos: str, frame_width: int) -> Any | None:
    """按提名的横向位置挑脸,并做一致性校验(移植 mneme _pick_face,宁缺毋滥)。

    - 有 pos:挑横向最接近该三分位中心的脸,再校验这张脸自己实际所在三分位 == pos,否则拒
      (防"被提名的人没检出、却把旁边人的脸塞进他的档");
    - 无 pos:仅当只有一张脸才收,多脸直接拒(分不清)。
    """
    if not dets:
        return None

    def _cr(d: Any) -> float:
        x0, _, x1, _ = d.bbox
        return (x0 + x1) / 2 / max(1, frame_width)

    if pos not in _THIRDS:
        return dets[0] if len(dets) == 1 else None
    best = min(dets, key=lambda d: abs(_cr(d) - _THIRDS[pos]))
    nearest = min(_THIRDS, key=lambda k: abs(_cr(best) - _THIRDS[k]))
    return best if nearest == pos else None


def _body_crop_b64(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """全身 crop:人脸框按人体比例外扩(宽×4、上 1.2 脸高、下 7 脸高),PIL 编码 JPG(RGB 正确)。

    体态/服装是认人主力之一(脸小而糊时尤其),即便脸不清也要给这张料。无 embedding。
    """
    import base64
    import io

    from PIL import Image
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    cx = (x0 + x1) / 2
    bx0, bx1 = int(max(0, cx - fw * 2.0)), int(min(w, cx + fw * 2.0))
    by0, by1 = int(max(0, y0 - fh * 1.2)), int(min(h, y1 + fh * 7.0))
    crop = frame[by0:by1, bx0:bx1]
    if crop.size == 0:
        crop = frame
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _face_q(det: Any) -> float:
    """FaceDet → 质量 q∈[0,1]:有 AdaFace norm-质量用它,否则用归一化 blur(清晰度)兜底。"""
    q = float(getattr(det, "quality", -1.0))
    return q if q >= 0 else float(getattr(det, "blur_score", 0.0))


def _frame_at(path: str, t: float) -> np.ndarray | None:
    import av
    c = av.open(path)
    try:
        stream = c.streams.video[0]
        try:
            c.seek(max(0, int(t / stream.time_base)), stream=stream, any_frame=False, backward=True)
        except Exception:  # noqa: BLE001  seek 失败退化为从头解
            pass
        for frame in c.decode(video=0):
            if frame.time is not None and frame.time >= t - 0.05:
                return frame.to_ndarray(format="rgb24")
        return None
    finally:
        c.close()


def _wav_at(path: str, t0: float, t1: float) -> bytes:
    """抽 [t0,t1] 音频 → 16k 单声道 PCM16 wav 字节;失败/无音轨返回 b""(声纹优雅跳过)。"""
    try:
        import av
        import soundfile as sf
    except ImportError:
        return b""
    c = av.open(path)
    try:
        if not c.streams.audio:
            return b""
        astream = c.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            c.seek(max(0, int(t0 / astream.time_base)), stream=astream, backward=True)
        except Exception:  # noqa: BLE001
            pass
        parts: list[np.ndarray] = []
        for frame in c.decode(audio=0):
            if frame.time is None:
                continue
            if frame.time > t1 + 0.1:
                break
            if frame.time + (frame.samples / max(1, frame.rate)) < t0:
                continue
            for rf in resampler.resample(frame):
                parts.append(rf.to_ndarray().reshape(-1))
        if not parts:
            return b""
        pcm = np.concatenate(parts).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001  抽音频不稳时不阻塞:该段无声纹
        logger.warning(f"抽音频失败 [{t0:.1f},{t1:.1f}]: {e}")
        return b""
    finally:
        c.close()


def harvest_clip(clip_path: str, script: ClipScript,
                 backends: dict[str, Any]) -> dict[str, CastEvidence]:
    """按剧本抽素材,返回 {local_id: CastEvidence}。SW 不抽脸(佩戴者拍不到自己)。"""
    face_det = backends["face_detector"]
    voiceprint = backends["voiceprint"]
    ev: dict[str, CastEvidence] = {}

    def _ev(local_id: str) -> CastEvidence:
        return ev.setdefault(local_id, CastEvidence(cast_id=local_id))

    # 人脸:提名帧单帧检测 + 横向三分位挑框 + 一致性校验(SW 跳过)
    for nom in script.nominations:
        if nom.local_id == "SW":
            continue
        frame = _frame_at(clip_path, nom.t)
        if frame is None:
            continue
        w = frame.shape[1]
        dets = face_det.detect(frame)
        box = pick_face(dets, nom.pos, w)
        if box is None:
            continue   # 位置不符/多脸无位置分不清 → 本次不抽(宁缺毋滥)
        _ev(nom.local_id).faces.append(FacePick(
            t=nom.t, embedding=np.asarray(box.embedding, dtype=np.float32),
            q=_face_q(box), crop_b64=getattr(box, "crop_b64", ""),
            body_crop_b64=_body_crop_b64(frame, tuple(box.bbox)), descriptor=nom.desc))

    # 声纹:语音段抽音频 + embed(对齐 mneme:<0.4s 跳过;q=min(1,dur/4);仲裁音频截前 10s)
    for vr in script.voice_ranges:
        dur = vr.t1 - vr.t0
        if dur < _MIN_VOICE_SEC:
            continue
        wav = _wav_at(clip_path, vr.t0, vr.t1)
        if not wav:
            continue
        try:
            emb = np.asarray(voiceprint.embed(wav), dtype=np.float32)   # embedding 用全段
        except Exception as e:  # noqa: BLE001
            logger.warning(f"声纹 embed 失败 {vr.local_id} [{vr.t0:.1f},{vr.t1:.1f}]: {e}")
            continue
        # 仲裁只听前 _VOICE_WAV_MAX_SEC 秒(样本太长模型也只听开头)
        wav_sample = wav if dur <= _VOICE_WAV_MAX_SEC else \
            (_wav_at(clip_path, vr.t0, vr.t0 + _VOICE_WAV_MAX_SEC) or wav)
        _ev(vr.local_id).voices.append(VoiceSample(
            t0=vr.t0, t1=vr.t1, embedding=emb,
            q=min(1.0, dur / _VOICE_FULL_Q_SEC), wav_bytes=wav_sample))
    return ev
