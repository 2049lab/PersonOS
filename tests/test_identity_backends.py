"""V1 backend 骨架:mock profile 冒烟(不下模型、不联网)。"""

from __future__ import annotations

import numpy as np

from personos.identity.backends.base import FaceDet
from personos.identity.backends.factory import make_backends


def test_mock_factory_has_three_backends():
    b = make_backends("mock")
    assert set(b) == {"mm_runner", "face_detector", "voiceprint"}


def test_mock_face_detect_returns_normalized_emb():
    det = make_backends("mock")["face_detector"]
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[10:20, 10:20] = 200
    out = det.detect(frame)
    assert len(out) == 1 and isinstance(out[0], FaceDet)
    assert out[0].embedding.shape == (512,)
    assert abs(float(np.linalg.norm(out[0].embedding)) - 1.0) < 1e-5
    # 确定性:同帧同向量
    assert np.allclose(out[0].embedding, det.detect(frame)[0].embedding)


def test_mock_face_detect_empty_frame():
    det = make_backends("mock")["face_detector"]
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_mock_voiceprint_embed():
    vp = make_backends("mock")["voiceprint"]
    emb = vp.embed(b"pcm-bytes")
    assert emb.shape == (192,) and abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    assert np.allclose(emb, vp.embed(b"pcm-bytes"))          # 确定
    assert not np.allclose(emb, vp.embed(b"other"))          # 不同输入不同向量


def test_mock_omni_chat_canned():
    omni = make_backends("mock")["mm_runner"]
    omni.canned = "CAST|P1|...\nEND"
    assert omni.chat("prompt", video_url="oss://x") == "CAST|P1|...\nEND"
    assert omni.calls[-1]["video_url"] == "oss://x"


def test_unknown_profile_raises():
    import pytest
    with pytest.raises(ValueError, match="mock/real"):
        make_backends("bogus")


def test_omni_video_call_uses_long_timeout(monkeypatch):
    """带视频的剧本调用必须走长超时,纯图/文调用仍用文本口径。

    回归的是一个真事故:视频剧本共用 MAAS_MLLM_TIMEOUT(120s),而 2min clip 的剧本调用
    实测就要顶穿它 → 读超时 → 重试 → 判毒消息 → **这条 clip 的记忆彻底丢**。
    """
    import httpx

    from personos.identity.backends import omni as omni_mod

    seen: list[float] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def post(self, url, **kw):
            seen.append(kw["timeout"].read)
            return _Resp()

    r = omni_mod.OmniRunner()
    r._client = _Client()
    monkeypatch.setattr(omni_mod, "VIDEO_TIMEOUT_S", 600.0)
    r.cfg = type("C", (), {"mllm_key": "k", "mllm_endpoint": "http://x",
                           "mllm_timeout": 120.0})()

    r.chat("p", video_url="https://oss/a.mp4")
    r.chat("p", images_b64=["Zg=="])                  # 仲裁:纯图,无需长超时
    assert seen == [600.0, 120.0], f"超时路由错了:{seen}"
    assert isinstance(httpx.Timeout(1.0).read, float)
