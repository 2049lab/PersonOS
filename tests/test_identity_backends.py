"""V1 backend skeleton: a smoke test of the mock profile (no model download, no network)."""

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
    # Deterministic: the same frame yields the same vector.
    assert np.allclose(out[0].embedding, det.detect(frame)[0].embedding)


def test_mock_face_detect_empty_frame():
    det = make_backends("mock")["face_detector"]
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_mock_voiceprint_embed():
    vp = make_backends("mock")["voiceprint"]
    emb = vp.embed(b"pcm-bytes")
    assert emb.shape == (192,) and abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    assert np.allclose(emb, vp.embed(b"pcm-bytes"))          # deterministic
    assert not np.allclose(emb, vp.embed(b"other"))          # different input, different vector


def test_mock_omni_chat_canned():
    omni = make_backends("mock")["mm_runner"]
    omni.canned = "CAST|P1|...\nEND"
    assert omni.chat("prompt", video_url="oss://x") == "CAST|P1|...\nEND"
    assert omni.calls[-1]["video_url"] == "oss://x"


def test_unknown_profile_raises():
    import pytest
    with pytest.raises(ValueError, match="none/real/mock"):
        make_backends("bogus")


def test_omni_video_call_uses_long_timeout(monkeypatch):
    """A screenplay call carrying video must use the long timeout; image-only and text-only
    calls keep using the text timeout.

    This is a regression test for a real incident: video screenplay calls shared the multimodal
    text timeout (120s), but a screenplay call over a 2-minute clip was measured blowing right
    through it -> read timeout -> retry -> the message got flagged as poisonous -> the memory
    for that entire clip was lost.
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
    r.cfg = type("C", (), {"mllm_api_key": "k", "mllm_endpoint": "http://x",
                           "mllm_timeout": 120.0})()

    r.chat("p", video_url="https://oss/a.mp4")
    r.chat("p", images_b64=["Zg=="])                  # adjudication: images only, no long timeout needed
    assert seen == [600.0, 120.0], f"timeout routing is wrong: {seen}"
    assert isinstance(httpx.Timeout(1.0).read, float)


def test_none_profile_is_the_default_and_explains_how_to_enable(monkeypatch):
    """The default is none: when someone touches video without installing the dependencies,
    give them one actionable sentence instead of an ImportError buried inside a worker thread.

    This default was changed DELIBERATELY (it used to be real). The reasoning is in the factory
    module docstring: real is right for an internal deployment, where a missing dependency
    should blow up immediately, but wrong for a pip-installed library — the identity extra is
    roughly 2GB, and text-only users should neither pay for it nor find out about it via an
    import explosion.
    """
    import pytest

    from personos.config import Config, reset_config, set_config
    from personos.identity.backends.factory import make_backends

    set_config(Config())          # all defaults = video_backend 'none'
    try:
        with pytest.raises(RuntimeError, match=r"personos\[identity\]"):
            make_backends()
    finally:
        reset_config()
