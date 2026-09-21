"""Entry validation for video messages on /ingest (validation only: no DB writes, no consumer
or model calls).

The key regression: when one message carries both a video and text/image, it **must return an
explicit 400**. An early implementation took only the video field and silently dropped the text,
and silently losing data is the worst class of bug.

No real user is registered (that would write to the DB and be blocked by the test guard); the
_ctx dependency is overridden with a fake context instead. The 400 path has no side effects and
the 202 path only enqueues into Redis, whose keys are cleaned up in teardown.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.runtime import UserContext, rt
from server.app import app
from server.api import _ctx
from server.signing import verify_signature

U = "vtest_ingest_valid"
_SIDS = ("valid", "vurl", "vkey", "vbad", "vempty", "vtext", "vbig")


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[verify_signature] = lambda: None      # bypass signature check
    app.dependency_overrides[_ctx] = lambda: UserContext(U, rt.db)  # fake context, no DB writes
    yield TestClient(app)
    app.dependency_overrides.pop(_ctx, None)
    mq = rt.msg_queue()                                             # clear queue keys left by the 202 path
    for sid in _SIDS:
        for kf in (getattr(mq, "_mk", None), getattr(mq, "_pk", None),
                   getattr(mq, "_sk", None), getattr(mq, "_ck", None)):
            if kf is None:
                continue
            try:
                mq._c.delete(kf(U, sid))
            except Exception:  # noqa: BLE001
                pass


def _post(c, msgs, sid="valid"):
    return c.post("/api/v1/ingest", headers={"X-User-Token": "x"},
                  json={"session_id": sid, "messages": msgs})


def test_same_message_video_plus_text_rejected(client):
    """One message carrying both video and text -> 400 (never silently drop the text)."""
    r = _post(client, [{"speaker": "user", "text": "看这个", "video_url": "https://x/c.mp4"}])
    assert r.status_code == 400 and "text/image_b64" in r.text


def test_same_message_video_plus_image_rejected(client):
    r = _post(client, [{"speaker": "user", "image_b64": "aGk=", "video_oss_key": "k.mp4"}])
    assert r.status_code == 400


def test_mixed_batch_video_and_text_rejected(client):
    """A batch mixing video messages with text messages -> 400 (callers must split the batch)."""
    r = _post(client, [{"speaker": "user", "text": "先说句话"},
                       {"speaker": "user", "video_url": "https://x/c.mp4"}])
    assert r.status_code == 400


def test_video_url_accepted(client):
    """video_url (a URL in the caller's own bucket is fine too) -> 202 and enqueued."""
    r = _post(client, [{"speaker": "user", "video_url": "https://caller-bucket/a.mp4",
                        "clip_index": 0}], sid="vurl")
    assert r.status_code == 202


def test_video_oss_key_still_accepted(client):
    """The shortcut path for objects already in our own object storage stays supported."""
    r = _post(client, [{"speaker": "user", "video_oss_key": "personos/mem/x/clip/a.mp4"}], sid="vkey")
    assert r.status_code == 202


def test_non_http_video_url_rejected(client):
    r = _post(client, [{"speaker": "user", "video_url": "file:///etc/passwd"}], sid="vbad")
    assert r.status_code == 400


def test_empty_message_rejected(client):
    """None of the three content kinds present -> 400."""
    r = _post(client, [{"speaker": "user"}], sid="vempty")
    assert r.status_code == 400


def test_plain_text_unaffected(client):
    """Backward compatible: existing plain-text calls are unaffected."""
    r = _post(client, [{"speaker": "user", "text": "普通一句"}], sid="vtext")
    assert r.status_code == 202


def test_caller_duration_over_limit_rejected(client):
    """A caller-declared duration above the limit -> 400 right at the entry point (a cheap check)."""
    from server.api import _MAX_CLIP_DURATION_S
    r = _post(client, [{"speaker": "user", "video_url": "https://x/a.mp4",
                        "duration_sec": _MAX_CLIP_DURATION_S + 1}], sid="vlong")
    assert r.status_code == 400 and "video duration" in r.text


def test_unreachable_url_rejected_at_ingest(client, monkeypatch):
    """An unreachable external URL (4xx/5xx) -> the entry-point HEAD precheck returns 400 on the
    spot, instead of handing the caller a 202 and then failing quietly."""
    import server.api as api

    monkeypatch.setattr(api, "_precheck_video_url", lambda url: "not reachable (HTTP 403)")
    r = _post(client, [{"speaker": "user", "video_url": "https://expired/a.mp4"}], sid="vdead")
    assert r.status_code == 400 and "not reachable" in r.text


def test_precheck_passes_on_network_flake(monkeypatch):
    """The precheck itself flakes, or the peer forbids HEAD -> let it through and leave the
    verdict to the real download on the consumer side, so valid data is not rejected."""
    import httpx

    import server.api as api

    def _boom(*a, **k):
        raise httpx.ConnectTimeout("flaky")

    monkeypatch.setattr(httpx, "head", _boom)
    assert api._precheck_video_url("https://x/a.mp4") == ""


def test_oversized_clip_rejected_at_ingest_both_paths(client, monkeypatch):
    """A clip larger than the upstream fetch limit -> 400 at the entry point, on both paths.

    This regresses a real incident: a 2min@7.3Mbps (106MB) clip was accepted with a 202, and only
    minutes later did the screenplay model report `Download multimodal file timed out` -- by then
    the caller was long gone and the only recourse was digging through logs afterwards. The size is
    knowable at the entry point (Content-Range for an external URL, an object-size lookup for our
    own key), so there is no reason to defer the discovery to the consumer side.
    """
    from server import api as service_api

    big = service_api._MAX_CLIP_UPSTREAM_BYTES + 1

    # (1) External URL: the total length reported by the precheck is over the limit
    monkeypatch.setattr(service_api, "_precheck_video_url",
                        lambda url: service_api._too_big_for_upstream(big))
    r = client.post("/api/v1/ingest", json={
        "session_id": "vbig", "messages": [
            {"speaker": "user", "video_url": "https://caller/huge.mp4"}]})
    assert r.status_code == 400, r.text
    assert "lower the bitrate" in r.text, r.text

    # (2) Our own key: the object size returned by object storage is over the limit
    monkeypatch.setattr(service_api.rt, "_media",
                        lambda: type("M", (), {"object_size": staticmethod(lambda k: big)})())
    r = client.post("/api/v1/ingest", json={
        "session_id": "vbig", "messages": [
            {"speaker": "user", "video_oss_key": "our/huge.mp4"}]})
    assert r.status_code == 400, r.text
    assert "lower the bitrate" in r.text, r.text


def test_oss_size_lookup_failure_does_not_block_ingest(client, monkeypatch):
    """If the object size cannot be read (missing key, or object storage flaking) the request must
    be **let through** and judged on the consumer side -- accept rather than reject. Entry
    validation is a cheap safety net; one flaky storage lookup must not bounce a valid call."""
    from server import api as service_api

    def _boom():
        raise RuntimeError("oss down")

    monkeypatch.setattr(service_api.rt, "_media",
                        lambda: type("M", (), {"object_size": staticmethod(
                            lambda k: (_ for _ in ()).throw(RuntimeError("oss down")))})())
    r = client.post("/api/v1/ingest", json={
        "session_id": "vkey", "messages": [
            {"speaker": "user", "video_oss_key": "our/unknown.mp4"}]})
    assert r.status_code == 202, r.text


# -- Image input on /recall (shares _decode_image_b64 with /ingest, so the contract must match) --

def test_recall_rejects_bad_image_base64(client):
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": "!!!不是base64!!!"})
    assert r.status_code == 400 and "base64" in r.text


def test_recall_rejects_oversized_image(client):
    import base64

    from server import api as service_api
    big = base64.b64encode(b"x" * (service_api._MAX_IMAGE_BYTES + 1)).decode()
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": big})
    assert r.status_code == 413, r.text


def test_recall_image_validation_runs_before_anything_expensive(client, monkeypatch):
    """A bad image must be rejected **before** entering the recall path -- do not take a
    concurrency slot and assemble heavy models only to then discover the image is broken."""
    from server import api as service_api

    def _boom(*a, **k):
        raise AssertionError("a bad image must never reach run_recall")

    monkeypatch.setattr(service_api, "run_recall", _boom)
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": "@@@"})
    assert r.status_code == 400


# -- Response echo and format validation for /recall with an image --

def test_recall_rejects_unrecognizable_image(client):
    """Valid base64 that is not an image -> 400.

    This used to degrade silently: the caller got a 200 plus an answer that quietly ignored the
    image, with no way to notice. A query image is not like an ingest image -- it is the **query
    input**, so if its format cannot be recognized there is no visual understanding to speak of.
    """
    import base64
    bad = base64.b64encode(b"this is definitely not an image" * 4).decode()
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": bad})
    assert r.status_code == 400 and "not a recognizable image" in r.text, r.text


def test_recall_accepts_real_png(client, monkeypatch):
    """A real PNG passes format validation (do not reject legitimate calls)."""
    import base64
    import io

    from PIL import Image

    from server import api as service_api
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")

    seen = {}

    def _fake(*a, **k):
        seen["image"] = k.get("image")
        raise RuntimeError("far enough: format validation already passed")

    monkeypatch.setattr(service_api, "run_recall", _fake)
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?",
        "image_b64": base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code != 400, r.text
    assert seen.get("image"), "the image should be decoded and passed through to the recall path"
