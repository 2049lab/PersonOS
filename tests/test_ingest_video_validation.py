"""/ingest 视频消息的入口校验(纯校验,不写 DB、不触消费/模型)。

重点回归:同一条消息既带视频又带 text/image 时,**必须明确 400**——早期实现会只取视频字段、
把文本静默丢弃(静默丢数据是最糟的一类 bug)。

不注册真 user(会写 DB 被测试护栏拦):直接覆写 _ctx 依赖造假上下文;400 路径无副作用,
202 路径只进 Redis 队列,teardown 清键。
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
    app.dependency_overrides[verify_signature] = lambda: None      # 绕验签
    app.dependency_overrides[_ctx] = lambda: UserContext(U, rt.db)  # 假上下文,不写 DB
    yield TestClient(app)
    app.dependency_overrides.pop(_ctx, None)
    mq = rt.msg_queue()                                             # 清 202 路径留下的队列键
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
    """同一条消息同时带视频与文本 → 400(绝不静默丢文本)。"""
    r = _post(client, [{"speaker": "user", "text": "看这个", "video_url": "https://x/c.mp4"}])
    assert r.status_code == 400 and ("同时带视频" in r.text or "text/image_b64" in r.text)


def test_same_message_video_plus_image_rejected(client):
    r = _post(client, [{"speaker": "user", "image_b64": "aGk=", "video_oss_key": "k.mp4"}])
    assert r.status_code == 400


def test_mixed_batch_video_and_text_rejected(client):
    """一批里视频消息 + 文本消息混排 → 400(要求分批)。"""
    r = _post(client, [{"speaker": "user", "text": "先说句话"},
                       {"speaker": "user", "video_url": "https://x/c.mp4"}])
    assert r.status_code == 400


def test_video_url_accepted(client):
    """video_url(调用方自己桶的地址亦可)→ 202 入队。"""
    r = _post(client, [{"speaker": "user", "video_url": "https://caller-bucket/a.mp4",
                        "clip_index": 0}], sid="vurl")
    assert r.status_code == 202


def test_video_oss_key_still_accepted(client):
    """已在我方 OSS 的快捷路径仍兼容。"""
    r = _post(client, [{"speaker": "user", "video_oss_key": "personos/mem/x/clip/a.mp4"}], sid="vkey")
    assert r.status_code == 202


def test_non_http_video_url_rejected(client):
    r = _post(client, [{"speaker": "user", "video_url": "file:///etc/passwd"}], sid="vbad")
    assert r.status_code == 400


def test_empty_message_rejected(client):
    """三种内容都没有 → 400。"""
    r = _post(client, [{"speaker": "user"}], sid="vempty")
    assert r.status_code == 400


def test_plain_text_unaffected(client):
    """向前兼容:老的纯文本调用不受影响。"""
    r = _post(client, [{"speaker": "user", "text": "普通一句"}], sid="vtext")
    assert r.status_code == 202


def test_caller_duration_over_limit_rejected(client):
    """调用方声明的时长超上限 → 入口就 400(便宜的一道)。"""
    from server.api import _MAX_CLIP_DURATION_S
    r = _post(client, [{"speaker": "user", "video_url": "https://x/a.mp4",
                        "duration_sec": _MAX_CLIP_DURATION_S + 1}], sid="vlong")
    assert r.status_code == 400 and "时长" in r.text


def test_unreachable_url_rejected_at_ingest(client, monkeypatch):
    """外链不可达(4xx/5xx)→ 入口 HEAD 预检当场 400,不让调用方收了 202 才悄悄失败。"""
    import server.api as api

    monkeypatch.setattr(api, "_precheck_video_url", lambda url: "不可访问(HTTP 403)")
    r = _post(client, [{"speaker": "user", "video_url": "https://expired/a.mp4"}], sid="vdead")
    assert r.status_code == 400 and "不可访问" in r.text


def test_precheck_passes_on_network_flake(monkeypatch):
    """预检本身抖动/对端禁 HEAD → 放行(交给消费侧真下载判定),不误杀正常数据。"""
    import httpx

    import server.api as api

    def _boom(*a, **k):
        raise httpx.ConnectTimeout("flaky")

    monkeypatch.setattr(httpx, "head", _boom)
    assert api._precheck_video_url("https://x/a.mp4") == ""


def test_oversized_clip_rejected_at_ingest_both_paths(client, monkeypatch):
    """体积超「上游可拉取上限」→ 入口 400,两条路径都要拦。

    回归实测事故:2min@7.3Mbps(106MB)的 clip 被受理成 202,几分钟后剧本 MLLM 才报
    `Download multimodal file timed out`——调用方早就走了,只能靠留痕事后查。
    体积在入口就能知道(外链看 Content-Range,我方 key 问 OSS),没有理由拖到消费侧才发现。
    """
    from server import api as service_api

    big = service_api._MAX_CLIP_UPSTREAM_BYTES + 1

    # ① 外链:预检拿到的总长超限
    monkeypatch.setattr(service_api, "_precheck_video_url",
                        lambda url: service_api._too_big_for_upstream(big))
    r = client.post("/api/v1/ingest", json={
        "session_id": "vbig", "messages": [
            {"speaker": "user", "video_url": "https://caller/huge.mp4"}]})
    assert r.status_code == 400, r.text
    assert "降低码率" in r.text, r.text

    # ② 我方 key:问 OSS 拿到的对象大小超限
    monkeypatch.setattr(service_api.rt, "_media",
                        lambda: type("M", (), {"object_size": staticmethod(lambda k: big)})())
    r = client.post("/api/v1/ingest", json={
        "session_id": "vbig", "messages": [
            {"speaker": "user", "video_oss_key": "our/huge.mp4"}]})
    assert r.status_code == 400, r.text
    assert "降低码率" in r.text, r.text


def test_oss_size_lookup_failure_does_not_block_ingest(client, monkeypatch):
    """取不到对象大小(key 不存在 / OSS 抖动)必须**放行**,交消费侧判定——宁放勿杀。
    入口校验是快速兜底,不能因为一次 OSS 抖动就把正常调用打回去。"""
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


# —— /recall 的图片入参(与 /ingest 共用 _decode_image_b64,契约必须一致)——

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
    """坏图必须在进召回链路**之前**就拒掉 —— 别先占并发闸、再去装配重模型才发现图是坏的。"""
    from server import api as service_api

    def _boom(*a, **k):
        raise AssertionError("坏图不该走到 run_recall")

    monkeypatch.setattr(service_api, "run_recall", _boom)
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": "@@@"})
    assert r.status_code == 400


# —— /recall 带图的响应回显与格式校验 ——

def test_recall_rejects_unrecognizable_image(client):
    """合法 base64 但不是图片 → 400。

    此前是静默降级:调用方拿到 200 和一段"没用上图"的答案,完全无从察觉。
    查询图与 ingest 的图片不同——它是**查询输入**,认不出格式则整个视觉理解无从谈起。
    """
    import base64
    bad = base64.b64encode(b"this is definitely not an image" * 4).decode()
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?", "image_b64": bad})
    assert r.status_code == 400 and "图片" in r.text, r.text


def test_recall_accepts_real_png(client, monkeypatch):
    """真 PNG 能过格式校验(别把正常调用误杀)。"""
    import base64
    import io

    from PIL import Image

    from server import api as service_api
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")

    seen = {}

    def _fake(*a, **k):
        seen["image"] = k.get("image")
        raise RuntimeError("到这里就够了:格式校验已通过")

    monkeypatch.setattr(service_api, "run_recall", _fake)
    r = client.post("/api/v1/recall", json={
        "session_id": "vrimg", "query": "他是谁?",
        "image_b64": base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code != 400, r.text
    assert seen.get("image"), "图片应被解码后透传给召回链路"
