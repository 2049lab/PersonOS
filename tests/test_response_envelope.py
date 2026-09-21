"""对外响应统一信封 {code, data, msg} 的守则测试。

两层:
1. 纯函数层(response.py):ResponseUtils.ok/error + _wrap_response 的成功/错误/非信封/已包 分支。
2. 端到端(TestClient + 真 rt):各端点响应都是 {code,data,msg},成功 code=0、失败 code=HTTP 状态号、
   msg 完整;平台探针 /healthz 不套信封。用后即删注册的临时 user(共享 SIT 库不留残留)。
"""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse

from server.response import ResponseUtils, _wrap_response


# —— 纯函数层 ——

def test_ok_envelope():
    r = ResponseUtils.ok(data={"x": 1})
    body = json.loads(r.body)
    assert r.status_code == 200 and body == {"code": 0, "data": {"x": 1}, "msg": "ok"}


def test_error_envelope_code_mirrors_http():
    r = ResponseUtils.error(404, "not found")
    body = json.loads(r.body)
    assert r.status_code == 404 and body == {"code": 404, "data": None, "msg": "not found"}


def test_wrap_success_dict():
    wrapped = _wrap_response(JSONResponse(status_code=201, content={"token": "t"}))
    body = json.loads(wrapped.body)
    assert wrapped.status_code == 201 and body["code"] == 0 and body["data"] == {"token": "t"}


def test_wrap_error_extracts_msg_and_preserves_headers():
    src = JSONResponse(status_code=503, content={"error": "背压"}, headers={"Retry-After": "1"})
    wrapped = _wrap_response(src)
    body = json.loads(wrapped.body)
    assert wrapped.status_code == 503 and body == {"code": 503, "data": None, "msg": "背压"}
    assert wrapped.headers.get("Retry-After") == "1"          # 限流头必须透传


def test_wrap_skips_already_enveloped():
    src = JSONResponse(status_code=200, content={"code": 0, "data": 1, "msg": "ok"})
    assert _wrap_response(src) is src                          # 不二次包


# —— 端到端(只读:注册会写 DB,被测试护栏拦——鉴权改用 dependency_overrides 注入只读上下文)——

def _assert_envelope(body):
    assert set(body.keys()) == {"code", "data", "msg"}
    assert isinstance(body["code"], int) and isinstance(body["msg"], str)


@pytest.fixture()
def client():
    """真 app,覆写掉 AK/SK 验签(生产无条件强制,测试用 dependency_overrides 绕过)。"""
    from fastapi.testclient import TestClient
    from server.app import app
    from server.signing import verify_signature
    app.dependency_overrides[verify_signature] = lambda: None
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def auth_client():
    """真 app + 覆写 _ctx 注入只读 UserContext(不写 DB,绕过 register 护栏)+ 绕过验签。"""
    from fastapi.testclient import TestClient
    from server import api as service_api
    from server.runtime import rt
    from server.app import app
    from server.signing import verify_signature

    app.dependency_overrides[service_api._ctx] = lambda: rt.for_user("envtest_ro")
    app.dependency_overrides[verify_signature] = lambda: None
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_health_success_envelope(client):
    r = client.get("/api/v1/health")
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 200 and body["code"] == 0 and body["msg"] == "ok"
    assert "users" in body["data"]                            # 原 payload 落在 data 下


def test_401_without_token(client):
    r = client.post("/api/v1/recall", json={"session_id": "s", "query": "q"})
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 401 and body["code"] == 401 and body["data"] is None
    assert body["msg"]                                        # msg 非空、完整


def test_healthz_probe_not_enveloped(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}   # 探针原样,不套信封


def test_api_doc_page_served(client):
    """/ 与 /api-doc 提供接口文档 HTML(业务方/agent 直接访问域名即可阅读)。"""
    for path in ("/", "/api-doc"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "personos-mem" in r.text and "curl -s" in r.text   # 标题 + 可复制 curl 就位


def test_422_validation_envelope(auth_client):
    r = auth_client.post("/api/v1/ingest", json={"session_id": "s"})   # 缺 messages
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 422 and body["code"] == 422 and "messages" in body["msg"]


def test_400_bad_mode_envelope(auth_client):
    r = auth_client.post("/api/v1/recall", json={"session_id": "s", "query": "q", "mode": "wtf"})
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 400 and body["code"] == 400 and body["msg"]


def test_404_unknown_task_envelope(auth_client):
    r = auth_client.get("/api/v1/tasks/nope")
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 404 and body["code"] == 404


def test_profile_success_envelope(auth_client):
    r = auth_client.get("/api/v1/profile")
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 200 and body["code"] == 0
    assert "exists" in body["data"]                           # data 承载业务体(exists 字段)
