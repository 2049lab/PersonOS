"""Rules tests for the uniform public response envelope {code, data, msg}.

Two layers:
1. The pure function layer (response.py): ResponseUtils.ok/error plus the success / error /
   not-an-envelope / already-wrapped branches of _wrap_response.
2. End to end (TestClient against the real runtime): every endpoint responds with
   {code, data, msg}, where success is code=0, failure sets code to the HTTP status number,
   and msg is complete. The platform probe /healthz is not wrapped in the envelope. Any
   temporary user registered here is deleted immediately afterwards, so the shared database
   keeps no residue.
"""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse

from server.response import ResponseUtils, _wrap_response

# -- The pure function layer --

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
    assert wrapped.headers.get("Retry-After") == "1"          # rate-limit headers must pass through


def test_wrap_skips_already_enveloped():
    src = JSONResponse(status_code=200, content={"code": 0, "data": 1, "msg": "ok"})
    assert _wrap_response(src) is src                          # never wrapped twice


# -- End to end (read-only: registering would write to the database and the test guard blocks
# that, so auth is instead injected as a read-only context via dependency_overrides) --

def _assert_envelope(body):
    assert set(body.keys()) == {"code", "data", "msg"}
    assert isinstance(body["code"], int) and isinstance(body["msg"], str)


@pytest.fixture()
def client():
    """The real app with AK/SK signature verification overridden (production enforces it
    unconditionally; tests bypass it via dependency_overrides)."""
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
    """The real app with _ctx overridden to inject a read-only UserContext (no database writes,
    which sidesteps the register guard) and signature verification bypassed."""
    from fastapi.testclient import TestClient

    from server import api as service_api
    from server.app import app
    from server.runtime import rt
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
    assert "users" in body["data"]                            # the original payload lands under data


def test_401_without_token(client):
    r = client.post("/api/v1/recall", json={"session_id": "s", "query": "q"})
    body = r.json()
    _assert_envelope(body)
    assert r.status_code == 401 and body["code"] == 401 and body["data"] is None
    assert body["msg"]                                        # msg is non-empty and complete


def test_healthz_probe_not_enveloped(client):
    r = client.get("/healthz")
    # The probe is returned verbatim, not wrapped in the envelope.
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_api_doc_page_served(client):
    """/ and /api-doc serve the API documentation as HTML, so a consumer or an agent can just
    hit the domain and read it."""
    for path in ("/", "/api-doc"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        # The title and a copy-pasteable curl are both in place.
        assert "personos-mem" in r.text and "curl -s" in r.text


def test_422_validation_envelope(auth_client):
    r = auth_client.post("/api/v1/ingest", json={"session_id": "s"})   # messages is missing
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
    assert "exists" in body["data"]                           # data carries the business body (the exists field)
