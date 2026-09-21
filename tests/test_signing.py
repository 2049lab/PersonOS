"""AK/SK request signing (signing.py): sign-verify round trip / tampering / expiry / replay /
unknown AK / missing headers / the guard being a no-op.

The pure _verify logic is tested offline by injecting the AK-to-SK cache and forcing the
in-memory nonce store, so neither Redis nor the key management service is contacted.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from server import signing

AK, SK = "rokid", "testsk-abc"


@pytest.fixture(autouse=True)
def _iso(monkeypatch):
    """For every case: inject the AK-to-SK cache, force the in-memory nonce store (no Redis),
    and clear the cache afterwards."""
    signing._aksk_cache = {AK: SK}
    signing._mem_nonce.clear()
    monkeypatch.setattr(signing, "settings", types.SimpleNamespace(redis_url=""))
    yield
    signing._reset_cache()


def _signed(method, path, *, query_items=None, headers=None, body=b""):
    """Sign from the caller's point of view, then merge into the (lowercased) headers the
    server would actually receive."""
    sh = signing.sign_request(AK, SK, method, path, query_items=query_items, headers=headers, body=body)
    merged = {**(headers or {}), **sh}
    return {k.lower(): v for k, v in merged.items()}


def test_roundtrip_ok():
    body = b'{"session_id":"s1"}'
    h = _signed("POST", "/api/v1/ingest",
                headers={"content-type": "application/json", "x-user-token": "tok"}, body=body)
    assert signing._verify("POST", "/api/v1/ingest", [], h, body) == AK


def test_tampered_body_rejected():
    body = b'{"n":1}'
    h = _signed("POST", "/api/v1/ingest", headers={"content-type": "application/json"}, body=body)
    with pytest.raises(signing._SigError, match="signature mismatch"):
        signing._verify("POST", "/api/v1/ingest", [], h, b'{"n":2}')   # the body was altered


def test_tampered_query_rejected():
    h = _signed("GET", "/api/v1/episodes", query_items=[("page", "1")],
                headers={"x-user-token": "tok"})
    with pytest.raises(signing._SigError, match="signature mismatch"):
        signing._verify("GET", "/api/v1/episodes", [("page", "2")], h, b"")   # the query was altered


def test_expired_timestamp_rejected():
    body = b""
    old = str(int(time.time()) - signing._WINDOW - 10)
    nonce = "n1"
    canon = signing.canonical_string("GET", "/api/v1/health", [], {}, body, old, nonce)
    h = {"x-access-key": AK, "x-timestamp": old, "x-nonce": nonce,
         "x-signature": signing.sign(SK, canon), "x-signed-headers": ""}
    with pytest.raises(signing._SigError, match="outside the allowed window"):
        signing._verify("GET", "/api/v1/health", [], h, body)


def test_replay_nonce_rejected():
    body = b""
    h = _signed("GET", "/api/v1/health", headers={"x-user-token": "tok"}, body=body)
    assert signing._verify("GET", "/api/v1/health", [], h, body) == AK   # the first time passes
    with pytest.raises(signing._SigError, match="replayed nonce"):
        signing._verify("GET", "/api/v1/health", [], h, body)            # the same nonce again is rejected


def test_unknown_ak_rejected():
    body = b""
    h = _signed("GET", "/api/v1/health", headers={"x-user-token": "tok"}, body=body)
    h["x-access-key"] = "ghost"                                          # swap in an unregistered AK
    with pytest.raises(signing._SigError, match="unknown or disabled AK"):
        signing._verify("GET", "/api/v1/health", [], h, body)


def test_missing_headers_rejected():
    with pytest.raises(signing._SigError, match="missing signature headers"):
        signing._verify("GET", "/api/v1/health", [], {"x-access-key": AK}, b"")


def test_stripped_signed_header_rejected():
    """A request that carries x-user-token but did not include it in the signature is rejected,
    which is what stops an attacker stripping security headers."""
    body = b""
    # Done by hand: x-user-token is left out of the signature but present in the request headers.
    ts, nonce = str(int(time.time())), "n2"
    canon = signing.canonical_string("GET", "/api/v1/recall", [], {}, body, ts, nonce)
    h = {"x-access-key": AK, "x-timestamp": ts, "x-nonce": nonce,
         "x-signature": signing.sign(SK, canon), "x-signed-headers": "",
         "x-user-token": "tok"}
    with pytest.raises(signing._SigError, match="x-user-token"):
        signing._verify("GET", "/api/v1/recall", [], h, body)


def test_aksk_empty_not_cached_selfheals(monkeypatch):
    """An empty map on the first read is not cached, so once the credentials are in place the
    next call loads them automatically with no restart, and subsequent calls hit the cache."""
    signing._reset_cache()
    seq = iter(["", '{"rokid":"sk1"}', "SHOULD-NOT-BE-READ"])
    monkeypatch.setattr(signing, "get_secret", lambda *a, **k: next(seq))
    assert signing._load_aksk() == {}                    # empty -> not cached
    assert signing._load_aksk() == {"rokid": "sk1"}      # now present -> self-healing load
    assert signing._load_aksk() == {"rokid": "sk1"}      # already cached -> get_secret is not read again


def _req(method, path, headers, body=b""):
    from starlette.requests import Request
    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    async def receive(): return {"type": "http.request", "body": body, "more_body": False}
    return Request(scope, receive)


def test_dependency_rejects_unsigned():
    """Unconditionally enforced: an unsigned request raises 401, with no environment-variable
    bypass any more."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(signing.verify_signature(_req("POST", "/api/v1/recall",
                                                  {"content-type": "application/json"}, b"{}")))
    assert ei.value.status_code == 401


def test_dependency_passes_signed():
    """A correct signature is let through and caller_ak is written into request.state."""
    body = b'{"q":1}'
    hdrs = {"content-type": "application/json"}
    sh = signing.sign_request(AK, SK, "POST", "/api/v1/recall", headers=hdrs, body=body)
    req = _req("POST", "/api/v1/recall", {**hdrs, **sh}, body)
    assert asyncio.run(signing.verify_signature(req)) is None
    assert req.state.caller_ak == AK
