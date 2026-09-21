"""AK/SK 验签(signing.py):签验往返 / 篡改 / 过期 / 重放 / 未知AK / 缺头 / 守卫 no-op。

纯 _verify 逻辑离线测(注入 AK/SK 缓存 + 强制内存 nonce),不连 Redis/KMS。
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from personos.app import signing

AK, SK = "rokid", "testsk-abc"


@pytest.fixture(autouse=True)
def _iso(monkeypatch):
    """每个用例:注入 AK→SK 缓存,强制走内存 nonce(不连 Redis),用后清缓存。"""
    signing._aksk_cache = {AK: SK}
    signing._mem_nonce.clear()
    monkeypatch.setattr(signing, "settings", types.SimpleNamespace(redis_cluster=""))
    yield
    signing._reset_cache()


def _signed(method, path, *, query_items=None, headers=None, body=b""):
    """调用方视角签好,再合并成服务端拿到的(小写)头。"""
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
    with pytest.raises(signing._SigError, match="签名不匹配"):
        signing._verify("POST", "/api/v1/ingest", [], h, b'{"n":2}')   # body 被改


def test_tampered_query_rejected():
    h = _signed("GET", "/api/v1/episodes", query_items=[("page", "1")],
                headers={"x-user-token": "tok"})
    with pytest.raises(signing._SigError, match="签名不匹配"):
        signing._verify("GET", "/api/v1/episodes", [("page", "2")], h, b"")   # query 被改


def test_expired_timestamp_rejected():
    body = b""
    old = str(int(time.time()) - signing._WINDOW - 10)
    nonce = "n1"
    canon = signing.canonical_string("GET", "/api/v1/health", [], {}, body, old, nonce)
    h = {"x-access-key": AK, "x-timestamp": old, "x-nonce": nonce,
         "x-signature": signing.sign(SK, canon), "x-signed-headers": ""}
    with pytest.raises(signing._SigError, match="超窗"):
        signing._verify("GET", "/api/v1/health", [], h, body)


def test_replay_nonce_rejected():
    body = b""
    h = _signed("GET", "/api/v1/health", headers={"x-user-token": "tok"}, body=body)
    assert signing._verify("GET", "/api/v1/health", [], h, body) == AK   # 首次通过
    with pytest.raises(signing._SigError, match="重放"):
        signing._verify("GET", "/api/v1/health", [], h, body)            # 同 nonce 再来 → 拒


def test_unknown_ak_rejected():
    body = b""
    h = _signed("GET", "/api/v1/health", headers={"x-user-token": "tok"}, body=body)
    h["x-access-key"] = "ghost"                                          # 换成未登记 AK
    with pytest.raises(signing._SigError, match="未知"):
        signing._verify("GET", "/api/v1/health", [], h, body)


def test_missing_headers_rejected():
    with pytest.raises(signing._SigError, match="缺少签名头"):
        signing._verify("GET", "/api/v1/health", [], {"x-access-key": AK}, b"")


def test_stripped_signed_header_rejected():
    """请求带了 x-user-token 却没纳入签名 → 拒(防剥离安全头)。"""
    body = b""
    # 手工:签名时不含 x-user-token,但请求头里带了它
    ts, nonce = str(int(time.time())), "n2"
    canon = signing.canonical_string("GET", "/api/v1/recall", [], {}, body, ts, nonce)
    h = {"x-access-key": AK, "x-timestamp": ts, "x-nonce": nonce,
         "x-signature": signing.sign(SK, canon), "x-signed-headers": "",
         "x-user-token": "tok"}
    with pytest.raises(signing._SigError, match="x-user-token"):
        signing._verify("GET", "/api/v1/recall", [], h, body)


def test_aksk_empty_not_cached_selfheals(monkeypatch):
    """map 首次空不缓存;凭证补好后下次自动加载(无需重启),之后走缓存。"""
    signing._reset_cache()
    seq = iter(["", '{"rokid":"sk1"}', "SHOULD-NOT-BE-READ"])
    monkeypatch.setattr(signing, "get_secret", lambda *a, **k: next(seq))
    assert signing._load_aksk() == {}                    # 空 → 不缓存
    assert signing._load_aksk() == {"rokid": "sk1"}      # 补好 → 自愈加载
    assert signing._load_aksk() == {"rokid": "sk1"}      # 已缓存 → 不再读 get_secret


def _req(method, path, headers, body=b""):
    from starlette.requests import Request
    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
    async def receive(): return {"type": "http.request", "body": body, "more_body": False}
    return Request(scope, receive)


def test_dependency_rejects_unsigned():
    """无条件强制:未签名请求 → 抛 401(不再有 env 旁路)。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(signing.verify_signature(_req("POST", "/api/v1/recall",
                                                  {"content-type": "application/json"}, b"{}")))
    assert ei.value.status_code == 401


def test_dependency_passes_signed():
    """正确签名 → 放行并把 caller_ak 写入 request.state。"""
    body = b'{"q":1}'
    hdrs = {"content-type": "application/json"}
    sh = signing.sign_request(AK, SK, "POST", "/api/v1/recall", headers=hdrs, body=body)
    req = _req("POST", "/api/v1/recall", {**hdrs, **sh}, body)
    assert asyncio.run(signing.verify_signature(req)) is None
    assert req.state.caller_ak == AK
