"""MaasClient._post 限流退避单测:429 短退避(2 次尝试,带抖动),其余 4xx 终态。

背景:429 曾走独立 6 次深预算(退避 2/4/8/16/30s 共 62s),并发下所有请求同步挂等、
醒来一齐重发再撞限流,worker 被睡满 = 重试地狱;2026-09-08 收紧为 2 次尝试 +
短退避抖动,耗尽即抛(API 层透传 429)。深预算仍可构造处显式传参(bench 用)。
只测退避编排,不发真实网络请求。
"""

from __future__ import annotations

import httpx
import pytest

from personos.clients.maas import MaasClient


class _Cfg:  # _post 只读 maas_base_url,测试替身免真实 .env
    maas_base_url = "https://maas.example"


def _client(handler) -> MaasClient:
    c = MaasClient(cfg=_Cfg(), max_retries=3)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_429_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("personos.clients.maas.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"msg": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    assert _client(handler)._post("/score", {}, {}, timeout=1.0) == {"ok": True}
    assert calls["n"] == 2                            # 1 次 429 + 1 次成功(默认预算内)


def test_429_exhausted_reraises(monkeypatch):
    monkeypatch.setattr("personos.clients.maas.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"msg": "rate limited"})

    c = MaasClient(cfg=_Cfg(), max_retries=3, rate_limit_attempts=3)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        c._post("/score", {}, {}, timeout=1.0)
    assert calls["n"] == 3                            # rate_limit_attempts=3 次耗尽,原样抛(上层降级)


def test_429_default_budget_two_attempts(monkeypatch):
    """默认 429 预算 = 2 次尝试(1s 基数带抖动)即抛——并发下快速失败,由 API 层透传 429。"""
    calls = {"n": 0}
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"msg": "rate limited"})

    monkeypatch.setattr("personos.clients.maas.time.sleep", waits.append)
    monkeypatch.setattr("personos.clients.maas.random.uniform", lambda a, b: 1.0)   # 抖动固定 ×1
    c = MaasClient(cfg=_Cfg(), max_retries=3)         # rate_limit_attempts 用默认
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        c._post("/score", {}, {}, timeout=1.0)
    assert calls["n"] == 2                            # 第 2 次 429 即抛(语义同 max_retries)
    assert waits == [1.0]                             # 1s 基数 × 抖动 1.0


def test_other_4xx_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"msg": "bad request"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler)._post("/embeddings", {}, {}, timeout=1.0)
    assert calls["n"] == 1                            # 400 仍是终态,一次都不重试
