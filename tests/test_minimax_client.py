"""MinimaxClient 429 长退避单测:SDK 内建重试耗尽后的限流在客户端层再兜一层(S4-H1)。

只测退避编排(几次失败/重试几次/非限流异常不重试),不发真实网络请求。
"""

from __future__ import annotations

import httpx
import anthropic
import pytest

from personos.clients.minimax import MinimaxClient


class _Cfg:  # 客户端只读三个字段,测试替身免真实 .env
    minimax_api_key = "test-key"
    minimax_base_url = "https://api.minimax.example"
    minimax_chat_model = "test-model"


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _TextResp:  # anthropic 响应替身:chat 只读 content 里的 text 块
    def __init__(self, text):
        self.content = [_Block(text)]


def _rate_limit_err() -> anthropic.RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"),
                          json={"type": "error"})
    return anthropic.RateLimitError("Error code: 429 - plan limit", response=resp, body=None)


class _FlakyMessages:
    """create 前 n_fail 次抛 429 后成功;n_fail<0 永远 429。"""

    def __init__(self, n_fail):
        self.n_fail, self.calls = n_fail, 0

    def create(self, **kw):
        self.calls += 1
        if self.n_fail < 0 or self.calls <= self.n_fail:
            raise _rate_limit_err()
        return _TextResp("ok")


def _client(flaky) -> MinimaxClient:
    c = MinimaxClient(cfg=_Cfg(), rate_limit_retries=3, rate_limit_base_sleep=0.001)
    c._client.messages = flaky
    return c


def test_rate_limit_retried_then_succeeds():
    flaky = _FlakyMessages(2)
    assert _client(flaky).chat([{"role": "user", "content": "hi"}]) == "ok"
    assert flaky.calls == 3                       # 2 次失败 + 1 次成功


def test_rate_limit_exhausted_reraises():
    flaky = _FlakyMessages(-1)
    with pytest.raises(anthropic.RateLimitError):
        _client(flaky).chat([{"role": "user", "content": "hi"}])
    assert flaky.calls == 4                       # 首调 + 3 次长退避,仍失败原样抛


def test_non_rate_limit_error_not_retried():
    flaky = _FlakyMessages(0)

    def boom(**kw):
        flaky.calls += 1
        raise RuntimeError("connection reset")

    flaky.create = boom
    with pytest.raises(RuntimeError):
        _client(flaky).chat([{"role": "user", "content": "hi"}])
    assert flaky.calls == 1                       # 非限流异常不进长退避
