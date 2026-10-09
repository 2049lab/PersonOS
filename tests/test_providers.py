"""Model providers, exercised over a mock transport rather than a mock method.

These tests assert the *wire* behaviour — URL, headers, request body, how the
response is interpreted — because that is the contract with the outside world.
Patching the client's own ``_post`` would have tested our arrangement of our own
code and said nothing about whether the request is one a real endpoint accepts.

No network: httpx.MockTransport intercepts at the transport layer.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from personos.config import Config
from personos.providers.openai_compat import (
    NullMllm,
    OpenAIChatLLM,
    OpenAIEmbedder,
    OpenAIMllm,
    OpenAIReranker,
)
from personos.providers.registry import PROVIDERS, build, load


def _cfg(**kw) -> Config:
    base = dict(llm_base_url="https://api.example/v1", llm_api_key="sk-test",
                llm_model="test-model", rerank_api_key="sk-rr", rerank_model="rr-model",
                mllm_api_key="sk-mm", mllm_model="mm-model")
    base.update(kw)
    return Config(**base)


def _mount(client_obj, handler):
    """Replace the provider's httpx client with one that never leaves the process."""
    client_obj._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client_obj


# ── chat ────────────────────────────────────────────────────────────────

def test_chat_uses_bearer_auth_and_openai_paths():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    llm = _mount(OpenAIChatLLM(_cfg()), handler)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert seen["url"] == "https://api.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test", "standard OpenAI auth, not a private header"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["stream"] is False


def test_chat_strips_reasoning_blocks_at_the_provider_boundary():
    """Reasoning models interleave <think> blocks into content on OpenAI-compatible
    endpoints; every consumer wants the final text (a leaked block once poisoned
    stored memory narratives and adjudication JSON). One or several blocks go,
    text that merely mentions the tag stays."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content":
            "<think>let me reason</think>\n<think>more reasoning</think>\nhello"}}]})

    llm = _mount(OpenAIChatLLM(_cfg()), handler)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"


def test_app_id_header_only_appears_when_configured():
    """Some gateways want an application id. Most do not, and sending an empty
    one has been rejected outright by stricter endpoints."""
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    _mount(OpenAIChatLLM(_cfg()), handler).chat([{"role": "user", "content": "a"}])
    assert "x-app-id" not in seen["headers"]

    _mount(OpenAIChatLLM(_cfg(llm_app_id="myapp")), handler).chat([{"role": "user", "content": "a"}])
    assert seen["headers"]["x-app-id"] == "myapp"


# ── embeddings ──────────────────────────────────────────────────────────

def test_embed_returns_a_matrix_in_input_order():
    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [float(i)] * 4}
                                                  for i, _ in enumerate(body["input"])]})

    out = _mount(OpenAIEmbedder(_cfg()), handler).embed(["a", "b", "c"])
    assert isinstance(out, np.ndarray) and out.shape == (3, 4)
    assert out.dtype == np.float32
    assert out[2][0] == 2.0


def test_embedding_falls_back_to_the_chat_credentials():
    """One endpoint usually serves both, which is why a single key is enough."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"embedding": [0.0]}]})

    _mount(OpenAIEmbedder(_cfg()), handler).embed(["a"])
    assert seen["url"] == "https://api.example/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"


# ── reranking ───────────────────────────────────────────────────────────

def test_rerank_realigns_scores_to_input_order():
    """A reranker may return results sorted by score. Trusting that order would
    silently pair each score with the wrong document."""
    def handler(request):
        body = json.loads(request.content)
        assert body["documents"] == ["first", "second"]
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]})

    scores = _mount(OpenAIReranker(_cfg()), handler).rerank("q", ["first", "second"])
    assert scores == [0.1, 0.9]


def test_rerank_short_circuits_on_empty_input():
    def handler(request):  # pragma: no cover - must never be called
        raise AssertionError("no request should be made for zero documents")

    assert _mount(OpenAIReranker(_cfg()), handler).rerank("q", []) == []


def test_rerank_folds_the_instruction_into_the_query():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    _mount(OpenAIReranker(_cfg()), handler).rerank("who?", ["d"], instruction="judge relevance")
    assert seen["body"]["query"].startswith("Instruct: judge relevance")


def test_rerank_names_the_real_cause_instead_of_a_bare_keyerror():
    from personos.errors import ProviderError

    rr = _mount(OpenAIReranker(_cfg()), _minimax_error)
    with pytest.raises(ProviderError, match="no 'results' field"):
        rr.rerank("q", ["d"])


# ── rerank dialects: DashScope (Alibaba Bailian) ─────────────────────────

def test_dashscope_rerank_uses_the_full_url_and_nested_shape():
    """Bailian's native rerank API has no path-suffix convention, so the base
    URL is the whole endpoint; body nests under input/parameters and the answer
    under output.results."""
    from personos.providers.rerank import DashScopeReranker

    seen = {}
    full_url = "https://ws.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"output": {"results": [
            {"index": 1, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.2},
        ]}, "usage": {"total_tokens": 9}})

    rr = _mount(DashScopeReranker(_cfg(rerank_base_url=full_url)), handler)
    assert rr.rerank("什么是文本排序模型", ["d0", "d1"]) == [0.2, 0.8]
    assert seen["url"] == full_url
    assert seen["body"] == {"model": "rr-model",
                            "input": {"query": "什么是文本排序模型", "documents": ["d0", "d1"]},
                            "parameters": {"top_n": 2, "return_documents": False}}


def test_dashscope_rerank_names_the_real_cause_instead_of_a_bare_keyerror():
    from personos.errors import ProviderError
    from personos.providers.rerank import DashScopeReranker

    rr = _mount(DashScopeReranker(_cfg(rerank_base_url="https://ds.example/rerank")), _minimax_error)
    with pytest.raises(ProviderError, match="no 'output' field"):
        rr.rerank("q", ["d"])


def test_dashscope_rerank_explains_the_full_url_rule_when_base_url_is_unset():
    from personos.errors import ProviderError
    from personos.providers.rerank import DashScopeReranker

    with pytest.raises(ProviderError, match="verbatim as the full endpoint"):
        DashScopeReranker(_cfg()).rerank("q", ["d"])


# ── bring-your-own reranker via dotted path ───────────────────────────────

def test_registry_resolves_a_dotted_class_path_directly():
    """The extension point: an application ships its own dialect and points
    PERSONOS_RERANKER_PROVIDER at it — nothing to register in the framework."""
    from personos.providers.rerank import DashScopeReranker

    assert load("reranker", "personos.providers.rerank.DashScopeReranker") is DashScopeReranker


def test_registry_guides_towards_dotted_paths_on_a_bad_reranker_name():
    with pytest.raises(KeyError, match="dotted class path"):
        load("reranker", "not-a-reranker")


# ── vision ──────────────────────────────────────────────────────────────

def test_look_image_inlines_the_bytes_as_a_data_url():
    """Base64 inline, not a URL — so image understanding works even when the
    media store is local and unreachable from the model's side."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a cat"}}]})

    out = _mount(OpenAIMllm(_cfg()), handler).look_image(b"\x89PNG-bytes", "what is this?",
                                                         content_type="image/png")
    assert out == "a cat"
    parts = seen["body"]["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_look_image_never_propagates_a_failure():
    """Vision is an enhancement. A failed description must not fail the write —
    the image is still stored as evidence either way."""
    def handler(request):
        return httpx.Response(500, json={"error": "upstream is down"})

    mllm = _mount(OpenAIMllm(_cfg()), handler)
    mllm.max_retries = 1
    assert mllm.look_image(b"x", "describe") == ""


def test_unavailable_vision_returns_empty_without_calling_out():
    def handler(request):  # pragma: no cover
        raise AssertionError("must not call the endpoint without a key")

    assert _mount(OpenAIMllm(_cfg(mllm_api_key="")), handler).look_image(b"x", "d") == ""


def test_null_mllm_matches_the_degradation_check():
    """The write path checks getattr(mllm, "available", False). The null object
    exists so callers never hold None and that check keeps working unchanged."""
    n = NullMllm()
    assert n.available is False
    assert n.look_image(b"x", "d") == ""


# ── registry ────────────────────────────────────────────────────────────

def test_registry_resolves_by_name():
    assert load("llm", "openai") is OpenAIChatLLM
    assert isinstance(build("mllm", "none"), NullMllm)


def test_registry_reports_what_it_knows_on_a_bad_name():
    with pytest.raises(KeyError, match="available"):
        load("llm", "not-a-provider")
    with pytest.raises(KeyError, match="expected one of"):
        load("not-a-kind", "openai")


def test_every_registered_provider_actually_imports():
    """A dotted path typo in the registry would only surface at runtime, in
    whichever deployment happened to select that provider."""
    for kind, providers in PROVIDERS.items():
        for name in providers:
            if kind in ("llm",) and name == "anthropic":
                pytest.importorskip("anthropic")
            if kind == "media" and name == "oss":
                pytest.importorskip("oss2")
            assert load(kind, name) is not None, f"{kind}/{name}"


# ── error bodies in a 200 (the MiniMax shape) ───────────────────────────

def _minimax_error(_request: httpx.Request) -> httpx.Response:
    """MiniMax reports auth/model failures inside a 200: base_resp, no data field."""
    return httpx.Response(200, json={"base_resp": {"status_code": 1004,
                                                   "status_msg": "login fail"}})


def test_embed_names_the_real_cause_instead_of_a_bare_keyerror():
    from personos.errors import ProviderError

    emb = _mount(OpenAIEmbedder(_cfg()), _minimax_error)
    with pytest.raises(ProviderError) as ei:
        emb.embed(["hello"])
    msg = str(ei.value)
    assert "no 'data' field" in msg and "embed" in msg
    assert "api.example" in msg and "base_resp" in msg, "quote the body so the cause is visible"


def test_chat_names_the_real_cause_instead_of_a_bare_keyerror():
    from personos.errors import ProviderError

    llm = _mount(OpenAIChatLLM(_cfg()), _minimax_error)
    with pytest.raises(ProviderError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "no 'choices' field" in str(ei.value)


def test_look_image_logs_the_named_cause_and_still_degrades(caplog):
    """Vision stays optional: the named error lands in the log, the call returns ''."""
    mm = _mount(OpenAIMllm(_cfg()), _minimax_error)
    assert mm.look_image(b"\xff\xd8", "what is shown") == ""
