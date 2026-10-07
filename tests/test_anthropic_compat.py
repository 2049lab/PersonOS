"""AnthropicChatLLM must work with both SDK generations: 0.x takes `temperature` as a keyword of
messages.create(), 1.x removed it, so there it has to travel via extra_body."""

from __future__ import annotations

import types

from personos.providers.anthropic_compat import AnthropicChatLLM


def _llm(create, temperature_is_kwarg: bool) -> AnthropicChatLLM:
    llm = object.__new__(AnthropicChatLLM)
    llm.cfg = types.SimpleNamespace(anthropic_model="m")
    llm._sdk = types.SimpleNamespace(NOT_GIVEN=object(), RateLimitError=RuntimeError)
    llm._rl_retries, llm._rl_base_sleep = 0, 0.0
    llm._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    llm._temperature_is_kwarg = temperature_is_kwarg
    return llm


def _resp(text="ok"):
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])


def test_temperature_passed_as_keyword_on_sdk_0x():
    seen = {}

    def create(*, model, max_tokens, temperature, system, messages):    # the 0.x signature
        seen.update(temperature=temperature)
        return _resp()

    assert _llm(create, True).chat([{"role": "user", "content": "hi"}], temperature=0.2) == "ok"
    assert seen["temperature"] == 0.2


def test_temperature_travels_via_extra_body_on_sdk_1x():
    seen = {}

    def create(*, model, max_tokens, system, messages, extra_body=None):  # the 1.x signature: no temperature
        seen.update(extra_body=extra_body)
        return _resp()

    assert _llm(create, False).chat([{"role": "user", "content": "hi"}], temperature=0.2) == "ok"
    assert seen["extra_body"] == {"temperature": 0.2}


def test_init_detects_the_installed_sdk_signature():
    import pytest
    anthropic = pytest.importorskip("anthropic")
    import inspect
    from personos.config import Config

    llm = AnthropicChatLLM(Config(anthropic_api_key="k"))
    assert llm._temperature_is_kwarg == ("temperature" in inspect.signature(anthropic.Anthropic(api_key="k").messages.create).parameters)
