"""Contract unit tests for chat_json: fence stripping / retry on parse failure via num_tries
(the retry carries a self-correction hint, and network errors are not retried)."""

from __future__ import annotations

import pytest

from personos.online.llm import chat_json, strip_fences


class QueueLLM:
    """Replies from a queue and records the messages received on each call, so retries can be
    asserted on."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, temperature=0.0, max_tokens=1024) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


class BoomLLM:
    def chat(self, messages, temperature=0.0, max_tokens=1024) -> str:
        raise ConnectionError("network down")


def test_strip_fences_unwraps_code_block():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('  {"a": 1}  ') == '{"a": 1}'


def test_default_single_try_raises_with_raw():
    llm = QueueLLM("不是 JSON")
    with pytest.raises(ValueError) as ei:
        chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10)
    assert ei.value.raw == "不是 JSON"
    assert len(llm.calls) == 1                                  # no retry by default


def test_num_tries_recovers_on_retry():
    llm = QueueLLM("坏了", '```json\n{"atoms": []}\n```')
    data, raw = chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)
    assert data == {"atoms": []} and raw.endswith("```")
    assert len(llm.calls) == 2
    retry = llm.calls[1]
    # The self-correction hint.
    assert retry[-1]["content"].startswith("Your last output was not valid JSON")
    # The previous raw output is attached.
    assert retry[-2]["role"] == "assistant" and retry[-2]["content"] == "坏了"


def test_num_tries_exhausted_raises_after_all_tries():
    llm = QueueLLM("坏1", "坏2", "坏3")
    with pytest.raises(ValueError, match="after 3 tries"):
        chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)
    assert len(llm.calls) == 3


def test_network_error_is_not_retried():
    llm = BoomLLM()
    with pytest.raises(ConnectionError):
        chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)


def test_strip_fences_removes_a_leading_reasoning_block():
    """Reasoning models put <think>...</think> inside content, before the answer.

    DeepSeek-R1, QwQ and MiniMax-M3 all do it, and JSON parsing fails on the
    whole string. Some providers offer a flag to turn it off, but relying on
    that would make the library work with one vendor's parameter and silently
    break with the next — so the block is stripped instead.
    """
    from personos.online.llm import strip_fences

    assert strip_fences('<think>reasoning</think>\n\n{"ok": 1}') == '{"ok": 1}'
    assert strip_fences('<thinking>x</thinking>{"ok": 1}') == '{"ok": 1}'
    # Combined with a code fence, which models also add unprompted.
    assert strip_fences('<think>x</think>\n```json\n{"ok": 1}\n```') == '{"ok": 1}'


def test_strip_fences_leaves_unrelated_text_alone():
    """Only a well-formed leading block is removed. Content that merely mentions
    the tag must survive, or stripping would corrupt legitimate payloads."""
    from personos.online.llm import strip_fences

    assert strip_fences('{"note": "we use <think> tags"}') == '{"note": "we use <think> tags"}'
    assert strip_fences('{"ok": 1}') == '{"ok": 1}'


def test_chat_json_accepts_raw_newlines_inside_strings():
    """A literal newline inside a JSON string is invalid under strict JSON but is exactly what models emit
    for multi-line text; it must parse on the first try rather than burn the retries and degrade the stage."""
    llm = QueueLLM('{"episode": "line one\nline two", "ops": []}')
    data, _ = chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=2)
    assert data["episode"] == "line one\nline two" and len(llm.calls) == 1
