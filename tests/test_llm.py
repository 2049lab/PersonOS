"""chat_json 契约单测:围栏剥离 / num_tries 解析失败重试(重试附自纠提示,网络异常不重试)。"""

from __future__ import annotations

import pytest

from personos.online.llm import chat_json, strip_fences


class QueueLLM:
    """按队列回响应,记录每次收到的消息(供重试断言)。"""

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
    assert len(llm.calls) == 1                                  # 默认不重试


def test_num_tries_recovers_on_retry():
    llm = QueueLLM("坏了", '```json\n{"atoms": []}\n```')
    data, raw = chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)
    assert data == {"atoms": []} and raw.endswith("```")
    assert len(llm.calls) == 2
    retry = llm.calls[1]
    assert retry[-1]["content"].startswith("你上次的输出不是合法 JSON")   # 自纠提示
    assert retry[-2]["role"] == "assistant" and retry[-2]["content"] == "坏了"  # 附上次原文


def test_num_tries_exhausted_raises_after_all_tries():
    llm = QueueLLM("坏1", "坏2", "坏3")
    with pytest.raises(ValueError, match="重试 3 次"):
        chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)
    assert len(llm.calls) == 3


def test_network_error_is_not_retried():
    llm = BoomLLM()
    with pytest.raises(ConnectionError):
        chat_json(llm, [{"role": "user", "content": "q"}], max_tokens=10, num_tries=3)
