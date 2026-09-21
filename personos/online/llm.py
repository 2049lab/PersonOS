"""共享 LLM 调用契约:ChatLLM 协议 + JSON 输出清洗。

写入(W1/W2)、检索(R0)、核判(R3')共用一份,协议站点不再各自定义。
"""

from __future__ import annotations

import json
from typing import Protocol

from loguru import logger

from .. import obs


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> str: ...


def strip_fences(s: str) -> str:
    """去掉 ```/```json 围栏;LLM 常把 JSON 包在代码块里。"""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json"):
            s = s[4:]
    return s.strip()


def _user_content(messages: list[dict]) -> str:
    """拼接 user 角色内容(动态输入:transcript/episode/候选链/query 等);跳过静态 system prompt。"""
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


# —— 业务方场景注入(留口子:高杠杆 LLM 环节的 system prompt 里插一段调用方场景描述)——
_SCEN_HEADER = "# Caller scenario (business context from the calling application)"


def with_scenario(prompt: str, anchor: str, scenario: str, directive: str) -> str:
    """在 anchor 段【前】插入业务方场景区块;scenario 为空 → 原样返回(默认链路逐字节不变)。

    业务偏好只调「关注度/详略」,directive 里写死「不改事实、不编造、不漏」的守则,防跑偏。
    anchor 取各 prompt 的下一个 section 标题(避开 JSON 花括号,replace 一次即可);
    anchor 万一不在 prompt 里(日后改了标题),兜底追加到末尾——scenario 绝不静默丢失。
    """
    s = (scenario or "").strip()
    if not s:
        return prompt
    block = f"{_SCEN_HEADER}\n{s}\n{directive}\n\n"
    if anchor and anchor in prompt:
        return prompt.replace(anchor, block + anchor, 1)
    return prompt.rstrip() + "\n\n" + block.rstrip()


def chat_json(llm: ChatLLM, messages: list[dict], *, max_tokens: int,
              temperature: float = 0.0, num_tries: int = 1, stage: str = ""):
    """调 LLM 并把输出解析为 JSON;解析失败抛 ValueError(附 .raw=模型原文,调用方决定兜底)。

    num_tries > 1 时 JSON 解析失败自动重发(附上次原文与"只输出 JSON"提示,让模型自我纠正)。
    只重试解析失败,不重试网络异常——后者由调用方各自的降级路径接住。
    stage:LLM 阶段名(如 rewrite_query/answer),经 obs.stage 传给 MaasClient 给 langfuse span 命名。
    """
    msgs = messages
    for attempt in range(1, num_tries + 1):
        with obs.stage(stage):
            raw = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        # 完整打这次 LLM 看到的动态输入 + 输出原文(排障用;静态 system 不打)。重试每次各一条
        logger.info(f"LLM[{stage or 'chat'}] attempt={attempt}/{num_tries}\n"
                    f"  ── 输入(user)──\n{_user_content(msgs)}\n"
                    f"  ── 输出(raw)──\n{raw}")
        try:
            return json.loads(strip_fences(raw)), raw
        except json.JSONDecodeError as e:
            if attempt == num_tries:
                err = ValueError(f"LLM 输出不是合法 JSON(重试 {num_tries} 次仍失败): {e}")
                err.raw = raw   # 降级方仍能拿到原文(网络异常等无 .raw,getattr 兜底)
                raise err from e
            msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "你上次的输出不是合法 JSON,无法解析。"
                                            "请重新输出,只输出 JSON 本体,不要解释、不要代码块围栏。"},
            ]
            logger.warning(f"chat_json 第 {attempt} 次输出非 JSON,重试(共 {num_tries} 次)")
