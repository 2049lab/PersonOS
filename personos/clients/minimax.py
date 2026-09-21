"""MiniMax 客户端(Anthropic 兼容接口):评测等场景的第二 LLM 供应商。

满足框架的 ChatLLM 协议(chat(messages, temperature, max_tokens) -> str),
可直接替换 MaasClient 注入 ingest/recall/light_dream 各环节。
- system 消息从 messages 里拆出走 Anthropic 的 system 参数;
- 只取 text 块拼接返回,thinking 块跳过(M3 默认关 thinking,此为兜底);
- 超时/重试交给 anthropic SDK 内建(max_retries 带指数退避)。
密钥只从环境变量/.env 读(config),不落代码。
"""

from __future__ import annotations

import time

import anthropic
from loguru import logger

from ..config import Settings, settings


class MinimaxClient:
    """Anthropic 兼容协议的 MiniMax chat 客户端。未配置 MINIMAX_API_KEY 时初始化即抛。"""

    def __init__(self, cfg: Settings = settings, timeout: float = 120.0, max_retries: int = 2,
                 rate_limit_retries: int = 2, rate_limit_base_sleep: float = 30.0):
        if not cfg.minimax_api_key:
            raise ValueError("MINIMAX_API_KEY 未配置(检查 .env);MiniMax 客户端不可用")
        self.cfg = cfg
        # Token Plan 限流是套餐级,SDK 内建重试的秒级退避等不到恢复——chat 里再加一层长退避;
        # 预算收紧为 2 次(旧 3 次 30/60/120 最坏挂 4 分钟),长断供靠评测外层重试/探针续跑兜底
        self._rl_retries = rate_limit_retries
        self._rl_base_sleep = rate_limit_base_sleep
        self._client = anthropic.Anthropic(
            base_url=cfg.minimax_base_url,
            api_key=cfg.minimax_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 10240) -> str:
        # system 消息拆出(Anthropic 格式);其余按序透传
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        rest = [{"role": m["role"], "content": m["content"]}
                for m in messages if m.get("role") != "system"]
        resp = self._create_with_rate_limit_retry(
            model=self.cfg.minimax_chat_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or anthropic.NOT_GIVEN,
            messages=rest,
        )
        # 只取 text 块;M3 默认不出 thinking 块,若有则跳过(抽取/判分只要正文)
        content = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        logger.debug(f"minimax chat 返回 {len(content)} 字符")
        return content

    def _create_with_rate_limit_retry(self, **kw):
        """429 长退避:SDK 内建重试(秒级,2 次)耗尽仍限流时,再等 30/60s 各试一次。

        Token Plan 限流窗口远长于 SDK 退避间隔,这层缺失曾让评测把 'Error code: 429 - …'
        异常文本漏成 R5 答案(S4-H1)。仍失败则原样抛出,由调用方降级路径接住。
        """
        for attempt in range(self._rl_retries + 1):
            try:
                return self._client.messages.create(**kw)
            except anthropic.RateLimitError as e:
                if attempt == self._rl_retries:
                    raise
                wait = self._rl_base_sleep * (2 ** attempt)
                logger.warning(f"MiniMax 限流(第 {attempt + 1}/{self._rl_retries} 次长退避),"
                               f"{wait:.0f}s 后重试: {e}")
                time.sleep(wait)

    def embed(self, texts: list[str]):  # pragma: no cover - 明确不支持,防误用
        raise NotImplementedError("MiniMax 客户端只做 chat;embedding 仍走 MAAS(qwen3-embedding)")

    def close(self) -> None:
        self._client.close()
