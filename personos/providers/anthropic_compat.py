"""Chat provider for the Anthropic message protocol.

Covers Anthropic itself and anything that speaks the same wire format — MiniMax
among them — by pointing ``base_url`` elsewhere. Satisfies the same ``ChatLLM``
protocol as the OpenAI-compatible provider, so it can be swapped into any stage
of the pipeline.

Requires ``pip install personos[anthropic]``.
"""

from __future__ import annotations

import inspect
import time

from loguru import logger

from ..config import Config, get_config


class AnthropicChatLLM:
    """Chat over the Anthropic messages API."""

    def __init__(self, cfg: Config | None = None, *, timeout: float = 120.0,
                 max_retries: int = 2, rate_limit_retries: int = 2,
                 rate_limit_base_sleep: float = 30.0):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the anthropic provider needs the SDK: pip install 'personos[anthropic]'"
            ) from e

        self._sdk = anthropic
        self.cfg = cfg or get_config()
        if not self.cfg.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set, so the anthropic provider cannot be used")
        # Plan-level rate limits are enforced over windows far longer than the
        # SDK's own second-scale backoff, so a second, slower layer sits on top.
        # Budget kept to 2 attempts: the previous 3 (30/60/120s) could stall a
        # worker for four minutes, and a long outage is better handled by the
        # caller retrying than by holding a thread hostage.
        self._rl_retries = rate_limit_retries
        self._rl_base_sleep = rate_limit_base_sleep
        self._client = anthropic.Anthropic(
            base_url=self.cfg.anthropic_base_url or None,
            api_key=self.cfg.anthropic_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        # SDK 1.x dropped `temperature` (and top_p/top_k) from messages.create(); 0.x has it. The
        # wire protocol still accepts the field, so on 1.x it travels via extra_body instead of
        # raising "unexpected keyword argument" and degrading every LLM stage.
        self._temperature_is_kwarg = "temperature" in inspect.signature(
            self._client.messages.create).parameters

    @property
    def available(self) -> bool:
        return bool(self.cfg.anthropic_api_key)

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 10240) -> str:
        # Anthropic takes the system prompt as its own parameter rather than as
        # a message, so it is split out here and the rest passed through in order.
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        rest = [{"role": m["role"], "content": m["content"]}
                for m in messages if m.get("role") != "system"]
        resp = self._create_with_rate_limit_retry(
            model=self.cfg.anthropic_model,
            max_tokens=max_tokens,
            system=system or self._sdk.NOT_GIVEN,
            messages=rest,
            **({"temperature": temperature} if self._temperature_is_kwarg
               else {"extra_body": {"temperature": temperature}}),
        )
        # Text blocks only. A thinking block, if the model emits one, is not part
        # of the answer — and letting it through once turned a rate-limit error
        # string into what looked like a memory answer.
        content = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        logger.debug(f"anthropic chat returned {len(content)} characters")
        return content

    def _create_with_rate_limit_retry(self, **kw):
        for attempt in range(self._rl_retries + 1):
            try:
                return self._client.messages.create(**kw)
            except self._sdk.RateLimitError as e:
                if attempt == self._rl_retries:
                    raise
                wait = self._rl_base_sleep * (2 ** attempt)
                logger.warning(f"rate limited (long backoff "
                               f"{attempt + 1}/{self._rl_retries}), retrying in {wait:.0f}s: {e}")
                time.sleep(wait)

    def embed(self, texts: list[str]):  # pragma: no cover - explicit, to prevent misuse
        raise NotImplementedError(
            "this provider only does chat; configure an embedder separately")

    def close(self) -> None:
        self._client.close()
