"""Shared LLM call contract: the ChatLLM protocol + cleanup of JSON output.

The write path (W1/W2), retrieval (R0) and adjudication (R3') all share this one definition, so no
call site declares its own protocol any more.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from loguru import logger

from .. import obs


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> str: ...


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE)


def strip_fences(s: str) -> str:
    """Normalise a model's raw text down to the payload we asked for.

    Two things get in the way, both common enough to handle rather than treat
    as the caller's problem:

    - **Code fences.** Models wrap JSON in ``` blocks despite being told not to.
    - **Reasoning blocks.** Models that think out loud (DeepSeek-R1, QwQ,
      MiniMax-M3 and others) emit `<think>...</think>` *inside* content, before
      the answer. Some let you disable it with a vendor-specific parameter;
      relying on that would make the library work with one provider's flag and
      silently fail with the next. Stripping the block is provider-neutral.

    Only a leading block is removed, and only a well-formed one — text that
    merely mentions the tag is left alone.
    """
    s = _THINK_BLOCK.sub("", s.strip(), count=1).strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json"):
            s = s[4:]
    return s.strip()


def _user_content(messages: list[dict]) -> str:
    """Join the content of the user-role messages (the dynamic input: transcript, episode, candidate
    chains, query, ...); the static system prompt is skipped."""
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


# -- Caller scenario injection (an extension point: insert a description of the caller's scenario into
# the system prompt of the high-leverage LLM steps) --
_SCEN_HEADER = "# Caller scenario (business context from the calling application)"


def with_scenario(prompt: str, anchor: str, scenario: str, directive: str) -> str:
    """Insert the caller-scenario block BEFORE the anchor section; an empty scenario returns the
    prompt unchanged (so the default path stays byte-for-byte identical).

    Business preferences may only tune attention and level of detail; the directive hard-codes the
    rules "do not alter facts, do not invent, do not omit" to keep it from drifting. The anchor is
    the next section heading of each prompt (which avoids the JSON braces, so a single replace is
    enough). If the anchor is not in the prompt (say a heading gets renamed later), fall back to
    appending at the end — the scenario is never silently dropped.
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
    """Call the LLM and parse its output as JSON; a parse failure raises ValueError (with .raw set to
    the model's original text, so the caller decides how to fall back).

    With num_tries > 1, a JSON parse failure is retried automatically (resending the previous output
    plus an "output JSON only" instruction so the model can correct itself). Only parse failures are
    retried, never network errors — those are caught by each caller's own degradation path.
    stage: the name of the LLM stage (e.g. rewrite_query / answer), passed through obs.stage to
    MaasClient to name the langfuse span.
    """
    msgs = messages
    for attempt in range(1, num_tries + 1):
        with obs.stage(stage):
            raw = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        # Log the full dynamic input this LLM call saw plus its raw output (for troubleshooting; the
        # static system prompt is not logged). One line per retry attempt
        logger.info(f"LLM[{stage or 'chat'}] attempt={attempt}/{num_tries}\n"
                    f"  -- input(user) --\n{_user_content(msgs)}\n"
                    f"  -- output(raw) --\n{raw}")
        try:
            # strict=False: models routinely put a literal newline or tab inside a string value
            # (multi-line episode / answer text). The structure is otherwise valid, and rejecting it
            # cost a whole stage (unchained atoms, a raw-text answer) even after the retries.
            return json.loads(strip_fences(raw), strict=False), raw
        except json.JSONDecodeError as e:
            if attempt == num_tries:
                err = ValueError(f"LLM output is not valid JSON (still failing after {num_tries} tries): {e}")
                err.raw = raw   # the fallback path can still read the original text (network errors and the like have no .raw, so callers use getattr)
                raise err from e
            msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your last output was not valid JSON and could not "
                                            "be parsed. Output again: the JSON body only, no "
                                            "explanation, no code fences."},
            ]
            logger.warning(f"chat_json attempt {attempt} produced non-JSON output, retrying ({num_tries} tries total)")
