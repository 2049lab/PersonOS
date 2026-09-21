"""Shared HTTP behaviour for model providers: timeouts, retries, backoff.

Lifted verbatim from the internal gateway client. The transport policy encoded
here was tuned against real failures and is worth keeping even though the
authentication it used to carry is gone:

- **429 gets a shallow budget, not a deep one.** A generous retry budget looks
  kinder but behaves worse under concurrency: every rate-limited request sleeps
  in lockstep and then wakes together to collide again, so the worker pool ends
  up entirely asleep. Failing fast and letting the caller decide is the correct
  shape.
- **Backoff is jittered.** Fixed backoff has the same lockstep problem in
  miniature: sleep together, wake together, re-collide.
- **4xx is never retried.** It will not become a different answer.
"""

from __future__ import annotations

import random
import time

import httpx
from loguru import logger


def jitter(base: float) -> float:
    """Randomise a backoff by 0.5x-1.5x so concurrent retries spread out."""
    return base * random.uniform(0.5, 1.5)


def usage_of(data: dict) -> dict | None:
    """Token usage from an OpenAI-shaped response, filtered to numbers only
    (tracing backends reject anything else)."""
    u = (data or {}).get("usage") or {}
    out = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
           "total": u.get("total_tokens")}
    out = {k: v for k, v in out.items() if isinstance(v, int)}
    return out or None


def post_json(client: httpx.Client, url: str, headers: dict, payload: dict, *,
              timeout: float, max_retries: int = 3, rate_limit_attempts: int = 2,
              label: str = "") -> dict:
    """POST with retries. Network errors and 5xx back off; 429 has its own
    shallow budget; other 4xx raise immediately."""
    attempt = 0     # network errors / 5xx
    rl = 0          # 429, budgeted separately
    what = label or url
    while True:
        try:
            resp = client.post(url, headers=headers, json=payload,
                               timeout=httpx.Timeout(timeout, connect=10.0))
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                rl += 1
                if rl >= rate_limit_attempts:
                    logger.error(f"{what}: rate limited, retry budget of "
                                 f"{rate_limit_attempts} exhausted, re-raising")
                    raise
                wait = jitter(min(1.0 * (2 ** (rl - 1)), 4.0))
                logger.warning(f"{what}: rate limited, backing off {wait:.1f}s "
                               f"({rl}/{rate_limit_attempts})")
                time.sleep(wait)
                continue
            if status and 400 <= status < 500:
                body = getattr(getattr(e, "response", None), "text", "")
                logger.error(f"{what}: {status}, not retrying: {body[:200]}")
                raise
            attempt += 1
            if attempt >= max_retries:
                raise
            logger.warning(f"{what}: attempt {attempt} failed ({e}; limit {timeout:g}s), retrying")
            time.sleep(jitter(0.5 * attempt))
