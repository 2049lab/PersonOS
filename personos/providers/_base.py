"""Shared HTTP plumbing for providers: transport, auth, and loud parse errors.

Kept separate from any vendor dialect module so dialects (``openai_compat``,
``rerank``) can both build on it without importing each other.
"""

from __future__ import annotations

import httpx

from ..config import Config, get_config
from ..errors import ProviderError
from ._http import post_json


def _expect(data: dict, key: str, label: str, url: str) -> None:
    """Fail loudly when a 200 carries an error body instead of the expected field.

    Several gateways say no inside a successful response (MiniMax's ``base_resp``
    error block, for one). Without this check the failure surfaces as a bare
    ``KeyError`` at the parse site, which points at our code instead of at the
    real cause — almost always a wrong base URL or a model the endpoint does
    not serve.
    """
    if key not in data:
        raise ProviderError(
            f"{label}: response from {url} has no {key!r} field — the endpoint is not "
            f"speaking the OpenAI {label} shape (wrong base URL, or a model it does not "
            f"serve?). body: {str(data)[:300]}")


class _Base:
    def __init__(self, cfg: Config | None = None, *, timeout: float | None = None,
                 max_retries: int = 3, rate_limit_attempts: int = 2):
        self.cfg = cfg or get_config()
        self.timeout = timeout
        self.max_retries = max_retries
        # A shallow 429 budget on purpose: see providers/_http.
        self.rate_limit_attempts = rate_limit_attempts
        # trust_env=False: a system proxy configured for browsing (Clash and
        # friends) gets inherited by the shell and then silently intercepts
        # calls to a private gateway, which fails as "connection refused" at a
        # confusing distance from the cause. Providers are addressed explicitly.
        self._client = httpx.Client(trust_env=False)

    def _post(self, url: str, headers: dict, payload: dict, timeout: float, label: str) -> dict:
        return post_json(self._client, url, headers, payload, timeout=timeout,
                         max_retries=self.max_retries,
                         rate_limit_attempts=self.rate_limit_attempts, label=label)

    def _auth(self, api_key: str) -> dict:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}
        if self.cfg.llm_app_id:      # some gateways require an application id
            headers["x-app-id"] = self.cfg.llm_app_id
        return headers

    def close(self) -> None:
        self._client.close()
