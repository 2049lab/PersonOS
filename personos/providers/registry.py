"""Provider registry: a name in configuration, a class at runtime.

Classes are referenced by dotted path and imported on demand. That indirection
is what lets optional dependencies stay optional — nothing imports ``anthropic``
or an identity backend unless a configuration actually selects it.

Adding a provider is one line here plus the class. A fork that needs a private
gateway registers it the same way, without touching the shipped code.
"""

from __future__ import annotations

import importlib
from typing import Any

PROVIDERS: dict[str, dict[str, str]] = {
    "llm": {
        "openai": "personos.providers.openai_compat.OpenAIChatLLM",
        "anthropic": "personos.providers.anthropic_compat.AnthropicChatLLM",
    },
    "embedder": {
        "openai": "personos.providers.openai_compat.OpenAIEmbedder",
    },
    "mllm": {
        "openai": "personos.providers.openai_compat.OpenAIMllm",
        "none": "personos.providers.openai_compat.NullMllm",
    },
    "reranker": {
        "openai": "personos.providers.openai_compat.OpenAIReranker",
        "noop": "personos.online.rerank.NoopReranker",
    },
    "database": {
        "sqlite": "personos.storage.db.sqlite.SQLiteDatabase",
        "mysql": "personos.storage.db.mysql.MySQLDatabase",
    },
    "media": {
        "local": "personos.storage.media.local.LocalMediaStore",
        "oss": "personos.storage.media.oss.OSSMediaStore",
    },
}


def load(kind: str, provider: str) -> type:
    """Resolve ``(kind, provider)`` to a class, importing it on first use."""
    known = PROVIDERS.get(kind)
    if known is None:
        raise KeyError(f"unknown provider kind {kind!r}; expected one of {sorted(PROVIDERS)}")
    path = known.get(provider)
    if path is None:
        raise KeyError(
            f"unknown {kind} provider {provider!r}; available: {sorted(known)}")
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def build(kind: str, provider: str, **kwargs: Any) -> Any:
    return load(kind, provider)(**kwargs)
