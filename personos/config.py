"""Configuration, read from environment variables and an optional `.env` file.

Two properties this module must have, and the reasons they are not negotiable:

**Nothing is evaluated at import time.** ``import personos`` has to work in a
completely empty environment — no API key, no database, nothing. A library that
reads secrets while being imported cannot honour a zero-configuration promise,
and the failure surfaces as an import error, which is the least debuggable place
for it to happen. ``scripts/check_cold_start.py`` guards this.

**Loading configuration never raises.** A missing key is not an error here; it is
an unavailable *capability*. The error belongs at the point of use, where we can
say which feature needs which variable. See ``personos/errors.py``.

Everything has a default that works. The only variable a user must set is
``PERSONOS_LLM_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: bool = True) -> bool:
    return _env(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


@dataclass(frozen=True)
class Config:
    """Resolved settings. Construct through :func:`get_config`, not directly."""

    # ── Required capability: a chat/embedding endpoint ────────────────────
    # One OpenAI-compatible endpoint serves both, so a single key is enough to
    # start. Point base_url at OpenAI, vLLM, Ollama, LiteLLM, or any gateway
    # that speaks /chat/completions and /embeddings.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 120.0      # non-streaming with large max_tokens: 60s+ is normal
    llm_app_id: str = ""            # some gateways require an application id header

    embedding_base_url: str = ""    # falls back to llm_base_url
    embedding_api_key: str = ""     # falls back to llm_api_key
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    io_timeout: float = 30.0        # embeddings/rerank are fast; a tight timeout surfaces faults

    # ── Optional: reranker. Unset keeps the fusion order from retrieval. ──
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""

    # ── Optional: multimodal. Unset means text memory is unaffected, images
    #    are stored but not understood, and video is rejected with a clear error.
    mllm_base_url: str = ""         # falls back to llm_base_url
    mllm_endpoint: str = ""         # full URL override, when a gateway does not follow /v1
    mllm_api_key: str = ""
    mllm_model: str = ""
    mllm_timeout: float = 120.0

    # ── Storage ──────────────────────────────────────────────────────────
    # Empty db_url means the local SQLite file under data_dir, tables created
    # automatically. Set it to a SQLAlchemy URL for MySQL.
    db_url: str = ""
    data_dir: Path = field(default_factory=lambda: Path.home() / ".personos")
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # Legacy discrete MySQL settings, still read by storage/db.py. Phase 3 of the
    # open-sourcing plan replaces them with db_url plus a SQLite default; kept
    # until then so the existing storage layer and its tests are untouched.
    mysql_host: str = ""
    mysql_port: int = 3306
    mysql_user: str = "personos"
    mysql_password: str = ""
    mysql_database: str = "personos"

    # Object storage. Empty means the local filesystem under data_dir/media.
    media_backend: str = "local"    # local | oss
    media_base_url: str = ""        # public prefix; required for video + a remote MLLM
    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""
    oss_bucket: str = ""
    oss_endpoint: str = ""
    oss_region: str = ""
    oss_prefix: str = "personos/"
    oss_url_expires_seconds: int = 3600

    # Redis. Empty means single-process: segment state, session locks, the
    # message queue and identity drafts all live in memory.
    redis_url: str = ""
    redis_cluster: str = ""
    env: str = "local"              # key namespace prefix, so deployments cannot collide

    # ── Video identity ───────────────────────────────────────────────────
    video_backend: str = "none"     # none | real | mock

    # ── Retrieval / writing behaviour ────────────────────────────────────
    deep_write: bool = True         # let the deep track write new atoms back

    # ── Concurrency (server mode; the library uses none of these) ────────
    ingest_pool_size: int = 50
    # Video concurrency has exactly one knob. Budget per slot: one temp clip
    # file held from download until harvesting finishes, across the whole
    # screenplay call. So worst-case disk is video_pool_size x clip size.
    video_pool_size: int = 8
    recall_pool_size: int = 50
    profile_pool_size: int = 50
    profile_ep_chars_trigger: int = 15000
    # Fairness valve: a chatty session returns its pool slot after this many
    # messages and re-queues behind everyone else. MUST stay well below
    # max_queue_depth, or the valve never fires: the queue holds at most
    # max_queue_depth messages, so the drain loop always exits at "queue
    # empty" first and the session keeps its slot until fully drained.
    max_drain_per_cycle: int = 5
    max_ingest_retries: int = 5
    max_queue_depth: int = 15
    dispatcher_tick_s: float = 0.05
    dispatcher_idle_tick_s: float = 0.5

    # ── Observability. Unset means every trace call is a no-op. ──────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "local"
    langfuse_release: str = ""

    log_dir: Path = field(default_factory=lambda: ROOT / "logs")

    # ── Secondary LLM provider, used by the evaluation harness only ──────
    minimax_base_url: str = ""
    minimax_api_key: str = ""
    minimax_chat_model: str = "MiniMax-M3"

    # ── Derived accessors: "unset means inherit" made explicit ───────────
    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_mllm_base_url(self) -> str:
        return self.mllm_base_url or self.llm_base_url

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "personos.db"

    @property
    def media_root(self) -> Path:
        return self.data_dir / "media"


def _path(value: str, default: Path) -> Path:
    if not value:
        return default
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config() -> Config:
    """Read the environment into a :class:`Config`. Never raises."""
    data_dir = _path(_env("PERSONOS_DATA_DIR"), Path.home() / ".personos")
    return Config(
        llm_base_url=_env("PERSONOS_LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=_env("PERSONOS_LLM_API_KEY"),
        llm_model=_env("PERSONOS_LLM_MODEL", "gpt-4o-mini"),
        llm_timeout=float(_env("PERSONOS_LLM_TIMEOUT", "120")),
        llm_app_id=_env("PERSONOS_LLM_APP_ID"),
        embedding_base_url=_env("PERSONOS_EMBEDDING_BASE_URL"),
        embedding_api_key=_env("PERSONOS_EMBEDDING_API_KEY"),
        embedding_model=_env("PERSONOS_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(_env("PERSONOS_EMBEDDING_DIM", "1536")),
        io_timeout=float(_env("PERSONOS_IO_TIMEOUT", "30")),
        rerank_base_url=_env("PERSONOS_RERANK_BASE_URL"),
        rerank_api_key=_env("PERSONOS_RERANK_API_KEY"),
        rerank_model=_env("PERSONOS_RERANK_MODEL"),
        mllm_base_url=_env("PERSONOS_MLLM_BASE_URL"),
        mllm_endpoint=_env("PERSONOS_MLLM_ENDPOINT"),
        mllm_api_key=_env("PERSONOS_MLLM_API_KEY"),
        mllm_model=_env("PERSONOS_MLLM_MODEL"),
        mllm_timeout=float(_env("PERSONOS_MLLM_TIMEOUT", "120")),
        db_url=_env("PERSONOS_DB_URL"),
        data_dir=data_dir,
        db_pool_size=int(_env("PERSONOS_DB_POOL_SIZE", "10")),
        db_max_overflow=int(_env("PERSONOS_DB_MAX_OVERFLOW", "20")),
        mysql_host=_env("MYSQL_HOST"),
        mysql_port=int(_env("MYSQL_PORT", "3306")),
        mysql_user=_env("MYSQL_USER", "personos"),
        mysql_password=_env("MYSQL_PASSWORD"),
        mysql_database=_env("MYSQL_DATABASE", "personos"),
        media_backend=_env("PERSONOS_MEDIA_BACKEND", "local").strip().lower(),
        media_base_url=_env("PERSONOS_MEDIA_BASE_URL").rstrip("/"),
        oss_access_key_id=_env("OSS_ACCESS_KEY_ID"),
        oss_access_key_secret=_env("OSS_ACCESS_KEY_SECRET"),
        oss_bucket=_env("OSS_BUCKET"),
        oss_endpoint=_env("OSS_ENDPOINT"),
        oss_region=_env("OSS_REGION"),
        oss_prefix=_env("OSS_PREFIX", "personos/"),
        oss_url_expires_seconds=int(_env("OSS_URL_EXPIRES_SECONDS", "3600")),
        redis_url=_env("PERSONOS_REDIS_URL"),
        redis_cluster=_env("PERSONOS_REDIS_CLUSTER"),
        env=_env("PERSONOS_ENV", "local"),
        video_backend=_env("PERSONOS_VIDEO_BACKEND", "none").strip().lower(),
        deep_write=_flag("PERSONOS_DEEP_WRITE", True),
        ingest_pool_size=int(_env("PERSONOS_INGEST_POOL", "50")),
        video_pool_size=int(_env("PERSONOS_VIDEO_POOL", "8")),
        recall_pool_size=int(_env("PERSONOS_RECALL_POOL", "50")),
        profile_pool_size=int(_env("PERSONOS_PROFILE_POOL", "50")),
        profile_ep_chars_trigger=int(_env("PERSONOS_PROFILE_EP_CHARS", "15000")),
        max_drain_per_cycle=int(_env("PERSONOS_MAX_DRAIN", "5")),
        max_ingest_retries=int(_env("PERSONOS_MAX_INGEST_RETRIES", "5")),
        max_queue_depth=int(_env("PERSONOS_MAX_QUEUE_DEPTH", "15")),
        dispatcher_tick_s=float(_env("PERSONOS_DISPATCH_TICK", "0.05")),
        dispatcher_idle_tick_s=float(_env("PERSONOS_DISPATCH_IDLE_TICK", "0.5")),
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        langfuse_environment=_env("LANGFUSE_ENVIRONMENT", _env("PERSONOS_ENV", "local")),
        langfuse_release=_env("LANGFUSE_RELEASE"),
        log_dir=_path(_env("PERSONOS_LOG_DIR"), ROOT / "logs"),
        minimax_base_url=_env("MINIMAX_BASE_URL"),
        minimax_api_key=_env("MINIMAX_API_KEY"),
        minimax_chat_model=_env("MINIMAX_CHAT_MODEL", "MiniMax-M3"),
    )


def get_secret(name: str, *, required: bool = False) -> str:
    """Read a secret from the environment.

    This used to fall back to an internal key-management service. That fallback
    is gone: an open-source library has no business reaching for a proprietary
    secret store, and every deployment platform can inject environment
    variables. The seam is kept because callers want the "missing is fine"
    versus "missing is fatal" distinction stated at the call site.
    """
    value = os.environ.get(name, "")
    if not value and required:
        raise RuntimeError(f"{name} is required but not set in the environment")
    return value


_cached: Config | None = None


def get_config() -> Config:
    """The process-wide configuration, loaded on first use and cached."""
    global _cached
    if _cached is None:
        _cached = load_config()
    return _cached


def set_config(cfg: Config) -> None:
    """Override the cached configuration. Intended for tests and embedding hosts."""
    global _cached
    _cached = cfg


def reset_config() -> None:
    """Drop the cache so the next :func:`get_config` re-reads the environment."""
    global _cached
    _cached = None


class _LazyConfig:
    """Attribute-forwarding stand-in for :data:`Config`, resolved on first access.

    This exists so that ``from personos.config import settings`` — used in a
    dozen modules, and as a default argument in several client constructors —
    stops being an import-time read without those call sites having to change.
    A default argument binds *this* object at definition time; the environment
    is only consulted when an attribute is actually touched.

    Prefer :func:`get_config` in new code. ``settings`` is a compatibility
    bridge, and will disappear once the ``Memory`` facade owns configuration.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)

    def __repr__(self) -> str:
        return f"<lazy {get_config()!r}>"


settings = _LazyConfig()

# Backwards-compatible alias: the type used to be called Settings.
Settings = Config
