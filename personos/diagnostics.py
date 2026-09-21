"""What the current configuration can and cannot do.

The capability tiers are documented in the README, but a table in a README goes
stale and cannot see your environment. This reads the actual configuration and
answers the only question that matters at setup time: *what works right now, and
what do I type to unlock the rest?*

Used by ``personos doctor`` and by the entry-point checks, so both speak with
one voice.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .config import Config, get_config


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str            # what is (or is not) configured
    remedy: str = ""       # empty when available
    required: bool = False


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def inspect(cfg: Config | None = None) -> list[Capability]:
    """Report every optional and required capability, in dependency order."""
    c = cfg or get_config()
    out: list[Capability] = []

    out.append(Capability(
        "chat model", bool(c.llm_api_key),
        f"{c.llm_model} at {c.llm_base_url}" if c.llm_api_key else "no API key",
        "" if c.llm_api_key else "set PERSONOS_LLM_API_KEY",
        required=True))

    emb_key = c.effective_embedding_api_key
    out.append(Capability(
        "embeddings", bool(emb_key),
        f"{c.embedding_model} ({c.embedding_dim}d)" if emb_key else "no API key",
        "" if emb_key else "set PERSONOS_LLM_API_KEY, or PERSONOS_EMBEDDING_API_KEY",
        required=True))

    using_mysql = bool(c.db_url)
    out.append(Capability(
        "storage", True,
        "MySQL" if using_mysql else f"SQLite at {c.sqlite_path}",
        "" if using_mysql else "set PERSONOS_DB_URL for MySQL (optional; needed for "
                               "multiple workers)"))

    out.append(Capability(
        "media storage", True,
        "object storage" if c.media_backend == "oss" else f"local files at {c.media_root}",
        "" if c.media_backend == "oss" else "set PERSONOS_MEDIA_BACKEND=oss for object storage"))

    out.append(Capability(
        "image understanding", bool(c.mllm_api_key),
        f"{c.mllm_model}" if c.mllm_api_key else "not configured — images are stored "
                                                 "but contribute nothing to retrieval",
        "" if c.mllm_api_key else "set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL"))

    video_ready = bool(c.mllm_api_key) and c.video_backend == "real" and _has("av")
    if not c.mllm_api_key:
        why, fix = "needs a multimodal model", "set PERSONOS_MLLM_API_KEY"
    elif c.video_backend != "real":
        why, fix = ("disabled (PERSONOS_VIDEO_BACKEND=%s)" % c.video_backend,
                    "pip install 'personos[identity]' and set PERSONOS_VIDEO_BACKEND=real")
    elif not _has("av"):
        why, fix = "dependencies missing", "pip install 'personos[identity]'"
    else:
        why, fix = c.video_backend, ""
    out.append(Capability("video + person identity", video_ready, why, fix))

    if video_ready and c.media_backend != "oss" and not c.media_base_url:
        out.append(Capability(
            "  video with local media", False,
            "the model fetches clips by URL and cannot read a local path",
            "set PERSONOS_MEDIA_BASE_URL, or use PERSONOS_MEDIA_BACKEND=oss"))

    deep_ok = _has("langchain_classic")
    out.append(Capability(
        "deep recall", deep_ok,
        "multi-step agent" if deep_ok else "not installed — recall uses the fast path only",
        "" if deep_ok else "pip install 'personos[deep]'"))

    rerank_ok = bool(c.rerank_api_key and c.rerank_model)
    out.append(Capability(
        "reranking", rerank_ok,
        c.rerank_model if rerank_ok else "not configured — retrieval keeps its fusion order",
        "" if rerank_ok else "set PERSONOS_RERANK_API_KEY and PERSONOS_RERANK_MODEL"))

    multi = bool(c.redis_url)
    out.append(Capability(
        "multiple workers", multi,
        "shared state on Redis" if multi else "single process — session state is in memory",
        "" if multi else "set PERSONOS_REDIS_URL"))

    return out


def render(caps: list[Capability]) -> str:
    lines = ["PersonOS configuration", ""]
    blocked = [c for c in caps if c.required and not c.available]
    for c in caps:
        mark = "OK  " if c.available else ("FAIL" if c.required else "--  ")
        lines.append(f"  [{mark}] {c.name:<26} {c.detail}")
        if c.remedy:
            lines.append(f"           -> {c.remedy}")
    lines.append("")
    if blocked:
        lines.append(f"Not usable yet: {', '.join(c.name for c in blocked)} missing.")
    else:
        lines.append("Ready. Optional capabilities marked '--' are off; "
                     "everything else degrades gracefully without them.")
    return "\n".join(lines)
