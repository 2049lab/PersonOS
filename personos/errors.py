"""Errors that tell you what to do about them.

The library degrades where degrading is honest and raises where it is not. The
line is:

    **Cannot do it at all → raise. Did it partially → return, and say so.**

A server was the original consumer, and for a server the instinct to never fail
is right: one unconfigured optional feature should not turn into a 500. But for
someone who just installed a library, silence is the wrong answer. Passing a
video with no vision model configured should produce a sentence, not a shrug —
and certainly not a ModuleNotFoundError from inside a worker thread.

So every error here carries a remedy that can be copied and run.
"""

from __future__ import annotations


class PersonOSError(Exception):
    """Base class, so callers can catch everything this library raises."""


class ProviderError(PersonOSError):
    """The model endpoint answered 200, but not in the shape that was asked for.

    Some gateways report failure inside a successful response (a ``base_resp``
    error block, an HTML error page that JSON-parsed anyway). Surfacing that as
    a bare ``KeyError`` three frames down tells the caller nothing; this names
    what was expected and quotes what arrived, which is almost always a wrong
    base URL or a model id the endpoint does not serve.
    """


class QueueBusy(PersonOSError):
    """Backpressure: the session's backlog is full, or enqueue contention timed out.

    Not a failure — the caller should slow down and retry, the same contract
    the HTTP API expresses as 503 + Retry-After.
    """


class MissingCapability(PersonOSError):
    """A requested feature needs configuration or dependencies that are absent.

    Raised at the entry points (``Memory.add`` / ``Memory.search``) rather than
    deep in the pipeline, so the message can name the feature the caller asked
    for instead of the internal component that happened to notice.
    """

    def __init__(self, capability: str, reason: str, remedy: str) -> None:
        self.capability = capability
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"{capability} is unavailable: {reason}\n  To enable it: {remedy}")


# —— the specific cases, so the wording stays consistent ——

def no_llm() -> MissingCapability:
    return MissingCapability(
        "memory writing and recall",
        "no chat model is configured",
        "set PERSONOS_LLM_API_KEY (and PERSONOS_LLM_BASE_URL for a non-OpenAI endpoint)")


def no_embedder() -> MissingCapability:
    return MissingCapability(
        "memory retrieval",
        "no embedding model is configured",
        "set PERSONOS_LLM_API_KEY, or PERSONOS_EMBEDDING_API_KEY to use a separate endpoint")


def no_vision() -> MissingCapability:
    return MissingCapability(
        "video understanding",
        "no multimodal model is configured",
        "set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL")


def no_identity_backend() -> MissingCapability:
    return MissingCapability(
        "video identity",
        "the identity backends are disabled",
        "pip install 'personos[identity]' and set PERSONOS_VIDEO_BACKEND=real")


def no_deep_track() -> MissingCapability:
    return MissingCapability(
        "deep recall",
        "the agent dependencies are not installed",
        "pip install 'personos[deep]'")


def no_public_media_url() -> MissingCapability:
    return MissingCapability(
        "video understanding with local media storage",
        "the model service fetches clips by URL and cannot read a local path",
        "set PERSONOS_MEDIA_BASE_URL to a publicly reachable prefix serving the "
        "media directory, or use object storage with PERSONOS_MEDIA_BACKEND=oss")


# —— warnings: partial success that must not be silent ——

def image_not_understood() -> str:
    return ("image stored but not understood: no multimodal model is configured, so it "
            "contributes nothing to retrieval. Set PERSONOS_MLLM_API_KEY to enable this.")


def visual_recall_unavailable() -> str:
    return ("the image was not used to resolve this question: no multimodal model is "
            "configured. Set PERSONOS_MLLM_API_KEY to enable this.")


def deep_track_skipped() -> str:
    return ("the fast path could not fully answer and deep recall is unavailable "
            "(pip install 'personos[deep]'); returning the fast-path answer.")
