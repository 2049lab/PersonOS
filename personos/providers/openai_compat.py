"""Providers for any OpenAI-compatible endpoint.

One base URL plus one key covers chat, embeddings and vision, which is why the
minimum configuration for this library is a single variable. Point it at OpenAI,
vLLM, Ollama, LiteLLM, or a corporate gateway — the wire format is the same.

Timeouts are split by interface rather than shared: chat is non-streaming with a
large ``max_tokens``, so a minute or more is normal and a tight timeout would
cut off healthy requests. Embeddings and reranking are fast, so a tight timeout
there surfaces faults early instead of hiding them behind a long wait.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np
from loguru import logger

from .. import obs
from ..config import Config, get_config
from ._http import post_json, usage_of


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


class OpenAIChatLLM(_Base):
    """Chat completions. Satisfies :class:`personos.online.llm.ChatLLM`."""

    @property
    def available(self) -> bool:
        return bool(self.cfg.llm_api_key)

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 10240) -> str:
        cfg = self.cfg
        payload = {"model": cfg.llm_model, "messages": messages, "stream": False,
                   "temperature": temperature, "max_tokens": max_tokens}
        with obs.observation(obs.current_stage() or "llm.chat", as_type="generation",
                             model=cfg.llm_model, input=messages,
                             metadata={"stage": obs.current_stage() or "chat",
                                       "temperature": temperature,
                                       "max_tokens": max_tokens}) as gen:
            data = self._post(f"{cfg.llm_base_url}/chat/completions",
                              self._auth(cfg.llm_api_key), payload,
                              self.timeout if self.timeout is not None else cfg.llm_timeout,
                              "chat")
            content = data["choices"][0]["message"]["content"]
            obs.update(gen, output=content, usage=usage_of(data))
            logger.debug(f"chat returned {len(content)} characters")
            return content


class OpenAIEmbedder(_Base):
    """Batch embeddings. Satisfies :class:`personos.online.write_path.Embedder`."""

    @property
    def available(self) -> bool:
        return bool(self.cfg.effective_embedding_api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix of shape (n, dim)."""
        cfg = self.cfg
        payload = {"model": cfg.embedding_model, "input": texts, "encoding_format": "float"}
        with obs.observation("llm.embed", as_type="embedding", model=cfg.embedding_model,
                             metadata={"n_texts": len(texts)}) as gen:
            data = self._post(f"{cfg.effective_embedding_base_url}/embeddings",
                              self._auth(cfg.effective_embedding_api_key), payload,
                              self.timeout if self.timeout is not None else cfg.io_timeout,
                              "embed")
            vecs = [np.asarray(item["embedding"], dtype=np.float32) for item in data["data"]]
            obs.update(gen, output={"n_vectors": len(vecs)}, usage=usage_of(data))
            logger.debug(f"embed {len(texts)} texts -> dim={vecs[0].shape[0] if vecs else 0}")
            return np.vstack(vecs)


class OpenAIReranker(_Base):
    """Cross-encoder reranking over the widely-shared `/rerank` shape.

    Request ``{model, query, documents}`` and response ``{results: [{index,
    relevance_score}]}`` is what Cohere, Jina, Voyage and text-embeddings-
    inference all speak, so one implementation covers the realistic options.

    Scores come back keyed by index and are re-expanded in input order, because
    a reranker is free to return them sorted — silently accepting that order
    would scramble the mapping between scores and documents.
    """

    @property
    def available(self) -> bool:
        return bool(self.cfg.rerank_api_key and self.cfg.rerank_model)

    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        if not documents:
            return []
        cfg = self.cfg
        text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
        payload = {"model": cfg.rerank_model, "query": text, "documents": documents}
        with obs.observation("llm.rerank", as_type="span", model=cfg.rerank_model,
                             input=query, metadata={"n_docs": len(documents)}) as gen:
            data = self._post(f"{cfg.rerank_base_url or cfg.llm_base_url}/rerank",
                              self._auth(cfg.rerank_api_key), payload,
                              self.timeout if self.timeout is not None else cfg.io_timeout,
                              "rerank")
            results = data.get("results") or data.get("data") or []
            by_index = {int(r["index"]): float(r.get("relevance_score", r.get("score", 0.0)))
                        for r in results}
            scores = [by_index.get(i, 0.0) for i in range(len(documents))]
            obs.update(gen, output={"top_score": max(scores) if scores else 0.0})
            return scores


class OpenAIMllm(_Base):
    """Vision: describe an image, used when a message carries one.

    Unavailable without a key, and the caller is expected to check ``available``
    and carry on — an un-described image is still stored as evidence, so text
    memory is unaffected.
    """

    # The caller supplies a PURPOSE built from the surrounding conversation, and
    # this prompt is what makes the model honour it. A generic "describe the
    # image" instruction produces whole-image narration, which buries the one
    # relevant fact under furniture and lighting and puts all of it into memory.
    # Ported verbatim apart from the sentinel: prompt text is pipeline
    # behaviour, not prose to be rewritten.
    _EMPTY = "NO_VISUAL_FACT"
    _SYSTEM = f"""You describe ONE image as factual text, guided by a PURPOSE.

# Iron rule: look with purpose, do not narrate the whole image
Report ONLY what serves the PURPOSE. Ignore everything unrelated. If the image shows nothing
relevant to the purpose, reply with exactly "{_EMPTY}". Never invent details you cannot see;
state only what is visually verifiable — text/signs on the image, countable objects, the scene,
who/what is shown.

# Output
Plain factual sentences (no markdown, no preamble, in the same language as the purpose). Prefer
concrete, verifiable statements over vague description. Quote on-image text literally when present."""

    @property
    def available(self) -> bool:
        return bool(self.cfg.mllm_api_key)

    def _endpoint(self) -> str:
        cfg = self.cfg
        return cfg.mllm_endpoint or f"{cfg.effective_mllm_base_url}/chat/completions"

    def look_image(self, image: bytes, purpose: str, *,
                   content_type: str = "image/jpeg") -> str:
        if not self.available:
            return ""
        cfg = self.cfg
        b64 = base64.b64encode(image).decode("ascii")
        payload = {
            "model": cfg.mllm_model,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"PURPOSE: {purpose}\n\nDescribe what in the image "
                             f"serves this purpose."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{content_type};base64,{b64}"}},
                ]},
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        with obs.observation("mllm.look_image", as_type="generation", model=cfg.mllm_model,
                             input=purpose) as gen:
            try:
                data = self._post(self._endpoint(), self._auth(cfg.mllm_api_key), payload,
                                  cfg.mllm_timeout, "look_image")
            except Exception as e:  # noqa: BLE001  vision is optional; never block a write
                logger.warning(f"look_image failed, continuing without it: {e}")
                return ""
            text = (data["choices"][0]["message"]["content"] or "").strip()
            obs.update(gen, output=text, usage=usage_of(data))
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return "" if not text or text == self._EMPTY else text


class NullMllm:
    """Stand-in used when no vision model is configured.

    Exists so callers never hold ``None`` and the existing
    ``getattr(mllm, "available", False)`` degradation check keeps working
    unchanged — the write path already does the right thing with an
    unavailable vision model, and that logic should not have to learn a second
    way of being absent.
    """

    available = False

    def look_image(self, image: bytes, purpose: str, *,
                   content_type: str = "image/jpeg") -> str:
        return ""
