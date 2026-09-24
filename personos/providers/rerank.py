"""HTTP rerankers: one transport, one dialect per vendor family.

Rerank endpoints disagree on wire shape — Cohere, Jina, Voyage and
text-embeddings-inference share one, Alibaba's DashScope speaks another — but
they agree on everything else: one POST, a query with documents in, relevance
scores keyed by document index out. So the transport (auth, retries, timeouts,
plain-language errors) lives once in :class:`BaseHTTPReranker` and a dialect
is three small hooks.

To speak to an endpoint nobody has tried: subclass, override the hooks, and
point ``PERSONOS_RERANKER_PROVIDER`` at the dotted path of your class — no
framework change needed (see ``providers/registry.load``).
"""

from __future__ import annotations

from .. import obs
from ..config import Config
from ..errors import ProviderError
from ._base import _Base, _expect
from ._http import usage_of


class BaseHTTPReranker(_Base):
    """Cross-encoder reranking over HTTP. Satisfies the ``Reranker`` protocol.

    Scores come back keyed by index and are re-expanded in input order, because
    a reranker is free to return them sorted — silently accepting that order
    would scramble the mapping between scores and documents.
    """

    def endpoint(self, cfg: Config) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def build_payload(self, cfg: Config, text: str, documents: list[str]) -> dict:
        """The request body. ``text`` is the query, instruction-prefixed when the
        caller passed one (the format instruction-tuned rerankers expect)."""
        raise NotImplementedError  # pragma: no cover - overridden

    def extract_scores(self, data: dict, n: int, url: str) -> list[float]:
        """Map a response body to ``n`` scores in input order."""
        raise NotImplementedError  # pragma: no cover - overridden

    @property
    def available(self) -> bool:
        return bool(self.cfg.rerank_api_key and self.cfg.rerank_model)

    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        if not documents:
            return []
        cfg = self.cfg
        text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
        with obs.observation("llm.rerank", as_type="span", model=cfg.rerank_model,
                             input=query, metadata={"n_docs": len(documents)}) as gen:
            url = self.endpoint(cfg)
            data = self._post(url, self._auth(cfg.rerank_api_key),
                              self.build_payload(cfg, text, documents),
                              self.timeout if self.timeout is not None else cfg.io_timeout,
                              "rerank")
            scores = self.extract_scores(data, len(documents), url)
            obs.update(gen, output={"top_score": max(scores) if scores else 0.0},
                       usage=usage_of(data))
            return scores

    @staticmethod
    def _scores_by_index(results: list[dict], n: int) -> list[float]:
        by_index = {int(r["index"]): float(r.get("relevance_score", r.get("score", 0.0)))
                    for r in results}
        return [by_index.get(i, 0.0) for i in range(n)]


class CohereReranker(BaseHTTPReranker):
    """The widely-shared ``/rerank`` shape: ``{model, query, documents}`` in,
    ``{results: [{index, relevance_score}]}`` out. Covers Cohere, Jina, Voyage,
    TEI and most OpenAI-compatible gateways."""

    def endpoint(self, cfg: Config) -> str:
        return f"{cfg.rerank_base_url or cfg.llm_base_url}/rerank"

    def build_payload(self, cfg: Config, text: str, documents: list[str]) -> dict:
        return {"model": cfg.rerank_model, "query": text, "documents": documents}

    def extract_scores(self, data: dict, n: int, url: str) -> list[float]:
        # key-presence, not truthiness: a legitimately empty result list is []
        results = data["results"] if "results" in data else data.get("data")
        if results is None:
            raise ProviderError(
                f"rerank: response from {url} has no 'results' field — the endpoint is "
                f"not speaking the Cohere rerank shape (wrong base URL, or a dialect this "
                f"class does not cover — subclass BaseHTTPReranker?). body: {str(data)[:300]}")
        return self._scores_by_index(results, n)


class DashScopeReranker(BaseHTTPReranker):
    """Alibaba Bailian's native rerank API (``gte-rerank`` / ``qwen3.*-rerank``).

    The body nests query and documents under ``input`` and the answer under
    ``output.results``. There is no path suffix convention, so
    ``PERSONOS_RERANK_BASE_URL`` is used verbatim as the full endpoint, e.g.
    ``https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank``.
    """

    def endpoint(self, cfg: Config) -> str:
        if not cfg.rerank_base_url:
            raise ProviderError(
                "rerank: DashScopeReranker uses PERSONOS_RERANK_BASE_URL verbatim as the "
                "full endpoint (no path suffix is appended), e.g. "
                "https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/"
                "text-rerank/text-rerank — it is currently unset")
        return cfg.rerank_base_url

    def build_payload(self, cfg: Config, text: str, documents: list[str]) -> dict:
        return {"model": cfg.rerank_model,
                "input": {"query": text, "documents": documents},
                "parameters": {"top_n": len(documents), "return_documents": False}}

    def extract_scores(self, data: dict, n: int, url: str) -> list[float]:
        _expect(data, "output", "rerank", url)
        results = data["output"].get("results")
        if results is None:
            raise ProviderError(
                f"rerank: response from {url} has no 'output.results' field. "
                f"body: {str(data)[:300]}")
        return self._scores_by_index(results, n)
