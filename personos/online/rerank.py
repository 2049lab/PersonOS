"""The R2 rerank station (fused architecture §3): oversampled first-stage retrieval (R1, 40) ->
cross-encoder rerank -> narrowing downstream (gate 5 / answering 20).

Input = (query, cell material): the metadata lead (dialogue time + topic) + episode — the same
convention as the R3/R5 answering material; atoms are the retrieval unit (used by R1 for locating)
and do not go into the rerank material. The dual-time format of the episode lets the cross-encoder
read temporal semantics like "valid until 2026-07". The instruction knob is not the default
"retrieve relevant passages" but a memory-QA criterion instead. If scoring fails it automatically
falls back to preserving the order, so R2 never blocks the main path.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from .retrieval import CellHit, cell_lead

# The rerank criterion for memory QA (not the default relevance wording; synonym rewriting is left to
# R0's expansion terms)
RERANK_INSTRUCTION = ("Judge whether the memory contains specific facts, entities, names, dates or "
                      "details that can answer the question")


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        """Score each document for relevance to the query (same length and same order as documents)."""
        ...


class NoopReranker:
    """Pass-through: returns strictly decreasing scores in input order, i.e. keeps the R1 fusion order
    (the default station; a real reranker can replace it at any time)."""

    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        return [float(len(documents) - i) for i in range(len(documents))]


class ScoringReranker:
    """Real reranking stage: wraps anything with ``.rerank(query, documents)``.

    The instruction is folded into the query using the Instruct/Query template
    that instruction-tuned rerankers expect.

    A failed scoring call degrades to pass-through order rather than
    propagating. Reranking improves an answer; it is never the reason there is
    no answer, so it must not be able to break the main path.
    """

    def __init__(self, score_api):
        self.api = score_api

    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        if not documents:
            return []
        text_1 = (f"Instruct: {instruction}\nQuery: {query}" if instruction else query)
        try:
            scores = self.api.rerank(text_1, documents)
            if len(scores) != len(documents):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for {len(documents)} documents")
            return scores
        except Exception as e:  # noqa: BLE001  never block the main path
            logger.warning(f"rerank scoring failed, falling back to fusion order: {e}")
            return NoopReranker().rerank(query, documents)


def cell_head(hit: CellHit) -> str:
    """The R2 rerank document: metadata lead (dialogue time + topic) + episode — the same convention
    as the R3/R5 material, just without the numbered separator line."""
    c = hit.cell
    return "\n".join([cell_lead(c), c.episode or "(no episode)"])


def rerank_cells(reranker: Reranker, query: str, hits: list[CellHit]) -> list[CellHit]:
    """Reorder cells by their rerank document (topic + episode); scores are written back to
    CellHit.rerank_score. With the noop reranker the R1 order is preserved.

    The sort is stable: cells tied on rerank score keep their previous relative order (the R1 fusion
    order acts as the secondary key).
    """
    if len(hits) <= 1:
        return hits   # a single hit needs no reordering, so rerank_score stays None (never ran)
    docs = [cell_head(h) for h in hits]
    scores = reranker.rerank(query, docs, instruction=RERANK_INSTRUCTION)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    order = sorted(range(len(hits)), key=lambda i: -scores[i])   # sorted is stable -> ties keep their original order
    out = [hits[i] for i in order]
    moved = sum(1 for i, h in enumerate(out) if h is not hits[i])
    logger.info(f"R2 rerank cells={len(out)} reordered={moved} top=«{out[0].cell.topic[:24]}» "
                f"score={out[0].rerank_score:.3f} q={query!r}")
    return out
