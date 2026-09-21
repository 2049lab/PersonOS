"""R2 精排工位(融合架构 §3):过采样粗排(R1,40)→ cross-encoder 精排 → 下游收窄(闸门 5/作答 20)。

输入 = (query, cell 材料):元信息头(对话时间+topic)+ episode——与 R3/R5 作答材料同口径;
atoms 是检索单元(R1 定位用),不进精排材料。episode 的双时间格式让 cross-encoder
读得出"有效至 2026-07"这类时间语义。instruction 旋钮不用默认"retrieve relevant passages",
换成记忆问答判据。打分挂了自动退保序,R2 永不阻塞主链路。
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from .retrieval import CellHit, cell_lead

# 记忆问答的精排判据(不用默认相关性措辞;同义改写留给 R0 扩展词)
RERANK_INSTRUCTION = ("Judge whether the memory contains specific facts, entities, names, dates or "
                      "details that can answer the question")


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], *, instruction: str = "") -> list[float]:
        """对每个 document 打一个与 query 相关性的分(与 documents 等长、同序)。"""
        ...


class NoopReranker:
    """透传:按输入序返回严格递减分数 = 保持 R1 融合序(默认工位,真 reranker 随时替换)。"""

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
    """R2 精排文档:元信息头(对话时间+topic)+ episode——与 R3/R5 材料同口径,只少个编号分隔行。"""
    c = hit.cell
    return "\n".join([cell_lead(c), c.episode or "(no episode)"])


def rerank_cells(reranker: Reranker, query: str, hits: list[CellHit]) -> list[CellHit]:
    """按精排文档(topic+episode)对 cell 重排;分数写回 CellHit.rerank_score。Noop 时保持 R1 序。

    排序稳定:rerank 分并列的 cell 保持原有相对次序(R1 融合序作为次级序)。
    """
    if len(hits) <= 1:
        return hits   # 单条无需重排,rerank_score 保持 None(未跑)
    docs = [cell_head(h) for h in hits]
    scores = reranker.rerank(query, docs, instruction=RERANK_INSTRUCTION)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    order = sorted(range(len(hits)), key=lambda i: -scores[i])   # sorted 稳定 → 并列保原序
    out = [hits[i] for i in order]
    moved = sum(1 for i, h in enumerate(out) if h is not hits[i])
    logger.info(f"R2 精排 cells={len(out)} 换位={moved} top=«{out[0].cell.topic[:24]}» "
                f"score={out[0].rerank_score:.3f} q={query!r}")
    return out
