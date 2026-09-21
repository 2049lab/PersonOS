"""R2 精排单测:NoopReranker 保序降分 / ScoringReranker(前缀+容错) / 伪 reranker 重排 / 材料头格式。"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.online.rerank import ScoringReranker, NoopReranker, cell_head, rerank_cells
from personos.online.retrieval import AtomHit, CellHit, QueryRewrite


def _hit(cid: str, topic="t") -> CellHit:
    from personos.models import MemCell, MemoryAtom
    c = MemCell(id=cid, topic=topic, episode="叙事不该出现在材料头里")
    a = MemoryAtom(id=f"{cid}_a", memcell_id=cid, text="原子内容")
    return CellHit(cell=c, score=0.1, best_sim=0.5, atoms=[AtomHit(atom=a, similarity=0.5)])


def test_noop_keeps_order_with_descending_scores():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = rerank_cells(NoopReranker(), "q", hits)
    assert [h.cell.id for h in out] == ["a", "b", "c"]        # 保序
    scores = [h.rerank_score for h in out]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1]   # 严格递减


def test_fake_reranker_resorts_and_stamps_score():
    hits = [_hit("a"), _hit("b"), _hit("c")]

    class FakeReranker:
        def rerank(self, query, documents, *, instruction=""):
            assert query == "画展哪天"
            assert len(documents) == 3
            assert instruction                              # 必须带判据指令
            return [0.1, 0.9, 0.5]                          # b 最相关

    out = rerank_cells(FakeReranker(), "画展哪天", hits)
    assert [h.cell.id for h in out] == ["b", "c", "a"]       # 按精比分重排
    assert out[0].rerank_score == 0.9 and out[2].rerank_score == 0.1


def test_single_hit_short_circuits_without_reranker():
    class Boom:
        def rerank(self, *a, **k):  # pragma: no cover
            raise AssertionError("单条不该调 rerank")

    hits = [_hit("a")]
    out = rerank_cells(Boom(), "q", hits)
    assert len(out) == 1 and out[0].rerank_score is None      # 未跑


def test_maas_reranker_prefixes_instruction_and_maps_scores():
    calls = []

    class FakeScoreApi:
        def rerank(self, query, documents):
            calls.append((query, documents))
            return [0.2, 0.8]

    r = ScoringReranker(FakeScoreApi())
    scores = r.rerank("画展哪天", ["甲", "乙"], instruction="记忆判据")
    assert scores == [0.2, 0.8]                          # 与文档等长同序
    assert calls[0][0] == "Instruct: 记忆判据\nQuery: 画展哪天"   # qwen3 Instruct/Query 模板
    assert calls[0][1] == ["甲", "乙"]
    assert r.rerank("q", [], instruction="i") == []       # 空输入不调 API


def test_maas_reranker_falls_back_to_order_on_failure():
    class BoomApi:
        def rerank(self, query, documents):
            raise RuntimeError("网关抖动")

    docs = ["a", "b", "c"]
    out = ScoringReranker(BoomApi()).rerank("q", docs, instruction="i")
    assert out == NoopReranker().rerank("q", docs)        # 保序透传,R2 不阻塞主链路

    class ShortApi:   # 返回条数与文档不等长 → 同样保序兜底
        def rerank(self, query, documents):
            return [0.5]

    assert ScoringReranker(ShortApi()).rerank("q", docs) == NoopReranker().rerank("q", docs)
