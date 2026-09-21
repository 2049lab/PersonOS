"""R2 rerank unit tests: NoopReranker keeps the order with descending scores, ScoringReranker handles
the instruction prefix and failures, a fake reranker really reorders, and the material header format."""

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
    assert [h.cell.id for h in out] == ["a", "b", "c"]        # order preserved
    scores = [h.rerank_score for h in out]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1]   # strictly decreasing


def test_fake_reranker_resorts_and_stamps_score():
    hits = [_hit("a"), _hit("b"), _hit("c")]

    class FakeReranker:
        def rerank(self, query, documents, *, instruction=""):
            assert query == "画展哪天"
            assert len(documents) == 3
            assert instruction                              # the judging instruction must be present
            return [0.1, 0.9, 0.5]                          # b is the most relevant

    out = rerank_cells(FakeReranker(), "画展哪天", hits)
    assert [h.cell.id for h in out] == ["b", "c", "a"]       # reordered by the rerank score
    assert out[0].rerank_score == 0.9 and out[2].rerank_score == 0.1


def test_single_hit_short_circuits_without_reranker():
    class Boom:
        def rerank(self, *a, **k):  # pragma: no cover
            raise AssertionError("rerank must not be called for a single hit")

    hits = [_hit("a")]
    out = rerank_cells(Boom(), "q", hits)
    assert len(out) == 1 and out[0].rerank_score is None      # never ran


def test_maas_reranker_prefixes_instruction_and_maps_scores():
    calls = []

    class FakeScoreApi:
        def rerank(self, query, documents):
            calls.append((query, documents))
            return [0.2, 0.8]

    r = ScoringReranker(FakeScoreApi())
    scores = r.rerank("画展哪天", ["甲", "乙"], instruction="记忆判据")
    assert scores == [0.2, 0.8]                          # same length and same order as the documents
    assert calls[0][0] == "Instruct: 记忆判据\nQuery: 画展哪天"   # the qwen3 Instruct/Query template
    assert calls[0][1] == ["甲", "乙"]
    assert r.rerank("q", [], instruction="i") == []       # empty input does not call the API


def test_maas_reranker_falls_back_to_order_on_failure():
    class BoomApi:
        def rerank(self, query, documents):
            raise RuntimeError("gateway blip")

    docs = ["a", "b", "c"]
    out = ScoringReranker(BoomApi()).rerank("q", docs, instruction="i")
    assert out == NoopReranker().rerank("q", docs)        # pass through in order, so R2 never blocks the main path

    class ShortApi:   # returns fewer scores than documents, which also falls back to the original order
        def rerank(self, query, documents):
            return [0.5]

    assert ScoringReranker(ShortApi()).rerank("q", docs) == NoopReranker().rerank("q", docs)
