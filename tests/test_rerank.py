"""R2 精排单测:NoopReranker 保序降分 / MaasReranker(前缀+容错) / 伪 reranker 重排 / 材料头格式。"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.online.rerank import MaasReranker, NoopReranker, cell_head, rerank_cells
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

    r = MaasReranker(FakeScoreApi())
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
    out = MaasReranker(BoomApi()).rerank("q", docs, instruction="i")
    assert out == NoopReranker().rerank("q", docs)        # 保序透传,R2 不阻塞主链路

    class ShortApi:   # 返回条数与文档不等长 → 同样保序兜底
        def rerank(self, query, documents):
            return [0.5]

    assert MaasReranker(ShortApi()).rerank("q", docs) == NoopReranker().rerank("q", docs)


def test_maas_client_rerank_maps_index_and_uses_score_endpoint(monkeypatch):
    from personos.clients.maas import MaasClient
    from personos.config import settings

    m = MaasClient()
    seen = {}

    def fake_post(path, headers, payload, timeout):
        seen.update(path=path, headers=headers, payload=payload, timeout=timeout)
        return {"data": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]}   # 乱序返回

    monkeypatch.setattr(m, "_post", fake_post)
    assert m.rerank("q", ["甲", "乙"]) == [0.1, 0.9]      # 按 index 对齐文档序
    assert seen["timeout"] == settings.maas_io_timeout    # rerank 走紧档
    assert seen["path"] == "/score"
    assert seen["payload"]["model"] == settings.rerank_model
    assert seen["payload"]["text_2"] == ["甲", "乙"]
    assert seen["headers"]["api-key"] == (settings.rerank_key or settings.chat_key)
    assert m.rerank("q", []) == []                        # 空输入不打网关


def test_maas_client_chat_timeout_tiering_and_override(monkeypatch):
    """chat 走宽档(Settings),显式 timeout 覆盖全档。"""
    from personos.clients.maas import MaasClient
    from personos.config import settings

    m = MaasClient()
    seen = {}

    def fake_post(path, headers, payload, timeout):
        seen.update(timeout=timeout)
        return {"choices": [{"message": {"content": "ok"}}],
                "data": [{"index": 0, "score": 0.5}]}   # chat/rerank 两种形状都给

    monkeypatch.setattr(m, "_post", fake_post)
    m.chat([{"role": "user", "content": "hi"}])
    assert seen["timeout"] == settings.maas_chat_timeout    # 宽档(非流式长生成)

    m2 = MaasClient(timeout=180.0)
    monkeypatch.setattr(m2, "_post", fake_post)
    m2.chat([{"role": "user", "content": "hi"}])
    assert seen["timeout"] == 180.0                          # 显式覆盖
    m2.rerank("q", ["甲"])
    assert seen["timeout"] == 180.0                          # 覆盖对 io 档同样生效


def test_cell_head_is_lead_plus_episode_no_atoms():
    """R2 精排文档 = 元信息头(对话时间+topic)+ episode;atoms 不进精排材料。"""
    from personos.models import MemCell, MemoryAtom
    day = datetime(2026, 8, 18, tzinfo=timezone.utc)
    c = MemCell(id="cell_x", topic="画展", episode="Caroline 在筹备画展,展期 2026-09。",
                t_start=day, t_end=day)
    a1 = MemoryAtom(id="a1", memcell_id="cell_x", text="展期 2026-09", occurrence_time=day)
    a2 = MemoryAtom(id="a2", memcell_id="cell_x", text="场地租好了", occurrence_time=None)
    h = CellHit(cell=c, score=0.0, best_sim=0.0,
                atoms=[AtomHit(atom=a1, similarity=0.9), AtomHit(atom=a2, similarity=0.4)])
    head = cell_head(h)
    lines = head.splitlines()
    assert lines[0] == "[dialogue 2026-08-18 | topic: 画展]"   # 元信息头:固定英文框架,与 episode 区分
    assert "Caroline 在筹备画展" in head                       # episode 进精排文档
    assert "场地租好了" not in head                            # atom 文本不进
