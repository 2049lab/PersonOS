"""召回编排单测:快链一条龙 R0→R1(atom池)→单元组装→R2→R5草稿→R3'核判(重排后),
role 路由 LLM 断言各工位该调/不该调;双层处置(defect 重答一次/insufficient 直升深轨)与枚举扩面。

run_recall 入参传显式依赖(不耦合 rt/FastAPI),这里全部用测试替身。
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.app.recall_flow import run_recall
from personos.storage.cell_store import CellStore

from .test_retrieval import Env, TableEmbedder, _v

_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


class RoutingLLM:
    """按 system prompt 里的角色词路由到预设响应;响应给 list 则按序出队(重答/再核判两轮)。

    注意路由次序:核判 prompt 里也出现 "answerer" 字样(指正写给谁看),
    必须先匹配 "answer reviewer" 再匹配 "answerer"。
    """

    def __init__(self, *, rewrite=None, review=None, answer=None):
        def _seq(v):
            return list(v) if isinstance(v, list) else [v]
        self.responses = {
            "query preprocessor": _seq(rewrite or [
                '{"resolved":"画展筹备","subject":"","expansions":[],"time_start":null,'
                '"time_end":null,"domains":[]}']),
            "answer reviewer": _seq(review or ['{"verdict":"ok","critique":""}']),
            "answerer": _seq(answer or ['{"answer":"展期在 2026-09。","cells":["m1"]}']),
        }
        self.calls = {k: 0 for k in self.responses}
        self.last_user: dict[str, str] = {}

    def chat(self, messages, temperature=0.3, max_tokens=2048):
        sys_prompt = messages[0]["content"]
        for key, seq in self.responses.items():
            if key in sys_prompt:
                if not seq:
                    raise AssertionError(f"角色 {key} 的预设响应耗尽")
                self.calls[key] += 1
                self.last_user[key] = messages[-1]["content"]
                return seq.pop(0)
        raise AssertionError(f"收到未知角色的 prompt: {sys_prompt[:50]!r}")


def _env_with_two_cells(db):
    """两个 cell:第一个与检索面同向(强),第二个弱——R1 序可预期。"""
    env = Env(db)
    c1 = env.add_cell(topic="画展", episode="Caroline 筹备画展,展期 2026-09。",
                      atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="跑步", episode="开始跑步训练。",
                 atoms=[{"text": "跑三公里", "vec": _v(0.3, 0.95, 0, 0)}])
    return env, c1


def test_full_fast_chain(db, evidence_store):
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})    # resolved 可 embed,扩展词无
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls == {"query preprocessor": 1, "answer reviewer": 1, "answerer": 1}   # 各工位恰好一次
    assert o.rw.resolved == "画展筹备"                          # R0 产物贯通全链
    assert [ah.atom.memcell_id for ah in o.hits] == [c1.id, o.hits[1].atom.memcell_id]   # R1 池:画展在前
    assert o.ranked[0].cell.id == c1.id and o.ranked[0].rerank_score is not None   # R2 Noop 保序+落分
    assert o.reviews[-1].verdict == "ok"                        # 核判通过
    assert o.draft is not None and o.draft.answer == "展期在 2026-09。"  # 草稿=首答
    assert not o.retried and not o.escalated
    assert o.ans.answer == "展期在 2026-09。" and o.ans.cited_cells == [c1.id]      # c1 → 真实 cell id


def test_r2_order_feeds_answer_and_review(db, evidence_store):
    """R2 精排序必须贯通到 R5/R3' 材料序(真 reranker 不白跑):
    倒序 reranker 把弱格提到第一 → 作答/核判材料的 m1 块就是弱格。"""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM()

    class ReverseReranker:
        def rerank(self, query, documents, *, instruction=""):
            return [float(i) for i in range(len(documents))]   # 分随位次递增 → 整体倒序

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T,
                   reranker=ReverseReranker())
    assert o.ranked[0].cell.topic == "跑步"                      # 精排确实把弱格提到第一
    m1 = llm.last_user["answerer"].split("━━━ m2")[0]            # 作答材料的 m1 块
    assert "开始跑步训练" in m1 and "画展" not in m1
    m1r = llm.last_user["answer reviewer"].split("━━━ m2")[0]    # 核判同款材料序
    assert "开始跑步训练" in m1r


def test_rewrite_off_skips_r0(db, evidence_store):
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展什么时候": _v(1, 0, 0, 0)})  # 原始 query 本身可 embed
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, rewrite=False)
    assert llm.calls["query preprocessor"] == 0                     # R0 没跑
    assert o.rw.resolved == "画展什么时候" and o.rw.subject == ""
    assert o.hits and o.hits[0].atom.memcell_id == c1.id      # 直接用原 query 检索
    assert not hasattr(o.rw, "question_type")                 # qtype 已拆(D-C1),字段不存在


def test_empty_store_short_circuits_no_answer_no_review(db, evidence_store):
    """空检索短路:不作答也不核判,直接判空;无答案不空手而归(客观交代,确定性拼装)。"""
    env = Env(db)                                            # 空库:无 cell 无 atom
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.hits == [] and o.ranked == [] and o.reviews == []
    assert llm.calls["answerer"] == 0 and llm.calls["answer reviewer"] == 0   # 两工位都不调
    assert o.escalated                                       # auto:升深轨(测试替身下深轨崩,回退)
    assert "结论:记忆库中没有足以回答该问题的相关记忆" in o.ans.answer
    assert "可能原因" in o.ans.answer and "检索范围" in o.ans.answer


def test_defect_retries_once_then_ok(db, evidence_store):
    """双层处置·答案缺陷:带指正重答一次,再核判 ok → 采纳重答案,不升深轨。"""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"answer_defect","critique":"材料 m1 有展期,草稿漏了——补上"}',
                '{"verdict":"ok","critique":""}'],
        answer=['{"answer":"不知道。","cells":[]}',
                '{"answer":"展期在 2026-09。","cells":["m1"]}'])

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls["answerer"] == 2 and llm.calls["answer reviewer"] == 2   # 重答+再核判各一次
    assert o.retried and not o.escalated
    assert o.draft.answer == "不知道。"                       # 首份草稿留档
    assert o.ans.answer == "展期在 2026-09。"                 # 终答=重答案
    assert "JUDGE FEEDBACK" in llm.last_user["answerer"]      # 指正喂进了重答 prompt
    assert "材料 m1 有展期" in llm.last_user["answerer"]
    assert [r.verdict for r in o.reviews] == ["answer_defect", "ok"]


def test_defect_retry_still_bad_escalates(db, evidence_store):
    """重答后仍缺陷 → 升深轨(测试替身深轨崩,回退保留重答案),escalated 如实。"""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"answer_defect","critique":"还差日期"}',
                '{"verdict":"answer_defect","critique":"仍不完整"}'],
        answer=['{"answer":"答一。","cells":[]}', '{"answer":"答二。","cells":[]}'])

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.retried and o.escalated
    assert o.ans.answer == "答二。"                           # 深轨崩 → 保留快链重答案


def test_insufficient_escalates_directly(db, evidence_store):
    """双层处置·材料不足:不重答直接升深轨;指正当缺口方向进无答案交代。"""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"insufficient_material","critique":"只有画展筹备,没有具体展期"}'],
        answer=['{"answer":"","cells":[]}'])                  # 草稿拒答(空)

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls["answerer"] == 1 and not o.retried       # 不重答
    assert o.escalated                                        # 直接深轨(替身崩→回退)
    assert "缺口:只有画展筹备,没有具体展期" in o.ans.answer    # 指正进客观交代


def test_fast_mode_insufficient_keeps_draft(db, evidence_store):
    """fast 模式无深轨:材料不足 → 保留草稿(R5 自己的交代已诚实),不标升级。"""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"insufficient_material","critique":"材料缺展期"}'],
        answer=['{"answer":"记忆里只有筹备,没有展期。","cells":["m1"]}'])

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="fast")
    assert not o.retried and not o.escalated
    assert o.ans.answer == "记忆里只有筹备,没有展期。"


def test_boundary_note_is_universal(db, evidence_store):
    """通用面(D-C6/D-C10):材料 = 池 top-30 atom 所在 cell 去重,不分题型;
    池外还有 atom → 边界提示进作答与核判同款材料;池装得下 → 无提示。"""
    env = Env(db)
    for i in range(32):                                      # 32 格全同向 → 前 30 进池,2 个池外
        env.add_cell(topic=f"事项{i}", episode=f"第 {i} 件事。",
                     atoms=[{"text": f"条目{i}", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"都做了什么": _v(1, 0, 0, 0)})
    rw = ('{"resolved":"都做了什么","subject":"","expansions":[],"time_start":null,'
          '"time_end":null,"domains":[]}')

    llm = RoutingLLM(rewrite=rw)
    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="都做了什么", now_dt=_T, mode="fast")
    assert "BOUNDARY" in llm.last_user["answerer"]            # 32 atom > 30 → 边界提示
    assert "2 more matching atoms exist beyond the shown materials" in llm.last_user["answerer"]
    assert "BOUNDARY" in llm.last_user["answer reviewer"]     # 核判看同款(含边界)
    assert llm.last_user["answerer"].count("━━━ m") == 30     # 材料恰好 30 格(池内 atom 的格)
    assert o.asm is not None and o.asm.n_pool == 30           # 透视字段贯通


def test_no_boundary_when_pool_covers_all(db, evidence_store):
    """池 atom ≤ 30:池装得下全集 → 无边界提示。"""
    env = Env(db)
    for i in range(3):
        env.add_cell(topic=f"少{i}", episode=f"第 {i} 件事。",
                     atoms=[{"text": f"少条{i}", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"都做了什么": _v(1, 0, 0, 0)})
    llm = RoutingLLM()
    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="都做了什么", now_dt=_T, mode="fast")
    assert "BOUNDARY" not in llm.last_user["answerer"]
    assert o.asm is not None and o.asm.n_pool == 3 and not o.asm.boundary


# —— 视觉改写接线:只在"图 + 依赖"都给了才触发,且结果 query 要真的喂给 R0 ——

def test_no_image_skips_visual_rewrite(db, evidence_store):
    """不带图 → 完全不碰视觉改写,纯文本召回逐字节不变(out.vis 为 None)。"""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.vis is None


def test_image_without_deps_is_a_noop(db, evidence_store):
    """给了图但没装配 visual_deps(纯文本 pod)→ 不改写也不报错。"""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, image=b"\x89PNG")
    assert o.vis is None


def test_visual_rewrite_output_feeds_r0(db, evidence_store, monkeypatch):
    """视觉改写的结果必须成为 R0 的输入 —— 否则认出了人也白认。"""
    from personos.app import recall_flow
    from personos.online.visual_query import VisualRewrite

    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    seen = {}

    def _fake_enrich(deps, *, query, image, content_type, history, scenario, now_dt):
        seen["history"], seen["now_dt"] = history, now_dt
        return VisualRewrite(query="李四的画展什么时候", faces=1,
                             matched=[{"character_id": "char_A", "name": "李四"}])

    monkeypatch.setattr(recall_flow, "enrich_query_with_image", _fake_enrich)
    llm = RoutingLLM()
    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="他的画展什么时候", now_dt=_T,
                   image=b"img", visual_deps=object())

    assert o.vis is not None and o.vis.matched[0]["name"] == "李四"
    assert "李四的画展什么时候" in llm.last_user["query preprocessor"], \
        "改写后的 query 没进 R0"
    assert seen["history"] is not None, "视觉改写必须拿到历史上下文"
    assert seen["now_dt"] == _T, "视觉改写必须拿到当前时间锚(与 R0 同口径)"


def test_visual_block_surfaces_what_the_image_resolved(db, evidence_store, monkeypatch):
    """带图召回要把「认出了谁 / 问题被改写成什么」回显给调用方。

    没有这个,调用方分不清两种完全不同的情况:
      A 认出了人但记忆里确实没有 → 该让用户换个问法
      B 根本没认出人,按原问题检索   → 该让用户换张清楚的照片
    两者的响应此前长得一模一样。(我自己写回归时也因此只能用"答案里有没有人名"做
    间接判据,结果被 "没有关于 Bob 的任何信息" 骗出过假阳性。)
    """
    from personos.app import recall_flow
    from personos.online.visual_query import VisualRewrite

    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    monkeypatch.setattr(
        recall_flow, "enrich_query_with_image",
        lambda deps, **k: VisualRewrite(
            query="李四的画展什么时候", faces=2,
            matched=[{"face_index": 0, "person": "p1",
                      "character_id": "char_A", "name": "李四"}]))
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="他的画展什么时候", now_dt=_T,
                   image=b"img", visual_deps=object())

    # 服务层据此组装 visual 块(此处直接验 outcome,HTTP 层同源)
    assert o.vis is not None
    assert o.vis.faces == 2
    assert o.vis.query == "李四的画展什么时候"
    assert o.vis.matched[0]["character_id"] == "char_A"


def test_no_visual_block_without_image(db, evidence_store):
    """不带图 → outcome 里没有 vis,响应也就不会出现 visual 块(纯增量,老调用不变)。"""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.vis is None
