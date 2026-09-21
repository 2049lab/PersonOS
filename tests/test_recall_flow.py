"""Unit tests for recall orchestration: the whole fast path, R0 -> R1 (atom pool) -> unit
assembly -> R2 -> R5 draft -> R3' adjudication after rerank. A role-routing LLM double asserts
which stations should and should not be called. Also covers the two-tier handling (a defect
triggers one re-answer, insufficient material escalates straight to the deep track) and the
broadened enumeration surface.

run_recall takes its dependencies explicitly rather than through the runtime or FastAPI, so
everything here is a test double.
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.online.recall_flow import run_recall
from personos.storage.cell_store import CellStore

from .test_retrieval import Env, TableEmbedder, _v

_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


class RoutingLLM:
    """Routes to a canned response by the role keyword in the system prompt; if a response is
    given as a list it is dequeued in order, which covers the re-answer and second
    adjudication rounds.

    Mind the routing order: the adjudication prompt also contains the word "answerer" (it says
    who the critique is addressed to), so "answer reviewer" must be matched before "answerer".
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
                    raise AssertionError(f"canned responses for role {key} are exhausted")
                self.calls[key] += 1
                self.last_user[key] = messages[-1]["content"]
                return seq.pop(0)
        raise AssertionError(f"prompt received for an unknown role: {sys_prompt[:50]!r}")


def _env_with_two_cells(db):
    """Two cells: the first points the same way as the query vector (a strong match) and the
    second is weak, so the R1 ordering is predictable."""
    env = Env(db)
    c1 = env.add_cell(topic="画展", episode="Caroline 筹备画展,展期 2026-09。",
                      atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="跑步", episode="开始跑步训练。",
                 atoms=[{"text": "跑三公里", "vec": _v(0.3, 0.95, 0, 0)}])
    return env, c1


def test_full_fast_chain(db, evidence_store):
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})    # the resolved query is embeddable, the expansions are not
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls == {"query preprocessor": 1, "answer reviewer": 1, "answerer": 1}   # each station called exactly once
    assert o.rw.resolved == "画展筹备"                          # the R0 output flows through the whole chain
    assert [ah.atom.memcell_id for ah in o.hits] == [c1.id, o.hits[1].atom.memcell_id]   # R1 pool: the strong cell comes first
    assert o.ranked[0].cell.id == c1.id and o.ranked[0].rerank_score is not None   # the no-op R2 keeps the order and still records a score
    assert o.reviews[-1].verdict == "ok"                        # adjudication passed
    assert o.draft is not None and o.draft.answer == "展期在 2026-09。"  # the draft is the first answer
    assert not o.retried and not o.escalated
    assert o.ans.answer == "展期在 2026-09。" and o.ans.cited_cells == [c1.id]      # the short label resolves to the real cell id


def test_r2_order_feeds_answer_and_review(db, evidence_store):
    """The R2 rerank order must carry through to the material order given to R5 and R3',
    otherwise a real reranker would run for nothing. A reversing reranker promotes the weak
    cell to first place, so the m1 block in both the answering and adjudication materials is
    the weak cell."""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM()

    class ReverseReranker:
        def rerank(self, query, documents, *, instruction=""):
            return [float(i) for i in range(len(documents))]   # score grows with position, which reverses the whole list

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T,
                   reranker=ReverseReranker())
    assert o.ranked[0].cell.topic == "跑步"                      # the rerank really did promote the weak cell to first
    m1 = llm.last_user["answerer"].split("━━━ m2")[0]            # the m1 block of the answering materials
    assert "开始跑步训练" in m1 and "画展" not in m1
    m1r = llm.last_user["answer reviewer"].split("━━━ m2")[0]    # adjudication sees the same material order
    assert "开始跑步训练" in m1r


def test_rewrite_off_skips_r0(db, evidence_store):
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展什么时候": _v(1, 0, 0, 0)})  # the raw query is embeddable on its own
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, rewrite=False)
    assert llm.calls["query preprocessor"] == 0                     # R0 did not run
    assert o.rw.resolved == "画展什么时候" and o.rw.subject == ""
    assert o.hits and o.hits[0].atom.memcell_id == c1.id      # retrieval uses the raw query directly
    assert not hasattr(o.rw, "question_type")                 # qtype was removed (D-C1), so the field is gone


def test_empty_store_short_circuits_no_answer_no_review(db, evidence_store):
    """An empty retrieval short-circuits: neither answering nor adjudication runs, the result
    is declared empty right away. Even with no answer the caller does not leave empty-handed,
    it gets a factual, deterministically assembled explanation."""
    env = Env(db)                                            # empty store: no cells, no atoms
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM()

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.hits == [] and o.ranked == [] and o.reviews == []
    assert llm.calls["answerer"] == 0 and llm.calls["answer reviewer"] == 0   # neither station is called
    assert o.escalated                                       # auto mode escalates to the deep track, which fails under the doubles and falls back
    assert "No memory relevant enough to answer this was found" in o.ans.answer
    assert "Possible reason" in o.ans.answer and "Searched:" in o.ans.answer


def test_defect_retries_once_then_ok(db, evidence_store):
    """Two-tier handling, answer defect: re-answer once with the critique attached, and if the
    second adjudication says ok the re-answer is adopted without escalating to the deep
    track."""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"answer_defect","critique":"材料 m1 有展期,草稿漏了——补上"}',
                '{"verdict":"ok","critique":""}'],
        answer=['{"answer":"不知道。","cells":[]}',
                '{"answer":"展期在 2026-09。","cells":["m1"]}'])

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls["answerer"] == 2 and llm.calls["answer reviewer"] == 2   # one re-answer and one second adjudication
    assert o.retried and not o.escalated
    assert o.draft.answer == "不知道。"                       # the first draft is kept on record
    assert o.ans.answer == "展期在 2026-09。"                 # the final answer is the re-answer
    assert "JUDGE FEEDBACK" in llm.last_user["answerer"]      # the critique was fed into the re-answer prompt
    assert "材料 m1 有展期" in llm.last_user["answerer"]
    assert [r.verdict for r in o.reviews] == ["answer_defect", "ok"]


def test_defect_retry_still_bad_escalates(db, evidence_store):
    """If the re-answer is still defective the flow escalates to the deep track, and
    escalated reports that faithfully. Under the test doubles the deep track fails, so the
    fallback keeps the re-answer."""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"answer_defect","critique":"还差日期"}',
                '{"verdict":"answer_defect","critique":"仍不完整"}'],
        answer=['{"answer":"答一。","cells":[]}', '{"answer":"答二。","cells":[]}'])

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.retried and o.escalated
    assert o.ans.answer == "答二。"                           # the deep track failed, so the fast path's re-answer is kept


def test_insufficient_escalates_directly(db, evidence_store):
    """Two-tier handling, insufficient material: no re-answer, escalate straight to the deep
    track, and the critique goes into the no-answer explanation as the direction of the
    gap."""
    env, c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    llm = RoutingLLM(
        review=['{"verdict":"insufficient_material","critique":"只有画展筹备,没有具体展期"}'],
        answer=['{"answer":"","cells":[]}'])                  # the draft declines to answer (empty)

    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert llm.calls["answerer"] == 1 and not o.retried       # no re-answer
    assert o.escalated                                        # straight to the deep track, which fails under the double and falls back
    assert "Gap: 只有画展筹备,没有具体展期" in o.ans.answer    # the critique lands in the factual explanation


def test_fast_mode_insufficient_keeps_draft(db, evidence_store):
    """fast mode has no deep track, so insufficient material simply keeps the draft, whose own
    R5 explanation is already honest, and does not mark the result as escalated."""
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
    """The general surface (D-C6/D-C10): the materials are the deduplicated cells of the top
    30 pooled atoms, regardless of question type. If atoms remain beyond the pool, the
    boundary note appears in the identical materials given to both answering and adjudication;
    if the pool covers everything there is no note."""
    env = Env(db)
    for i in range(32):                                      # 32 cells all point the same way -> the top 30 enter the pool, 2 stay beyond
        env.add_cell(topic=f"事项{i}", episode=f"第 {i} 件事。",
                     atoms=[{"text": f"条目{i}", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"都做了什么": _v(1, 0, 0, 0)})
    rw = ('{"resolved":"都做了什么","subject":"","expansions":[],"time_start":null,'
          '"time_end":null,"domains":[]}')

    llm = RoutingLLM(rewrite=rw)
    o = run_recall(llm, emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="都做了什么", now_dt=_T, mode="fast")
    assert "BOUNDARY" in llm.last_user["answerer"]            # 32 atoms > 30 -> boundary note
    assert "2 more matching atoms exist beyond the shown materials" in llm.last_user["answerer"]
    assert "BOUNDARY" in llm.last_user["answer reviewer"]     # adjudication sees the same materials, boundary included
    assert llm.last_user["answerer"].count("━━━ m") == 30     # exactly 30 cells of materials, one per pooled atom
    assert o.asm is not None and o.asm.n_pool == 30           # the introspection fields carry through


def test_no_boundary_when_pool_covers_all(db, evidence_store):
    """With at most 30 pooled atoms the pool holds everything, so there is no boundary
    note."""
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


# -- Visual rewrite wiring: it fires only when both an image and its dependencies are given,
#    and the resulting query must really be fed to R0 --

def test_no_image_skips_visual_rewrite(db, evidence_store):
    """Without an image the visual rewrite is never touched and text-only recall is unchanged
    byte for byte, with out.vis left as None."""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.vis is None


def test_image_without_deps_is_a_noop(db, evidence_store):
    """An image passed to a deployment that has no visual_deps wired up (a text-only
    deployment) is neither rewritten nor treated as an error."""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, image=b"\x89PNG")
    assert o.vis is None


def test_visual_rewrite_output_feeds_r0(db, evidence_store, monkeypatch):
    """The visual rewrite's output must become R0's input, otherwise recognising the person in
    the picture achieves nothing."""
    from personos.online import recall_flow
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
        "the rewritten query never reached R0"
    assert seen["history"] is not None, "the visual rewrite must receive the conversation history"
    assert seen["now_dt"] == _T, "the visual rewrite must receive the current time anchor, the same one R0 uses"


def test_visual_block_surfaces_what_the_image_resolved(db, evidence_store, monkeypatch):
    """Image-based recall has to echo back who was recognised and what the question was
    rewritten into.

    Without that the caller cannot tell apart two completely different situations:
      A: the person was recognised but memory genuinely holds nothing, so the user should
         rephrase the question;
      B: nobody was recognised at all and retrieval ran on the original question, so the user
         should supply a clearer photo.
    The responses for the two used to look identical. Writing regression tests, that left only
    the indirect criterion of whether a person's name appeared in the answer, which produced a
    false positive off an answer that read "no information about Bob at all".
    """
    from personos.online import recall_flow
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

    # the service layer builds the visual block from these fields; the outcome is checked
    # directly here, and the HTTP layer reads from the same source
    assert o.vis is not None
    assert o.vis.faces == 2
    assert o.vis.query == "李四的画展什么时候"
    assert o.vis.matched[0]["character_id"] == "char_A"


def test_no_visual_block_without_image(db, evidence_store):
    """Without an image the outcome carries no vis, so no visual block appears in the
    response. The feature is purely additive and existing callers are unaffected."""
    env, _c1 = _env_with_two_cells(db)
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})
    o = run_recall(RoutingLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T)
    assert o.vis is None
