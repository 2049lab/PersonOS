"""R3' adjudication unit tests: the three verdict tiers / the ok contract / conservative
answer_defect on parse failure / rendering of materials and the draft.

Adjudication and answering are deliberately split (two calls, two personalities); this file
tests only the adjudicator itself: it checks the draft against the same materials item by item.
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.models import MemCell, MemoryAtom
from personos.online.arbitrate import review_answer
from personos.online.retrieval import AtomHit, CellHit, MemoryAnswer

from .fakes import FakeLLM


def _hit(topic="画展", episode="Caroline 筹备画展", atom_text="展期定在 2026-09",
         when=None) -> CellHit:
    c = MemCell(topic=topic, episode=episode, t_start=datetime(2026, 8, 20, tzinfo=timezone.utc),
                t_end=datetime(2026, 8, 21, tzinfo=timezone.utc))
    a = MemoryAtom(memcell_id=c.id, text=atom_text, occurrence_time=when)
    return CellHit(cell=c, score=0.02, best_sim=0.8, atoms=[AtomHit(atom=a, similarity=0.8)])


def test_verdict_and_critique_parse():
    llm = FakeLLM(['{"verdict":"answer_defect","critique":"材料共 4 处地点(c2,c5,c7,c9),'
                   '草稿只列 2 处——补全"}'])
    res = review_answer(llm, query="在哪些地方练瑜伽", draft=MemoryAnswer(answer="两处。"),
                        hits=[_hit()])
    assert res.verdict == "answer_defect"
    assert "补全" in res.critique


def test_ok_forces_empty_critique():
    """The ok contract: even if the model insists on emitting a critique, clear it (ok means
    there is nothing to correct)."""
    llm = FakeLLM(['{"verdict":"ok","critique":"顺手提一句措辞可以更好"}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "ok" and res.critique == ""


def test_insufficient_empty_critique_gets_synthesized_gap():
    """Learned the hard way against the live model: when the draft already refuses honestly,
    the adjudicator often writes no critique at all. But the critique is what tells the deep
    track which gap to chase, so an empty one loses that direction — synthesize an objective
    gap statement from the question plus the cell count instead."""
    llm = FakeLLM(['{"verdict":"insufficient_material","critique":""}'])
    res = review_answer(llm, query="我的驾照什么时候到期?", draft=MemoryAnswer(answer="无法确定。"),
                        hits=[_hit(), _hit()])
    assert res.verdict == "insufficient_material"
    # Carries the question and the cell count, so the deep track has a direction.
    assert "驾照" in res.critique and "2" in res.critique
    # The model wrote its own gap statement -> keep it verbatim, do not overwrite.
    llm2 = FakeLLM(['{"verdict":"insufficient_material","critique":"缺驾照到期日"}'])
    res2 = review_answer(llm2, query="驾照", draft=MemoryAnswer(answer="无法确定。"), hits=[_hit()])
    assert res2.critique == "缺驾照到期日"


def test_unknown_verdict_degrades_to_conservative_defect():
    llm = FakeLLM(['{"verdict":"maybe","critique":""}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "answer_defect"
    # Conservative critique: re-answering once is the cheap fallback.
    assert "re-answer" in res.critique


def test_parse_failure_keeps_raw_and_defect():
    llm = FakeLLM(["核判器跑偏了,不是 JSON"])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="答"), hits=[_hit()])
    assert res.verdict == "answer_defect"
    # Traceability: keep the model's raw output.
    assert "核判器跑偏" in res.raw


def test_prompt_carries_draft_materials_and_subject():
    """The user prompt must contain: a DRAFT ANSWER section, the material blocks (numbered
    separator line + header + episode, with atoms excluded), and the subject of the question."""
    llm = FakeLLM(['{"verdict":"ok"}'])
    res = review_answer(llm, query="画展哪天", draft=MemoryAnswer(answer="2026 年 9 月。"),
                        hits=[_hit(atom_text="展期 2026-09")],
                        resolved="Caroline 的画展哪天", subject="Caroline")
    u = res.user
    # The draft goes into the prompt; it is the thing being checked.
    assert "DRAFT ANSWER\n2026 年 9 月。" in u
    # Hard numbered separator line (mN, matching the fast-chain window).
    assert "━━━ m1 ━━━" in u
    assert "[dialogue 2026-08-20 to 2026-08-21 | topic: 画展]" in u
    assert "Caroline 筹备画展" in u                         # the episode is the main material
    assert "展期" not in u                                  # atom text does not enter the materials
    assert "QUESTION SUBJECT\nCaroline" in u                # the ownership anchor enters the materials
    # Listed separately only when the resolved question differs from the raw query.
    assert "Resolved question (references resolved)" in u


def test_empty_draft_marked_as_refusal():
    """An empty draft is explicitly labelled a refusal in the prompt, so the adjudicator checks
    the materials along the refusal path."""
    llm = FakeLLM(['{"verdict":"ok"}'])
    res = review_answer(llm, query="q", draft=MemoryAnswer(answer="  "), hits=[_hit()])
    assert "DRAFT ANSWER\n(the draft is EMPTY — treat as a refusal)" in res.user


def test_review_prompt_carries_itemized_checks():
    """Guard over the four itemized checks, the objectivity clause, and the two-sided
    conservative calibration (which only applies to factual doubt): stops a later prompt
    rewrite from quietly dropping a clause."""
    from personos.online.arbitrate import _REVIEW_SYS
    for clause in ("1. Grounded:", "2. Complete:", "3. Current:", "4. Refusals:",
                   "Judge facts objectively", "not a defect",
                   # Loosening: terse but on-point is not a defect.
                   "BREVITY IS NOT A DEFECT",
                   "never convert phrasing, hedging, tense, or brevity into a defect",
                   # Factual doubt stays conservative, but only for core claims.
                   "Unsure whether a CORE claim is grounded",
                   # Doubt about richness goes the other way: let it through.
                   'Unsure whether coverage is "full enough" → ok',
                   "Unsure between answer_defect and insufficient_material"):
        assert clause in _REVIEW_SYS
