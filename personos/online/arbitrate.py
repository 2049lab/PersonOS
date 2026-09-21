"""R3' adjudication (fused architecture §5, decision 1/2): after R5 produces a draft answer, check
that draft against the same materials, claim by claim.

Deliberately split into two calls instead of one: adjudication only checks and never answers (it
matches each claim against the materials and gives no impression score), while answering is allowed
to make inferences — those two temperaments cannot live in one prompt. The verdict drives a
two-tier disposition: ok -> take the draft; answer_defect -> re-answer once with the critique
attached, and only escalate to the deep track if it is still defective; insufficient_material ->
go straight to the deep track (the critique becomes the gap to chase).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger

from .llm import ChatLLM, chat_json
from .retrieval import _MOST_RECENT_RULE, _R5_ORDER_ENV, _order_hits_for_answer, CellHit, MemoryAnswer, cell_block

_REVIEW_SYS = (
    "# Role\n"
    "You are the answer reviewer of a personal memory system: a DRAFT ANSWER was produced from the "
    "given MEMORY MATERIALS; check the draft against those materials, item by item. You are a "
    "checker, not a scorer — no impressions, no style judgment.\n\n"
    "# How to read the materials\n"
    "Each material block is one topic cell, opened by a \"━━━ mN ━━━\" separator line; the next line is a "
    "header with the dialogue date and topic, followed by the segment narrative (episode) — the only "
    "basis for judgment.\n\n"
    "# Check the draft (all hard)\n"
    "1. Grounded: walk the draft's claims one by one; each must trace to some material block "
    "(numbers, dates, names, attribution included — a wrong number is an ungrounded claim). The "
    "draft's own citations are hints, not proof.\n"
    "2. Complete: read the question first and judge ONLY what it asks. List/enumeration question → "
    "count the distinct items the materials actually contain, then count the items the draft lists "
    "— fewer = missing items. Single-fact question → the draft answers that fact, not a neighbor "
    "fact; materials containing OTHER facts the question never asked for is NOT a defect (e.g. "
    "asked \"when did X happen\" and one date is recorded → giving that date is complete even if "
    "the materials mention unrelated events too).\n"
    "3. Current: " + _MOST_RECENT_RULE + " The draft must report that most-recent statement as the "
    "current state; an explicit correction always wins.\n"
    "4. Refusals: defined BROADLY — any draft that fails to deliver the fact the question asks "
    "for counts as a refusal, however it is phrased — 'not found', 'the materials do not state/"
    "contain/record X', an empty draft, an evasive non-answer, or describing the thing while "
    "withholding its name. For every refusal: verify against the materials YOURSELF — re-scan "
    "every block for the asked fact. The materials contain the answer → answer_defect (name the "
    "block). They truly lack it → insufficient_material. On a specific-fact question (a name, a "
    "date, a title, a place), default to answer_defect unless your own re-scan of ALL blocks "
    "confirms absence. A draft's own claim that it 'checked every block m1..mN' is a checklist "
    "for your spot-check, never proof — do not trust it without verifying.\n\n"
    "# Verdict (pick one)\n"
    "- ok: every check passes. Judge facts objectively — if the draft's claims are grounded and it "
    "delivers what the materials support, pass it. Wording, style, and tense framing are not "
    "defects: asked \"where is it held?\" about an event that has not started, a draft answering "
    "\"planned at X on <date>\" is correct, not a defect. BREVITY IS NOT A DEFECT: a short draft "
    "that directly delivers the asked fact passes — never demand fuller coverage, background, or "
    "extra items the question did not ask for; exact-wording attribution differences (\"announced\" "
    "vs \"framed it as\") are not defects either.\n"
    "- answer_defect: the materials suffice but the draft is wrong / incomplete / an unsupported "
    "refusal — fixable from the materials alone. Reject ONLY for one of these three: (i) it "
    "answers the wrong / a neighbor fact, (ii) it contradicts the materials or itself (a wrong "
    "number, date, name, attribution), (iii) it misses the core fact the question asks (or items "
    "of an enumeration). Minor unsupported garnish that does not change the answer to the "
    "question is not a defect.\n"
    "- insufficient_material: the materials themselves lack the core of the answer — no re-draft "
    "can fix this; deeper search is needed.\n\n"
    "# critique (answer_defect only; else \"\")\n"
    "Write the fix instruction the answerer will see — concrete, pointing at blocks (\"materials "
    "contain 4 locations (m2, m5, m7, m9); the draft lists 2 — add the other two\"; \"m6 states the "
    "current value; the draft reports the old one\").\n\n"
    "# Calibration (hard)\n"
    "- Unsure whether a CORE claim is grounded → answer_defect (one cheap re-answer catches it; "
    "a wrong 'ok' ships the defect). This conservatism covers factual doubts only — never "
    "convert phrasing, hedging, tense, or brevity into a defect.\n"
    "- Unsure whether coverage is \"full enough\" → ok: doubts about thoroughness resolve to "
    "pass, only a MISSING core fact is a defect.\n"
    "- Unsure between answer_defect and insufficient_material → answer_defect (re-answer is "
    "cheaper than deep search; if the re-answer still fails it escalates anyway).\n\n"
    "# Output (JSON only)\n"
    '{"verdict":"ok|answer_defect|insufficient_material","critique":"..."}'
)


@dataclass
class ReviewResult:
    verdict: str                        # ok | answer_defect | insufficient_material
    critique: str = ""                  # fix instruction (fed to R5 for the re-answer on defect; used as the deep-track gap direction on insufficient)
    system: str = ""
    user: str = ""
    raw: str = ""


def review_answer(
    llm: ChatLLM, *, query: str, draft: MemoryAnswer, hits: list[CellHit],
    resolved: str = "", subject: str = "", boundary: str = "",
) -> ReviewResult:
    """Adjudicate the R5 draft (draft + the same materials, same renderer / window / boundary as answering).

    The boundary matches the answering side: on an enumeration question the reviewer has to know
    that some cells were left unexpanded, otherwise it mistakes the truncated view for the full set.
    Material ordering also matches answering (_R5_ORDER_ENV): answering, adjudication and re-answering
    all see the same window and the same set of mN handles, so the block numbers in the critique
    still line up during the re-answer round.
    Parse failure -> conservatively return answer_defect: one re-answer is a cheap fallback, whereas
    a mistaken "ok" lets the defect through.
    """
    order = os.environ.get(_R5_ORDER_ENV, "").strip() or "relevance"
    hits = _order_hits_for_answer(hits, order)
    handles = [f"m{i + 1}" for i in range(len(hits))]   # same material window as answering, same set of mN handles
    block = "\n\n".join(cell_block(h, hd) for h, hd in zip(hits, handles)) or "(no materials)"
    bnd = f"\n\nBOUNDARY\n{boundary}" if boundary else ""   # same header as the answering side, so adjudication lines up with it
    rd = (f"\n\nResolved question (references resolved)\n{resolved}" if resolved and resolved != query else "")
    subj = f"\n\nQUESTION SUBJECT\n{subject or '(not determined)'}"
    draft_txt = (draft.answer or "").strip() or "(the draft is EMPTY — treat as a refusal)"
    user = (f"USER QUESTION\n{query}{rd}{subj}\n\nDRAFT ANSWER\n{draft_txt}"
            f"\n\nMEMORY MATERIALS\n{block}{bnd}")
    messages = [{"role": "system", "content": _REVIEW_SYS}, {"role": "user", "content": user}]
    try:
        obj, raw = chat_json(llm, messages, max_tokens=800, stage="review_answer")
        verdict = str(obj.get("verdict") or "").strip()
        if verdict not in ("ok", "answer_defect", "insufficient_material"):
            raise ValueError(f"unknown verdict: {verdict!r}")
        critique = str(obj.get("critique") or "").strip()
        if verdict == "ok":
            critique = ""                       # ok contract: no critique
        elif verdict == "insufficient_material" and not critique:
            # Hit in production: when the draft has already honestly refused, the reviewer often
            # writes no critique. But the critique is the gap direction handed to the deep track,
            # and an empty one loses that direction — so synthesize one objective gap statement from
            # the question, giving the deep track something to aim at when it re-scans the store.
            critique = (f"none of the {len(hits)} retrieved cells contains the core answer to "
                        f"the question ({query!r}); deeper search is needed")
        res = ReviewResult(verdict=verdict, critique=critique,
                           system=_REVIEW_SYS, user=user, raw=raw)
    except Exception as e:
        logger.warning(f"review parse failed, conservatively ruling answer_defect: {e}")
        res = ReviewResult(verdict="answer_defect",
                           critique="review parse failure; one conservative re-answer",
                           system=_REVIEW_SYS, user=user,
                           raw=getattr(e, "raw", "") or str(e))
    logger.info(f"R3' review verdict={res.verdict} cells={len(hits)} q={query!r} "
                f"critique={res.critique!r}")
    return res
