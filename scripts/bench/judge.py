"""The answerer and judge for the LoCoMo benchmark (mode A: memories + brief ->
a single answerer -> an LLM judge).

The protocol follows the Mem0 paper: the judge makes a binary decision
(CORRECT/WRONG, allowing semantic equivalence and unit conversion), and the
judge prompt descends from the MemGPT -> Mem0 lineage. The judge model is
written explicitly into the report at run time (a lesson from the benchmark
wars: if the judge and answerer are not declared, the numbers are marketing).

judge v2: relative times are converted deterministically into absolute dates
and windows by Python inside the harness, annotated in parentheses, and only
then handed to the judge, so the LLM compares rather than doing calendar
arithmetic in its head. In practice the model gets weekdays wrong when it does
the arithmetic itself (marking WRONG an answer of 7/18, which really is the
Tuesday before 7/20), while a v1 judge that only matched literal wording marked
correctly-converted answers wrong.

Two scoring conventions: answerer rule 7 is "trust the brief" (adopt the brief
when it already answers the question and the materials support it), and the
judge runs twice on each question — the Mem0 convention judges the answerer's
output, and the product convention judges the R5 answer directly.

Both functions take any ``ChatLLM`` — the judge model is a configuration
choice (``PERSONOS_JUDGE_*``), not a property of the protocol. The prompts and
the relative-time conversion below ARE the protocol: change them and the
numbers are no longer comparable across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from personos.online.llm import ChatLLM

# The mode-A answerer: treat what personos produced (the brief plus memories
# grouped by cell) as materials and produce a short English answer. Structurally
# identical to the Mem0/MIRIX "memory -> answerer -> answer" protocol.
# Answering discipline (a lightweight CoT modelled on EverMemOS's ANSWER_PROMPT,
# covering the answer shapes the LoCoMo question types need):
#   1. For list questions, gather everything first and then re-check for
#      omissions (one overlooked group is one wrong answer);
#   2. Use only the absolute dates annotated in the materials; never do calendar
#      arithmetic (the model gets weekdays wrong, the same lesson as judge v2);
#   3. Use the materials' own specific nouns, numbers and activity names; do not
#      generalize to a hypernym;
#   4. A single direct inference and a common-sense connection are allowed
#      (open-domain questions require world knowledge by definition);
#   5. Hedged materials get a hedged answer; questions of degree are concluded
#      according to the strength of the evidence.
# The answerer is written as a CoT procedure (modelled on EverMemOS's 7-step
# CoT) rather than the earlier list of rules. Error attribution over 226 genuine
# failures: 54 refusals / 46 list omissions / 33 reasoning errors / 28 wrong
# values or nouns / 12 wrong dates. The root cause was that, faced with a list
# of rules, the model takes the cheapest path (refuse, answer only one item, or
# guess in a single step). Hence the forced walk through every reasoning step
# and a much narrower exit for refusals.
_ANSWERER_SYS = (
    "You answer a question about a person using the memory materials below (a memory brief, plus "
    "memory items grouped by dialogue segment, each group headed by its date and topic) together "
    "with your general knowledge. Work through these steps IN ORDER, then give the answer.\n"
    "\n"
    "STEP 1 — Gather: read EVERY group. Collect every item relevant to the question; do not stop "
    "at the first hit.\n"
    "STEP 2 — Connect: link facts across different groups when the question needs it (who did "
    "what, where, why). The materials may state pieces in separate segments — join them.\n"
    "STEP 3 — Infer when needed: you MAY draw a direct, single-step inference from the materials "
    "plus common world knowledge (e.g. 'runs the shop personally, handling everything' → it is a "
    "small operation; 'adopts as a single parent' → currently single). Do NOT dismiss a "
    "reasonable inference as mere speculation. Only avoid chaining several speculative leaps.\n"
    "STEP 4 — Time: use absolute dates exactly as they appear; never do calendar arithmetic "
    "yourself (no weekday counting, no deriving 'a week after X'). If a value is hedged "
    "('about September 2023'), answer with that hedged value — do not refuse for lack of precision.\n"
    "STEP 5 — Verify (list/count questions): re-scan every group for any matching item you "
    "missed — one overlooked item is a wrong answer. Use the materials' own specific nouns, "
    "numbers and activity names verbatim; never generalize to a hypernym ('yoga and hiking' stays "
    "'yoga and hiking', not 'exercise'), and when several values exist pick the most specific. "
    "For COUNT questions ('how many ...'): list the distinct instances first, then count them; "
    "MERGE duplicates — the same event mentioned in several groups, or rephrased, counts ONCE — "
    "and exclude near-duplicates and plans/intentions that never happened. Do not answer 'at "
    "least N' — give the exact count of distinct instances you found.\n"
    "STEP 6 — Answer: concise, in English, absolute dates from the materials.\n"
    "\n"
    "REFUSAL IS A LAST RESORT. If the materials give ANY direct, indirect, or inferable evidence "
    "— even hedged, even needing a single-step inference — you MUST answer (state the evidential "
    "basis if indirect). Only reply exactly 'I don't have that information.' when the materials "
    "are genuinely silent AND no reasonable inference is possible. A yes/no question that the "
    "materials clearly imply (e.g. 'does he employ many people?' when he runs it alone) must be "
    "answered, not refused. If you can DESCRIBE the thing but hesitate to name it (e.g. you know "
    "which movie/book/place is meant but withhold the title), commit to the specific name — "
    "describing-but-refusing-to-name counts as a refusal and is not allowed.\n"
    "The MEMORY BRIEF is the memory system's own reviewed answer; when it already answers the "
    "question and the materials support it, adopt it (light rewording is fine) — do not "
    "second-guess it into a refusal or a different value. EXCEPTION: if the brief is itself a "
    "refusal or a negative conclusion (it claims the materials do not contain the answer), do NOT "
    "adopt it blindly — work through STEP 1-5 on the materials yourself; if you find any usable "
    "evidence the brief missed, you MUST answer from it."
)


def answer_mode_a(llm: ChatLLM, *, question: str, brief: str, mem_block: str) -> str:
    user = (f"MEMORY BRIEF\n{brief or '(empty)'}\n\n"
            f"MEMORY MATERIALS\n{mem_block or '(no memory items)'}\n\n"
            f"QUESTION\n{question}")
    return llm.chat([{"role": "system", "content": _ANSWERER_SYS},
                     {"role": "user", "content": user}], temperature=0.2, max_tokens=512).strip()


# -- Deterministic relative-time conversion (judge v2): the LLM only compares,
#    it never does calendar arithmetic --

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
_WEEKDAYS = {w: i for i, w in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
_DATE = (rf"(?:(?P<d>\d{{1,2}})\s*(?P<mon>{_MONTHS})"                # 25 May 2023 / 7November(H2)
         rf"|(?P<mon2>{_MONTHS})\s*(?P<d2>\d{{1,2}}))"               # October 20, 2023 / November7
         rf"\s*,?\s+(?P<y>\d{{4}})")


def _last_weekday_before(d: date, target: int) -> date:
    """The most recent `target` weekday strictly before `d` (if the anchor day
    itself falls on that weekday, go back a full week)."""
    return d - timedelta(days=((d.weekday() - target) % 7) or 7)


def _fmt_rng(a: date, b: date) -> str:
    return f"{a:%Y-%m-%d} to {b:%Y-%m-%d}"


def normalize_relative_times(text: str) -> str:
    """Annotate relative-time expressions in the gold answer or the prediction
    in place with their deterministic conversion, in parentheses.

    Covers the phrasings actually observed in LoCoMo: "the Sunday before
    25 May 2023", "the week before X", "the weekend before X", "two weekends
    before X", "the week of X".
    Getting the conversion right is the whole point: left to itself, the model
    works out the weekday incorrectly and marks 7/18 (the Tuesday before 7/20)
    as wrong.
    """
    if not text:
        return text

    def _d(m) -> date:
        mon = m.group("mon") or m.group("mon2")
        day = m.group("d") or m.group("d2")
        return date(int(m.group("y")), _MONTH_NUM[mon], int(day))

    # the Sunday before <D> -> a single day
    def _wd_before(m):
        d = _d(m)
        return m.group(0) + f" (= {_last_weekday_before(d, _WEEKDAYS[m.group(1).lower()]):%Y-%m-%d})"
    _WD_ALT = "|".join(_WEEKDAYS)
    text = re.sub(rf"\bthe\s+({_WD_ALT})\s+before\s+{_DATE}", _wd_before, text, flags=re.I)

    # the week before <D> -> a 7-day window ending the day before
    def _week_before(m):
        d = _d(m)
        return m.group(0) + f" (= {_fmt_rng(d - timedelta(days=7), d - timedelta(days=1))})"
    text = re.sub(rf"\bthe\s+week\s+before\s+{_DATE}", _week_before, text, flags=re.I)

    # (two|three|...) weekends before <D> -> the two days of that weekend
    def _weekends_before(m):
        d = _d(m)
        n = _NUM_WORDS.get(m.group(1).lower(), 2)
        sun = _last_weekday_before(d, 6)
        sun -= timedelta(days=7 * (n - 1))
        return m.group(0) + f" (= {_fmt_rng(sun - timedelta(days=1), sun)})"
    text = re.sub(rf"\b({'|'.join(_NUM_WORDS)})\s+weekends\s+before\s+{_DATE}",
                  _weekends_before, text, flags=re.I)

    # the weekend before <D> -> the most recent Saturday/Sunday
    def _weekend_before(m):
        d = _d(m)
        sun = _last_weekday_before(d, 6)
        return m.group(0) + f" (= {_fmt_rng(sun - timedelta(days=1), sun)})"
    text = re.sub(rf"\bthe\s+weekend\s+before\s+{_DATE}", _weekend_before, text, flags=re.I)

    # the week of <D> -> the calendar week it falls in (Monday to Sunday)
    def _week_of(m):
        d = _d(m)
        mon = d - timedelta(days=d.weekday())
        return m.group(0) + f" (= {_fmt_rng(mon, mon + timedelta(days=6))})"
    text = re.sub(rf"\bthe\s+week\s+of\s+{_DATE}", _week_of, text, flags=re.I)

    return text


# The Mem0-style binary judge (CORRECT/WRONG scoring inherited from MemGPT,
# allowing semantic equivalence; the superset rule keeps list questions fair).
# Misjudgement attribution over 48 cases: faced with a long answer, several
# listed items, or an opening disclaimer, the model would mark WRONG without
# comparing patiently. The remedy is to make it locate the part of the answer
# corresponding to each gold item before deciding, and to state explicitly that
# "longer", "lists more" and "more specific" are not reasons to mark it wrong.
_JUDGE_SYS = (
    "You are an impartial judge evaluating an answer to a question. "
    "Given the question, the ground-truth answer, and the model's answer, decide whether "
    "the model's answer is factually correct compared to the ground truth.\n"
    "\n"
    "HOW TO JUDGE (do this before deciding): the model's answer may be long, list several items, "
    "or open with a hedge/disclaimer. Do NOT skim and reject. First LOCATE, inside the model's "
    "answer, the part(s) that correspond to each ground-truth item; judge on that, ignoring "
    "surrounding extra text. Length, extra items, disclaimers, or reasoning shown do NOT make an "
    "answer wrong.\n"
    "\n"
    "- Semantic equivalence counts as CORRECT (paraphrase, different units/format, a MORE "
    "SPECIFIC value that is an instance of the gold — 'Nintendo Wii' for 'nintendo console', "
    "'grow his fanbase' for 'wants to be more popular', a more specific date that matches).\n"
    "- Hedged but matching values count as CORRECT: 'around 2022' vs '2022', "
    "'about a week before X' vs the actual date, 'roughly N' vs 'N' — approximation or "
    "extra certainty language does NOT make it wrong as long as the value matches.\n"
    "- Multi-part ground truths (lists of items/places/occurrences): an answer that contains "
    "ALL the gold items is CORRECT even if it lists additional items (a superset is correct); "
    "check each gold item against the WHOLE answer before deciding. Only a genuinely MISSING gold "
    "item makes it WRONG. But when the question asks for an exact count, a wrong number is WRONG "
    "however it is phrased.\n"
    "- A yes/no ground truth: the model's answer is CORRECT if it clearly conveys the same "
    "yes/no stance, even via explanation rather than the literal word.\n"
    "- Relative-time conversion (judge v2): ground truths often phrase dates relative to an "
    "anchor ('the week before 9 June 2023', 'the Sunday before 25 May 2023', "
    "'two weekends before 17 July 2023'). Such expressions are annotated with the precomputed "
    "absolute date or window in parentheses, e.g. 'the Sunday before 25 May 2023 (= 2023-05-21)' "
    "— USE that annotation instead of redoing calendar arithmetic yourself. An answer that gives "
    "an absolute date inside the annotated window is CORRECT even if phrased differently; "
    "identical phrasing is NOT required. If a relative expression carries no annotation, "
    "convert it conservatively (a ±few-days window) rather than requiring exact phrasing.\n"
    "- Missing a key part of a multi-part ground truth, a wrong value/date/name, or "
    "contradicting the ground truth counts as WRONG.\n"
    "- Extra unrelated details or quoting the source do not make it wrong as long as "
    "the core matches.\n"
    "Output EXACTLY one word: CORRECT or WRONG."
)


@dataclass
class Verdict:
    ok: bool
    raw: str


def judge(llm: ChatLLM, *, question: str, gold: str, prediction: str) -> Verdict:
    # Convert relative times deterministically on both sides: common in the
    # gold answers, and the answerer occasionally emits "the weekend before X" too.
    gold = normalize_relative_times(str(gold))
    prediction = normalize_relative_times(str(prediction))
    user = (f"QUESTION\n{question}\n\nGROUND TRUTH\n{gold}\n\n"
            f"MODEL ANSWER\n{prediction}")
    raw = llm.chat([{"role": "system", "content": _JUDGE_SYS},
                    {"role": "user", "content": user}], temperature=0.0, max_tokens=64).strip()
    m = re.search(r"CORRECT|WRONG", raw.upper())
    return Verdict(ok=bool(m and m.group(0) == "CORRECT"), raw=raw)
