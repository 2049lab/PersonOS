"""LoCoMo 评测的 answerer 与 judge(模式 A:memories+brief → 统一 answerer → LLM judge)。

协议对齐 Mem0 论文:judge 为二元判定(CORRECT/WRONG,允许语义等价/单位换算),
judge prompt 沿袭 MemGPT→Mem0 一系;运行时把 judge 模型显式写进报告(评测战争教训:
judge/answerer 不声明,数字就是营销)。

judge v2(2026-08-26):相对时间在 harness 里用 Python 确定性换算成绝对日期/窗口
(括号内标注后交给 judge),LLM 只做比对不做日历心算——实测 M3 心算星期会错
(7/18 明明是 7/20 前的周二,判 WRONG);v1 只认字面则把换算对的答案判错。

双口径(S4 决策 3,2026-09-03):answerer 规则 7 = 简报信任(简报已答且材料支持时
采用简报);同一题 judge 跑两遍——Mem0 口径判 answerer 产物,产品口径直判 R5 答案。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from personos.clients.minimax import MinimaxClient

# 模式 A answerer:把 personos 的产物(brief + 按 cell 分组的 memories)当材料,生成英文简答。
# 与 Mem0/MIRIX 的"记忆 → answerer → 答案"协议同构。
# 作答纪律(对标 EverMemOS ANSWER_PROMPT 的轻量 CoT,覆盖 LoCoMo 题型的作答形态):
#   ① 清单题先全量收集再自查漏项(一个漏看的组就是一道错题);
#   ② 日期只用材料标注的绝对日期,不做日历心算(M3 实测心算星期会错,judge v2 同源教训);
#   ③ 用材料自带的具体名词/数字/活动名,不上位泛化;
#   ④ 允许一步直接推理与常识连接(cat3 开放域题按定义需要世界知识推断);
#   ⑤ hedged 材料 hedged 答;程度题按证据强度下结论。
# answerer:CoT 流程式(对标 EverMemOS 7 步 CoT,替代原规则罗列)。错题归因(2026-09-08,
# 226 道真错):拒答 54/清单漏项 46/推理错 33/值名词错 28/日期错 12 —— 病根是 M3 面对
# 规则罗列时挑最省力路径(拒答/只答一项/一步蒙)。改为强制走完思考步骤,并收紧拒答出口。
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


def answer_mode_a(llm: MinimaxClient, *, question: str, brief: str, mem_block: str) -> str:
    user = (f"MEMORY BRIEF\n{brief or '(empty)'}\n\n"
            f"MEMORY MATERIALS\n{mem_block or '(no memory items)'}\n\n"
            f"QUESTION\n{question}")
    return llm.chat([{"role": "system", "content": _ANSWERER_SYS},
                     {"role": "user", "content": user}], temperature=0.2, max_tokens=512).strip()


# —— 相对时间确定性换算(judge v2):LLM 只比对,不做日历心算 ——

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
_WEEKDAYS = {w: i for i, w in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}
_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
_DATE = (rf"(?:(?P<d>\d{{1,2}})\s*(?P<mon>{_MONTHS})"                # 25 May 2023 / 7November(H2)
         rf"|(?P<mon2>{_MONTHS})\s*(?P<d2>\d{{1,2}}))"               # October 20, 2023 / November7
         rf"\s*,?\s+(?P<y>\d{{4}})")


def _last_weekday_before(d: date, target: int) -> date:
    """严格早于 d 的最近一个 target 星期几(锚点当天同星期也取前一周)。"""
    return d - timedelta(days=((d.weekday() - target) % 7) or 7)


def _fmt_rng(a: date, b: date) -> str:
    return f"{a:%Y-%m-%d} to {b:%Y-%m-%d}"


def normalize_relative_times(text: str) -> str:
    """把 gold/prediction 里的相对时间表达就地补上确定性换算(括号标注)。

    覆盖 LoCoMo 实测句式:"the Sunday before 25 May 2023" / "the week before X" /
    "the weekend before X" / "two weekends before X" / "the week of X"。
    换算错不了是关键——实测 M3 自己心算星期会把 7/18(7/20 前的周二)判错。
    """
    if not text:
        return text

    def _d(m) -> date:
        mon = m.group("mon") or m.group("mon2")
        day = m.group("d") or m.group("d2")
        return date(int(m.group("y")), _MONTH_NUM[mon], int(day))

    # the Sunday before <D> → 单日
    def _wd_before(m):
        d = _d(m)
        return m.group(0) + f" (= {_last_weekday_before(d, _WEEKDAYS[m.group(1).lower()]):%Y-%m-%d})"
    _WD_ALT = "|".join(_WEEKDAYS)
    text = re.sub(rf"\bthe\s+({_WD_ALT})\s+before\s+{_DATE}", _wd_before, text, flags=re.I)

    # the week before <D> → 7 天窗(含端点,止于前一天)
    def _week_before(m):
        d = _d(m)
        return m.group(0) + f" (= {_fmt_rng(d - timedelta(days=7), d - timedelta(days=1))})"
    text = re.sub(rf"\bthe\s+week\s+before\s+{_DATE}", _week_before, text, flags=re.I)

    # (two|three|…) weekends before <D> → 对应周末两天窗
    def _weekends_before(m):
        d = _d(m)
        n = _NUM_WORDS.get(m.group(1).lower(), 2)
        sun = _last_weekday_before(d, 6)
        sun -= timedelta(days=7 * (n - 1))
        return m.group(0) + f" (= {_fmt_rng(sun - timedelta(days=1), sun)})"
    text = re.sub(rf"\b({'|'.join(_NUM_WORDS)})\s+weekends\s+before\s+{_DATE}",
                  _weekends_before, text, flags=re.I)

    # the weekend before <D> → 最近一个周六/周日
    def _weekend_before(m):
        d = _d(m)
        sun = _last_weekday_before(d, 6)
        return m.group(0) + f" (= {_fmt_rng(sun - timedelta(days=1), sun)})"
    text = re.sub(rf"\bthe\s+weekend\s+before\s+{_DATE}", _weekend_before, text, flags=re.I)

    # the week of <D> → 所在日历周(周一至周日)
    def _week_of(m):
        d = _d(m)
        mon = d - timedelta(days=d.weekday())
        return m.group(0) + f" (= {_fmt_rng(mon, mon + timedelta(days=6))})"
    text = re.sub(rf"\bthe\s+week\s+of\s+{_DATE}", _week_of, text, flags=re.I)

    return text


# Mem0 式二元 judge(沿袭 MemGPT 的 CORRECT/WRONG 判分,允许语义等价;超集判定补清单题公平性)。
# 误判归因(2026-09-08,48 道):M3 面对长答案/多列/带免责声明时不耐心比对就判 WRONG。
# 对策:判前先在 ans 里定位对应 gold 的部分,并显式声明"更长/多列/更具体"不是判错理由。
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


def judge(llm: MinimaxClient, *, question: str, gold: str, prediction: str) -> Verdict:
    # 相对时间两侧都确定性换算(gold 常见;answerer 偶尔也输出 "the weekend before X")
    gold = normalize_relative_times(str(gold))
    prediction = normalize_relative_times(str(prediction))
    user = (f"QUESTION\n{question}\n\nGROUND TRUTH\n{gold}\n\n"
            f"MODEL ANSWER\n{prediction}")
    raw = llm.chat([{"role": "system", "content": _JUDGE_SYS},
                    {"role": "user", "content": user}], temperature=0.0, max_tokens=64).strip()
    m = re.search(r"CORRECT|WRONG", raw.upper())
    return Verdict(ok=bool(m and m.group(0) == "CORRECT"), raw=raw)
