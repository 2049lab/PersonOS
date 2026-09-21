"""Generate a large, realistic, first-person multi-session corpus that covers a
wide range of memory phenomena.

Two stages (parallel generation, yet coherent across periods):
  A. One serial call: produce the *persona bible + timeline outline* — stable
     facts plus N periods (each with a theme, dates and a few sessions) — with
     memory phenomena deliberately **planted** into the outline (updates,
     corrections, preference shifts, goals, expiries, cross-period callbacks,
     contradictions, one-off events, recurring habits, noise), along with a set
     of after-the-fact probe questions. This outline is the "ground-truth
     spine" that pins down cross-period consistency.
  B. N parallel calls: each period is expanded into realistic human-assistant
     dialogue from the persona, a summary of every period's arc, and that
     period's own detail. Since each call depends only on the immutable outline
     and never on text another call generated, they can run in parallel and
     still get cross-period corrections ("what I told you last time was
     actually Y") right.

Authentication goes through OpenAIChatLLM, which reads PERSONOS_LLM_API_KEY
from .env; no key is hardcoded here.

Run:      python -m scripts.generate_corpus --periods 10 --months 18
Output:   data/persona_gen/corpus.json (plus raw/ responses for debugging)
Next:     load corpus.json into PersonOS session by session; see
          scripts/load_corpus.py.

Note: the prompts below are written in Chinese on purpose — they generate a
Chinese-language persona corpus. They are model input, so translating them
would change what this script produces.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from personos.providers.openai_compat import OpenAIChatLLM
from personos.logging_setup import setup_logging

OUT_DEFAULT = "data/persona_gen/corpus.json"

# Catalogue of memory phenomena: shown to the model, and also used to report
# how well the generated corpus covers them.
PHENOMENA = [
    "new(新事实)", "update(属性更新)", "supersede(整体顶替:搬家/换工作/换设备/关系变化)",
    "correction(纠错:之前说错后面更正)", "preference_shift(偏好演化)",
    "goal(目标→兑现/放弃)", "expiry(时效到期:会员/租约/证件)",
    "cross_ref(跨会话埋点:前段随口提,后段才相关)", "dispute(前后矛盾/存疑)",
    "one_off(一次性事件)", "recurring(周期性习惯)", "noise(闲聊噪声,不该成记忆)",
    "emotion(情绪起伏)", "relationship(人际关系及其演变)",
]

_PHEN_TEXT = "\n".join(f"  - {p}" for p in PHENOMENA)

# ============ Stage A: persona bible + timeline outline ============
_A_SYS = "你在为一个【个人长期记忆系统】造评测语料。你要设定一个真实可信的中国都市普通人,并规划其一段人生时间线。输出必须是严格 JSON,不要任何解释文字。"

def _a_user(periods: int, months: int) -> str:
    return f"""设定一个真实、有血有肉的中国都市个体(第一人称"我"将在后续对话里跟一个 AI 助手日常聊天),规划其约 {months} 个月、{periods} 个时间段的人生时间线。

要求:
1. persona(人设圣经):姓名、年龄、性别、城市、职业、性格、家庭与社交圈(列几个具体的人:姓名/关系/特征)、长期爱好、日常作息习惯。要具体、可信、有个性,不要脸谱化。
2. stable_facts:5~10 条长期稳定事实(不太会变的,如籍贯、母校、家人)。
3. timeline:{periods} 个时间段,按时间先后,覆盖约 {months} 个月。每段:
   - period_id(p1,p2...)、date_start、date_end(YYYY-MM-DD)、theme(这段的主题,如"入职新公司")、arc(1~2句这段发生了什么)
   - sessions:2~4 个会话,每个含 session_id(p1_s1...)、date(YYYY-MM-DD,落在该段内)、focus(这次聊什么)
   - planted:这段【刻意埋入】的记忆现象数组,每个 {{"phenomenon":<下列之一>, "about":"主题词", "detail":"具体埋什么", "refs":"若是跨段纠错/更新/回指,写它指向之前哪个 period/about,否则 null"}}
4. 现象要【贯穿多段】,尤其这些跨段的要有:
{_PHEN_TEXT}
   典型:p1 说养的猫叫A → p4 纠正其实叫B(correction,refs=p1);p2 住在X → p6 搬到Y(supersede,refs=p2);
   p3 爱喝某饮品 → p7 戒了改喝别的(preference_shift);p2 随口提"我妈生日中秋" → p8 才问起(cross_ref)。
5. probes:12~18 个"事后提问",每个 {{"question":"用户会问的问题", "expected":"依据时间线的正确答案", "tests":"考哪个现象", "kind":"single|update|correction|cross_session|temporal|aggregate|abstain"}}。
   要包含:被更新/纠错后的正确值、跨会话回忆、时效判断、多事实聚合,以及【拒答类】(时间线里从没出现过的信息,期望"不知道/未提及")。

严格输出以下 JSON(只输出 JSON):
{{
 "persona": {{"name":"","age":0,"gender":"","city":"","occupation":"","personality":"","social_circle":[{{"name":"","relation":"","note":""}}],"hobbies":[],"routines":[]}},
 "stable_facts": [""],
 "timeline": [{{"period_id":"p1","date_start":"","date_end":"","theme":"","arc":"","sessions":[{{"session_id":"p1_s1","date":"","focus":""}}],"planted":[{{"phenomenon":"","about":"","detail":"","refs":null}}]}}],
 "probes": [{{"question":"","expected":"","tests":"","kind":""}}]
}}"""


# ============ Stage B: expand one period into realistic dialogue ============
_B_SYS = "你在把一段人生时间线展开成真实的人机日常对话(用户第一人称跟 AI 助手聊天)。语气像真人随手说话:有具体细节、有情绪、有琐碎,不是填表。输出必须是严格 JSON,不要解释文字。"

def _b_user(persona: dict, arcs_brief: str, period: dict) -> str:
    return f"""【人设】{json.dumps(persona, ensure_ascii=False)}

【全时间线概览】(让你知道前后文,才能正确处理纠错/更新/回指)
{arcs_brief}

【要展开的时间段】
{json.dumps(period, ensure_ascii=False)}

把这一段的每个 session 展开成真实对话。要求:
- 每个 session 生成 4~8 轮。每轮 = 用户说的一句(第一人称、口语、有生活细节)+ AI 助手的自然简短回应。
- 把该段 planted 里的现象【自然融入】对话,不要生硬报菜名。correction/supersede/cross_ref 这类要显式指向过去(如"上次跟你说的X,其实是Y""我之前不是在A嘛,现在搬到B了")。
- 真实感:混入一些日常琐碎和闲聊(noise),别每句都是"值得记的事实";偶尔带情绪。
- 每轮附 tags(该轮体现的现象标签数组,取自 planted 的 phenomenon 或 noise/emotion)和 note(一句话说明埋了什么;纯闲聊写 "noise")。

严格输出以下 JSON(只输出 JSON,period_id 用给定值):
{{"period_id":"{period.get('period_id','')}","sessions":[{{"session_id":"","date":"","turns":[{{"user":"","assistant":"","tags":[],"note":""}}]}}]}}"""


def _extract_json(s: str) -> dict:
    """Strip a ```fence``` and keep from the first { to the last }, tolerating
    the occasional explanatory sentence the model adds around the JSON."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    if i > 0 or j < len(s) - 1:
        s = s[i:j + 1]
    return json.loads(s)


def _chat_json(llm: OpenAIChatLLM, system: str, user: str, *, max_tokens: int, retries: int = 2) -> dict:
    """Call the LLM for JSON, retrying on a parse failure with an appended
    "emit valid JSON only" instruction."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = ""
    for attempt in range(1, retries + 2):
        raw = llm.chat(msgs, temperature=0.9, max_tokens=max_tokens)
        last = raw
        try:
            return _extract_json(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed (attempt {attempt}): {e}; tail={raw[-120:]!r}")
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user + "\n\n上次输出不是合法 JSON(可能被截断或多了解释)。请只输出完整合法 JSON。"}]
    raise ValueError(f"still no valid JSON after retries; tail of last response: {last[-300:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=10, help="number of periods (stage A outline)")
    ap.add_argument("--months", type=int, default=18, help="timeline span in months")
    ap.add_argument("--workers", type=int, default=6, help="stage B parallelism")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    setup_logging(Path("logs"))
    out = Path(args.out)
    raw_dir = out.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    llm = OpenAIChatLLM(timeout=180.0)   # generation is slow, so allow a long timeout

    # -- Stage A --
    print(f"[A] building persona + timeline outline ({args.periods} periods / {args.months} months)...")
    outline = _chat_json(llm, _A_SYS, _a_user(args.periods, args.months), max_tokens=8000)
    (raw_dir / "outline.json").write_text(json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")
    persona = outline["persona"]
    timeline = outline["timeline"]
    print(f"    persona: {persona.get('name')} - {persona.get('occupation')} - {persona.get('city')}"
          f" | periods {len(timeline)} | probes {len(outline.get('probes', []))}")

    # Arc summary of the whole timeline, passed to every stage B call as context.
    arcs_brief = "\n".join(
        f"{p['period_id']} [{p.get('date_start','')}~{p.get('date_end','')}] {p.get('theme','')}: {p.get('arc','')}"
        for p in timeline
    )

    # -- Stage B: expand every period in parallel --
    print(f"[B] expanding {len(timeline)} periods in parallel (workers={args.workers})...")
    persona_slim = {k: persona.get(k) for k in ("name", "age", "gender", "city", "occupation", "personality", "social_circle")}

    def expand(period: dict) -> dict:
        pid = period.get("period_id", "?")
        try:
            res = _chat_json(llm, _B_SYS, _b_user(persona_slim, arcs_brief, period), max_tokens=6000)
            (raw_dir / f"{pid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            n_turns = sum(len(s.get("turns", [])) for s in res.get("sessions", []))
            print(f"    ok  {pid} {period.get('theme','')} -> "
                  f"{len(res.get('sessions',[]))} sessions / {n_turns} turns")
            return res
        except Exception as e:
            logger.error(f"{pid} expansion failed: {e}")
            print(f"    ERR {pid} failed: {e}")
            return {"period_id": pid, "sessions": [], "error": str(e)}

    expanded: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(expand, p): p.get("period_id") for p in timeline}
        for f in as_completed(futs):
            r = f.result()
            expanded[r["period_id"]] = r

    # -- Assemble: keep timeline order, merging outline metadata with the dialogue --
    periods_out = []
    total_turns = 0
    phen_count: dict[str, int] = {}
    for p in timeline:
        pid = p["period_id"]
        ex_p = expanded.get(pid, {})
        sessions = ex_p.get("sessions", [])
        for s in sessions:
            for t in s.get("turns", []):
                total_turns += 1
                for tag in t.get("tags", []):
                    key = str(tag).split("(")[0].split(":")[0].strip()
                    phen_count[key] = phen_count.get(key, 0) + 1
        periods_out.append({
            "period_id": pid, "theme": p.get("theme", ""),
            "date_start": p.get("date_start", ""), "date_end": p.get("date_end", ""),
            "arc": p.get("arc", ""), "planted": p.get("planted", []),
            "sessions": sessions,
        })

    corpus = {
        "persona": persona,
        "stable_facts": outline.get("stable_facts", []),
        "periods": periods_out,
        "probes": outline.get("probes", []),
        "meta": {"periods": len(periods_out), "turns": total_turns,
                 "months": args.months, "model": llm.cfg.llm_model},
    }
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    n_sessions = sum(len(p["sessions"]) for p in periods_out)
    print(f"\nwritten -> {out}")
    print(f"  persona {persona.get('name')} | periods {len(periods_out)} | sessions {n_sessions} "
          f"| turns {total_turns} | probes {len(corpus['probes'])}")
    print(f"  phenomenon coverage: {dict(sorted(phen_count.items(), key=lambda x: -x[1]))}")
    failed = [p['period_id'] for p in periods_out if not p['sessions']]
    if failed:
        print(f"  WARNING: periods that failed to expand: {failed} (re-run to retry)")
    llm.close()


if __name__ == "__main__":
    main()
