"""Offline re-scoring: re-grade an existing trace with the current judge
(judge v2). The pipeline is not re-run; only the scoring is.

Why: to realign a baseline after the judge criteria change. A change in judge
criteria moves scores systematically (v1 marked relative-time questions wrong),
and without re-scoring there is no way to answer "how much of the improvement
came from the pipeline and how much from the judge".

Usage:
  PYTHONPATH=. python -m scripts.bench.rejudge data/bench/conv26.trace.json
Output: <name>.judgev2.json (the original trace is left untouched); the old and
new scores plus the list of flipped questions are printed to the terminal.
"""
from __future__ import annotations

import json
import sys
import time

from personos.providers.anthropic_compat import AnthropicChatLLM

from .judge import judge


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    trace = json.load(open(src))
    qas = trace.get("qa") or []
    if not qas:
        print("the trace contains no qa records")
        sys.exit(1)

    llm = AnthropicChatLLM()
    old_ok = sum(1 for q in qas if q.get("judge"))
    flips, new_ok = [], 0
    t0 = time.time()
    for i, q in enumerate(qas, 1):
        v = judge(llm, question=q["question"], gold=str(q["gold"]), prediction=q["answer"])
        q["judge_v1"], q["judge_raw_v1"] = q.get("judge"), q.get("judge_raw")   # keep the old verdict
        q["judge"], q["judge_raw"] = v.ok, v.raw
        new_ok += int(v.ok)
        if v.ok != q["judge_v1"]:
            flips.append((i, q["judge_v1"], v.ok, q["question"][:70]))
        if i % 25 == 0:
            print(f"  {i}/{len(qas)} … {time.time() - t0:.0f}s", flush=True)

    trace.setdefault("meta", {})["rejudged"] = {
        "at": time.strftime("%Y-%m-%d %H:%M"), "judge": "v2 with relative-time conversion",
        "v1_ok": old_ok, "v2_ok": new_ok, "n": len(qas)}

    out = f"{src.rstrip('.json')}.judgev2.json"
    json.dump(trace, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\njudge v1 -> v2: {old_ok}/{len(qas)} -> {new_ok}/{len(qas)} "
          f"({old_ok / len(qas):.1%} -> {new_ok / len(qas):.1%}), {len(flips)} questions flipped")
    for i, old, new, qs in flips:
        print(f"  #{i} {'WRONG→CORRECT' if new else 'CORRECT→WRONG'}  {qs}")
    print(f"output: {out} (the original trace is unchanged)")


if __name__ == "__main__":
    main()
