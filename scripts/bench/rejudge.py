"""离线重判:用当前 judge(judge v2)对既有 trace 重新打分——不重跑链路,只重打分。

用途:judge 口径升级后对齐 baseline。judge 口径变化会系统性移动分数(相对时间题
v1 判错),不重判就无法回答"新版提升多少是链路的、多少是 judge 的"。

用法:
  PYTHONPATH=. python -m scripts.bench.rejudge data/bench/conv26.trace.json
产物:<原名>.judgev2.json(原 trace 不动),终端打印新旧对比与翻转清单。
"""
from __future__ import annotations

import json
import sys
import time

from personos.clients.minimax import MinimaxClient

from .judge import judge


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    trace = json.load(open(src))
    qas = trace.get("qa") or []
    if not qas:
        print("trace 里没有 qa 记录")
        sys.exit(1)

    llm = MinimaxClient()
    old_ok = sum(1 for q in qas if q.get("judge"))
    flips, new_ok = [], 0
    t0 = time.time()
    for i, q in enumerate(qas, 1):
        v = judge(llm, question=q["question"], gold=str(q["gold"]), prediction=q["answer"])
        q["judge_v1"], q["judge_raw_v1"] = q.get("judge"), q.get("judge_raw")   # 留旧判定
        q["judge"], q["judge_raw"] = v.ok, v.raw
        new_ok += int(v.ok)
        if v.ok != q["judge_v1"]:
            flips.append((i, q["judge_v1"], v.ok, q["question"][:70]))
        if i % 25 == 0:
            print(f"  {i}/{len(qas)} … {time.time() - t0:.0f}s", flush=True)

    trace.setdefault("meta", {})["rejudged"] = {
        "at": time.strftime("%Y-%m-%d %H:%M"), "judge": "v2 相对时间换算",
        "v1_ok": old_ok, "v2_ok": new_ok, "n": len(qas)}

    out = f"{src.rstrip('.json')}.judgev2.json"
    json.dump(trace, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\njudge v1 → v2:{old_ok}/{len(qas)} → {new_ok}/{len(qas)} "
          f"({old_ok / len(qas):.1%} → {new_ok / len(qas):.1%}),翻转 {len(flips)} 题")
    for i, old, new, qs in flips:
        print(f"  #{i} {'WRONG→CORRECT' if new else 'CORRECT→WRONG'}  {qs}")
    print(f"产物:{out}(原 trace 未动)")


if __name__ == "__main__":
    main()
