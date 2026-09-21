"""LoCoMo 批量执行器:逐 conversation 调 run_locomo,每 conv 独立 trace/报告 + 轻量聚合。

设计:
- 每 conv 一个子进程跑 run_locomo(共享 MySQL 里独立 user 命名空间,互不污染;内存不累积);
- 产物按 conv 分目录:<outdir>/<conv>/{trace.json, report.html, report.md}——
  单个 trace 不至于过大,也便于单独重跑/审阅某个 conv;
- 断点续跑:该 conv 的 trace.json 已存在即跳过(--force 重跑);
- 聚合:只读各 conv trace 的 meta/summary/错题归因,产出 aggregate md(不复制大 JSON)。

用法:
  python -m scripts.bench.run_all \
    --data data/locomo10.json --outdir data/bench/runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.bench.locomo_adapter import load_conversations
from scripts.bench.report import _CAT_NAME, _error_kind

_PY = sys.executable


def _run_one(conv: str, args, outdir: Path) -> bool:
    """跑单个 conv(子进程,失败自动重试:长批跑的网络抖动/DNS 瞬断不至报废整个 conv)。

    run_locomo 的 --out 是产物前缀:传 <outdir>/<conv>/<conv> 得到
    <conv>.trace.json / <conv>.report.html / <conv>.report.md;MySQL 侧按
    locomo-<conv> user 命名空间隔离,重跑自清。
    """
    cdir = outdir / conv
    cdir.mkdir(parents=True, exist_ok=True)
    if (cdir / f"{conv}.trace.json").exists() and not args.force:
        print(f"[skip] {conv}: trace 已存在(--force 重跑)")
        return False
    cmd = [_PY, "-m", "scripts.bench.run_locomo",
           "--conv", conv, "--n-sessions", str(args.n_sessions),
           "--n-questions", str(args.n_questions), "--mode", args.mode,
           "--data", args.data,
           "--out", str(cdir / conv)]
    for attempt in (1, 2, 3):
        print(f"[run ] {conv}: {time.strftime('%H:%M:%S')}" + (f" (第{attempt}次)" if attempt > 1 else ""))
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode == 0:
            print(f"[done] {conv} ({time.time()-t0:.0f}s)")
            return True
        print(f"[FAIL] {conv} 第{attempt}次失败 ({time.time()-t0:.0f}s)")
        if attempt < 3:
            time.sleep(60)   # 网络抖动类失败:等一分钟再试
    print(f"[GIVE-UP] {conv}: 三次失败,跳过(可稍后单跑)")
    return False


def _aggregate(outdir: Path) -> Path:
    """汇总所有 conv 的 summary → aggregate md(只读 meta/summary,不复制大 JSON)。"""
    rows, by_cat, errs, tot_ok, tot_n = [], {}, {}, 0, 0
    for cdir in sorted(outdir.iterdir() if outdir.exists() else []):
        if not cdir.is_dir():
            continue
        tj = cdir / f"{cdir.name}.trace.json"   # run_locomo 按 --out 前缀写 <conv>.trace.json
        if not tj.exists():
            tj = cdir / "trace.json"            # 兼容历史目录布局
        if not tj.exists():
            continue
        t = json.loads(tj.read_text(encoding="utf-8"))
        s, m = t.get("summary", {}), t.get("meta", {})
        ok, n = s.get("score", 0), s.get("n_questions", 0)
        tot_ok += ok
        tot_n += n
        esc = s.get("escalated", 0)
        rows.append(f"| {cdir.name} | {m.get('n_sessions')}s | {ok}/{n} "
                    f"| {100*ok/max(1,n):.0f}% | {esc}/{n} | {m.get('git','')} |")
        for k, v in s.get("by_category", {}).items():
            a = by_cat.setdefault(k, [0, 0])
            a[0] += v[0]
            a[1] += v[1]
        for q in t.get("qa", []):
            if not q.get("judge"):
                k = q.get("break_point") or _error_kind(q)
                errs[k] = errs.get(k, 0) + 1
    lines = ["# LoCoMo × personos · 全量聚合报告", ""]
    lines.append(f"- 生成: {time.strftime('%Y-%m-%d %H:%M')} · 覆盖 {len(rows)} 个 conversation")
    lines.append(f"- 总分: **{tot_ok}/{tot_n} ({100*tot_ok/max(1,tot_n):.1f}%)**")
    lines.append("")
    lines.append("## 各 conversation")
    lines.append("| conv | sessions | 得分 | 正确率 | partial+empty | git |")
    lines.append("|---|---|---|---|---|---|")
    lines += rows
    lines += ["", "## 分题型(全量聚合)", "| 题型 | 对/总 | 正确率 |", "|---|---|---|"]
    for k, v in sorted(by_cat.items()):
        lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {v[0]}/{v[1]} | {100*v[0]/max(1,v[1]):.0f}% |")
    lines += ["", "## 错题断点分布(金标证据链最先断的环节)", ""]
    lines += [f"- {k} ×{v}" for k, v in sorted(errs.items(), key=lambda x: -x[1])] or ["- 无错题"]
    lines.append("")
    lines.append("> 逐 conv 细节见各目录 report.md / report.html;判定以链路展开为准。")
    p = outdir / "aggregate.report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"聚合报告: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/locomo10.json")
    ap.add_argument("--outdir", default="data/bench/runs")
    ap.add_argument("--n-sessions", type=int, default=99, help="每 conv 灌的 session 上限(99=全量)")
    ap.add_argument("--n-questions", type=int, default=999, help="每 conv 答题上限(999=全部有效题)")
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--convs", default="", help="逗号分隔的 conv 名,空=全部 10 个")
    ap.add_argument("--force", action="store_true", help="忽略已存在的 trace 重跑")
    ap.add_argument("--aggregate-only", action="store_true", help="只重算聚合报告,不跑评测")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        all_ids = [c.sample_id for c in load_conversations(args.data)]
        todo = [c for c in (args.convs.split(",") if args.convs else all_ids) if c]
        t0 = time.time()
        for i, conv in enumerate(todo, 1):
            print(f"\n===== [{i}/{len(todo)}] {conv} =====")
            _run_one(conv, args, outdir)
        print(f"\n全部完成 ({time.time()-t0:.0f}s)")
    _aggregate(outdir)


if __name__ == "__main__":
    main()
