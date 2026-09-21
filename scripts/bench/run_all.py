"""Batch runner for LoCoMo: invoke run_locomo per conversation, with a separate
trace and report for each, plus a lightweight aggregate.

Design:
- one subprocess per conversation running run_locomo, each in its own user
  namespace so they cannot contaminate one another and memory does not
  accumulate;
- artefacts are laid out per conversation as
  <outdir>/<conv>/{trace.json, report.html, report.md}, which keeps any single
  trace from growing too large and makes it easy to re-run or review one
  conversation on its own;
- resumable: a conversation whose trace.json already exists is skipped (use
  --force to re-run it);
- aggregation reads only the meta, summary and error attribution from each
  trace to produce an aggregate markdown file; the large JSON is never copied.

Usage:
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
    """Run one conversation in a subprocess, retrying on failure so that a
    network blip or a momentary DNS failure during a long batch does not write
    off the whole conversation.

    run_locomo's --out is an artefact prefix: passing <outdir>/<conv>/<conv>
    yields <conv>.trace.json / <conv>.report.html / <conv>.report.md. On the
    database side each conversation is isolated in the locomo-<conv> user
    namespace, which a re-run clears for itself.
    """
    cdir = outdir / conv
    cdir.mkdir(parents=True, exist_ok=True)
    if (cdir / f"{conv}.trace.json").exists() and not args.force:
        print(f"[skip] {conv}: a trace already exists (use --force to re-run)")
        return False
    cmd = [_PY, "-m", "scripts.bench.run_locomo",
           "--conv", conv, "--n-sessions", str(args.n_sessions),
           "--n-questions", str(args.n_questions), "--mode", args.mode,
           "--data", args.data,
           "--out", str(cdir / conv)]
    for attempt in (1, 2, 3):
        print(f"[run ] {conv}: {time.strftime('%H:%M:%S')}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode == 0:
            print(f"[done] {conv} ({time.time()-t0:.0f}s)")
            return True
        print(f"[FAIL] {conv} attempt {attempt} failed ({time.time()-t0:.0f}s)")
        if attempt < 3:
            time.sleep(60)   # network-blip failures: wait a minute and try again
    print(f"[GIVE-UP] {conv}: failed three times, skipping (run it on its own later)")
    return False


def _aggregate(outdir: Path) -> Path:
    """Aggregate every conversation's summary into an aggregate markdown file,
    reading only meta and summary and never copying the large JSON."""
    rows, by_cat, errs, tot_ok, tot_n = [], {}, {}, 0, 0
    for cdir in sorted(outdir.iterdir() if outdir.exists() else []):
        if not cdir.is_dir():
            continue
        tj = cdir / f"{cdir.name}.trace.json"   # run_locomo writes <conv>.trace.json from the --out prefix
        if not tj.exists():
            tj = cdir / "trace.json"            # compatibility with the older directory layout
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
    lines = ["# LoCoMo x personos - aggregate report", ""]
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M')} - covering {len(rows)} conversations")
    lines.append(f"- total: **{tot_ok}/{tot_n} ({100*tot_ok/max(1,tot_n):.1f}%)**")
    lines.append("")
    lines.append("## Per conversation")
    lines.append("| conv | sessions | score | accuracy | partial+empty | git |")
    lines.append("|---|---|---|---|---|---|")
    lines += rows
    lines += ["", "## Per question category (aggregated)",
              "| category | correct/total | accuracy |", "|---|---|---|"]
    for k, v in sorted(by_cat.items()):
        lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {v[0]}/{v[1]} | {100*v[0]/max(1,v[1]):.0f}% |")
    lines += ["", "## Break points of wrong answers (first stage where the gold evidence chain broke)", ""]
    lines += [f"- {k} x{v}" for k, v in sorted(errs.items(), key=lambda x: -x[1])] or ["- no wrong answers"]
    lines.append("")
    lines.append("> Per-conversation detail is in each directory's report.md / report.html; "
                 "judge the result by expanding the pipeline, not by the score alone.")
    p = outdir / "aggregate.report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"aggregate report: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/locomo10.json")
    ap.add_argument("--outdir", default="data/bench/runs")
    ap.add_argument("--n-sessions", type=int, default=99,
                    help="maximum sessions loaded per conversation (99 = all of them)")
    ap.add_argument("--n-questions", type=int, default=999,
                    help="maximum questions answered per conversation (999 = every valid one)")
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--convs", default="",
                    help="comma-separated conversation names; empty means all ten")
    ap.add_argument("--force", action="store_true",
                    help="re-run even when a trace already exists")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only recompute the aggregate report, without running the benchmark")
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
        print(f"\nall done ({time.time()-t0:.0f}s)")
    _aggregate(outdir)


if __name__ == "__main__":
    main()
