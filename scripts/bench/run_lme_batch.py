"""Batch orchestration for LongMemEval: run every question across N concurrent
subprocesses, resumable after an interruption.

- Isolation: one subprocess per question, each in its own user namespace
  (lme-<qid>), whose four tables are cleared and rebuilt at start-up, so a
  failure or a re-run of one question only ever touches its own data. Database
  credentials are inherited by the subprocess via env={**os.environ}, needing
  no code changes.
- Pre-splitting: the full 265MB JSON is loaded once in the parent, which writes
  a small split/<qid>.json per question; each subprocess only loads its own
  (5 workers x ~1.5GB of parse memory would otherwise be prohibitive).
- Resumable: questions that already have a <qid>.trace.json are skipped, so
  restarting after a kill simply continues.
- Each question's subprocess is retried 3 times, 120s apart; after three
  failures it is recorded in the failed list without blocking the rest.
- One progress line per completed question, with summary.json (question type x
  score x break point) aggregated every 25 questions and again at the end.

Usage (all 500 questions, 5 workers):
  nohup python -m scripts.bench.run_lme_batch --workers 5 \
    > data/longmemeval/runs/full/batch.log 2>&1 &
Smoke test (3 questions, 3 workers, only the first 4 sessions, ~4 minutes):
  python -m scripts.bench.run_lme_batch --outdir data/longmemeval/runs/smoke \
    --workers 3 --limit 3 --limit-sessions 4
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from loguru import logger

LME_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
# Subprocesses run under the same interpreter as the parent, so a virtualenv or
# conda environment carries over without anyone having to configure a path.
PY = sys.executable


def _judge_of(outdir: Path, qid: str) -> tuple[bool | None, str]:
    """Read this question's score and break point from its trace; a missing or
    corrupt file yields (None, '')."""
    try:
        qa = json.loads((outdir / f"{qid}.trace.json").read_text(encoding="utf-8"))["qa"]
        return bool(qa.get("judge")), str(qa.get("break_point") or "")
    except Exception:   # noqa: BLE001
        return None, ""


def run_one(qid: str, qfile: Path, outdir: Path, limit_sessions: int, tries: int) -> bool:
    """Run one question in a subprocess, retrying up to `tries` times. Success
    means exit code 0 *and* a trace written to disk."""
    out = outdir / qid
    cmd = [PY, "-m", "scripts.bench.run_longmemeval",
           "--qid", qid, "--data", str(qfile), "--out", str(out)]
    if limit_sessions:
        cmd += ["--limit-sessions", str(limit_sessions)]
    env = {**os.environ, "PYTHONPATH": "."}
    for attempt in range(1, tries + 1):
        with open(outdir / f"{qid}.log", "a", encoding="utf-8") as lf:
            lf.write(f"\n===== attempt {attempt}/{tries} {time.strftime('%F %T')} =====\n")
            lf.flush()
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        if r.returncode == 0 and out.with_suffix(".trace.json").exists():
            return True
        logger.warning(f"[{qid}] attempt {attempt}/{tries} failed (rc={r.returncode})")
        if attempt < tries:
            time.sleep(120)
    return False


def summary(outdir: Path) -> dict:
    """Scan every trace, aggregate by question type and break point, and write
    summary.json under both conventions (judge = Mem0, judge_r5 = product)."""
    by_type: dict[str, Counter] = {}
    bps: Counter = Counter()
    n = ok = ok5 = n5 = 0
    for f in sorted(outdir.glob("*.trace.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except Exception:   # noqa: BLE001
            continue
        qtype, judged = t["qa"].get("question_type"), bool(t["qa"].get("judge"))
        c = by_type.setdefault(qtype, Counter())
        c["n"] += 1
        c["ok"] += judged
        if "judge_r5" in t["qa"]:   # only count traces that carry the key; older baseline traces do not, and must not be mixed in
            judged5 = bool(t["qa"]["judge_r5"])
            c["ok_r5"] += judged5
            n5 += 1
            ok5 += judged5
        if not judged:   # break points are only tallied for wrong answers; correct ones carry one too, and would distort the distribution
            bps[t["qa"].get("break_point") or "?"] += 1
        n += 1
        ok += judged
    data = {"n": n, "ok": ok, "acc": round(ok / n, 4) if n else None,
            "n_r5": n5, "ok_r5": ok5, "acc_r5": round(ok5 / n5, 4) if n5 else None,
            "by_type": {k: dict(v) for k, v in by_type.items()},
            "break_points": dict(bps)}
    (outdir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


def stratified_sample(raw_list: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional stratified random sample, keyed by question type combined
    with abstention; rounding differences are absorbed by the larger strata.
    A fixed seed keeps the sample identical across restarts, which is what makes
    resuming safe."""
    rng = random.Random(seed)
    key = lambda d: d["question_type"] + ("+abs" if d["question_id"].endswith("_abs") else "")
    by_type: dict[str, list[dict]] = {}
    for d in raw_list:
        by_type.setdefault(key(d), []).append(d)
    quota = {t: min(round(len(v) * n / len(raw_list)), len(v)) for t, v in by_type.items()}
    diff = n - sum(quota.values())
    order = sorted(quota, key=lambda t: -len(by_type[t]))
    i = 0
    while diff != 0 and i < 1000:
        t = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if 0 <= quota[t] + step <= len(by_type[t]):
            quota[t] += step
            diff -= step
        i += 1
    picked = [d for t in sorted(quota) for d in rng.sample(by_type[t], quota[t])]
    logger.info("stratified sample: " + ", ".join(f"{t}x{quota[t]}" for t in sorted(quota)))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(LME_PATH))
    ap.add_argument("--outdir", default="data/longmemeval/runs/full")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help=">0 runs only the first N questions (debugging)")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 takes a proportional stratified sample of N questions by type; the "
                         "seed is fixed, so a restart re-draws the same sample and resuming is safe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="passed through to the single-question runner (smoke tests)")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_list = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit:
        raw_list = raw_list[:args.limit]
    if args.sample:
        raw_list = stratified_sample(raw_list, args.sample, args.seed)
    todo = [d["question_id"] for d in raw_list
            if not (outdir / f"{d['question_id']}.trace.json").exists()]
    total_done = len(list(outdir.glob("*.trace.json")))
    logger.info(f"{len(todo)} questions to run ({total_done} already done), workers={args.workers}")
    if not todo:
        logger.info(f"aggregate: {summary(outdir)}")
        return

    # Pre-split: write only the split files that are missing for questions still
    # to run. The parent loads the full file once and releases it immediately.
    split_dir = outdir / "split"
    split_dir.mkdir(exist_ok=True)
    raw_by_qid = {d["question_id"]: d for d in raw_list}
    files = {}
    for qid in todo:
        f = split_dir / f"{qid}.json"
        if not f.exists():
            f.write_text(json.dumps([raw_by_qid[qid]], ensure_ascii=False), encoding="utf-8")
        files[qid] = f
    raw_by_qid.clear()

    work: queue.Queue = queue.Queue()
    for qid in todo:
        work.put(qid)
    total = len(todo)
    lock = threading.Lock()
    stats = {"done": 0, "ok": 0, "fail": 0}
    failed: list[str] = []
    t0 = time.time()

    def worker():
        while True:
            try:
                qid = work.get_nowait()
            except queue.Empty:
                return
            run_one(qid, files[qid], outdir, args.limit_sessions, args.tries)
            judged, bp = _judge_of(outdir, qid)
            with lock:
                stats["done"] += 1
                if judged is None:
                    stats["fail"] += 1
                    failed.append(qid)
                    (outdir / "failed.json").write_text(json.dumps(failed, ensure_ascii=False))
                else:
                    stats["ok"] += judged
                elapsed = (time.time() - t0) / 60
                per = elapsed / stats["done"]
                eta = per * (total - stats["done"])
                mark = "ok" if judged else "X"
                logger.info(f"[{stats['done']}/{total}] {qid} {mark} "
                            f"({elapsed:.0f}m elapsed, ~{eta:.0f}m left) bp={bp[:24]}")
                if stats["done"] % 10 == 0:
                    logger.info(f"interim aggregate: {summary(outdir)}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = summary(outdir)
    logger.info(f"done: {data['ok']}/{data['n']} correct; {len(failed)} questions failed with "
                f"no trace: {failed}")
    if failed:
        (outdir / "failed.json").write_text(json.dumps(failed, ensure_ascii=False))


if __name__ == "__main__":
    main()
