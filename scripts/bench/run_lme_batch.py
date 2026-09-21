"""LongMemEval 批量评测编排:N 并发子进程跑全量题,断点续跑。

- 隔离:每题独立子进程,共享 SIT MySQL 里独立 user 命名空间(lme-<qid>,
  启动即清本 user 四表重建)——单题失败/重跑只动自己的 user 数据。
  MYSQL_* 经 env={**os.environ} 自动透传给子进程,零代码改动。
- 预切题:265MB 全量 JSON 父进程只载一次,逐题切 split/<qid>.json 小文件,
  子进程只载自己的题(5 并发 × ~1.5GB 解析内存 → 忽略不计)。
- 断点续跑:已有 <qid>.trace.json 的题跳过;被杀后重启即续。
- 每题子进程失败重试 ×3(间隔 120s);3 败记入 failed 列表,不阻塞整体。
- 每题完成打一行进度;每 25 题及结束时聚合 summary.json(题型 × 判分 × 断点)。

用法(全量 500 题 5 并发):
  nohup caffeinate -i python -m scripts.bench.run_lme_batch --workers 5 \
    > data/longmemeval/runs/full/batch.log 2>&1 &
冒烟(3 题 3 并发,只灌前 4 个 session,~4 分钟):
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
import threading
import time
from collections import Counter
from pathlib import Path

from loguru import logger

LME_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
PY = Path.home() / "miniconda3/envs/personos/bin/python"


def _judge_of(outdir: Path, qid: str) -> tuple[bool | None, str]:
    """读该题 trace 的判分与断点;文件缺失/损坏返回 (None, '')。"""
    try:
        qa = json.loads((outdir / f"{qid}.trace.json").read_text(encoding="utf-8"))["qa"]
        return bool(qa.get("judge")), str(qa.get("break_point") or "")
    except Exception:   # noqa: BLE001
        return None, ""


def run_one(qid: str, qfile: Path, outdir: Path, limit_sessions: int, tries: int) -> bool:
    """单题子进程,失败重试 ×3;成功判据 = rc 0 且 trace 落盘。"""
    out = outdir / qid
    cmd = [str(PY), "-m", "scripts.bench.run_longmemeval",
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
        logger.warning(f"[{qid}] 第 {attempt}/{tries} 次失败(rc={r.returncode})")
        if attempt < tries:
            time.sleep(120)
    return False


def summary(outdir: Path) -> dict:
    """扫全部 trace → 按题型/断点聚合,写 summary.json(双口径:judge=Mem0,judge_r5=产品)。"""
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
        if "judge_r5" in t["qa"]:   # 产品口径仅统计有该键的 trace(旧基线 trace 无此键,不混算)
            judged5 = bool(t["qa"]["judge_r5"])
            c["ok_r5"] += judged5
            n5 += 1
            ok5 += judged5
        if not judged:   # H3:断点分布只统计错题——对题也带 break_point,混入让归因分布失真
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
    """按「题型 × abstention」复合键比例分层随机抽样;取整差额在大题型上增减找平。
    seed 固定 → 重启抽样一致(断点续跑安全)。"""
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
    logger.info("分层抽样 " + ", ".join(f"{t}×{quota[t]}" for t in sorted(quota)))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(LME_PATH))
    ap.add_argument("--outdir", default="data/longmemeval/runs/full")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 题(调试)")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 按题型分层比例抽样 N 题(seed 固定,重启重抽结果一致,断点续跑安全)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-sessions", type=int, default=0, help="透传单题 runner(冒烟用)")
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
    logger.info(f"待跑 {len(todo)} 题(已完成 {total_done}),workers={args.workers}")
    if not todo:
        logger.info(f"聚合结果: {summary(outdir)}")
        return

    # 预切题:只写待跑且缺失的 split 文件(父进程载一次全量,切完即释放)
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
                mark = "✓" if judged else "✗"
                logger.info(f"[{stats['done']}/{total}] {qid} {mark} "
                            f"({elapsed:.0f}m elapsed, ~{eta:.0f}m left) bp={bp[:24]}")
                if stats["done"] % 10 == 0:
                    logger.info(f"中期聚合: {summary(outdir)}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = summary(outdir)
    logger.info(f"完成: {data['ok']}/{data['n']} 正确;失败(无 trace)题 {len(failed)}: {failed}")
    if failed:
        (outdir / "failed.json").write_text(json.dumps(failed, ensure_ascii=False))


if __name__ == "__main__":
    main()
