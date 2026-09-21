"""LoCoMo × personos 评测 harness(融合架构版)——带全流程 trace 落盘。

流程:选定对话的前 N 个 session → SessionWriter 逐句 feed(W0 证据/W1 边界/W2 建 cell,
now_dt=会话时间)→ 挑 evidence 全落在已灌 session 内的题 → 新快链一条龙 run_recall
(R0 五件套→R1 双路→R2 精排→R5 草稿→R3' 核判)→ 模式 A answerer 出英文答案 → Mem0 式二元 judge。

评测协议(2026-09-03 起双口径,S4 决策 3):
- **Mem0 口径**(对外可比):answerer=模式A(brief=R5 直答,memories=精排前 20 cell 命中
  atoms 带日期、按 cell 分组,与 R5 材料面对齐;简报信任规则),judge 判 answerer 产物;
- **产品口径**(产品读数):无第二棒,judge 直判 R5 答案(judge_r5)。
对外对比时声明协议差异。分数差异归因到记忆系统(写入+检索+R5 材料组织)的变化。

透明度是本 harness 的一等公民:每轮的边界判定、每个闭合 cell 的 topic/episode/atoms、
每题的五件套/双路排名/核判/作答,全写进 trace JSON,report.py 渲染自包含 HTML 报告。

LLM:MiniMax-M3(Anthropic 兼容);embedding:MAAS qwen3-embedding(只换 LLM)。

用法:
  python -m scripts.bench.run_locomo --conv conv-26 --n-sessions 99 --n-questions 999
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from personos.app.recall_flow import run_recall
from personos.clients.maas import MaasClient
from personos.clients.minimax import MinimaxClient
from personos.models import stamped_atom_text
from personos.online.rerank import MaasReranker
from personos.online.retrieval import cell_lead
from personos.online.trust import evidence_entries
from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from scripts.bench.judge import Verdict, answer_mode_a, judge
from scripts.bench.locomo_adapter import LocomoConversation, load_conversations, pick_answerable

LOCOMO_PATH = Path("data/locomo10.json")   # 默认本地;服务器/共享盘跑时用 --data 覆盖

# trace 里长文本的截断上限(报告里可展开的都是这些;完整原话在库里,不截)
_CAP_RAW = 2000

# answerer 的 memories 面宽:精排后前 N 个 cell 的命中 atoms(与对外 /recall 的依据记忆同口径)
_ANSWER_MEM_CELLS = 20    # answerer 材料面与 R5 对齐(双口径决策 3;原 10 格两层不一致)


def _grouped_materials(groups: list[tuple]) -> str:
    """answerer 材料组织:按 cell 分组——cell_lead 头(对话时间+topic)+ 组内 stamped atoms。

    与产品面 answer_from_cells 的材料形态同构:枚举题的自查按组扫,组头提供时间/topic 锚。
    """
    parts = []
    for c, atom_list in groups:
        lines = "\n".join(f"- {stamped_atom_text(a)}" for a in atom_list)
        parts.append(f"{cell_lead(c)}\n{lines}" if lines else cell_lead(c))
    return "\n\n".join(parts)


def _cap(s: str | None, n: int = _CAP_RAW) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" …[截断,全文{len(s)}字符]"


def _atom_brief(a) -> dict:
    return {
        "id": a.id, "cell_id": a.memcell_id, "holder": a.holder, "text": a.text,
        "type": a.object_type, "domains": a.domains, "kind": a.kind,
        "occurrence": a.occurrence_time.isoformat() if a.occurrence_time else None,
        "evidence": [r.evidence_id for r in a.evidence_refs],
    }


def _cell_brief(cb) -> dict:
    """闭合 cell 的 trace 视图:topic/episode/atoms + W2 两次调用溯源。"""
    c = cb.cell
    return {
        "cell_id": c.id, "topic": c.topic, "domains": c.domains,
        "t_start": c.t_start.isoformat() if c.t_start else None,
        "t_end": c.t_end.isoformat() if c.t_end else None,
        "episode": _cap(c.episode, 600),
        "atoms": [_atom_brief(a) for a in cb.atoms],
        "gen": {k: {"system": _cap(v.get("system"), 400), "user": _cap(v.get("user"), 800),
                    "raw": _cap(v.get("raw"))}
                for k, v in (cb.gen or {}).items()},
    }


def ingest_sessions(lc: LocomoConversation, n_sessions: int, llm, embedder,
                    atoms: AtomStore, cells: CellStore, ev_store: EvidenceStore):
    """逐会话 SessionWriter 快进(W0/W1/W2 同一循环,与产品在线等价);返回 (末次会话时间, trace)。"""
    trace_sessions = []
    last_dt = None
    for sess in lc.sessions[:n_sessions]:
        writer = SessionWriter(llm, embedder, ev_store, cells, atoms,
                               session_id=f"{lc.sample_id}-s{sess.idx}")
        t0 = time.time()
        ex_records, closed_in_session = [], []
        for ex in sess.exchanges:
            r = writer.feed(ex.holder, ex.text, now_dt=sess.dt)
            ex_records.append({
                "dia": ex.dia, "holder": ex.holder, "text": _cap(ex.text, 800),
                "evidence_id": r.evidence_id,        # dia↔证据id 映射,金标证据链追踪用
                "boundary": ({"should_end": r.boundary.should_end,
                              "confidence": r.boundary.confidence,
                              "topic_summary": r.boundary.topic_summary}
                             if r.boundary else None),   # None = 段首句(无界可判)
                "forced_close": r.forced_close,      # 30 轮安全阀(非 LLM 判定)
            })
            if r.closed_cell:
                closed_in_session.append(r.closed_cell)
        # end_session 返回全会话 cell 清单:只取未记录的尾部(段内闭合 + 会话末强制闭合,不重复)
        closed_in_session.extend(writer.end_session()[len(closed_in_session):])
        trace_sessions.append({
            "session": sess.idx, "dt": sess.dt.isoformat(),
            "exchanges": ex_records,
            "closed_cells": [_cell_brief(cb) for cb in closed_in_session],
            "cells_total": len(writer.cells),
            "atoms_total": len(atoms.list(limit=100000)),
            "secs": round(time.time() - t0, 1),
        })
        logger.info(f"[{lc.sample_id}] s{sess.idx} 完成: {len(sess.exchanges)} exchanges -> "
                    f"{len(closed_in_session)} cells / 全局 {trace_sessions[-1]['atoms_total']} atoms "
                    f"({time.time()-t0:.0f}s)")
        last_dt = sess.dt
    return last_dt, trace_sessions


def pick_diverse(qas, n: int) -> list:
    """按题型轮转取样,保证类别覆盖。"""
    by_cat = defaultdict(list)
    for qa in qas:
        by_cat[qa.category].append(qa)
    out = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop(0))
    return out


def _atom_view(ah, rank: int) -> dict:
    """R1 atom 池一条命中的 trace 视图。"""
    return {"rank": rank, "atom_id": ah.atom.id, "text": ah.atom.text,
            "holder": ah.atom.holder, "type": ah.atom.object_type,
            "chain_id": ah.atom.chain_id,
            "rrf": round(ah.rrf, 5), "sim": round(ah.similarity, 4)}


def _hit_view(h, rank: int) -> dict:
    """一个材料单元的 trace 视图(组装序/精排序共用;covers 非空=织写 memcell′)。"""
    return {
        "rank": rank, "cell_id": h.cell.id, "topic": h.cell.topic,
        "domains": h.cell.domains, "covers": h.covers,
        "rrf": round(h.score, 5), "best_sim": round(h.best_sim, 4),
        "rerank_score": h.rerank_score,
        "atoms": [{"id": a.atom.id, "text": a.atom.text, "holder": a.atom.holder,
                   "type": a.atom.object_type, "sim": round(a.similarity, 4)}
                  for a in h.atoms],
    }


def _qa_trace(o, qa, answer, verdict, r5_verdict, secs, mem_atoms) -> dict:
    """把一次问答的全链路收进 trace(含深轨轨迹)。memories 的支撑原话在 main 里补(需要 ev_store)。

    双口径:judge=Mem0 口径(answerer 产物);judge_r5=产品口径(直判 R5 答案)。"""
    memories = [{"atom_id": a.id, "text": stamped_atom_text(a), "type": a.object_type,
                 "evidence": []} for a in mem_atoms]
    return {
        "category": qa.category, "question": qa.question, "gold": qa.answer,
        "evidence": qa.evidence, "secs": secs,
        "rewrite": {"resolved": o.rw.resolved if o.rw else None,
                    "subject": o.rw.subject if o.rw else "",
                    "expansions": o.rw.expansions if o.rw else [],
                    "time_window": ([o.rw.time_start, o.rw.time_end]
                                    if (o.rw and (o.rw.time_start or o.rw.time_end)) else []),
                    "domains": o.rw.domains if o.rw else []},
        "pool": [_atom_view(a, i + 1) for i, a in enumerate(o.hits[:10])],     # R1 atom 池
        "hits": [_hit_view(h, i + 1) for i, h in
                 enumerate((o.asm.units if o.asm else [])[:10])],               # 材料单元(组装序)
        "ranked": [_hit_view(h, i + 1) for i, h in enumerate(o.ranked[:10])],  # R2 精排序
        "review": ({"verdict": o.reviews[-1].verdict, "critique": o.reviews[-1].critique,
                    "retried": o.retried} if o.reviews else None),
        "r5": {"answer": o.ans.answer, "cited_cells": o.ans.cited_cells} if o.ans else None,
        "deep": ({"steps": o.deep.steps, "remembered": o.deep.remembered,
                  "escalated": o.escalated, "answer": o.deep.ans.answer,
                  "cited_cells": o.deep.ans.cited_cells}
                 if o.deep else None),                                       # 深轨轨迹(未跑为 None)
        "memories": memories,
        "answer": answer, "judge": verdict.ok, "judge_raw": verdict.raw,
        "judge_r5": r5_verdict.ok, "judge_r5_raw": r5_verdict.raw,
    }


def _dia_evidence(lc, store_evidence: list) -> dict:
    """dia → 证据id:按 utterance 文本前缀把 adapter 的 dia 对齐到库里的证据。

    仅 --skip-ingest(无 ingest 记录)时的兜底:合并轮的非首句 dia 前缀对不上
    整段证据的前缀,可能漏映;有 ingest 记录时 main 走精确映射(dia→evidence_id)。
    """
    prefix2ev = {}
    for e in store_evidence:
        c = e.get("content") or ""
        if c:
            prefix2ev.setdefault(c[:60], e["id"])
    out = {}
    for s in lc.sessions:
        for ex in s.exchanges:
            dias = [d for d in (ex.dia or "").split(",") if d]
            for d, line in zip(dias, ex.text.split("\n")):
                ev_id = prefix2ev.get(line[:60])
                if ev_id:
                    out[d] = ev_id
    return out


def _gold_chain(qa_rec: dict, dia2ev: dict, store_atoms: list) -> list:
    """金标证据链:金标指向的每条原话(dia)在本题链路里走到了哪一环。

    环节:证据层(入库了吗)→抽取层(抽成原子了吗)→检索层(进 R1 top10 cells 了吗)→
    材料层(进 R5 作答材料 top20 了吗)→采信层(R5 引用了该 cell 吗)。
    断点=最先断的环节,即"错误点"。
    """
    hit_atom_ids = {a["id"] for h in qa_rec.get("hits", []) for a in h["atoms"]}
    ranked_atom_ids = {a["id"] for h in qa_rec.get("ranked", []) for a in h["atoms"]}
    cited = set((qa_rec.get("r5") or {}).get("cited_cells") or [])
    ev2atoms: dict[str, list] = defaultdict(list)
    for a in store_atoms:
        for e in a.get("evidence") or []:
            ev2atoms[e].append(a)
    chains = []
    for d in qa_rec.get("evidence", []):
        ev_id = dia2ev.get(d)
        hit_atoms = ev2atoms.get(ev_id, []) if ev_id else []
        chains.append({
            "dia": d, "ev_id": ev_id,
            "n_extracted": len(hit_atoms),
            "extracted_atoms": [a["text"][:60] for a in hit_atoms[:3]],
            "cell_ids": sorted({a["cell_id"] for a in hit_atoms if a.get("cell_id")}),
            "in_hits": any(a["id"] in hit_atom_ids for a in hit_atoms),
            "in_material": any(a["id"] in ranked_atom_ids for a in hit_atoms),
            "cited": bool(hit_atoms and (cited & {a["cell_id"] for a in hit_atoms})),
        })
    return chains


def _break_point(qa_rec: dict, chains: list) -> str:
    """断点判定:链路能力上限走到哪一环,错误点就在那环之后。"""
    if not chains:
        return "无evidence(金标未指位置)"
    if not any(c["ev_id"] for c in chains):
        return "证据层(dia未入库)"
    if not any(c["n_extracted"] for c in chains):
        return "抽取层(原话在库但没抽成原子)"
    if any(c["cited"] for c in chains):
        # 金标证据已被 R5 引用仍错:死在作答/判分层
        ans = (qa_rec.get("answer") or "")
        if ans.lower().startswith("i don't have"):
            return "answerer(已引用仍拒答)"
        return "作答层(answerer答偏或judge误判,人工复核)"
    if any(c["in_material"] for c in chains):
        return "采信层(进了作答材料但未被R5引用)"
    if any(c["in_hits"] for c in chains):
        return "材料层(检索到了但被top20截断)"
    return "检索层(原子在库,双路没召回)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--n-sessions", type=int, default=3)
    ap.add_argument("--n-questions", type=int, default=8)
    ap.add_argument("--out", default="data/bench/locomo_demo")
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "deep"])
    ap.add_argument("--data", default=str(LOCOMO_PATH),
                    help="locomo10.json 路径(默认 data/locomo10.json;持久盘/GPU 服务器跑时覆盖)")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="复用该 conv 已灌的 user 数据,只跑问答阶段——调试判分/检索时省灌库时间")
    ap.add_argument("--ingest-only", action="store_true",
                    help="只灌库不作答,灌完即停——批量场景先并发灌满多个 conv,再统一 --skip-ingest 作答")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="作答阶段 conv 内题目并发度(默认 1=串行;每题独立 session_id,互不污染)")
    ap.add_argument("--questions-file", default="",
                    help="JSON 数组文件(题面精确匹配),只跑这些题——小回归用,跳过 pick_diverse 抽样")
    args = ap.parse_args()

    convs = load_conversations(args.data)
    lc = next(c for c in convs if c.sample_id == args.conv)
    print(f"对话 {lc.sample_id}: {len(lc.sessions)} sessions / {len(lc.qa)} 题;取前 {args.n_sessions} 个 session")

    # 共享 MySQL:每 conv 独立 user 命名空间,重跑前清本 user 四表(幂等,原子数不翻倍)
    uid = f"locomo-{lc.sample_id}"
    db = Database()
    have = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
    if args.skip_ingest:
        assert have > 0, f"--skip-ingest 需要已灌的 user 数据: {uid}"
    elif have:
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
    ev_store, atoms = EvidenceStore(db, uid), AtomStore(db, uid)
    cells = CellStore(db, uid)
    llm = MinimaxClient()                       # 评测 LLM(MiniMax-M3):W1/W2/R0/R3/R5/answerer/judge
    embedder = MaasClient()                     # embedding:MAAS qwen3(写入 topic/atom 向量 + 检索查询面)

    trace: dict = {"meta": {"conv": lc.sample_id, "n_sessions": args.n_sessions,
                            "mode": args.mode, "llm": "MiniMax-M3",
                            "judge": "MiniMax-M3(Mem0式二元,v2含相对时间换算)",
                            "answerer": "模式A双口径(brief=R5直答, memories=精排前20 cell命中atoms按cell分组+简报信任; judge_r5=直判R5)",
                            "chain": "融合架构步骤1写入+步骤2快链",
                            "run_at": time.strftime("%Y-%m-%d %H:%M"), "data": args.data}}
    try:   # git 短哈希进 meta,报告可追溯到代码版本
        import subprocess
        trace["meta"]["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    # —— 灌库(SessionWriter:W0 证据/W1 边界/W2 建 cell)——
    if args.skip_ingest:
        # n_sessions 常传 99 表示"全部";出题时刻 = 实际已灌的最后一个 session
        last_dt = lc.sessions[min(args.n_sessions, len(lc.sessions)) - 1].dt
        trace["ingest"] = []
        print(f"复用已灌 user {uid}(全局 {len(atoms.list(limit=100000))} atoms / "
              f"{len(cells.iter_all())} cells),出题时刻 {last_dt.isoformat()}")
    else:
        t0 = time.time()
        last_dt, trace["ingest"] = ingest_sessions(lc, args.n_sessions, llm, embedder,
                                                   atoms, cells, ev_store)
        print(f"\n灌库完成({time.time()-t0:.0f}s):")
        for s in trace["ingest"]:
            cc = s["closed_cells"]
            print(f"  s{s['session']} @{s['dt'][:10]}: {len(s['exchanges'])} exchanges -> "
                  f"{len(cc)} cells / 全局 {s['atoms_total']} atoms ({s['secs']}s)")

    # 只灌库:落一份灌库 trace 便于核对,不进作答(批量先灌满再统一作答)
    if args.ingest_only:
        n_atoms = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
        n_cells = db.fetch_one("SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s", (uid,))["n"]
        print(f"\n[ingest-only] {lc.sample_id} 灌库结束:user={uid} atoms={n_atoms} cells={n_cells}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix(".ingest.json")).write_text(
            json.dumps({"meta": trace["meta"], "ingest": trace["ingest"],
                        "atoms": n_atoms, "cells": n_cells}, ensure_ascii=False, indent=1))
        print(f"灌库 trace 写入 {out.with_suffix('.ingest.json')}")
        return

    # —— 选题:全部 evidence 落在已灌 session 内 ——
    sidx = set(range(1, args.n_sessions + 1))
    answerable = pick_answerable(lc, sidx)
    if args.questions_file:
        wanted = [q.strip() for q in json.loads(
            Path(args.questions_file).read_text(encoding="utf-8"))]
        by_text = {q.question.strip(): q for q in answerable}
        qas = [by_text[w] for w in wanted if w in by_text]
        miss = len(wanted) - len(qas)
        if miss:
            print(f"警告:{miss}/{len(wanted)} 题未命中(evidence 范围外或题面不一致)")
    else:
        qas = pick_diverse(answerable, args.n_questions)
    print(f"\n可答题 {len(answerable)} 道(evidence 全在前 {args.n_sessions} session、剔 cat5),抽 {len(qas)} 道作答\n")

    # —— 逐题:新快链一条龙 → answerer → judge(conv 内 N 题并发)——
    reranker = MaasReranker(embedder)   # R2 真精排(qwen3-reranker;与产品 /recall 同链路)

    def answer_one(i: int, qa) -> dict:
        """一道题的完整作答+判分,返回 trace 记录。并发安全:
        - 每题独立 session_id({conv}-qa-{i}),互不污染 build_history / session_context;
        - 只读 atoms/cells/evidence(DB engine 连接池线程安全);LLM 客户端无共享可变态。
        """
        t1 = time.time()
        for attempt in range(1, 4):   # 逐题重试×3,间隔 60s(单题网络抖动不拖垮整轮)
            try:
                o = run_recall(llm, embedder, atoms, cells, ev_store,
                               session_id=f"{lc.sample_id}-qa-{i}", query=qa.question,
                               now_dt=last_dt, mode=args.mode, reranker=reranker)
                if o.deep and not o.ranked:
                    groups = [(c, atoms.list_by_cell(c.id))
                              for c in (cells.get(cid) for cid in (o.ans.cited_cells
                                                                   if o.ans else [])) if c]
                else:
                    groups = [(h.cell, [sa.atom for sa in h.atoms])
                              for h in o.ranked[:_ANSWER_MEM_CELLS]]
                mem_atoms = [a for _, atom_list in groups for a in atom_list]
                answer = answer_mode_a(llm, question=qa.question, brief=o.ans.answer,
                                       mem_block=_grouped_materials(groups))
                v = judge(llm, question=qa.question, gold=qa.answer, prediction=answer)
                try:
                    v5 = judge(llm, question=qa.question, gold=qa.answer,
                               prediction=o.ans.answer if o.ans else "")
                except Exception:   # noqa: BLE001
                    logger.exception("产品口径 judge 失败")
                    v5 = Verdict(ok=False, raw="ERROR: judge_r5 失败(见 run log)")
                rec = _qa_trace(o, qa, answer, v, v5, round(time.time() - t1, 1), mem_atoms)
                for m, a in zip(rec["memories"], mem_atoms):
                    m["evidence"] = [{"holder": e["holder"], "content": _cap(e["content"], 400),
                                      "at": (e.get("captured_at") or "")[:10]}
                                     for e in evidence_entries(a, ev_store)]
                verdict = o.reviews[-1].verdict if o.reviews else "?"
                deep_part = f", deep={len(o.deep.steps)}步" if o.deep else ""
                print(f"[{i}/{len(qas)}] {'✓' if v.ok else '✗'}/{'✓' if v5.ok else '✗'} "
                      f"cat{qa.category} ({rec['secs']}s, 核判={verdict}{deep_part}) "
                      f"Q: {qa.question[:60]}", flush=True)
                return rec
            except Exception:   # noqa: BLE001
                logger.exception(f"Q{i} 第 {attempt}/3 次失败")
                if attempt < 3:
                    time.sleep(60)
        print(f"[{i}/{len(qas)}] ✗/✗ cat{qa.category} (3 次重试失败) Q: {qa.question[:60]}", flush=True)
        return {"category": qa.category, "question": qa.question, "gold": qa.answer,
                "evidence": qa.evidence, "secs": round(time.time() - t1, 1),
                "memories": [], "answer": "", "judge": False,
                "judge_raw": "ERROR: 3 次重试仍失败(见 run log)",
                "judge_r5": False, "judge_r5_raw": "ERROR: 3 次重试仍失败"}

    # 线程池并发;结果按题号回填,保证 qa_records 顺序与串行一致(可复现/可 diff)
    results: dict[int, dict] = {}
    conc = max(1, args.concurrency)
    print(f"作答并发度: {conc}\n", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(answer_one, i, qa): i for i, qa in enumerate(qas, 1)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    qa_records = [results[i] for i in sorted(results)]

    # —— 库全貌(评测后的最终态)——
    trace["store"] = {
        "cells": [{"id": c.id, "session": c.session_id, "topic": c.topic,
                   "domains": c.domains,
                   "t_start": c.t_start.isoformat()[:19] if c.t_start else "",
                   "t_end": c.t_end.isoformat()[:19] if c.t_end else "",
                   "episode": _cap(c.episode, 400),
                   "n_atoms": len(atoms.list_by_cell(c.id))}
                  for c in cells.iter_all(limit=10000)],
        "atoms": [_atom_brief(a) for a in atoms.list(limit=100000)],
        "evidence": [{"id": e.id, "holder": e.holder,
                      "at": e.captured_at.isoformat()[:19] if e.captured_at else "",
                      "content": _cap(e.content_inline, 500)}
                     for e in ev_store.iter_all()],
    }
    # —— 金标证据链与断点:每道题(尤其错题)可定位到"错误发生在哪一环"——
    if trace["ingest"]:
        # 刚灌过库:ingest 记录里 dia↔evidence_id 精确对应(合并轮的多 dia 共享一条证据)
        dia2ev = {d: ex["evidence_id"]
                  for s in trace["ingest"] for ex in s["exchanges"]
                  for d in filter(None, (ex["dia"] or "").split(","))}
    else:   # --skip-ingest:只能按文本前缀对齐
        dia2ev = _dia_evidence(lc, trace["store"]["evidence"])
    for q in qa_records:
        q["gold_chain"] = _gold_chain(q, dia2ev, trace["store"]["atoms"])
        q["break_point"] = _break_point(q, q["gold_chain"])
    n_ok = sum(r["judge"] for r in qa_records)
    n_ok5 = sum(bool(r.get("judge_r5")) for r in qa_records)   # 产品口径
    by_cat = defaultdict(lambda: [0, 0])
    for r in qa_records:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["judge"])
    verdicts = defaultdict(int)
    for r in qa_records:
        verdicts[(r.get("review") or {}).get("verdict", "?")] += 1
    trace["qa"] = qa_records
    trace["summary"] = {"score": n_ok, "score_r5": n_ok5, "n_questions": len(qa_records),
                        "by_category": {str(k): v for k, v in sorted(by_cat.items())},
                        "verdicts": dict(verdicts),
                        "retried": sum(bool((r.get("review") or {}).get("retried")) for r in qa_records),
                        "escalated": sum(bool((r.get("deep") or {}).get("escalated"))
                                         for r in qa_records)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".trace.json")).write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    n = max(1, len(qa_records))
    print(f"\n===== 结果 Mem0口径(answerer): {n_ok}/{len(qa_records)} ({100*n_ok/n:.0f}%) · "
          f"产品口径(R5直判): {n_ok5}/{len(qa_records)} ({100*n_ok5/n:.0f}%) =====")
    for cat, (ok, tot) in sorted(by_cat.items()):
        print(f"  cat{cat}: {ok}/{tot}")
    print(f"核判判级: {dict(verdicts)}")
    print(f"trace 已写入 {out.with_suffix('.trace.json')}")

    # 渲染 HTML 全流程报告 + markdown 最终报告(自包含,浏览器/编辑器直接打开)
    from scripts.bench.report import render, render_md
    html_path = out.with_suffix(".report.html")
    render(trace, html_path)
    md_path = out.with_suffix(".report.md")
    render_md(trace, md_path)
    print(f"全流程报告: {html_path}\n最终报告(总分/分题型/错题分析): {md_path}")
    embedder.close(); llm.close(); db.close()


if __name__ == "__main__":
    main()
