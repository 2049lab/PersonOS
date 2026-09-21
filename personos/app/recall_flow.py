"""召回编排(融合架构 §5 决策 1/2):快链一条龙 R0→R1(atom 池)→单元组装(链织写)→R2→R5→R3'核判;深轨分岔。

对外服务 API 调它渲染干净视图;评测/场景脚本也直调。入参传显式依赖(不耦合 rt/FastAPI),
便于测试与复用。

分岔:mode=deep 只跑 R0(给深轨resolved/日期窗/域)后直达深轨 agent(独立评测口径);
mode=auto 快链全跑——R5 先出草稿,R3' 核判「草稿+同款材料」:答案缺陷 → 带指正重答一次,
仍缺陷或材料不足 → 升深轨(escalated=True),深轨终答覆盖快链作答。
深轨崩溃不拖垮主链路:回退快链作答,escalated 如实保留。

评测断点统计:每问一条汇总 INFO(recall 完成 mode=… R0=..s R1=..s units=..s R2=..s R5=..s …),
各工位另有自己的 INFO/WARNING;按 trace-id 聚起来即可定位"错在哪一环"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from ..online.arbitrate import ReviewResult, review_answer
from ..online.chain_face import UnitAssembly, assemble_units
from ..online.deep_recall import DeepOutcome, run_deep
from ..online.rerank import NoopReranker, Reranker, rerank_cells
from ..online.retrieval import (
    AtomHit, CellHit, MemoryAnswer, QueryRewrite, answer_from_cells, rewrite_query, search_atoms,
)
from ..online.session_context import build_history
from ..online.visual_query import VisualRewrite, enrich_query_with_image
from ..storage.atom_store import AtomStore
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.evidence_store import EvidenceStore

# 对外三档:auto(自动升级)| fast(只快链)| deep(深轨直达,独立评测)
PUBLIC_MODES = ("auto", "fast", "deep")


def _fmt_day(dt) -> str:
    """日期格式化,兼容 datetime / ISO 串 / None。rewrite 的 time_start/end 可能是字符串
    (LLM JSON 原样带出),此前按 datetime 格式化会 ValueError 把 recall 拖成 502。"""
    if not dt:
        return ""
    if isinstance(dt, str):
        return dt[:10]                      # ISO 串取日期部分;非 ISO 截断亦无害
    try:
        return f"{dt:%Y-%m-%d}"
    except (ValueError, TypeError):
        return str(dt)[:10]


def _no_answer_note(rw, review, deep) -> str:
    """无答案时的客观交代:查了什么 / 结论 / 可能原因。

    记忆框架的诚实义务:答不出时不给似是而非的结论,也不静默空手而归——
    把检索范围与缺口说清楚,调用方可据此换问法或补记忆。确定性拼装,不调 LLM。
    """
    parts = []
    if rw:
        scope = []
        if rw.subject:
            scope.append(f"主体「{rw.subject}」")
        tw = f"{_fmt_day(rw.time_start)}~{_fmt_day(rw.time_end)}".strip("~")
        if tw:
            scope.append(f"时间窗 {tw}")
        if rw.domains:
            scope.append("域 " + "/".join(rw.domains))
        parts.append("检索范围:" + ("、".join(scope) if scope else "全库语义检索") + ";")
    if deep:
        parts.append("结论:深轨多步翻阅记忆后,未能在限定步数内定位到可作答的记忆。")
    else:
        parts.append("结论:记忆库中没有足以回答该问题的相关记忆。")
    reason = "该信息可能从未在过往对话中出现,或出现时未被收入记忆(检索索引未覆盖)。"
    if review and review.critique:
        reason += f"缺口:{review.critique}。"
    parts.append("可能原因:" + reason)
    return "".join(parts)


@dataclass
class RecallOutcome:
    """一次召回的原始产物,交给各 API 层各自渲染视图。"""
    query: str
    mode: str
    rw: QueryRewrite | None = None      # R0 五件套(无答案交代要引用检索范围;deep 直达也先跑 R0)
    hits: list[AtomHit] = field(default_factory=list)      # R1 atom 池(两路 RRF 融合序)
    ranked: list[CellHit] = field(default_factory=list)    # R2 精排后的材料单元序(普通+织写)
    draft: MemoryAnswer | None = None        # R5 首份草稿(核判/重答前的原始作答)
    reviews: list[ReviewResult] = field(default_factory=list)   # 核判历史(0~2 条)
    retried: bool = False                    # 判 answer_defect 后重答过一次
    ans: MemoryAnswer | None = None          # 终答(ok 草稿/重答案/auto 升级时=深轨作答)
    escalated: bool = False                  # auto 且核判仍缺陷或材料不足 → 升了深轨
    deep: DeepOutcome | None = None          # 深轨产物(轨迹/写回数;未跑深轨为 None)
    asm: UnitAssembly | None = None          # 单元组装透视(池/链/织写/普通计数 + 残缺提示)
    vis: VisualRewrite | None = None       # 视觉改写(仅当调用方带了图片;None=纯文本召回)
    secs: dict[str, float] = field(default_factory=dict)   # 各工位耗时(汇总日志/评测统计用)


def run_recall(
    llm, embedder, atoms: AtomStore, cells: CellStore, evidence: EvidenceStore,
    *,
    session_id: str,
    query: str,
    now_dt: datetime,
    mode: str = "auto",
    top_k: int = 30,
    rewrite: bool = True,
    reranker: Reranker | None = None,
    media_store=None, mllm=None,   # 图片看图依赖(透传给深轨;未注入=深轨读 content_inline 不看图)
    image: bytes | None = None, image_content_type: str = "image/jpeg",   # 调用方随问题带的图
    visual_deps=None,      # 视觉改写依赖(rt.visual_deps);与 image 同时给才触发,否则纯文本链路
    profile_full: str = "", profile_traits: str = "",   # 画像注入(full→R0/深轨;traits→R5);空=无画像,行为与今天一致
    scenario: str = "",   # 业务方场景描述(可空)→ R0/R5/深轨;空=默认链路逐字节不变
) -> RecallOutcome:
    """快链完整编排:R0 预处理 → R1 两路 atom 检索(联想/域 RRF,top_k 池)→ 单元组装
    (atom→链去重,≥2 节点链织写 memcell′,单节点/游离→普通格)→ R2 精排(单元统一精排)
    → R5 草稿 → R3' 核判(草稿+同款材料)。mode=deep 直达深轨;auto 判答案缺陷先重答一次,
    仍缺陷/材料不足升深轨覆盖终答。"""
    out = RecallOutcome(query=query, mode=mode)
    t0 = time.perf_counter()

    def mark(stage: str):
        out.secs[stage] = round(time.perf_counter() - t0, 3)   # 累计锚点:各段耗时 = 差分

    # 指代补全的依据(滚动摘要 + 近几轮)。带 cells:视频段折叠成 episode 而非 20+ 行逐字对白。
    history = build_history(evidence, session_id, llm=llm, cell_store=cells)
    mark("hist")

    # R0 之前的视觉理解:把图里的人/场景写进 query,让后面纯文本的链路也能用上视觉信息。
    # 只在调用方带了图片时触发;失败恒退回原 query(见 visual_query),纯文本召回逐字节不变。
    q0 = query
    if image is not None and visual_deps is not None:
        out.vis = enrich_query_with_image(
            visual_deps, query=query, image=image, content_type=image_content_type,
            history=history, scenario=scenario, now_dt=now_dt)
        q0 = out.vis.query
    mark("vis")

    out.rw = (rewrite_query(llm, raw_query=q0, history=history, now_dt=now_dt,
                            profile=profile_full, scenario=scenario)
              if rewrite else QueryRewrite(original=q0, resolved=q0))
    mark("R0")

    def run_deep_safe(**kw) -> DeepOutcome | None:
        """深轨包装:崩溃不拖垮主链路(回退快链作答),escalated 如实保留。"""
        try:
            return run_deep(llm, embedder, atoms, cells, evidence, reranker=reranker,
                            media_store=media_store, mllm=mllm, profile=profile_full,
                            scenario=scenario, **kw)
        except Exception as e:   # noqa: BLE001
            logger.exception(f"深轨失败,回退快链结果 q={query!r}: {e}")
            return None

    if mode == "deep":
        # 直达深轨(独立评测口径):R0 的 resolved/日期窗/域喂交接包,快链工位全跳过
        out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw)
        out.ans = out.deep.ans if out.deep else MemoryAnswer(answer="")
        mark("deep")
    else:
        pool = search_atoms(embedder, atoms, rewrite=out.rw, top_n=top_k)
        out.hits = pool.atoms
        mark("R1")
        # 单元组装(§5.2):池 atom → 链去重 → 织写 memcell′/普通单元 + 通用残缺提示(D-C10)
        out.asm = assemble_units(out.hits, ChainStore(atoms.db, atoms.user_id), cells,
                                 llm, query=out.rw.resolved, beyond=pool.beyond)
        mark("units")
        materials = out.asm.units
        if not materials:
            # 空检索短路(§5 决策 1):不作答不核判,直接判空;auto 升深轨
            out.ans = MemoryAnswer(answer="")
            mark("R5")
            if mode == "auto":
                out.escalated = True
                out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw,
                                         review=None, fast_hits=[])
                if out.deep and out.deep.ans.answer:
                    out.ans = out.deep.ans   # 深轨终答覆盖;空答不覆盖
                mark("deep")
        else:
            # R2 精排:普通格与织写 memcell′ 同构统一精排(临时视图,无特殊逻辑);
            # 下游(R5/R3'/重答/深轨交接)一律消费精排序——否则真 reranker 白跑
            out.ranked = rerank_cells(reranker or NoopReranker(), out.rw.resolved, materials)
            materials = out.ranked
            mark("R2")
            boundary = out.asm.boundary
            # R5 先答(草稿),核判看「草稿+同款材料」逐条核对(§5 决策 1 的重排)
            out.draft = answer_from_cells(llm, query=out.rw.resolved, subject=out.rw.subject,
                                          hits=materials, now_dt=now_dt, boundary=boundary,
                                          profile=profile_traits, scenario=scenario)
            out.ans = out.draft
            mark("R5")
            out.reviews.append(review_answer(llm, query=query, draft=out.draft, hits=materials,
                                             resolved=out.rw.resolved, subject=out.rw.subject,
                                             boundary=boundary))
            mark("R3")
            # 双层处置(§5 决策 2):答案缺陷 → 带指正重答一次;材料不足 → 直接深轨
            if out.reviews[-1].verdict == "answer_defect":
                out.ans = answer_from_cells(llm, query=out.rw.resolved, subject=out.rw.subject,
                                            hits=materials, now_dt=now_dt, boundary=boundary,
                                            feedback=out.reviews[-1].critique, profile=profile_traits,
                                            scenario=scenario)
                out.retried = True
                mark("R5r")
                out.reviews.append(review_answer(
                    llm, query=query, draft=out.ans, hits=materials,
                    resolved=out.rw.resolved, subject=out.rw.subject, boundary=boundary))
                mark("R3r")
            verdict = out.reviews[-1].verdict if out.reviews else "ok"
            if mode == "auto" and verdict != "ok":
                out.escalated = True
                out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw,
                                         review=out.reviews[-1] if out.reviews else None,
                                         fast_hits=materials)
                if out.deep and out.deep.ans.answer:
                    out.ans = out.deep.ans      # 深轨终答覆盖;空答不覆盖(保留快链说法)
                mark("deep")

    stages = [s for s in ("hist", "R0", "R1", "units", "R2", "R5", "R3", "R5r", "R3r", "deep")
              if s in out.secs]
    parts = " ".join(f"{b}={out.secs[b] - out.secs[a]:.1f}s" for a, b in zip(stages, stages[1:]))
    deep_part = (f" deep_steps={len(out.deep.steps)} remembered={out.deep.remembered}"
                 if out.deep else "")
    unit_part = (f" units={len(out.asm.units)}/{out.asm.n_chains}ch/{out.asm.n_woven}wv"
                 if out.asm else "")
    logger.info(f"recall 完成 mode={mode}{' escalated' if out.escalated else ''}{deep_part} "
                f"{' retried' if out.retried else ''} {parts} pool={len(out.hits)}{unit_part} "
                f"verdict={out.reviews[-1].verdict if out.reviews else '-'} "
                f"cited={len(out.ans.cited_cells) if out.ans else 0} "
                f"ans={len(out.ans.answer) if out.ans else 0}字 q={query!r}")
    # 无答案兜底:空答案 → 客观交代(查了什么/结论/原因),不静默空手而归
    if out.ans is None or not out.ans.answer.strip():
        out.ans = MemoryAnswer(answer=_no_answer_note(out.rw, out.reviews[-1] if out.reviews else None,
                                                      out.deep))
    # 召回结果全貌:终答 + 引用 cell + 精排材料单元(topic/命中 atom 文本),排障用
    ranked_detail = "\n".join(
        f"    [{i}] score={h.rerank_score if h.rerank_score is not None else h.score:.3f} "
        f"topic={h.cell.topic!r} atoms={[a.atom.text for a in h.atoms]}"
        for i, h in enumerate(out.ranked, 1)) if out.ranked else "    (无)"
    logger.info(f"召回结果 q={query!r} cited={out.ans.cited_cells}\n"
                f"  ── 终答 ──\n{out.ans.answer}\n"
                f"  ── 精排材料单元(n={len(out.ranked)})──\n{ranked_detail}")
    return out
