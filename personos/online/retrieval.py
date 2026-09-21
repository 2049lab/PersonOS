"""快链召回(融合架构 §3):R0 预处理 → R1 两路 atom 检索+RRF → R5 作答。

检索单元是 atom(联想/域两路,topic 路已删);材料单元是 memcell(普通格或链织写的
memcell′ 临时视图,同构无感知)——atoms 不进任何精排/作答材料(检索单元不充当参考答案)。
写入不消解——重复=冗余索引,冲突在作答时消费(D1)。
开段(未闭合)内容不在任何向量池里,当前对话靠会话上下文回答。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
from loguru import logger

from ..models import (
    DOMAIN_VOCAB, MemCell, MemoryAtom, atom_anchor, ensure_aware, now, vocab_menu,
)
from ..storage.atom_store import AtomStore
from .llm import ChatLLM, chat_json, with_scenario

# 业务方场景注入 directive(只调关注度/详略,不改事实、不编造、不漏)——见 with_scenario
_SCEN_DIR_REWRITE = ("When the question is ambiguous, bias domain guesses and expansion terms toward "
                     "the caller's subject area; never override the literal question.")
_SCEN_DIR_ANSWER = ("Use it only to shape emphasis and level of detail in the answer; it never changes "
                    "which facts are true, nor licenses stating anything absent from the materials.")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


# —— 命中结构:atom 是检索单元(R1),memcell 是材料/精排单元(R2 起) ——

@dataclass
class AtomHit:
    atom: MemoryAtom
    similarity: float        # 该 atom 对查询面的最大 cosine(联想/域两路取最优)
    rrf: float = 0.0         # RRF 融合分(排名量纲;深轨单路检索不用,保持 0)


@dataclass
class CellHit:
    """材料单元:普通 memcell 或链织写的 memcell′(临时视图,下游完全无感知)。

    memcell′:cell.id=链id、topic=链title、episode=织文、t_start/t_end=成员格跨度,
    covers=成员格 id 全集——covers 只用于引用长短映射(mN → 成员格),不进任何 prompt 文本。
    普通单元 covers 空,引用展开视为 [自身 cell.id]。
    """

    cell: MemCell
    score: float                     # 单元最佳 atom 的 RRF 融合分(排名量纲,非相似度)
    best_sim: float                  # 单元最佳 atom 相似度(可解释:语义近不近)
    atoms: list[AtomHit] = field(default_factory=list)   # 该单元命中的池内 atoms(相似度降序)
    rerank_score: float | None = None                   # R2 精比分(未跑 rerank 时 None)
    covers: list[str] = field(default_factory=list)     # 织写单元=成员 cell id 全集;空=视为[自身 cell.id]


def _cosine(qmat: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """qmat:(f,d) 查询面 × mat:(n,d) 文档 → (n,f) 余弦矩阵。零向量安全。"""
    if mat.size == 0:
        return np.zeros((0, qmat.shape[0]), dtype=np.float32)
    qn = qmat / (np.linalg.norm(qmat, axis=1, keepdims=True) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mn @ qn.T


def _maxsim_ranking(
    query_vecs: list[np.ndarray], pool: list[tuple[MemoryAtom, np.ndarray]],
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """MaxSim:每条 atom 对各查询面取最大 cosine,按 memcell_id 聚合取 max → cell 排名。

    一个 cell 只出它最强的 atom(候选池自动多样化,防单个富 cell 灌满)。
    返回 (cell 排名序列[相似度降序], cell→MaxSim 分, cell→atom→最佳面相似度)。
    """
    if not pool:
        return [], {}, {}
    qmat = np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs])
    sims = _cosine(qmat, np.stack([v for (_, v) in pool])).max(axis=1)   # 每条 atom 的最佳面
    cell_scores: dict[str, float] = {}
    atom_sims: dict[str, dict[str, float]] = {}
    for (atom, _), s in zip(pool, sims):
        atom_sims.setdefault(atom.memcell_id, {})[atom.id] = float(s)
        s = float(s)
        if s > cell_scores.get(atom.memcell_id, -1.0):
            cell_scores[atom.memcell_id] = s
    ranking = sorted(cell_scores, key=lambda c: cell_scores[c], reverse=True)
    return ranking, cell_scores, atom_sims


def _vec_ranking(query_vecs: list[np.ndarray], entries: list[tuple[MemCell, np.ndarray]]) -> list[str]:
    """1 cell 1 向量的池(如 topic 向量):对各查询面取最大 cosine → cell id 排名(降序)。"""
    if not entries:
        return []
    sims = _cosine(np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs]),
                   np.stack([v for (_, v) in entries])).max(axis=1)
    order = sorted(range(len(entries)), key=lambda i: sims[i], reverse=True)
    return [entries[i][0].id for i in order]


_RRF_K = 60   # RRF 平滑常数(融合架构 §3 定死;k 越大名次差异越平)


def _rrf(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """多路排名 → 融合分:Σ 1/(k+rank),rank 从 1 起。

    多路都认的浮上来,单路噪声被稀释;只在一路出现的条目只累计那一路的贡献。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


# —— R0 · 查询预处理:一次轻 LLM 产五件套(qtype 已拆,D-C1)——
_REWRITE_SYS = (
    "# Role\n"
    "You are the query preprocessor of a memory retrieval system: turn the user's CURRENT QUESTION into "
    "the five retrieval fields below.\n\n"
    "# Input\n"
    "Current time (the anchor for converting relative times), Recent dialogue (the basis for resolving "
    "references), Current question.\n\n"
    "# Output (strict JSON, all five fields at once)\n"
    "1. resolved: apply exactly TWO minimal edits to the question and change nothing else —\n"
    "   (a) resolve references/ellipsis: replace 'it / that / afterwards / this matter' with the actual "
    "referent from the Recent dialogue; no reference word may remain in resolved;\n"
    "   (b) relative → absolute time: convert 'last week / a few days ago / yesterday / last year' into "
    "absolute dates using the Current time (e.g. if now is 2026-08-25, 'yesterday' → '2026-08-24').\n"
    "   Do not stuff dates into a question that carries no time meaning; apart from these two edits keep "
    "the original wording — no rephrasing, no added interpretation.\n"
    "2. subject: whose affairs the question asks about — 'what have I been busy with' → 'user'; "
    "'how is Caroline's exhibition going' → 'Caroline'. Not about a specific person, or unclear → null.\n"
    "3. expansions: 0-5 retrieval-boosting terms or paraphrases, in the SAME language as the question "
    "('leg pain' → 'sports; injury'; 'busy with which project' → 'preparing which event; any plans'). "
    "When the subject is a proper name, include name-carrying paraphrases of the asked relation "
    "('what movie did Joanna watch' → 'Joanna's movies; film Joanna saw'; 'what did Nate do for "
    "Joanna' → \"Nate's gift to Joanna; Nate's favor for Joanna\") — memories may phrase the fact in "
    "first person ('user') or third person, and a name-carrying face bridges the wording gap. "
    "Be restrained: no wild leaps, no far-fetched associations; when unsure, give fewer or [].\n"
    "4. time_start / time_end: when the question carries its own time constraint, convert it into an "
    "absolute-date window (ISO yyyy-MM-dd) using the Current time — 'what did we talk about last week' → "
    "start = that Monday, end = that Sunday. No time constraint → both null.\n"
    "5. domains: which life domains the ANSWER will live in (not which domain the question's surface "
    "belongs to), 0-3, Dxx codes only. Vocabulary:\n  "
    + vocab_menu(DOMAIN_VOCAB) + "\n"
    "   Memories are filed by domain — a correct guess lets the domain route hit fast. Guess when you "
    "reasonably can (1-3 codes); don't stay empty out of over-caution, and don't force clearly "
    "irrelevant domains. [] only when there is truly no basis.\n\n"
    "# Output format\n"
    'JSON only, no extra text: {"resolved":"the full question after reference resolution and '
    'relative-to-absolute time conversion","subject":"user|name|null","expansions":["reasonable '
    'terms"],"time_start":"2026-08-11|null","time_end":"2026-08-17|null","domains":["D05"]}'
)


@dataclass
class QueryRewrite:
    original: str
    resolved: str               # 指代补全、相对时间转绝对后的问题(检索与作答都用它)
    subject: str = ""           # 疑问主体(R5 归属校验锚点;"user"/人名;空=未判出)
    expansions: list[str] = field(default_factory=list)   # 扩展词(联想路查询面)
    time_start: str = ""        # 绝对日期窗(ISO yyyy-MM-dd;空=无界;直供深轨工具 start/end_date)
    time_end: str = ""
    domains: list[str] = field(default_factory=list)      # 域路锚点(D 轴)
    system: str = ""
    user: str = ""
    raw: str = ""


def _iso_date(v) -> str:
    """LLM 输出 → 规整 ISO 日期串(yyyy-MM-dd);null/解析失败 → 空串(该侧无界)。"""
    d = ensure_aware(v)
    return d.date().isoformat() if d else ""


def rewrite_query(llm: ChatLLM, *, raw_query: str, history: list[tuple[str, str]] | None = None,
                  now_dt: datetime | None = None, profile: str = "", scenario: str = "") -> QueryRewrite:
    """R0:一次 LLM 产五件套(补全指代/疑问主体/日期窗/扩展词/判域),解析失败退化为原 query(容错不阻塞)。

    profile:用户画像文本块(可空)——注入以补全指代/省略、主体判定、时间习惯换算。
    scenario:业务方场景描述(可空)——偏置判域/扩展词到调用方主题域。
    """
    now_dt = now_dt or now()
    hist = "\n".join(f"{h}: {t}" for h, t in (history or [])) or "(no history)"
    prof = f"\n\n{profile}" if profile else ""
    user = (f"Current time: {now_dt.isoformat()} (the anchor for converting relative times in the "
            f"dialogue into absolute dates)\n\nRecent dialogue:\n{hist}{prof}\n\nCurrent question:\n{raw_query}")
    sys = with_scenario(_REWRITE_SYS, "# Input", scenario, _SCEN_DIR_REWRITE)
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    try:
        obj, raw = chat_json(llm, messages, max_tokens=800, temperature=0.2, stage="rewrite_query")
        resolved = str(obj.get("resolved") or "").strip() or raw_query
        subject = str(obj.get("subject") or "").strip()
        if subject.lower() in ("null", "none", "无"):
            subject = ""
        exp = [s for s in (str(x).strip() for x in (obj.get("expansions") or obj.get("associations") or []))
               if s][:5]
        doms = [d for d in (str(x).strip() for x in (obj.get("domains") or [])) if d in DOMAIN_VOCAB]
        ts, te = _iso_date(obj.get("time_start")), _iso_date(obj.get("time_end"))
        logger.info(f"R0 改写 subject={subject or '-'} exp={exp} "
                    f"窗={ts or '-'}/{te or '-'} 域={doms or '[]'} "
                    f"resolved={resolved!r}")
        return QueryRewrite(original=raw_query, resolved=resolved, subject=subject,
                            expansions=exp, time_start=ts, time_end=te, domains=doms,
                            system=sys, user=user, raw=raw)
    except Exception as e:
        logger.warning(f"R0 改写解析失败,退回原 query: {e}")
        return QueryRewrite(original=raw_query, resolved=raw_query,
                            system=sys, user=user,
                            raw=getattr(e, "raw", "") or str(e))


# —— R1 · 两路召回:联想路(发散,全库) ∥ 域路(收敛,域内子集)→ atom 级 RRF ——

_PER_CELL_CAP = 10   # 池内同格 atom 数上限(防富格灌满挤掉别家——atom 级池选择的护栏)


def _date_window(start_date: str, end_date: str) -> tuple[datetime | None, datetime | None]:
    """yyyy-MM-dd 串 → (下界, 上界+1天);日期窗按【含端点一整天】算。坏值忽略该侧。

    快链(无过滤)与深轨工具(search_atoms/find_cells)共用同一份窗语义,防两处各写漂移。
    """
    lo = ensure_aware(start_date) if start_date else None
    hi = ensure_aware(end_date) if end_date else None
    if hi:
        hi = hi + timedelta(days=1)
    return lo, hi


def _in_window(anchor: datetime | None, lo: datetime | None, hi: datetime | None) -> bool:
    """锚点是否落在窗内;时间筛选下无锚对象不可见(不知道何时成立,不能冒充实效)。"""
    return not ((lo and (anchor is None or anchor < lo)) or (hi and (anchor is None or anchor >= hi)))


def _atom_maxsim(query_vecs: list[np.ndarray],
                 rows: list[tuple[MemoryAtom, np.ndarray]]) -> dict[str, float]:
    """每条 atom 对各查询面取最大 cosine → {atom_id: sim}(atom 级排名原料,无 cell 聚合)。"""
    if not rows:
        return {}
    qmat = np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs])
    sims = _cosine(qmat, np.stack([v for (_, v) in rows])).max(axis=1)
    return {a.id: float(s) for (a, _), s in zip(rows, sims)}


@dataclass
class AtomPool:
    """R1 产物:RRF 融合序 atom 池 + 池外候选(残缺提示点名其链 title 用)。

    beyond 只收池满之后的名次——被同格上限挤出的 atom 其事实仍在所示格 episode 里,不算缺料。
    """
    atoms: list[AtomHit] = field(default_factory=list)
    beyond: list[AtomHit] = field(default_factory=list)


def search_atoms(
    embedder: Embedder, atom_store: AtomStore,
    *,
    rewrite: QueryRewrite,
    top_n: int = 30,
    per_cell_cap: int = _PER_CELL_CAP,
    start_date: str = "",
    end_date: str = "",
    holder: str = "",
    domains_filter: list[str] | None = None,
) -> AtomPool:
    """两路 atom 检索 → RRF(k=60) 融合 → 池选择,取 top_n 个 atom(D-C6:召回单位 = atom)。

    - 联想路:resolved + 各扩展词【分别 embed】成多个查询面,每条 atom 取对面最大 cosine,
      对全库 atom 池——召回优先,防漏。
    - 域路:resolved 原样不扩展,只对判出域内的 atom 子集——精度优先,防误
      (域对但语义没匹配上的仍可命中)。判不出域 → 该路空,RRF 自然退化为单路。
    - 每路各取前 2×top_n 名进 RRF(过采样),融合后取 top_n。
    - 池选择:同格最多 per_cell_cap 个 atom;池满后其余名次进 beyond(链面残缺提示用)。
    - start/end_date / holder / domains_filter:深轨工具的结构化【硬过滤】(快链全留空);
      日期窗按 atom 锚点(occurrence_time,回退 recorded_at),含端点一整天。
      注意 domains_filter ≠ rewrite.domains:前者是硬过滤(agent 显式设的条件),
      后者是域路锚点(R0 猜的召回提示,不滤掉任何 atom)。
    """
    lo, hi = _date_window(start_date, end_date)
    rows = []
    for a, v in atom_store.all_with_embeddings():
        if not a.memcell_id:
            continue
        if (lo or hi) and not _in_window(atom_anchor(a), lo, hi):
            continue
        if holder and a.holder != holder:
            continue
        if domains_filter and not (set(a.domains) & set(domains_filter)):
            continue
        rows.append((a, v))

    # 联想路查询面(face[0] 恒为 resolved,域路复用该向量,只 embed 一次)
    faces = [rewrite.resolved] + list(rewrite.expansions)
    qvecs = list(embedder.embed(faces))
    assoc_sims = _atom_maxsim(qvecs, rows)
    assoc_ranking = sorted(assoc_sims, key=lambda aid: -assoc_sims[aid])

    dom_sims: dict[str, float] = {}
    if rewrite.domains and rows:
        dset = set(rewrite.domains)
        dom_sims = _atom_maxsim([qvecs[0]], [(a, v) for a, v in rows if dset & set(a.domains)])
    dom_ranking = sorted(dom_sims, key=lambda aid: -dom_sims[aid])

    route_depth = 2 * top_n                                # 每路过采样深度(60 = 2×30)
    fused = _rrf([assoc_ranking[:route_depth], dom_ranking[:route_depth]])
    best_sim = {aid: max(assoc_sims.get(aid, 0.0), dom_sims.get(aid, 0.0)) for aid in fused}
    order = sorted(fused, key=lambda aid: (-fused[aid], -best_sim[aid]))   # 并列按相似度定序

    atom_map = {a.id: a for a, _ in rows}
    per_cell: dict[str, int] = {}
    pool = AtomPool()
    for aid in order:
        atom = atom_map.get(aid)
        if atom is None:
            continue                                       # 防御:fused 键必来自 rows,理论不可达
        if len(pool.atoms) < top_n:
            n = per_cell.get(atom.memcell_id, 0)
            if n < per_cell_cap:
                per_cell[atom.memcell_id] = n + 1
                pool.atoms.append(AtomHit(atom=atom, similarity=best_sim[aid], rrf=fused[aid]))
            # 同格超限被挤出:该格事实已在所示 episode 里,不算缺料 → 跳过(不进 beyond)
        else:
            pool.beyond.append(AtomHit(atom=atom, similarity=best_sim[aid], rrf=fused[aid]))
    pool_detail = "\n".join(
        f"    [{i}] sim={h.similarity:.3f} rrf={h.rrf:.4f} {h.atom.text!r}"
        for i, h in enumerate(pool.atoms, 1))
    logger.info(f"R1 search_atoms rows={len(rows)} faces={len(faces)} 域={rewrite.domains or '[]'} "
                f"assoc={len(assoc_ranking)} 域路={len(dom_ranking)} pool={len(pool.atoms)} "
                f"beyond={len(pool.beyond)} q={rewrite.resolved!r}\n"
                f"  命中 atom 池(n={len(pool.atoms)}):\n{pool_detail}")
    return pool


# —— R5 · 作答:episode 主料 + 命中 atoms,一次 LLM ——

# 冲突消费规则(D1 的落点):作答与核判两个 prompt 共用同一份措辞,防多处复述漂移。
_MOST_RECENT_RULE = (
    "A fact stated several times and never mentioned again afterwards: the MOST RECENT statement "
    "is the current state — even if older statements are more frequent or more detailed."
)
CONFLICT_RULE = (
    "Conflict consumption: when multiple cells in the materials state different things about the same "
    "fact (one says 'loves apples', another says 'allergic to apples'), order them by the cell time "
    "window and treat the later one as the new state; present the evolution ('earlier ... later ...'); "
    "an explicit correction always wins. " + _MOST_RECENT_RULE + " Never average, never pick one "
    "arbitrarily."
)

# R5 v2(2026-09-09,错题归因:拒答桶 68 道的病根是规则罗列下模型挑最省力路径——软拒答写得
# 有理有据,核判对无 claim 草稿无从核对而放行)。规则罗列 → CoT 流程式,拒答抬高成本并留痕
# (逐块声明),给 R3' 提供可核对的抓手。保留全部硬规则语义(冲突/归属/保真/程度/引用)。
_ANSWER_SYS = (
    "# Role\n"
    "You are the answerer of a personal memory system: answer the USER QUESTION strictly from the given "
    "MEMORY MATERIALS.\n\n"
    "# How to read the materials\n"
    "Each material block is one topic unit, opened by a \"━━━ mN ━━━\" separator line; the next "
    "line is a header with the dialogue time and topic, followed by the segment narrative "
    "(episode) — the ONLY primary material. Times, attribution and qualifiers are already "
    "written into the narrative itself.\n\n"
    "# Working procedure (follow IN ORDER, then answer)\n"
    "STEP 1 — Scan: read EVERY block m1..mN and collect every fact relevant to the question; do not "
    "stop at the first hit — one overlooked block is a wrong answer.\n"
    "STEP 2 — Connect: link facts across blocks when the question needs it (who did what, where, "
    "why); the materials may state pieces in separate blocks — join them.\n"
    "STEP 3 — Infer when needed: you MAY draw a direct, single-step inference supported by the "
    "materials ('planned for 2026-09' → 'has not happened yet; it is a plan'; 'runs the shop "
    "personally, handling everything' → a small operation). Do NOT dismiss a reasonable inference as "
    "mere speculation; only avoid chaining several speculative leaps.\n"
    "STEP 4 — Time: use only two anchors — the dual-time format inside the materials "
    "('last week (2026-08-18)') and the Current time. Convert 'how long ago / is it still ...' "
    "against the Current time. Never do calendar arithmetic from memory, never use times from "
    "outside the materials, never guess a date the materials do not state.\n"
    "STEP 5 — Verify (list/count questions): re-scan every block for any matching item you missed, "
    "then count the distinct items the materials actually contain and verify your list has exactly "
    "that many. For COUNT questions ('how many ...'): list the distinct instances first, then count "
    "them; MERGE duplicates — the same event mentioned in several blocks counts ONCE — and exclude "
    "plans/intentions that never happened. Give the exact count, never 'at least N'.\n"
    "STEP 6 — Answer, observing these hard rules:\n"
    "- " + CONFLICT_RULE + "\n"
    "- Detail fidelity: full names/numbers/frequencies exactly as written in the materials — no "
    "generalization ('3 pizzas' stays '3 pizzas'; 'hot yoga' stays 'hot yoga', never 'exercise').\n"
    "- Degree questions: for qualitative/how-much questions where the materials give indirect but "
    "real evidence, answer at the degree the evidence supports ('shows real interest — brought it "
    "up twice herself') and state the evidential boundary; do not refuse the whole question just "
    "because the evidence is indirect, and do not inflate the degree beyond the evidence.\n"
    "- Attribution check: when the answer's subject is NOT the QUESTION SUBJECT, say 'that was said/"
    "done by X' and keep the attribution straight — never pass one person's matter off as the asked "
    "subject's. If the asked subject has nothing in the materials, say so honestly; never substitute "
    "someone else's.\n"
    "- Citations: besides answer, list the cell handles you relied on in cells (m1, m2 ...); the "
    "answer body itself must not contain handles.\n\n"
    "# Refusal is a last resort\n"
    "If the materials give ANY direct, indirect, or single-step-inferable evidence — even hedged — "
    "you MUST answer (state the evidential basis if indirect). If you can DESCRIBE the thing but "
    "hesitate to name it, commit to the specific name — describing-but-refusing-to-name counts as a "
    "refusal and is not allowed. Only when the materials contain genuinely nothing usable may you "
    "say the information is not found — and then the answer MUST state that you checked every block "
    "m1..mN, name the closest block (by handle) and what exactly it lacks; never a bare 'I don't "
    "know', never a plausible-sounding guess dressed as fact.\n\n"
    "# Requirements\n"
    "- Answer in the language of the question.\n"
    "- Style: a standard factual statement — third person, neutral, conclusion first, stating the facts "
    "directly; no conversational tone (never 'you told me / I remember / I guess'); the tone does not "
    "change with how the question is phrased. This is the memory service's factual output — the caller "
    "(an external agent) composes the conversation.\n"
    "- Include time cues (absolute dates). If the materials hold nothing relevant, say so honestly — no "
    "fabrication, no guessing.\n\n"
    "# Output (JSON only)\n"
    '{"answer":"...","cells":["m1"]}'
)


@dataclass
class MemoryAnswer:
    answer: str                 # 对问题的直接作答(空串=记忆无相关信息)
    cited_cells: list[str] = field(default_factory=list)   # 引用的 cell id(规则 6)
    system: str = ""
    user: str = ""
    raw: str = ""


def _date_str(dt: datetime | None) -> str:
    """日期 → 裸 'yyyy-MM-dd'(cell 时间窗用,外面不再套中括号);无日期 → 空串。"""
    dt = ensure_aware(dt)
    return f"{dt:%Y-%m-%d}" if dt else ""


def cell_lead(c: MemCell) -> str:
    """程序化元信息拼一行语言中立的材料头:对话时间(证据时间戳,无需 LLM 识别)+ topic。

    R2 精排文档与 R3/R5 材料块共用,保证三个工位看到的单元口径一致。
    英文固定格式(元数据框架不随库内容语言变,材料主体语言由 episode/topic 自带)。
    """
    ts, te = _date_str(c.t_start), _date_str(c.t_end)
    if ts and te:
        when = ts if ts == te else f"{ts} to {te}"
    else:
        when = "date unknown"
    return f"[dialogue {when} | topic: {c.topic or '(no topic)'}]"


def cell_block(hit: CellHit, handle: str) -> str:
    """一个材料单元:「━━━ 编号 ━━━」分隔行 + 元信息头(时间+topic)+ episode(唯一主料)。

    普通格与织写 memcell′ 同构渲染(临时视图,下游无感知):memcell′ 的 topic=链 title、
    时间=成员格跨度、episode=织文。covers 只影响引用展开(mN→成员格),不进材料文本。
    R3' 核判与 R5 作答共用同一渲染器(两个 prompt 的材料口径一致,不各写各的漂移)。
    atoms 不进材料:它们是检索单元(R1 定位用),喂给判级/作答器会被当"参考答案"混淆视听。
    """
    c = hit.cell
    return "\n".join([f"━━━ {handle} ━━━", cell_lead(c), c.episode or "(no episode)"])


# —— 材料渲染顺序(P1-B A/B 开关)——
# relevance=按 R2 精排序(现行基线);time_asc/time_desc=按 cell 时间窗排。
# CONFLICT_RULE 要求「后者覆盖前者」:时间序下 LLM 的自然阅读顺序即冲突消费顺序,
# 不再依赖它自己从各格 header 读日期重排(重排错一次就取旧值)。
# env 逐次读取(测试/多臂对比友好);A/B 定档后把默认值改为胜者。
_R5_ORDER_ENV = "PERSONOS_R5_ORDER"
_R5_ORDER_FLOOR = datetime(1, 1, 1, tzinfo=timezone.utc)   # t_start 缺失的排序下界(aware)


def _order_hits_for_answer(hits: list[CellHit], order: str) -> list[CellHit]:
    """按 order 重排作答材料;stable sort——同刻/无时刻的格保持精排序(相关度仍是同序参照)。"""
    if order == "relevance":
        return hits
    return sorted(hits, key=lambda h: ensure_aware(h.cell.t_start) or _R5_ORDER_FLOOR,
                  reverse=(order == "time_desc"))


def answer_from_cells(
    llm: ChatLLM, *, query: str, subject: str, hits: list[CellHit],
    now_dt: datetime | None = None,
    feedback: str = "", boundary: str = "", profile: str = "", scenario: str = "",
) -> MemoryAnswer:
    """R5 作答:材料单元(普通格或织写 memcell′,同构)的 episode 为唯一主料,按规则出答复。

    now_dt 进 prompt 作【当前时间】锚("多久了"类问题换算相对量的唯一基准)。
    feedback:核判判 defect 后的修正指令(重答轮);boundary:枚举扩面的边界提示(5b)。
    材料为空不调 LLM(空作答即"无相关信息");解析失败原样透出;
    基建异常(限流/超时)空答交上层无答案交代——异常文本不当答案。
    """
    if not hits:
        logger.info(f"R5 作答 空材料,直接空答 q={query!r}")
        return MemoryAnswer(answer="")
    order = os.environ.get(_R5_ORDER_ENV, "").strip() or "relevance"
    hits = _order_hits_for_answer(hits, order)
    handles = [f"m{i + 1}" for i in range(len(hits))]   # 快链窗口自用编号 mN(深轨目录用 cN,不共享)
    block = "\n\n".join(cell_block(h, hd) for h, hd in zip(hits, handles))
    bnd = f"\n\nBOUNDARY\n{boundary}" if boundary else ""
    fb = (f"\n\nJUDGE FEEDBACK\nA reviewer checked your previous draft against the SAME materials "
          f"and found: {feedback}\nRe-answer the question, fixing exactly what it points out."
          if feedback else "")
    subj = subject or "(not determined)"
    tnow = f"\n\nCurrent time: {(now_dt or now()).isoformat()}" if now_dt else ""
    prof = f"\n\n{profile}" if profile else ""       # 仅影响作答组织(详略/语言),不改事实选择
    user = (f"MEMORY MATERIALS\n{block}{bnd}{tnow}{fb}{prof}"
            f"\n\nQUESTION SUBJECT\n{subj}\n\nUSER QUESTION\n{query}")
    sys = with_scenario(_ANSWER_SYS, "# How to read the materials", scenario, _SCEN_DIR_ANSWER)
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    # mN → 该单元覆盖的 cell id(织写单元展开成员格全集,普通单元=自身格);跨单元去重保序
    h2ids = {hd: (h.covers or [h.cell.id]) for hd, h in zip(handles, hits)}
    try:
        obj, raw = chat_json(llm, messages, max_tokens=2000, num_tries=3, stage="answer")
        answer = str(obj.get("answer") or "").strip()
        cited: list[str] = []
        for x in (obj.get("cells") or []):
            for cid in h2ids.get(str(x).strip(), []):
                if cid not in cited:
                    cited.append(cid)
        res = MemoryAnswer(answer=answer,
                           cited_cells=cited,   # 未知编号丢弃,不崩
                           system=sys, user=user, raw=raw)
    except Exception as e:
        # 只有 JSON 解析失败才有 .raw(模型其实答了,只是格式坏)——原样透出;
        # 限流/超时等基建异常没有可用原文,异常文本绝不当答案(H1),空答走上层无答案交代
        raw = getattr(e, "raw", "")
        if raw:
            logger.warning(f"R5 作答解析失败,原样透出: {e}")
            res = MemoryAnswer(answer=raw, system=sys, user=user)
        else:
            logger.warning(f"R5 作答调用失败(基建/网络),空答交上层兜底: {type(e).__name__}: {e}")
            res = MemoryAnswer(answer="", system=sys, user=user)
    logger.info(f"R5 作答 cells={len(hits)} cited={len(res.cited_cells)} "
                f"ans={len(res.answer)}字 q={query!r}")
    return res


# 作答材料/对外视图共用的认识状态中文名(object_type 三分类,影响作答措辞)。
_TYPE_CN = {"event": "经历", "fact": "事实", "claim": "说法"}
