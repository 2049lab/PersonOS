"""深轨(融合架构 §4):单 agent 六工具翻阅记忆,langchain JSON 协议 harness。

与快链分工:快链一次检索一次作答,快而浅;深轨是"看得见记忆长什么样"的 agent——
拿交接包开局(任务/快链已得/判级缺口/目录/当前时间),用工具自主翻格、翻原话、(克制地)写回,
自己给出终答。入口:run_recall mode=deep 直达;mode=auto 且 R3 判 partial/empty 时升级。

口径贯穿:atoms 只是检索面——agent 一切可见材料不含 atom 文本,search_atoms 用 atoms 定位
但返回的是整格材料(topic+episode);open_cell 只带"已抽 N 条索引"的数量。

Trace:每步工具调用记 INFO + steps 透出(评测/工作台透视"深轨走到了哪")。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger
from pydantic import (
    AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator,
)

from .. import obs
from ..config import settings
from ..models import (
    MemCell, MemoryAtom, ensure_aware, stamped_atom_text,
)
from ..storage.atom_store import AtomStore
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.evidence_store import EvidenceStore
from .arbitrate import ReviewResult
from .llm import ChatLLM, strip_fences, with_scenario
from .chain_face import assemble_units
from .rerank import NoopReranker, Reranker, rerank_cells
from .retrieval import (
    CellHit, MemoryAnswer, QueryRewrite, _date_window, _in_window, _vec_ranking,
    cell_block, cell_lead, search_atoms as _pool_search,
)
from .write_path import _match_evidence_refs

# langchain 1.x:classic agents(JSON 协议,不依赖网关 tool-call)在 langchain-classic
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad import format_log_to_messages
from langchain_classic.agents.output_parsers import JSONAgentOutputParser
from langchain_core.agents import AgentAction
from langchain_core.language_models.chat_models import (
    BaseChatModel, ChatGeneration, ChatResult,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import StructuredTool
from langchain_core.tools.render import render_text_description

_MAX_STEPS = 9         # agent 工具调用上限(每步可批量并行调用;每步观察附剩余预算提示)
_REMEMBER_CAP = 8      # 单会话 remember 写回上限(护栏;超出拒写)
_CATALOG_SIZE = 10     # 开局目录:时间倒序近 N 格
_PAGE_CELLS = 10       # find_cells 每页格数
_PAGE_LINES = 30       # get_cell_evidence 每页句子数
_DEFAULT_LIMIT = 5     # search_atoms 默认返回格数
_MAX_LIMIT = 8         # 上限(agent 可调,非固定)


def _coerce_int(v, default: int) -> int:
    """schema 入参容错:模型偶尔把 page/limit 传成字符串("abc"/None),静默回落默认值。

    放在 pydantic 入参层而不是靠 handle_tool_error——那层只接工具函数内异常,
    入参校验错误会直接穿出 AgentExecutor 打断整轮对话。
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# —— 短编号注册表(R5 h2id 模式在深轨的复用)——

class HandleRegistry:
    """cell 真 id ↔ 短编号(c1..cN)的会话内注册表。

    编号按【首次进入上下文】的顺序分配(开局目录 c1=最新),此后稳定:agent 通篇引用
    c3 就是同一格,终答 cited 回译成真 id 交给上层。
    """

    def __init__(self):
        self._h2id: dict[str, str] = {}
        self._id2h: dict[str, str] = {}

    def ensure(self, cell_id: str) -> str:
        h = self._id2h.get(cell_id)
        if h is None:
            h = f"c{len(self._id2h) + 1}"
            self._id2h[cell_id] = h
            self._h2id[h] = cell_id
        return h

    def real(self, handle: str) -> str | None:
        return self._h2id.get((handle or "").strip())

    def __len__(self) -> int:
        return len(self._id2h)


# —— 渲染器(工具返回的文本形态)——

def _fmt_ts(dt: datetime | None) -> str:
    dt = ensure_aware(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "(time unknown)"


def cell_full(cell: MemCell, handle: str, n_atoms: int, n_lines: int) -> str:
    """一格完整材料:分隔行 + 元信息头(与快链 cell_lead 同口径)+ episode 主料 + 索引/原话提示。

    atom 文本刻意不出现:它们只是检索面,给 agent 看会被当成"参考答案"混淆视听。
    """
    lines = [
        f"━━━ {handle} ━━━",
        cell_lead(cell),
        cell.episode or "(no episode)",
        f"(this segment has {n_atoms} extracted index atoms and {n_lines} raw utterances; "
        f"if a fact looks doubtful, verify with get_cell_evidence against the raw transcript)",
    ]
    return "\n".join(lines)


def cell_row(handle: str, cell: MemCell) -> str:
    """目录轻量行:编号 | 起始时间(yyyy-MM-dd HH:mm:ss) | topic。"""
    return f"{handle} | {_fmt_ts(cell.t_start)} | {cell.topic or '(no topic)'}"


def _look_image_note(d: "DeepDeps", rec) -> str:
    """带用户原始问题(d.task_query)重看这条图片证据的原图,返回可拼进原话行的补充事实。

    写入时 content_inline 已含「带对话上下文」的泛化理解;这里是「带精确问题」的定向追问,
    补 ingest 那次可能漏掉的细节(堵 caption 信息损失的坑)。任何缺失/失败都返回空串(降级)。
    """
    if rec.modality not in ("image", "mixed") or not rec.content_ref:
        return ""
    if d.media_store is None or d.mllm is None or not getattr(d.mllm, "available", False):
        return ""
    purpose = d.task_query or ""
    if not purpose.strip():
        return ""
    # 内容类型从 OSS key 后缀推断(EvidenceRecord 不单独存 content_type)
    ext = rec.content_ref.rsplit(".", 1)[-1].lower() if "." in rec.content_ref else "jpg"
    ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "webp": "image/webp", "gif": "image/gif", "heic": "image/heic"}.get(ext, "image/jpeg")
    try:
        img = d.media_store.read_bytes(rec.content_ref)
        text = d.mllm.look_image(img, purpose, content_type=ctype)
    except Exception:   # noqa: BLE001  看图失败不影响原话展示
        return ""
    return f"  ↳[看图·针对「{purpose[:30]}」] {text}" if text else ""


def _render_evidence_line(d: "DeepDeps", r) -> str:
    """一条证据渲染成一行;图片证据额外带一次针对性看图补充。"""
    base = f"[{_fmt_ts(r.captured_at)}] {r.holder}: {r.content_inline or ''}"
    note = _look_image_note(d, r)
    return base + ("\n" + note if note else "")


def evidence_page(records: list, page: int, per_page: int = _PAGE_LINES,
                  d: "DeepDeps | None" = None) -> tuple[str, int]:
    """原话分页渲染:逐行 [时刻] 说话人: 原话。说话人照证据 holder(第三人称是真名)。

    d 非空且某行是图片证据时,额外带 d.task_query 重看原图,补一行针对性事实。
    返回 (渲染文本, 总页数);空记录给一句可行动说明。
    """
    if not records:
        return ("(no raw utterances kept for this segment: evidence may not have been captured, or was "
                "purged. Try another cell.)", 1)
    total = (len(records) + per_page - 1) // per_page
    page = max(1, min(page, total))
    chunk = records[(page - 1) * per_page: page * per_page]
    if d is not None:
        lines = [_render_evidence_line(d, r) for r in chunk]
    else:
        lines = [f"[{_fmt_ts(r.captured_at)}] {r.holder}: {r.content_inline or ''}" for r in chunk]
    return "\n".join(lines), total


# —— 工具实现(纯函数,只碰 DeepDeps;不碰 langchain,便于直测)——

@dataclass
class DeepDeps:
    """深轨工具的依赖束 + 会话态(短编号注册表、写回计数)。"""
    embedder: Any
    reranker: Reranker
    atoms: AtomStore
    cells: CellStore
    evidence: EvidenceStore
    llm: Any = None              # 织写器(search_atoms 单元组装用;None=链全降级普通格)
    reg: HandleRegistry = field(default_factory=HandleRegistry)
    deep_write: bool = True
    remembered: int = 0
    calls_made: int = 0          # 已用工具步数(批量算 1 步;观察尾部预算提示的数据源)
    # 图片看图:task_query=用户原始问题(看图目的);media_store/mllm 未注入=不看图(退化为读 content_inline)
    task_query: str = ""
    media_store: Any = None
    mllm: Any = None


_EMPTY_HINT = ("No hits. Try: a more specific search term (person name / matter / date verbatim), "
               "relaxing the date-window/domain/holder filters, or find_cells to browse by time. "
               "When the retrieval chain keeps coming up empty, fall back to search_evidence to "
               "keyword-search the raw transcript directly.")


def tool_search_atoms(d: DeepDeps, *, query: str, start_date: str = "", end_date: str = "",
                      domains: list[str] | None = None, holder: str = "", limit: int = _DEFAULT_LIMIT) -> str:
    """按事实检索 = 可定制的快链:结构化过滤 → R1 atom 池 → 单元组装(织写/普通)→ R2 精排。

    与 fast-recall 同机制同代码(池/组装/精排全复用),agent 只多三样东西:检索词自己写、
    过滤条件(日期窗/域/holder)、返回单元数 limit。池 = 2×limit(agent 步窗口比一锤定音
    的快链小)。织写单元(memcell′)按覆盖格注册 handle,块头多 cN 并列——链 id 不进
    注册表,open_cell/get_cell_evidence/remember 只认真实 cell,长短 id 永不撞。
    """
    limit = max(1, min(_MAX_LIMIT, int(limit or _DEFAULT_LIMIT)))
    rw = QueryRewrite(original=query, resolved=query, domains=domains or [])
    pool = _pool_search(d.embedder, d.atoms, rewrite=rw, top_n=2 * limit,
                        start_date=start_date, end_date=end_date, holder=holder,
                        domains_filter=domains or None)   # agent 设的域条件是硬过滤
    if not pool.atoms:
        return _EMPTY_HINT
    asm = assemble_units(pool.atoms, ChainStore(d.atoms.db, d.atoms.user_id), d.cells,
                         d.llm, query=query, beyond=pool.beyond)
    units = rerank_cells(d.reranker, query, asm.units)[:limit]
    if not units:
        return _EMPTY_HINT
    n_woven = sum(1 for u in units if u.covers)
    woven_note = (f"; {n_woven} woven from fact-chains (a woven unit's header lists its "
                  f"member cell handles — open any of them to drill into the raw segments)"
                  if n_woven else "")
    out = (f"search_atoms hit {len(units)} unit(s) (atom pool → chain assembly → reranker)"
           f"{woven_note}:\n\n" + "\n\n".join(_unit_block(d, u) for u in units))
    if asm.boundary:   # 与快链同口径的残缺提示(已滤掉被织写/同格覆盖的假缺料)
        out += f"\n\nBOUNDARY\n{asm.boundary}"
    return out


def _unit_block(d: DeepDeps, u: CellHit) -> str:
    """一个材料单元的深轨渲染:handle = 覆盖格逐个注册(织写单元多 cN 并列)。

    索引/原话规模按覆盖格汇总——memcell′ 是临时视图,自身的链 id 无任何存储对应,
    下钻走成员格 handle。
    """
    ids = _covered_ids(u)
    handle = ", ".join(d.reg.ensure(cid) for cid in ids)
    n_atoms = sum(len(d.atoms.list_by_cell(cid)) for cid in ids)
    n_lines = 0
    for cid in ids:
        c = d.cells.get(cid)
        if c is not None:
            n_lines += len(c.evidence_refs)
    return cell_full(u.cell, handle, n_atoms, n_lines)


def tool_find_cells(d: DeepDeps, *, query: str = "", start_date: str = "", end_date: str = "",
                    domains: list[str] | None = None, page: int = 1) -> str:
    """按条件列格:时间窗(按 cell 起始时间)+ 域过滤;有 query 时按 topic 相似度排,否则时间倒序;分页。"""
    lo, hi = _date_window(start_date, end_date)
    domains = set(domains or [])
    selected = []
    for c in d.cells.iter_all():                   # old → new
        if (lo or hi) and not _in_window(ensure_aware(c.t_start), lo, hi):
            continue                               # 时间筛选下无时间锚的格不可见
        if domains and not (set(c.domains) & domains):
            continue
        selected.append(c)
    if not selected:
        return _EMPTY_HINT
    order_note = "reverse chronological"
    if query:
        sel_ids = {c.id for c in selected}
        entries = [(c, v) for c, v in d.cells.all_with_embeddings() if c.id in sel_ids]
        ranked_ids = [cid for cid in _vec_ranking([d.embedder.embed([query])[0]], entries)
                      if cid in sel_ids]
        selected.sort(key=lambda c: ranked_ids.index(c.id)
                      if c.id in ranked_ids else len(ranked_ids))   # 无 topic 向量的格排后面
        order_note = "topic similarity descending (cells without topic vectors last, by time)"
    else:
        selected.reverse()
    total = len(selected)
    pages = (total + _PAGE_CELLS - 1) // _PAGE_CELLS
    page = max(1, min(int(page or 1), pages))
    chunk = selected[(page - 1) * _PAGE_CELLS: page * _PAGE_CELLS]
    rows = "\n".join(cell_row(d.reg.ensure(c.id), c) for c in chunk)
    rest = total - page * _PAGE_CELLS
    tail = (f"\n({rest} more cell(s) after this page; continue with page={page + 1})" if rest > 0
            else "\n(last page)")
    return f"find_cells page {page}/{pages} ({total} cell(s) after filtering, {order_note}):\n{rows}{tail}"


def _resolve_cell(d: DeepDeps, handle: str) -> MemCell | None:
    cid = d.reg.real(handle)
    return d.cells.get(cid) if cid else None


def tool_open_cell(d: DeepDeps, *, c: str) -> str:
    """展开一格:叙事(episode)全文 + 索引/原话规模。"""
    if not d.reg.real(c):
        return (f"Unknown handle {c!r}: only handles that appeared in find_cells or search_atoms "
                f"results can be opened.")
    cell = _resolve_cell(d, c)
    if cell is None:
        return "This cell no longer exists (it may have been purged)."
    return cell_full(cell, d.reg.ensure(cell.id),
                     len(d.atoms.list_by_cell(cell.id)), len(cell.evidence_refs))


def tool_get_cell_evidence(d: DeepDeps, *, c: str, page: int = 1) -> str:
    """翻某格的对话原话(逐句带说话人与时刻),每页 30 句,page 翻页。"""
    if not d.reg.real(c):
        return f"Unknown handle {c!r}: locate the cell first via find_cells / search_atoms."
    cell = _resolve_cell(d, c)
    if cell is None:
        return "This cell no longer exists (it may have been purged)."
    recs = [r for r in (d.evidence.get(ref.evidence_id) for ref in cell.evidence_refs) if r]
    recs.sort(key=lambda r: ensure_aware(r.captured_at)
              or datetime.min.replace(tzinfo=timezone.utc))
    body, total = evidence_page(recs, int(page or 1), d=d)
    head = (f"Raw transcript of cell {d.reg.ensure(cell.id)}: {len(recs)} utterance(s), "
            f"page {page}/{total}:\n")
    return head + body


def tool_search_evidence(d: DeepDeps, *, keywords: list[str], holder: str = "",
                         limit: int = 15) -> str:
    """关键词直搜原话底稿(全库,不经索引):所有关键词须同一句命中,返回逐句原话及所属格。

    兜底路径:search_atoms/find_cells 建立在 atoms/cells 索引上,抽取漏项时原话在
    真相层却无路可达——这把路补上(原话是完整真相层,LIKE 扫描不吃任何抽取损失)。
    """
    kws = [str(k).strip() for k in (keywords or []) if str(k).strip()][:4]
    if not kws:
        return ("keywords is empty: give 1-4 verbatim words (person name / matter / brand / place); "
                "all keywords must hit the SAME utterance.")
    limit = max(1, min(30, int(limit or 15)))
    recs = d.evidence.search_keyword(kws, holder=holder or "", limit=limit)
    if not recs:
        return (f"No hits in the raw transcript either ({' / '.join(kws)}). The wording may differ: "
                "try words more likely to appear verbatim, or relax/drop the holder filter.")
    ev2cell = {}
    for c in d.cells.iter_all():
        for ref in c.evidence_refs:
            ev2cell[ref.evidence_id] = c
    lines = []
    for r in recs:
        cell = ev2cell.get(r.id)
        h = f"(cell {d.reg.ensure(cell.id)})" if cell else "(no cell)"
        # 命中图片证据时带 task_query 追加一次针对性看图(_render_evidence_line 内部判 modality)
        lines.append(_render_evidence_line(d, r) + f" {h}")
    return (f"search_evidence hit {len(recs)} utterance(s) (keywords AND in the same utterance: "
            f"{' / '.join(kws)}):\n" + "\n".join(lines)
            + "\n(use open_cell / get_cell_evidence on the owning cell for context)")


def tool_remember(d: DeepDeps, *, c: str, text: str = "", quote: str = "", holder: str = "user",
                  kind: str = "", domains: list[str] | None = None, episode_append: str = "") -> str:
    """写回(克制):给某格新增一条检索索引(quote 逐字回链原话)和/或在叙事末尾追加。

    只新增不改写;全会话上限 _REMEMBER_CAP 条;deep_write=False 时整体只读。
    """
    if not d.deep_write:
        return "Write-back is disabled (deep_write=False); this run is read-only."
    if not (text or "").strip() and not (episode_append or "").strip():
        return "Give at least one of text / episode_append."
    if d.remembered >= _REMEMBER_CAP:
        return f"Write-back cap for this session reached ({_REMEMBER_CAP}); continue searching/answering."
    cell = _resolve_cell(d, c)
    if cell is None:
        return f"Cannot write back: handle {c!r} never appeared in this session's cells."

    wrote = []
    if text.strip():
        records = [r for r in (d.evidence.get(ref.evidence_id) for ref in cell.evidence_refs) if r]
        refs = _match_evidence_refs(records, quote)   # W2② 同款:逐字子串回链
        atom = MemoryAtom(memcell_id=cell.id, text=text.strip(),
                          holder=(holder or "user").strip() or "user",
                          domains=list(domains or []), kind=kind or None,
                          occurrence_time=cell.t_start, evidence_refs=refs, source="deep")
        d.atoms.upsert(atom, embedding=d.embedder.embed([stamped_atom_text(atom)])[0])
        d.remembered += 1
        wrote.append(f"added 1 index atom (linked to {len(refs)} source utterance(s))")
    if episode_append.strip():
        new_episode = ((cell.episode or "").rstrip() + "\n" + episode_append.strip()).strip()
        d.cells.upsert(cell.model_copy(update={"episode": new_episode}))   # embedding=None 保留 topic 向量
        wrote.append("appended 1 passage to the episode")
    logger.info(f"deep remember cell={cell.id} wrote={wrote}")
    return ("Written back: " + "; ".join(wrote) + ". The new index is retrievable on the next search.")


# —— langchain 接线 ——

class MaasChatModel(BaseChatModel):
    """ChatLLM 协议(MaasClient / FakeLLM)的 langchain 适配器。

    基础设施仍走 .env + clients/maas.py(超时分档/重试都在那层);这里只做
    消息映射(role)与 stop 的客户端截断——网关不传 stop,而 JSON agent 靠它防幻觉续写。
    """
    client: Any = None
    temperature: float = 0.2
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "personos-chatllm"

    @property
    def identifying_params(self) -> dict:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        role = {"system": "system", "human": "user", "ai": "assistant"}
        msgs = [{"role": role.get(m.type, "user"), "content": m.content} for m in messages]
        text = self.client.chat(msgs, temperature=self.temperature, max_tokens=self.max_tokens)
        for s in (stop or []):
            i = text.find(s)
            if i >= 0:
                text = text[:i]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# schema 硬化(2026-09-09,实测 19% 工具调用带 schema 外参数):高频错名收为 AliasChoices 别名,
# time_range dict 由 before-validator 消费映射;extra="forbid" 让剩余怪参数响亮报错自纠,
# 不再被静默丢弃(agent 以为过滤了日期其实没有 → 计数题答错的直接机制)。
def _absorb_time_range(data, *, has_date_fields: bool = True):
    """入参 dict 里的 time_range={start,end} → start_date/end_date(extra=forbid 前必须消费掉)。

    has_date_fields=False 的 schema(search_evidence 无日期过滤能力):仅 pop 掉防 forbid 报错——
    策略段已声明该工具无日期窗,命中行的时间戳由 agent 自查。
    """
    if isinstance(data, dict):
        tr = data.pop("time_range", None)
        if isinstance(tr, dict) and has_date_fields:
            lo = tr.get("start") or tr.get("start_date") or ""
            hi = tr.get("end") or tr.get("end_date") or ""
            data.setdefault("start_date", lo)
            data.setdefault("end_date", hi)
    return data


def _wrap_batch(data, **required_placeholder):
    """action_input 传裸 list = 一步内多次调用同一工具:包进 batch 字段交给执行层。

    agent 的连发式调用(实测:连发 5 个单关键词 search_evidence)每次烧一步预算;
    批量形态一步打完,省步数。执行层串行跑(存储层共享 pymysql 连接非线程安全),
    省的是步数预算而非墙钟。批量形态下 schema 的必填主字段(query/c/keywords)用占位
    值放行——真正的逐项校验在执行层对 batch 里每个参数对象做。
    """
    if isinstance(data, list):
        data = {"batch": data}
    if isinstance(data, dict) and data.get("batch") is not None:
        for k, v in required_placeholder.items():
            data.setdefault(k, v)
    return data


# 批量入参字段(六 schema 共用):agent 既可 action_input 传裸 list,也可显式 {"batch": [...]}
_BATCH_FIELD = Field(default=None, description="Run several calls of this tool in ONE step: "
                                               "a list of argument objects, each as specified "
                                               "above (e.g. three keyword sweeps at once)")


class _SearchAtomsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(validation_alias=AliasChoices("query", "q", "kw", "keyword", "keywords", "term"),
                       description="Search terms: person name / matter verbatim, one short phrase "
                                   "(drawn from materials already read)")
    start_date: str = Field(default="", validation_alias=AliasChoices("start_date", "start", "date_from", "time_start"),
                            description="Start date yyyy-MM-dd (endpoint day inclusive); "
                                        "empty = unbounded")
    end_date: str = Field(default="", validation_alias=AliasChoices("end_date", "end", "date_to", "time_end"),
                          description="End date yyyy-MM-dd (endpoint day inclusive); "
                                      "empty = unbounded")
    domains: list[str] = Field(default_factory=list,
                               validation_alias=AliasChoices("domains", "domain"),
                               description="D-axis domain codes to filter "
                                           "(e.g. ['D13']); empty = unbounded")
    holder: str = Field(default="", validation_alias=AliasChoices("holder", "speaker"),
                        description="Restrict to speaker/owner ('user' or a name); "
                                    "empty = unbounded")
    limit: int = Field(default=_DEFAULT_LIMIT,
                       validation_alias=AliasChoices("limit", "top_k", "max_results", "n"),
                       description="Material units returned, 1-8, default 5 (a unit is a plain "
                                   "segment cell or a woven chain narrative)")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _tr(cls, data):
        return _absorb_time_range(_wrap_batch(data, query=""))

    @field_validator("limit", mode="before")
    @classmethod
    def _int_limit(cls, v):
        return _coerce_int(v, _DEFAULT_LIMIT)


class _FindCellsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="", validation_alias=AliasChoices("query", "q", "topic", "keyword"),
                       description="Search term for topic-similarity ordering; "
                                   "empty = plain reverse-chronological")
    start_date: str = Field(default="", validation_alias=AliasChoices("start_date", "start", "date_from", "time_start"),
                            description="Start date yyyy-MM-dd, empty = unbounded")
    end_date: str = Field(default="", validation_alias=AliasChoices("end_date", "end", "date_to", "time_end"),
                          description="End date yyyy-MM-dd, empty = unbounded")
    domains: list[str] = Field(default_factory=list,
                               validation_alias=AliasChoices("domains", "domain"),
                               description="D-axis domain codes to filter, "
                                           "empty = unbounded")
    page: int = Field(default=1, description="Page number, 10 cells per page")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _tr(cls, data):
        return _absorb_time_range(_wrap_batch(data))

    @field_validator("page", mode="before")
    @classmethod
    def _int_page(cls, v):
        return _coerce_int(v, 1)


class _OpenCellArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    c: str = Field(validation_alias=AliasChoices("c", "cell", "cell_id", "handle", "cell_handle"),
                   description="Cell handle, e.g. 'c3' (one that appeared in the catalog or "
                               "search results)")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _bw(cls, data):
        return _wrap_batch(data, c="")


class _GetCellEvidenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    c: str = Field(validation_alias=AliasChoices("c", "cell", "cell_id", "handle", "cell_handle"),
                   description="Cell handle, e.g. 'c3'")
    page: int = Field(default=1, description="Page number, 30 utterances per page")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _bw(cls, data):
        return _wrap_batch(data, c="")

    @field_validator("page", mode="before")
    @classmethod
    def _int_page(cls, v):
        return _coerce_int(v, 1)


class _SearchEvidenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(validation_alias=AliasChoices("keywords", "query", "q", "kw", "keyword",
                                                              "words", "terms"),
                                description="Keyword list (1-4 verbatim words: person name / matter / "
                                            "brand / place); all keywords must hit the SAME utterance")
    holder: str = Field(default="", validation_alias=AliasChoices("holder", "speaker"),
                        description="Restrict to speaker ('user' or a name); "
                                    "empty = unbounded")
    limit: int = Field(default=15, validation_alias=AliasChoices("limit", "top_k", "max_results", "n"),
                       description="Max utterances returned, 1-30, default 15")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _tr(cls, data):
        data = _absorb_time_range(_wrap_batch(data, keywords=[]), has_date_fields=False)
        # 实测模型常把 keywords 传成裸字符串("beach")——包成单元素列表,不打断自纠;
        # 注意别名键(query/kw/keyword/…)在 before 阶段还是原样,逐个键检查
        if isinstance(data, dict):
            for k in ("keywords", "query", "q", "kw", "keyword", "words", "terms"):
                if isinstance(data.get(k), str):
                    data[k] = [data[k]]
        return data

    @field_validator("limit", mode="before")
    @classmethod
    def _int_limit(cls, v):
        return _coerce_int(v, 15)


class _RememberArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    c: str = Field(validation_alias=AliasChoices("c", "cell", "cell_id", "handle", "cell_handle"),
                   description="Target cell handle to write into, e.g. 'c3'")
    text: str = Field(default="", description="New retrieval index atom (one self-contained atomic "
                                              "fact, same spec as the extractor: third person, dual "
                                              "time format)")
    quote: str = Field(default="", description="Verbatim quote from that cell's transcript (links the "
                                               "provenance; a non-verbatim quote leaves provenance "
                                               "empty)")
    holder: str = Field(default="user", description="Who said it / whose attribute ('user' or a name)")
    kind: str = Field(default="", description="K-axis type code (K01..K14); empty = unset")
    domains: list[str] = Field(default_factory=list, description="D-axis domain codes (0-3)")
    episode_append: str = Field(default="", description="Supplement appended at the end of the cell's "
                                                        "narrative (never rewrites the original)")
    batch: list[dict] | None = _BATCH_FIELD

    @model_validator(mode="before")
    @classmethod
    def _bw(cls, data):
        return _wrap_batch(data, c="")


class _BatchTolerantJSONParser(JSONAgentOutputParser):
    """action_input 传 list(一步批量调用)→ 构造 AgentAction 前包成 {"batch": [...]}。

    langchain 的 AgentAction.tool_input 只收 str/dict,裸 list 会在基类 parse 内部
    实例化时抛 ValidationError——必须复写整个 parse 在构造前转接(实测 agent 按批量
    协议连发时整轮卡死在解析重试,9/9 步全是 _Exception)。批量语义本身由工具的
    batch 字段执行,这里只做形态转接。
    """

    def parse(self, text: str):
        from langchain_core.agents import AgentFinish
        from langchain_core.exceptions import OutputParserException
        from langchain_core.utils.json import parse_json_markdown
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["action_input"]}, text)
            action_input = response.get("action_input") or {}
            if isinstance(action_input, list):
                action_input = {"batch": action_input}
            return AgentAction(response["action"], action_input, text)
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e


def _build_agent(model: BaseChatModel, tools: list[StructuredTool],
                 scenario: str = "") -> RunnablePassthrough:
    """复刻 create_json_chat_agent 的默认装配,仅把输出解析器换成批量容忍版。

    scenario 非空 → 系统提示词插入业务方场景段(见 _agent_prompt);空 → 逐字节不变。
    """
    prompt = _agent_prompt(scenario).partial(tools=render_text_description(list(tools)),
                                             tool_names=", ".join(t.name for t in tools))
    return (RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_messages(x["intermediate_steps"]))
            | prompt | model.bind(stop=["\nObservation"]) | _BatchTolerantJSONParser())


def _tool_error_as_observation(err: Exception) -> str:
    """工具故障 → 观察文本。框架的 handle_tool_error 只接 ToolException,
    通用异常会穿出打断整轮 agent——所以这层兜底放我们自己的闭包里。
    """
    return (f"Tool call failed: {err}. Check the arguments (field names and types per the tool spec), "
            f"or try another tool / relax the filters.")


def _args_validation_hint(err) -> str:
    """入参校验失败(字段名写错/类型离谱)→ 可行动的观察提示,agent 自纠。"""
    return (f"Tool argument validation failed: {err}. Retry with field names and types exactly as "
            f"specified (the cell-handle field is named c, e.g. 'c3').")


def build_tools(d: DeepDeps) -> list[StructuredTool]:
    """六个工具 → langchain StructuredTool(闭包捕获 DeepDeps,agent 会话态随之走)。"""
    def _mk(name: str, desc: str, schema: type[BaseModel], fn) -> StructuredTool:
        def _invoke(kw: dict) -> str:                        # 绑定会话态 DeepDeps
            try:
                return fn(d, **kw)
            except Exception as e:                           # noqa: BLE001 任何故障降级为观察
                logger.warning(f"deep 工具 {name} 故障: {e!r} args={kw}")
                return _tool_error_as_observation(e)

        def _bound(**kw):
            batch = kw.pop("batch", None)
            if batch:                                        # 一步批量:逐项校验后串行执行
                try:                                         # (共享 pymysql 连接非线程安全,不并行)
                    calls = [{k: v for k, v in schema.model_validate(item)
                              .model_dump(exclude_none=True).items() if k != "batch"}
                             for item in batch]
                except ValidationError as e:
                    return _args_validation_hint(e)
                out = "\n\n".join(f"── batch {i}/{len(calls)} ──\n{_invoke(c)}"
                                  for i, c in enumerate(calls, 1))
            else:
                out = _invoke(kw)
            d.calls_made += 1                                # 批量只算一步
            left = _MAX_STEPS - d.calls_made
            tail = f"≈{left} tool steps left"
            if left <= 2:                                    # 预算将尽:逼 agent 收手作答
                tail += (" — budget nearly exhausted: give your Final Answer now with the best "
                         "material in hand (state honestly what is missing) instead of opening "
                         "new lines of search")
            return f"{out}\n\n---\n[{tail}]"

        return StructuredTool.from_function(
            func=_bound, name=name, description=desc, args_schema=schema,
            handle_validation_error=_args_validation_hint)

    return [
        _mk("search_atoms",
            "Search by fact: atom-level retrieval + fact-chain assembly + reranker, returning "
            "the material units most likely to contain the fact. A unit is either a full cell "
            "(date/topic/whole narrative) or a woven chain narrative (the same matter's dated "
            "statements merged into one episode; its separator lists the member cell handles). "
            "Use for 'was this specific thing ever discussed / where did it happen'.",
            _SearchAtomsArgs, tool_search_atoms),
        _mk("find_cells",
            "List cells by condition: time window / domain filter (optionally topic-similarity "
            "ordering), paginated lightweight catalog (handle | start date | topic). Use for 'what "
            "was discussed in that period / browsing by time / getting the lay of the land'.",
            _FindCellsArgs, tool_find_cells),
        _mk("open_cell", "Open one cell: read its full narrative (episode) plus its index/transcript "
                         "sizes.",
            _OpenCellArgs, tool_open_cell),
        _mk("get_cell_evidence",
            "Read a cell's raw transcript: utterance by utterance with speaker and timestamp, 30 "
            "lines per page. Use it to verify a doubtful detail in the narrative, or when you "
            "suspect the segment holds facts that never made it into the narrative.",
            _GetCellEvidenceArgs, tool_get_cell_evidence),
        _mk("search_evidence",
            "Keyword-search the whole raw transcript directly: all keywords must hit the same "
            "utterance; returns utterances verbatim (speaker, timestamp, owning cell handle). "
            "Bypasses every index — use as the fallback when search_atoms/find_cells keep coming "
            "up empty, or when you suspect a fact was never extracted into the narrative/index.",
            _SearchEvidenceArgs, tool_search_evidence),
        _mk("remember",
            "Write back (use sparingly): add one retrieval index atom to a cell, and/or append a "
            "supplement at the end of its narrative. Only for facts that ARE in the raw transcript "
            "but were missed by the narrative/index; quote must be a verbatim quote of one raw "
            "utterance in that cell.",
            _RememberArgs, tool_remember),
    ]


# —— 系统提示词:全貌教育 + 多跳要诀 + JSON 输出协议 ——

_AGENT_SYS = (
    "# Role\n"
    "You are the deep-retrieval agent of a personal memory system: given a question about the "
    "user's past conversations, dig through the memory store and produce the final answer "
    "yourself.\n\n"
    "# What the memory store looks like (four layers)\n"
    "Memory is organized into cells: one cell = one bounded topic conversation —\n"
    "- topic: the chapter title (a one-line theme);\n"
    "- episode: the chapter's narrative (third-person retelling; your main answering material);\n"
    "- retrieval index (atoms): single-fact cards extracted from the raw transcript, used only "
    "for search — you never see their full text;\n"
    "- evidence: the utterance-by-utterance transcript of that segment (most trustworthy; read "
    "it with get_cell_evidence).\n"
    "Cells chain in chronological order; every cell's material opens with the dialogue date.\n\n"
    "# Working method (multi-hop / reasoning)\n"
    "1. Each step, first ask 'what is still missing': is the subject right? the time window? is "
    "the evolution chain complete? does every instance of a counting question hold in hand?\n"
    "2. Draw the next hop's search terms from materials ALREADY read (person name / matter / "
    "date verbatim); never repeat the same query.\n"
    "3. When the retrieval chain (search_atoms/find_cells) comes up empty two or three times in "
    "a row → fall back: search_evidence keyword-searches the raw transcript directly — the raw "
    "transcript is the complete ground-truth layer; facts the index missed can only be found "
    "there.\n"
    "4. Multiple cells stating different things about the same fact: order them by cell time "
    "(the later one is the new state) and present the evolution; an explicit correction always "
    "wins.\n"
    "5. When the narrative and the raw transcript disagree, the raw transcript wins.\n"
    "6. Two consecutive rounds with no new lead → stop: answer from the materials in hand and "
    "state the uncertain parts in the answer.\n"
    "7. Every tool result ends with your remaining step budget — when it runs low, stop opening "
    "new lines of search and answer from the materials in hand (state what is missing).\n\n"
    "# Tool strategy\n"
    "- Retrieval priority: search_atoms (the extracted fact index, most precise) first; "
    "find_cells (cell catalog) when you don't know where the fact lives; search_evidence "
    "(raw-transcript LIKE keyword search, lowest precision) only as the fallback when both "
    "keep coming up empty.\n"
    "- search_atoms: first stop for a specific fact — phrase the query with verbatim person "
    "names / matters you have already read. Results are material units: a plain segment cell, "
    "or a woven chain narrative (several dated statements about the same matter merged into "
    "one episode) whose separator line lists several cN handles — its member cells, open any "
    "of them to drill into the raw segment. A trailing BOUNDARY note means more matching "
    "atoms exist beyond the shown units — it names what is NOT covered.\n"
    "- find_cells: when you don't know where the fact lives — browse by time window and/or "
    "topic to locate the right period, then open the candidates.\n"
    "- search_evidence: fallback and verbatim sweep. AND semantics: ALL keywords must appear in "
    "the SAME utterance — every extra keyword shrinks the hit set. Use 1-2 highly distinctive "
    "words (a proper name / brand / place); never 3-4 generic words. For a time-bounded sweep, "
    "scan the hits' timestamps yourself (there is no date filter here).\n"
    "- open_cell / get_cell_evidence: once you hold a handle; when a detail is doubtful, the raw "
    "transcript wins.\n"
    "- COUNT/LIST questions: before the Final Answer you MUST do one closing sweep — "
    "search_evidence with the single most distinctive word, or find_cells over the relevant "
    "period — to confirm no instance was missed; the same event mentioned several times counts "
    "ONCE.\n\n"
    "# Available tools\n"
    "{tools}\n\n"
    "# Output protocol (each step outputs exactly one json code block and nothing else)\n"
    "```json\n"
    '{{"thought": "judgment for this step: what is in hand, what is missing, why this move",\n'
    ' "action": "tool name or Final Answer",\n'
    ' "action_input": {{...}}}}\n'
    "```\n"
    "- Calling a tool: action = tool name, action_input = that tool's argument object (field "
    "names exactly as in the tool specs above).\n"
    "- Running several INDEPENDENT lookups in one step: action_input may be a list of argument "
    "objects (e.g. three search_evidence sweeps with different distinctive words, or opening "
    "two cells at once) — they run together and consume only ONE step of your budget. Do not "
    "burn one step per lookup when the lookups do not depend on each other.\n"
    '- Final answer: action = "Final Answer", action_input is\n'
    '{{"answer": "a standard factual statement: third person, neutral, conclusion first, with '
    "absolute dates, no conversational tone (never write 'you told me / I remember'); state "
    "honestly what the materials lack — objectively report what was checked, what the closest "
    "material is, and why it does not suffice (what is missing); never fabricate, never give "
    'plausible-sounding but unfounded conclusions",'
    ' "cited": ["c1", "c2"]}}\n'
    "  (cited lists only the cell handles you actually relied on).\n\n"
    "# You may only use these tool names\n"
    "{tool_names}\n"
)

_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _AGENT_SYS),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad", optional=True),
])

_PARSE_NUDGE = ("Wrong output format: reply with a single json code block containing "
                "thought/action/action_input; for the final answer, action must be "
                "\"Final Answer\".")

# 业务方场景注入 directive(只调检索方向,不许编造)——见 llm.with_scenario
_SCEN_DIR_AGENT = ("Use it to steer which matters to prioritize digging into and which leads to "
                   "follow first; it never licenses fabricating facts absent from the store.")


def _agent_prompt(scenario: str) -> ChatPromptTemplate:
    """深轨 agent 系统提示词模板;scenario 空 → 复用常量(逐字节不变)。

    scenario 拼进 system 前先转义花括号——_AGENT_SYS 靠 {tools}/{tool_names} 做模板变量,
    业务方文本里的 {} 若不转义会被 ChatPromptTemplate 当变量解析而报错。
    """
    s = (scenario or "").strip()
    if not s:
        return _AGENT_PROMPT
    safe = s.replace("{", "{{").replace("}", "}}")
    sys = with_scenario(_AGENT_SYS, "# What the memory store looks like (four layers)",
                        safe, _SCEN_DIR_AGENT)
    return ChatPromptTemplate.from_messages([
        ("system", sys), ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad", optional=True)])


# —— 交接包(自然语言六件套;给证据不给作答草稿,防锚定)——

_REVIEW_TEXT = {"answer_defect": "a draft was answered from the materials above and reviewed; "
               "even the re-answer failed itemized checks",
               "insufficient_material": "the retrieved materials lack the core of the answer"}


def _covered_ids(h: CellHit) -> list[str]:
    """材料单元覆盖的 cell id:织写单元=成员格全集(covers);普通单元=自身格。"""
    return h.covers or [h.cell.id]


def _translate_handles(text: str, fast_hits: list[CellHit], d: DeepDeps) -> str:
    """把 critique 里的快链窗口编号(mN,核判侧自用)回译成深轨目录编号 cN。

    两套编号服务不同窗口:快链 m1..mK 按 R5/R3' 材料顺序,深轨 c1..cN 按目录注册序。
    核判只见过自己的材料窗口,它写的 mN 只能指快链材料——按 fast_hits 顺序回译,
    织写单元展开为其覆盖的成员格(逐个 ensure,多格逗号并列);越界/不认识的编号
    原样保留(可见的陌生符号,而不是静默指错格)。兼容模型手滑写 cN。
    """
    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return (",".join(d.reg.ensure(cid) for cid in _covered_ids(fast_hits[i - 1]))
                if 1 <= i <= len(fast_hits) else m.group(0))

    # 边界用 ASCII 字母数字界定(而非 \b):中文 critique「材料m1」无空格写法 \b 不成立(CJK 同为 \w)
    return re.sub(r"(?<![A-Za-z0-9])[mc](\d+)(?![A-Za-z0-9])", repl, text)


def build_handoff(d: DeepDeps, *, query: str, now_dt: datetime,
                  rw: QueryRewrite | None, review: ReviewResult | None,
                  fast_hits: list[CellHit] | None, total_cells: int,
                  catalog: list[MemCell], profile: str = "") -> str:
    """快链 → 深轨的交接包:①任务 ②快链已得证据 ③核判结论与缺口 ④记忆库目录 ⑤当前时间。

    系统提示词(全貌教育)常驻,不占交接包。快链的作答草稿刻意不给:深轨要独立判断,
    只继承"已检索到什么材料 + 核判指出的缺口"(critique 是方向,草稿是锚,给方向不给锚)。
    critique 里的快链编号 mN 在此回译成深轨 cN——材料区用深轨编号渲染,缺口必须对得上号。
    目录格先进注册表(c1=最新),材料区复用目录编号:深轨编号只由库内容决定,
    与快链命中过什么、什么顺序命中无关——两套 id 体系各自独立。
    """
    for c in catalog:
        d.reg.ensure(c.id)
    parts: list[str] = []
    task = [f"Answer the user's question: {query}"]
    if rw:
        if rw.resolved and rw.resolved != query:
            task.append(f"(references resolved: {rw.resolved})")
        if rw.subject:
            task.append(f"question subject: {rw.subject} (whose affairs are being asked about)")
        if rw.time_start or rw.time_end:
            task.append(f"date window: {rw.time_start or '…'} ~ {rw.time_end or '…'}")
        if rw.domains:
            task.append(f"initial domains: {','.join(rw.domains)}")
        if rw.expansions:
            task.append("starting search leads: " + "; ".join(rw.expansions[:4])
                        + " (suggested first hops from the query preprocessor — a starting "
                          "point, not proven coverage)")
    parts.append("## Task\n" + "\n".join(task))

    if fast_hits:
        top = fast_hits[:5]
        # 织写单元的编号 = 其覆盖成员格(逗号并列);普通单元单格,行为与旧版一致
        blocks = [cell_block(h, ", ".join(d.reg.ensure(i) for i in _covered_ids(h)))
                  for h in top]   # 与 R3/R5 同渲染器
        rows = [cell_row(", ".join(d.reg.ensure(i) for i in _covered_ids(h)), h.cell)
                for h in fast_hits[5:20]]
        more = "\n".join(rows)
        parts.append("## Materials already retrieved by the fast chain (starting point, not "
                     "proven complete; dig only where these do not cover)\n"
                     + "\n\n".join(blocks) + (f"\nOther relevant cells (light rows):\n{more}"
                                              if more else ""))

    if review:
        s = f"## Fast-chain review\n{_REVIEW_TEXT.get(review.verdict, review.verdict)}"
        if review.critique:
            s += ". Gap: " + _translate_handles(review.critique, fast_hits or [], d)
        parts.append(s)

    rows = [cell_row(d.reg.ensure(c.id), c) for c in catalog]
    rest = total_cells - len(catalog)
    parts.append(f"## Memory catalog (reverse chronological, latest {len(catalog)} cells; "
                 f"{max(rest, 0)} older cells exist, browse with find_cells)\n" + "\n".join(rows))

    if profile:
        parts.append("## User profile (context for search direction — goals/relationships/habits; "
                     "NOT a source of answer facts)\n" + profile)

    parts.append(f"## Current time\n{now_dt:%Y-%m-%d %H:%M:%S} (timezone {now_dt.tzinfo})")
    return "\n\n".join(parts)


# —— 编排 ——

@dataclass
class DeepOutcome:
    """一次深轨的产物:终答(MemoryAnswer,cited 已回译真 id)+ 工具轨迹 + 写回数。"""
    ans: MemoryAnswer
    steps: list[dict] = field(default_factory=list)   # [{tool, args, obs_head}]
    handoff: str = ""
    remembered: int = 0
    secs: dict[str, float] = field(default_factory=dict)


def _parse_final(out: Any, reg: HandleRegistry) -> MemoryAnswer:
    """executor 终态 → MemoryAnswer:支持 dict / JSON 串 / 纯文本;cited 回译真 id。"""
    obj = out if isinstance(out, dict) else None
    if obj is None and isinstance(out, str):
        try:
            obj = json.loads(strip_fences(out))
        except (json.JSONDecodeError, ValueError):
            obj = None
    if isinstance(obj, dict) and "answer" in obj:
        cited = [reg.real(str(x).strip()) for x in (obj.get("cited") or [])]
        return MemoryAnswer(answer=str(obj.get("answer") or "").strip(),
                            cited_cells=[c for c in cited if c], raw=str(out))
    if isinstance(out, str) and out.startswith("Agent stopped"):
        logger.warning(f"深轨触发步数上限,无终答 steps={len(reg)}")
        return MemoryAnswer(answer="")
    return MemoryAnswer(answer=out if isinstance(out, str)
                        else json.dumps(out, ensure_ascii=False), raw=str(out))


def run_deep(llm: ChatLLM, embedder, atoms: AtomStore, cells: CellStore, evidence: EvidenceStore, *,
             query: str, now_dt: datetime, rw: QueryRewrite | None = None,
             review: ReviewResult | None = None, fast_hits: list[CellHit] | None = None,
             reranker: Reranker | None = None, deep_write: bool | None = None,
             media_store=None, mllm=None, profile: str = "", scenario: str = "") -> DeepOutcome:
    """深轨一次完整运行:交接包开局 → agent 六工具循环(上限 9 步,每步可批量调用)→ 终答回译。

    失败语义:agent 崩溃向上抛(调用方决定回退);步数耗尽 → 空答案(如实"没答出来")。
    """
    t0 = time.perf_counter()
    # 看图目的 = 用户原始问题(resolved 优先,含指代消解);贯穿所有证据工具的看图点
    task_query = (rw.resolved if rw and rw.resolved else query) or query
    d = DeepDeps(embedder=embedder, reranker=reranker or NoopReranker(),
                 atoms=atoms, cells=cells, evidence=evidence, llm=llm,
                 deep_write=settings.deep_write if deep_write is None else deep_write,
                 task_query=task_query, media_store=media_store, mllm=mllm)

    all_cells = list(cells.iter_all())               # old → new
    catalog = list(reversed(all_cells))[:_CATALOG_SIZE]
    handoff = build_handoff(d, query=query, now_dt=now_dt, rw=rw, review=review,
                            fast_hits=fast_hits, total_cells=len(all_cells), catalog=catalog,
                            profile=profile)

    tools = build_tools(d)
    model = MaasChatModel(client=llm)
    agent = _build_agent(model, tools, scenario)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=_MAX_STEPS,
                             handle_parsing_errors=_PARSE_NUDGE,
                             return_intermediate_steps=True)   # 工具异常兜底在 build_tools 的闭包层
    logger.info(f"deep 交接包(agent 输入)q={query!r}\n{handoff}")
    with obs.observation("deep_recall", as_type="agent", input=task_query,
                         metadata={"max_steps": _MAX_STEPS}):   # 嵌在 recall 根 trace 下
        result = executor.invoke({"input": handoff})
    raw_steps = result.get("intermediate_steps") or []
    steps = [{"tool": a.tool, "args": a.tool_input, "obs_head": str(o)[:160]}
             for a, o in raw_steps]
    for i, (a, o) in enumerate(raw_steps, 1):   # 完整打每步:工具 + 全参 + 工具返回(观测)全文
        logger.info(f"deep step{i}/{_MAX_STEPS} tool={a.tool} args={a.tool_input}\n"
                    f"  ── 工具返回 ──\n{o}")
    ans = _parse_final(result.get("output"), d.reg)
    secs = {"deep": round(time.perf_counter() - t0, 3)}
    logger.info(f"deep 完成 steps={len(steps)} remembered={d.remembered} "
                f"cited={len(ans.cited_cells)} ans={len(ans.answer)}字 q={query!r}\n"
                f"  ── 终答 ──\n{ans.answer}")
    return DeepOutcome(ans=ans, steps=steps, handoff=handoff,
                       remembered=d.remembered, secs=secs)
