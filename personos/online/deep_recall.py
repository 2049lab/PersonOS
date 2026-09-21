"""The deep track (fused architecture §4): a single agent with six tools that leafs through memory,
harnessed by langchain's JSON protocol.

How it divides labour with the fast path: the fast path retrieves once and answers once — fast and
shallow; the deep track is an agent that CAN SEE what memory looks like — it starts from a handoff
package (task / what the fast path already found / the gap adjudication named / the catalogue /
the current time), uses its tools to open cells, read raw transcripts and (sparingly) write back, and
produces the final answer itself. Entry points: run_recall with mode=deep goes straight here;
mode=auto escalates when R3 returns partial or empty.

One convention runs through it: atoms are only a retrieval face — none of the material the agent can
see contains atom text; search_atoms uses atoms to locate but returns whole-cell material
(topic + episode); open_cell only reports the COUNT of extracted index atoms.

Trace: every tool call logs an INFO line and is surfaced in `steps` (so benchmarking and the
workbench can see how far the deep track got).
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

# langchain 1.x: the classic agents (JSON protocol, no dependency on gateway tool-calling) live in
# langchain-classic
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

_MAX_STEPS = 9         # cap on the agent's tool calls (each step may batch several calls; every observation carries a remaining-budget hint)
_REMEMBER_CAP = 8      # cap on remember write-backs per session (a guard; writes past it are refused)
_CATALOG_SIZE = 10     # the opening catalogue: the N most recent cells in reverse chronological order
_PAGE_CELLS = 10       # cells per page in find_cells
_PAGE_LINES = 30       # utterances per page in get_cell_evidence
_DEFAULT_LIMIT = 5     # default number of cells search_atoms returns
_MAX_LIMIT = 8         # the cap (the agent can tune the limit; it is not fixed)


def _coerce_int(v, default: int) -> int:
    """Argument-schema tolerance: the model occasionally passes page/limit as a string ("abc") or
    None, so fall back to the default silently.

    This lives at the pydantic argument layer rather than relying on handle_tool_error — that layer
    only catches exceptions raised inside the tool function, whereas an argument validation error
    would escape straight out of AgentExecutor and cut the whole round short.
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# -- The short-handle registry (the deep track's reuse of R5's h2id pattern) --

class HandleRegistry:
    """A per-session registry mapping a cell's real id <-> its short handle (c1..cN).

    Handles are assigned in order of FIRST ENTRY INTO THE CONTEXT (in the opening catalogue c1 is the
    most recent) and are stable from then on: whenever the agent writes c3 it means the same cell, and
    the `cited` of the final answer is translated back into real ids for the caller.
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


# -- Renderers (the textual shape of what the tools return) --

def _fmt_ts(dt: datetime | None) -> str:
    dt = ensure_aware(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "(time unknown)"


def cell_full(cell: MemCell, handle: str, n_atoms: int, n_lines: int) -> str:
    """One cell's full material: the separator line + the metadata header (the same convention as the
    fast path's cell_lead) + the episode as primary material + a note about its index and transcript.

    Atom text is deliberately absent: atoms are only a retrieval face, and showing them to the agent
    would let them be read as a "reference answer" and muddy its judgment.
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
    """A lightweight catalogue row: handle | start time (yyyy-MM-dd HH:mm:ss) | topic."""
    return f"{handle} | {_fmt_ts(cell.t_start)} | {cell.topic or '(no topic)'}"


def _look_image_note(d: "DeepDeps", rec) -> str:
    """Look at this image evidence's original again, carrying the user's own question (d.task_query),
    and return supplementary facts that can be appended to the transcript line.

    At write time, content_inline already holds a general understanding produced WITH DIALOGUE
    CONTEXT; this is a targeted follow-up WITH A PRECISE QUESTION, filling in details the ingest pass
    may have missed (closing the caption information-loss gap). Anything missing or failing returns an
    empty string (degrade).
    """
    if rec.modality not in ("image", "mixed") or not rec.content_ref:
        return ""
    if d.media_store is None or d.mllm is None or not getattr(d.mllm, "available", False):
        return ""
    purpose = d.task_query or ""
    if not purpose.strip():
        return ""
    # The content type is inferred from the OSS key's extension (EvidenceRecord does not store
    # content_type separately)
    ext = rec.content_ref.rsplit(".", 1)[-1].lower() if "." in rec.content_ref else "jpg"
    ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "webp": "image/webp", "gif": "image/gif", "heic": "image/heic"}.get(ext, "image/jpeg")
    try:
        img = d.media_store.read_bytes(rec.content_ref)
        text = d.mllm.look_image(img, purpose, content_type=ctype)
    except Exception:   # noqa: BLE001  a failed image read does not affect showing the raw transcript
        return ""
    return f"  ↳[看图·针对「{purpose[:30]}」] {text}" if text else ""


def _render_evidence_line(d: "DeepDeps", r) -> str:
    """Render one piece of evidence as one line; image evidence additionally carries one targeted
    re-reading of the image."""
    base = f"[{_fmt_ts(r.captured_at)}] {r.holder}: {r.content_inline or ''}"
    note = _look_image_note(d, r)
    return base + ("\n" + note if note else "")


def evidence_page(records: list, page: int, per_page: int = _PAGE_LINES,
                  d: "DeepDeps | None" = None) -> tuple[str, int]:
    """Render the raw transcript page by page, one line each: [timestamp] speaker: utterance. The
    speaker follows the evidence holder (in the third person that is the real name).

    When `d` is given and a line is image evidence, the original is re-read with d.task_query and one
    line of targeted facts is added.
    Returns (rendered text, total page count); with no records it returns one actionable note.
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


# -- Tool implementations (pure functions that touch only DeepDeps; they do not touch langchain, which
# makes them directly testable) --

@dataclass
class DeepDeps:
    """The dependency bundle of the deep-track tools + the session state (handle registry, write-back
    counter)."""
    embedder: Any
    reranker: Reranker
    atoms: AtomStore
    cells: CellStore
    evidence: EvidenceStore
    llm: Any = None              # the weaver (used by search_atoms' unit assembly; None means every chain degrades to plain cells)
    reg: HandleRegistry = field(default_factory=HandleRegistry)
    deep_write: bool = True
    remembered: int = 0
    calls_made: int = 0          # tool steps used so far (a batch counts as 1; the data behind the budget hint at the end of each observation)
    # Image viewing: task_query is the user's own question (the purpose of looking); without
    # media_store/mllm injected no image is read and it degrades to reading content_inline
    task_query: str = ""
    media_store: Any = None
    mllm: Any = None


_EMPTY_HINT = ("No hits. Try: a more specific search term (person name / matter / date verbatim), "
               "relaxing the date-window/domain/holder filters, or find_cells to browse by time. "
               "When the retrieval chain keeps coming up empty, fall back to search_evidence to "
               "keyword-search the raw transcript directly.")


def tool_search_atoms(d: DeepDeps, *, query: str, start_date: str = "", end_date: str = "",
                      domains: list[str] | None = None, holder: str = "", limit: int = _DEFAULT_LIMIT) -> str:
    """Search by fact = a customizable fast path: structured filters -> the R1 atom pool -> unit
    assembly (woven or plain) -> R2 rerank.

    Same mechanism and same code as fast-recall (the pool, the assembly and the rerank are all
    reused); the agent just gets three extra things: it writes the search terms itself, it sets the
    filters (date window / domains / holder), and it sets `limit`, the number of units returned. The
    pool is 2 x limit (an agent's per-step window is smaller than the one-shot fast path's). A woven
    unit (memcell') registers a handle for each cell it covers, so its block header lists several cN
    side by side — the chain id itself never enters the registry, and open_cell / get_cell_evidence /
    remember only accept real cells, so long and short ids can never collide.
    """
    limit = max(1, min(_MAX_LIMIT, int(limit or _DEFAULT_LIMIT)))
    rw = QueryRewrite(original=query, resolved=query, domains=domains or [])
    pool = _pool_search(d.embedder, d.atoms, rewrite=rw, top_n=2 * limit,
                        start_date=start_date, end_date=end_date, holder=holder,
                        domains_filter=domains or None)   # a domain condition the agent set is a hard filter
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
    if asm.boundary:   # the same boundary note as the fast path (fake gaps covered by a weave or by the same cell are already filtered out)
        out += f"\n\nBOUNDARY\n{asm.boundary}"
    return out


def _unit_block(d: DeepDeps, u: CellHit) -> str:
    """Deep-track rendering of one material unit: the handle is built by registering each covered cell
    (so a woven unit shows several cN side by side).

    The index and transcript sizes are summed over the covered cells — a memcell' is a temporary view
    whose own chain id corresponds to nothing in storage, so drilling down goes through the member
    cells' handles.
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
    """List cells by condition: a time window (on the cell's start time) plus a domain filter; ordered
    by topic similarity when a query is given, otherwise reverse chronological; paginated."""
    lo, hi = _date_window(start_date, end_date)
    domains = set(domains or [])
    selected = []
    for c in d.cells.iter_all():                   # old → new
        if (lo or hi) and not _in_window(ensure_aware(c.t_start), lo, hi):
            continue                               # under a time filter a cell with no time anchor is invisible
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
                      if c.id in ranked_ids else len(ranked_ids))   # cells without a topic vector go last
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
    """Open one cell: the full narrative (episode) plus its index and transcript sizes."""
    if not d.reg.real(c):
        return (f"Unknown handle {c!r}: only handles that appeared in find_cells or search_atoms "
                f"results can be opened.")
    cell = _resolve_cell(d, c)
    if cell is None:
        return "This cell no longer exists (it may have been purged)."
    return cell_full(cell, d.reg.ensure(cell.id),
                     len(d.atoms.list_by_cell(cell.id)), len(cell.evidence_refs))


def tool_get_cell_evidence(d: DeepDeps, *, c: str, page: int = 1) -> str:
    """Page through a cell's raw dialogue (utterance by utterance with speaker and timestamp), 30
    utterances per page, navigated with `page`."""
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
    """Keyword-search the raw transcript directly (whole store, bypassing every index): all keywords
    must hit the same utterance, and the matching utterances are returned with their owning cell.

    The fallback path: search_atoms and find_cells are built on the atom and cell indexes, so when an
    extraction misses something, the utterance sits in the ground-truth layer with no route to it —
    this supplies that route (the raw transcript is the complete ground-truth layer, and a LIKE scan
    suffers no extraction loss at all).
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
        # When the hit is image evidence, one targeted re-reading with task_query is appended
        # (_render_evidence_line checks the modality internally)
        lines.append(_render_evidence_line(d, r) + f" {h}")
    return (f"search_evidence hit {len(recs)} utterance(s) (keywords AND in the same utterance: "
            f"{' / '.join(kws)}):\n" + "\n".join(lines)
            + "\n(use open_cell / get_cell_evidence on the owning cell for context)")


def tool_remember(d: DeepDeps, *, c: str, text: str = "", quote: str = "", holder: str = "user",
                  kind: str = "", domains: list[str] | None = None, episode_append: str = "") -> str:
    """Write back (sparingly): add one retrieval index atom to a cell (with `quote` linking verbatim
    back to the raw utterance) and/or append to the end of its narrative.

    It only adds, never rewrites; the whole session is capped at _REMEMBER_CAP entries; with
    deep_write=False everything is read-only.
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
        refs = _match_evidence_refs(records, quote)   # same as W2 step 2: link back by verbatim substring
        atom = MemoryAtom(memcell_id=cell.id, text=text.strip(),
                          holder=(holder or "user").strip() or "user",
                          domains=list(domains or []), kind=kind or None,
                          occurrence_time=cell.t_start, evidence_refs=refs, source="deep")
        d.atoms.upsert(atom, embedding=d.embedder.embed([stamped_atom_text(atom)])[0])
        d.remembered += 1
        wrote.append(f"added 1 index atom (linked to {len(refs)} source utterance(s))")
    if episode_append.strip():
        new_episode = ((cell.episode or "").rstrip() + "\n" + episode_append.strip()).strip()
        d.cells.upsert(cell.model_copy(update={"episode": new_episode}))   # embedding=None keeps the existing topic vector
        wrote.append("appended 1 passage to the episode")
    logger.info(f"deep remember cell={cell.id} wrote={wrote}")
    return ("Written back: " + "; ".join(wrote) + ". The new index is retrievable on the next search.")


# -- langchain wiring --

class MaasChatModel(BaseChatModel):
    """A langchain adapter for the ChatLLM protocol (MaasClient / FakeLLM).

    The infrastructure still goes through the provider layer (timeout tiers and retries all live
    in that layer); this only maps message roles and applies `stop` client-side — the gateway does not
    forward stop, and the JSON agent relies on it to prevent the model hallucinating a continuation.
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


# Schema hardening (2026-09-09; measured: 19% of tool calls carried arguments outside the schema). The
# frequent wrong names are absorbed as AliasChoices aliases and a time_range dict is consumed and
# mapped by a before-validator; extra="forbid" makes any remaining odd argument fail loudly so the
# agent corrects itself, instead of being silently dropped (the agent believing it filtered by date
# when it did not is the direct mechanism behind wrong answers on counting questions).
def _absorb_time_range(data, *, has_date_fields: bool = True):
    """A time_range={start,end} in the argument dict -> start_date/end_date (it must be consumed
    before extra=forbid sees it).

    For schemas with has_date_fields=False (search_evidence has no date-filtering ability), it is only
    popped to avoid a forbid error — the strategy section already states that tool has no date window,
    and the agent checks the timestamps of the hits itself.
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
    """A bare list as action_input means calling the same tool several times in one step: wrap it into
    the `batch` field and hand it to the execution layer.

    The agent's rapid-fire pattern (observed: five consecutive single-keyword search_evidence calls)
    burns one step of budget each time; the batch form does it all in one step and saves steps. The
    execution layer runs them serially (the storage layer's shared pymysql connection is not
    thread-safe), so what is saved is step budget, not wall clock. In the batch form the schema's
    required primary fields (query / c / keywords) are let through with placeholder values — the real
    per-item validation happens in the execution layer, against each argument object inside the batch.
    """
    if isinstance(data, list):
        data = {"batch": data}
    if isinstance(data, dict) and data.get("batch") is not None:
        for k, v in required_placeholder.items():
            data.setdefault(k, v)
    return data


# The batch argument field (shared by all six schemas): the agent may pass a bare list as action_input
# or an explicit {"batch": [...]}
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
        # In practice the model often passes keywords as a bare string ("beach") — wrap it into a
        # one-element list rather than interrupting its self-correction.
        # Note that the alias keys (query / kw / keyword / ...) are still unrenamed at the before
        # stage, so each key is checked individually
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
    """A list as action_input (one batched step) -> wrap it into {"batch": [...]} before constructing
    the AgentAction.

    langchain's AgentAction.tool_input only accepts a str or a dict, and a bare list raises a
    ValidationError while the base class's parse instantiates it — so the whole parse has to be
    overridden to adapt the shape before construction (observed: when the agent fires off batched
    calls as the protocol allows, the entire round jams on parse retries, with all 9/9 steps being
    _Exception). The batch semantics themselves are executed by the tools' `batch` field; this only
    adapts the shape.
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
    """Replicate create_json_chat_agent's default assembly, swapping only the output parser for the
    batch-tolerant one.

    A non-empty scenario inserts the caller-scenario section into the system prompt (see
    _agent_prompt); an empty one leaves it byte-for-byte unchanged.
    """
    prompt = _agent_prompt(scenario).partial(tools=render_text_description(list(tools)),
                                             tool_names=", ".join(t.name for t in tools))
    return (RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_messages(x["intermediate_steps"]))
            | prompt | model.bind(stop=["\nObservation"]) | _BatchTolerantJSONParser())


def _tool_error_as_observation(err: Exception) -> str:
    """A tool failure -> observation text. The framework's handle_tool_error only catches
    ToolException, and a generic exception would escape and cut the whole agent round short — so this
    backstop lives in our own closure instead.
    """
    return (f"Tool call failed: {err}. Check the arguments (field names and types per the tool spec), "
            f"or try another tool / relax the filters.")


def _args_validation_hint(err) -> str:
    """Argument validation failed (a misspelled field name, a wildly wrong type) -> an actionable
    observation hint so the agent corrects itself."""
    return (f"Tool argument validation failed: {err}. Retry with field names and types exactly as "
            f"specified (the cell-handle field is named c, e.g. 'c3').")


def build_tools(d: DeepDeps) -> list[StructuredTool]:
    """The six tools -> langchain StructuredTools (the closure captures DeepDeps, so the agent's
    session state travels with them)."""
    def _mk(name: str, desc: str, schema: type[BaseModel], fn) -> StructuredTool:
        def _invoke(kw: dict) -> str:                        # bind the session-state DeepDeps
            try:
                return fn(d, **kw)
            except Exception as e:                           # noqa: BLE001 any failure degrades into an observation
                logger.warning(f"deep tool {name} failed: {e!r} args={kw}")
                return _tool_error_as_observation(e)

        def _bound(**kw):
            batch = kw.pop("batch", None)
            if batch:                                        # one batched step: validate each item, then execute serially
                try:                                         # (the shared pymysql connection is not thread-safe, so no parallelism)
                    calls = [{k: v for k, v in schema.model_validate(item)
                              .model_dump(exclude_none=True).items() if k != "batch"}
                             for item in batch]
                except ValidationError as e:
                    return _args_validation_hint(e)
                out = "\n\n".join(f"── batch {i}/{len(calls)} ──\n{_invoke(c)}"
                                  for i, c in enumerate(calls, 1))
            else:
                out = _invoke(kw)
            d.calls_made += 1                                # a batch counts as a single step
            left = _MAX_STEPS - d.calls_made
            tail = f"≈{left} tool steps left"
            if left <= 2:                                    # budget nearly gone: push the agent to stop and answer
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


# -- The system prompt: teaching the big picture + the multi-hop method + the JSON output protocol --

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

# Directive for caller-scenario injection (it only steers the search direction; fabrication is
# forbidden) -- see llm.with_scenario
_SCEN_DIR_AGENT = ("Use it to steer which matters to prioritize digging into and which leads to "
                   "follow first; it never licenses fabricating facts absent from the store.")


def _agent_prompt(scenario: str) -> ChatPromptTemplate:
    """The deep-track agent's system prompt template; an empty scenario reuses the constant (leaving it
    byte-for-byte unchanged).

    The braces in the scenario are escaped before it is spliced into the system prompt — _AGENT_SYS
    relies on {tools}/{tool_names} as template variables, so unescaped {} in the caller's text would
    be parsed by ChatPromptTemplate as variables and raise an error.
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


# -- The handoff package (six natural-language sections; it hands over evidence but not the draft
# answer, to avoid anchoring) --

_REVIEW_TEXT = {"answer_defect": "a draft was answered from the materials above and reviewed; "
               "even the re-answer failed itemized checks",
               "insufficient_material": "the retrieved materials lack the core of the answer"}


def _covered_ids(h: CellHit) -> list[str]:
    """The cell ids a material unit covers: a woven unit covers its full set of member cells
    (`covers`), a plain unit covers itself."""
    return h.covers or [h.cell.id]


def _translate_handles(text: str, fast_hits: list[CellHit], d: DeepDeps) -> str:
    """Translate the fast-path window handles in a critique (mN, used on the adjudication side) into
    deep-track catalogue handles (cN).

    The two numbering schemes serve different windows: the fast path's m1..mK follow the order of the
    R5/R3' material, while the deep track's c1..cN follow catalogue registration order. Adjudication
    only ever saw its own material window, so the mN it writes can only refer to fast-path material —
    translated back through the order of fast_hits, with a woven unit expanded into the member cells
    it covers (each ensured in turn, several joined by commas). An out-of-range or unrecognized
    handle is left as-is (a visibly strange token beats silently pointing at the wrong cell). It also
    tolerates the model slipping and writing cN.
    """
    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return (",".join(d.reg.ensure(cid) for cid in _covered_ids(fast_hits[i - 1]))
                if 1 <= i <= len(fast_hits) else m.group(0))

    # The boundary is defined by ASCII alphanumerics rather than \b: in a Chinese critique, a
    # no-space form like "材料m1" does not satisfy \b, because CJK characters are \w too
    return re.sub(r"(?<![A-Za-z0-9])[mc](\d+)(?![A-Za-z0-9])", repl, text)


def build_handoff(d: DeepDeps, *, query: str, now_dt: datetime,
                  rw: QueryRewrite | None, review: ReviewResult | None,
                  fast_hits: list[CellHit] | None, total_cells: int,
                  catalog: list[MemCell], profile: str = "") -> str:
    """The fast path -> deep track handoff package: (1) the task, (2) the evidence the fast path
    already found, (3) the adjudication verdict and the gap, (4) the memory catalogue, (5) the current
    time.

    The system prompt (which teaches the big picture) is always resident and does not take up space in
    the handoff. The fast path's draft answer is deliberately withheld: the deep track has to judge
    independently, inheriting only "what material was already retrieved + the gap adjudication named"
    (a critique is a direction, a draft is an anchor — give the direction, not the anchor).
    The fast-path handles mN inside the critique are translated here into deep-track cN — the material
    section is rendered with deep-track handles, and the gap has to line up with them.
    The catalogue cells enter the registry first (c1 = the most recent) and the material section
    reuses those handles: the deep-track numbering is determined solely by the store's content and has
    nothing to do with what the fast path hit or in what order — the two id systems are independent.
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
        # A woven unit's handle is its covered member cells (comma-joined); a plain unit is a single
        # cell, behaving exactly as before
        blocks = [cell_block(h, ", ".join(d.reg.ensure(i) for i in _covered_ids(h)))
                  for h in top]   # the same renderer as R3/R5
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


# -- Orchestration --

@dataclass
class DeepOutcome:
    """The product of one deep-track run: the final answer (a MemoryAnswer whose `cited` has already
    been translated back into real ids) + the tool trace + the write-back count."""
    ans: MemoryAnswer
    steps: list[dict] = field(default_factory=list)   # [{tool, args, obs_head}]
    handoff: str = ""
    remembered: int = 0
    secs: dict[str, float] = field(default_factory=dict)


def _parse_final(out: Any, reg: HandleRegistry) -> MemoryAnswer:
    """The executor's final state -> a MemoryAnswer: accepts a dict, a JSON string, or plain text;
    `cited` is translated back into real ids."""
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
        logger.warning(f"deep track hit the step limit with no final answer steps={len(reg)}")
        return MemoryAnswer(answer="")
    return MemoryAnswer(answer=out if isinstance(out, str)
                        else json.dumps(out, ensure_ascii=False), raw=str(out))


def run_deep(llm: ChatLLM, embedder, atoms: AtomStore, cells: CellStore, evidence: EvidenceStore, *,
             query: str, now_dt: datetime, rw: QueryRewrite | None = None,
             review: ReviewResult | None = None, fast_hits: list[CellHit] | None = None,
             reranker: Reranker | None = None, deep_write: bool | None = None,
             media_store=None, mllm=None, profile: str = "", scenario: str = "") -> DeepOutcome:
    """One complete deep-track run: open with the handoff package -> the agent's six-tool loop (capped
    at 9 steps, each of which may batch calls) -> translate the handles in the final answer.

    Failure semantics: an agent crash propagates upwards (the caller decides how to fall back); an
    exhausted step budget yields an empty answer (an honest "could not answer it").
    """
    t0 = time.perf_counter()
    # The image-viewing purpose is the user's own question (preferring `resolved`, which has
    # references resolved); it runs through every image-viewing point of the evidence tools
    task_query = (rw.resolved if rw and rw.resolved else query) or query
    d = DeepDeps(embedder=embedder, reranker=reranker or NoopReranker(),
                 atoms=atoms, cells=cells, evidence=evidence, llm=llm,
                 deep_write=settings.deep_write if deep_write is None else deep_write,
                 task_query=task_query, media_store=media_store, mllm=mllm)

    all_cells = list(cells.iter_all())               # old -> new
    catalog = list(reversed(all_cells))[:_CATALOG_SIZE]
    handoff = build_handoff(d, query=query, now_dt=now_dt, rw=rw, review=review,
                            fast_hits=fast_hits, total_cells=len(all_cells), catalog=catalog,
                            profile=profile)

    tools = build_tools(d)
    model = MaasChatModel(client=llm)
    agent = _build_agent(model, tools, scenario)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=_MAX_STEPS,
                             handle_parsing_errors=_PARSE_NUDGE,
                             return_intermediate_steps=True)   # the backstop for tool exceptions lives in build_tools' closure layer
    logger.info(f"deep handoff package (agent input) q={query!r}\n{handoff}")
    with obs.observation("deep_recall", as_type="agent", input=task_query,
                         metadata={"max_steps": _MAX_STEPS}):   # nested under the recall root trace
        result = executor.invoke({"input": handoff})
    raw_steps = result.get("intermediate_steps") or []
    steps = [{"tool": a.tool, "args": a.tool_input, "obs_head": str(o)[:160]}
             for a, o in raw_steps]
    for i, (a, o) in enumerate(raw_steps, 1):   # log every step in full: the tool + all its arguments + the tool's entire returned observation
        logger.info(f"deep step{i}/{_MAX_STEPS} tool={a.tool} args={a.tool_input}\n"
                    f"  -- tool observation --\n{o}")
    ans = _parse_final(result.get("output"), d.reg)
    secs = {"deep": round(time.perf_counter() - t0, 3)}
    logger.info(f"deep finished steps={len(steps)} remembered={d.remembered} "
                f"cited={len(ans.cited_cells)} ans_chars={len(ans.answer)} q={query!r}\n"
                f"  -- final answer --\n{ans.answer}")
    return DeepOutcome(ans=ans, steps=steps, handoff=handoff,
                       remembered=d.remembered, secs=secs)
