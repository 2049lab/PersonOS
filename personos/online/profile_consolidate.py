"""画像整理(consolidate):强模型一次出**补丁**,harness 打回 + 超限打回;短↔长 id 映射。

单阶段(无 digest):输入 = 当前画像(结构化,剥掉 sources 长 id)+ 上次出版本以来的新 cell
(episode + atoms,标短标 c1..cN)+ 今天。LLM 只产补丁(增改汰,带自己选),profile_merge 合并,
profile_harness 校验;两类打回:字段非法 / 某带超限。用尽:字段级不过保留旧版(返 None);
仅超限则引擎踢最旧兜底后落库。

id 映射规范(见记忆 llm-id-reference-short-long-mapping):LLM 只见/只写短标 c1..cN;
校验按短标集;合并前短→长回填成真实 cell_id。
"""

from __future__ import annotations

import json
from datetime import date

from loguru import logger

from .. import obs
from ..models import MemCell, MemoryAtom
from ..storage.profile_store import UserProfile
from .llm import ChatLLM, chat_json, with_scenario
from .profile_harness import validate_patch
from .profile_merge import BAND_CAPS, apply_patch, enforce_caps, over_cap

# 业务方场景注入 directive(只调关注度,不许编造画像)——见 llm.with_scenario
_SCEN_DIR_PROFILE = ("Weight traits and facts in the caller's focus area as higher-value to keep and "
                     "keep current; it never licenses inventing profile content unsupported by the "
                     "memory.")

_CONSOLIDATE_SYS = """# Role
You are the user-profile consolidator in a personal memory system. Given the CURRENT PROFILE
(structured JSON, may be empty), NEW MEMORY (recently closed topic cells: a narrative episode plus
extracted atoms each, labelled c1, c2, ...), and TODAY's date, output a PATCH that updates the profile.

The profile answers two things about the user:
- traits: what kind of person the user is (a stable portrait, inference allowed but marked)
- facts: noteworthy things that happened to / about the user, organized into recency bands

# Output: a PATCH, not a full rewrite (JSON only)
Emit ONLY what changes; anything you do not mention is preserved as-is. Never restate unchanged content.
{
  "traits": { "<domain>": {"text": "...", "status": "confirmed|inferred", "sources": ["c1"]},
              "<domain>": null },
  "facts": {
    "add":     [ {"band": "today|week|month|long", "text": "...", "sources": ["c1"]} ],
    "rewrite": [ {"id": "f_xxx", ...only the fields you change..., "band": "week"(optional, to move)} ],
    "drop":    [ "f_xxx" ]
  }
}
- sources MUST be short labels (c1, c2, ...) of cells in NEW MEMORY — never invent one, never write a long id.
- Do NOT write dates as a separate field — write dates INSIDE the text in natural language
  ("2026-09-14 had ramen"; "married in 2020"); resolve relative words to absolute dates.
- "traits": {"domain": null} clears a domain. Omit a domain to leave it untouched.

# traits — the portrait (PMO-16 domains in three layers; use these EXACT keys)
- L1 dispositional (what the person is like overall): personality, communication_style, social_style
- L2 characteristic adaptations (what they want / value / how they operate): occupation, goals, values,
  work_style, learning_style, tech_environment, lifestyle, health, finance
- L3 narrative identity (how they understand their own life): identity, location, family, interests
Rules:
- ONE concise sentence per domain (<= 500 chars). Prefer an if-then behavioral signature
  ("in technical talk prefers direct feedback; brief in small talk") over a flat trait ("is direct").
- REWRITE-INTEGRATE, do not accumulate: when a domain changes, rewrite the whole sentence folding
  old + new into one smooth sentence; never append fragments or let it grow longer and longer.
- status: confirmed = user stated it; inferred = you deduced from behavior (allowed, but mark it).
- A domain no dialogue exposes stays null — never fabricate to fill a slot.

# facts — recency bands (YOU place each fact in a band; the engine only enforces per-band caps)
Band responsibilities:
- today  (max 1): TODAY's facts only, integrated into ONE item.
- week   (max 3): facts from roughly the last 7 days.
- month  (max 5): facts from roughly the last 30 days.
- long   (max 21): only genuinely ENDURING facts (identity, lasting preferences, stable relationships,
  values). Put a fact here by "band":"long".
Near bands may hold plain objective facts (a log); long holds only what stays true.

YOU maintain the bands as of TODAY (there is no automatic date sorting):
- Roll aged facts DOWN yourself: a fact that was "today" yesterday is no longer today's — move it to
  week (rewrite with "band":"week"), and free the today slot for today's new content. Likewise week→month.
- Never keep extending a past day's fact to stuff today's events into it — create a NEW fact for today,
  and let the older one roll down (or drop / promote it).
- A single fact MAY span multiple days as a theme ("worked Mon–Fri on project X" = one week fact) —
  the band is your judgement of recency/importance, not a per-day limit.
- If a band would exceed its cap, MERGE related items, DROP the least important, or PROMOTE the enduring
  one to long — do not overflow.

Fact writing discipline (mirror atom extraction):
- Keep every qualifier (when / where / with whom / how / how often / until when); proper nouns and
  numbers verbatim; third person; dates written in the text (dual time: relative + absolute).
- One fact may INTEGRATE several atoms sharing one bounding condition (e.g. same day).
- Facts are observed, not guessed — record what was said.

# Language
Write trait/fact text in the SAME language as the source dialogue.

# Output JSON only (no prose, no code fences)."""


def _sanitized_profile_json(current: UserProfile | None) -> str:
    """给 LLM 看的当前画像:剥掉 sources(真实长 id,LLM 不需要也不该抄),保留 id/text/band。"""
    if current is None:
        return "{}"
    d = current.model_dump()
    for t in d.get("traits", {}).values():
        if t:
            t.pop("sources", None)
    for band in d.get("facts", {}).values():
        for f in band:
            f.pop("sources", None)
    return json.dumps(d, ensure_ascii=False)


def _render_input(current: UserProfile | None, cells: list[MemCell],
                  atoms_by_cell: dict[str, list[MemoryAtom]], today: date) -> tuple[str, dict[str, str]]:
    """渲染用户消息 + 返回 {短标: 真实cell_id} 映射(新 cell 标 c1..cN)。"""
    cell_map: dict[str, str] = {}
    blocks = []
    for i, c in enumerate(cells, 1):
        label = f"c{i}"
        cell_map[label] = c.id
        lines = [f"━━━ {label} ━━━ topic: {c.topic}", f"episode: {c.episode}"]
        atoms = atoms_by_cell.get(c.id, [])
        if atoms:
            lines.append("atoms:")
            lines.extend(f"- {a.text}" for a in atoms)
        blocks.append("\n".join(lines))
    material = "\n\n".join(blocks) if blocks else "(none)"
    msg = (f"TODAY: {today.isoformat()}\n\n"
           f"CURRENT PROFILE (JSON; {{}} = no profile yet):\n{_sanitized_profile_json(current)}\n\n"
           f"NEW MEMORY (topic cells closed since last consolidation; cite these labels as sources):\n"
           f"{material}")
    return msg, cell_map


def _remap_sources(patch: dict, cell_map: dict[str, str]) -> dict:
    """短标 → 真实 cell_id 回填(校验已保证短标都在集内)。返回新 patch,不改入参。"""
    def mp(src):
        return [cell_map[s] for s in (src or []) if s in cell_map]

    out = json.loads(json.dumps(patch))          # 深拷贝
    for t in (out.get("traits") or {}).values():
        if isinstance(t, dict) and "sources" in t:
            t["sources"] = mp(t["sources"])
    facts = out.get("facts") or {}
    for a in facts.get("add") or []:
        if isinstance(a, dict) and "sources" in a:
            a["sources"] = mp(a["sources"])
    for rw in facts.get("rewrite") or []:
        if isinstance(rw, dict) and "sources" in rw:
            rw["sources"] = mp(rw["sources"])
    return out


def _retry_msg(errs: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errs[:20])
    return ("你上一版补丁有以下问题,请逐条修正后重新**只输出 JSON 补丁本体**"
            f"(不要解释、不要代码块围栏):\n{bullet}")


def consolidate(llm: ChatLLM, *, current: UserProfile | None, cells: list[MemCell],
                atoms_by_cell: dict[str, list[MemoryAtom]], today: date,
                max_retries: int = 2, max_tokens: int = 4096,
                scenario: str = "") -> UserProfile | None:
    """跑一次整理。无新 cell → None。字段级打回用尽仍不过 → None(保留旧版);
    仅超限收敛不了 → 引擎踢最旧兜底后返回。
    scenario:业务方场景描述(可空)——偏重调用方关注域的 traits/facts。"""
    if not cells:
        return None
    today_str = today.isoformat()
    user_msg, cell_map = _render_input(current, cells, atoms_by_cell, today)
    valid = set(cell_map.keys())
    sys = with_scenario(_CONSOLIDATE_SYS, "# Output: a PATCH, not a full rewrite (JSON only)",
                        scenario, _SCEN_DIR_PROFILE)
    messages = [{"role": "system", "content": sys},
                {"role": "user", "content": user_msg}]
    last_valid: UserProfile | None = None         # 字段合法但超限的候选(兜底用)

    for attempt in range(max_retries + 1):
        raw = ""
        try:
            patch, raw = chat_json(llm, messages, max_tokens=max_tokens, temperature=0.2,
                                   stage="profile_consolidate")
            errs = validate_patch(patch, valid_cell_ids=valid)
        except ValueError as e:                   # 输出非合法 JSON
            raw = getattr(e, "raw", "")
            patch, errs = None, ["你的输出不是合法 JSON,请只输出 JSON 补丁本体"]

        if patch is not None and not errs:
            merged = apply_patch(current, _remap_sources(patch, cell_map),
                                 today_str=today_str, evict=False)
            over = over_cap(merged)
            if not over:
                logger.info(f"consolidate 成功 attempt={attempt + 1} cells={len(cells)}")
                return merged
            last_valid = merged                    # 字段合法,仅超限
            errs = [f"{b} 带现有 {n} 条,超上限 {BAND_CAPS[b]};请合并/删除/升 long 收敛到上限内"
                    for b, n in over.items()]

        if attempt < max_retries:
            logger.warning(f"consolidate 打回 attempt={attempt + 1} errs={errs[:3]}")
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _retry_msg(errs)},
            ]

    if last_valid is not None:                     # 仅超限收敛不了 → 引擎兜底淘汰最旧后出版
        enforce_caps(last_valid)
        logger.warning(f"consolidate 超限打回未收敛,引擎踢最旧兜底后出版(cells={len(cells)})")
        return last_valid
    logger.warning(f"consolidate 字段打回 {max_retries} 次仍不过,保留旧版(cells={len(cells)})")
    return None


# —— 触发退火 + 单 user 编排(runtime 接线用;拆出来便于单测)——

def anneal_step(version_count: int) -> int:
    """退火步长(按已出版本数):1,2,2,3,4,5…封顶 5。冷启动第 1 版尽早成形。"""
    ramp = (1, 2, 2, 3, 4, 5)
    return ramp[version_count] if version_count < len(ramp) else 5


def should_consolidate(*, n_new: int, ep_chars: int, version_count: int,
                       ep_chars_trigger: int) -> bool:
    """触发判定:新 cell 数达退火步长 或 episode 累计字数达上界,任一即触发。"""
    return n_new >= anneal_step(version_count) or ep_chars >= ep_chars_trigger


def run_user_consolidation(llm: ChatLLM, *, cells_store, atoms_store, profile_store,
                           today: date, scenario: str = "") -> int | None:
    """一 user 的整理编排:读游标后新 cell + atoms → consolidate → 出版本。返回新版本号或 None。

    幂等:读的是"上次出版本以来的新 cell";被单飞锁挡回/漏触发,下次触发自愈(读的仍是全部新 cell)。
    """
    cur = profile_store.current()
    cursor = cur.up_to_cell_id if cur else ""
    cells = cells_store.cells_after(cursor)
    if not cells:
        return None
    atoms_by_cell = {c.id: atoms_store.list_by_cell(c.id) for c in cells}
    _inp = "; ".join(c.topic for c in cells if c.topic)[:500] or f"{len(cells)} new cells"
    with obs.root_span("profile.consolidate", user_id=getattr(profile_store, "user_id", None),
                       session_id="profile", input=_inp, metadata={"n_cells": len(cells)}):
        merged = consolidate(llm, current=cur.profile if cur else None,
                             cells=cells, atoms_by_cell=atoms_by_cell, today=today,
                             scenario=scenario)
        if merged is None:
            return None
        version = profile_store.save_version(merged, up_to_cell_id=cells[-1].id)
        traits_detail = "\n".join(f"    {dom}: {t.text!r} ({t.status})"
                                  for dom, t in merged.traits.items() if t)
        facts_detail = "\n".join(
            f"    [{band}] " + " | ".join(f.text for f in merged.facts.get(band, []))
            for band in merged.facts)
        logger.info(f"画像出版 v{version} user={getattr(profile_store, 'user_id', '?')} "
                    f"cells={len(cells)}\n  ── traits ──\n{traits_detail}\n  ── facts ──\n{facts_detail}")
        return version
