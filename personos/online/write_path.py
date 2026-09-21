"""写入链路(融合架构 §2):W0 逐句落证据 → W1 边界检测 → W2 cell 生成。

写入侧总纲:只做"切段、织 episode、拆 atom",不做任何跨 cell 消解——
重复允许存在(冗余索引),冲突是作答时的事(D1)。
评测批量写入与产品在线走同一循环,前者对全对话快进执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

import numpy as np
from loguru import logger

from ..models import (
    AXIS_MENU, ATOM_TEXT_SPEC, DOMAIN_VOCAB, EvidenceRecord, EvidenceRef,
    MemCell, MemoryAtom, ensure_aware, normalize_domains, normalize_kind, now, vocab_menu,
)
from ..storage.atom_store import AtomStore
from .events import EVENT_ADD, emit
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.evidence_store import EvidenceStore
from ..storage.seg_store import MemorySegStore, SegStore
from . import chain_build
from .llm import ChatLLM, chat_json, with_scenario

# 业务方场景注入 directive(只调关注度/详略,不改事实、不编造、不漏)——见 with_scenario
_SCEN_DIR_BOUNDARY = ("Use it as background for the caller's typical dialogue rhythm and what counts "
                      "as one coherent topic; it informs the judgment below but never overrides the "
                      "priority rules.")
_SCEN_DIR_EPISODE = ("Narrate the facts this caller cares about more fully and faithfully. This sets "
                     "emphasis (level of detail) only — still cover the whole segment, never drop a "
                     "stated fact.")
_SCEN_DIR_ATOM = ("Facts matching this focus are high-value retrieval anchors: make sure each DISTINCT "
                  "one is covered by an atom. This only raises priority within what is worth "
                  "remembering; it never licenses inventing facts absent from the transcript, dropping "
                  "other worth-remembering facts, nor logging restatements as separate atoms.")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


# 段长安全阀:W1 判歪时兜底,超此轮数强制闭合(融合架构 W1 维度 5)
MAX_SEGMENT_TURNS = 30
# W1 prompt 里"当前段最近几轮"窗口:防长段把边界检测 prompt 撑爆
_BOUNDARY_WINDOW = 6
# W2② quote 匹配不中时的处理:refs 留空(该 atom 仍可被检索,溯源走 cell→evidence)
_D_MENU = "- domains (D axis, 0-3 Dxx codes; never output the labels):\n  " + vocab_menu(DOMAIN_VOCAB)


# —— W0 · 逐句落证据(零 LLM,被动;不再对证据 embed——全库只有 atom/topic 两种向量)——

def append_utterance(
    evidence_store: EvidenceStore, *, session_id: str, speaker: str, text: str,
    now_dt: Optional[datetime] = None,
    modality: str = "text", content_ref: Optional[str] = None, sha256: str = "",
) -> str:
    rec = EvidenceRecord(holder=speaker, content_inline=text,
                         modality=modality, content_ref=content_ref, sha256=sha256,
                         source={"session_id": session_id}, captured_at=now_dt or now())
    evidence_id = evidence_store.append(rec)
    logger.debug(f"W0 证据落库 ev={evidence_id} speaker={speaker} modality={modality} len={len(text)}")
    return evidence_id


def _transcript(records: list[EvidenceRecord]) -> str:
    """段内原话渲染:带说话人与时间戳(供 W1/W2 的 prompt 与 LLM 写双时间格式)。"""
    lines = []
    for r in records:
        ts = r.captured_at.strftime("%Y-%m-%d %H:%M") if r.captured_at else "?"
        lines.append(f"[{ts}] {r.holder}: {r.content_inline or ''}")
    return "\n".join(lines)


# —— W1 · 边界检测(逐句 1 次小 LLM)——

@dataclass
class BoundaryDecision:
    should_end: bool
    confidence: float = 0.0
    topic_summary: str = ""    # 当前段一句话主题,供 W2① 的 topic 参考
    raw: str = ""              # 溯源


_BOUNDARY_SYSTEM = """# Role
You are the dialogue boundary detector in a memory write pipeline: decide whether a newly arrived
BATCH of utterances continues the current topic segment or opens a new one. The batch is atomic —
its utterances always stay together in the same segment (a question and its answer arriving in one
batch must never be split).

# Decision dimensions (by priority)
1. Substantive topic shift (highest priority): the new batch's core topic clearly differs from the
   current segment → end the segment.
2. Intent switch: the segment's core question is resolved and the new batch opens a brand-new
   task/intent → end.
3. Small talk / farewells are NOT shifts: closing phrases like "Thanks!", "Talk soon!" stay in the segment.
4. Time gap: hours or days since the previous utterance → end; only a few minutes → lean continue.
5. Segment length 3-20 turns: a long segment showing signs of drifting → lean end; under 3 turns,
   continue unless the topic clearly switched.

# Output (JSON only, no explanation)
{"should_end": true|false, "confidence": 0-to-1 decimal, "topic_summary": "one-sentence topic of the CURRENT segment"}
topic_summary states what the CURRENT segment is about, as a reference for downstream cell-topic
generation: one sentence making clear WHO is doing WHAT; keep names/brands/places/numbers/times
explicit and detailed — no pronouns, no generalization. Write it in the language of the dialogue."""


def detect_boundary(
    llm: ChatLLM, seg: list[EvidenceRecord], new_records: list[EvidenceRecord],
    *, gap_minutes: Optional[float] = None, scenario: str = "",
) -> BoundaryDecision:
    """判"新到的整批是否开启新段"。seg 为空时不调 LLM(新段首批无界可判)。

    new_records 是调用方声明的一个原子批(如一对 QA):批内永远不切,
    判的是整批 vs 当前段的去留。解析失败/异常 → 保守延续(should_end=False):
    错闭合伤 episode 连贯,延续只伤段长,且有安全阀兜底。
    scenario:业务方场景描述(可空)——注入切段校准(对话形态/何为一个话题)。
    """
    if not seg:
        return BoundaryDecision(should_end=False)
    gap = "unknown" if gap_minutes is None else f"{gap_minutes:.0f} minutes"
    user = (
        f"Current segment turns: {len(seg)}\nTime gap since the previous utterance: {gap}\n\n"
        f"—— Current segment (most recent turns) ——\n{_transcript(seg[-_BOUNDARY_WINDOW:])}\n\n"
        f"—— Newly arrived batch ({len(new_records)} utterances) ——\n{_transcript(new_records)}"
    )
    sys = with_scenario(_BOUNDARY_SYSTEM, "# Decision dimensions (by priority)",
                        scenario, _SCEN_DIR_BOUNDARY)
    try:
        data, raw = chat_json(llm, [{"role": "system", "content": sys},
                                    {"role": "user", "content": user}], stage="boundary_detect",
                              max_tokens=200)
        d = BoundaryDecision(should_end=bool(data.get("should_end")),
                             confidence=float(data.get("confidence") or 0.0),
                             topic_summary=str(data.get("topic_summary") or ""), raw=raw)
        logger.info(f"W1 边界 end={d.should_end} conf={d.confidence:.2f} seg_len={len(seg)} "
                    f"gap={gap} summary={d.topic_summary!r}")
        return d
    except Exception as e:   # noqa: BLE001  解析失败/网络抖动一律保守延续(错闭合伤 episode,延续有安全阀兜底)
        logger.warning(f"W1 边界检测异常,保守延续: {e}")
        return BoundaryDecision(should_end=False, raw=str(e))


# —— W2 · cell 生成(每段 2 次 LLM,顺序)——

@dataclass
class CellBuild:
    """一个 cell 的生成产物(含溯源)。"""
    cell: MemCell
    atoms: list[MemoryAtom]
    gen: dict = field(default_factory=dict)   # {call1: {system,user,raw}, call2: {...}} 工作台透视用
    chain_assign: "chain_build.ChainAssignResult | None" = None   # W2.5 判链产物(关闭/失败=None 或全游离)


_EPISODE_SYSTEM = """# Role
You are the episode weaver in a memory write pipeline: weave one topic segment of dialogue into a
third-person narrative, plus a one-sentence topic and life-domain codes.

# Task (on the FULL segment transcript provided)
1. topic: one sentence stating what this segment is about — WHO is doing WHAT + key proper nouns kept
   verbatim (names/brands/places/numbers), no pronouns, no generalization (e.g. "Caroline preparing a
   joint exhibition with Rob"). The topic is this segment's retrieval face and material header — word
   it so a later search can hit it.
2. episode: a third-person narrative covering the whole segment in dialogue order — this is the primary
   material for all future answering.
3. domains: life-domain codes for this segment (0-3 D-axis codes).

# Hard constraints on episode (detail preservation carries most of the retrieval & answering score)
- Coverage first: before writing, walk the transcript utterance by utterance — every fact-bearing
  line (a date, time, amount, count, name, address, or one-off mention) must appear in the
  episode. Never trade a "small" fact (a departure day, a ticket price, a one-time plan) for
  narrative flow: write "departed on July 11", never "departed in mid-July".
- Full names, never pronouns: "Caroline and Rob", not "she and her friend".
- Proper nouns verbatim: brands, place names, book titles, restaurant names enter the text as-is.
- Numbers exact: "3 pizzas" not "some pizzas"; frequency specific: "every Tue & Thu" not "often".
- Dates/times of departures, appointments and deadlines, amounts, and identifiers enter the episode
  verbatim — a bare date is often the entire answer to a later question.
- Specific activities stay specific: "hot yoga" not "exercise".
- Dual time format: relative wording + absolute date written together — "last week (2026-08-18)".
- Complete when/who/where/how bounds: state clearly who did what and when; every qualifier present in
  the source (where / with whom / how / how often / until when) enters the narrative — never drop a
  stated qualifier, never invent an unstated dimension.
- Attribution explicit: make clear who said/did what; with multiple speakers, keep each person's
  matters distinct.
- Language follows the source dialogue: write topic and episode in the SAME language as the dialogue.
- Only what was said in this segment: no inference, no addition; every narrative sentence must trace
  back to the transcript.

""" + _D_MENU + """

# Output (JSON only)
{"topic": "...", "episode": "...", "domains": ["D13"]}"""


_ATOM_SYSTEM = """# Role
You are the atomic-memory extractor in a memory write pipeline: batch-extract atomic facts (atoms)
from one dialogue segment.

# Input
An episode (narrative — navigate by it) plus the FULL segment transcript (the ground truth — decide by
it): speaker attribution, times and numbers are always settled from the transcript.

# What to extract
- Worth remembering: stable identity & attributes / long-term preferences & habits / relationships /
  goals, plans, commitments & schedules (clear intent or a time attached — including near-term
  bookings, useful until they happen) / constraints & rules / important events & experiences / stable
  facts about the people and things that matter to someone / concrete items of interest (books read,
  places visited, activities enjoyed, family members' likes — record even passing mentions).
- Not worth remembering: passing whims & momentary desires / small talk / immediate instructions about
  the current task / transition chatter.
- Read question-and-answer exchanges as a whole: "guess where I live" + "Shanghai?" + "yes" → "lives
  in Shanghai" holds. The assistant's words are context only — its suggestions/guesses count as fact
  only when the user explicitly endorses them.
- Multiple speakers: attribute every statement to its owner — holder is the speaker the proposition
  TRULY belongs to (the name used in the dialogue; the person talking to the assistant is "user").
  Never attribute someone else's experience to user.
- A question or follow-up is not itself a fact; an unanswered cliffhanger is not extracted.

# Coverage as retrieval anchors (atoms ARE the retrieval face — the episode already carries the full
# narrative for answering; atoms exist so a later query can FIND this segment, NOT to re-log every line)
- One atom per DISTINCT fact: give each distinct worth-remembering thing exactly ONE atom. Together the
  atoms are MECE — no two restate the same fact, and collectively they cover every distinct
  worth-remembering point in the segment. Do NOT chase completeness by adding near-duplicate atoms.
- Merge repetition; never one atom per utterance: the same fact stated, rephrased, reacted to or argued
  over across many lines is still ONE atom. Blow-by-blow narration — every gesture, each repeated
  complaint, the back-and-forth bickering, momentary reactions — belongs in the EPISODE, not in atoms.
- **Always return at least one atom — an empty list is never a valid answer.** How many is decided by
  the segment itself: one atom per DISTINCT worth-remembering fact. When the segment genuinely carries
  no durable fact (pure greetings, acknowledgements), return exactly ONE atom that states in one
  sentence what happened in this segment. An argument is still a source of facts — what got damaged,
  who owns what, stated numbers, majors, habits — extract those even when the surrounding lines are
  repetitive bickering. Reason: atoms are the ONLY retrieval entry to this segment; return none and
  the whole segment becomes permanently unreachable, however rich its episode is.
- Skip process trivia: small talk, filler, transient reactions, and repeated micro-actions get NO atom;
  they are neither searched for nor worth remembering.
- Distinct items & attributes DO each get their own atom — they are separate search anchors, NOT
  repetition. A list of things (bought / visited / people present) → one atom each. Parallel attributes
  of one subject that are each asked about on their own — schedule, venue, coach, price, a stated
  frequency/count ("every Tue & Thu", "the third time") — → separate atoms; a bare name / address /
  number is often exactly what gets looked up later. This is coverage of distinct facts, the opposite
  of logging restatements. (A qualifier bounding one occurrence — its when/where — stays inside its atom.)

# Fields per atom
- text: follow the spec below (third-person single sentence, self-contained subject-verb-object, dual
  time format, proper nouns & numbers verbatim).
- object_type (epistemic state; shapes future answer wording): fact = normalized persistent state
  ("Caroline lives in Shanghai"); claim = opinion/intention/plan/hearsay (always attributed wording,
  e.g. "Caroline said she plans to hold the exhibition next month"); event = a one-time occurrence
  with a definite moment ("Caroline sprained her right ankle last week (2026-08-18)").
- holder: who said it / whose attribute (the name from the dialogue, or "user").
- kind: one K-axis code.
- domains: 0-3 D-axis codes.
- when: the date the fact itself happened or holds (ISO yyyy-MM-dd, e.g. "2026-08-18") — the day it
  HAPPENED, not the day it was talked about. Resolve relative words (yesterday / last Wednesday /
  next month) into absolute dates against the transcript timestamps. Anchor to the time the fact
  refers to, NOT the dialogue day: "made a limited edition line last week" said on 2023-08-23 → when
  is a date within 08-16 ~ 08-22 (the week BEFORE the dialogue), never 08-23. For persistent states
  with no intrinsic date, give null (code falls back to the dialogue date, meaning "already true as
  of that day").
- quote: the atom's single most relevant source sentence, a CONTIGUOUS SUBSTRING copied verbatim from
  the transcript (used only to link back to the evidence — an anchor, NOT required to cover the whole
  atom text; when the atom merges several lines, pick the one line that best evidences it). Never
  rewrite it; if no single line fits, leave it empty.

""" + ATOM_TEXT_SPEC + "\n\n" + AXIS_MENU + """

# Output (JSON only; output {"atoms":[]} when nothing is worth remembering)
{"atoms":[{"text":"...","object_type":"fact","holder":"user","kind":"K01","domains":["D01"],"when":"2026-08-18","quote":"..."}]}"""


def _match_evidence_refs(records: list[EvidenceRecord], quote: str) -> list[EvidenceRef]:
    """quote 子串 → 命中的证据(逐字包含即算;多句命中取全部,时序)。

    未命中 → refs 留空并告警:atom 仍可检索,但金标证据链断在"回链"这一环,
    评测统计断点时要能从日志里数出来。
    """
    q = (quote or "").strip()
    if not q:
        return []
    refs = [EvidenceRef(evidence_id=r.id)
            for r in records if q in (r.content_inline or "")]
    if not refs:
        logger.warning(f"W2② quote 未命中原话,refs 留空(证据链断点)quote={q[:40]!r}")
    return refs[:3]   # 一条 atom 的出处顶多几句话,防 LLM 给整段原文


# episode 分类:仅当调用方传了 task_type 词表才追加(否则 LLM 根本不知道有这字段,默认 unknown)。
# 挂在 W2① 那次调用上,零额外 LLM 开销。选不中列表 → 解析侧回落 unknown。
_CLASSIFY_SUFFIX = """

# Episode classification (REQUIRED extra field)
Also classify this whole segment into EXACTLY ONE type from this list: {types}
Add an "episode_type" field to your JSON with the chosen value verbatim; if none fits, use "unknown".
Output JSON: {{"topic": "...", "episode": "...", "domains": [...], "episode_type": "..."}}"""


def build_cell(
    llm: ChatLLM, embedder: Embedder,
    evidence_store: EvidenceStore, cell_store: CellStore, atom_store: AtomStore,
    records: list[EvidenceRecord], *, session_id: str, topic_hint: str = "",
    chain_store: ChainStore | None = None, task_type: list[str] | None = None,
    scenario: str = "",
) -> CellBuild:
    """W2:一段闭合的原话 → MemCell(topic/episode/域)+ atoms,批量 embed 后落库。

    Call① 失败 → 降级用原话当 episode(信息不丢,格式受损);Call② 失败 → cell 落库、
    atoms 留空(漏抽由深轨 remember / 后续 dream 补抽自愈,见 D3/D8)。
    atoms 落库后接 W2.5 判链(chain_store 未注入则从
    atom_store 派生);判链失败非阻塞——本格 atom 留游离,绝不动 cell/atoms 已落库结果。
    """
    transcript = _transcript(records)
    gen: dict = {}

    # Call① topic + episode + cell 域(时间戳缺失按 "?" 渲染,不让格式化把 W2 整个炸穿)
    t0s = (records[0].captured_at.strftime("%Y-%m-%d %H:%M") if records[0].captured_at else "?")
    t1s = (records[-1].captured_at.strftime("%Y-%m-%d %H:%M") if records[-1].captured_at else "?")
    user1 = (f"Session window: {t0s} ~ {t1s}\n"
             + (f"(session topic: {topic_hint})\n" if topic_hint else "")
             + f"\n—— Full segment transcript ——\n{transcript}")
    types = [t for t in (task_type or []) if isinstance(t, str) and t.strip()]
    sys1 = with_scenario(_EPISODE_SYSTEM, "# Task (on the FULL segment transcript provided)",
                         scenario, _SCEN_DIR_EPISODE)
    sys1 += (_CLASSIFY_SUFFIX.format(types=types) if types else "")
    topic, episode, cell_domains, episode_type = "", "", [], "unknown"
    try:
        data, raw = chat_json(llm, [{"role": "system", "content": sys1},
                                    {"role": "user", "content": user1}], max_tokens=3000,
                              stage="episode_weave")
        topic = str(data.get("topic") or "").strip()
        episode = str(data.get("episode") or "").strip()
        cell_domains = normalize_domains(data.get("domains"))[:3]
        if types:   # 传了词表才认:LLM 输出须命中列表,否则回落 unknown(选不中不硬套)
            et = str(data.get("episode_type") or "").strip()
            episode_type = et if et in set(types) else "unknown"
        gen["call1"] = {"system": sys1, "user": user1, "raw": raw}
    except Exception as e:   # noqa: BLE001  解析失败/网络抖动降级为原话 episode(信息不丢,格式受损)
        logger.warning(f"W2① 失败,降级为原话 episode: {e}")
        topic = topic_hint or (records[0].content_inline or "")[:50]
        episode = transcript
        gen["call1"] = {"system": sys1, "user": user1, "raw": str(e)}

    cell = MemCell(session_id=session_id, topic=topic, episode=episode, domains=cell_domains,
                   episode_type=episode_type,
                   t_start=records[0].captured_at, t_end=records[-1].captured_at,
                   evidence_refs=[EvidenceRef(evidence_id=r.id) for r in records])

    # Call② atom 批量提取(episode + 原话双喂)
    atoms: list[MemoryAtom] = []
    user2 = (f"—— Episode (episodic memory compressed from the original dialogue) ——\n{episode}\n\n"
             f"—— Full original transcript ——\n{transcript}")
    sys2 = with_scenario(_ATOM_SYSTEM, "# Input", scenario, _SCEN_DIR_ATOM)
    seg_date = records[0].captured_at

    def _extract(nudge: str = "") -> tuple[list[MemoryAtom], str]:
        msgs = [{"role": "system", "content": sys2}, {"role": "user", "content": user2}]
        if nudge:   # 重抽时追加纠正语:原样重发等于再摇一次同样偏向空的骰子
            msgs += [{"role": "assistant", "content": '{"atoms":[]}'},
                     {"role": "user", "content": nudge}]
        data, raw_out = chat_json(llm, msgs, max_tokens=4000, num_tries=3, stage="atom_extract")
        out: list[MemoryAtom] = []
        for item in (data.get("atoms") or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            out.append(MemoryAtom(
                memcell_id=cell.id,
                object_type=item.get("object_type") if item.get("object_type") in ("fact", "claim", "event") else "claim",
                text=text,
                holder=str(item.get("holder") or "user"),
                kind=normalize_kind(item.get("kind")) or "K01",
                domains=normalize_domains(item.get("domains"))[:3],
                occurrence_time=ensure_aware(item.get("when")) or seg_date,
                evidence_refs=_match_evidence_refs(records, str(item.get("quote") or "")),
            ))
        return out, raw_out

    try:
        atoms, raw = _extract()
        # 0 atom 再抽一次:atom 是**检索触角**,一条都没有 = 这个 cell 任何查询都召不回,
        # 成了孤岛(episode 还在,但找不到它)。而 chat_json 的 num_tries 只管 JSON 解析失败,
        # 合法的 {"atoms":[]} 会被直接放行。实测同一份输入在不同进程里给过 0 也给过 2,
        # 说明空结果有相当概率是漏抽而非"确实无可记"。第二次仍空则认定确实无可记。
        if not atoms:
            logger.warning(f"W2② 抽取 0 atoms,带纠正语重抽 turns={len(records)} topic={topic[:30]!r}")
            atoms, raw = _extract(
                "你上次返回了空列表。请重新通读这一段:哪怕整段以争执/闲聊为主,其中提到的"
                "**持久事实**(谁的什么东西、明确的数字、身份/专业/习惯、发生过的具体事件)仍然要抽出来。"
                "只有当这一段确实通篇只有寒暄与应答、没有任何值得记住的事实时,才允许返回空列表。"
                "只输出 JSON 本体。")
        gen["call2"] = {"system": sys2, "user": user2, "raw": raw}
    except Exception as e:   # noqa: BLE001  解析失败/网络抖动 → cell 落库、atoms 留空(补抽自愈,见 D3/D8)
        logger.warning(f"W2② 失败,atoms 留空: {e}")
        gen["call2"] = {"system": sys2, "user": user2, "raw": str(e)}

    # —— 兜底:有 memcell 就必须有至少一条检索触角 ——
    # 快链 R1(retrieval.search_atoms)只检索 **atom** 向量;cell 的 topic_embedding 只在深轨
    # 对已选中的 cell 子集用。所以 0 atom 的 cell 在快链里**彻底不可达** —— episode 还在库里,
    # 但任何问法都召不到它(实测 4 种问法全 miss)。
    # 信息量再低的一段也该留一句概括当触角,而不是整段消失。topic 本身就是"谁在做什么"的
    # 一句话概括,直接拿来当这条 atom 的正文;evidence_refs 挂全段,溯源不断。
    # 标 source="w2_fallback" 便于事后统计这条路径被触发的频率(正常路径恒为 "w2")。
    if not atoms:
        summary = (topic or "").strip() or (episode or "").strip()[:120]
        if summary:
            atoms = [MemoryAtom(
                memcell_id=cell.id, object_type="event", text=summary,
                holder="user",          # 段概括不属于某一个人,用默认 holder
                kind="K04",             # event/experience:这一段发生过的事
                domains=cell_domains, occurrence_time=seg_date, source="w2_fallback",
                evidence_refs=[EvidenceRef(evidence_id=r.id) for r in records])]
            logger.warning(f"W2② 重抽后仍 0 atoms → 合成 1 条概括 atom 兜底(否则该 cell 永久"
                           f"不可召回)turns={len(records)} text={summary[:60]!r}")
        else:
            logger.error(f"W2② 0 atoms 且 topic/episode 皆空,该 cell 将不可召回 cell={cell.id}")

    # 批量 embed:topic 一个 + atom 逐条(全库仅有的两种向量),一次调用;topic 空不占位
    texts = ([topic] if topic else []) + [a.text for a in atoms]
    vecs = list(embedder.embed(texts)) if texts else []
    topic_vec = np.asarray(vecs[0]) if topic and vecs else None
    atom_vecs = vecs[1:] if topic else vecs

    cell_store.upsert(cell, topic_embedding=topic_vec)
    if atoms:
        atom_store.upsert_many(list(zip(atoms, atom_vecs)))
    logger.info(f"W2 cell 落库 id={cell.id} turns={len(records)} atoms={len(atoms)} "
                f"episode_type={episode_type} domains={cell_domains}\n"
                f"  topic={topic!r}\n"
                f"  episode={episode!r}\n"
                f"  atoms(n={len(atoms)})=" + "\n".join(
                    f"    [{i}] {a.object_type}/{a.kind} {a.text!r} holder={a.holder} "
                    f"domains={a.domains} occ={a.occurrence_time}" for i, a in enumerate(atoms, 1)))
    # 广播「这些记忆刚形成」。记忆服务不关心谁在听——投递、签名、订阅关系
    # 都是业务层的事。无订阅者时零开销,回调失败也不影响本次写入。
    if atoms:
        emit(
            atom_store.user_id,
            EVENT_ADD,
            {
                "cell_id": cell.id,
                "session_id": cell.session_id,
                "atom_ids": [a.id for a in atoms],
                "count": len(atoms),
                # 只带 id 不带正文:正文属于记忆本身,要内容让订阅方回调 API 取
                "domains": sorted({d for a in atoms for d in (a.domains or [])}),
            },
        )

    # W2.5 判链(有 atom 才走;失败在 assign_chains 内部消化,绝不阻塞)
    chain_assign = None
    if atoms:
        cs = chain_store or ChainStore(atom_store.db, atom_store.user_id)
        try:
            chain_assign = chain_build.assign_chains(
                llm, cs, list(zip(atoms, atom_vecs)), origin_cell_id=cell.id)
            gen["w25"] = {"system": chain_build._ASSIGN_SYSTEM, "result": {
                "new": [c.id for c in chain_assign.new_chains],
                "appended": chain_assign.appended, "free": len(chain_assign.free)}}
        except Exception as e:   # noqa: BLE001  双保险:assign_chains 已自捕获,这里兜未知路径
            logger.warning(f"W2.5 判链异常,本格 atom 留游离: {e}")
    return CellBuild(cell=cell, atoms=atoms, gen=gen, chain_assign=chain_assign)


# —— 编排:SessionWriter(产品在线逐句 / 评测批量快进,同一循环)——

@dataclass
class StepResult:
    """feed 一句话后的全部中间态(场景/工作台透视用)。"""
    evidence_id: str
    record: EvidenceRecord
    boundary: Optional[BoundaryDecision] = None   # None = 段首句(无界可判)或 30 轮强制闭合
    forced_close: bool = False                    # True = 安全阀闭合,非 LLM 判定
    closed_cell: Optional[CellBuild] = None       # 本句触发闭合的 cell(新句归新段)


@dataclass
class FeedMsg:
    """feed_batch 的一个原子成员(一条消息:说话人 + 文本/图片)。"""
    speaker: str
    text: str = ""
    image: Optional[bytes] = None
    image_content_type: str = "image/jpeg"


@dataclass
class BatchStepResult:
    """feed_batch 一批后的中间态:整批原子——要么整批并入旧段,要么整批开启新段。"""
    evidence_ids: list[str]
    records: list[EvidenceRecord]
    boundary: Optional[BoundaryDecision] = None   # None = 段首批(无界可判)或安全阀强制闭合
    forced_close: bool = False                    # True = 安全阀闭合,非 LLM 判定
    closed_cell: Optional[CellBuild] = None       # 本批触发闭合的 cell(闭合的是旧段,不含本批)
    # Things that partially succeeded. A write is never rejected for a missing
    # optional capability, but it must not silently do less than asked either —
    # an image stored without understanding contributes nothing to retrieval,
    # and the caller deserves to know that rather than discover it at recall.
    warnings: list[str] = field(default_factory=list)


class SessionWriter:
    """一段会话的写入状态机:逐句 feed,段闭合时 W2 建 cell;session 末强制闭合。

    无状态化:未闭合段存 seg_store(服务侧注入 Redis 实现 → 跨副本共享/重部署不丢;
    缺省进程内私有实例,单进程脚本语义不变)。本实例只保留 cells 累积——批量脚本
    的进度视图;服务侧每次请求新建 writer,列表随请求生灭,不再常驻。
    """

    def __init__(self, llm: ChatLLM, embedder: Embedder,
                 evidence_store: EvidenceStore, cell_store: CellStore, atom_store: AtomStore,
                 *, session_id: str, user_id: str = "", max_turns: int = MAX_SEGMENT_TURNS,
                 seg_store: SegStore | None = None, chain_store: ChainStore | None = None,
                 media_store=None, mllm=None):
        self.llm = llm
        self.embedder = embedder
        self.evidence_store = evidence_store
        self.cell_store = cell_store
        self.atom_store = atom_store
        self.chain_store = chain_store   # None=build_cell 里从 atom_store 派生(同库同 user)
        self.session_id = session_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.seg_store = seg_store or MemorySegStore()   # 未注入=私有进程内段状态(单进程语义)
        self.cells: list[CellBuild] = []                 # 本实例产出的 cell(批量脚本进度用)
        # 图片输入依赖(可选;未注入=不支持图片,纯文本链路不受影响)
        self.media_store = media_store
        self.mllm = mllm

    @property
    def seg(self) -> list[EvidenceRecord]:
        """当前未闭合段(从 seg_store 实时读,副本;兼容旧调用方断言)。"""
        return self.seg_store.load(self.user_id, self.session_id)

    def _ingest_image(self, image: bytes, content_type: str, speaker: str,
                      context: Optional[list[EvidenceRecord]] = None) -> tuple[Optional[str], str, str]:
        """存图 + 带上下文看图。返回 (content_ref, sha256, 图片理解文本);任一步失败降级。

        看图目的(purpose)= 当前段已有对话 + 本批已备好的前几句——让 MLLM 带着
        「这段在聊什么」看图,而非漫无目的描述。存储失败仍尝试用内存字节看图
        (原图没留底但理解不丢)。
        """
        content_ref, sha256 = None, ""
        if self.media_store is not None:
            try:
                stored = self.media_store.save_image(
                    image, owner=self.user_id or "user", content_type=content_type)
                content_ref, sha256 = stored.key, stored.sha256
            except Exception as e:   # noqa: BLE001  存储失败不阻塞:仍可看图,只是没留底
                logger.warning(f"图片存 OSS 失败,原图不留底(理解仍尝试): {e}")
        img_text = ""
        if self.mllm is not None and getattr(self.mllm, "available", False):
            ctx = _transcript((context or [])[-_BOUNDARY_WINDOW:]) if context else ""
            purpose = (f"本段对话正在聊:\n{ctx}\n\n带着这段上下文,说明图片里与之相关的事实。"
                       if ctx else "客观说明图片里可见的事实(人物/文字/场景/可数对象)。")
            img_text = self.mllm.look_image(image, purpose, content_type=content_type)
        return content_ref, sha256, img_text

    def _prepare(self, m: FeedMsg, *, now_dt: datetime,
                 source_extra: dict | None, context: list[EvidenceRecord]) -> EvidenceRecord:
        """W0:一条消息 → 证据落库 + 内存 rec。带图先走存图/看图(失败降级,不阻塞)。"""
        modality, content_ref, sha256 = "text", None, ""
        content_inline = m.text
        if m.image:
            # 有图即标 image/mixed(与存储是否成功无关:即便原图没留底,这仍是一条图片消息)
            modality = "mixed" if m.text.strip() else "image"
            content_ref, sha256, img_text = self._ingest_image(
                m.image, m.image_content_type, m.speaker, context)
            if img_text:
                # 图片理解文本并进原话:用户配文在前,看图事实在后(供 W1/W2 一并读)
                content_inline = (m.text + "\n" if m.text.strip() else "") + f"[图片] {img_text}"
        evidence_id = append_utterance(self.evidence_store, session_id=self.session_id,
                                       speaker=m.speaker, text=content_inline, now_dt=now_dt,
                                       modality=modality, content_ref=content_ref, sha256=sha256)
        return EvidenceRecord(id=evidence_id, holder=m.speaker, content_inline=content_inline,
                              modality=modality, content_ref=content_ref, sha256=sha256,
                              source={"session_id": self.session_id, **(source_extra or {})},
                              captured_at=now_dt)

    def feed_batch(self, msgs: list[FeedMsg], *,
                   now_dt: Optional[datetime] = None,
                   source_extra: dict | None = None,
                   task_type: list[str] | None = None,
                   scenario: str = "") -> BatchStepResult:
        """喂一整批(原子):全批 W0 落证据 → (段非空时)W1 判整批去留 → 闭合则 W2 → 入段。

        批的地位等于以前的一句:要么整批并入当前段,要么整批开启新段——批内(如一对
        QA)永远不被切开。带图消息逐条存图/看图,看图上下文含本批已备好的前几句。
        """
        now_dt = now_dt or now()
        recs: list[EvidenceRecord] = []
        # W0 阶段(持写锁)段不会变,读一次即可;看图上下文 = 当前段 + 本批已备好的前几句
        ctx_seg = self.seg_store.load(self.user_id, self.session_id)
        for m in msgs:
            recs.append(self._prepare(m, now_dt=now_dt, source_extra=source_extra,
                                      context=ctx_seg + recs))

        seg = ctx_seg   # W0 只写证据库不动段,直接复用
        boundary, forced, closed = None, False, None
        if seg:
            # 安全阀:段已超限,或并入本批后将超限 → 先闭合旧段再接批,不调 LLM
            if len(seg) >= self.max_turns or len(seg) + len(recs) > self.max_turns:
                forced = True
            else:
                gap = ((now_dt - seg[-1].captured_at).total_seconds() / 60
                       if now_dt and seg[-1].captured_at else None)
                boundary = detect_boundary(self.llm, seg, recs, gap_minutes=gap, scenario=scenario)
            if forced or boundary.should_end:
                closed = build_cell(self.llm, self.embedder, self.evidence_store,
                                    self.cell_store, self.atom_store, seg,
                                    session_id=self.session_id,
                                    topic_hint=boundary.topic_summary if boundary else "",
                                    chain_store=self.chain_store, task_type=task_type,
                                    scenario=scenario)
                self.cells.append(closed)
                seg = []                             # 闭合即清段;下方 save 整体覆写键
        self.seg_store.save(self.user_id, self.session_id, seg + recs)
        return BatchStepResult(evidence_ids=[r.id for r in recs], records=recs,
                               boundary=boundary, forced_close=forced, closed_cell=closed)

    def feed(self, speaker: str, text: str, *,
             now_dt: Optional[datetime] = None, source_extra: dict | None = None,
             image: bytes | None = None, image_content_type: str = "image/jpeg",
             task_type: list[str] | None = None, scenario: str = "") -> StepResult:
        """喂一句话(单句便利壳:等价 feed_batch 一条,内部脚本/评测沿用)。"""
        r = self.feed_batch([FeedMsg(speaker=speaker, text=text, image=image,
                                     image_content_type=image_content_type)],
                            now_dt=now_dt, source_extra=source_extra, task_type=task_type,
                            scenario=scenario)
        return StepResult(evidence_id=r.evidence_ids[0], record=r.records[0],
                          boundary=r.boundary, forced_close=r.forced_close,
                          closed_cell=r.closed_cell)

    def end_session(self, *, task_type: list[str] | None = None,
                    scenario: str = "") -> list[CellBuild]:
        """session 结束:未闭合段强制闭合(W1 分段覆盖整个会话的保证)。task_type 用于尾段 episode 分类。"""
        seg = self.seg_store.load(self.user_id, self.session_id)
        if seg:
            closed = build_cell(self.llm, self.embedder, self.evidence_store,
                                self.cell_store, self.atom_store, seg,
                                session_id=self.session_id, chain_store=self.chain_store,
                                task_type=task_type, scenario=scenario)
            self.cells.append(closed)
            self.seg_store.clear(self.user_id, self.session_id)
            logger.info(f"session 末强制闭合 session={self.session_id} cells={len(self.cells)}")
        return self.cells
