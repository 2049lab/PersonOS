"""视频身份 → 记忆主管线的桥(行归属接线)。

会话末 commit 后调用:把 session 缓冲的剧本行(draft.all_lines)按 commit 交付的 {chain_ref→
character_id} 归属映射,翻译成带人物归属的 evidence,再一次性喂 build_cell 产 memcell/atoms。
让"Bob 说他喜欢爬山"落成 holder=Bob 的 atom,跨 session/模态收敛到同一人物。

设计(已与用户对齐):
- holder = 展示名(有名取名、无名取稳定短标 人物#N;SW→"user"、ENV→"env"),精确人物 id 走
  source.character_id(渲染层看到干净说话人名、不被裸 ULID 污染归属/指代;精确 id 持久在 payload)。
- clip 存为原始媒体证据(modality=video, content_ref=oss_key, 无 content_inline),派生行经
  source.raw_evidence_id 回指;raw_clip 不进 build_cell 的 records(无文本)。
- 每 session 一次 build_cell(全 clip 行时序合并 = 一段录像一个 episode)。
- 独立干净入口:将来 session_end 消费路径可直接调(不绑脚本)。

- 行文本里裸露的 id 一并改写成展示名(对齐 mneme _rewrite_label_mentions):clip 处理时已把
  local id(P1)换成 cast id(S1),这里再换成展示名 —— 否则同一个人在记忆里有三种叫法。
- 非 speech 行(action)加 `(action)` 前缀:transcript 渲染是统一的 "holder: text",不加标记
  的话动作描述会被 episode LLM 读成这个人**说**的话。

本轮不做:per-person 画像整理(profile_consolidate 现只产用户画像,独立新功能)。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from personos.identity.draft import DraftStore
from personos.identity.screenplay import ENV_WHO, WEARER_CAST_ID, rewrite_ids
from personos.identity.store import CharacterStore
from personos.models import EvidenceRecord
from personos.online.write_path import CellBuild, build_cell

# 第一人称视频场景提示,引导 episode/atom 抽取识别环境/动作/多人对话(喂 build_cell 的 scenario)。
VIDEO_SCENARIO = ("first-person wearable/robot video: the transcript below is a screenplay of one"
                  " recording — speaker lines are spoken words, plus action and environment"
                  " observations. 'user' is the camera wearer; other names are people in view."
                  " A speaker labelled 人物#N is a person whose name is not known yet; the first"
                  " lines of the transcript describe what each of them looks like.")

_ANON_PREFIX = "人物#"      # 无名人物的稳定短标前缀


class _DisplayResolver:
    """会话 cast → (holder 展示名, character_id)。无名人物给稳定短标 人物#N(本次 flush 内一致)。"""

    def __init__(self, draft: DraftStore, char_store: CharacterStore,
                 by_chain: dict[str, str], session_id: str) -> None:
        self.draft = draft
        self.store = char_store
        self.by_chain = by_chain
        self.session_id = session_id
        self.wearer = char_store.session_wearer(session_id) or char_store.wearer_character() or ""
        self._handle: dict[str, str] = {}   # cid → 人物#N(稳定)

    def resolve(self, who: str) -> tuple[str, str]:
        if who == ENV_WHO:
            return "env", ""
        if who == WEARER_CAST_ID:
            return "user", self.wearer
        ref = self.draft.chain_ref(self.session_id, who)
        cid = self.by_chain.get(ref) or self.by_chain.get(self.draft.canonical_chain(ref)) or ""
        if not cid:
            return who, ""                      # 兜底:无归属就用会话 cast 标签(不该发生)
        names = self.store.names_for(cid)
        if names:
            return names[0], cid
        ch = self.store.get_character(cid)
        primary = (ch or {}).get("primary_name")
        if primary:
            return primary, cid
        return self._handle.setdefault(cid, f"{_ANON_PREFIX}{len(self._handle) + 1}"), cid

    def describe(self, cast_id: str) -> str:
        """人物的外观描述:优先本会话链上的(最新),退回持久档的 appearance(老熟人)。"""
        ref = self.draft.canonical_chain(self.draft.chain_ref(self.session_id, cast_id))
        desc = ((self.draft.get_chain(ref) or {}).get("desc_text") or "").strip()
        if desc:
            return desc
        cid = self.by_chain.get(ref) or ""
        profile = ((self.store.get_character(cid) or {}).get("text_profile") or {}) if cid else {}
        return str(profile.get("appearance") or "").strip()

    def name_map(self, cast_ids) -> dict[str, str]:
        """cast id → 展示名,供改写行文本里裸露的 id。

        注意必须一次性建好整张表再改写:resolve 对无名人物是**按调用顺序**发 人物#N 的,
        边改写边 resolve 会让同一个人在不同行拿到不同编号。
        """
        return {c: self.resolve(c)[0] for c in cast_ids if c}


def flush_session_to_memory(
    draft: DraftStore, by_chain: dict[str, str], *, session_id: str,
    char_store: CharacterStore, evidence_store: Any, cell_store: Any, atom_store: Any,
    chain_store: Any, llm: Any, embedder: Any, media_store: Any = None,
    clip_keys: Optional[dict[int, str]] = None, scenario: str = VIDEO_SCENARIO,
) -> Optional[CellBuild]:
    """把 session 剧本行 flush 成带人物归属的 evidence + 一个 memcell。无行则返回 None。"""
    lines = draft.all_lines(session_id)
    if not lines:
        logger.info(f"视频记忆 flush:session={session_id} 无剧本行,跳过")
        return None
    clip_keys = clip_keys or {}
    resolver = _DisplayResolver(draft, char_store, by_chain, session_id)

    # ① 原始媒体证据:每个有 key 的 clip 一条(modality=video,无 content_inline,不进 records)
    raw_id: dict[int, str] = {}
    for ci in sorted({r["clip"] for r in lines}):
        key = clip_keys.get(ci)
        if not key:
            continue
        rec = EvidenceRecord(modality="video", content_ref=key, sha256=f"clip:{session_id}:{ci}",
                             source={"session_id": session_id, "clip_index": ci, "kind": "raw_clip"})
        raw_id[ci] = evidence_store.append(rec)

    # ② 派生行证据:holder=展示名,精确人物 id 走 source.character_id;时序收集喂 build_cell
    # 先把全量 cast→展示名表建好(含 roster 里出场但没说过话的人:他们的 id 仍可能出现在别人
    # 的动作行文本里),再逐行改写——不能边走边 resolve,否则无名人物的 人物#N 编号会漂。
    # 顺序 = 首次出场顺序,不是 set 迭代序:人物#N 的编号必须可复现(否则同一段录像重跑,
    # 同一个人会时而 人物#1 时而 人物#2)。没说过话的 roster 成员排在后面,按 id 排序兜底。
    seen_order = list(dict.fromkeys(ln["who"] for ln in lines))
    # 人物全集不能只取"说过话的 + roster":一个人可能整段没开口(佩戴者尤其常见),
    # 却被别人的动作行提到("... while SW films")。commit 交付的 by_chain 才是权威全集。
    others = ({r.split(":")[-1] for r in by_chain} | set(draft.load_roster(session_id))
              | {WEARER_CAST_ID, ENV_WHO}) - set(seen_order)
    all_casts = seen_order + sorted(others)
    names = resolver.name_map(all_casts)

    # ②.5 无名人物的说明行:每人**只列一次**,排在对话之前。
    # 不然记忆里「人物#1」就是个空壳编号——episode/atom 读到它完全不知道是谁,
    # 既没法判断跨会话是不是同一人,作答时也只能干巴巴复述这个编号。
    # 有名字的人不用列:名字本身就是身份。挂 holder=env(这是旁白式说明,不是谁说的话)。
    intro: list[EvidenceRecord] = []
    introduced: set[str] = set()
    for cast_id in all_casts:
        holder, cid = resolver.resolve(cast_id)
        if not holder.startswith(_ANON_PREFIX) or cid in introduced:
            continue      # 两条 cast 可能在终审后并到同一人,按 character_id 去重
        desc = rewrite_ids(resolver.describe(cast_id), names)
        if not desc:
            continue
        introduced.add(cid)
        intro.append(EvidenceRecord(
            holder="env", content_inline=f"{holder} is {desc}", modality="video",
            source={"session_id": session_id, "kind": "cast_intro",
                    "cast_id": cast_id, "character_id": cid}))
    for rec in intro:
        evidence_store.append(rec)

    records: list[EvidenceRecord] = list(intro)
    for ln in lines:
        text = rewrite_ids((ln.get("text") or "").strip(), names)
        if not text:
            continue
        holder, cid = resolver.resolve(ln["who"])
        if (ln.get("kind") or "") == "action":
            text = f"(action) {text}"      # 否则动作描述会被读成这个人说的话
        ci = ln["clip"]
        rec = EvidenceRecord(
            holder=holder, content_inline=text, modality="video",
            content_ref=clip_keys.get(ci),
            source={"session_id": session_id, "clip_index": ci, "t0": ln["t0"], "t1": ln["t1"],
                    "kind": ln.get("kind"), "cast_id": ln["who"], "character_id": cid,
                    "raw_evidence_id": raw_id.get(ci)})
        evidence_store.append(rec)
        records.append(rec)
    if len(records) == len(intro):     # 只有说明行、一句对话/动作都没有 → 没内容可记
        logger.info(f"视频记忆 flush:session={session_id} 无可渲染行,跳过 build_cell")
        return None

    # ③ 一次 build_cell:全 session 行 = 一段录像一个 episode
    logger.info(f"视频记忆 flush:session={session_id} lines={len(records) - len(intro)} "
                f"intro={len(intro)} raw_clips={len(raw_id)} → build_cell")
    return build_cell(llm, embedder, evidence_store, cell_store, atom_store, records,
                      session_id=session_id, chain_store=chain_store, scenario=scenario)
