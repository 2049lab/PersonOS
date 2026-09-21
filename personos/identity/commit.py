"""commit_session(⑦):会话末两阶段终审 + 结算落 MySQL。

终审 = 批量模型调用(材料 = 会话最佳脸/全身 + desc + names + 声纹;候选 = 库(≤small_library_max
全量,否则每链粗召回)+ pending 链互为候选)。verdict 先一致化(链-链并;同框撞同档→保证据优选、其余降
NEW),再按 canonical 链结算:NEW 此刻才建档,素材入库 / 名字归并 / 学云一趟做完。
佩戴者走同一终审(SW 链,声纹是它唯一生物特征):结算档成为本会话佩戴者指针 + 标 is_wearer。
终审调用失败:每链回退到最近一次成功评估(链假设);崩在提交前的链留 pending,finalize 重跑兜底。

有意偏离(对齐 personos):持久层 MySQL CharacterStore;素材从 OSS 取 b64;结算原子性——测试下
rollback_scope pin 连接 → SAVEPOINT 真原子(store 写都走 pinned 连接),prod 下 CharacterStore
逐语句 autocommit(Database facade 规范),靠 pending 状态 + 重跑幂等兜底(draft.commit_chain 在该链
MySQL 写成功后才标 committed);personos 无 anchor_line 表,故不移植 line 改写,只交付归属映射。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from personos.identity.chains import ChainBook
from personos.identity.cloud import CloudEngine
from personos.identity.draft import read_b64
from personos.identity.recognize import (
    build_arbitration_prompt,
    parse_verdicts_detailed,
)
from personos.identity.registry import AnchorRegistry
from personos.identity.screenplay import WEARER_CAST_ID
from personos.identity.store import CharacterStore
from personos.identity.types import CandidateCard

# 终审不属于任何 clip;评估台账用 -1 作标记。
FINAL_CLIP_INDEX = -1
# 单批过大退化注意力(观测到批量自绑);分批保住注意力。
FINAL_REVIEW_BATCH_SIZE = 12

# 两条通用规则(chain id / 同框)已在 arbiter.PROMPT_HEADER;终审尾注加:verdict 即终判 +
# 自绑无效 + note 是强先验。
_FINAL_CONSTRAINT = (
    "\n\nFINAL REVIEW NOTES:\n"
    "- This is the LAST review before permanent registration: verdicts here are final"
    " for this recording. Re-examine each query against the evidence rules above;"
    " when not reasonably sure, answer NEW.\n"
    "- A query's OWN chain id is never a valid target: answering it is treated as"
    " abstention and discarded. Choose a REGISTERED character, a DIFFERENT chain"
    " id, or NEW.\n"
    "- When a query carries a mid-run evaluation note, treat it as a strong prior:"
    " contradict it only with clear visual/voice evidence."
)


def commit_session(store: CharacterStore, clouds: CloudEngine, registry: AnchorRegistry,
                   book: ChainBook, *, session_id: str, omni: Any, media_store: Any = None,
                   ) -> dict[str, Any]:
    """对全 pending 链做终审并落库;返回 by_chain/registered/verdicts/fallbacks/wearer。"""
    report: dict[str, Any] = {"by_chain": {}, "registered": [], "verdicts": {}, "fallbacks": {}}
    chains = book.store.pending_chains(session_id)
    if not chains:
        return report

    verdicts, issues, defaulted = _final_arbitration(
        store, registry, book, chains, session_id=session_id, omni=omni, media_store=media_store)
    if verdicts is None:
        # 调用失败:每链回退到当前假设(=最近一次成功评估;NEW 仍 NEW)。
        for chain in chains:
            report["fallbacks"][chain["chain_ref"]] = (
                "hypothesis" if chain["hypothesis"] != "NEW" else "new")
    else:
        # 自绑 = 弃权(超大批模型会把每个 query 自绑,击穿安全网)——指向自身链的 verdict 作无答处理,
        # 走默认回退。
        for chain in chains:
            cast, verdict = chain["cast_id"], verdicts.get(chain["cast_id"])
            if (verdict and book.store.is_chain_ref(verdict)
                    and book.store.canonical_chain(verdict)
                    == book.store.canonical_chain(chain["chain_ref"])):
                issues.append(f"self-bind abstention: {cast}")
                defaulted.add(cast)
        # 缺答/弃权不得默认 NEW(终审 verdict 即刻注册,默认 NEW 会永久建重复档)。回退序:强多数
        # (中途证据共识)→ 链假设 → NEW。显式 NEW 但中途一致强多数指向某档 → 取强多数(防懒 NEW)。
        for chain in chains:
            cast, ref = chain["cast_id"], chain["chain_ref"]
            majority = book.strong_majority(ref)
            if cast in defaulted:
                if majority:
                    verdicts[cast] = majority; report["fallbacks"][ref] = "majority"
                elif chain["hypothesis"] != "NEW":
                    verdicts[cast] = chain["hypothesis"]; report["fallbacks"][ref] = "hypothesis"
                else:
                    verdicts[cast] = "NEW"; report["fallbacks"][ref] = "new"
            elif verdicts.get(cast) == "NEW" and majority:
                verdicts[cast] = majority; report["fallbacks"][ref] = "majority_over_new"
        _apply_final_verdicts(book, chains, verdicts, issues, session_id=session_id, report=report)
    _resolve_final_collisions(book, session_id, report)
    _merge_same_name_chains(book, session_id, report)

    # ── 结算:每 canonical 链落 MySQL ────────────────────────────────
    with store.db.transaction():
        for chain in book.store.pending_chains(session_id):
            ref = chain["chain_ref"]
            aliases = book.store.aliases_of(ref)
            refs = [ref, *aliases]
            final = chain["hypothesis"]
            if final != "NEW" and store.get_character(final) is None:
                logger.warning(f"链 {ref} 假设 {final} 不存在 → NEW")
                final = "NEW"
            if final == "NEW":
                desc = chain["desc_text"]
                profile = {"appearance": desc} if desc else {}
                is_wearer = chain["cast_id"] == WEARER_CAST_ID
                if is_wearer:
                    profile.setdefault("role", "camera wearer")
                final = store.create_character(session_id=session_id, is_wearer=is_wearer,
                                               text_profile=profile)
                report["registered"].append(final)
            # 赢家 staged → character_assets 行 + 学云(跨 ref+aliases 收集一次,避免两链并入同档双学)。
            self_enroll_staged(store, clouds, book, refs, final, session_id)
            # 名字归并(会话内链名 → 持久档)。
            for r in refs:
                for name in book.store.names_for(r):
                    store.add_name_claim(final, name, "chain")
            store.touch_character(final, session_id)
            book.store.commit_chain(ref, final)
            if chain["cast_id"] == WEARER_CAST_ID:
                store.mark_wearer(final)
                store.set_session_wearer(session_id, final)
                report["wearer"] = final
            for r in refs:
                report["by_chain"][r] = final
            logger.info(f"链 {ref} 提交 → {final}(aliases={aliases})")
    return report


def self_enroll_staged(store: CharacterStore, clouds: CloudEngine, book: ChainBook,
                       refs: list[str], final: str, session_id: str) -> None:
    """把链(及别名)的暂存素材落成持久 asset + 学云。oss_key 已在 OSS,直接复用不重传。"""
    for r in refs:
        for a in book.store.active_staged(r, "face"):
            emb = a.get("embedding")
            if emb is None:
                continue
            store.add_asset(final, "face", quality=float(a["q"]), embedding=emb,
                            payload={"oss_key": a.get("oss_key", ""), "t": a.get("t"),
                                     "session": a.get("session", session_id), "clip": a.get("clip")})
            clouds.learn(final, "face", emb, float(a["q"]),
                         payload={"session": session_id, "clip": a.get("clip")})
        for a in book.store.active_staged(r, "body"):
            store.add_asset(final, "body", quality=float(a["q"]), embedding=None,
                            payload={"oss_key": a.get("oss_key", ""), "t": a.get("t"),
                                     "session": a.get("session", session_id), "clip": a.get("clip")})
        for a in book.store.active_staged(r, "voice"):
            emb = a.get("embedding")
            if emb is None:
                continue
            store.add_asset(final, "voice", quality=float(a["q"]), embedding=emb,
                            payload={"oss_key": a.get("oss_key", ""), "t0": a.get("t0"),
                                     "t1": a.get("t1"), "session": a.get("session", session_id),
                                     "clip": a.get("clip")})
            clouds.learn(final, "voice", emb, float(a["q"]),
                         payload={"session": session_id, "clip": a.get("clip")})


# ── 终审调用(与 s4 同形,带声纹;链多时分批保注意力)──────────────────
def _final_arbitration(store: CharacterStore, registry: AnchorRegistry, book: ChainBook,
                       chains: list[dict[str, Any]], *, session_id: str, omni: Any,
                       media_store: Any,
                       ) -> tuple[Optional[dict[str, str]], list[str], set[str]]:
    """返回 ({cast_id: verdict}, issues, 被默认成 NEW 的 cast 集)。分批但每批给全候选池
    (跨批链-链并仍可能)。某批失败 → 该批 cast 进 defaulted;全批失败才返回 (None,[],set())。"""
    queries, pool = _final_materials(store, registry, book, chains, session_id, media_store)
    candidate_ids = [card.character_id for card in pool]
    verdicts: dict[str, str] = {}
    issues: list[str] = []
    defaulted: set[str] = set()
    raws: list[str] = []
    for start in range(0, len(chains), FINAL_REVIEW_BATCH_SIZE):
        batch_no = start // FINAL_REVIEW_BATCH_SIZE + 1
        batch_chains = chains[start:start + FINAL_REVIEW_BATCH_SIZE]
        prompt, images, audios = build_arbitration_prompt(
            queries[start:start + FINAL_REVIEW_BATCH_SIZE], pool)
        prompt += _FINAL_CONSTRAINT
        try:
            raw = omni.chat(prompt, images_b64=images, audio_b64_list=audios, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"终审批 {batch_no} 失败 → 链回退: {exc}")
            issues.append(f"final batch failed: {exc}")
            defaulted.update(c["cast_id"] for c in batch_chains)
            continue
        raws.append(raw)
        bv, bi, bd = parse_verdicts_detailed(
            raw, cast_ids=[c["cast_id"] for c in batch_chains], candidate_ids=candidate_ids)
        verdicts.update(bv); issues.extend(bi); defaulted.update(bd)
    if not raws:
        logger.warning("终审全批失败 → 每链回退")
        return None, [], set()
    return verdicts, issues, defaulted


def _final_materials(store: CharacterStore, registry: AnchorRegistry, book: ChainBook,
                     chains: list[dict[str, Any]], session_id: str, media_store: Any,
                     ) -> tuple[list[dict[str, Any]], list[CandidateCard]]:
    """query 卡 = 链最佳素材(desc/names 取链台账,key_lines 取 roster 卡);候选 = 库(≤small_lib
    全量,否则每链粗召回 + 精确名直通)+ pending 链互为候选。"""
    roster = book.store.load_roster(session_id)
    queries: list[dict[str, Any]] = []
    chain_cards: list[CandidateCard] = []
    names_by_ref: dict[str, list[str]] = {}
    for chain in chains:
        ref = chain["chain_ref"]
        refs = [ref, *book.store.aliases_of(ref)]
        pair = _chain_best_pair(book, refs)
        voice = _chain_best_voice(book, refs)
        names = list(dict.fromkeys(n for r in refs for n in book.store.names_for(r)))
        names_by_ref[ref] = names
        card = roster.get(chain["cast_id"]) or {}
        summary = book.vote_summary(ref)
        majority = book.strong_majority(ref)
        note_parts = [f"own chain id ({ref}) is NOT a valid target for this query"]
        if majority:
            note_parts.append(
                f"mid-run evidence-triggered evaluations bound this person to {majority}"
                f" in {summary['counts'][majority]}/{summary['total']} evaluations")
        queries.append({
            "cast_id": chain["cast_id"],
            "desc": chain["desc_text"],
            "name": names[0] if names else None,
            "key_lines": list(card.get("key_lines") or [])[:3],
            "face_b64": read_b64(media_store, str(pair.get("face_oss_key") or "")),
            "body_b64": read_b64(media_store, str(pair.get("body_oss_key") or "")),
            "voice_b64": read_b64(media_store, str(voice.get("oss_key") or "")),
            "note": "; ".join(note_parts),
        })
        chain_cards.append(CandidateCard(
            character_id=ref, name=names[0] if names else "",
            desc=chain["desc_text"] or "",
            face_b64=read_b64(media_store, str(pair.get("face_oss_key") or "")),
            body_b64=read_b64(media_store, str(pair.get("body_oss_key") or "")),
            voice_b64=read_b64(media_store, str(voice.get("oss_key") or "")),
            last_seen_session=session_id))

    characters = store.list_active_characters(include_wearer=True)
    by_id = {c["id"]: c for c in characters}
    if len(characters) <= registry.small_library_max:
        chosen = list(by_id)
    else:
        chosen = []
        for chain in chains:
            evidence = book.synthetic_evidence(chain["chain_ref"])
            for cid, _score in registry.clouds.coarse_recall(
                    evidence, list(by_id), k=registry.arbiter_top_k):
                if cid not in chosen:
                    chosen.append(cid)
            for name in names_by_ref[chain["chain_ref"]]:   # 精确名直通
                for cid in by_id:
                    if cid not in chosen and name in store.names_for(cid):
                        chosen.append(cid)
    library_cards = [registry.candidate_card(by_id[cid]) for cid in chosen]
    return queries, library_cards + chain_cards


def _chain_best_pair(book: ChainBook, refs: list[str]) -> dict[str, Any]:
    pairs = [pair for r in refs if (pair := book.store.best_pair(r))]
    return max(pairs, key=lambda p: float(p["quality"]), default={})


def _chain_best_voice(book: ChainBook, refs: list[str]) -> dict[str, Any]:
    voices = [v for r in refs if (v := book.store.best_voice(r))]
    return max(voices, key=lambda v: float(v["quality"]), default={})


# ── verdict 一致化(链并 → 同框碰撞降级)────────────────────────────
def _apply_final_verdicts(book: ChainBook, chains: list[dict[str, Any]],
                          verdicts: dict[str, str], issues: list[str],
                          *, session_id: str, report: dict[str, Any]) -> None:
    ref_by_cast = {c["cast_id"]: c["chain_ref"] for c in chains}
    # 先链-链并(unions),再档/NEW verdict(应用到并后 canonical)。
    for cast_id, verdict in verdicts.items():
        ref = ref_by_cast[cast_id]
        report["verdicts"][ref] = verdict
        if not book.store.is_chain_ref(verdict):
            continue
        if book.store.canonical_chain(verdict) == book.store.canonical_chain(ref):
            continue
        if book.copresent(ref, verdict):
            book.apply_verdict(ref, "NEW", session_id=session_id, clip_index=FINAL_CLIP_INDEX,
                               reason="final_arbitration",
                               issues=[f"copresent merge rejected: {cast_id} -> {verdict}"])
            report["verdicts"][ref] = "NEW"
            continue
        book.apply_verdict(ref, verdict, session_id=session_id, clip_index=FINAL_CLIP_INDEX,
                           reason="final_arbitration", issues=issues)
    for cast_id, verdict in verdicts.items():
        if book.store.is_chain_ref(verdict):
            continue
        book.apply_verdict(ref_by_cast[cast_id], verdict, session_id=session_id,
                           clip_index=FINAL_CLIP_INDEX, reason="final_arbitration", issues=issues)


def _merge_same_name_chains(book: ChainBook, session_id: str, report: dict[str, Any]) -> None:
    """名字兜底合并(personos 新增,mneme 无——名字不单独定夺,故加护栏):终审后仍分裂的同名链,
    合并到 presence 最长的那条。护栏:①只并 hypothesis 仍为 NEW 的链(已绑具体注册档的尊重模型的
    区分,不按名字强并——防两个真重名的人被错并);②同框(presence 交)绝不并(物理不可能同人)。
    重名极少,此为收拾"模型视觉犹豫没合并的残留分裂"的最后一层,落审计日志可回溯。"""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for chain in book.store.pending_chains(session_id):
        if chain["hypothesis"] != "NEW":
            continue                                     # 护栏①:已绑档的不动
        names = book.store.names_for(chain["chain_ref"])
        if names:
            by_name.setdefault(names[0], []).append(chain)
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        # presence 最长者留(早出场破平)
        group.sort(key=lambda c: (-len(c["presence"]), min(c["presence"] or [10**9])))
        keep, merged = group[0], []
        for other in group[1:]:
            if book.copresent(other["chain_ref"], keep["chain_ref"]):
                continue                                 # 护栏②:同框不并
            book.store.merge_chain(other["chain_ref"], keep["chain_ref"])
            merged.append(other["chain_ref"])
        if merged:
            report.setdefault("name_merges", []).append(
                {"name": name, "kept": keep["chain_ref"], "merged": merged})
            logger.info(f"名字兜底合并 name={name!r} 保留={keep['chain_ref']} 并入={merged}")


def _resolve_final_collisions(book: ChainBook, session_id: str, report: dict[str, Any]) -> None:
    """同框链撞同档 → 组内按证据序(best_face_q 高者留;早出场破平);恰一条强多数则升到留位;
    同框其余降 NEW。与 chains.resolve_chain_collisions 的降级分支同取舍,终审后不再调模型。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for chain in book.store.pending_chains(session_id):
        if chain["hypothesis"] != "NEW":
            groups.setdefault(chain["hypothesis"], []).append(chain)
    for target, group in groups.items():
        if len(group) < 2:
            continue
        strong = [c for c in group if book.strong_majority(c["chain_ref"]) == target]
        group.sort(key=lambda c: (-c["best_face_q"], min(c["presence"] or [10**9])))
        if len(strong) == 1:
            group.remove(strong[0]); group.insert(0, strong[0])
        kept = [group[0]]
        for chain in group[1:]:
            if any(set(chain["presence"]) & set(k["presence"]) for k in kept):
                book.apply_verdict(chain["chain_ref"], "NEW", session_id=session_id,
                                   clip_index=FINAL_CLIP_INDEX, reason="final_collision_degrade",
                                   issues=[f"copresent chains bound to {target}"])
                report["verdicts"][chain["chain_ref"]] = "NEW"
                logger.info(f"终审碰撞 {target}:链 {chain['chain_ref']} 降 NEW")
            else:
                kept.append(chain)                       # 非同框可合法共享一档
