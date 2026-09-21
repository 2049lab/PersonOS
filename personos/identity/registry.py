"""AnchorRegistry(⑥b):cast 映射 / 候选构造 / 名字采集 + roster 更新。

移植自 mneme anchor/registry.py,逐处对齐:
- cast 映射:local 角色 → 会话 cast(roster 续接 + CONT,信任模型的续接输出);
- 候选构造:小库直通 / 大库概率云粗召回 + 精确名命中 + extra_cards(pending 链卡);
- 名字采集 + roster 更新(会话末归并进档在 commit.py)。

不做绑定/注册(归属只是链假设,见 chains.py);不调模型(编排层注入 verdict),故纯逻辑可单测。
有意偏离:持久档在 MySQL CharacterStore;会话 roster 在 DraftStore;候选卡素材从 OSS 取 b64。
"""

from __future__ import annotations

from typing import Any, Optional

from personos.identity.cloud import CloudEngine
from personos.identity.draft import DraftStore, read_b64
from personos.identity.screenplay import WEARER_CAST_ID, CastDecl, ClipScript
from personos.identity.store import CharacterStore
from personos.identity.types import CandidateCard, CastEvidence


class AnchorRegistry:
    def __init__(self, store: CharacterStore, clouds: CloudEngine, draft: DraftStore,
                 media_store: Any = None, *, small_library_max: int = 20,
                 arbiter_top_k: int = 3) -> None:
        self.store = store
        self.clouds = clouds
        self.draft = draft
        self.media_store = media_store
        self.small_library_max = small_library_max
        self.arbiter_top_k = arbiter_top_k

    # ── cast 映射:local 角色 → 会话 cast ────────────────────────────
    def map_casts(self, session_id: str, script: ClipScript) -> dict[str, str]:
        """回填 script.cast_map。信任模型续接输出,两种续接形式都接受:
        - CAST 直接复用 roster id(S#)→ 隐式续接;
        - CAST P# + CONT prev=S# → 显式续接。
        prev=none/缺失/未知 id → 铸新 S#。佩戴者恒 SW(从不可见,不参与 CONT)。"""
        roster = self.draft.load_roster(session_id)
        used = {cid for cid in roster if cid != WEARER_CAST_ID}
        mapping: dict[str, str] = {}

        def mint() -> str:
            index = 1 + max((int(cid[1:]) for cid in used if cid[1:].isdigit()), default=0)
            new_id = f"S{index}"
            used.add(new_id)
            return new_id

        for cast in script.casts:
            if cast.is_wearer or cast.local_id == WEARER_CAST_ID:
                mapping[cast.local_id] = WEARER_CAST_ID
                continue
            prev = script.cont.get(cast.local_id, "none")
            if prev != "none" and prev != WEARER_CAST_ID and prev in roster:
                mapping[cast.local_id] = prev
                used.add(prev)
            elif cast.local_id in roster:
                mapping[cast.local_id] = cast.local_id     # 复用 roster id = 隐式续接
                used.add(cast.local_id)
            else:
                if prev not in ("none", WEARER_CAST_ID) and prev not in roster:
                    script.issues.append(
                        f"CONT prev={prev} not in roster for {cast.local_id} → new cast")
                mapping[cast.local_id] = mint()
        script.cast_map = mapping
        return mapping

    # ── 候选构造(小库全量直通)──────────────────────────────────────
    def build_candidates(self, unbound_cast_ids: list[str],
                         evidence_by_cast: dict[str, CastEvidence], script: ClipScript,
                         *, extra_cards: Optional[dict[str, list[CandidateCard]]] = None,
                         ) -> dict[str, list[CandidateCard]]:
        """extra_cards:每 cast 的附加候选(如 pending 链卡),不占小库席位、不走粗召回。"""
        extras = extra_cards or {}
        characters = self.store.list_active_characters(include_wearer=True)
        if not characters:
            return {cid: list(extras.get(cid, [])) for cid in unbound_cast_ids}
        cards_by_id = {c["id"]: c for c in characters}
        out: dict[str, list[CandidateCard]] = {}
        small_library = len(characters) <= self.small_library_max
        for cast_id in unbound_cast_ids:
            if small_library:
                chosen = list(cards_by_id)
            else:
                evidence = evidence_by_cast.get(cast_id) or CastEvidence(cast_id=cast_id)
                ranked = self.clouds.coarse_recall(evidence, list(cards_by_id), k=self.arbiter_top_k)
                chosen = [cid for cid, _score in ranked]
                declared = self._decl_for_cast(script, cast_id)
                if declared and declared.name:            # 精确名命中直通(不受分数限制)
                    for cid in cards_by_id:
                        if declared.name in self.store.names_for(cid) and cid not in chosen:
                            chosen.append(cid)
            out[cast_id] = ([self.candidate_card(cards_by_id[cid]) for cid in chosen]
                            + list(extras.get(cast_id, [])))
        return out

    def candidate_card(self, character: dict[str, Any]) -> CandidateCard:
        """人物 → 候选卡(名字/描述 + 脸/全身/声音素材 b64)。

        公开:召回侧的视觉改写(online/visual_query)也要按同一口径组装候选卡——
        抄一份必然漂(素材 key 的取法、last_seen_session 在 payload 里这类细节最容易抄错)。
        只依赖 store + media_store,不碰 clouds/draft/ClipScript。
        """
        cid = character["id"]
        names = self.store.names_for(cid)
        name = (names[0] if names else None) or character.get("primary_name") or ""
        tp = (character.get("payload") or {}).get("text_profile")
        desc = tp.get("appearance", "") if isinstance(tp, dict) else ""
        return CandidateCard(
            character_id=cid, name=name, desc=desc,
            face_b64=read_b64(self.media_store, self._asset_oss_key(cid, "face")),
            body_b64=read_b64(self.media_store, self._asset_oss_key(cid, "body")),
            voice_b64=read_b64(self.media_store, self._asset_oss_key(cid, "voice")),
            last_seen_session=(character.get("payload") or {}).get("last_seen_session") or "")

    def _asset_oss_key(self, character_id: str, kind: str) -> str:
        asset = self.store.best_asset(character_id, kind)
        return (asset.get("payload") or {}).get("oss_key", "") if asset else ""

    @staticmethod
    def _decl_for_cast(script: ClipScript, cast_id: str) -> Optional[CastDecl]:
        for local_id, mapped in script.cast_map.items():
            if mapped == cast_id:
                decl = script.cast_decl(local_id)
                if decl is not None:
                    return decl
        return None

    # ── 名字采集(会话末把绑定档的名字归并;此处采集台账)─────────────
    def harvest_names(self, session_id: str, script: ClipScript,
                      bindings: dict[str, str]) -> dict[str, list[str]]:
        """bindings: {cast_id -> character_id}。把本 clip 的名字 claim 记到绑定档。"""
        harvested: dict[str, list[str]] = {}
        for cast in script.casts:
            if not cast.name or cast.name_evidence == "none":
                continue
            cast_id = script.cast_map.get(cast.local_id)
            character_id = bindings.get(cast_id or "")
            if not character_id:
                continue
            self.store.add_name_claim(character_id, cast.name, cast.name_evidence)
            harvested.setdefault(character_id, []).append(cast.name)
        return harvested

    # ── roster 更新(会话续接名册,存 DraftStore)─────────────────────
    def update_roster(self, session_id: str, clip_index: int, script: ClipScript,
                      bindings: dict[str, str], evidence_by_cast: dict[str, CastEvidence]) -> None:
        roster = self.draft.load_roster(session_id)
        for cast_id in script.session_cast_ids():
            decl = self._decl_for_cast(script, cast_id)
            evidence = evidence_by_cast.get(cast_id)
            card = dict(roster.get(cast_id) or {})
            card.pop("character_id", None)
            if decl:
                if decl.name:
                    card["name"] = decl.name
                if decl.desc:
                    card["desc"] = decl.desc
            key_lines = list(card.get("key_lines") or [])
            for line in script.lines:
                if line.kind == "speech" and script.cast_map.get(line.who) == cast_id:
                    key_lines.append(line.text)
            card["key_lines"] = key_lines[-3:]
            # SW 卡只带 name/lines 作 s1 上下文(区分近麦说话 vs 画外音);佩戴者物理不可见→不带脸。
            # roster 素材内联 b64(仅每 cast 最佳一张,量有界):供下一 clip 剧本 prompt 的 ROSTER 块看图。
            if cast_id != WEARER_CAST_ID:
                best_q = float(card.get("best_q") or -1.0)
                best = evidence.best_face() if evidence else None
                if best is not None and best.q >= best_q:
                    card["best_q"] = best.q
                    card["face_b64"] = best.crop_b64
                    body = evidence.best_body() if evidence else None
                    card["body_b64"] = body.body_crop_b64 if body else ""
            card["last_clip"] = clip_index
            self.draft.save_roster_entry(session_id, cast_id,
                                         character_id=bindings.get(cast_id), card=card)

    def roster_cards_for_prompt(self, session_id: str) -> list[dict[str, Any]]:
        roster = self.draft.load_roster(session_id)
        cards = []
        for cast_id, card in sorted(roster.items()):
            character_id = card.get("character_id")
            name = card.get("name")
            if not name and character_id:
                names = self.store.names_for(str(character_id))
                name = names[0] if names else None
            cards.append({
                "cast_id": cast_id,
                "character_id": character_id,
                "name": name,
                "desc": card.get("desc", ""),
                "key_lines": card.get("key_lines") or [],
                "face_b64": card.get("face_b64", ""),
                "body_b64": card.get("body_b64", ""),
                "is_wearer": cast_id == WEARER_CAST_ID,
            })
        return cards
