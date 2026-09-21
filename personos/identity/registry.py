"""AnchorRegistry: cast mapping, candidate construction, name harvesting and roster updates.

- cast mapping: local character -> session cast, via the roster and CONT records,
  trusting the model's own continuation output;
- candidate construction: a small library passes through whole, a large one goes
  through probability-cloud coarse recall, plus exact name hits, plus extra_cards
  (the pending chain cards);
- name harvesting and roster updates. Merging names into the profile at
  end-of-session happens in commit.py.

This module neither binds nor enrolls — attribution here is only a chain
hypothesis, see chains.py — and it never calls a model, since the orchestration
layer injects the verdict. That keeps it pure logic, and unit-testable.

Where things live: persistent profiles in the MySQL-backed CharacterStore,
the session roster in DraftStore, and candidate-card assets read out of OSS as b64.
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

    # ── Cast mapping: local character -> session cast ────────────────
    def map_casts(self, session_id: str, script: ClipScript) -> dict[str, str]:
        """Fill in script.cast_map.

        We trust the model's continuation output and accept both forms it can
        take:

        - a cast that reuses a roster id (S#) directly, which is an implicit
          continuation;
        - a cast P# plus a CONT with prev=S#, which is an explicit one.

        prev=none, a missing prev, or an unknown id all mint a fresh S#. The
        wearer is always SW: never visible, and never part of a CONT.
        """
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
                mapping[cast.local_id] = cast.local_id     # reusing a roster id is an implicit continuation
                used.add(cast.local_id)
            else:
                if prev not in ("none", WEARER_CAST_ID) and prev not in roster:
                    script.issues.append(
                        f"CONT prev={prev} not in roster for {cast.local_id} → new cast")
                mapping[cast.local_id] = mint()
        script.cast_map = mapping
        return mapping

    # ── Candidate construction: a small library passes through whole ──
    def build_candidates(self, unbound_cast_ids: list[str],
                         evidence_by_cast: dict[str, CastEvidence], script: ClipScript,
                         *, extra_cards: Optional[dict[str, list[CandidateCard]]] = None,
                         ) -> dict[str, list[CandidateCard]]:
        """extra_cards holds per-cast additional candidates, such as pending chain
        cards. They take no small-library seat and skip coarse recall.
        """
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
                if declared and declared.name:            # an exact name hit passes through regardless of score
                    for cid in cards_by_id:
                        if declared.name in self.store.names_for(cid) and cid not in chosen:
                            chosen.append(cid)
            out[cast_id] = ([self.candidate_card(cards_by_id[cid]) for cid in chosen]
                            + list(extras.get(cast_id, [])))
        return out

    def candidate_card(self, character: dict[str, Any]) -> CandidateCard:
        """Character -> candidate card: name and description plus face, body shot
        and voice assets as b64.

        This is deliberately public, because the retrieval side's visual rewrite
        (online/visual_query) has to build candidate cards the same way. A second
        copy would drift — details like how the asset key is fetched, or the fact
        that last_seen_session lives inside payload, are exactly what gets copied
        wrong. It depends only on store and media_store, and touches neither
        clouds, draft nor ClipScript.
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

    # ── Name harvesting. Merging names into the bound profile happens at
    #    end-of-session; here we only record them into the ledger ──────
    def harvest_names(self, session_id: str, script: ClipScript,
                      bindings: dict[str, str]) -> dict[str, list[str]]:
        """bindings is {cast_id -> character_id}. Record this clip's name claims onto the bound profile."""
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

    # ── Roster updates: the session's continuation register, kept in DraftStore ──
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
            # The SW card carries only name and lines, as context for telling
            # close-mic speech apart from off-screen voice. The wearer is
            # physically invisible, so it carries no face.
            # Roster assets are inlined as b64 — only the single best one per cast,
            # so the volume stays bounded — for the ROSTER block of the next clip's
            # screenplay prompt to actually look at.
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
