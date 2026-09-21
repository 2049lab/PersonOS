"""commit_session: the end-of-session two-phase final adjudication, then settlement
into MySQL.

Final adjudication is a batched model call. The material for each query is the
session's best face and body shot plus desc, names and a voiceprint; the
candidates are the library (in full when it is no larger than small_library_max,
otherwise coarse recall per chain) together with the pending chains, which serve
as candidates for one another.

The verdicts are first made consistent — chain-to-chain merges are applied, and
where same-frame chains collide on one profile the evidence picks a winner and the
rest degrade to NEW — and then settled per canonical chain. A NEW profile is
created only at this point, and persisting assets, merging names and learning into
the cloud all happen in one pass.

The wearer goes through the same adjudication on the SW chain, where the
voiceprint is the only biometric available. The profile it settles on becomes this
session's wearer pointer and is marked is_wearer.

If the adjudication call fails, each chain falls back to its most recent
successful evaluation, that is, its chain hypothesis. A chain that crashes before
commit stays pending, and a re-run of finalize picks it up.

On atomicity: under test, rollback_scope pins the connection so SAVEPOINT makes
settlement genuinely atomic, since every store write goes through the pinned
connection. In production CharacterStore autocommits statement by statement, per
the Database facade's contract, and the safety net is the pending status plus
idempotent re-runs — draft.commit_chain only marks a chain committed after its
MySQL writes have succeeded. Line rewriting is not done here; this only delivers
the attribution mapping.
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

# Final adjudication belongs to no clip; the evaluation ledger marks it with -1.
FINAL_CLIP_INDEX = -1
# Too large a batch degrades the model's attention — we observed it self-binding
# every query in one — so we split into batches to keep attention intact.
FINAL_REVIEW_BATCH_SIZE = 12

# The two general rules (chain ids and same-frame exclusivity) are already in
# recognize.PROMPT_HEADER. This trailing note adds three things specific to final
# adjudication: the verdict is final, a self-bind is invalid, and the note is a
# strong prior.
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
    """Run final adjudication over every pending chain and persist the result.

    Returns by_chain, registered, verdicts, fallbacks and wearer.
    """
    report: dict[str, Any] = {"by_chain": {}, "registered": [], "verdicts": {}, "fallbacks": {}}
    chains = book.store.pending_chains(session_id)
    if not chains:
        return report

    verdicts, issues, defaulted = _final_arbitration(
        store, registry, book, chains, session_id=session_id, omni=omni, media_store=media_store)
    if verdicts is None:
        # The call failed, so every chain falls back to its current hypothesis,
        # which is its most recent successful evaluation. NEW stays NEW.
        for chain in chains:
            report["fallbacks"][chain["chain_ref"]] = (
                "hypothesis" if chain["hypothesis"] != "NEW" else "new")
    else:
        # A self-bind counts as abstention. With an oversized batch the model binds
        # every query to itself, which would punch straight through the safety net,
        # so a verdict pointing at the query's own chain is treated as no answer and
        # takes the default fallback.
        for chain in chains:
            cast, verdict = chain["cast_id"], verdicts.get(chain["cast_id"])
            if (verdict and book.store.is_chain_ref(verdict)
                    and book.store.canonical_chain(verdict)
                    == book.store.canonical_chain(chain["chain_ref"])):
                issues.append(f"self-bind abstention: {cast}")
                defaulted.add(cast)
        # A missing answer or an abstention must not default to NEW: a final
        # verdict registers immediately, so defaulting to NEW would permanently
        # create a duplicate profile. The fallback order is strong majority (the
        # evidence consensus built up mid-run), then the chain hypothesis, then
        # NEW. And an explicit NEW is overridden when a consistent strong majority
        # mid-run pointed at a profile, which guards against a lazy NEW.
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

    # ── Settlement: each canonical chain lands in MySQL ─────────────
    with store.db.transaction():
        for chain in book.store.pending_chains(session_id):
            ref = chain["chain_ref"]
            aliases = book.store.aliases_of(ref)
            refs = [ref, *aliases]
            final = chain["hypothesis"]
            if final != "NEW" and store.get_character(final) is None:
                logger.warning(f"chain {ref} hypothesis {final} does not exist -> NEW")
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
            # The winner's staged assets become character_assets rows and are
            # learned into the cloud. We collect across ref plus its aliases in one
            # go, so two chains merging into one profile do not learn it twice.
            self_enroll_staged(store, clouds, book, refs, final, session_id)
            # Merge names: the chain's session-scoped name onto the persistent profile.
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
            logger.info(f"chain {ref} committed -> {final} (aliases={aliases})")
    return report


def self_enroll_staged(store: CharacterStore, clouds: CloudEngine, book: ChainBook,
                       refs: list[str], final: str, session_id: str) -> None:
    """Turn the staged assets of a chain and its aliases into persistent assets and
    learn them into the cloud.

    The oss_key already points at OSS, so it is reused rather than re-uploaded.
    """
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


# ── The final adjudication call. Same shape as clip-level arbitration, but
#    carrying voiceprints, and batched when there are many chains so the model's
#    attention holds up ───────────────────────────────────────────────────
def _final_arbitration(store: CharacterStore, registry: AnchorRegistry, book: ChainBook,
                       chains: list[dict[str, Any]], *, session_id: str, omni: Any,
                       media_store: Any,
                       ) -> tuple[Optional[dict[str, str]], list[str], set[str]]:
    """Returns ({cast_id: verdict}, issues, the set of casts defaulted to NEW).

    The queries are batched, but every batch is given the full candidate pool, so
    a chain-to-chain merge across batches remains possible. If one batch fails its
    casts go into defaulted; only if every batch fails do we return
    (None, [], set()).
    """
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
            logger.warning(f"final review batch {batch_no} failed -> chains fall back: {exc}")
            issues.append(f"final batch failed: {exc}")
            defaulted.update(c["cast_id"] for c in batch_chains)
            continue
        raws.append(raw)
        bv, bi, bd = parse_verdicts_detailed(
            raw, cast_ids=[c["cast_id"] for c in batch_chains], candidate_ids=candidate_ids)
        verdicts.update(bv); issues.extend(bi); defaulted.update(bd)
    if not raws:
        logger.warning("every final review batch failed -> every chain falls back")
        return None, [], set()
    return verdicts, issues, defaulted


def _final_materials(store: CharacterStore, registry: AnchorRegistry, book: ChainBook,
                     chains: list[dict[str, Any]], session_id: str, media_store: Any,
                     ) -> tuple[list[dict[str, Any]], list[CandidateCard]]:
    """Build the query cards and the candidate pool.

    A query card is the chain's best assets, with desc and names taken from the
    chain ledger and key_lines from the roster card. The candidates are the
    library — in full when it is small enough, otherwise coarse recall per chain
    plus exact name hits — together with the pending chains, which serve as
    candidates for one another.
    """
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
            for name in names_by_ref[chain["chain_ref"]]:   # exact name hits pass through
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


# ── Making the verdicts consistent: apply chain merges, then degrade
#    same-frame collisions ───────────────────────────────────────────
def _apply_final_verdicts(book: ChainBook, chains: list[dict[str, Any]],
                          verdicts: dict[str, str], issues: list[str],
                          *, session_id: str, report: dict[str, Any]) -> None:
    ref_by_cast = {c["cast_id"]: c["chain_ref"] for c in chains}
    # Apply the chain-to-chain unions first, then the profile and NEW verdicts,
    # which land on the canonical chain that resulted from those unions.
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
    """A last-resort merge by name: same-name chains still split after final
    adjudication are merged into the one with the longest presence.

    A name never decides an identity on its own, so this needs guards. First, only
    chains whose hypothesis is still NEW are merged — a chain already bound to a
    specific registered profile keeps the distinction the model drew, rather than
    being forced together by name, which is what stops two genuinely different
    people who share a name from being merged. Second, chains that have appeared
    in the same frame (their presence sets intersect) are never merged, since they
    cannot physically be the same person.

    Shared names are rare, and this is the last layer, there to clean up residual
    splits the model left behind when it hesitated visually. Every merge is
    written to the audit log so it can be traced back.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for chain in book.store.pending_chains(session_id):
        if chain["hypothesis"] != "NEW":
            continue                                     # guard one: leave chains already bound to a profile alone
        names = book.store.names_for(chain["chain_ref"])
        if names:
            by_name.setdefault(names[0], []).append(chain)
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        # The longest presence is kept, with earlier appearance breaking ties.
        group.sort(key=lambda c: (-len(c["presence"]), min(c["presence"] or [10**9])))
        keep, merged = group[0], []
        for other in group[1:]:
            if book.copresent(other["chain_ref"], keep["chain_ref"]):
                continue                                 # guard two: never merge chains seen in the same frame
            book.store.merge_chain(other["chain_ref"], keep["chain_ref"])
            merged.append(other["chain_ref"])
        if merged:
            report.setdefault("name_merges", []).append(
                {"name": name, "kept": keep["chain_ref"], "merged": merged})
            logger.info(f"name fallback merge name={name!r} kept={keep['chain_ref']} merged={merged}")


def _resolve_final_collisions(book: ChainBook, session_id: str, report: dict[str, Any]) -> None:
    """Resolve same-frame chains colliding on one profile.

    Within the group, evidence decides: the highest best_face_q is kept, with
    earlier appearance breaking ties, and if exactly one chain holds a strong
    majority it is promoted into that kept position. The remaining same-frame
    chains degrade to NEW. This makes the same trade-off as the degrade branch of
    chains.resolve_chain_collisions, but calls no model, since adjudication is
    already over.
    """
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
                logger.info(f"final collision on {target}: chain {chain['chain_ref']} degraded to NEW")
            else:
                kept.append(chain)                       # chains never seen together may legitimately share one profile
