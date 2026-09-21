"""ChainBook: the evidence ledger for identity chains, binding transactionally
within a session.

A chain carries identity inside a session, held together by roster continuation,
and global attribution is only ever the chain's **current hypothesis**. This
module maintains the chain's evidence ledger and decides when "new best evidence"
should trigger a refresh. That decision is a pure relative comparison: no new
thresholds, and no model calls. Invoking the evaluation and running the final
adjudication belong to the orchestration layer and commit.py.

Storage follows draft.py: the store is a DraftStore backed by Redis; assets live
in the draft as oss_keys and are fetched as b64 when a card is built; and vector
types come from our own types module (FacePick.crop_b64 / body_crop_b64,
VoiceSample.wav_bytes).
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from loguru import logger

from personos.identity.draft import DraftStore, read_b64
from personos.identity.screenplay import ClipScript
from personos.identity.types import CandidateCard, CastEvidence, FacePick, VoiceSample

# Trigger reasons, written into the chain evaluation's reason field.
FIRST_SEEN = "first_seen"
BETTER_FACE = "better_face"
BETTER_VOICE = "better_voice"
FIRST_NAME = "first_name"

# The ballot box counts only evidence-triggered evaluations; collision
# re-arbitration (collision_*) and final adjudication (final_*) do not count.
# A strong majority means a stable evidence consensus, which serves as the
# fallback when final adjudication abstains and as protection for the stronger
# side of a collision.
EVIDENCE_REASONS = frozenset({FIRST_SEEN, BETTER_FACE, BETTER_VOICE, FIRST_NAME})
MAJORITY_MIN_COUNT = 3
MAJORITY_MIN_RATIO = 0.6


class ChainBook:
    """The identity-chain ledger for one session.

    store is the session draft storage; media_store turns a draft asset's oss_key
    back into b64.
    """

    def __init__(self, store: DraftStore, media_store: Any = None) -> None:
        self.store = store
        self.media_store = media_store

    def _b64(self, oss_key: str) -> str:
        return read_b64(self.media_store, oss_key)

    # ── Ledger update and refresh decision. Runs after harvest, calls no model ──
    def observe_clip(
        self, session_id: str, clip_index: int, script: ClipScript,
        evidence_by_cast: dict[str, CastEvidence],
    ) -> dict[str, list[str]]:
        refreshed: dict[str, list[str]] = {}
        for cast_id in script.session_cast_ids():
            self.store.ensure_chain(session_id, cast_id)
            canonical = self.store.canonical_chain(self.store.chain_ref(session_id, cast_id))
            chain = self.store.get_chain(canonical)
            evidence = evidence_by_cast.get(cast_id) or CastEvidence(cast_id=cast_id)
            decl = self._decl(script, cast_id)

            # Ledger update: best q, named, desc and presence are recorded whether
            # or not a refresh is triggered.
            updates: dict[str, Any] = {}
            best_face = max((p for p in evidence.faces if p.embedding is not None),
                            key=lambda p: p.q, default=None)
            face_improved = best_face is not None and best_face.q > chain["best_face_q"]
            if face_improved:
                updates["best_face_q"] = best_face.q
            best_voice = max((v for v in evidence.voices if v.embedding is not None),
                             key=lambda v: v.q, default=None)
            voice_improved = best_voice is not None and best_voice.q > chain["best_voice_q"]
            if voice_improved:
                updates["best_voice_q"] = best_voice.q
            # Being introduced is strong name evidence, since it comes from someone
            # else's speech. Acquiring a name for the first time must therefore
            # trigger FIRST_NAME; otherwise an introduced name never carries the
            # chain back into arbitration against the enrolled profiles. The weaker
            # evidence kinds (spoken, self_introduction, visible_text) still do not
            # trigger it.
            name_first = bool(
                decl and decl.name
                and decl.name_evidence in ("explicit_dialogue", "introduction")
                and not chain["named"])
            if name_first:
                updates["named"] = 1
            if decl and decl.desc:
                updates["desc_text"] = decl.desc
            updates["presence"] = sorted(set(chain["presence"]) | {clip_index})

            # Refresh decision. first_seen means this chain root had never been
            # observed (presence went from empty to non-empty), which already
            # implies "first best evidence", so it is not listed alongside
            # better_* or first_name.
            reasons: list[str] = []
            if not chain["presence"]:
                reasons.append(FIRST_SEEN)
            if FIRST_SEEN not in reasons:
                if face_improved:
                    reasons.append(BETTER_FACE)
                if voice_improved:
                    reasons.append(BETTER_VOICE)
                if name_first:
                    reasons.append(FIRST_NAME)

            self.store.update_chain(canonical, **updates)
            if reasons:
                merged = refreshed.setdefault(canonical, [])
                merged.extend(r for r in reasons if r not in merged)
                logger.debug(f"chain {canonical} refreshed at clip {clip_index}: {reasons}")
        return refreshed

    @staticmethod
    def _decl(script: ClipScript, cast_id: str):
        for local_id, mapped in script.cast_map.items():
            if mapped == cast_id:
                decl = script.cast_decl(local_id)
                if decl is not None:
                    return decl
        return None

    # ── Applying a verdict: this changes the hypothesis only, it never enrolls ──
    def apply_verdict(self, chain_ref: str, verdict: str, *, session_id: str,
                      clip_index: int, reason: str, issues: list[str]) -> str:
        canonical = self.store.canonical_chain(chain_ref)
        chain = self.store.get_chain(canonical)
        # The same-frame guard, the same rule final adjudication applies, pulled
        # forward to clip level. The arbitration candidate pool is the union across
        # casts, so the model can judge two chains that appeared in the same frame
        # to be one person. That merge is irreversible, and afterwards the
        # collision rule sees them as a single chain recurring — permanently blind
        # to the mistake.
        is_merge = self.store.is_chain_ref(verdict)
        target = self.store.canonical_chain(verdict) if is_merge else None
        rejected = (is_merge and target != canonical and self.copresent(canonical, target))
        if rejected:
            issues = [*issues, f"copresent merge rejected: {target}"]
        self.store.add_chain_evaluation(
            canonical, session_id=session_id, clip_index=clip_index, reason=reason,
            verdict=verdict, issues=issues,
            evidence={"best_face_q": chain["best_face_q"],
                      "best_voice_q": chain["best_voice_q"], "named": chain["named"]})
        if rejected:
            return canonical                             # the hypothesis is unchanged
        if is_merge:
            if target != canonical:
                self.store.merge_chain(canonical, target)
            return target
        self.store.update_chain(canonical, hypothesis=verdict, hypo_method=reason)
        return canonical

    # ── Evaluation material: query cards, synthetic recall evidence, pending candidate cards ──
    def query_card(self, session_id: str, cast_id: str, script: ClipScript, *,
                   evidence: Optional[CastEvidence] = None) -> dict[str, Any]:
        canonical = self.store.canonical_chain(self.store.chain_ref(session_id, cast_id))
        chain = self.store.get_chain(canonical) or {}
        pair = self.store.best_pair(canonical) or {}
        # Evaluation happens before staging, so on first sight the staged set is
        # empty, and on a re-evaluation this clip's evidence has not been staged
        # yet. So we use the current image when its quality is higher, and the best
        # staged one otherwise. The face and the body shot come from the same
        # higher-q source (staged or current), though the two images are not
        # guaranteed to be from the same instant.
        staged_q = float(pair["quality"]) if pair.get("quality") is not None else -1.0
        best_face = evidence.best_face() if evidence is not None else None
        if best_face is not None and best_face.q > staged_q:
            best_body = evidence.best_body()
            face_b64 = best_face.crop_b64 or ""
            body_b64 = (best_body.body_crop_b64 or "") if best_body else ""
        else:
            face_b64 = self._b64(str(pair.get("face_oss_key") or ""))
            body_b64 = self._b64(str(pair.get("body_oss_key") or ""))
        names = self.store.names_for(canonical)
        decl = self._decl(script, cast_id)
        key_lines = [line.text for line in script.lines
                     if line.kind == "speech" and script.cast_map.get(line.who) == cast_id][:3]
        # The voice sample follows the same policy: best staged versus this clip's
        # evidence, higher q wins, and it must carry real wav bytes.
        staged_voice = self.store.best_voice(canonical) or {}
        staged_voice_q = (float(staged_voice["quality"])
                          if staged_voice.get("quality") is not None else -1.0)
        cur_voice = evidence.best_voice_wav() if evidence is not None else None
        if cur_voice is not None and cur_voice.q > staged_voice_q:
            voice_b64 = base64.b64encode(cur_voice.wav_bytes).decode() if cur_voice.wav_bytes else ""
        else:
            voice_b64 = self._b64(str(staged_voice.get("oss_key") or ""))
        return {
            "cast_id": cast_id,
            "desc": chain.get("desc_text") or (decl.desc if decl else ""),
            "name": (decl.name if decl and decl.name else None) or (names[0] if names else None),
            "key_lines": key_lines,
            "face_b64": face_b64,
            "body_b64": body_b64,
            "voice_b64": voice_b64,
        }

    def synthetic_evidence(self, chain_ref: str) -> CastEvidence:
        """Synthesize evidence from the best staged assets, for coarse recall over a
        large library. Only embedding and q are needed.
        """
        canonical = self.store.canonical_chain(chain_ref)
        evidence = CastEvidence(cast_id=canonical)
        for asset in self.store.active_staged(canonical, "face")[:1]:   # already ordered by q descending
            if asset.get("embedding") is not None:
                evidence.faces.append(FacePick(t=float(asset.get("t") or 0),
                                               embedding=asset["embedding"], q=float(asset["q"])))
        for asset in self.store.active_staged(canonical, "voice")[:1]:
            if asset.get("embedding") is not None:
                evidence.voices.append(VoiceSample(t0=float(asset.get("t0") or 0),
                                                   t1=float(asset.get("t1") or 0),
                                                   embedding=asset["embedding"], q=float(asset["q"])))
        return evidence

    def copresent(self, ref_a: str, ref_b: str) -> bool:
        a = self.store.get_chain(self.store.canonical_chain(ref_a))
        b = self.store.get_chain(self.store.canonical_chain(ref_b))
        if a is None or b is None:
            return False
        return bool(set(a["presence"]) & set(b["presence"]))

    def vote_summary(self, chain_ref: str) -> dict[str, Any]:
        """Tally the evidence-triggered evaluations across the canonical chain and
        its aliases. Chain-to-chain merge verdicts are not counted.
        """
        canonical = self.store.canonical_chain(chain_ref)
        counts: dict[str, int] = {}
        total = 0
        for ref in (canonical, *self.store.aliases_of(canonical)):
            for row in self.store.evaluations_for(ref):
                reasons = set(str(row["reason"] or "").split("+"))
                if not reasons & EVIDENCE_REASONS:
                    continue
                verdict = str(row["verdict"] or "")
                if not verdict or self.store.is_chain_ref(verdict):
                    continue
                total += 1
                counts[verdict] = counts.get(verdict, 0) + 1
        return {"counts": counts, "total": total}

    def strong_majority(self, chain_ref: str) -> Optional[str]:
        """The strong-majority profile: the character_id with at least
        MAJORITY_MIN_COUNT votes and at least MAJORITY_MIN_RATIO of the share.

        NEW is never a target, but it does count toward the denominator.
        """
        summary = self.vote_summary(chain_ref)
        total = summary["total"]
        if not total:
            return None
        target, count = max(((t, c) for t, c in summary["counts"].items() if t != "NEW"),
                            key=lambda item: item[1], default=(None, 0))
        if target is None:
            return None
        if count >= MAJORITY_MIN_COUNT and count / total >= MAJORITY_MIN_RATIO:
            return target
        return None

    def pending_cards(self, session_id: str, *, for_chain: str) -> list[CandidateCard]:
        """Other pending chains offered as candidates, so a broken chain can be
        merged back when the person returns.

        Chains that have appeared in the same frame cannot physically be the same
        person, so they are excluded.
        """
        cards: list[CandidateCard] = []
        for chain in self.store.pending_chains(session_id):
            ref = chain["chain_ref"]
            if ref == self.store.canonical_chain(for_chain) or self.copresent(ref, for_chain):
                continue
            pair = self.store.best_pair(ref) or {}
            voice = self.store.best_voice(ref) or {}
            names = self.store.names_for(ref)
            cards.append(CandidateCard(
                character_id=ref, name=names[0] if names else "",
                desc=chain["desc_text"] or "",
                face_b64=self._b64(str(pair.get("face_oss_key") or "")),
                body_b64=self._b64(str(pair.get("body_oss_key") or "")),
                voice_b64=self._b64(str(voice.get("oss_key") or "")),
                last_seen_session=session_id))
        return cards


# ── The chain-level collision rule: same-frame chains sharing a hypothesis.
#    Detect, re-arbitrate, then degrade with evidence taking priority ──────
def _chain_proposed(book: ChainBook, script: ClipScript, session_id: str) -> dict[str, str]:
    from personos.identity.inspect import present_casts
    proposed: dict[str, str] = {}
    for cast_id in present_casts(script):
        canonical = book.store.canonical_chain(book.store.chain_ref(session_id, cast_id))
        chain = book.store.get_chain(canonical)
        if chain is None:
            continue
        proposed[cast_id] = (chain["hypothesis"] if chain["hypothesis"] != "NEW" else canonical)
    return proposed


def resolve_chain_collisions(book: ChainBook, script: ClipScript, *, session_id: str,
                             clip_index: int, queries_by_cast: dict[str, dict[str, Any]],
                             pool: list[CandidateCard], omni: Any,
                             dump: Any = None) -> list[dict[str, Any]]:
    """Detect same-frame chains sharing a hypothesis, re-arbitrate (at most two
    rounds), then degrade with evidence taking priority.
    """
    from personos.identity.inspect import inspect_bind_collisions
    from personos.identity.recognize import build_arbitration_prompt, parse_verdicts
    log: list[dict[str, Any]] = []

    def _violations() -> list:
        vs = inspect_bind_collisions(script, _chain_proposed(book, script, session_id))
        # Two casts colliding on the same canonical chain is not a violation — that
        # is one chain recurring — so filter out groups whose casts all map to the
        # same canonical.
        return [v for v in vs if len({
            book.store.canonical_chain(book.store.chain_ref(session_id, c))
            for c in v.cast_ids}) > 1]

    violations = _violations()
    if not violations:
        return log
    candidate_ids = [card.character_id for card in pool]

    # Protect the stronger side: when exactly one chain holds a strong majority on
    # the contested profile, that is the identity the evidence best supports, so it
    # is neither re-arbitrated nor degraded.
    def _contested_target(violation):
        proposed = _chain_proposed(book, script, session_id)
        targets = {proposed.get(c) for c in violation.cast_ids}
        targets.discard(None)
        return next(iter(targets)) if len(targets) == 1 else None

    def _protected(violation) -> set[str]:
        target = _contested_target(violation)
        if not target or book.store.is_chain_ref(target):
            return set()
        strong = [c for c in violation.cast_ids
                  if book.strong_majority(book.store.chain_ref(session_id, c)) == target]
        return set(strong) if len(strong) == 1 else set()

    for attempt in (1, 2):
        conflicted = [c for v in violations for c in v.cast_ids
                      if c in queries_by_cast and c not in _protected(v)]
        conflicted = list(dict.fromkeys(conflicted))
        if not conflicted or not pool:
            break
        prompt, images, audios = build_arbitration_prompt(
            [queries_by_cast[c] for c in conflicted], pool)
        prompt += ("\n\nCONSTRAINT VIOLATION FEEDBACK:\n"
                   + "\n".join(f"- {v.detail}" for v in violations)
                   + "\nPeople appearing together in the same clip are physically"
                     " distinct people — they can never be the same character."
                     " Re-judge ONLY the queries above: pick a different registered"
                     " character, or NEW if unsure.")
        if dump:
            dump(f"s4_chain_rearbitration_prompt_a{attempt}.txt", prompt)
        try:
            raw = omni.chat(prompt, images_b64=images, audio_b64_list=audios, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            log.append({"kind": "chain_bind", "attempt": attempt,
                        "outcome": "call_failed", "error": str(exc)})
            break
        if dump:
            dump(f"s4_chain_rearbitration_raw_a{attempt}.txt", raw)
        verdicts, issues = parse_verdicts(raw, cast_ids=conflicted, candidate_ids=candidate_ids)
        for cast_id, verdict in verdicts.items():
            book.apply_verdict(book.store.chain_ref(session_id, cast_id), verdict,
                               session_id=session_id, clip_index=clip_index,
                               reason="collision_rearbitration", issues=issues)
        log.append({"kind": "chain_bind", "attempt": attempt,
                    "outcome": "rearbitrated", "verdicts": verdicts})
        violations = _violations()
        if not violations:
            return log
    # Degrade with evidence taking priority: keep the strong-majority chain, or
    # failing that the one with the highest best_face_q, breaking ties by who
    # appeared earlier. Everything else degrades to NEW, because a wrong split can
    # be recovered from and a wrong merge cannot.
    for violation in violations:
        chains = {c: book.store.get_chain(book.store.canonical_chain(
            book.store.chain_ref(session_id, c))) for c in violation.cast_ids}
        protected = _protected(violation)
        keep = (next(iter(protected)) if protected else
                max(chains, key=lambda c: (chains[c]["best_face_q"],
                                           -min(chains[c]["presence"] or [10**9]))))
        demoted = []
        for cast_id, chain in chains.items():
            if cast_id == keep:
                continue
            book.apply_verdict(chain["chain_ref"], "NEW", session_id=session_id,
                               clip_index=clip_index, reason="collision_degrade",
                               issues=[violation.detail])
            demoted.append(cast_id)
        log.append({"kind": "chain_bind", "outcome": "degraded", "keep": keep,
                    "demoted": demoted, "detail": violation.detail})
    return log
