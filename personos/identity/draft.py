"""Session drafts: session-scoped state for identity chains, the roster, the
evaluation ledger and staged assets.

The chain/roster machinery is a union-find over canonical pointers, with presence
accumulation, an append-only evaluation ledger, and a commit_chain marker. All
four kinds of session state live in Redis, keyed through redis_client.key() with a
sliding TTL; sqlite is not allowed here. The persistent layer remains the
MySQL-backed CharacterStore.

Assets are handled deliberately differently from the rest: a crop is uploaded to
OSS the moment harvest produces it, and the draft keeps only the oss_key, the
normalized vector and the quality. Nothing is inlined as b64, so Redis does not
balloon; arbitration cards fetch b64 from OSS on demand. At commit time the
winner's staged assets become character_assets rows and feed cloud.learn (see
commit.py).

Storage layout: one STRING key per session holding the whole state JSON (chains,
evals, roster, staged). Read-modify-write happens inside the session write lock,
which identity processing already holds, and the write itself is a single
SET ... EX command so value and TTL land atomically. We avoid pipelines and Lua
here for the same reason seg_store does: the Redis proxy we run behind mishandles
them. One key and one command means no multi-key access, so no hash tag is needed.

The governing principle: a chain only ever holds the *current hypothesis*. Global
attribution is deferred to end-of-session final adjudication in commit.py.
canonical=None marks a chain root, aliases point at the root (merging flattens
them to one level deep), and pending means uncommitted and a root.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

import numpy as np
from loguru import logger

from personos.identity.types import CastEvidence
from personos.storage.redis_client import key as _redis_key

# Sliding TTL for the session draft, renewed on every write. If a session goes a
# whole day without another clip the draft expires. Losing uncommitted chains that
# way is acceptable: final adjudication only handles pending chains that are still
# there, and a re-run is the fallback. Same order of magnitude as seg_store's
# SEG_TTL_S.
IDDRAFT_TTL_S = 24 * 3600

# The whitelist of chain fields that may be updated.
_CHAIN_FIELDS = frozenset({"canonical", "status", "hypothesis", "hypo_method", "best_face_q",
                           "best_voice_q", "named", "desc_text", "presence", "final_character_id"})


def read_b64(media_store: Any, oss_key: str) -> str:
    """Read an asset out of OSS as b64.

    A failed read returns an empty string rather than raising: with one angle
    missing the decision can still be made from the others, so this must not
    block. Shared by everything in the identity layer that builds cards
    (candidate cards, query cards, roster), and kept here in the lowest-level
    draft module so there is only one copy.
    """
    if not oss_key or media_store is None:
        return ""
    try:
        return base64.b64encode(media_store.read_bytes(oss_key)).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"failed to read asset {oss_key}: {e}")
        return ""


def _emb_list(emb: Optional[np.ndarray]) -> Optional[list[float]]:
    return None if emb is None else np.asarray(emb, dtype=np.float32).tolist()


def _emb_arr(x: Any) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float32)


class DraftStore:
    """Base class for session-draft access.

    All the chain, roster, evaluation and asset logic lives here as pure
    operations on a state dict; subclasses supply the IO through _read/_write.
    An instance is bound to one user_id, fencing it to that user's own social
    circle just as CharacterStore does.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    # ── Implemented by subclasses: read/write the whole state, isolated per session ──
    def _read(self, session_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def _blank_state() -> dict[str, Any]:
        return {"chains": {}, "evals": {}, "roster": {}, "staged": {}, "lines": [],
                "clip_seq": 0, "clip_keys": {}, "clips_done": {}}

    def next_clip_seq(self, session_id: str, clip_key: str = "") -> int:
        """Hand out the next session-global monotonic clip sequence number (one per
        video message) and record clip_index -> oss_key.

        The consuming side takes its clip_index from here rather than trusting the
        caller. An awkward caller that interleaves text and clips (text -> clips ->
        text -> clips) would renumber from 0 on each batch, and those repeated
        indices would collide in presence and lines. A session-global counter
        makes that impossible.
        """
        state = self._read(session_id)
        seq = int(state.get("clip_seq", 0))
        state["clip_seq"] = seq + 1
        if clip_key:
            state.setdefault("clip_keys", {})[str(seq)] = clip_key
        self._write(session_id, state)
        return seq

    def clip_keys(self, session_id: str) -> dict[int, str]:
        """clip_index -> oss_key within the session, so a flush can link raw_clip evidence back."""
        return {int(k): v for k, v in self._read(session_id).get("clip_keys", {}).items()}

    # ── Per-clip idempotency ──────────────────────────────────────
    # A queue redelivery replays the **whole message**, and one message carries up
    # to 20 clips, while processing a clip accumulates presence, lines and staged
    # assets. Without deduplication, a transient failure on the 19th clip of a
    # batch re-runs the first 18: appearances counted twice, dialogue entering a
    # memcell twice, assets uploaded twice. That is not wasted compute, it is
    # **remembering something false**.
    #
    # The dedup marker has to hang off "finished", not off the sequence allocation
    # in next_clip_seq. The sequence number is handed out *before* the screenplay
    # model runs, so keying on it would make a clip whose screenplay failed look
    # already-processed on redelivery and get skipped — trading double-counting for
    # silently losing the memory, which is worse. Hence a separate record of
    # clip_key -> the clip_index it completed as.
    def clip_done_index(self, session_id: str, clip_key: str) -> Optional[int]:
        """Whether this clip has already been processed **to completion**.

        Returns the clip_index it had at the time, or None.
        """
        if not clip_key:
            return None
        v = self._read(session_id).get("clips_done", {}).get(clip_key)
        return int(v) if v is not None else None

    def mark_clip_done(self, session_id: str, clip_key: str, clip_index: int) -> None:
        """Call once every stage of the clip has landed in the draft; a redelivery of the same key then skips it."""
        if not clip_key:
            return
        state = self._read(session_id)
        state.setdefault("clips_done", {})[clip_key] = int(clip_index)
        self._write(session_id, state)

    # ── The chain_ref convention ─────────────────────────────────────
    @staticmethod
    def chain_ref(session_id: str, cast_id: str) -> str:
        return f"chain:{session_id}:{cast_id}"

    @staticmethod
    def is_chain_ref(ref: str) -> bool:
        return ref.startswith("chain:")

    @staticmethod
    def _session_of(chain_ref: str) -> str:
        parts = chain_ref.split(":")
        return parts[1] if len(parts) == 3 and parts[0] == "chain" else ""

    @staticmethod
    def _default_chain(session_id: str, cast_id: str) -> dict[str, Any]:
        """The initial state of a new chain."""
        return {"chain_ref": DraftStore.chain_ref(session_id, cast_id),
                "session_id": session_id, "cast_id": cast_id,
                "canonical": None, "status": "pending", "hypothesis": "NEW",
                "hypo_method": "first_seen", "best_face_q": -1.0, "best_voice_q": -1.0,
                "named": 0, "desc_text": "", "presence": [], "final_character_id": None}

    # ── chains ────────────────────────────────────────────────────────
    def ensure_chain(self, session_id: str, cast_id: str) -> tuple[dict[str, Any], bool]:
        ref = self.chain_ref(session_id, cast_id)
        state = self._read(session_id)
        row = state["chains"].get(ref)
        if row is not None:
            return dict(row), False
        row = self._default_chain(session_id, cast_id)
        state["chains"][ref] = row
        self._write(session_id, state)
        return dict(row), True

    def get_chain(self, chain_ref: str) -> Optional[dict[str, Any]]:
        row = self._read(self._session_of(chain_ref))["chains"].get(chain_ref)
        return dict(row) if row else None

    def canonical_chain(self, chain_ref: str) -> str:
        """Follow canonical pointers to the chain root, guarding against cycles.

        merge_chain already refuses to create one; this is the second line of defence.
        """
        chains = self._read(self._session_of(chain_ref))["chains"]
        seen: set[str] = set()
        current = chain_ref
        while current not in seen:
            seen.add(current)
            row = chains.get(current)
            if row is None or not row.get("canonical"):
                return current
            current = row["canonical"]
        return current

    def update_chain(self, chain_ref: str, **fields: Any) -> None:
        unknown = set(fields) - _CHAIN_FIELDS
        if unknown:
            raise ValueError(f"unknown chain fields: {sorted(unknown)}")
        session_id = self._session_of(chain_ref)
        state = self._read(session_id)
        row = state["chains"].get(chain_ref)
        if row is None:
            raise ValueError(f"update unknown chain: {chain_ref}")
        row.update(fields)
        self._write(session_id, state)

    def merge_chain(self, src_ref: str, dst_ref: str) -> None:
        """Merge src into dst.

        src.canonical is set to dst's root and src's existing aliases are flattened
        onto that root too, keeping the structure one level deep — commit's
        single-level collection depends on that. The evidence (best_*, named,
        presence) is merged into the dst root.
        """
        dst = self.canonical_chain(dst_ref)
        if self.canonical_chain(src_ref) == dst:
            return
        src_row, dst_row = self.get_chain(src_ref), self.get_chain(dst)
        if src_row is None or dst_row is None:
            raise ValueError(f"merge unknown chain: {src_ref} -> {dst_ref}")
        session_id = self._session_of(src_ref)
        state = self._read(session_id)
        state["chains"][src_ref]["canonical"] = dst
        for ref, row in state["chains"].items():
            if row.get("canonical") == src_ref:
                row["canonical"] = dst
        dst_state_row = state["chains"][dst]
        dst_state_row["best_face_q"] = max(src_row["best_face_q"], dst_row["best_face_q"])
        dst_state_row["best_voice_q"] = max(src_row["best_voice_q"], dst_row["best_voice_q"])
        dst_state_row["named"] = max(src_row["named"], dst_row["named"])
        dst_state_row["presence"] = sorted(set(src_row["presence"]) | set(dst_row["presence"]))
        self._write(session_id, state)

    def aliases_of(self, chain_ref: str) -> list[str]:
        chains = self._read(self._session_of(chain_ref))["chains"]
        return sorted(ref for ref, row in chains.items() if row.get("canonical") == chain_ref)

    def pending_chains(self, session_id: str) -> list[dict[str, Any]]:
        """Chains that are uncommitted and are roots (canonical=None), ordered by cast_id."""
        chains = self._read(session_id)["chains"]
        rows = [dict(r) for r in chains.values()
                if r["status"] == "pending" and not r.get("canonical")]
        return sorted(rows, key=lambda r: r["cast_id"])

    def commit_chain(self, canonical_ref: str, final_character_id: str) -> None:
        """The draft side of a final-adjudication commit: mark the chain and its
        aliases committed and record final_character_id.

        Actually persisting assets and names is commit.py's job, through
        CharacterStore.
        """
        session_id = self._session_of(canonical_ref)
        state = self._read(session_id)
        for ref in (canonical_ref, *[r for r, row in state["chains"].items()
                                     if row.get("canonical") == canonical_ref]):
            row = state["chains"].get(ref)
            if row is not None:
                row["status"] = "committed"
                row["final_character_id"] = final_character_id
        self._write(session_id, state)

    # ── The chain evaluation ledger, append-only ────────────────────
    def add_chain_evaluation(self, chain_ref: str, *, session_id: str, clip_index: int,
                             reason: str, verdict: Optional[str],
                             evidence: Optional[dict[str, Any]] = None,
                             issues: Optional[list[str]] = None) -> None:
        state = self._read(session_id)
        state["evals"].setdefault(chain_ref, []).append(
            {"chain_ref": chain_ref, "session_id": session_id, "clip_index": clip_index,
             "reason": reason, "verdict": verdict, "evidence": evidence or {},
             "issues": issues or []})
        self._write(session_id, state)

    def evaluations_for(self, chain_ref: str) -> list[dict[str, Any]]:
        return list(self._read(self._session_of(chain_ref))["evals"].get(chain_ref, []))

    # ── The roster: this session's continuation register ────────────
    def load_roster(self, session_id: str) -> dict[str, dict[str, Any]]:
        roster: dict[str, dict[str, Any]] = {}
        for cast_id, entry in self._read(session_id)["roster"].items():
            card = dict(entry.get("card") or {})
            card["character_id"] = entry.get("character_id")
            roster[cast_id] = card
        return roster

    def names_for(self, chain_ref: str) -> list[str]:
        """The chain's name within this session, read off the roster card
        (chain_ref -> cast_id -> roster[cast_id].name).

        Names on the persistent profile live in CharacterStore.names_for; the
        chain's name is merged into the profile at end-of-session by commit.py.
        """
        cast_id = chain_ref.split(":")[-1]
        card = self._read(self._session_of(chain_ref))["roster"].get(cast_id, {}).get("card") or {}
        name = card.get("name")
        return [name] if name else []

    def save_roster_entry(self, session_id: str, cast_id: str, *,
                          character_id: Optional[str], card: dict[str, Any]) -> None:
        payload = {k: v for k, v in card.items() if k != "character_id"}
        state = self._read(session_id)
        state["roster"][cast_id] = {"character_id": character_id, "card": payload}
        self._write(session_id, state)

    # ── The screenplay line buffer. At end-of-session it is flushed into
    #    evidence carrying person attribution; see online/video_memory ──
    def stage_lines(self, session_id: str, clip_index: int,
                    lines: list[tuple[float, float, str, str, str]]) -> None:
        """Buffer one clip's screenplay lines.

        lines is [(t0, t1, who, kind, text)], where who is the already-mapped
        session cast id (cast_map.get(line.who)), or 'SW', or 'ENV' for
        environment lines. Lines append within the session, and their order is
        determined by clip_index plus t0.
        """
        state = self._read(session_id)
        for t0, t1, who, kind, text in lines:
            state["lines"].append({"clip": clip_index, "t0": float(t0), "t1": float(t1),
                                   "who": who, "kind": kind, "text": text})
        self._write(session_id, state)

    def all_lines(self, session_id: str) -> list[dict[str, Any]]:
        """All of the session's screenplay lines in (clip_index, t0) order, for the commit flush."""
        return sorted(self._read(session_id)["lines"], key=lambda r: (r["clip"], r["t0"]))

    # ── Staged assets: crops go to OSS, the draft keeps only key, vector and quality ──
    def stage_evidence(self, chain_ref: str, *, session_id: str, clip_index: int,
                       evidence: CastEvidence, media_store: Any) -> None:
        """Stage one cast's evidence assets from this clip: the crop is uploaded to
        OSS immediately and the metadata (oss_key, vector, q, t) goes into the draft.

        This is what arbitration cards draw their assets from, and what the winner
        learns into the cloud and persists from at final adjudication (commit.py).
        """
        state = self._read(session_id)
        st = state["staged"].setdefault(chain_ref, {"face": [], "body": [], "voice": []})
        for f in evidence.faces:
            if f.embedding is None and not f.crop_b64:
                continue
            st["face"].append({
                "oss_key": self._save_img(media_store, f.crop_b64, "image/png"),
                "emb": _emb_list(f.embedding), "q": float(f.q), "t": float(f.t),
                "clip": clip_index, "session": session_id, "descriptor": f.descriptor})
            body_key = self._save_img(media_store, f.body_crop_b64, "image/jpeg")
            if body_key:
                st["body"].append({"oss_key": body_key, "q": float(f.q), "t": float(f.t),
                                   "clip": clip_index, "session": session_id})
        for v in evidence.voices:
            if v.embedding is None:
                continue
            st["voice"].append({
                "oss_key": self._save_audio(media_store, v.wav_bytes),
                "emb": _emb_list(v.embedding), "q": float(v.q),
                "t0": float(v.t0), "t1": float(v.t1),
                "clip": clip_index, "session": session_id})
        self._write(session_id, state)

    def active_staged(self, chain_ref: str, kind: str) -> list[dict[str, Any]]:
        """Staged assets of one kind, ordered by q descending.

        Used by final adjudication for cloud learning and persistence; emb is
        restored to a numpy vector.
        """
        items = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {}).get(kind, [])
        out = [{**it, "embedding": _emb_arr(it.get("emb"))} for it in items]
        return sorted(out, key=lambda a: a["q"], reverse=True)

    def best_pair(self, chain_ref: str) -> Optional[dict[str, Any]]:
        """The highest-quality face plus the best body shot, as arbitration card material.

        Returns oss_keys; the card side then fetches the b64 from OSS.
        """
        staged = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {})
        faces = staged.get("face") or []
        bodies = staged.get("body") or []
        if not faces and not bodies:
            return None
        best_face = max(faces, key=lambda a: a["q"], default=None)
        best_body = max(bodies, key=lambda a: a["q"], default=None)
        return {"face_oss_key": (best_face or {}).get("oss_key", ""),
                "body_oss_key": (best_body or {}).get("oss_key", ""),
                "quality": (best_face or best_body or {}).get("q", -1.0)}

    def best_voice(self, chain_ref: str) -> Optional[dict[str, Any]]:
        voices = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {}).get("voice") or []
        if not voices:
            return None
        best = max(voices, key=lambda a: a["q"])
        return {"oss_key": best.get("oss_key", ""), "quality": best["q"]}

    # ── Landing in OSS. The crop goes up immediately, and a failure does not
    #    block: the vector is still in the draft ─────────────────────────
    def _save_img(self, media_store: Any, b64: str, content_type: str) -> str:
        if media_store is None or not b64:
            return ""
        try:
            return media_store.save_image(base64.b64decode(b64), owner=self.user_id,
                                          content_type=content_type).key
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to store draft asset in the object store: {e}")
            return ""

    def _save_audio(self, media_store: Any, wav_bytes: Optional[bytes]) -> str:
        if media_store is None or not wav_bytes:
            return ""
        try:
            return media_store.save_audio(wav_bytes, owner=self.user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to store draft voiceprint wav in the object store: {e}")
            return ""


class RedisDraftStore(DraftStore):
    """The production implementation: one STRING key per session on Redis, holding
    the whole state JSON.

    key = {env}:personos:iddraft:{user}:{session}, written with a single
    SET ... EX so the value and its TTL land atomically. No pipelines and no Lua,
    for the same reason as seg_store: the Redis proxy we run behind can misframe
    pipelined commands and cross responses between them.
    """

    def __init__(self, client, user_id: str, ttl_s: int = IDDRAFT_TTL_S) -> None:
        super().__init__(user_id)
        self._c = client
        self._ttl = ttl_s

    def _k(self, session_id: str) -> str:
        return _redis_key("iddraft", self.user_id, session_id)

    def _read(self, session_id: str) -> dict[str, Any]:
        raw = self._c.get(self._k(session_id))
        return json.loads(raw) if raw else self._blank_state()

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        self._c.set(self._k(session_id), json.dumps(state, ensure_ascii=False), ex=self._ttl)

    def clear(self, session_id: str) -> None:
        self._c.delete(self._k(session_id))


class MemoryDraftStore(DraftStore):
    """The in-process implementation, for local scripts, unit tests, and single-replica
    deployments with no Redis configured. State is held in memory per (user, session).
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(user_id)
        self._d: dict[str, dict[str, Any]] = {}

    def _read(self, session_id: str) -> dict[str, Any]:
        return self._d.setdefault(session_id, self._blank_state())

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        self._d[session_id] = state

    def clear(self, session_id: str) -> None:
        self._d.pop(session_id, None)
