"""Evidence storage: append-only and immutable. This is the single source of truth.

The rule is absolute: insert only, never update or delete. Deletion goes through
the governance layer's forget, and is not something this store does day to day.

For multi-tenancy, an instance is bound to one user through the constructor, and
every read and write is confined to that user automatically. The caller receives a
store that is already that user's, so there is no surface on which someone can
"forget to pass user_id" and leak. user_id="" is the default namespace, kept for
compatibility with older databases and single-tenant use.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from loguru import logger

from ..models import EvidenceRecord
from .db import Database, blob_param


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def append(self, rec: EvidenceRecord, embedding: Optional[np.ndarray] = None) -> str:
        """Append one piece of evidence.

        If that sha256 already exists for this user it counts as a duplicate and
        the existing id is returned, which makes this deduplicating and idempotent.

        embedding is the vector of the evidence text, used for the evidence-level
        semantic fallback search when the fact layer finds nothing.
        """
        if not rec.sha256 and rec.content_inline is not None:
            rec.sha256 = sha256_of(rec.content_inline)
        existing = self.by_sha256(rec.sha256) if rec.sha256 else None
        if existing:
            logger.debug(f"evidence deduplicated sha={rec.sha256[:8]} -> {existing.id}")
            return existing.id
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        self.db.execute(
            "INSERT INTO evidence(id, user_id, sha256, holder, modality, captured_at, payload, embedding) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,UNHEX(%s))",
            (rec.id, self.user_id, rec.sha256, rec.holder, rec.modality,
             rec.captured_at.isoformat(), rec.model_dump_json(), blob_param(emb_blob)),
        )
        logger.info(f"evidence stored id={rec.id} user={self.user_id or '(default)'} holder={rec.holder} len={len(rec.content_inline or '')}")
        return rec.id

    def all_with_embeddings(self) -> list[tuple[EvidenceRecord, np.ndarray]]:
        """All of this user's evidence that carries a vector, for batch scoring in the fallback search."""
        rows = self.db.fetch_all(
            "SELECT payload, HEX(embedding) AS emb FROM evidence "
            "WHERE user_id=%s AND embedding IS NOT NULL",
            (self.user_id,),
        )
        return [
            (EvidenceRecord.model_validate_json(r["payload"]),
             np.frombuffer(bytes.fromhex(r["emb"]), dtype=np.float32))
            for r in rows
        ]

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.db.fetch_one(
            "SELECT payload FROM evidence WHERE id=%s AND user_id=%s", (evidence_id, self.user_id)
        )
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def by_sha256(self, sha: str) -> EvidenceRecord | None:
        row = self.db.fetch_one(
            "SELECT payload FROM evidence WHERE sha256=%s AND user_id=%s LIMIT 1", (sha, self.user_id)
        )
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def list(self, limit: int = 100) -> list[EvidenceRecord]:
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s ORDER BY captured_at DESC LIMIT %s",
            (self.user_id, limit),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]

    def reply_for(self, user_evidence_id: str) -> EvidenceRecord | None:
        """The assistant evidence that replies to a given piece of user evidence,
        identified by its source.reply_to. Used to pair question with answer when
        showing retrieval results.

        holder is a real column and can be filtered directly, but reply_to lives
        inside payload, so we scan a window of recent assistant evidence and
        compare. That is sufficient at the current scale.
        """
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s AND holder='assistant' "
            "ORDER BY captured_at DESC LIMIT 500",
            (self.user_id,),
        )
        for r in rows:
            rec = EvidenceRecord.model_validate_json(r["payload"])
            if (rec.source or {}).get("reply_to") == user_evidence_id:
                return rec
        return None

    def in_session(self, session_id: str, limit: int = 20) -> list[EvidenceRecord]:
        """Evidence from the same session, oldest to newest, so expand can pull in neighbouring raw material."""
        rows = self.by_session(session_id)
        return rows[:limit]

    def session_stats(self) -> dict[str, dict]:
        """Per-session evidence counts for this user, plus the time of the most
        recent piece as an ISO string, or an empty string if there is none.

        This is what lets a console list sessions by recent activity, so when
        something is being diagnosed the session that just ran sorts first.
        """
        counts: dict[str, dict] = {}
        for rec in self.iter_all():
            sid = (rec.source or {}).get("session_id")
            if not sid:
                continue
            st = counts.setdefault(sid, {"count": 0, "last_at": ""})
            st["count"] += 1
            if rec.captured_at:
                iso = rec.captured_at.isoformat()
                if iso > st["last_at"]:
                    st["last_at"] = iso
        return counts

    def by_session(self, session_id: str) -> list[EvidenceRecord]:
        """**All** of this session's evidence, oldest to newest.

        Session context compression needs the entire history in view before it can
        roll the older part up.
        """
        recs = [r for r in self.iter_all() if (r.source or {}).get("session_id") == session_id]
        recs.sort(key=lambda r: r.captured_at)
        return recs

    def iter_all(self) -> list[EvidenceRecord]:
        """All of this user's evidence, in time order.

        Fine at the current scale; once one user's memory grows large this will
        need pagination and a column index.
        """
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s ORDER BY captured_at ASC, id ASC",
            (self.user_id,),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]

    def search_keyword(self, keywords: list[str], *, holder: str = "", limit: int = 30) -> list[EvidenceRecord]:
        """Search the raw utterances by keyword, as a fallback path. Every keyword
        must hit **the same utterance**, and results come back in time order.

        This is the only retrieval path that depends on no index at all: when atom
        extraction misses something, the original words are still in the truth
        layer and reachable from here.

        The LIKE runs against the payload JSON. Ordinary words are stored verbatim,
        but a term containing quotes or backslashes will not match.
        """
        kws = [k.strip() for k in keywords if k and k.strip()]
        if not kws:
            return []
        conds = ["payload LIKE %s ESCAPE '!'"] * len(kws)
        if holder:
            conds.append("holder = %s")
        params: list = [f"%{_like_escape(k)}%" for k in kws]
        if holder:
            params.append(holder)
        rows = self.db.fetch_all(
            f"SELECT payload FROM evidence WHERE user_id=%s AND {' AND '.join(conds)} "
            "ORDER BY captured_at ASC, id ASC LIMIT %s",
            (self.user_id, *params, limit),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]


def _like_escape(s: str) -> str:
    """Escape the LIKE wildcards (!, %, _) to pair with ESCAPE '!'.

    A single-character escape is used because it is unaffected by whatever
    sql_mode decides backslashes mean.
    """
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")
