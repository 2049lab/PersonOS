"""MemCell storage: the organizing unit at segment granularity.

The topic vector gets its own column and forms the segment-level retrieval
surface, used by find_cells and as the header of rerank material.

Multi-tenancy works as it does in AtomStore: an instance is bound to one user.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from ..models import MemCell, ensure_aware
from .db import Database, blob_of, blob_param

# The VALUES(col) form works on 5.6, 5.7 and 8.0 alike, as in atom_store, and
# topic_embedding travels through the UNHEX hex channel for the same reason.
_UPSERT_SQL = (
    "INSERT INTO memcells(id, user_id, session_id, t_start, t_end, episode_type, payload, topic_embedding) "
    "VALUES(%s,%s,%s,%s,%s,%s,%s,UNHEX(%s)) "
    "ON DUPLICATE KEY UPDATE session_id=VALUES(session_id), t_start=VALUES(t_start), "
    "t_end=VALUES(t_end), episode_type=VALUES(episode_type), payload=VALUES(payload), "
    "topic_embedding=VALUES(topic_embedding)"
)


class CellStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def upsert(self, cell: MemCell, topic_embedding: Optional[np.ndarray] = None) -> str:
        """Insert a cell or replace it wholesale, which is what remember does when
        it revises an episode. When embedding is None the existing vector is kept.
        """
        emb_blob = topic_embedding
        if emb_blob is None:
            row = self.db.fetch_one(
                "SELECT HEX(topic_embedding) AS emb FROM memcells WHERE id=%s AND user_id=%s",
                (cell.id, self.user_id),
            )
            emb_blob = blob_of(row["emb"]) if row else None
        else:
            emb_blob = np.asarray(emb_blob, dtype=np.float32).tobytes()
        self.db.execute(_UPSERT_SQL, (
            cell.id, self.user_id, cell.session_id,
            cell.t_start.isoformat() if cell.t_start else None,
            cell.t_end.isoformat() if cell.t_end else None,
            cell.episode_type or "unknown",
            cell.model_dump_json(), blob_param(emb_blob),
        ))
        logger.info(f"cell upsert id={cell.id} session={cell.session_id} "
                    f"user={self.user_id or '(default)'} topic={cell.topic[:40]!r}")
        return cell.id

    def get(self, cell_id: str) -> MemCell | None:
        row = self.db.fetch_one(
            "SELECT payload FROM memcells WHERE id=%s AND user_id=%s", (cell_id, self.user_id)
        )
        return MemCell.model_validate_json(row["payload"]) if row else None

    def list_session(self, session_id: str) -> list[MemCell]:
        """Every cell of one session, by segment start time oldest to newest, which is conversational order."""
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s AND session_id=%s "
            "ORDER BY t_start, id",
            (self.user_id, session_id),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def count_session(self, session_id: str) -> int:
        """How many cells this session has produced.

        This is the authoritative count across redeploys and replicas, replacing a
        counter accumulated in memory.
        """
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s AND session_id=%s",
            (self.user_id, session_id),
        )
        return int(row["n"]) if row else 0

    def iter_all(self, limit: int = 1000) -> list[MemCell]:
        """All of this user's cells, by segment start time oldest to newest."""
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s ORDER BY t_start, id LIMIT %s",
            (self.user_id, limit),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def cells_after(self, up_to_cell_id: str = "") -> list[MemCell]:
        """All of this user's cells after the cursor cell, under the total order on
        (t_start, id), oldest to newest.

        This is how profile consolidation picks up the cells added since the last
        published version. An empty cursor means all of them. A cursor cell that no
        longer exists — deleted by forget, say — also degrades to all of them,
        triggering a full re-distillation, which is consistent with a profile being
        recompilable.

        Ordering on the (t_start, id) tuple means id breaks ties when two cells
        share a t_start, so nothing is skipped and nothing repeats. t_start is an
        ISO string, where lexical order equals chronological order.
        """
        if not up_to_cell_id:
            return self.iter_all()
        cur = self.get(up_to_cell_id)
        if cur is None or cur.t_start is None:
            return self.iter_all()
        ct = cur.t_start.isoformat()
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s "
            "AND (t_start > %s OR (t_start = %s AND id > %s)) ORDER BY t_start, id",
            (self.user_id, ct, ct, up_to_cell_id),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def _type_where(self, episode_type: Optional[str], start: Optional[str], end: Optional[str]):
        """Build the WHERE clause from user plus optional type and time range.

        Filter values always go through bound parameters and the clause names are
        fixed, so there is no injection surface.
        """
        clauses, params = ["user_id=%s"], [self.user_id]
        if episode_type:
            clauses.append("episode_type=%s"); params.append(episode_type)
        if start:
            clauses.append("t_start >= %s"); params.append(start)      # t_start holds an ISO string, so lexical order is chronological order
        if end:
            clauses.append("t_start <= %s"); params.append(end)
        return " AND ".join(clauses), params

    def list_by_type(self, *, episode_type: Optional[str] = None, start: Optional[str] = None,
                     end: Optional[str] = None, limit: int = 20, offset: int = 0) -> list[MemCell]:
        """Page through this user's cells by episode_type and time range, newest
        t_start first. Every filter is optional; limit and offset are required.
        """
        where, params = self._type_where(episode_type, start, end)
        rows = self.db.fetch_all(
            f"SELECT payload FROM memcells WHERE {where} ORDER BY t_start DESC, id DESC LIMIT %s OFFSET %s",
            (*params, limit, offset),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def count_by_type(self, *, episode_type: Optional[str] = None, start: Optional[str] = None,
                      end: Optional[str] = None) -> int:
        """The total under the same filters as list_by_type, for pagination."""
        where, params = self._type_where(episode_type, start, end)
        row = self.db.fetch_one(f"SELECT COUNT(*) AS n FROM memcells WHERE {where}", tuple(params))
        return int(row["n"]) if row else 0

    def all_with_embeddings(self) -> list[tuple[MemCell, np.ndarray]]:
        """All of this user's cells that carry a topic vector, for batch scoring in find_cells."""
        rows = self.db.fetch_all(
            "SELECT payload, HEX(topic_embedding) AS emb FROM memcells "
            "WHERE user_id=%s AND topic_embedding IS NOT NULL",
            (self.user_id,),
        )
        return [
            (MemCell.model_validate_json(r["payload"]),
             np.frombuffer(blob_of(r["emb"]), dtype=np.float32))
            for r in rows
        ]

    def in_window(self, cells: list[MemCell], start: Optional[datetime], end: Optional[datetime]) -> list[MemCell]:
        """Filter by segment start time.

        At this scale filtering in Python is fine. An omitted bound means that side
        is unbounded.
        """
        s, e = ensure_aware(start), ensure_aware(end)
        out = []
        for c in cells:
            t = ensure_aware(c.t_start)
            if t is None:
                continue          # a cell with no time metadata stays out of the window rather than being guessed into it
            if s and t < s:
                continue
            if e and t > e:
                continue
            out.append(c)
        return out
