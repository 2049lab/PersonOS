"""Atom storage: the retrieval unit that points at a cell, with upsert semantics.

Vectors get their own column so MaxSim can score them in bulk.

For multi-tenancy, an instance is bound to one user through the constructor and
every read and write is confined to that user. user_id="" is the default
namespace, kept for compatibility with older databases and single-tenant use.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from ..models import MemoryAtom
from .db import Database, blob_of, blob_param

# The VALUES(col) form works on 5.6, 5.7 and 8.0 alike. One database proxy we
# deploy behind announces the 5.6 protocol, so the 8.0.19+ "AS new" alias is not
# available to us.
# embedding travels through the UNHEX hex channel, because that same proxy is not
# binary-safe for _binary literals; see db.blob_param.
# The three chain columns appear in the INSERT but deliberately not in the
# ON DUPLICATE KEY UPDATE, so replaying an upsert cannot wipe out a chain
# attribution. Nothing collides today — the write path and remember both insert
# new atoms — so this is structural insurance for future replay paths.
_UPSERT_SQL = (
    "INSERT INTO atoms(id, user_id, memcell_id, chain_id, prev_atom_id, next_atom_id, "
    "object_type, holder, recorded_at, updated_at, payload, embedding) "
    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,UNHEX(%s)) "
    "ON DUPLICATE KEY UPDATE memcell_id=VALUES(memcell_id), "
    "object_type=VALUES(object_type), holder=VALUES(holder), "
    "recorded_at=VALUES(recorded_at), updated_at=VALUES(updated_at), "
    "payload=VALUES(payload), embedding=VALUES(embedding)"
)


class AtomStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def _row_args(self, atom: MemoryAtom, emb_blob) -> tuple:
        return (atom.id, self.user_id, atom.memcell_id or None,
                atom.chain_id or None, atom.prev_atom_id or None, atom.next_atom_id or None,
                atom.object_type, atom.holder,
                atom.recorded_at.isoformat(), atom.updated_at.isoformat(),
                atom.model_dump_json(), blob_param(emb_blob))

    def _existing_emb(self, atom_id: str, tx=None) -> bytes | None:
        """Read the existing vector, which is reused when embedding is None.

        When tx is given the read goes through the same connection, so it sees
        what the transaction has written.
        """
        q = tx if tx is not None else self.db
        row = q.fetch_one(
            "SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        return blob_of(row["emb"]) if row else None

    def upsert(self, atom: MemoryAtom, embedding: Optional[np.ndarray] = None) -> str:
        """Insert an atom or replace it wholesale.

        When embedding is None the existing vector is kept, which is what remember
        uses when it edits fields without re-embedding.
        """
        emb_blob = (np.asarray(embedding, dtype=np.float32).tobytes()
                    if embedding is not None else self._existing_emb(atom.id))
        self.db.execute(_UPSERT_SQL, self._row_args(atom, emb_blob))
        logger.info(f"atom upsert id={atom.id} cell={atom.memcell_id} user={self.user_id or '(default)'} "
                    f"type={atom.object_type} holder={atom.holder}")
        return atom.id

    def upsert_many(self, items: list[tuple[MemoryAtom, Optional[np.ndarray]]]) -> int:
        """Upsert in bulk, committing **once, in a single transaction**, so either
        all of it is visible or none of it is.

        This is what the write path uses: every atom of a cell arrives and departs
        together, so a reader on the fast path never sees a half-written cell.
        """
        with self.db.transaction() as tx:
            for atom, embedding in items:
                emb_blob = (np.asarray(embedding, dtype=np.float32).tobytes()
                            if embedding is not None else self._existing_emb(atom.id, tx))
                tx.execute(_UPSERT_SQL, self._row_args(atom, emb_blob))
        logger.info(f"atom upsert_many committed {len(items)} rows in one transaction user={self.user_id or '(default)'}")
        return len(items)

    def get(self, atom_id: str) -> MemoryAtom | None:
        row = self.db.fetch_one(
            "SELECT payload FROM atoms WHERE id=%s AND user_id=%s", (atom_id, self.user_id)
        )
        return MemoryAtom.model_validate_json(row["payload"]) if row else None

    def get_embedding(self, atom_id: str) -> Optional[np.ndarray]:
        row = self.db.fetch_one(
            "SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        emb = blob_of(row["emb"]) if row else None
        if emb is None:
            return None
        return np.frombuffer(emb, dtype=np.float32)

    def list_by_cell(self, memcell_id: str) -> list[MemoryAtom]:
        """Every atom of one cell, ordered as they were written, which is the order the extraction step produced them."""
        rows = self.db.fetch_all(
            "SELECT payload FROM atoms WHERE user_id=%s AND memcell_id=%s ORDER BY recorded_at, id",
            (self.user_id, memcell_id),
        )
        return [MemoryAtom.model_validate_json(r["payload"]) for r in rows]

    def list(self, limit: int = 500) -> list[MemoryAtom]:
        rows = self.db.fetch_all(
            "SELECT payload FROM atoms WHERE user_id=%s ORDER BY recorded_at DESC LIMIT %s",
            (self.user_id, limit),
        )
        return [MemoryAtom.model_validate_json(r["payload"]) for r in rows]

    def all_with_embeddings(self) -> list[tuple[MemoryAtom, np.ndarray]]:
        """All of this user's atoms that carry a vector, for batch MaxSim scoring on
        both branches of the fast path.

        The three chain columns are fetched alongside and overwrite the
        corresponding model fields, because the columns are the only source of
        truth for chain attribution — this stops a stale payload from misleading
        the hot path.
        """
        rows = self.db.fetch_all(
            "SELECT payload, HEX(embedding) AS emb, chain_id, prev_atom_id, next_atom_id "
            "FROM atoms WHERE user_id=%s AND embedding IS NOT NULL",
            (self.user_id,),
        )
        out = []
        for r in rows:
            a = MemoryAtom.model_validate_json(r["payload"])
            a.chain_id = r["chain_id"] or ""
            a.prev_atom_id = r["prev_atom_id"] or ""
            a.next_atom_id = r["next_atom_id"] or ""
            out.append((a, np.frombuffer(blob_of(r["emb"]), dtype=np.float32)))
        return out
