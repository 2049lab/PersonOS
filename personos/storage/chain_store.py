"""atom_chain storage: chain-level information plus the lifecycle of the three
chain columns on atoms.

A chain is a derived view that groups without resolving: it never picks a winner
between an older and a newer statement in the chain, and conflicts are consumed at
answer time.

Membership deliberately has no table of its own. It is strictly 1:1 with an atom,
and the single-valued chain_id column is itself the database-level constraint that
an atom belongs to at most one chain. The occurrence and recorded times are already
in the atom payload.

Chain order is append order, which is the natural order in which facts were
extracted from the conversation. A new member is appended at the tail, with its
prev pointing at the old tail, and the doubly-linked pointers are the only source
of truth for chain order. occurrence_time is display metadata only and is never
used to order a chain — a retrospective mention would scramble it, and it is
unstable when two entries tie.

Multi-tenancy works as it does in AtomStore: an instance is bound to one user.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from ..models import ChainInfo, MemoryAtom, now
from .db import Database, _Tx, blob_of, blob_param

_INSERT_CHAIN = (
    "INSERT INTO atom_chains(id, user_id, n_atoms, head_atom_id, tail_atom_id, "
    "centroid, payload, created_at, updated_at) VALUES(%s,%s,%s,%s,%s,UNHEX(%s),%s,%s,%s)"
)
_UPDATE_CHAIN = (
    "UPDATE atom_chains SET n_atoms=%s, head_atom_id=%s, tail_atom_id=%s, "
    "centroid=UNHEX(%s), payload=%s, updated_at=%s WHERE id=%s AND user_id=%s"
)
# Write the chain columns and the payload together, following the precedent set by
# memcell_id. The caller guarantees the atom row exists.
_SET_LINK = (
    "UPDATE atoms SET chain_id=%s, prev_atom_id=%s, next_atom_id=%s, payload=%s "
    "WHERE id=%s AND user_id=%s"
)


def _centroid_blob(c: Optional[np.ndarray]) -> Optional[str]:
    return blob_param(np.asarray(c, dtype=np.float32).tobytes()) if c is not None else None


class ChainStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    # —— Internal: the paired write of the chain columns. Read the current
    #    payload, validate, change the chain fields, write it back. ——

    def _set_link(self, tx: _Tx, atom_id: str, *, chain_id: str, next_: str,
                  prev: Optional[str], expect_chain: str) -> None:
        """Write one node's chain fields.

        expect_chain is the chain this node should currently be in, where "" means
        it should be unattached. A mismatch raises and rolls back, because linking
        an atom twice or into the wrong chain corrupts the linked-list structure —
        and a structural error is better left raised than quietly repaired.

        prev=None keeps the current value, which is what the old tail needs when it
        gains a next pointer and its predecessor must not move. An empty string or
        an id sets it explicitly.
        """
        row = tx.fetch_one(
            "SELECT payload, chain_id, prev_atom_id FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        if row is None:
            raise RuntimeError(f"chain link target atom does not exist: {atom_id}")
        if (row["chain_id"] or "") != expect_chain:
            raise RuntimeError(
                f"chain link conflict: atom {atom_id} is already on chain {row['chain_id'] or '(none)'} "
                f"(expected {expect_chain or '(unchained)'}) -- double/wrong link, rolling the whole answer back"
            )
        atom = MemoryAtom.model_validate_json(row["payload"])
        new_prev = (row["prev_atom_id"] or "") if prev is None else prev  # the column is the source of truth
        atom.chain_id, atom.prev_atom_id, atom.next_atom_id = chain_id, new_prev, next_
        tx.execute(_SET_LINK, (chain_id or None, new_prev or None, next_ or None,
                               atom.model_dump_json(), atom_id, self.user_id))

    # —— Creating a chain and appending to one. Each is a single transaction, and
    #    append's three writes are atomic together. ——

    def create_chain(self, info: ChainInfo, first_atom: MemoryAtom,
                     centroid: Optional[np.ndarray] = None) -> ChainInfo:
        """Create a chain together with its first member, so head and tail are both
        first_atom and n is 1. This is what fills in info's counting fields.
        """
        info.user_id = self.user_id
        info.n_atoms, info.head_atom_id, info.tail_atom_id = 1, first_atom.id, first_atom.id
        info.created_at = info.updated_at = now()
        with self.db.transaction() as tx:
            self._set_link(tx, first_atom.id, chain_id=info.id, prev="", next_="", expect_chain="")
            tx.execute(_INSERT_CHAIN, (
                info.id, self.user_id, info.n_atoms, info.head_atom_id, info.tail_atom_id,
                _centroid_blob(centroid), info.model_dump_json(),
                info.created_at.isoformat(), info.updated_at.isoformat(),
            ))
        logger.info(f"chain created id={info.id} first={first_atom.id} title={info.title[:30]!r} "
                    f"user={self.user_id or '(default)'}")
        return info

    def append_atom(self, chain: ChainInfo, atom: MemoryAtom,
                    centroid: Optional[np.ndarray] = None) -> ChainInfo:
        """Append at the tail: the old tail gains a next pointer, the new member
        gets its prev, and the chain row's tail, n and centroid move forward — all
        in one transaction.

        The chain object is updated in place and returned, so a caller holding a
        reference already has the latest state.

        centroid=None keeps the current centroid. Otherwise you must pass the full
        centroid *including* the new member; the incremental mean is computed in
        chain_build, and an exact recomputation goes through recompute.
        """
        if not chain.tail_atom_id:
            raise RuntimeError(f"chain {chain.id} has no tail; an empty chain cannot be appended to, use create_chain")
        if centroid is None:   # keep the current value, so UNHEX(NULL) cannot wipe the centroid
            row = self.db.fetch_one(
                "SELECT HEX(centroid) AS c FROM atom_chains WHERE id=%s AND user_id=%s",
                (chain.id, self.user_id),
            )
            c = blob_of(row["c"]) if row else None
            centroid = np.frombuffer(c, dtype=np.float32) if c else None
        with self.db.transaction() as tx:
            self._set_link(tx, chain.tail_atom_id, chain_id=chain.id, prev=None,
                           next_=atom.id, expect_chain=chain.id)
            self._set_link(tx, atom.id, chain_id=chain.id, prev=chain.tail_atom_id,
                           next_="", expect_chain="")
            chain.n_atoms += 1
            chain.tail_atom_id = atom.id
            chain.updated_at = now()
            tx.execute(_UPDATE_CHAIN, (
                chain.n_atoms, chain.head_atom_id, chain.tail_atom_id,
                _centroid_blob(centroid), chain.model_dump_json(),
                chain.updated_at.isoformat(), chain.id, self.user_id,
            ))
        return chain

    # —— Reads ——

    def get_chain(self, chain_id: str) -> Optional[ChainInfo]:
        row = self.db.fetch_one(
            "SELECT payload FROM atom_chains WHERE id=%s AND user_id=%s",
            (chain_id, self.user_id),
        )
        return ChainInfo.model_validate_json(row["payload"]) if row else None

    def list_chains(self) -> list[tuple[ChainInfo, Optional[np.ndarray]]]:
        """All of this user's chains with their centroids: the candidate pool for
        deciding chain membership. There are far fewer chains than atoms.
        """
        rows = self.db.fetch_all(
            "SELECT payload, HEX(centroid) AS c FROM atom_chains WHERE user_id=%s",
            (self.user_id,),
        )
        out = []
        for r in rows:
            c = blob_of(r["c"])
            out.append((ChainInfo.model_validate_json(r["payload"]),
                        np.frombuffer(c, dtype=np.float32) if c else None))
        return out

    def full_chain(self, chain_id: str) -> list[MemoryAtom]:
        """Every member of a chain, in chain order.

        One indexed query fetches them, and the order is then walked out locally
        from the head along the next pointers, since those pointers are the only
        source of truth for chain order. The returned order is therefore append
        order.

        If the chain row is missing or the chain is broken, we return the prefix we
        could walk and log a warning: a damaged derived index should not crash
        anything, and the rebuild script is what puts it right.
        """
        info = self.get_chain(chain_id)
        if info is None:
            return []
        rows = self.db.fetch_all(
            "SELECT payload, chain_id, prev_atom_id, next_atom_id FROM atoms "
            "WHERE user_id=%s AND chain_id=%s",
            (self.user_id, chain_id),
        )
        by_id: dict[str, MemoryAtom] = {}
        for r in rows:
            a = MemoryAtom.model_validate_json(r["payload"])
            a.chain_id = r["chain_id"] or ""
            a.prev_atom_id = r["prev_atom_id"] or ""
            a.next_atom_id = r["next_atom_id"] or ""
            by_id[a.id] = a
        ordered, cur, seen = [], info.head_atom_id, set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            ordered.append(by_id[cur])
            cur = by_id[cur].next_atom_id
        if len(ordered) != len(by_id) or info.n_atoms != len(by_id):
            logger.warning(f"chain {chain_id} is structurally inconsistent: rows={len(by_id)} walked={len(ordered)} "
                           f"recorded n={info.n_atoms} (orphaned or broken links, left for rebuild to repair)")
        return ordered

    # —— Rebuild and repair ——

    def clear_user(self) -> int:
        """The precondition for a rebuild: clear all of this user's chains, deleting
        the chain rows and zeroing both the three chain columns on atoms and the
        chain fields inside their payloads.

        Every statement carries a WHERE clause, since the test guard blocks
        TRUNCATE and unqualified DELETE. Returns how many chains were cleared.

        The payload's chain fields are wiped along with the columns so the column
        and its copy are zeroed from the same place, leaving no half-stale state.
        """
        chains = self.list_chains()
        with self.db.transaction() as tx:
            for info, _ in chains:
                rows = tx.fetch_all(
                    "SELECT id, payload FROM atoms WHERE user_id=%s AND chain_id=%s",
                    (self.user_id, info.id),
                )
                for r in rows:
                    a = MemoryAtom.model_validate_json(r["payload"])
                    a.chain_id = a.prev_atom_id = a.next_atom_id = ""
                    tx.execute(_SET_LINK, (None, None, None, a.model_dump_json(),
                                           r["id"], self.user_id))
                tx.execute("DELETE FROM atom_chains WHERE id=%s AND user_id=%s",
                           (info.id, self.user_id))
        logger.info(f"chains cleared user={self.user_id or '(default)'} chains={len(chains)}")
        return len(chains)

    def recompute(self, chain_id: str) -> Optional[ChainInfo]:
        """Recompute the centroid from the member vectors and bring n_atoms back in
        line, for backfill and repair. This is where the floating-point drift of
        the incremental centroid gets corrected.
        """
        info = self.get_chain(chain_id)
        if info is None:
            return None
        rows = self.db.fetch_all(
            "SELECT id, HEX(embedding) AS emb FROM atoms "
            "WHERE user_id=%s AND chain_id=%s AND embedding IS NOT NULL",
            (self.user_id, chain_id),
        )
        if rows:
            mat = np.stack([np.frombuffer(blob_of(r["emb"]), dtype=np.float32) for r in rows])
            centroid = mat.mean(axis=0)
        else:
            centroid = None
        info.n_atoms = len(rows)
        info.updated_at = now()
        self.db.execute(_UPDATE_CHAIN, (
            info.n_atoms, info.head_atom_id, info.tail_atom_id,
            _centroid_blob(centroid), info.model_dump_json(),
            info.updated_at.isoformat(), info.id, self.user_id,
        ))
        return info
