"""Bounded, user-scoped inspection of memories and their provenance.

Browsing does not resolve providers or load vectors. Chain membership comes
from its columns, never the potentially stale copy in an atom's JSON payload.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ..models import DOMAIN_VOCAB, KindLiteral
from .db import Database

_ATOM_COLUMNS = "a.id, a.payload, a.memcell_id, a.chain_id, a.prev_atom_id, a.next_atom_id"
_MAX_OFFSET = 100_000


def _page(limit: int, offset: int = 0) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= _MAX_OFFSET:
        raise ValueError(f"offset must be an integer between 0 and {_MAX_OFFSET}")


def _text(value: str | None, name: str, maximum: int) -> str | None:
    if value is not None and (not isinstance(value, str) or not value or len(value) > maximum):
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    return value


def _like(value: str) -> str:
    return "%" + value.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


def _metadata(row) -> dict:
    payload = json.loads(row["payload"])
    return {
        "chain_id": row["id"], "title": payload.get("title", ""),
        "n_atoms": row["n_atoms"], "head_atom_id": row["head_atom_id"] or None,
        "tail_atom_id": row["tail_atom_id"] or None,
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def _version(chain: dict) -> list:
    return [chain[k] for k in ("updated_at", "n_atoms", "head_atom_id", "tail_atom_id")]


def _encode_cursor(chain: dict, last: str, count: int) -> str:
    value = [chain["chain_id"], _version(chain), last, count]
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).decode()


def _decode_cursor(cursor: str, chain_id: str) -> tuple[list, str, int]:
    try:
        if not isinstance(cursor, str) or not cursor or len(cursor) > 2048:
            raise ValueError
        decoded = base64.b64decode(cursor, altchars=b"-_", validate=True)
        cid, version, last, count = json.loads(decoded)
        if (cid != chain_id or not isinstance(version, list) or len(version) != 4
                or not isinstance(last, str) or not 1 <= len(last) <= 64
                or isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError
        return version, last, count
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid chain cursor") from exc


class MemoryQuery:
    """Read one user's memories, lightweight chain index and paged chain history."""

    def __init__(self, db: Database, user_id: str):
        self.db = db
        self.user_id = user_id

    def _json_text(self, column: str, field: str) -> str:
        # Only fixed field names from this module reach this helper. MySQL
        # JSON_EXTRACT retains JSON quotes; SQLite returns a SQL scalar.
        value = f"JSON_EXTRACT({column}, '$.{field}')"
        return value if self.db.DIALECT == "sqlite" else f"JSON_UNQUOTE({value})"

    def memories(self, *, session_id: str | None = None, object_type: str | None = None,
                 holder: str | None = None, kind: str | None = None,
                 domain: str | None = None, query: str | None = None,
                 limit: int = 20, offset: int = 0) -> dict:
        """Newest IDs first, with exact facets and a literal text substring filter.

        ID ordering is stable even for legacy records whose ISO timestamps use
        different offsets. Offset pagination can shift when records are added.
        """
        _page(limit, offset)
        _text(session_id, "session_id", 128)
        _text(holder, "holder", 64)
        _text(query, "query", 1000)
        if object_type is not None and (not isinstance(object_type, str)
                                        or object_type not in {"fact", "claim", "event"}):
            raise ValueError("object_type must be fact, claim or event")
        if kind is not None and (not isinstance(kind, str) or kind not in KindLiteral):
            raise ValueError("kind must be a recognized K-axis code")
        if domain is not None and (not isinstance(domain, str) or domain not in DOMAIN_VOCAB):
            raise ValueError("domain must be a recognized D-axis code")
        clauses, params = ["a.user_id=%s"], [self.user_id]
        if session_id is not None:
            clauses.append("c.session_id=%s")
            params.append(session_id)
        for column, value in (("object_type", object_type), ("holder", holder)):
            if value is not None:
                clauses.append(f"a.{column}=%s")
                params.append(value)
        if kind is not None:
            clauses.append(self._json_text("a.payload", "kind") + "=%s")
            params.append(kind)
        if domain is not None:
            clauses.append("JSON_EXTRACT(a.payload, '$.domains') LIKE %s")
            params.append('%"' + domain + '"%')
        if query is not None:
            clauses.append(self._json_text("a.payload", "text") + " LIKE %s ESCAPE '!'")
            params.append(_like(query))
        rows = self.db.fetch_all(
            f"SELECT {_ATOM_COLUMNS} FROM atoms a LEFT JOIN memcells c "
            "ON c.id=a.memcell_id AND c.user_id=a.user_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY a.id DESC LIMIT %s OFFSET %s",
            (*params, limit + 1, offset),
        )
        return {"items": self._items(rows[:limit]), "limit": limit, "offset": offset,
                "has_more": len(rows) > limit}

    def chains(self, *, limit: int = 20, offset: int = 0) -> dict:
        """Read metadata only: no member history, evidence, vectors or LLM calls."""
        _page(limit, offset)
        rows = self.db.fetch_all(
            "SELECT id, payload, n_atoms, head_atom_id, tail_atom_id, created_at, updated_at "
            "FROM atom_chains WHERE user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s",
            (self.user_id, limit + 1, offset),
        )
        return {"items": [_metadata(r) for r in rows[:limit]], "limit": limit,
                "offset": offset, "has_more": len(rows) > limit}

    def _chain(self, chain_id: str) -> dict | None:
        row = self.db.fetch_one(
            "SELECT id, payload, n_atoms, head_atom_id, tail_atom_id, created_at, updated_at "
            "FROM atom_chains WHERE user_id=%s AND id=%s", (self.user_id, chain_id),
        )
        return _metadata(row) if row else None

    def _atom(self, atom_id: str, chain_id: str):
        return self.db.fetch_one(
            f"SELECT {_ATOM_COLUMNS} FROM atoms a "
            "WHERE a.user_id=%s AND a.chain_id=%s AND a.id=%s",
            (self.user_id, chain_id, atom_id),
        )

    def chain(self, chain_id: str, *, limit: int = 20, cursor: str | None = None) -> dict | None:
        """Walk at most one page in append order and report incomplete indices.

        A cursor includes the chain's version and consumed count. It is an
        untrusted paging hint, never an authorization token. Every lookup still
        constrains both user and chain. A changed chain asks callers to restart.
        """
        _page(limit)
        if chain_id is None:
            raise ValueError("chain_id must be a nonempty string")
        _text(chain_id, "chain_id", 64)
        decoded = _decode_cursor(cursor, chain_id) if cursor is not None else None
        chain = self._chain(chain_id)
        if chain is None:
            return None
        state, count, previous = "more", 0, ""
        current = chain["head_atom_id"] or ""
        if decoded:
            version, previous, count = decoded
            if version != _version(chain):
                return self._chain_result(chain, [], limit, "changed")
            row = self._atom(previous, chain_id)
            if row is None or count >= chain["n_atoms"]:
                raise ValueError("invalid chain cursor")
            current = row["next_atom_id"] or ""

        # Most IDs follow append order. Fetch one bounded window in bulk, then
        # follow the authoritative links. Out-of-order IDs need a point lookup;
        # they never make us load the whole chain or sort a partial timeline.
        candidates = self.db.fetch_all(
            f"SELECT {_ATOM_COLUMNS} FROM atoms a "
            "WHERE a.user_id=%s AND a.chain_id=%s AND a.id >= %s "
            "ORDER BY a.id LIMIT %s", (self.user_id, chain_id, current, limit + 1),
        ) if current else []
        by_id = {r["id"]: r for r in candidates}
        rows, seen = [], {previous} if previous else set()
        while current and len(rows) < limit:
            if current in seen or (count and current == chain["head_atom_id"]):
                state = "incomplete"
                break
            row = by_id.get(current)
            if row is None:
                row = self._atom(current, chain_id)
            if row is None or (row["prev_atom_id"] or "") != previous:
                state = "incomplete"
                break
            if count >= chain["n_atoms"]:
                state = "incomplete"
                break
            rows.append(row)
            seen.add(current)
            previous, current = current, row["next_atom_id"] or ""
            count += 1
        if state != "incomplete":
            if not current:
                state = ("complete" if count == chain["n_atoms"]
                         and previous == (chain["tail_atom_id"] or "") else "incomplete")
            elif count >= chain["n_atoms"] or current in seen or current == chain["head_atom_id"]:
                state = "incomplete"
        # A writer may append while we read. Do not publish a page assembled
        # from two different versions, including its possibly stale evidence.
        latest = self._chain(chain_id)
        if latest is None or _version(latest) != _version(chain):
            return self._chain_result(latest or chain, [], limit, "changed")
        next_cursor = _encode_cursor(chain, previous, count) if state == "more" else None
        return self._chain_result(chain, self._items(rows), limit, state, next_cursor)

    @staticmethod
    def _chain_result(chain, items, limit, state, cursor=None) -> dict:
        return {"chain": chain, "items": items, "limit": limit,
                "has_more": state == "more", "next_cursor": cursor, "state": state}

    def _items(self, rows: list) -> list[dict]:
        """Batch-load only this page's referenced cells and evidence."""
        atoms = [(row, json.loads(row["payload"])) for row in rows]
        cell_ids = list(dict.fromkeys(r["memcell_id"] for r, _ in atoms if r["memcell_id"]))
        evidence_ids = list(dict.fromkeys(
            ref["evidence_id"] for _, a in atoms for ref in a.get("evidence_refs", [])
            if isinstance(ref, dict) and ref.get("evidence_id")
        ))
        cells = self._payloads("memcells", cell_ids)
        evidence = self._payloads("evidence", evidence_ids)
        out = []
        for row, atom in atoms:
            cell = cells.get(row["memcell_id"])
            refs = atom.get("evidence_refs", [])
            evs = [evidence[r["evidence_id"]] for r in refs
                   if isinstance(r, dict) and r.get("evidence_id") in evidence]
            out.append({
                "atom_id": row["id"], "text": atom.get("text", ""),
                "object_type": atom.get("object_type", "fact"),
                "holder": atom.get("holder", "user"), "kind": atom.get("kind"),
                "domains": atom.get("domains", []),
                "recorded_at": atom.get("recorded_at"),
                "occurrence_time": atom.get("occurrence_time"),
                "chain_id": row["chain_id"] or None,
                "prev_atom_id": row["prev_atom_id"] or None,
                "next_atom_id": row["next_atom_id"] or None,
                "cell_id": row["memcell_id"] or None,
                "cell": ({"cell_id": row["memcell_id"], "session_id": cell.get("session_id"),
                          "topic": cell.get("topic", ""), "episode": cell.get("episode", ""),
                          "episode_type": cell.get("episode_type", "unknown"),
                          "start_time": cell.get("t_start"), "end_time": cell.get("t_end")}
                         if cell else None),
                "evidence_refs": refs,
                "evidence": [{"id": e["id"], "holder": e.get("holder"),
                              "content": e.get("content_inline"), "modality": e.get("modality"),
                              "captured_at": e.get("captured_at")} for e in evs],
            })
        return out

    def _payloads(self, table: str, ids: list[str]) -> dict[str, Any]:
        # Chunk references so a page with many citations stays below SQLite's
        # parameter limit. Both possible table names are constants above.
        found = {}
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            placeholders = ",".join(["%s"] * len(batch))
            rows = self.db.fetch_all(
                f"SELECT id, payload FROM {table} WHERE user_id=%s AND id IN ({placeholders})",
                (self.user_id, *batch),
            )
            found.update((r["id"], json.loads(r["payload"])) for r in rows)
        return found
