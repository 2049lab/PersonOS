"""Access to the rolling compressed cache of a session's conversation history.

It is derived data: overwritable and persistent. One row per (user, session),
holding summary — the already-compressed earlier history — and covered, the number
of turns folded into it, which acts as a high-water mark.

It is kept apart from evidence deliberately: evidence is the immutable source of
truth, while a summary is only temporary context in service of conversational
continuity.

For multi-tenancy, an instance is bound to one user through the constructor, and
the primary key is the composite (user_id, session_id).
"""

from __future__ import annotations

from ..models import now
from .db import Database


class SessionContextStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def get(self, session_id: str) -> tuple[str, int]:
        row = self.db.fetch_one(
            "SELECT summary, covered FROM session_context WHERE user_id=%s AND session_id=%s",
            (self.user_id, session_id),
        )
        return (row["summary"], row["covered"]) if row else ("", 0)

    def save(self, session_id: str, summary: str, covered: int) -> None:
        # A hit on the composite primary key (user_id, session_id) updates in place,
        # so no conflict-target clause is needed.
        self.db.execute(
            "INSERT INTO session_context(user_id, session_id, summary, covered, updated_at) "
            "VALUES(%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE summary=VALUES(summary), "
            "covered=VALUES(covered), updated_at=VALUES(updated_at)",
            (self.user_id, session_id, summary, covered, now().isoformat()),
        )
