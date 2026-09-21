"""The user registry: token to user. This is the entry point for multi-tenancy —
registration issues a token, and the caller uses it to find their own memories.

The users table lives in the same database as the business tables. The token is
returned in full exactly once, in the registration response, and is stored in
plain text. That is acceptable only for an internal deployment; before a real
production rollout it should be stored hashed and be revocable.
"""

from __future__ import annotations

import secrets

from sqlalchemy.exc import IntegrityError
from ulid import ULID

from ..models import now
from .db import Database


class UserStore:
    def __init__(self, db: Database):
        self.db = db

    def register(self, user_id: str | None = None) -> dict:
        """Register a user.

        user_id may be supplied, in which case a duplicate raises, or generated
        automatically. Either way a random token is issued.
        """
        user_id = (user_id or "").strip() or f"u_{ULID()}"
        row = self.db.fetch_one("SELECT 1 FROM users WHERE user_id=%s", (user_id,))
        if row:
            raise ValueError(f"user_id already exists: {user_id}")
        token = secrets.token_urlsafe(24)
        try:
            self.db.execute(
                "INSERT INTO users(token, user_id, created_at) VALUES(%s,%s,%s)",
                (token, user_id, now().isoformat()),
            )
        except IntegrityError:
            # A concurrent registration landing in the window between the SELECT and
            # the INSERT is caught by UNIQUE(user_id), which keeps the 409 semantics.
            raise ValueError(f"user_id already exists: {user_id}")
        return {"user_id": user_id, "token": token}

    def user_id_by_token(self, token: str) -> str | None:
        row = self.db.fetch_one("SELECT user_id FROM users WHERE token=%s", (token,))
        return row["user_id"] if row else None

    def list_users(self) -> list[dict]:
        rows = self.db.fetch_all("SELECT user_id, created_at FROM users ORDER BY created_at DESC")
        return [{"user_id": r["user_id"], "created_at": r["created_at"]} for r in rows]

    def count(self) -> int:
        return self.db.fetch_one("SELECT COUNT(*) AS n FROM users")["n"]
