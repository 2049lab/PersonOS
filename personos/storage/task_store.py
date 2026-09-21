"""The async task register, backed by the tasks table: task state for the
202-plus-polling pattern, queryable from any replica.

This used to be an in-memory dict, which meant a redeploy or a second replica lost
it and polling returned 404. Now that it is in the database:

- the submitting replica writes pending -> running -> done/error, and any replica
  can look a task up by task_id;
- the work itself still runs in the submitting replica's thread pool, so if that
  pod dies mid-execution the zombie sweep marks the task worker-lost;
- done and error rows are kept around for a while so polling can still see them,
  and are then expired by the sweep, since MySQL has no native TTL.
"""

from __future__ import annotations

import json
from datetime import timedelta

from ..models import now
from .db import Database

STALE_RUNNING_S = 600       # pending or running with no movement for 10 minutes means the worker died: judge it worker-lost
KEEP_DONE_S = 24 * 3600     # terminal rows are kept for a day, then swept


def _before(seconds: int) -> str:
    """The ISO timestamp N seconds ago.

    The column stores ISO strings, where lexical order equals chronological order,
    so it can be compared directly.
    """
    return (now() - timedelta(seconds=seconds)).isoformat()


class TaskStore:
    """A thin wrapper over the tasks table: state transitions, lookup, and an infrequent sweep."""

    def __init__(self, db: Database):
        self.db = db

    def create(self, task_id: str, kind: str, user_id: str, session_id: str) -> None:
        ts = now().isoformat()
        self.db.execute(
            "INSERT INTO tasks (task_id, kind, user_id, session_id, status, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, 'pending', %s, %s)",
            (task_id, kind, user_id, session_id, ts, ts),
        )

    def mark_running(self, task_id: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='running', updated_at=%s WHERE task_id=%s",
            (now().isoformat(), task_id),
        )

    def mark_done(self, task_id: str, result: dict) -> None:
        self.db.execute(
            "UPDATE tasks SET status='done', result=%s, updated_at=%s WHERE task_id=%s",
            (json.dumps(result, ensure_ascii=False), now().isoformat(), task_id),
        )

    def mark_error(self, task_id: str, error: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='error', error=%s, updated_at=%s WHERE task_id=%s",
            ((error or "unknown")[:60000], now().isoformat(), task_id),
        )

    def get(self, task_id: str) -> dict | None:
        """Look up one task.

        The result column holds a JSON string, which is parsed back into a dict so
        the shape matches what the old in-memory version returned.
        """
        row = self.db.fetch_one(
            "SELECT task_id, kind, user_id, session_id, status, result, error "
            "FROM tasks WHERE task_id=%s",
            (task_id,),
        )
        if row is None:
            return None
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (TypeError, ValueError):
                pass    # not JSON (hand-written or a malformed value): return it as-is
        return d

    def sweep(self, *, stale_running_s: int = STALE_RUNNING_S,
              keep_done_s: int = KEEP_DONE_S) -> None:
        """Reap zombies and delete expired rows.

        Idempotent, triggered infrequently by submit_task, and a failure is absorbed
        by the caller.
        """
        self.db.execute(
            "UPDATE tasks SET status='error', error='worker lost', updated_at=%s "
            "WHERE status IN ('pending','running') AND updated_at < %s",
            (now().isoformat(), _before(stale_running_s)),
        )
        self.db.execute(
            "DELETE FROM tasks WHERE status IN ('done','error') AND updated_at < %s",
            (_before(keep_done_s),),
        )
