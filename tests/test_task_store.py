"""TaskStore unit tests against the MySQL tasks table; the db fixture rolls everything back
atomically, so nothing is left behind."""

from __future__ import annotations

from datetime import timedelta

from personos.models import now
from personos.storage.task_store import TaskStore


def test_lifecycle_roundtrip(db):
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    assert ts.get("t1")["status"] == "pending"

    ts.mark_running("t1")
    assert ts.get("t1")["status"] == "running"

    ts.mark_done("t1", {"evidence_id": "e1", "boundary": None})
    got = ts.get("t1")
    assert got["status"] == "done"
    assert got["result"] == {"evidence_id": "e1", "boundary": None}   # the JSON was parsed back into a dict
    assert got["user_id"] == "u1" and got["session_id"] == "s1"

    ts.create("t2", "session-end", "u1", "s1")
    ts.mark_error("t2", "上游模型超时")
    assert ts.get("t2")["status"] == "error"
    assert ts.get("t2")["error"] == "上游模型超时"

    assert ts.get("不存在") is None


def test_user_scoping_by_caller(db):
    """user_id is stored on the row: the API layer uses this field to authorize polling, so GET /tasks
    can only return the caller's own tasks."""
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.create("t2", "ingest", "u2", "s1")
    assert ts.get("t1")["user_id"] == "u1"
    assert ts.get("t2")["user_id"] == "u2"


def test_sweep_reaps_stale_running(db):
    """A running row that has not been updated for a long time is marked worker lost, and terminal rows
    are cleaned up once they expire."""
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.mark_running("t1")
    # Wind updated_at back one hour by hand (it is an ISO string column, so UPDATE it directly)
    db.execute("UPDATE tasks SET updated_at=%s WHERE task_id=%s",
               ((now() - timedelta(hours=1)).isoformat(), "t1"))

    ts.create("t2", "session-end", "u1", "s1")
    ts.mark_done("t2", {"ok": True})
    db.execute("UPDATE tasks SET updated_at=%s WHERE task_id=%s",
               ((now() - timedelta(hours=25)).isoformat(), "t2"))

    ts.sweep(stale_running_s=600, keep_done_s=86400)

    got = ts.get("t1")
    assert got["status"] == "error" and got["error"] == "worker lost"
    assert ts.get("t2") is None                  # the expired terminal row was removed


def test_sweep_keeps_fresh_rows(db):
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.mark_running("t1")
    ts.create("t2", "ingest", "u1", "s1")
    ts.mark_done("t2", {"ok": True})
    ts.sweep()
    assert ts.get("t1")["status"] == "running"   # a fresh running row is not reaped
    assert ts.get("t2")["status"] == "done"      # a fresh terminal row is not cleaned up
