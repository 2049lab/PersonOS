"""TaskStore 单测(MySQL tasks 表;经 db fixture 原子回退,零污染)。"""

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
    assert got["result"] == {"evidence_id": "e1", "boundary": None}   # JSON 已解析回 dict
    assert got["user_id"] == "u1" and got["session_id"] == "s1"

    ts.create("t2", "session-end", "u1", "s1")
    ts.mark_error("t2", "上游模型超时")
    assert ts.get("t2")["status"] == "error"
    assert ts.get("t2")["error"] == "上游模型超时"

    assert ts.get("不存在") is None


def test_user_scoping_by_caller(db):
    """user_id 落库:轮询鉴权(GET /tasks 只能查自己)由 API 层用该字段判。"""
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.create("t2", "ingest", "u2", "s1")
    assert ts.get("t1")["user_id"] == "u1"
    assert ts.get("t2")["user_id"] == "u2"


def test_sweep_reaps_stale_running(db):
    """running 久未更新 → worker lost;终态行过期清理。"""
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.mark_running("t1")
    # 手动把 updated_at 拨回 1 小时前(ISO 串列,直接 UPDATE)
    db.execute("UPDATE tasks SET updated_at=%s WHERE task_id=%s",
               ((now() - timedelta(hours=1)).isoformat(), "t1"))

    ts.create("t2", "session-end", "u1", "s1")
    ts.mark_done("t2", {"ok": True})
    db.execute("UPDATE tasks SET updated_at=%s WHERE task_id=%s",
               ((now() - timedelta(hours=25)).isoformat(), "t2"))

    ts.sweep(stale_running_s=600, keep_done_s=86400)

    got = ts.get("t1")
    assert got["status"] == "error" and got["error"] == "worker lost"
    assert ts.get("t2") is None                  # 过期终态行已清


def test_sweep_keeps_fresh_rows(db):
    ts = TaskStore(db)
    ts.create("t1", "ingest", "u1", "s1")
    ts.mark_running("t1")
    ts.create("t2", "ingest", "u1", "s1")
    ts.mark_done("t2", {"ok": True})
    ts.sweep()
    assert ts.get("t1")["status"] == "running"   # 新鲜 running 不收割
    assert ts.get("t2")["status"] == "done"      # 新鲜终态不清理
