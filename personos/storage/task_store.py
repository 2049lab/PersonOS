"""异步任务登记簿(MySQL tasks 表):202+轮询模式的任务状态,跨副本可查。

此前在 rt.tasks 内存 dict——重部署/多副本即丢(轮询 404)。落库后:
- 提交副本写 pending→running→done/error,任意副本按 task_id 可查;
- 执行体仍在提交副本的线程池里,pod 死在执行中 → 僵尸收割判 worker lost;
- done/error 行保留一段时间供轮询,之后由 sweep 过期清理(MySQL 无原生 TTL)。
"""

from __future__ import annotations

import json
from datetime import timedelta

from ..models import now
from .db import Database

STALE_RUNNING_S = 600       # pending/running 超过 10 分钟没动静:执行体已死,判 worker lost
KEEP_DONE_S = 24 * 3600     # 终态行保留 1 天,过期清理


def _before(seconds: int) -> str:
    """N 秒之前的 ISO 时间串(列存 ISO 字符串,字典序=时间序,可直接比较)。"""
    return (now() - timedelta(seconds=seconds)).isoformat()


class TaskStore:
    """tasks 表的薄封装:状态流转 + 查询 + 低频清扫。"""

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
        """单条查询;result 列是 JSON 串,解析成 dict 返回(与旧内存版形状一致)。"""
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
                pass    # 非 JSON(手写/异常值):原样返回
        return d

    def sweep(self, *, stale_running_s: int = STALE_RUNNING_S,
              keep_done_s: int = KEEP_DONE_S) -> None:
        """僵尸收割 + 过期清理。幂等;由 submit_task 低频触发,失败由调用方兜住。"""
        self.db.execute(
            "UPDATE tasks SET status='error', error='worker lost', updated_at=%s "
            "WHERE status IN ('pending','running') AND updated_at < %s",
            (now().isoformat(), _before(stale_running_s)),
        )
        self.db.execute(
            "DELETE FROM tasks WHERE status IN ('done','error') AND updated_at < %s",
            (_before(keep_done_s),),
        )
