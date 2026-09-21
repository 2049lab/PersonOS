"""会话对话历史的滚动压缩缓存存取(派生物,可覆盖、可持久)。

每 (user, session) 单条:summary(已压缩的更早历史)+ covered(已折进 summary 的轮数=水位)。
与 evidence 分开存:evidence 是不可变真相源,summary 只是为对话连续性服务的临时上下文。

多租户:实例按 user 绑定(构造注入 user_id);复合主键 (user_id, session_id)。
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
        # 复合主键 (user_id, session_id) 命中即更新,无需冲突目标子句
        self.db.execute(
            "INSERT INTO session_context(user_id, session_id, summary, covered, updated_at) "
            "VALUES(%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE summary=VALUES(summary), "
            "covered=VALUES(covered), updated_at=VALUES(updated_at)",
            (self.user_id, session_id, summary, covered, now().isoformat()),
        )
