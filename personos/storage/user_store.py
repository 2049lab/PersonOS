"""用户注册表:token ↔ user。多租户的入口(注册签发 token,调用方凭 token 定位自己的记忆)。

users 表与业务表同库;token 只在注册响应里给一次全文,库内明文存
(P0 内网部署;上生产前换哈希存+可吊销,见 deploy-xhs 待办)。
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
        """注册一个用户:user_id 可指定(重复则报错)或自动生成;签发随机 token。"""
        user_id = (user_id or "").strip() or f"u_{ULID()}"
        row = self.db.fetch_one("SELECT 1 FROM users WHERE user_id=%s", (user_id,))
        if row:
            raise ValueError(f"user_id 已存在: {user_id}")
        token = secrets.token_urlsafe(24)
        try:
            self.db.execute(
                "INSERT INTO users(token, user_id, created_at) VALUES(%s,%s,%s)",
                (token, user_id, now().isoformat()),
            )
        except IntegrityError:
            # SELECT-then-INSERT 窗口期并发注册:UNIQUE(user_id) 兜底,保持 409 语义
            raise ValueError(f"user_id 已存在: {user_id}")
        return {"user_id": user_id, "token": token}

    def user_id_by_token(self, token: str) -> str | None:
        row = self.db.fetch_one("SELECT user_id FROM users WHERE token=%s", (token,))
        return row["user_id"] if row else None

    def list_users(self) -> list[dict]:
        rows = self.db.fetch_all("SELECT user_id, created_at FROM users ORDER BY created_at DESC")
        return [{"user_id": r["user_id"], "created_at": r["created_at"]} for r in rows]

    def count(self) -> int:
        return self.db.fetch_one("SELECT COUNT(*) AS n FROM users")["n"]
