"""MemCell 存储:段粒度组织单元。topic 向量单独列存,是段粒度检索面(find_cells / rerank 材料头)。

多租户:与 AtomStore 同规,实例按 user 绑定。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from ..models import MemCell, ensure_aware
from .db import Database, blob_of, blob_param

# VALUES(col) 写法:兼容 5.6/5.7/8.0 全版本(同 atom_store);topic_embedding 走 UNHEX hex 通道
_UPSERT_SQL = (
    "INSERT INTO memcells(id, user_id, session_id, t_start, t_end, episode_type, payload, topic_embedding) "
    "VALUES(%s,%s,%s,%s,%s,%s,%s,UNHEX(%s)) "
    "ON DUPLICATE KEY UPDATE session_id=VALUES(session_id), t_start=VALUES(t_start), "
    "t_end=VALUES(t_end), episode_type=VALUES(episode_type), payload=VALUES(payload), "
    "topic_embedding=VALUES(topic_embedding)"
)


class CellStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def upsert(self, cell: MemCell, topic_embedding: Optional[np.ndarray] = None) -> str:
        """插入或整体替换一个 cell(remember 修订 episode 时整体替换)。embedding None 保留原向量。"""
        emb_blob = topic_embedding
        if emb_blob is None:
            row = self.db.fetch_one(
                "SELECT HEX(topic_embedding) AS emb FROM memcells WHERE id=%s AND user_id=%s",
                (cell.id, self.user_id),
            )
            emb_blob = blob_of(row["emb"]) if row else None
        else:
            emb_blob = np.asarray(emb_blob, dtype=np.float32).tobytes()
        self.db.execute(_UPSERT_SQL, (
            cell.id, self.user_id, cell.session_id,
            cell.t_start.isoformat() if cell.t_start else None,
            cell.t_end.isoformat() if cell.t_end else None,
            cell.episode_type or "unknown",
            cell.model_dump_json(), blob_param(emb_blob),
        ))
        logger.info(f"cell upsert id={cell.id} session={cell.session_id} "
                    f"user={self.user_id or '(默认)'} topic={cell.topic[:40]!r}")
        return cell.id

    def get(self, cell_id: str) -> MemCell | None:
        row = self.db.fetch_one(
            "SELECT payload FROM memcells WHERE id=%s AND user_id=%s", (cell_id, self.user_id)
        )
        return MemCell.model_validate_json(row["payload"]) if row else None

    def list_session(self, session_id: str) -> list[MemCell]:
        """一个会话的全部 cell,按段起始时间 old→new(即对话时序)。"""
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s AND session_id=%s "
            "ORDER BY t_start, id",
            (self.user_id, session_id),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def count_session(self, session_id: str) -> int:
        """该会话已产出 cell 数:重部署/多副本下的权威口径(取代内存累积计数)。"""
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s AND session_id=%s",
            (self.user_id, session_id),
        )
        return int(row["n"]) if row else 0

    def iter_all(self, limit: int = 1000) -> list[MemCell]:
        """本 user 全部 cell,按段起始时间 old→new。"""
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s ORDER BY t_start, id LIMIT %s",
            (self.user_id, limit),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def cells_after(self, up_to_cell_id: str = "") -> list[MemCell]:
        """本 user 中,游标 cell 之后(按 (t_start, id) 全序)的所有 cell,old→new。

        供画像 consolidate 取"上次出版本以来的新 cell"。游标空 → 全部;游标 cell 已不存在
        (如被 forget 删)→ 退化为全部(触发全量重蒸馏,符合"画像可重编译")。
        (t_start, id) 元组序:t_start 同值时以 id 兜底,不漏不重。t_start 为 ISO 字符串,字典序=时间序。
        """
        if not up_to_cell_id:
            return self.iter_all()
        cur = self.get(up_to_cell_id)
        if cur is None or cur.t_start is None:
            return self.iter_all()
        ct = cur.t_start.isoformat()
        rows = self.db.fetch_all(
            "SELECT payload FROM memcells WHERE user_id=%s "
            "AND (t_start > %s OR (t_start = %s AND id > %s)) ORDER BY t_start, id",
            (self.user_id, ct, ct, up_to_cell_id),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def _type_where(self, episode_type: Optional[str], start: Optional[str], end: Optional[str]):
        """按 user + 可选(类型/时间范围)拼 WHERE;过滤值一律走参数,子句名固定,无注入面。"""
        clauses, params = ["user_id=%s"], [self.user_id]
        if episode_type:
            clauses.append("episode_type=%s"); params.append(episode_type)
        if start:
            clauses.append("t_start >= %s"); params.append(start)      # t_start 存 ISO 串,字典序=时间序
        if end:
            clauses.append("t_start <= %s"); params.append(end)
        return " AND ".join(clauses), params

    def list_by_type(self, *, episode_type: Optional[str] = None, start: Optional[str] = None,
                     end: Optional[str] = None, limit: int = 20, offset: int = 0) -> list[MemCell]:
        """按 episode_type + 时间范围分页取本 user 的 cell,t_start 倒序(最近在前)。过滤项皆可选,limit/offset 必给。"""
        where, params = self._type_where(episode_type, start, end)
        rows = self.db.fetch_all(
            f"SELECT payload FROM memcells WHERE {where} ORDER BY t_start DESC, id DESC LIMIT %s OFFSET %s",
            (*params, limit, offset),
        )
        return [MemCell.model_validate_json(r["payload"]) for r in rows]

    def count_by_type(self, *, episode_type: Optional[str] = None, start: Optional[str] = None,
                      end: Optional[str] = None) -> int:
        """同 list_by_type 的过滤条件下的总数(分页 total)。"""
        where, params = self._type_where(episode_type, start, end)
        row = self.db.fetch_one(f"SELECT COUNT(*) AS n FROM memcells WHERE {where}", tuple(params))
        return int(row["n"]) if row else 0

    def all_with_embeddings(self) -> list[tuple[MemCell, np.ndarray]]:
        """取本 user 所有带 topic 向量的 cell,供 find_cells 语义检索批量打分。"""
        rows = self.db.fetch_all(
            "SELECT payload, HEX(topic_embedding) AS emb FROM memcells "
            "WHERE user_id=%s AND topic_embedding IS NOT NULL",
            (self.user_id,),
        )
        return [
            (MemCell.model_validate_json(r["payload"]),
             np.frombuffer(blob_of(r["emb"]), dtype=np.float32))
            for r in rows
        ]

    def in_window(self, cells: list[MemCell], start: Optional[datetime], end: Optional[datetime]) -> list[MemCell]:
        """按段起始时间过滤(P0 规模 Python 侧过滤即可;不填 = 该侧无界)。"""
        s, e = ensure_aware(start), ensure_aware(end)
        out = []
        for c in cells:
            t = ensure_aware(c.t_start)
            if t is None:
                continue          # 无时间元数据的 cell 不进时间窗(宁缺勿错)
            if s and t < s:
                continue
            if e and t > e:
                continue
            out.append(c)
        return out
