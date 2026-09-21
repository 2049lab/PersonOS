"""原子存储:指向 cell 的检索单元,upsert 语义。向量单独列存,供 MaxSim 批量打分。

多租户:实例按 user 绑定(构造注入 user_id),所有读写自动限定在该 user 内。user_id="" 为
兼容旧库/单租户场景的默认命名空间。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from ..models import MemoryAtom
from .db import Database, blob_of, blob_param

# VALUES(col) 写法:兼容 5.6/5.7/8.0 全版本(RedHub 代理报 5.6 协议,不用 8.0.19+ 的 AS new 别名);
# embedding 走 UNHEX hex 通道(代理对 _binary 字面量非 binary-safe,见 db.blob_param)。
# 链三列只进 INSERT、不进 ON DUPLICATE KEY UPDATE:重放 upsert 不抹链归属
# (W2/remember 都写新 atom,本无碰撞;这是对将来重放路径的结构性保险)。
_UPSERT_SQL = (
    "INSERT INTO atoms(id, user_id, memcell_id, chain_id, prev_atom_id, next_atom_id, "
    "object_type, holder, recorded_at, updated_at, payload, embedding) "
    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,UNHEX(%s)) "
    "ON DUPLICATE KEY UPDATE memcell_id=VALUES(memcell_id), "
    "object_type=VALUES(object_type), holder=VALUES(holder), "
    "recorded_at=VALUES(recorded_at), updated_at=VALUES(updated_at), "
    "payload=VALUES(payload), embedding=VALUES(embedding)"
)


class AtomStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def _row_args(self, atom: MemoryAtom, emb_blob) -> tuple:
        return (atom.id, self.user_id, atom.memcell_id or None,
                atom.chain_id or None, atom.prev_atom_id or None, atom.next_atom_id or None,
                atom.object_type, atom.holder,
                atom.recorded_at.isoformat(), atom.updated_at.isoformat(),
                atom.model_dump_json(), blob_param(emb_blob))

    def _existing_emb(self, atom_id: str, tx=None) -> bytes | None:
        """读旧向量(embedding=None 复用);tx 传入时走同连接,事务内可见。"""
        q = tx if tx is not None else self.db
        row = q.fetch_one(
            "SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        return blob_of(row["emb"]) if row else None

    def upsert(self, atom: MemoryAtom, embedding: Optional[np.ndarray] = None) -> str:
        """插入或整体替换一条原子。embedding 为 None 时保留原向量(remember 只改字段不重embed 时用)。"""
        emb_blob = (np.asarray(embedding, dtype=np.float32).tobytes()
                    if embedding is not None else self._existing_emb(atom.id))
        self.db.execute(_UPSERT_SQL, self._row_args(atom, emb_blob))
        logger.info(f"原子 upsert id={atom.id} cell={atom.memcell_id} user={self.user_id or '(默认)'} "
                    f"type={atom.object_type} holder={atom.holder}")
        return atom.id

    def upsert_many(self, items: list[tuple[MemoryAtom, Optional[np.ndarray]]]) -> int:
        """批量 upsert,【单事务一次提交】(原子性):要么全可见、要么全不可见。

        W2 落库用它——一个 cell 的全部 atom 同进同出,快链读者不会读到"写了一半"的 cell。
        """
        with self.db.transaction() as tx:
            for atom, embedding in items:
                emb_blob = (np.asarray(embedding, dtype=np.float32).tobytes()
                            if embedding is not None else self._existing_emb(atom.id, tx))
                tx.execute(_UPSERT_SQL, self._row_args(atom, emb_blob))
        logger.info(f"原子 upsert_many 提交 {len(items)} 条(单事务)user={self.user_id or '(默认)'}")
        return len(items)

    def get(self, atom_id: str) -> MemoryAtom | None:
        row = self.db.fetch_one(
            "SELECT payload FROM atoms WHERE id=%s AND user_id=%s", (atom_id, self.user_id)
        )
        return MemoryAtom.model_validate_json(row["payload"]) if row else None

    def get_embedding(self, atom_id: str) -> Optional[np.ndarray]:
        row = self.db.fetch_one(
            "SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        emb = blob_of(row["emb"]) if row else None
        if emb is None:
            return None
        return np.frombuffer(emb, dtype=np.float32)

    def list_by_cell(self, memcell_id: str) -> list[MemoryAtom]:
        """一个 cell 的全部原子(时序=落库序,即 W2② 输出序)。"""
        rows = self.db.fetch_all(
            "SELECT payload FROM atoms WHERE user_id=%s AND memcell_id=%s ORDER BY recorded_at, id",
            (self.user_id, memcell_id),
        )
        return [MemoryAtom.model_validate_json(r["payload"]) for r in rows]

    def list(self, limit: int = 500) -> list[MemoryAtom]:
        rows = self.db.fetch_all(
            "SELECT payload FROM atoms WHERE user_id=%s ORDER BY recorded_at DESC LIMIT %s",
            (self.user_id, limit),
        )
        return [MemoryAtom.model_validate_json(r["payload"]) for r in rows]

    def all_with_embeddings(self) -> list[tuple[MemoryAtom, np.ndarray]]:
        """取本 user 所有带向量的原子,供快链双路批量 MaxSim 打分。

        链三列随行取回并覆盖模型字段(列是链归属的唯一事实源,防过期 payload 误导热路径)。
        """
        rows = self.db.fetch_all(
            "SELECT payload, HEX(embedding) AS emb, chain_id, prev_atom_id, next_atom_id "
            "FROM atoms WHERE user_id=%s AND embedding IS NOT NULL",
            (self.user_id,),
        )
        out = []
        for r in rows:
            a = MemoryAtom.model_validate_json(r["payload"])
            a.chain_id = r["chain_id"] or ""
            a.prev_atom_id = r["prev_atom_id"] or ""
            a.next_atom_id = r["next_atom_id"] or ""
            out.append((a, np.frombuffer(blob_of(r["emb"]), dtype=np.float32)))
        return out
