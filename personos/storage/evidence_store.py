"""证据存储:append-only、不可变。唯一真相源(DESIGN §1)。

铁律:只 insert,永不 update/delete(delete 走治理层的 forget,非本 store 日常能力)。

多租户:实例按 user 绑定(构造注入 user_id),所有读写自动限定在该 user 内——
调用方拿到的就是这个 user 的 store,不存在"忘带 user_id"的泄露面。user_id="" 为
兼容旧库/单租户场景的默认命名空间。
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from loguru import logger

from ..models import EvidenceRecord
from .db import Database, blob_param


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def append(self, rec: EvidenceRecord, embedding: Optional[np.ndarray] = None) -> str:
        """追加一条证据。若 (本user内) sha256 已存在则视为重复,返回既有 id(去重、幂等)。

        embedding:证据文本向量,供 fact 层查不到时的 evidence 兜底语义检索(S3b)。
        """
        if not rec.sha256 and rec.content_inline is not None:
            rec.sha256 = sha256_of(rec.content_inline)
        existing = self.by_sha256(rec.sha256) if rec.sha256 else None
        if existing:
            logger.debug(f"证据去重命中 sha={rec.sha256[:8]} -> {existing.id}")
            return existing.id
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        self.db.execute(
            "INSERT INTO evidence(id, user_id, sha256, holder, modality, captured_at, payload, embedding) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,UNHEX(%s))",
            (rec.id, self.user_id, rec.sha256, rec.holder, rec.modality,
             rec.captured_at.isoformat(), rec.model_dump_json(), blob_param(emb_blob)),
        )
        logger.info(f"证据入库 id={rec.id} user={self.user_id or '(默认)'} holder={rec.holder} len={len(rec.content_inline or '')}")
        return rec.id

    def all_with_embeddings(self) -> list[tuple[EvidenceRecord, np.ndarray]]:
        """取本 user 所有带向量的证据,供 evidence 兜底检索批量打分。"""
        rows = self.db.fetch_all(
            "SELECT payload, HEX(embedding) AS emb FROM evidence "
            "WHERE user_id=%s AND embedding IS NOT NULL",
            (self.user_id,),
        )
        return [
            (EvidenceRecord.model_validate_json(r["payload"]),
             np.frombuffer(bytes.fromhex(r["emb"]), dtype=np.float32))
            for r in rows
        ]

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.db.fetch_one(
            "SELECT payload FROM evidence WHERE id=%s AND user_id=%s", (evidence_id, self.user_id)
        )
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def by_sha256(self, sha: str) -> EvidenceRecord | None:
        row = self.db.fetch_one(
            "SELECT payload FROM evidence WHERE sha256=%s AND user_id=%s LIMIT 1", (sha, self.user_id)
        )
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def list(self, limit: int = 100) -> list[EvidenceRecord]:
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s ORDER BY captured_at DESC LIMIT %s",
            (self.user_id, limit),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]

    def reply_for(self, user_evidence_id: str) -> EvidenceRecord | None:
        """取"回应某条用户证据"的助手证据(其 source.reply_to 指向它)。检索展示时用来把 Q↔A 配对。

        holder 是列、可直接筛;reply_to 在 payload 里,故扫近段 assistant 证据再比对——P0 规模够用。
        """
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s AND holder='assistant' "
            "ORDER BY captured_at DESC LIMIT 500",
            (self.user_id,),
        )
        for r in rows:
            rec = EvidenceRecord.model_validate_json(r["payload"])
            if (rec.source or {}).get("reply_to") == user_evidence_id:
                return rec
        return None

    def in_session(self, session_id: str, limit: int = 20) -> list[EvidenceRecord]:
        """同一会话的证据,按时间 old→new。供 expand 展开"会话内相邻的原始信息"。"""
        rows = self.by_session(session_id)
        return rows[:limit]

    def session_stats(self) -> dict[str, dict]:
        """本 user 各 session 的证据条数 + 最近一条证据时间(ISO 串,无则空串)。

        供工作台列会话:按最近活跃排序——线上排查时"刚在跑的会话"排最前。
        """
        counts: dict[str, dict] = {}
        for rec in self.iter_all():
            sid = (rec.source or {}).get("session_id")
            if not sid:
                continue
            st = counts.setdefault(sid, {"count": 0, "last_at": ""})
            st["count"] += 1
            if rec.captured_at:
                iso = rec.captured_at.isoformat()
                if iso > st["last_at"]:
                    st["last_at"] = iso
        return counts

    def by_session(self, session_id: str) -> list[EvidenceRecord]:
        """本会话【全部】证据,时序 old→new。供会话上下文压缩:需要看到整段历史才能滚动折叠。"""
        recs = [r for r in self.iter_all() if (r.source or {}).get("session_id") == session_id]
        recs.sort(key=lambda r: r.captured_at)
        return recs

    def iter_all(self) -> list[EvidenceRecord]:
        """本 user 全部证据(时间序)。P0 规模够用;单 user 记忆量大后再加分页/列索引。"""
        rows = self.db.fetch_all(
            "SELECT payload FROM evidence WHERE user_id=%s ORDER BY captured_at ASC, id ASC",
            (self.user_id,),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]

    def search_keyword(self, keywords: list[str], *, holder: str = "", limit: int = 30) -> list[EvidenceRecord]:
        """关键词直搜原话(兜底路径):全部关键词须【同一句】命中,时序返回。

        这是唯一不依赖任何索引的检索——atoms 漏抽时原话仍在真相层,从这里可达。
        LIKE 打在 payload JSON 上(CJK 与普通 ASCII 词原样存,含引号/反斜杠的词命中不了)。
        """
        kws = [k.strip() for k in keywords if k and k.strip()]
        if not kws:
            return []
        conds = ["payload LIKE %s ESCAPE '!'"] * len(kws)
        if holder:
            conds.append("holder = %s")
        params: list = [f"%{_like_escape(k)}%" for k in kws]
        if holder:
            params.append(holder)
        rows = self.db.fetch_all(
            f"SELECT payload FROM evidence WHERE user_id=%s AND {' AND '.join(conds)} "
            "ORDER BY captured_at ASC, id ASC LIMIT %s",
            (self.user_id, *params, limit),
        )
        return [EvidenceRecord.model_validate_json(r["payload"]) for r in rows]


def _like_escape(s: str) -> str:
    """LIKE 通配符转义(! % _),配合 ESCAPE '!'(单字符转义符,不受 sql_mode 反斜杠语义影响)。"""
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")
