"""身份层存储:characters / character_assets / character_cloud 三表的 per-user 存取。

移植自 mneme-release anchor/store.py 的持久部分(character/asset/cloud),按 personos 规范改写:
- per-user 隔离:实例绑定 user_id,所有 SQL 带 `WHERE user_id=%s`(铁律,AGENTS.md §11);
- 向量走 hex 通道:MEDIUMBLOB 列写 `UNHEX(%s)`/读 `HEX()`(RedHub 非 binary-safe,复用 blob_param/blob_of);
- id=ULID(全局唯一),时间=ISO VARCHAR,无 FK;
- name_claim/trait 折进 `characters.payload`;cloud prototype+template 合表按 `slot` 区分。

原则沿用:名字是 claim 不是 id;retire≠delete;台账(payload 内)只追加。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from personos.models import _ulid
from personos.storage.db import Database, blob_of, blob_param


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emb_to_bytes(emb: Optional[np.ndarray]) -> Optional[bytes]:
    """向量 → float32 原始字节(与 mneme emb_to_blob 一致的存储格式)。"""
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32).tobytes()


def bytes_to_emb(b: Optional[bytes]) -> Optional[np.ndarray]:
    if not b:
        return None
    return np.frombuffer(b, dtype=np.float32)


class CharacterStore:
    """一个 user 的角色档案 + 素材 + 概率云存取。所有查询按 user_id 圈死在本用户社交圈内。"""

    def __init__(self, db: Database, user_id: str) -> None:
        self.db = db
        self.user_id = user_id

    # ── characters ───────────────────────────────────────────────────
    def create_character(self, *, session_id: str = "", is_wearer: bool = False,
                         text_profile: dict[str, Any] | None = None) -> str:
        cid = _ulid("char")
        now = _now_iso()
        payload = {"text_profile": text_profile or {}, "name_claims": [],
                   "first_session": session_id, "last_seen_session": session_id}
        self.db.execute(
            "INSERT INTO characters(id, user_id, is_wearer, primary_name, status, merged_into,"
            " face_tau, voice_tau, payload, created_at, updated_at)"
            " VALUES (%s,%s,%s,NULL,'active',NULL,0,0,%s,%s,%s)",
            (cid, self.user_id, 1 if is_wearer else 0,
             json.dumps(payload, ensure_ascii=False), now, now))
        return cid

    def get_character(self, character_id: str) -> Optional[dict[str, Any]]:
        row = self.db.fetch_one(
            "SELECT id, user_id, is_wearer, primary_name, status, merged_into, face_tau,"
            " voice_tau, payload, created_at, updated_at FROM characters"
            " WHERE user_id=%s AND id=%s", (self.user_id, character_id))
        return self._char_row(row) if row else None

    def _char_row(self, row) -> dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d.get("payload") or "{}")
        d["is_wearer"] = bool(d.get("is_wearer"))
        return d

    def list_active_characters(self, *, include_wearer: bool = False) -> list[dict[str, Any]]:
        sql = ("SELECT id, user_id, is_wearer, primary_name, status, merged_into, face_tau,"
               " voice_tau, payload, created_at, updated_at FROM characters"
               " WHERE user_id=%s AND status='active'")
        if not include_wearer:
            sql += " AND is_wearer=0"
        sql += " ORDER BY id"
        return [self._char_row(r) for r in self.db.fetch_all(sql, (self.user_id,))]

    def touch_character(self, character_id: str, session_id: str) -> None:
        """更新 last_seen_session(供跨会话"最近见于哪次"展示)。"""
        ch = self.get_character(character_id)
        if not ch:
            return
        pl = ch["payload"]
        pl["last_seen_session"] = session_id
        self.db.execute(
            "UPDATE characters SET payload=%s, updated_at=%s WHERE user_id=%s AND id=%s",
            (json.dumps(pl, ensure_ascii=False), _now_iso(), self.user_id, character_id))

    # ── wearer 指针(终审把 SW 链落成佩戴者档案)──────────────────────
    def wearer_character(self) -> Optional[str]:
        """最近的佩戴者档案(is_wearer 表"曾是佩戴者",可能多个;ULID 时序,取最新)。"""
        row = self.db.fetch_one(
            "SELECT id FROM characters WHERE user_id=%s AND is_wearer=1 AND status='active'"
            " ORDER BY id DESC LIMIT 1", (self.user_id,))
        return row["id"] if row else None

    def mark_wearer(self, character_id: str) -> None:
        """终审把 SW 链落到此档 → 标记它曾是佩戴者(非单例:换人换机会有多个)。"""
        self.db.execute(
            "UPDATE characters SET is_wearer=1, updated_at=%s WHERE user_id=%s AND id=%s",
            (_now_iso(), self.user_id, character_id))

    def set_session_wearer(self, session_id: str, character_id: str) -> None:
        """本会话佩戴者指针:SW 链落成的那个档案。personos 无 anchor_session 表,
        折进该档 payload.wearer_sessions(per-user 持久,不新增表)。"""
        ch = self.get_character(character_id)
        if not ch:
            return
        pl = ch["payload"]
        sessions = pl.setdefault("wearer_sessions", [])
        if session_id and session_id not in sessions:
            sessions.append(session_id)
        self.db.execute(
            "UPDATE characters SET payload=%s, updated_at=%s WHERE user_id=%s AND id=%s",
            (json.dumps(pl, ensure_ascii=False), _now_iso(), self.user_id, character_id))

    def session_wearer(self, session_id: str) -> Optional[str]:
        """本会话 SW 链落成的档案 id(从 wearer 档 payload.wearer_sessions 反查)。"""
        for ch in self.list_active_characters(include_wearer=True):
            if not ch["is_wearer"]:
                continue
            if session_id in (ch["payload"].get("wearer_sessions") or []):
                return ch["id"]
        return None

    # ── name claims(折进 payload;台账只追加,primary_name 取 cnt 最高)──
    def add_name_claim(self, character_id: str, name: str, evidence: str = "") -> None:
        ch = self.get_character(character_id)
        if not ch or not name:
            return
        pl = ch["payload"]
        claims = pl.setdefault("name_claims", [])
        for c in claims:
            if c["name"] == name:
                c["cnt"] = int(c.get("cnt", 1)) + 1
                break
        else:
            claims.append({"name": name, "evidence": evidence, "cnt": 1})
        primary = max(claims, key=lambda c: int(c.get("cnt", 1)))["name"]
        self.db.execute(
            "UPDATE characters SET payload=%s, primary_name=%s, updated_at=%s"
            " WHERE user_id=%s AND id=%s",
            (json.dumps(pl, ensure_ascii=False), primary, _now_iso(), self.user_id, character_id))

    def names_for(self, character_id: str) -> list[str]:
        ch = self.get_character(character_id)
        if not ch:
            return []
        claims = sorted(ch["payload"].get("name_claims", []),
                        key=lambda c: (-int(c.get("cnt", 1)), c["name"]))
        return [c["name"] for c in claims]

    # ── assets(crop 字节在 OSS,库存 key + 向量 + 元数据)──────────────
    def add_asset(self, character_id: str, kind: str, *, quality: float,
                  embedding: np.ndarray | None = None,
                  payload: dict[str, Any] | None = None) -> str:
        """kind ∈ face/body/voice。payload 携带 session/clip/t0/t1、crop 的 OSS key、descriptor。"""
        aid = _ulid("asset")
        dim = int(embedding.shape[0]) if embedding is not None else None
        self.db.execute(
            "INSERT INTO character_assets(id, user_id, character_id, kind, quality, embedding,"
            " dim, status, payload, created_at)"
            " VALUES (%s,%s,%s,%s,%s,UNHEX(%s),%s,'active',%s,%s)",
            (aid, self.user_id, character_id, kind, float(quality),
             blob_param(emb_to_bytes(embedding)), dim,
             json.dumps(payload or {}, ensure_ascii=False), _now_iso()))
        return aid

    def active_assets(self, character_id: str, kind: str) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT id, character_id, kind, quality, HEX(embedding) AS emb_hex, dim, status,"
            " payload, created_at FROM character_assets"
            " WHERE user_id=%s AND character_id=%s AND kind=%s AND status='active'"
            " ORDER BY quality DESC", (self.user_id, character_id, kind))
        return [self._asset_row(r) for r in rows]

    def _asset_row(self, row) -> dict[str, Any]:
        d = dict(row)
        d["embedding"] = bytes_to_emb(blob_of(d.pop("emb_hex")))
        d["payload"] = json.loads(d.get("payload") or "{}")
        return d

    def best_asset(self, character_id: str, kind: str) -> Optional[dict[str, Any]]:
        """某 kind 质量最高的 active asset(候选卡/终审材料取素材;active_assets 已按 quality DESC)。"""
        assets = self.active_assets(character_id, kind)
        return assets[0] if assets else None

    def retire_asset(self, asset_id: str) -> None:
        """retire≠delete:向量与出处保留,只是不再进候选卡。"""
        self.db.execute(
            "UPDATE character_assets SET status='retired' WHERE user_id=%s AND id=%s",
            (self.user_id, asset_id))

    def enforce_asset_caps(self, character_id: str, *, face_cap: int, session_face_cap: int,
                           voice_cap: int) -> int:
        """容量结算(对齐 mneme enforce_asset_caps):脸 per-session 席位 + 全局上限,声音全局上限,
        溢出按 quality 序 retire。body 跟随同 session/clip 的脸(脸退它也退)。retire≠delete。
        返回 retire 数。"""
        retired = 0
        faces = self.active_assets(character_id, "face")   # 已按 quality DESC
        per_session: dict[str, int] = {}
        keep_pairs: set[tuple[str, Any]] = set()           # (session, clip) 保留的脸帧,body 据此跟随
        for a in faces:
            pl = a["payload"]
            sid = str(pl.get("session", ""))
            if per_session.get(sid, 0) >= session_face_cap or len(keep_pairs) >= face_cap:
                self.retire_asset(a["id"]); retired += 1
                continue
            per_session[sid] = per_session.get(sid, 0) + 1
            keep_pairs.add((sid, pl.get("clip")))
        for a in self.active_assets(character_id, "body"):
            pl = a["payload"]
            if (str(pl.get("session", "")), pl.get("clip")) not in keep_pairs:
                self.retire_asset(a["id"]); retired += 1
        for a in self.active_assets(character_id, "voice")[voice_cap:]:
            self.retire_asset(a["id"]); retired += 1
        return retired

    # ── 概率云:prototype(slot=prototype,每 char/modality 一行)──────
    def load_prototype(self, character_id: str,
                       modality: str) -> tuple[np.ndarray | None, float, int]:
        row = self.db.fetch_one(
            "SELECT HEX(embedding) AS emb_hex, payload FROM character_cloud"
            " WHERE user_id=%s AND character_id=%s AND modality=%s AND slot='prototype'",
            (self.user_id, character_id, modality))
        if not row:
            return None, 0.0, 0
        pl = json.loads(row["payload"] or "{}")
        return (bytes_to_emb(blob_of(row["emb_hex"])),
                float(pl.get("tau", 0.0)), int(pl.get("n_obs", 0)))

    def save_prototype(self, character_id: str, modality: str,
                       mean: np.ndarray | None, tau: float, n_obs: int) -> None:
        pl = json.dumps({"tau": float(tau), "n_obs": int(n_obs)}, ensure_ascii=False)
        dim = int(mean.shape[0]) if mean is not None else None
        now = _now_iso()
        exist = self.db.fetch_one(
            "SELECT id FROM character_cloud WHERE user_id=%s AND character_id=%s"
            " AND modality=%s AND slot='prototype'", (self.user_id, character_id, modality))
        if exist:
            self.db.execute(
                "UPDATE character_cloud SET embedding=UNHEX(%s), dim=%s, payload=%s, updated_at=%s"
                " WHERE id=%s", (blob_param(emb_to_bytes(mean)), dim, pl, now, exist["id"]))
        else:
            self.db.execute(
                "INSERT INTO character_cloud(id, user_id, character_id, modality, slot, embedding,"
                " dim, payload, updated_at) VALUES (%s,%s,%s,%s,'prototype',UNHEX(%s),%s,%s,%s)",
                (_ulid("cld"), self.user_id, character_id, modality,
                 blob_param(emb_to_bytes(mean)), dim, pl, now))
        # 冗余快照:把成熟度 τ 镜像到 characters,方便展示/排序,不作为真相源
        col = {"face": "face_tau", "voice": "voice_tau"}.get(modality)
        if col:
            self.db.execute(
                f"UPDATE characters SET {col}=%s, updated_at=%s WHERE user_id=%s AND id=%s",
                (float(tau), now, self.user_id, character_id))

    # ── 概率云:templates(slot=template,多行多样性范例)─────────────
    def templates(self, character_id: str, modality: str) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT id, HEX(embedding) AS emb_hex, payload FROM character_cloud"
            " WHERE user_id=%s AND character_id=%s AND modality=%s AND slot='template'",
            (self.user_id, character_id, modality))
        out = []
        for r in rows:
            pl = json.loads(r["payload"] or "{}")
            out.append({"template_id": r["id"],
                        "embedding": bytes_to_emb(blob_of(r["emb_hex"])),
                        "q": float(pl.get("q", 0.0)), **pl})
        return out

    def add_template(self, character_id: str, modality: str, embedding: np.ndarray, q: float,
                     *, payload: dict[str, Any] | None = None) -> str:
        tid = _ulid("cld")
        pl = {"q": float(q), **(payload or {})}
        self.db.execute(
            "INSERT INTO character_cloud(id, user_id, character_id, modality, slot, embedding,"
            " dim, payload, updated_at) VALUES (%s,%s,%s,%s,'template',UNHEX(%s),%s,%s,%s)",
            (tid, self.user_id, character_id, modality, blob_param(emb_to_bytes(embedding)),
             int(embedding.shape[0]), json.dumps(pl, ensure_ascii=False), _now_iso()))
        return tid

    def remove_template(self, template_id: str) -> None:
        self.db.execute(
            "DELETE FROM character_cloud WHERE user_id=%s AND id=%s AND slot='template'",
            (self.user_id, template_id))
