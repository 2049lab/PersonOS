"""会话草稿(⑤):身份链 / roster / 评估台账 / 暂存素材的会话态存取。

移植 mneme AnchorStore 的 chain/roster 段(逐处对齐:union-find canonical、presence 累积、
评估台账 append、commit_chain 标记),但**有意偏离**:mneme 这四类会话态在单 SQLite,personos
禁 sqlite——落 Redis(key 经 redis_client.key() 收口,带滑动 TTL),持久层仍是 MySQL CharacterStore。

素材偏离:harvest 出的 crop 即时传 OSS,草稿只存 oss_key + 归一化向量 + 质量(不内联 b64,Redis 不膨胀);
仲裁卡按需从 OSS 取 b64。commit 时赢家 staged→character_assets 行 + cloud.learn(见 commit.py)。

存储布局:一个会话一个 STRING 键存整块 state JSON(chains/evals/roster/staged),读改写在会话写锁内
(身份处理已持 session_lock),单命令 SET ... EX 原子写值+TTL(对齐 seg_store 踩坑:corvus 不走
pipeline/Lua)。单键单命令,无多键 → 不需 hash-tag。

原则沿用 mneme:chain 只维护"当前假设"(hypothesis),全局归属推迟到会话末终审(commit.py);
canonical=None 为链根,别名指向根(合并时压平一层深),pending=未提交且是链根。
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

import numpy as np
from loguru import logger

from personos.identity.types import CastEvidence
from personos.storage.redis_client import key as _redis_key

# 会话草稿滑动 TTL:每次写续期;一个会话超过一天没有下一 clip,草稿作废(未提交链丢失=可接受,
# 终审只处理仍在的 pending 链,重跑兜底)。与 seg_store SEG_TTL_S 同量级。
IDDRAFT_TTL_S = 24 * 3600

# chain 可更新字段白名单(逐字对齐 mneme AnchorStore._CHAIN_FIELDS)。
_CHAIN_FIELDS = frozenset({"canonical", "status", "hypothesis", "hypo_method", "best_face_q",
                           "best_voice_q", "named", "desc_text", "presence", "final_character_id"})


def read_b64(media_store: Any, oss_key: str) -> str:
    """OSS 素材 → b64(取素材失败返回空串:该角度缺失仍可凭其他角度判,不阻塞)。
    身份层各处(候选卡/query卡/roster)共用,收口在最底层的 draft 模块。"""
    if not oss_key or media_store is None:
        return ""
    try:
        return base64.b64encode(media_store.read_bytes(oss_key)).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"取素材失败 {oss_key}: {e}")
        return ""


def _emb_list(emb: Optional[np.ndarray]) -> Optional[list[float]]:
    return None if emb is None else np.asarray(emb, dtype=np.float32).tolist()


def _emb_arr(x: Any) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float32)


class DraftStore:
    """会话草稿存取基类:所有链/roster/评估/素材逻辑在此(纯 state dict 操作),
    IO 由子类的 _read/_write 提供。实例绑定 user_id(圈死本用户社交圈,对齐 CharacterStore)。"""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    # ── 子类实现:整块 state 的读/写(按 session 隔离)──────────────────
    def _read(self, session_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def _blank_state() -> dict[str, Any]:
        return {"chains": {}, "evals": {}, "roster": {}, "staged": {}, "lines": [],
                "clip_seq": 0, "clip_keys": {}, "clips_done": {}}

    def next_clip_seq(self, session_id: str, clip_key: str = "") -> int:
        """会话全局单调 clip 序号(每条视频消息 +1)+ 记 clip_index→oss_key。消费侧据此定 clip_index,
        不信调用方——变态调用方(文→clips→文→clips)每批从 0 重编会撞 presence/lines,全局序号避免。"""
        state = self._read(session_id)
        seq = int(state.get("clip_seq", 0))
        state["clip_seq"] = seq + 1
        if clip_key:
            state.setdefault("clip_keys", {})[str(seq)] = clip_key
        self._write(session_id, state)
        return seq

    def clip_keys(self, session_id: str) -> dict[int, str]:
        """会话内 clip_index→oss_key(供 flush 时 raw_clip 证据回链)。"""
        return {int(k): v for k, v in self._read(session_id).get("clip_keys", {}).items()}

    # ── clip 级幂等 ────────────────────────────────────────────────
    # 队列重投是**整条消息**重放的(一条消息最多 20 个 clip),而 clip 处理会累积 presence/
    # lines/staged 素材。若不去重,批内第 19 个 clip 遇到瞬时故障就会让前 18 个重跑一遍,
    # 出场记两遍、台词进两次 memcell、素材传两份——不是浪费算力,是**记错**。
    #
    # 去重必须挂在「已完成」上,不能挂在 next_clip_seq 的序号分配上:序号是在剧本 MLLM
    # **之前**分配的,挂在那儿会让"剧本失败的 clip"在重投时被误判成已处理而跳过,
    # 把重复记账换成静默丢记忆——更糟。故单独记一份 clip_key→已完成的 clip_index。
    def clip_done_index(self, session_id: str, clip_key: str) -> Optional[int]:
        """该 clip 是否已**完整处理**过;是则返回它当时的 clip_index,否则 None。"""
        if not clip_key:
            return None
        v = self._read(session_id).get("clips_done", {}).get(clip_key)
        return int(v) if v is not None else None

    def mark_clip_done(self, session_id: str, clip_key: str, clip_index: int) -> None:
        """clip 全部环节落草稿后调用;此后同 key 重投即跳过。"""
        if not clip_key:
            return
        state = self._read(session_id)
        state.setdefault("clips_done", {})[clip_key] = int(clip_index)
        self._write(session_id, state)

    # ── chain_ref 约定(与 mneme 一致)────────────────────────────────
    @staticmethod
    def chain_ref(session_id: str, cast_id: str) -> str:
        return f"chain:{session_id}:{cast_id}"

    @staticmethod
    def is_chain_ref(ref: str) -> bool:
        return ref.startswith("chain:")

    @staticmethod
    def _session_of(chain_ref: str) -> str:
        parts = chain_ref.split(":")
        return parts[1] if len(parts) == 3 and parts[0] == "chain" else ""

    @staticmethod
    def _default_chain(session_id: str, cast_id: str) -> dict[str, Any]:
        """新链初始态(逐字对齐 mneme chain 表 DEFAULT)。"""
        return {"chain_ref": DraftStore.chain_ref(session_id, cast_id),
                "session_id": session_id, "cast_id": cast_id,
                "canonical": None, "status": "pending", "hypothesis": "NEW",
                "hypo_method": "first_seen", "best_face_q": -1.0, "best_voice_q": -1.0,
                "named": 0, "desc_text": "", "presence": [], "final_character_id": None}

    # ── chains ────────────────────────────────────────────────────────
    def ensure_chain(self, session_id: str, cast_id: str) -> tuple[dict[str, Any], bool]:
        ref = self.chain_ref(session_id, cast_id)
        state = self._read(session_id)
        row = state["chains"].get(ref)
        if row is not None:
            return dict(row), False
        row = self._default_chain(session_id, cast_id)
        state["chains"][ref] = row
        self._write(session_id, state)
        return dict(row), True

    def get_chain(self, chain_ref: str) -> Optional[dict[str, Any]]:
        row = self._read(self._session_of(chain_ref))["chains"].get(chain_ref)
        return dict(row) if row else None

    def canonical_chain(self, chain_ref: str) -> str:
        """跟随 canonical 到链根,带环路守卫(merge_chain 已拒环,双保险)。"""
        chains = self._read(self._session_of(chain_ref))["chains"]
        seen: set[str] = set()
        current = chain_ref
        while current not in seen:
            seen.add(current)
            row = chains.get(current)
            if row is None or not row.get("canonical"):
                return current
            current = row["canonical"]
        return current

    def update_chain(self, chain_ref: str, **fields: Any) -> None:
        unknown = set(fields) - _CHAIN_FIELDS
        if unknown:
            raise ValueError(f"unknown chain fields: {sorted(unknown)}")
        session_id = self._session_of(chain_ref)
        state = self._read(session_id)
        row = state["chains"].get(chain_ref)
        if row is None:
            raise ValueError(f"update unknown chain: {chain_ref}")
        row.update(fields)
        self._write(session_id, state)

    def merge_chain(self, src_ref: str, dst_ref: str) -> None:
        """src 并入 dst:src.canonical=dst 根,压平 src 的既有别名到 dst 根(保持一层深,
        commit 单层收集依赖此),证据(best_*/named/presence)并入 dst 根。逐字对齐 mneme。"""
        dst = self.canonical_chain(dst_ref)
        if self.canonical_chain(src_ref) == dst:
            return
        src_row, dst_row = self.get_chain(src_ref), self.get_chain(dst)
        if src_row is None or dst_row is None:
            raise ValueError(f"merge unknown chain: {src_ref} -> {dst_ref}")
        session_id = self._session_of(src_ref)
        state = self._read(session_id)
        state["chains"][src_ref]["canonical"] = dst
        for ref, row in state["chains"].items():
            if row.get("canonical") == src_ref:
                row["canonical"] = dst
        dst_state_row = state["chains"][dst]
        dst_state_row["best_face_q"] = max(src_row["best_face_q"], dst_row["best_face_q"])
        dst_state_row["best_voice_q"] = max(src_row["best_voice_q"], dst_row["best_voice_q"])
        dst_state_row["named"] = max(src_row["named"], dst_row["named"])
        dst_state_row["presence"] = sorted(set(src_row["presence"]) | set(dst_row["presence"]))
        self._write(session_id, state)

    def aliases_of(self, chain_ref: str) -> list[str]:
        chains = self._read(self._session_of(chain_ref))["chains"]
        return sorted(ref for ref, row in chains.items() if row.get("canonical") == chain_ref)

    def pending_chains(self, session_id: str) -> list[dict[str, Any]]:
        """未提交且是链根(canonical=None)的链,按 cast_id 序。"""
        chains = self._read(session_id)["chains"]
        rows = [dict(r) for r in chains.values()
                if r["status"] == "pending" and not r.get("canonical")]
        return sorted(rows, key=lambda r: r["cast_id"])

    def commit_chain(self, canonical_ref: str, final_character_id: str) -> None:
        """终审提交(草稿侧):链及其别名标 committed + final_character_id。
        素材/名字的实际落库由 commit.py 走 CharacterStore(personos 无 anchor_line 表,
        故 mneme commit_chain 的 line/asset 迁移不在此)。"""
        session_id = self._session_of(canonical_ref)
        state = self._read(session_id)
        for ref in (canonical_ref, *[r for r, row in state["chains"].items()
                                     if row.get("canonical") == canonical_ref]):
            row = state["chains"].get(ref)
            if row is not None:
                row["status"] = "committed"
                row["final_character_id"] = final_character_id
        self._write(session_id, state)

    # ── chain 评估台账(append-only)─────────────────────────────────
    def add_chain_evaluation(self, chain_ref: str, *, session_id: str, clip_index: int,
                             reason: str, verdict: Optional[str],
                             evidence: Optional[dict[str, Any]] = None,
                             issues: Optional[list[str]] = None) -> None:
        state = self._read(session_id)
        state["evals"].setdefault(chain_ref, []).append(
            {"chain_ref": chain_ref, "session_id": session_id, "clip_index": clip_index,
             "reason": reason, "verdict": verdict, "evidence": evidence or {},
             "issues": issues or []})
        self._write(session_id, state)

    def evaluations_for(self, chain_ref: str) -> list[dict[str, Any]]:
        return list(self._read(self._session_of(chain_ref))["evals"].get(chain_ref, []))

    # ── roster(会话续接名册)─────────────────────────────────────────
    def load_roster(self, session_id: str) -> dict[str, dict[str, Any]]:
        roster: dict[str, dict[str, Any]] = {}
        for cast_id, entry in self._read(session_id)["roster"].items():
            card = dict(entry.get("card") or {})
            card["character_id"] = entry.get("character_id")
            roster[cast_id] = card
        return roster

    def names_for(self, chain_ref: str) -> list[str]:
        """会话内链名:从 roster 卡取(chain_ref→cast_id→roster[cast_id].name)。
        持久档的名字在 CharacterStore.names_for;链名会话末归并进档(commit.py)。"""
        cast_id = chain_ref.split(":")[-1]
        card = self._read(self._session_of(chain_ref))["roster"].get(cast_id, {}).get("card") or {}
        name = card.get("name")
        return [name] if name else []

    def save_roster_entry(self, session_id: str, cast_id: str, *,
                          character_id: Optional[str], card: dict[str, Any]) -> None:
        payload = {k: v for k, v in card.items() if k != "character_id"}
        state = self._read(session_id)
        state["roster"][cast_id] = {"character_id": character_id, "card": payload}
        self._write(session_id, state)

    # ── 剧本行缓冲(会话末 flush 成带人物归属的 evidence,见 online/video_memory)──
    def stage_lines(self, session_id: str, clip_index: int,
                    lines: list[tuple[float, float, str, str, str]]) -> None:
        """缓存一个 clip 的剧本行。lines=[(t0,t1,who,kind,text)],who=已映射的会话 cast id
        (cast_map.get(line.who))/ 'SW' / 'ENV'(env 行)。会话内追加,时序由 clip_index+t0 定。"""
        state = self._read(session_id)
        for t0, t1, who, kind, text in lines:
            state["lines"].append({"clip": clip_index, "t0": float(t0), "t1": float(t1),
                                   "who": who, "kind": kind, "text": text})
        self._write(session_id, state)

    def all_lines(self, session_id: str) -> list[dict[str, Any]]:
        """全 session 剧本行,按 (clip_index, t0) 时序返回(供 commit flush)。"""
        return sorted(self._read(session_id)["lines"], key=lambda r: (r["clip"], r["t0"]))

    # ── 暂存素材(crop 走 OSS,草稿只存 key+向量+质量)──────────────────
    def stage_evidence(self, chain_ref: str, *, session_id: str, clip_index: int,
                       evidence: CastEvidence, media_store: Any) -> None:
        """把一个 cast 本 clip 的证据素材暂存:crop 即时传 OSS,元数据(oss_key/向量/q/t)入草稿。
        供仲裁卡取素材 + 终审时赢家学云入库(commit.py)。"""
        state = self._read(session_id)
        st = state["staged"].setdefault(chain_ref, {"face": [], "body": [], "voice": []})
        for f in evidence.faces:
            if f.embedding is None and not f.crop_b64:
                continue
            st["face"].append({
                "oss_key": self._save_img(media_store, f.crop_b64, "image/png"),
                "emb": _emb_list(f.embedding), "q": float(f.q), "t": float(f.t),
                "clip": clip_index, "session": session_id, "descriptor": f.descriptor})
            body_key = self._save_img(media_store, f.body_crop_b64, "image/jpeg")
            if body_key:
                st["body"].append({"oss_key": body_key, "q": float(f.q), "t": float(f.t),
                                   "clip": clip_index, "session": session_id})
        for v in evidence.voices:
            if v.embedding is None:
                continue
            st["voice"].append({
                "oss_key": self._save_audio(media_store, v.wav_bytes),
                "emb": _emb_list(v.embedding), "q": float(v.q),
                "t0": float(v.t0), "t1": float(v.t1),
                "clip": clip_index, "session": session_id})
        self._write(session_id, state)

    def active_staged(self, chain_ref: str, kind: str) -> list[dict[str, Any]]:
        """某 kind 的暂存素材,按 q 降序(终审学云/入库用;emb 还原成 np 向量)。"""
        items = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {}).get(kind, [])
        out = [{**it, "embedding": _emb_arr(it.get("emb"))} for it in items]
        return sorted(out, key=lambda a: a["q"], reverse=True)

    def best_pair(self, chain_ref: str) -> Optional[dict[str, Any]]:
        """质量最高的脸 + 最佳全身(仲裁卡素材)。返回 oss_key(卡侧再从 OSS 取 b64)。"""
        staged = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {})
        faces = staged.get("face") or []
        bodies = staged.get("body") or []
        if not faces and not bodies:
            return None
        best_face = max(faces, key=lambda a: a["q"], default=None)
        best_body = max(bodies, key=lambda a: a["q"], default=None)
        return {"face_oss_key": (best_face or {}).get("oss_key", ""),
                "body_oss_key": (best_body or {}).get("oss_key", ""),
                "quality": (best_face or best_body or {}).get("q", -1.0)}

    def best_voice(self, chain_ref: str) -> Optional[dict[str, Any]]:
        voices = self._read(self._session_of(chain_ref))["staged"].get(chain_ref, {}).get("voice") or []
        if not voices:
            return None
        best = max(voices, key=lambda a: a["q"])
        return {"oss_key": best.get("oss_key", ""), "quality": best["q"]}

    # ── OSS 落地(素材偏离:crop 即时进 OSS,失败不阻塞——向量仍在草稿)──
    def _save_img(self, media_store: Any, b64: str, content_type: str) -> str:
        if media_store is None or not b64:
            return ""
        try:
            return media_store.save_image(base64.b64decode(b64), owner=self.user_id,
                                          content_type=content_type).key
        except Exception as e:  # noqa: BLE001
            logger.warning(f"草稿素材存 OSS 失败: {e}")
            return ""

    def _save_audio(self, media_store: Any, wav_bytes: Optional[bytes]) -> str:
        if media_store is None or not wav_bytes:
            return ""
        try:
            return media_store.save_audio(wav_bytes, owner=self.user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"草稿声纹 wav 存 OSS 失败: {e}")
            return ""


class RedisDraftStore(DraftStore):
    """生产实现:corvus 上一个会话一个 STRING 键存整块 state JSON。

    key = {env}:personos:iddraft:{user}:{session};SET ... EX 单命令原子写值+TTL
    (不走 pipeline/Lua——对齐 seg_store:corvus 对 pipeline 会错位串包)。
    """

    def __init__(self, client, user_id: str, ttl_s: int = IDDRAFT_TTL_S) -> None:
        super().__init__(user_id)
        self._c = client
        self._ttl = ttl_s

    def _k(self, session_id: str) -> str:
        return _redis_key("iddraft", self.user_id, session_id)

    def _read(self, session_id: str) -> dict[str, Any]:
        raw = self._c.get(self._k(session_id))
        return json.loads(raw) if raw else self._blank_state()

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        self._c.set(self._k(session_id), json.dumps(state, ensure_ascii=False), ex=self._ttl)

    def clear(self, session_id: str) -> None:
        self._c.delete(self._k(session_id))


class MemoryDraftStore(DraftStore):
    """进程内实现(本地脚本/单测/未配 Redis 的单副本)。state 按 (user, session) 存内存。"""

    def __init__(self, user_id: str) -> None:
        super().__init__(user_id)
        self._d: dict[str, dict[str, Any]] = {}

    def _read(self, session_id: str) -> dict[str, Any]:
        return self._d.setdefault(session_id, self._blank_state())

    def _write(self, session_id: str, state: dict[str, Any]) -> None:
        self._d[session_id] = state

    def clear(self, session_id: str) -> None:
        self._d.pop(session_id, None)
