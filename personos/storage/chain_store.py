"""atom 链存储(docs/atom-chain-design.md §3):链级信息 + atoms 三列的生命周期。

链是派生视图、只分组不消解(D-C3):链内新旧陈述不选赢家,冲突消费在作答时。
成员关系不放独立表——与 atom 严格 1:1,atoms 链三列(单值 chain_id)即
「一 atom 至多一链」的数据库级约束;occurrence/recorded 时间本就在 atom payload。

链序 = 追加序 = 对话事实抽取的自然顺序(D-C7):新成员追加到链尾
(member.prev=旧尾),双向链接是链序唯一事实源;occurrence_time 只作展示元数据,
不用来排链序(回溯提及会让它乱序、并列时不稳定)。

多租户:与 AtomStore 同规,实例按 user 绑定(构造注入 user_id)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from ..models import ChainInfo, MemoryAtom, now
from .db import Database, _Tx, blob_of, blob_param

_INSERT_CHAIN = (
    "INSERT INTO atom_chains(id, user_id, n_atoms, head_atom_id, tail_atom_id, "
    "centroid, payload, created_at, updated_at) VALUES(%s,%s,%s,%s,%s,UNHEX(%s),%s,%s,%s)"
)
_UPDATE_CHAIN = (
    "UPDATE atom_chains SET n_atoms=%s, head_atom_id=%s, tail_atom_id=%s, "
    "centroid=UNHEX(%s), payload=%s, updated_at=%s WHERE id=%s AND user_id=%s"
)
# 链列 + payload 双写(仿 memcell_id 先例);调用方保证 atom 行存在
_SET_LINK = (
    "UPDATE atoms SET chain_id=%s, prev_atom_id=%s, next_atom_id=%s, payload=%s "
    "WHERE id=%s AND user_id=%s"
)


def _centroid_blob(c: Optional[np.ndarray]) -> Optional[str]:
    return blob_param(np.asarray(c, dtype=np.float32).tobytes()) if c is not None else None


class ChainStore:
    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    # —— 内部:链列双写(读当前 payload → 校验 → 改链字段 → 回写)——

    def _set_link(self, tx: _Tx, atom_id: str, *, chain_id: str, next_: str,
                  prev: Optional[str], expect_chain: str) -> None:
        """写一个节点的链字段。expect_chain=该节点当前应处的链(''=应游离)——
        不符即抛错回滚:防重复挂链/挂错链把链表结构写坏(结构性错误宁死不改)。
        prev=None 表示保留现值(旧尾接 next 时前驱不能动);''/str 则显式设定。"""
        row = tx.fetch_one(
            "SELECT payload, chain_id, prev_atom_id FROM atoms WHERE id=%s AND user_id=%s",
            (atom_id, self.user_id),
        )
        if row is None:
            raise RuntimeError(f"链挂接目标 atom 不存在: {atom_id}")
        if (row["chain_id"] or "") != expect_chain:
            raise RuntimeError(
                f"链挂接冲突: atom {atom_id} 已在链 {row['chain_id'] or '(无)'} "
                f"(期望 {expect_chain or '(游离)'}) —— 双挂/错挂,整答回滚"
            )
        atom = MemoryAtom.model_validate_json(row["payload"])
        new_prev = (row["prev_atom_id"] or "") if prev is None else prev  # 列为事实源
        atom.chain_id, atom.prev_atom_id, atom.next_atom_id = chain_id, new_prev, next_
        tx.execute(_SET_LINK, (chain_id or None, new_prev or None, next_ or None,
                               atom.model_dump_json(), atom_id, self.user_id))

    # —— 建链 / 追加(各一个事务;append 内三写原子)——

    def create_chain(self, info: ChainInfo, first_atom: MemoryAtom,
                     centroid: Optional[np.ndarray] = None) -> ChainInfo:
        """建链 + 首成员(链首=链尾=first_atom,n=1)。info 的计数字段由此填定。"""
        info.user_id = self.user_id
        info.n_atoms, info.head_atom_id, info.tail_atom_id = 1, first_atom.id, first_atom.id
        info.created_at = info.updated_at = now()
        with self.db.transaction() as tx:
            self._set_link(tx, first_atom.id, chain_id=info.id, prev="", next_="", expect_chain="")
            tx.execute(_INSERT_CHAIN, (
                info.id, self.user_id, info.n_atoms, info.head_atom_id, info.tail_atom_id,
                _centroid_blob(centroid), info.model_dump_json(),
                info.created_at.isoformat(), info.updated_at.isoformat(),
            ))
        logger.info(f"链建链 id={info.id} first={first_atom.id} title={info.title[:30]!r} "
                    f"user={self.user_id or '(默认)'}")
        return info

    def append_atom(self, chain: ChainInfo, atom: MemoryAtom,
                    centroid: Optional[np.ndarray] = None) -> ChainInfo:
        """追加到链尾:旧尾接 next、新成员写 prev、链行 tail/n/centroid 前移(单事务)。

        chain 对象就地更新并返回,调用方持引用即最新态。
        centroid=None 表示保留现质心(否则须传"含新成员"的全量质心——增量均值在
        chain_build 里算,精确重算走 recompute)。
        """
        if not chain.tail_atom_id:
            raise RuntimeError(f"链 {chain.id} 无尾(空链不可追加,应走 create_chain)")
        if centroid is None:   # 保留现值,防 UNHEX(NULL) 把质心清掉
            row = self.db.fetch_one(
                "SELECT HEX(centroid) AS c FROM atom_chains WHERE id=%s AND user_id=%s",
                (chain.id, self.user_id),
            )
            c = blob_of(row["c"]) if row else None
            centroid = np.frombuffer(c, dtype=np.float32) if c else None
        with self.db.transaction() as tx:
            self._set_link(tx, chain.tail_atom_id, chain_id=chain.id, prev=None,
                           next_=atom.id, expect_chain=chain.id)
            self._set_link(tx, atom.id, chain_id=chain.id, prev=chain.tail_atom_id,
                           next_="", expect_chain="")
            chain.n_atoms += 1
            chain.tail_atom_id = atom.id
            chain.updated_at = now()
            tx.execute(_UPDATE_CHAIN, (
                chain.n_atoms, chain.head_atom_id, chain.tail_atom_id,
                _centroid_blob(centroid), chain.model_dump_json(),
                chain.updated_at.isoformat(), chain.id, self.user_id,
            ))
        return chain

    # —— 读 ——

    def get_chain(self, chain_id: str) -> Optional[ChainInfo]:
        row = self.db.fetch_one(
            "SELECT payload FROM atom_chains WHERE id=%s AND user_id=%s",
            (chain_id, self.user_id),
        )
        return ChainInfo.model_validate_json(row["payload"]) if row else None

    def list_chains(self) -> list[tuple[ChainInfo, Optional[np.ndarray]]]:
        """本 user 全部链 + 质心(判链预筛的候选池;链数 ≪ atom 数)。"""
        rows = self.db.fetch_all(
            "SELECT payload, HEX(centroid) AS c FROM atom_chains WHERE user_id=%s",
            (self.user_id,),
        )
        out = []
        for r in rows:
            c = blob_of(r["c"])
            out.append((ChainInfo.model_validate_json(r["payload"]),
                        np.frombuffer(c, dtype=np.float32) if c else None))
        return out

    def full_chain(self, chain_id: str) -> list[MemoryAtom]:
        """整链成员(链序):一条索引查询取回,本地沿 next 链接从链首走出顺序。

        双向链接是链序唯一事实源;返回顺序即链序(追加序)。链行缺失/链断裂 →
        返回已走通的前缀并告警(派生索引受损不抛崩,留重建脚本收口)。
        """
        info = self.get_chain(chain_id)
        if info is None:
            return []
        rows = self.db.fetch_all(
            "SELECT payload, chain_id, prev_atom_id, next_atom_id FROM atoms "
            "WHERE user_id=%s AND chain_id=%s",
            (self.user_id, chain_id),
        )
        by_id: dict[str, MemoryAtom] = {}
        for r in rows:
            a = MemoryAtom.model_validate_json(r["payload"])
            a.chain_id = r["chain_id"] or ""
            a.prev_atom_id = r["prev_atom_id"] or ""
            a.next_atom_id = r["next_atom_id"] or ""
            by_id[a.id] = a
        ordered, cur, seen = [], info.head_atom_id, set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            ordered.append(by_id[cur])
            cur = by_id[cur].next_atom_id
        if len(ordered) != len(by_id) or info.n_atoms != len(by_id):
            logger.warning(f"链 {chain_id} 结构不一致: 行={len(by_id)} 走通={len(ordered)} "
                           f"登记 n={info.n_atoms} (游离断链待 rebuild 收口)")
        return ordered

    # —— 重建 / 修复 ——

    def clear_user(self) -> int:
        """重建前置:清本 user 全部链(链行删除 + atoms 链三列/payload 链字段归零)。

        均带 WHERE(测试护栏拦 TRUNCATE/裸 DELETE);返回清掉的链数。
        payload 链字段一并抹掉——列与副本同源归零,不留半旧状态。
        """
        chains = self.list_chains()
        with self.db.transaction() as tx:
            for info, _ in chains:
                rows = tx.fetch_all(
                    "SELECT id, payload FROM atoms WHERE user_id=%s AND chain_id=%s",
                    (self.user_id, info.id),
                )
                for r in rows:
                    a = MemoryAtom.model_validate_json(r["payload"])
                    a.chain_id = a.prev_atom_id = a.next_atom_id = ""
                    tx.execute(_SET_LINK, (None, None, None, a.model_dump_json(),
                                           r["id"], self.user_id))
                tx.execute("DELETE FROM atom_chains WHERE id=%s AND user_id=%s",
                           (info.id, self.user_id))
        logger.info(f"链清理 user={self.user_id or '(默认)'} chains={len(chains)}")
        return len(chains)

    def recompute(self, chain_id: str) -> Optional[ChainInfo]:
        """按成员向量重算 centroid 并对齐 n_atoms(回填/修复用;增量质心的浮点漂移在此收口)。"""
        info = self.get_chain(chain_id)
        if info is None:
            return None
        rows = self.db.fetch_all(
            "SELECT id, HEX(embedding) AS emb FROM atoms "
            "WHERE user_id=%s AND chain_id=%s AND embedding IS NOT NULL",
            (self.user_id, chain_id),
        )
        if rows:
            mat = np.stack([np.frombuffer(blob_of(r["emb"]), dtype=np.float32) for r in rows])
            centroid = mat.mean(axis=0)
        else:
            centroid = None
        info.n_atoms = len(rows)
        info.updated_at = now()
        self.db.execute(_UPDATE_CHAIN, (
            info.n_atoms, info.head_atom_id, info.tail_atom_id,
            _centroid_blob(centroid), info.model_dump_json(),
            info.updated_at.isoformat(), info.id, self.user_id,
        ))
        return info
