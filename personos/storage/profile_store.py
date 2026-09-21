"""用户画像存储:画像结构模型 + profile_versions 版本表访问。

画像是 atom 层之上的**有损派生视图**(可从 cell 重蒸馏),不是真相源。每次 consolidate
整版留档,当前画像 = 该 user 最新一版(单行读,无双真相源一致性问题)。

多租户:与其它 store 同规,实例按 user 绑定(构造注入 user_id),所有 SQL 自动带
`user_id=%s` 过滤——**用户隔离收敛在此层,严禁跨用户串画像信息**(P2 纪律)。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from ..models import _ulid, now
from .db import Database

# facts 四个时间带(key 固定;分带与淘汰见 online/profile_merge)
BANDS: tuple[str, ...] = ("today", "week", "month", "long")

# PMO-16 维度(traits 的合法 key,英文);按 McAdams 三层分组(见 docs/design/user-profile.md §1)。
# 空维度以 None 表示;整理 agent 只能写这 16 个 key(harness 拒未知域,保画像干净)。
PMO16: tuple[str, ...] = (
    # L1 dispositional traits(跨情境稳定性情)
    "personality", "communication_style", "social_style",
    # L2 characteristic adaptations(情境化的目标·价值·动机·应对)
    "occupation", "goals", "values", "work_style", "learning_style",
    "tech_environment", "lifestyle", "health", "finance",
    # L3 narrative identity(如何理解自己的人生)
    "identity", "location", "family", "interests",
)


class ProfileTrait(BaseModel):
    """a 部分一维:一句话侧写 + 认识状态 + 时效 + 出处。空维度以 None 表示(合法,不编造)。"""
    text: str = ""
    status: Literal["confirmed", "inferred"] = "inferred"
    last_confirmed: str = ""                     # YYYY-MM-DD,引擎盖戳
    sources: list[str] = Field(default_factory=list)   # cell_id 列表


class ProfileFact(BaseModel):
    """b 部分一条:一句(可跨天主题式整合)事实,日期写在正文(自然语言,双时间)。

    带(today/week/month/long)由 LLM 显式决定放哪(不靠结构化日期机械分带)——见设计文档。
    last_confirmed 是引擎盖戳的"最近更新时间",超限淘汰时踢最旧的它;无 event_time 概念。
    """
    id: str = Field(default_factory=lambda: _ulid("f"))
    text: str = ""                               # 事实叙事,日期在正文里
    last_confirmed: str = ""                     # YYYY-MM-DD,引擎盖戳(超限淘汰键)
    sources: list[str] = Field(default_factory=list)   # 真实 cell_id(consolidate 已短→长回填)


class UserProfile(BaseModel):
    """全量画像:traits(PMO-16 域→侧写|None)+ facts(四带→事实列表)。"""
    traits: dict[str, Optional[ProfileTrait]] = Field(default_factory=dict)
    facts: dict[str, list[ProfileFact]] = Field(
        default_factory=lambda: {b: [] for b in BANDS}
    )

    @classmethod
    def empty(cls) -> "UserProfile":
        return cls(traits={}, facts={b: [] for b in BANDS})


@dataclass
class ProfileVersion:
    """一版画像行(带元数据)。"""
    version: int
    profile: UserProfile
    up_to_cell_id: str
    created_at: datetime


class ProfileStore:
    """profile_versions 访问:整版留档,取当前版=最新版。实例按 user 绑定。"""

    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def current(self) -> ProfileVersion | None:
        """当前画像 = 该 user 最新一版;从未整理过 → None(消费侧走"无画像"路径)。"""
        row = self.db.fetch_one(
            "SELECT version, profile_json, up_to_cell_id, created_at "
            "FROM profile_versions WHERE user_id=%s ORDER BY version DESC LIMIT 1",
            (self.user_id,),
        )
        if not row:
            return None
        return ProfileVersion(
            version=int(row["version"]),
            profile=UserProfile.model_validate_json(row["profile_json"]),
            up_to_cell_id=row["up_to_cell_id"] or "",
            created_at=row["created_at"],
        )

    def version_count(self) -> int:
        """已整理过几版 = 画像成熟度(退火进度锚点)。"""
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM profile_versions WHERE user_id=%s", (self.user_id,)
        )
        return int(row["n"]) if row else 0

    def get_version(self, version: int) -> ProfileVersion | None:
        """取指定版本(审计/回滚)。"""
        row = self.db.fetch_one(
            "SELECT version, profile_json, up_to_cell_id, created_at "
            "FROM profile_versions WHERE user_id=%s AND version=%s",
            (self.user_id, version),
        )
        if not row:
            return None
        return ProfileVersion(
            version=int(row["version"]),
            profile=UserProfile.model_validate_json(row["profile_json"]),
            up_to_cell_id=row["up_to_cell_id"] or "",
            created_at=row["created_at"],
        )

    def save_version(self, profile: UserProfile, up_to_cell_id: str) -> int:
        """整版留档,返回新版本号。version 在事务内 MAX+1(uq_user_version 兜底防并发重号)。

        同 user 的 consolidate 已由 per-user 单飞锁串行,这里的事务只是二次护栏。
        """
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                "SELECT COALESCE(MAX(version), 0) AS v FROM profile_versions WHERE user_id=%s",
                (self.user_id,),
            )
            nv = int(row["v"]) + 1
            tx.execute(
                "INSERT INTO profile_versions(id, user_id, version, profile_json, "
                "up_to_cell_id, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (_ulid("pv"), self.user_id, nv, profile.model_dump_json(),
                 up_to_cell_id or "", now().strftime("%Y-%m-%d %H:%M:%S")),
            )
        logger.info(f"画像出版本 user={self.user_id or '(默认)'} version={nv} "
                    f"up_to_cell={up_to_cell_id or '(全量)'} "
                    f"traits={sum(1 for v in profile.traits.values() if v)} "
                    f"facts={sum(len(v) for v in profile.facts.values())}")
        return nv
