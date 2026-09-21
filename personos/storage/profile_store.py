"""User profile storage: the profile's structural model plus access to the
profile_versions table.

A profile is a **lossy derived view** sitting above the atom layer and can be
re-distilled from the cells; it is not a source of truth. Each consolidate
archives a whole version, and the current profile is that user's latest one — a
single-row read, so there is no two-sources-of-truth consistency problem.

Multi-tenancy works as in the other stores: an instance is bound to one user and
every statement carries `user_id=%s`. **User isolation is enforced at this layer,
and profile information must never cross between users.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from ..models import _ulid, now
from .db import Database

# The four time bands for facts. The keys are fixed; how facts are assigned to a
# band and evicted from one lives in online/profile_merge.
BANDS: tuple[str, ...] = ("today", "week", "month", "long")

# The PMO-16 dimensions: the legal keys for traits, grouped into McAdams' three
# levels. An empty dimension is represented by None. The consolidating agent may
# write only these sixteen keys — the harness rejects unknown ones, which is what
# keeps the profile clean.
PMO16: tuple[str, ...] = (
    # L1 dispositional traits: temperament that holds across situations
    "personality", "communication_style", "social_style",
    # L2 characteristic adaptations: situated goals, values, motives and coping
    "occupation", "goals", "values", "work_style", "learning_style",
    "tech_environment", "lifestyle", "health", "finance",
    # L3 narrative identity: how someone makes sense of their own life
    "identity", "location", "family", "interests",
)


class ProfileTrait(BaseModel):
    """One dimension of the traits section: a one-line portrait, how well it is
    known, when it was last confirmed, and where it came from.

    An empty dimension is None, which is legal — better than inventing one.
    """
    text: str = ""
    status: Literal["confirmed", "inferred"] = "inferred"
    last_confirmed: str = ""                     # YYYY-MM-DD, stamped by the engine
    sources: list[str] = Field(default_factory=list)   # a list of cell_ids


class ProfileFact(BaseModel):
    """One entry in the facts section: a single fact, which may thematically
    combine things said across several days, with the date written into the prose
    itself in natural language.

    Which band it goes in (today/week/month/long) is decided explicitly by the LLM
    rather than derived mechanically from a structured date.

    last_confirmed is the "last updated" timestamp the engine stamps on, and it is
    the key used to evict the oldest entry when a band is over its limit. There is
    no separate notion of event time here.
    """
    id: str = Field(default_factory=lambda: _ulid("f"))
    text: str = ""                               # the fact as prose, with the date inside it
    last_confirmed: str = ""                     # YYYY-MM-DD, stamped by the engine; the eviction key
    sources: list[str] = Field(default_factory=list)   # real cell_ids, which consolidate has already mapped back from short labels


class UserProfile(BaseModel):
    """The complete profile: traits maps each PMO-16 dimension to a portrait or
    None, and facts maps each of the four bands to a list of facts.
    """
    traits: dict[str, Optional[ProfileTrait]] = Field(default_factory=dict)
    facts: dict[str, list[ProfileFact]] = Field(
        default_factory=lambda: {b: [] for b in BANDS}
    )

    @classmethod
    def empty(cls) -> "UserProfile":
        return cls(traits={}, facts={b: [] for b in BANDS})


@dataclass
class ProfileVersion:
    """One archived profile version, with its metadata."""
    version: int
    profile: UserProfile
    up_to_cell_id: str
    created_at: datetime


class ProfileStore:
    """Access to profile_versions: whole versions are archived, and the current one
    is the latest. An instance is bound to one user.
    """

    def __init__(self, db: Database, user_id: str = ""):
        self.db = db
        self.user_id = user_id

    def current(self) -> ProfileVersion | None:
        """The current profile, which is this user's latest version.

        Returns None if consolidation has never run, and the consumer then takes
        its no-profile path.
        """
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
        """How many versions have been consolidated, which is the profile's maturity
        and the anchor for annealing progress.
        """
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM profile_versions WHERE user_id=%s", (self.user_id,)
        )
        return int(row["n"]) if row else 0

    def get_version(self, version: int) -> ProfileVersion | None:
        """Fetch a specific version, for auditing or rollback."""
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
        """Archive a whole version and return its new version number.

        The version is MAX+1 computed inside the transaction, with
        uq_user_version as the backstop against a concurrent duplicate.

        Consolidation for one user is already serialized by a per-user
        single-flight lock, so this transaction is only the second line of defence.
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
        logger.info(f"profile version published user={self.user_id or '(default)'} version={nv} "
                    f"up_to_cell={up_to_cell_id or '(all)'} "
                    f"traits={sum(1 for v in profile.traits.values() if v)} "
                    f"facts={sum(len(v) for v in profile.facts.values())}")
        return nv
