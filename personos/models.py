"""Domain data models (matching DESIGN section 2, the P0 text-phase subset).

Only the fields the online path needs are modeled; multimodal / offline-only fields are
left as defaults for now and filled in later phases. All times are timezone-aware
datetimes, serialized as ISO strings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator
from ulid import ULID

# The time context is UTC+8 (the zone used by the deployment); every record is
# timezone-aware.
TZ = ZoneInfo("Asia/Shanghai")


def now() -> datetime:
    return datetime.now(TZ)


def ensure_aware(dt) -> Optional[datetime]:
    """Normalize a datetime / ISO string into a timezone-aware datetime; a naive value
    is treated as being in the local TZ.

    This gives every timestamp the same "timezone identity" and avoids the TypeError
    raised when comparing a naive datetime with an aware one (the LLM often returns
    dates without a timezone, and an update patch may push a raw string in).
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except (ValueError, TypeError):
            return None
    if not isinstance(dt, datetime):
        return None
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt


def _ulid(prefix: str) -> str:
    return f"{prefix}_{ULID()}"


# -- Evidence layer (the immutable source of truth) --
class EvidenceRecord(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("ev"))
    modality: Literal["text", "image", "audio", "video", "mixed"] = "text"
    holder: str = "user"                       # user | assistant | third_party
    content_inline: Optional[str] = None       # text stored inline
    content_ref: Optional[str] = None          # ext: pointer to the media
    sha256: str = ""                           # content hash, for dedup and integrity
    source: dict[str, Any] = Field(default_factory=dict)  # system/session_id/message_id/turn_index/span
    captured_at: datetime = Field(default_factory=now)    # when the system received it (recorded_at)
    sensitivity: Literal["public", "normal", "sensitive", "secret"] = "normal"
    local_only: bool = False
    notes: Optional[str] = None


class EvidenceRef(BaseModel):
    evidence_id: str
    span: Optional[str] = None


# K axis: the controlled vocabulary of memory types (facet navigation). Maps code ->
# "label: typical examples"; the codes are stable while the descriptions may evolve.
KindLiteral: dict[str, str] = {
    "K01": "fact/attribute: birth year, languages spoken, device model",
    "K02": "relationship/role: A is the user's partner, B is a mentor",
    "K03": "status/metric: current job title, current weight, current asset allocation",
    "K04": "event/experience: started a job on a date, a trip, a surgery",
    "K05": "pattern/habit/routine: stays up late, studies weekly, work-procrastination pattern",
    "K06": "preference/dislike: prefers conclusions first, dislikes vague advice",
    "K07": "value/principle/belief: values freedom, long-termism, family boundaries",
    "K08": "need/request: needs proactive reminders, needs a quiet environment",
    "K09": "constraint/boundary/rule/obligation: never touch secrets, no automating high-risk external actions, never leak sensitive data",
    "K10": "goal/desired outcome: wants a career transition, building a personal knowledge system",
    "K11": "explicit intent/commitment: has explicitly decided to handle a matter",
    "K12": "plan/project/task: health-data bridge, long-running material project, knowledge-graph upkeep",
    "K13": "ability/skill: can code, can do data analysis, speaks a language",
    "K14": "assessment/feedback: evaluation of a piece of advice, a tool, an experience",
}

# D axis: the controlled vocabulary of life domains (shared by domain classification at
# write time and at retrieval time, so both ends use identical wording).
DOMAIN_VOCAB: dict[str, str] = {
    "D01": "identity & life context: basic identity, life stage, places, roles",
    "D02": "values, beliefs & meaning: worldview, value ranking, definition of success, life philosophy",
    "D03": "personality, cognition & communication style: Big Five, decision style, risk appetite, communication style",
    "D04": "emotion, motivation & subjective well-being: mood, stress, motivation, self-efficacy, happiness",
    "D05": "body, health & functioning: illness, symptoms, sleep, exercise, nutrition, functional status",
    "D06": "intimate relationships, family & caregiving: partner, marriage, parents, children, caregiving duties",
    "D07": "friends, social network & belonging: friends, mentors, communities, support network, belonging",
    "D08": "learning, knowledge & skills: education, courses, skills, knowledge base, study plans",
    "D09": "work, career & business: profession, projects, company, entrepreneurship, reputation",
    "D10": "finance, assets & economic security: income, spending, budget, investments, insurance, taxes",
    "D11": "time, attention, energy & execution: schedule, priorities, focus, task system, execution rhythm",
    "D12": "home, possessions, commute & living environment: housing, belongings, vehicle, commute, workspace",
    "D13": "interests, creativity, entertainment & travel: hobbies, aesthetics, games, music, travel, creation",
    "D14": "community, citizenship, contribution & life impact: volunteering, community roles, public issues, life impact",
    "D15": "legal, administrative, safety & institutional affairs: contracts, visas, documents, compliance, authorization, personal safety",
    "D16": "digital life, devices, accounts & data: devices, software, accounts, AI agents, data permissions, cyber security",
}


def _vocab_label(desc: str) -> str:
    """Vocabulary entry -> label name (the part before the ":")."""
    return desc.partition(":")[0].strip()


# code -> name (so the frontend/tool output can render labels; when the vocabulary
# evolves only the value changes, never the code)
KIND_LABELS: dict[str, str] = {c: _vocab_label(d) for c, d in KindLiteral.items()}
DOMAIN_LABELS: dict[str, str] = {c: _vocab_label(d) for c, d in DOMAIN_VOCAB.items()}


def vocab_menu(vocab: dict[str, str], indent: str = "  ") -> str:
    """Vocabulary -> prompt menu, one line per entry: code=name(typical examples). Every
    prompt site shares this, so the wording stays identical everywhere."""
    lines = [f"{c}={_vocab_label(d)}({d.partition(':')[2].strip()})" for c, d in vocab.items()]
    return ("\n" + indent).join(lines)


# -- Shared prompt spec blocks (single source of truth; referenced by reconcile,
# deep_recall and light_dream so the same rules aren't restated in several places and
# allowed to drift apart) --

# How to write a self-bounded proposition: the one spec for memory atom text, shared by
# W2 (2) batch extraction and the deep track's remember write-back.
# The dual time format is a hard requirement: relative wording and the absolute date
# both go into the text, so neither retrieval nor answering has to do calendar
# arithmetic.
# Language following is a hard requirement: stored content follows the source dialogue's
# language, which kills the cross-language retrieval gap.
ATOM_TEXT_SPEC = """Write each `text` as ONE atomic fact — third person, self-contained subject-verb-object:
  · Smallest retrievable unit: one atom = one minimal fact that can be independently retrieved and independently
    holds true. Split a turn into its distinct INDEPENDENTLY-QUERYABLE facts ("works late every Tue & Thu" +
    "review falls in September" → two atoms; parallel attributes each asked about on their own — schedule vs
    venue vs coach vs price — are separate anchors). But do NOT over-split or log restatements: qualifiers
    bounding ONE occurrence (the when/where of that one event) stay inside it; the SAME fact rephrased or
    reacted to across lines is ONE atom, not many; co-attributes never queried on their own stay fused.
  · MECE within the batch: atoms from the same segment never duplicate one another (no two phrasings of the
    same fact, no near-duplicates) and together cover every DISTINCT worth-remembering fact — not every line.
  · Pronouns → names: resolve "I/she/it" to the actual person ("I'm setting up with Rob" → "Caroline is
    setting up the exhibition with Rob").
  · Only settled propositions: if a reference cannot be resolved or subject-verb-object cannot be completed,
    skip it (the information survives in the episode) — never force an extraction.
  · Dual time format: relative wording AND absolute date go into the text together — "will hold the
    exhibition next month (2026-09)", "sprained the ankle last week (2026-08-18)".
  · Qualifiers stay inside the sentence: location/frequency/scope/expiry enter the text verbatim —
    "exhibition at Marina Bay Sands", "runs every Tue & Thu", "membership valid until 2026-07" — never drop
    a qualifier.
  · Proper nouns & numbers verbatim: brands/places/names/numbers copied exactly, never generalized
    ("3 pizzas" not "some pizzas", "hot yoga" not "exercise").
  · Language follows the source dialogue: write the atom text in the SAME language as the dialogue.
No dedup, no resolution: different cells each keep their own copy; whether an atom duplicates or conflicts
with existing memory is decided at answer time, never at write time.
Bad example: "likes americano" — no subject, no time.
Good example: "Caroline said her everyday favorite is americano (as of 2026-08)"."""

# D/K axis menus (shared by the prompts of write-time extraction and the later
# consolidation stages).
AXIS_MENU = (
    "- kind (K axis, exactly one Kxx code or null; never output the label):\n  "
    + vocab_menu(KindLiteral) + "\n"
    "- domains (D axis, one or more Dxx codes; never output the labels):\n  "
    + vocab_menu(DOMAIN_VOCAB)
)


def normalize_domains(value: Any) -> list[str]:
    """Any input -> a list of valid D axis codes (deduplicated, order preserved)."""
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(x for x in value if isinstance(x, str) and x in DOMAIN_VOCAB))


def normalize_kind(value: Any) -> Optional[str]:
    """Any input -> a valid K axis code, or None."""
    return value if isinstance(value, str) and value in KindLiteral else None


# -- MemCell: the processed output of one topical stretch of dialogue (the unit of
# organization; fused architecture section 1) --
class MemCell(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("cell"))
    session_id: str = ""
    topic: str = ""                            # One-sentence topic (the segment-level retrieval surface, embedded)
    episode: str = ""                          # Third-person narrative (the main material for answering; only the
                                               # deep track's remember may revise it)
    domains: list[str] = Field(default_factory=list)   # 0-3 D axis codes (decided in W2 (1))
    episode_type: str = "unknown"              # Classification against the caller's task_type vocabulary (not
                                               # provided / no match = unknown); used for paged retrieval by type
    t_start: Optional[datetime] = None         # Start/end of this stretch of dialogue (from boundary detection)
    t_end: Optional[datetime] = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)  # Utterances this segment covers, in order

    @field_validator("domains", mode="before")
    @classmethod
    def _coerce_domains(cls, v):
        return normalize_domains(v)


# -- Atom layer: the retrieval unit that points at a cell (fused architecture section 1;
# the old scoring / lifecycle / supersede fields have been removed) --
class MemoryAtom(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("atom"))
    memcell_id: str = ""                       # The cell this atom belongs to
    object_type: Literal["claim", "fact", "event"] = "fact"
    text: str = ""                             # See ATOM_TEXT_SPEC (the dual time format goes into the text)
    holder: str = "user"                       # Who said it / whose attribute it is
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)  # The original wording it came from

    # Chain membership: the three chain columns on the atoms
    # table are the single source of truth; these are display copies — ChainStore keeps
    # them in sync with a dual write, and AtomStore overrides them with the column values
    # on read so a stale payload can't mislead anyone. '' = not on a chain.
    chain_id: str = ""
    prev_atom_id: str = ""                     # Predecessor on the chain ('' at the head)
    next_atom_id: str = ""                     # Successor on the chain ('' at the tail)

    occurrence_time: Optional[datetime] = None  # The dialogue time this fact corresponds to
    recorded_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    source: str = "w2"                      # Where the write came from: w2 = batch extraction | deep = deep-track
                                            # write-back (rows from the old schema default to w2)

    domains: list[str] = Field(default_factory=list)  # 0-3 D axis codes
    kind: Optional[str] = None                        # A single K axis code (K01..K14)

    @field_validator("domains", mode="before")
    @classmethod
    def _coerce_domains(cls, v):
        """D axis guard: keep only D01..D16, deduplicated and in order; wording from an
        older vocabulary or a domain the LLM invented is dropped."""
        return normalize_domains(v)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v):
        """K axis guard: a value outside the vocabulary — from the LLM or from old rows —
        falls back to None, so loading never blows up."""
        return normalize_kind(v)


# -- atom chain: the timeline of atomic facts about the same "thing"
# Chain record models --
# A derived view that only groups and never resolves (D-C3): within a chain we pick no
# winner between the older and newer statement; conflicts are still consumed at answer
# time. Chain order = append order = the natural order in which facts were extracted
# from the dialogue (D-C7); the doubly linked list lives in the three atoms columns.
class ChainInfo(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("chn"))
    user_id: str = ""
    title: str = ""                    # A one-line phrase the chain-deciding LLM produced when the chain was
                                       # created (e.g. "where Caroline practices yoga"); never changes afterwards
    origin_cell_id: str = ""           # The cell the first member lived in when the chain was created (provenance)
    n_atoms: int = 0
    head_atom_id: str = ""             # Head of the chain
    tail_atom_id: str = ""             # Tail of the chain (the only place new members are appended)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


# -- Time anchor: surface the absolute "recorded/occurred" time as a prefix on the text
# that is fed to the embedding model / the LLM --
# The stored text itself is untouched (evidence is immutable and an atom stays neutral);
# the anchor is added only to the embed text used for retrieval and to the display text,
# which keeps the timeline visible throughout (so a downstream LLM can tell old from new
# and align "last week" to an absolute date).
def _stamp(dt: Optional[datetime]) -> str:
    dt = ensure_aware(dt)
    if dt is None:
        return ""
    return "[" + dt.strftime("%Y-%m-%d") + "]"


def atom_anchor(a: "MemoryAtom") -> Optional[datetime]:
    """An atom's time anchor: prefer the occurrence time, fall back to the recorded time
    (valid_from was removed along with the scoring fields)."""
    return a.occurrence_time or a.recorded_at


def stamped_atom_text(a: "MemoryAtom") -> str:
    """An atom's anchored embed text, at day granularity: `[YYYY-MM-DD] proposition`.
    The stored text is unchanged; the anchor only shows up here."""
    p = _stamp(atom_anchor(a))
    return f"{p} {a.text}" if p else a.text

