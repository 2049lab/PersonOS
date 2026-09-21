"""领域数据模型(对应 DESIGN §2,P0 文字阶段子集)。

只建模在线链路用得到的字段;多模态/离线专属字段留默认值占位,后续阶段填。
时间统一用带时区的 datetime,序列化为 ISO 字符串。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator
from ulid import ULID

# 用户常住新加坡,时间语境按 SGT(UTC+8,与上海/北京同步);记录一律带时区
TZ = ZoneInfo("Asia/Shanghai")


def now() -> datetime:
    return datetime.now(TZ)


def ensure_aware(dt) -> Optional[datetime]:
    """把 datetime / ISO 字符串规整成带时区的 datetime;naive 视为本地 TZ。

    统一时间的"时区身份",避免 naive 与 aware datetime 相比较时抛 TypeError
    (LLM 常给出无时区日期;update 的 patch 又可能塞进原始字符串)。
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


# —— 证据层(不可变真相源)——
class EvidenceRecord(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("ev"))
    modality: Literal["text", "image", "audio", "video", "mixed"] = "text"
    holder: str = "user"                       # user | assistant | third_party
    content_inline: Optional[str] = None       # 文字直接内联
    content_ref: Optional[str] = None          # ext: 媒体指针
    sha256: str = ""                           # 内容哈希,去重与完整性
    source: dict[str, Any] = Field(default_factory=dict)  # system/session_id/message_id/turn_index/span
    captured_at: datetime = Field(default_factory=now)    # 系统何时收到(recorded_at)
    sensitivity: Literal["public", "normal", "sensitive", "secret"] = "normal"
    local_only: bool = False
    notes: Optional[str] = None


class EvidenceRef(BaseModel):
    evidence_id: str
    span: Optional[str] = None


# K 轴:记忆类型受控词表(facet 导航)。code → "label: typical examples";code 稳定、说明可演进。
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

# D 轴:生活域受控词表(写入判域 + 检索判域共用,保证两端词面一致)。
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
    """词表条目 → 标签名(":"前的名称部分)。"""
    return desc.partition(":")[0].strip()


# code → 名称(供前端/工具输出渲染标签;词表演进只改 value,不动 code)
KIND_LABELS: dict[str, str] = {c: _vocab_label(d) for c, d in KindLiteral.items()}
DOMAIN_LABELS: dict[str, str] = {c: _vocab_label(d) for c, d in DOMAIN_VOCAB.items()}


def vocab_menu(vocab: dict[str, str], indent: str = "  ") -> str:
    """词表 → 提示词菜单(逐行:code=名称(典型示例))。所有 prompt 站点共用,保证词面一致。"""
    lines = [f"{c}={_vocab_label(d)}({d.partition(':')[2].strip()})" for c, d in vocab.items()]
    return ("\n" + indent).join(lines)


# —— 提示词共享规范块(单一来源;reconcile / deep_recall / light_dream 各环节引用,防止多处复述漂移)——

# 自限界命题写法:记忆原子 text 的统一规范(W2② 批量提取 / 深轨 remember 写回共用一份)。
# 双时间格式是硬要求:相对语义与绝对日期同时进文本,检索与作答都不再做日历算术。
# 语言跟随是硬要求:存储内容语言跟随源对话语言,kill 跨语检索断层。
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

# D/K 轴菜单(写入抽取与后续巩固环节的提示词共用)。
AXIS_MENU = (
    "- kind (K axis, exactly one Kxx code or null; never output the label):\n  "
    + vocab_menu(KindLiteral) + "\n"
    "- domains (D axis, one or more Dxx codes; never output the labels):\n  "
    + vocab_menu(DOMAIN_VOCAB)
)


def normalize_domains(value: Any) -> list[str]:
    """任意输入 → 合法 D 轴 code 列表(去重保序)。"""
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(x for x in value if isinstance(x, str) and x in DOMAIN_VOCAB))


def normalize_kind(value: Any) -> Optional[str]:
    """任意输入 → 合法 K 轴 code 或 None。"""
    return value if isinstance(value, str) and value in KindLiteral else None


# —— MemCell:一段话题对话的加工产物(组织单元;融合架构 §1)——
class MemCell(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("cell"))
    session_id: str = ""
    topic: str = ""                            # 一句话主题(段粒度检索面,带 embedding)
    episode: str = ""                          # 第三人称叙事(作答主料;仅可被深轨 remember 修订)
    domains: list[str] = Field(default_factory=list)   # D 轴 0-3 个(W2① 判)
    episode_type: str = "unknown"              # 调用方 task_type 词表分类(未传/选不中=unknown);按类型分页检索用
    t_start: Optional[datetime] = None         # 该段对话起止(边界检测给出)
    t_end: Optional[datetime] = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)  # 该段覆盖的 utterance(时序)

    @field_validator("domains", mode="before")
    @classmethod
    def _coerce_domains(cls, v):
        return normalize_domains(v)


# —— 原子层:指向 cell 的检索单元(融合架构 §1;旧打分/生命周期/supersede 字段已删)——
class MemoryAtom(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("atom"))
    memcell_id: str = ""                       # 归属 cell
    object_type: Literal["claim", "fact", "event"] = "fact"
    text: str = ""                             # 规范见 ATOM_TEXT_SPEC(双时间格式写进文本)
    holder: str = "user"                       # 谁说的/谁的属性
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)  # 出处原话

    # 链归属(docs/atom-chain-design.md):atoms 表链三列是唯一事实源,此处为展示副本——
    # ChainStore 双写维护;AtomStore 读取时以列覆盖,防过期 payload 误导。''=游离
    chain_id: str = ""
    prev_atom_id: str = ""                     # 链上前驱(链首='')
    next_atom_id: str = ""                     # 链上后继(链尾='')

    occurrence_time: Optional[datetime] = None  # 该事实对应的对话时间
    recorded_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    source: str = "w2"                      # 写入来源:w2 批量抽取 | deep 深轨写回(旧库缺省视为 w2)

    domains: list[str] = Field(default_factory=list)  # D 轴 0-3 个
    kind: Optional[str] = None                        # K 轴单选(K01..K14)

    @field_validator("domains", mode="before")
    @classmethod
    def _coerce_domains(cls, v):
        """D 轴兜底:只保留 D01..D16,去重保序;旧词面/LLM 自造域直接丢弃。"""
        return normalize_domains(v)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v):
        """K 轴兜底:LLM 或旧库给出词表外的值时归 None,保证加载不崩。"""
        return normalize_kind(v)


# —— atom 链:同「事情」原子事实的时间线(docs/atom-chain-design.md)——
# 派生视图、只分组不消解(D-C3):链内新旧陈述不选赢家,冲突消费仍在作答时。
# 链序 = 追加序 = 对话事实抽取的自然顺序(D-C7);双向链表挂在 atoms 三列上。
class ChainInfo(BaseModel):
    id: str = Field(default_factory=lambda: _ulid("chn"))
    user_id: str = ""
    title: str = ""                    # 建链时判链 LLM 生成的一行短语(如「Caroline 练瑜伽的地点」),建后不变
    origin_cell_id: str = ""           # 建链时首成员所在 cell(溯源)
    n_atoms: int = 0
    head_atom_id: str = ""             # 链首
    tail_atom_id: str = ""             # 链尾(新成员唯一追加点)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


# —— 时间锚点:把"记录/发生"的绝对时间显影到【喂给 embedding / LLM】的文本前缀 ——
# 存储原文不动(证据不可变、原子角度中立);只在检索用的 embed 文本与展示文本里加锚点,
# 让时间线全程可见(下游 LLM 能判新旧、把"上周"对齐到绝对日期)。
def _stamp(dt: Optional[datetime]) -> str:
    dt = ensure_aware(dt)
    if dt is None:
        return ""
    return "[" + dt.strftime("%Y-%m-%d") + "]"


def atom_anchor(a: "MemoryAtom") -> Optional[datetime]:
    """原子的时间锚:发生时刻优先,回退记录时刻(valid_from 已随打分字段删除)。"""
    return a.occurrence_time or a.recorded_at


def stamped_atom_text(a: "MemoryAtom") -> str:
    """原子的带锚 embed 文本(到天):`[YYYY-MM-DD] 命题`。text 存储不变,仅此处显影。"""
    p = _stamp(atom_anchor(a))
    return f"{p} {a.text}" if p else a.text

