"""LoCoMo-10 → personos 数据适配器。

职责(纯解析,不做任何 LLM/存储调用):
- 解析 locomo10.json 的 10 段对话 → 会话/轮次/题目结构;
- 图片消息 caption 内联:img_url/blip_caption(数据集 BLIP 离线生成)拼进文本,
  "[shared an image: …]"——图片语义经文字通道进入证据/抽取/检索全链;
- 双说话者映射:speaker_a → user,speaker_b → assistant_reply(连续同人回合合并成一个
  exchange;session 以 speaker_b 开头的孤立回合降级为 user 消息——personos 的 reconcile
  本就不按主语过滤,两位说话者的事实都会被抽取);
- 时间解析:"1:56 pm on 8 May, 2023" → 带时区 datetime(锚 personos 的 TZ),作为该
  session 所有轮的 now_dt 注入基准;
- 题目的 evidence dia_id("D1:3" = session 1 第 3 轮)→ 所属 session 编号,供按已灌
  session 过滤可回答的题/反推出题时刻。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from personos.models import TZ

# LoCoMo 的 session 时间格式,如 "1:56 pm on 8 May, 2023" / "10:33 am on 9 April, 2023"
_DT_FORMATS = ["%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"]
_DIA_RE = re.compile(r"^D(\d+):(\d+)$")


@dataclass
class Exchange:
    """一条 utterance(连续同人回合合并):LoCoMo 双方都是人,各自作为独立 message 喂入。

    holder:speaker_a→"user"(第一主体/记忆主人);speaker_b→其真名(人类参与者,非助手)。
    不走 assistant_reply 通道——那条通道在框架里是认识论上的"助手上下文"(建议/猜测需
    用户认可才算事实),而 LoCoMo 的 speaker_b 是真人,其自述事实应全量可抽、可归属
    (reconcile 的"按说话人归属"规则)、原话应可进证据检索池。
    """
    holder: str
    text: str
    dia: str = ""


@dataclass
class LocomoSession:
    idx: int                       # 1-based session 编号(dia_id 的 D# 与此一致)
    dt: datetime                   # 会话发生时间(now_dt 注入基准)
    exchanges: list[Exchange] = field(default_factory=list)


@dataclass
class LocomoQA:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)   # ["D1:3", ...]
    category: int = 0                                    # 1-5;5=adversarial(社区惯例弃)

    def evidence_sessions(self) -> set[int]:
        out = set()
        for e in self.evidence:
            m = _DIA_RE.match(e.strip())
            if m:
                out.add(int(m.group(1)))
        return out


@dataclass
class LocomoConversation:
    sample_id: str
    speaker_a: str                 # 映射为 user
    speaker_b: str                 # 映射为 assistant
    sessions: list[LocomoSession] = field(default_factory=list)
    qa: list[LocomoQA] = field(default_factory=list)


def _parse_dt(raw: str) -> datetime:
    s = (raw or "").strip()
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    raise ValueError(f"无法解析 LoCoMo 会话时间: {raw!r}")


def _group_turns(turns: list[dict], speaker_a: str, speaker_b: str) -> list[Exchange]:
    """连续同人回合合并成一条 utterance;holder 直接用【说话人真名】(如 "Caroline"/"Melanie")。

    EvidenceRecord.holder 是自由字符串,真名直传让 reconcile 的对话行渲染成
    "Melanie: ..."(模型看得到归属);抽出的原子 holder 也落在真名上。
    speaker_a(第一视角)固定为 "user"——她/他是记忆的主人;speaker_b 是人类对话参与者,
    不走 assistant 通道(那条通道是认识论上的"助手上下文",会降权抽取)。
    """
    out: list[Exchange] = []
    for t in turns or []:
        spk, text, dia = t["speaker"], (t.get("text") or "").strip(), t.get("dia_id", "")
        # 图片消息:caption 内联进文本(评测口径与 EverMemOS 对标——图片理解在上游
        # BLIP 已完成,记忆层只消费文本)。归属由渲染层 holder 前缀提供,模板不带
        # 说话人名;单 turn 单行,不改变行数(run_locomo 按 split("\n") zip dia 对齐 trace)。
        if t.get("img_url"):
            text = (f"[shared an image: {t.get('blip_caption', 'an image')}] {text}").strip()
        holder = "user" if spk == speaker_a else spk
        if out and out[-1].holder == holder:
            last = out[-1]
            last.text = (last.text + "\n" + text).strip()
            last.dia = (last.dia + "," + dia).lstrip(",")
        else:
            out.append(Exchange(holder=holder, text=text, dia=dia))
    return out


def load_conversations(path: str | Path) -> list[LocomoConversation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[LocomoConversation] = []
    for c in raw:
        conv_d = c["conversation"]
        speaker_a, speaker_b = conv_d["speaker_a"], conv_d["speaker_b"]
        lc = LocomoConversation(sample_id=c["sample_id"], speaker_a=speaker_a, speaker_b=speaker_b)
        sids = sorted(
            (k for k in conv_d if re.fullmatch(r"session_\d+", k)),
            key=lambda s: int(s.split("_")[1]),
        )
        for s in sids:
            idx = int(s.split("_")[1])
            lc.sessions.append(LocomoSession(
                idx=idx,
                dt=_parse_dt(conv_d[f"{s}_date_time"]),
                exchanges=_group_turns(conv_d[s], speaker_a, speaker_b),
            ))
        for qa in c.get("qa", []):
            lc.qa.append(LocomoQA(
                question=qa.get("question", ""),
                answer=qa.get("answer", ""),
                evidence=[str(e) for e in (qa.get("evidence") or [])],
                category=int(qa.get("category", 0) or 0),
            ))
        out.append(lc)
    return out


def pick_answerable(lc: LocomoConversation, session_idxs: set[int], *, exclude_categories=(5,)) -> list[LocomoQA]:
    """挑出【全部 evidence 都落在已灌 session 内】且非弃用题型的题——保证问的都在库里。"""
    return [
        qa for qa in lc.qa
        if qa.category not in exclude_categories and qa.evidence and qa.evidence_sessions() <= session_idxs
    ]
