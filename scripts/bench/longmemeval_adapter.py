"""LongMemEval → personos 数据适配器(纯解析,不做任何 LLM/存储调用)。

与 LoCoMo 适配器的关键差异:
- LongMemEval 是 user↔assistant 对话(非双真人):user 轮 holder="user"(记忆主人),
  assistant 轮 holder="assistant"(真名直传对话通道,全量可抽可归属——不走
  assistant_reply 降权通道;single-session-assistant 题专考助手侧内容,按 LoCoMo
  speaker_b 的同等口径处理);
- 每道题自带独立干草堆:一条实例 = 一段"人设历史"(48±个 session,按时间排序),
  灌库后在 question_date 时刻提问一次;
- 时间格式 "2023/05/20 (Sat) 02:21" → 带时区 datetime(锚 personos TZ);
- 证据定位是 session 级(answer_session_ids)+ 轮级(has_answer 轮标注),
  本适配器保留 turn_idx 级映射供金标证据链追踪。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from personos.models import TZ

_DT_RE = re.compile(r"(\d{4}/\d{2}/\d{2})(?: \([A-Za-z]{3}\))?(?: (\d{2}:\d{2}))?")


def parse_lme_dt(raw: str) -> datetime:
    """'2023/05/20 (Sat) 02:21' → 带 TZ 的 datetime;缺时间按 00:00。"""
    m = _DT_RE.match((raw or "").strip())
    if not m:
        raise ValueError(f"无法解析 LongMemEval 时间: {raw!r}")
    d = datetime.strptime(m.group(1), "%Y/%m/%d")
    if m.group(2):
        d = d.replace(hour=int(m.group(2)[:2]), minute=int(m.group(2)[3:5]))
    return d.replace(tzinfo=TZ)


@dataclass
class LmeTurn:
    """干草堆里的一轮:role→holder(user/assistant),turn_idx 供 has_answer 对齐。"""
    holder: str
    text: str
    turn_idx: int
    has_answer: bool = False


@dataclass
class LmeSession:
    idx: int                      # 在 haystack_session_ids 里的下标(证据定位用)
    sid: str                      # 原始 session_id
    dt: datetime
    turns: list[LmeTurn] = field(default_factory=list)


@dataclass
class LmeQuestion:
    """一条评测实例:干草堆(独立历史)+ 单题。"""
    qid: str
    qtype: str                    # single-session-user / temporal-reasoning / ... (abstention 并入类型后缀判断)
    abstention: bool
    question: str
    answer: object                # str 或 int(计数题)
    question_dt: datetime
    sessions: list[LmeSession] = field(default_factory=list)
    evidence_sidx: list[int] = field(default_factory=list)   # 证据 session 下标

    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


def load_questions(path: str | Path) -> list[LmeQuestion]:
    import json
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[LmeQuestion] = []
    for d in raw:
        sid_list = d["haystack_session_ids"]
        q = LmeQuestion(
            qid=d["question_id"], qtype=d["question_type"],
            abstention=d["question_id"].endswith("_abs"),
            question=d["question"], answer=d["answer"],
            question_dt=parse_lme_dt(d["question_date"]),
            evidence_sidx=[sid_list.index(a) for a in d.get("answer_session_ids", [])],
        )
        for i, (sid, dt_raw, sess) in enumerate(
                zip(sid_list, d["haystack_dates"], d["haystack_sessions"])):
            s = LmeSession(idx=i, sid=sid, dt=parse_lme_dt(dt_raw))
            for t_idx, t in enumerate(sess):
                s.turns.append(LmeTurn(
                    holder=t["role"], text=(t.get("content") or "").strip(),
                    turn_idx=t_idx, has_answer=bool(t.get("has_answer")),
                ))
            q.sessions.append(s)
        out.append(q)
    return out
