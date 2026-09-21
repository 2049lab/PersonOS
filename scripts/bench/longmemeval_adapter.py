"""LongMemEval -> personos data adapter (pure parsing; it makes no LLM or
storage calls).

The key differences from the LoCoMo adapter:
- LongMemEval is a user-assistant dialogue rather than two real people: user
  turns get holder="user" (the owner of the memory) and assistant turns get
  holder="assistant", passed straight through the dialogue channel so they are
  fully extractable and attributable. They deliberately avoid the down-weighted
  assistant_reply channel, since the single-session-assistant question type
  tests assistant-side content specifically, and it is treated on the same
  footing as LoCoMo's speaker_b;
- every question carries its own haystack: one instance is one persona history
  (roughly 48 sessions in chronological order), loaded into the store and then
  asked a single question as of question_date;
- timestamp format "2023/05/20 (Sat) 02:21" -> a timezone-aware datetime
  (anchored to the personos TZ);
- evidence is located at session level (answer_session_ids) and at turn level
  (the has_answer flag); this adapter keeps the turn_idx-level mapping so the
  gold evidence chain can be traced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from personos.models import TZ

_DT_RE = re.compile(r"(\d{4}/\d{2}/\d{2})(?: \([A-Za-z]{3}\))?(?: (\d{2}:\d{2}))?")


def parse_lme_dt(raw: str) -> datetime:
    """'2023/05/20 (Sat) 02:21' -> a timezone-aware datetime; a missing time
    defaults to 00:00."""
    m = _DT_RE.match((raw or "").strip())
    if not m:
        raise ValueError(f"cannot parse LongMemEval timestamp: {raw!r}")
    d = datetime.strptime(m.group(1), "%Y/%m/%d")
    if m.group(2):
        d = d.replace(hour=int(m.group(2)[:2]), minute=int(m.group(2)[3:5]))
    return d.replace(tzinfo=TZ)


@dataclass
class LmeTurn:
    """One turn in the haystack: role -> holder (user/assistant), with turn_idx
    used to line up the has_answer flag."""
    holder: str
    text: str
    turn_idx: int
    has_answer: bool = False


@dataclass
class LmeSession:
    idx: int                      # index within haystack_session_ids (used to locate evidence)
    sid: str                      # the original session_id
    dt: datetime
    turns: list[LmeTurn] = field(default_factory=list)


@dataclass
class LmeQuestion:
    """One benchmark instance: a haystack (its own history) plus a single question."""
    qid: str
    qtype: str                    # single-session-user / temporal-reasoning / ... (abstention is inferred from the id suffix)
    abstention: bool
    question: str
    answer: object                # str, or int for counting questions
    question_dt: datetime
    sessions: list[LmeSession] = field(default_factory=list)
    evidence_sidx: list[int] = field(default_factory=list)   # indices of the evidence sessions

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
