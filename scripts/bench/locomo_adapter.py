"""LoCoMo-10 -> personos data adapter.

Responsibilities (pure parsing; it makes no LLM or storage calls):
- parse the ten conversations in locomo10.json into a session / turn / question
  structure;
- inline image captions: img_url and blip_caption (generated offline by BLIP as
  part of the dataset) are folded into the text as "[shared an image: ...]", so
  that image semantics travel through the text channel into evidence,
  extraction and retrieval alike;
- map the two speakers: speaker_a -> user, speaker_b -> its real name
  (consecutive turns by the same speaker are merged into one exchange);
- parse timestamps: "1:56 pm on 8 May, 2023" -> a timezone-aware datetime
  (anchored to the personos TZ), used as the now_dt injected for every turn of
  that session;
- map a question's evidence dia_id ("D1:3" = session 1, turn 3) to its session
  number, so that answerable questions can be filtered by which sessions have
  been loaded and the question time can be derived.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from personos.models import TZ

# LoCoMo session timestamp formats, e.g. "1:56 pm on 8 May, 2023" / "10:33 am on 9 April, 2023"
_DT_FORMATS = ["%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"]
_DIA_RE = re.compile(r"^D(\d+):(\d+)$")


@dataclass
class Exchange:
    """One utterance (consecutive turns by the same speaker merged). Both LoCoMo
    participants are human, and each is fed in as an independent message.

    holder: speaker_a -> "user" (the first-person subject, the owner of the
    memory); speaker_b -> their real name (a human participant, not an
    assistant). Deliberately not routed through the assistant_reply channel:
    epistemically that channel is "assistant context" (a suggestion or guess
    only becomes fact once the user endorses it), whereas LoCoMo's speaker_b is
    a real person whose self-reported facts should be fully extractable,
    attributable (the "attribute by speaker" rule in reconcile), and whose
    verbatim words belong in the evidence retrieval pool.
    """
    holder: str
    text: str
    dia: str = ""


@dataclass
class LocomoSession:
    idx: int                       # 1-based session number (matches the D# in dia_id)
    dt: datetime                   # when the session happened (the now_dt baseline)
    exchanges: list[Exchange] = field(default_factory=list)


@dataclass
class LocomoQA:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)   # ["D1:3", ...]
    category: int = 0                                    # 1-5; 5 = adversarial (dropped by convention)

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
    speaker_a: str                 # mapped to user
    speaker_b: str                 # mapped to their real name
    sessions: list[LocomoSession] = field(default_factory=list)
    qa: list[LocomoQA] = field(default_factory=list)


def _parse_dt(raw: str) -> datetime:
    s = (raw or "").strip()
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    raise ValueError(f"cannot parse LoCoMo session timestamp: {raw!r}")


def _group_turns(turns: list[dict], speaker_a: str, speaker_b: str) -> list[Exchange]:
    """Merge consecutive turns by the same speaker into one utterance, using the
    speaker's **real name** as the holder (e.g. "Caroline" / "Melanie").

    EvidenceRecord.holder is a free-form string, so passing the real name
    straight through makes reconcile render the dialogue line as
    "Melanie: ..." (the model can see the attribution), and the extracted atoms
    carry the real name as their holder too.
    speaker_a, the first-person viewpoint, is always "user" - they own the
    memory. speaker_b is a human participant and does not go through the
    assistant channel, which is epistemically "assistant context" and would
    down-weight extraction.
    """
    out: list[Exchange] = []
    for t in turns or []:
        spk, text, dia = t["speaker"], (t.get("text") or "").strip(), t.get("dia_id", "")
        # Image messages: the caption is inlined into the text. This matches the
        # EverMemOS evaluation convention - image understanding already happened
        # upstream in BLIP, and the memory layer only ever consumes text.
        # Attribution comes from the holder prefix added by the rendering layer,
        # so the template carries no speaker name. One turn stays one line, which
        # keeps the line count unchanged (run_locomo zips split("\n") against dia
        # to align the trace).
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
    """Select questions whose evidence lies entirely within the loaded sessions
    and whose category is not excluded, so that everything asked is in the store."""
    return [
        qa for qa in lc.qa
        if qa.category not in exclude_categories and qa.evidence and qa.evidence_sessions() <= session_idxs
    ]
