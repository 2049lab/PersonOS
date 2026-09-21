"""The multimodal-LLM identity arbitration protocol (one batched call, one verdict)
plus persisting evidence and learning it into the cloud (enroll).

A single call judges every unbound cast in this clip:

- the query side: per cast, the sharpest face, the body shot, the appearance
  description and a couple of quoted lines, plus any mid-run evaluation note;
- the candidate side: one card per enrolled character — name, appearance, clearest
  face, body shot, and a voice sample;
- the output space: BIND|<cast>|<character_id> or BIND|<cast>|NEW;
- the verdict is final. There is no round-robin voting and no vector veto. Illegal
  output is treated as NEW and recorded as an issue, so a failed arbitration
  enrolls a temporary new profile: it never blocks, and it can be merged offline.

A couple of deliberate choices: a candidate card carries a single free-text desc
field rather than a structured profile, and there is no documentary profile, since
the scenes here are first-person from glasses or a robot. Asset b64 is fetched
from OSS upstream by ChainBook.query_card and registry.candidate_card and passed
in, so this module never touches OSS itself.

enroll_evidence lands one cast's evidence onto a character: crops go to OSS,
assets go into the database, and vectors are learned into the probability cloud.
It runs after a NEW verdict creates a profile and after a match verdict picks an
existing one, and final adjudication in commit.py reuses it.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from loguru import logger

from personos.identity.cloud import CloudEngine
from personos.identity.store import CharacterStore
from personos.identity.types import CandidateCard, CastEvidence

NEW = "NEW"

_BIND_RE = re.compile(r"^BIND\|", re.IGNORECASE)

PROMPT_HEADER = """You are identifying people across separate recordings.
For each QUERY person (seen in the current recording) decide whether they are one of the
REGISTERED characters below, or a new person never registered before.

EVIDENCE — judge like a human, strongest signal first:
1. Face, build, body shape, hairstyle, age impression.
2. Voice samples (audio #N) when attached: a similar voice supports a match but never
   decides alone; listen especially when a person has no face photo.
3. A heard name matching a registered character supports a match but never decides alone
   (different people can share a name).
4. Clothing is the weakest signal — it may have changed between recordings; never rely
   on clothing alone.

DECISION POLICY:
- If you are not reasonably sure a query matches a registered character, answer NEW.
  An unnecessary NEW is recoverable later; merging two different people is not.
- People appearing together at the same time are physically distinct people — they can
  never be the same character.
- Candidate ids starting with "chain:" are OTHER query people from this same recording.
  Bind a query to a chain id ONLY if the two are the same person seen in separate,
  non-overlapping segments. Never bind a query to its own chain id.

Answer with protocol lines ONLY, one per query, then END.

FORMAT EXAMPLE (format only; use the actual QUERY ids and ALLOWED TARGETS you are given):
BIND|<query_id>|<target id copied verbatim from ALLOWED TARGETS>
BIND|<query_id>|NEW
END
"""


def build_arbitration_prompt(queries: list[dict[str, Any]], candidates: list[CandidateCard],
                             ) -> tuple[str, list[str], list[str]]:
    """queries is [{cast_id, desc, name, key_lines, note?, face_b64, body_b64, voice_b64?}].

    Returns (prompt, images, audios); image #N and audio #N are numbered
    independently of each other.
    """
    images: list[str] = []
    audios: list[str] = []
    blocks: list[str] = [PROMPT_HEADER, "REGISTERED CHARACTERS:"]
    for card in candidates:
        rows = [f"CHARACTER {card.character_id}:"]
        if card.name:
            rows.append(f"    known name: {card.name}")
        if card.desc:
            rows.append(f"    appearance: {card.desc}")
        if card.last_seen_session:
            rows.append(f"    last seen: recording {card.last_seen_session}"
                        " (clothing may differ now)")
        attached = []
        if card.face_b64:
            images.append(card.face_b64); attached.append(f"image #{len(images)} = face photo")
        if card.body_b64:
            images.append(card.body_b64); attached.append(f"image #{len(images)} = full-body photo")
        if card.voice_b64:
            audios.append(card.voice_b64); attached.append(f"audio #{len(audios)} = voice sample")
        if attached:
            rows.append(f"    attached: {', '.join(attached)}")
        blocks.append("\n".join(rows))
    blocks.append("QUERIES (people in the current recording):")
    for query in queries:
        rows = [f"QUERY {query['cast_id']}:"]
        if query.get("name"):
            rows.append(f"    heard name: {query['name']}")
        if query.get("desc"):
            rows.append(f"    appearance: {query['desc']}")
        for line in (query.get("key_lines") or [])[:2]:
            rows.append(f"    said: {line}")
        if query.get("note"):
            rows.append(f"    note: {query['note']}")
        attached = []
        if query.get("face_b64"):
            images.append(query["face_b64"]); attached.append(f"image #{len(images)} = face photo")
        if query.get("body_b64"):
            images.append(query["body_b64"]); attached.append(f"image #{len(images)} = full-body photo")
        if query.get("voice_b64"):
            audios.append(query["voice_b64"]); attached.append(f"audio #{len(audios)} = voice sample")
        if attached:
            rows.append(f"    attached: {', '.join(attached)}")
        blocks.append("\n".join(rows))
    # Observed failure modes: the model copies the format example literally, or
    # invents an id that is not in the candidate set, which voids the verdict. An
    # explicit whitelist — plus stating that NEW is the only answer when there are
    # no candidates — pins the output space down.
    if candidates:
        ids = ", ".join(card.character_id for card in candidates)
        blocks.append(
            "ALLOWED TARGETS (the target id in each BIND line must be copied"
            f" verbatim from this list, or be NEW): {ids}")
    else:
        blocks.append(
            "ALLOWED TARGETS: none registered — NEW is the only valid answer;"
            " output BIND|<query_id>|NEW for every query.")
    blocks.append("Output BIND lines now.")
    return "\n\n".join(blocks), images, audios


def parse_verdicts(raw: str, *, cast_ids: list[str], candidate_ids: list[str],
                   ) -> tuple[dict[str, str], list[str]]:
    """Returns ({cast_id: character_id | "NEW"}, issues).

    A cast that is missing or out of bounds always falls back to NEW. With a
    single-shot protocol that is the only safe fallback: a spurious new profile
    can be undone, a misidentification cannot.
    """
    verdicts, issues, _defaulted = parse_verdicts_detailed(
        raw, cast_ids=cast_ids, candidate_ids=candidate_ids)
    return verdicts, issues


def parse_verdicts_detailed(raw: str, *, cast_ids: list[str], candidate_ids: list[str],
                            ) -> tuple[dict[str, str], list[str], set[str]]:
    """The detailed form of parse_verdicts, which also returns the set of casts that
    were defaulted to NEW.

    A cast is "defaulted" when the verdict did not come from the model explicitly —
    a missing BIND line, or a target outside the whitelist that fell back to NEW.
    Final adjudication uses this to fall back to the chain hypothesis instead of
    enrolling a duplicate new profile. An explicit BIND|x|NEW is a real verdict and
    is not counted as defaulted.
    """
    verdicts: dict[str, str] = {}
    issues: list[str] = []
    defaulted: set[str] = set()
    valid_candidates = set(candidate_ids)
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or not _BIND_RE.match(line):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            issues.append(f"bad BIND line: {line!r}")
            continue
        cast_id, target = parts[1], parts[2]
        if cast_id not in cast_ids:
            issues.append(f"BIND for unknown query {cast_id!r}")
            continue
        if target.upper() == "NEW":
            verdicts[cast_id] = NEW
            defaulted.discard(cast_id)
        elif target in valid_candidates:
            verdicts[cast_id] = target
            defaulted.discard(cast_id)
        else:
            issues.append(f"BIND to unknown character {target!r} → NEW")
            verdicts[cast_id] = NEW
            defaulted.add(cast_id)
    for cast_id in cast_ids:
        if cast_id not in verdicts:
            issues.append(f"no verdict for {cast_id!r} → NEW")
            verdicts[cast_id] = NEW
            defaulted.add(cast_id)
    return verdicts, issues, defaulted


# ── enroll: persist the evidence and learn it into the cloud. Used both when
#    creating a NEW profile and when appending to an existing match, and reused
#    by final adjudication in commit.py ────────────────────────────────────
def enroll_evidence(store: CharacterStore, cloud: CloudEngine, media_store: Any,
                    character_id: str, evidence: CastEvidence,
                    *, session_id: str = "", clip_index: int = 0) -> None:
    """Land one cast's evidence onto a character: upload the crops to OSS, insert
    the assets, and learn the vectors into the probability cloud.
    """
    def _save_img(b64: str, ct: str) -> str:
        if media_store is None or not b64:
            return ""
        try:
            return media_store.save_image(base64.b64decode(b64), owner=store.user_id,
                                          content_type=ct).key
        except Exception as e:  # noqa: BLE001  a failed OSS upload must not block: the vector is still learned
            logger.warning(f"failed to store asset in the object store: {e}")
            return ""

    for f in evidence.faces:
        if f.embedding is None:
            continue
        store.add_asset(character_id, "face", quality=f.q, embedding=f.embedding,
                        payload={"oss_key": _save_img(f.crop_b64, "image/png"), "t": f.t,
                                 "session": session_id, "clip": clip_index,
                                 "descriptor": f.descriptor})
        cloud.learn(character_id, "face", f.embedding, f.q,
                    payload={"session": session_id, "clip": clip_index})
        body_key = _save_img(f.body_crop_b64, "image/jpeg")
        if body_key:
            store.add_asset(character_id, "body", quality=f.q, embedding=None,
                            payload={"oss_key": body_key, "t": f.t, "session": session_id,
                                     "clip": clip_index})
    for v in evidence.voices:
        if v.embedding is None:
            continue
        voice_key = ""
        if media_store is not None and v.wav_bytes:
            try:
                voice_key = media_store.save_audio(v.wav_bytes, owner=store.user_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"failed to store voiceprint wav in the object store: {e}")
        store.add_asset(character_id, "voice", quality=v.q, embedding=v.embedding,
                        payload={"oss_key": voice_key, "t0": v.t0, "t1": v.t1,
                                 "session": session_id, "clip": clip_index})
        cloud.learn(character_id, "voice", v.embedding, v.q,
                    payload={"session": session_id, "clip": clip_index})
