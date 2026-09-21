"""A **visual-understanding query rewrite** that runs before R0: when the user asks a question with an
image, recognize the people in it and write them into the query.

Why it is needed: the user sends a photo and asks "what did he and I do last week?" — the text-only
path can neither see the image nor know who "he" is; `rewrite_query` can only resolve references from
the dialogue history, and no amount of history resolves a face in a photo. So visual information is
turned into text first:

    "what did he and I do last week?"  --[image + character roster + history]-->  "what did Li Si and I do last week?"

The rewritten query then goes into `rewrite_query` as usual, and not a single line of the retrieval
path that follows has to change.

The flow (sharing the same identity assets as recognition on the ingest side, rather than standing up
a second set):
  image bytes -> RGB frame -> face_detector detection -> per-face embedding / quality / crop
  -> character-roster recall (take everyone for a small store, coarse recall via the probability
  cloud for a large one) -> candidate cards carrying face / full-body / voice / name / description
  -> one MLLM call: look at the image + the candidate material, read the history and the original
  query -> the rewritten query + who was recognized

**Degradation along the whole path**: decode failure, no face, no candidates, the MLLM failing, bad
JSON — all of them return the query unchanged and only log. Recall is a read path: losing one layer
of understanding is fine, but failing to look at an image must never cost us the answer.

The query image is **not stored in OSS**: it is query input, not memory content (unlike an ingest
image, which is stored so the deep track can look at it again).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

from personos.identity.harvest import _body_crop_b64, _face_q
from personos.identity.types import CandidateCard, CastEvidence, FacePick
from personos.online.llm import strip_fences

# Number of candidates coarsely recalled per face for a large store; for a small one
# (<= small_library_max) everyone is taken directly
COARSE_TOP_K = 5
# How many faces one rewrite looks at at most: with many people in the image, only the highest-quality
# few are taken, to keep the prompt from exploding
MAX_FACES = 4

_SYSTEM = """# Role
You rewrite a user's question so that later text-only retrieval can work, using an image the user just sent.

# Input
- The image the user sent (image #1).
- REGISTERED PEOPLE: people already known from this user's memory, each with name/appearance and photos.
- Recent dialogue history (may include `video:` lines — those are narrative summaries of recordings the user watched earlier).
- The user's current question.

# Task
1. Decide, for each detected face in the image, whether it is one of the REGISTERED PEOPLE.
   - Match on facial features first; clothing and body shape are weaker evidence (people change clothes).
   - **It is correct and expected to answer "none of them".** Do NOT force a match. A wrong name is far
     worse than no name: it sends retrieval to the wrong person entirely.
2. Rewrite the question by replacing visual references ("he", "this person", "this place", "that thing")
   with what they actually are, so the question stands alone without the image.
   - A person you matched → use their known name.
   - A person you could NOT match → describe them briefly ("the man in the blue jacket"), do not invent a name.
   - No people in the image → still use what you see (place, object, text, scene) to make the question concrete.
3. Keep the question's original intent and time expressions untouched. Only resolve what the image resolves.
   If the image adds nothing, return the question unchanged.

# Output
Return a single JSON object, nothing else:
{"resolved": "<rewritten question>", "matched": [{"face_index": 0, "person": "<label>", "name": "<name>"}]}
- `matched` lists only faces you are confident about; leave it `[]` when unsure.
- `person` MUST be one of the short labels (p1, p2, …) from ALLOWED LABELS. Never invent one.

# Examples

Question: "他和我上周干嘛去了?"  → FACE 0 matches p2 (known name 李四)
{"resolved": "李四和我上周干嘛去了?", "matched": [{"face_index": 0, "person": "p2", "name": "李四"}]}

Question: "他和我上周干嘛去了?"  → FACE 0 matches nobody in the list
{"resolved": "照片里那个穿蓝色夹克的男生和我上周干嘛去了?", "matched": []}

Question: "这家店我上周去过吗?"  → no face; the image shows a restaurant sign reading 蜀香源
{"resolved": "蜀香源这家川菜馆我上周去过吗?", "matched": []}

Question: "我上周三下午在干嘛?"  → the image is an unrelated screenshot; it resolves nothing
{"resolved": "我上周三下午在干嘛?", "matched": []}"""


@dataclass
class VisualRewrite:
    """The result of a visual rewrite. Every failure path sets query to the original question, so the
    caller never has to check for errors."""

    query: str
    faces: int = 0                              # how many faces were detected in the image
    matched: list[dict] = field(default_factory=list)   # [{face_index, character_id, name}]
    system: str = ""
    user: str = ""
    raw: str = ""
    skipped: str = ""                           # non-empty = why no rewrite happened (for logging and observability)


@dataclass
class VisualDeps:
    """Dependencies of the visual rewrite (assembled by runtime.visual_deps). `backends` is a process
    singleton (heavy models)."""

    store: Any            # CharacterStore (this user's people store)
    cloud: Any            # CloudEngine (the probabilistic identity cloud, used for coarse recall on a large store)
    backends: dict        # {face_detector, mm_runner, ...} -- the same singletons ingest uses
    registry: Any         # AnchorRegistry: only its candidate_card is used (cards built exactly as on the ingest side)
    small_library_max: int = 20    # same convention as AnchorRegistry: with at most this many people in the store, all of them become candidates


def _to_frame(image: bytes) -> Optional[np.ndarray]:
    """Image bytes -> an RGB numpy frame (which is what face_detector consumes). A bad image returns
    None rather than raising."""
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"query image decode failed, skipping the visual rewrite: {e}")
        return None


def _detect(deps: VisualDeps, frame: np.ndarray) -> list[FacePick]:
    """Detect faces -> FacePick (carrying embedding, quality, face crop and body crop), taking the
    top MAX_FACES by descending quality.

    Reuses harvest's _face_q / _body_crop_b64: the same convention as the ingest side, so the
    definition of the quality score cannot drift between the two places.
    """
    det = deps.backends.get("face_detector")
    if det is None:
        return []
    try:
        dets = det.detect(frame)
    except Exception as e:  # noqa: BLE001  a local inference failure must not block recall
        logger.warning(f"query image face detection failed, skipping the visual rewrite: {e}")
        return []
    picks = [FacePick(t=0.0, embedding=np.asarray(d.embedding, dtype=np.float32),
                      q=_face_q(d), crop_b64=getattr(d, "crop_b64", ""),
                      body_crop_b64=_body_crop_b64(frame, tuple(d.bbox)))
             for d in dets]
    picks.sort(key=lambda p: p.q, reverse=True)
    return picks[:MAX_FACES]


def _candidates(deps: VisualDeps, picks: list[FacePick]) -> list[CandidateCard]:
    """Character-roster recall: take everyone for a small store; for a large one, run coarse recall
    through the probability cloud per face and dedup by character_id."""
    chars = deps.store.list_active_characters(include_wearer=True)
    if not chars:
        return []
    by_id = {c["id"]: c for c in chars}
    if len(chars) <= deps.small_library_max or not picks:
        chosen = list(by_id)
    else:
        chosen, seen = [], set()
        for i, p in enumerate(picks):
            ev = CastEvidence(cast_id=f"F{i}", faces=[p])
            for cid, _score in deps.cloud.coarse_recall(ev, list(by_id), k=COARSE_TOP_K):
                if cid not in seen:
                    seen.add(cid); chosen.append(cid)
    return [deps.registry.candidate_card(by_id[cid]) for cid in chosen]


def build_prompt(query: str, picks: list[FacePick], cands: list[CandidateCard],
                 history: list[tuple[str, str]] | None, now_dt: Any = None,
                 ) -> tuple[str, list[str], dict[str, str]]:
    """Assemble the prompt. Returns (user_prompt, images, {short handle: character_id}).

    - images[0] is always the original image the user sent; the image #N numbering follows the same
      style as build_arbitration_prompt, so the model can line its textual references up with
      specific images.
    - People are referred to by **short handles p1/p2**, never by the real character_id (a 26-char
      ULID) — an existing project convention: whenever an LLM has to reference content by repeating
      an id, it goes through a short <-> long mapping. Long ids are easy to mistype or hallucinate,
      while short handles are easy to copy and can be validated mechanically, with the real ids
      filled back in after parsing.
    """
    images: list[str] = []
    blocks: list[str] = []
    labels = {f"p{i + 1}": c.character_id for i, c in enumerate(cands)}

    blocks.append("THE USER'S IMAGE: image #1"
                  + (f" ({len(picks)} face(s) detected, cropped as the images that follow)"
                     if picks else " (no face detected in it)"))
    for i, p in enumerate(picks):
        attached = []
        if p.crop_b64:
            images.append(p.crop_b64); attached.append(f"image #{len(images) + 1} = face crop")
        if p.body_crop_b64:
            images.append(p.body_crop_b64); attached.append(f"image #{len(images) + 1} = body crop")
        blocks.append(f"  FACE {i}: " + (", ".join(attached) or "(no crop available)"))

    if cands:
        blocks.append("\nREGISTERED PEOPLE (from this user's memory):")
        for label, c in zip(labels, cands):
            rows = [f"  PERSON {label}:"]
            if c.name:
                rows.append(f"    known name: {c.name}")
            if c.desc:
                rows.append(f"    appearance: {c.desc}")
            attached = []
            if c.face_b64:
                images.append(c.face_b64); attached.append(f"image #{len(images) + 1} = face photo")
            if c.body_b64:
                images.append(c.body_b64)
                attached.append(f"image #{len(images) + 1} = full-body photo")
            if attached:
                rows.append(f"    attached: {', '.join(attached)}")
            blocks.append("\n".join(rows))
        blocks.append("\nALLOWED LABELS (use exactly one of these in `person`; never invent): "
                      + ", ".join(labels))
    else:
        blocks.append("\nREGISTERED PEOPLE: (none — this user's memory has no people yet;"
                      " `matched` must be [])")

    # The current-time anchor, following the same convention as rewrite_query. Without it, a season,
    # a holiday or a shop sign visible in the photo can make the model anchor "last week" or "last
    # year" to the wrong year — and the product of this rewrite becomes R0's input directly.
    if now_dt is not None:
        blocks.append(f"\nCURRENT TIME: {now_dt.isoformat()}"
                      " (the anchor for any relative time in the question or the image)")
    hist = "\n".join(f"{h}: {t}" for h, t in (history or [])) or "(no history)"
    blocks.append(f"\nRECENT DIALOGUE HISTORY:\n{hist}")
    blocks.append(f"\nTHE USER'S QUESTION:\n{query}")
    return "\n".join(blocks), images, labels


def enrich_query_with_image(
    deps: VisualDeps, *, query: str, image: bytes, content_type: str = "image/jpeg",
    history: list[tuple[str, str]] | None = None, scenario: str = "", now_dt: Any = None,
) -> VisualRewrite:
    """Rewrite the query by looking at the image. **Any failure returns the original query** (with
    the reason recorded in `skipped`); it never raises and never blocks recall."""
    import base64

    frame = _to_frame(image)
    if frame is None:
        return VisualRewrite(query=query, skipped="image decode failed")

    picks = _detect(deps, frame)
    cands = _candidates(deps, picks) if picks else []
    # Carry on even with no face: a place, an object or text in the image can just as well turn "that
    # shop" or "this thing" into something concrete (agreed with the user)
    omni = deps.backends.get("mm_runner")
    if omni is None:
        return VisualRewrite(query=query, faces=len(picks), skipped="no mm_runner configured")

    user, extra, labels = build_prompt(query, picks, cands, history, now_dt)
    sys = _SYSTEM + (f"\n\n# Caller scenario\n{scenario}" if scenario else "")
    images = [base64.b64encode(image).decode()] + extra
    try:
        raw = omni.chat(f"{sys}\n\n{user}", images_b64=images, max_tokens=1000, temperature=0.0)
        obj = json.loads(strip_fences(raw))
        resolved = str(obj.get("resolved") or "").strip() or query
        # Validate the short handles and fill the real ids back in: any label outside the allow-list
        # is dropped (which stops the model inventing people), while the public field stays character_id
        matched = [{**m, "character_id": labels[m["person"]]}
                   for m in (obj.get("matched") or [])
                   if isinstance(m, dict) and m.get("person") in labels]
        logger.info(f"visual rewrite faces={len(picks)} candidates={len(cands)} "
                    f"recognized={[m.get('name') for m in matched]}\n"
                    f"  original_question={query!r}\n  rewritten={resolved!r}")
        return VisualRewrite(query=resolved, faces=len(picks), matched=matched,
                             system=sys, user=user, raw=raw)
    except Exception as e:  # noqa: BLE001  a failed image read falls back to the original query: one layer of understanding less, but still an answer
        logger.warning(f"visual rewrite failed, falling back to the original query: {e}")
        return VisualRewrite(query=query, faces=len(picks), system=sys, user=user,
                             raw=str(e), skipped=f"{type(e).__name__}: {e}")
