"""Video in, questions in words out — including questions about *who*.

    pip install 'personos[identity]'          # ~2 GB of model dependencies
    export PERSONOS_LLM_API_KEY=sk-...
    export PERSONOS_MLLM_API_KEY=sk-...       # a model that accepts video
    export PERSONOS_MLLM_MODEL=...
    export PERSONOS_VIDEO_BACKEND=real
    export PERSONOS_MEDIA_BASE_URL=https://...   # see "A note on hosting" below

    python examples/prepare_video_sample.py   # fetches a sample, cuts 3 clips
    python examples/video.py

This is the part no other open memory framework does. Text and images are
described and indexed; a person is *resolved*. Faces, body shots and voice
samples from each clip accumulate into a character entity, so the same person
appearing in clip 3 and again in clip 1 of a later session is one person, with
one history — without anyone enrolling them beforehand.

What to watch:

- **Clips are consecutive and the session is closed once.** Identity evidence
  accumulates as a draft across clips and is committed when the session ends.
  A person seen briefly in one clip and clearly in another ends up as a single
  character, which is why the order and the closing matter.
- **Names are earned, not assumed.** Someone only gets a name when the dialogue
  supplies one. Until then they are a stable anonymous handle, and questions
  about them still work.
- **Answers cite clips.** Every claim traces back to the clip and timestamp it
  came from.

A note on hosting: the model service fetches each clip by URL, server-side, so
with local media storage it needs a publicly reachable prefix
(PERSONOS_MEDIA_BASE_URL). Without one the library says so rather than failing
somewhere inside the model call. Object storage (PERSONOS_MEDIA_BACKEND=oss)
avoids the question.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from personos import Memory, MissingCapability

USER, SESSION = "robot", "living-room"

QUESTIONS = [
    "Who appears in this recording, and what did each of them do?",
    "Did anyone bring something into the room?",
    "What did people talk about?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=str(Path.home() / ".personos" / "samples" / "clips"))
    args = ap.parse_args()

    clips = sorted(Path(args.clips_dir).glob("*.mp4"))
    if not clips:
        raise SystemExit(
            f"no clips in {args.clips_dir}\n"
            f"run: python examples/prepare_video_sample.py")

    with Memory() as m:
        for cap in m.capabilities():
            if cap.required and not cap.available:
                raise SystemExit(f"{cap.name}: {cap.remedy}")
        video = next(c for c in m.capabilities() if "video" in c.name)
        if not video.available:
            raise SystemExit(f"video identity is off — {video.remedy}")

        print(f"Ingesting {len(clips)} consecutive clips...\n")
        for i, clip in enumerate(clips):
            started = time.monotonic()
            try:
                m.add([{"role": "user", "content": "", "video": str(clip)}],
                      user_id=USER, session_id=SESSION)
            except MissingCapability as e:
                # The most common one here is local media with no public URL.
                raise SystemExit(str(e)) from e
            print(f"  clip {i}: {clip.name}  {time.monotonic() - started:.0f}s")

        # Identity is committed here: candidate people seen across clips are
        # adjudicated into characters, and the dialogue is written into memory
        # with each line attributed to whoever said it.
        print("\nClosing the session (identity is resolved here)...")
        started = time.monotonic()
        m.end_session(user_id=USER, session_id=SESSION)
        print(f"  {time.monotonic() - started:.0f}s")

        ctx = m.for_user(USER)
        print("\nPeople this recording produced:")
        from personos.identity.store import CharacterStore

        store = CharacterStore(m.db, USER)
        for ch in store.list_active_characters(include_wearer=True):
            label = ch.get("primary_name") or "(name not yet known)"
            kinds = [k for k in ("face", "body", "voice") if store.active_assets(ch["id"], k)]
            print(f"  {label:<24} id={ch['id'][:20]}  evidence={kinds or ['none']}"
                  f"{'   [the wearer — the camera itself]' if ch.get('is_wearer') else ''}")

        print("\nWhat was written into memory:")
        for c in ctx.cells.iter_all():
            print(f"  episode: {c.episode[:240]}...")
        for a in ctx.atoms.list(limit=25):
            print(f"  [{a.holder}] {a.text}")

        print("\n" + "─" * 70)
        for question in QUESTIONS:
            out = m.search(question, user_id=USER, session_id=SESSION)
            print(f"\nQ: {question}")
            print(f"A: {out.ans.answer if out.ans else '(no answer)'}")
            print(f"   cited clips/episodes: {out.ans.cited_cells if out.ans else []}")

        print("\n(keeping the data — re-run the questions without re-ingesting, "
              f"or call Memory().reset(user_id={USER!r}) to clear it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
