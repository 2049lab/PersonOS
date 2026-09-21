"""Images alongside text: write a photo into memory, then ask about it in words.

    export PERSONOS_LLM_API_KEY=sk-...
    export PERSONOS_MLLM_API_KEY=sk-...     # may be the same key
    export PERSONOS_MLLM_MODEL=gpt-4o
    python examples/images.py

An image travels through the same pipeline as text. It is described once at
write time, and from then on it is ordinary evidence: searchable by words,
citable in an answer, traceable back to the turn it arrived in.

Two things worth watching:

- **The original is kept.** The description is what makes the image findable,
  but the bytes are stored too, so a later question can go back to the source.
  With no object storage configured they land on the local filesystem.
- **Without a vision model nothing breaks.** The image is still stored and the
  text path is unaffected; the result simply carries a warning saying the image
  contributed nothing to retrieval. Run this without PERSONOS_MLLM_API_KEY to
  see that path — it is the difference between degrading and failing.
"""

from __future__ import annotations

import io
from pathlib import Path

from personos import Memory

USER, SESSION = "alice", "standup"


def _sample_image() -> bytes:
    """A whiteboard photo stand-in, drawn locally so the example needs no assets."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 360), (252, 252, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 624, 344], outline=(60, 60, 60), width=3)
    d.text((40, 60), "SPRINT 14 — GOALS", fill=(20, 20, 20))
    d.text((40, 120), "1. ship the memory API", fill=(20, 20, 20))
    d.text((40, 160), "2. cut recall p95 to 200ms", fill=(20, 20, 20))
    d.text((40, 200), "3. migrate storage to SQLite", fill=(20, 20, 20))
    d.text((40, 270), "owner: Priya   due: Friday", fill=(120, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> None:
    image = _sample_image()

    with Memory() as m:
        blocked = [c for c in m.capabilities() if c.required and not c.available]
        if blocked:
            raise SystemExit(f"{blocked[0].name}: {blocked[0].remedy}")

        vision = next(c for c in m.capabilities() if c.name == "image understanding")
        print(f"image understanding: {'on' if vision.available else 'OFF — ' + vision.remedy}\n")

        receipt = m.add(
            [{"role": "user", "content": "Photo of the whiteboard from today's standup.",
              "image": image, "image_content_type": "image/png"},
             {"role": "assistant", "content": "Got it. I'll remember what was on the board."},
             {"role": "user", "content": "Priya owns all three items this sprint."}],
            user_id=USER, session_id=SESSION)
        for w in receipt.warnings:
            print(f"warning: {w}\n")

        # sync=True: wait until the queued write (and the image's description)
        # has actually been built into memories before reading them back.
        m.end_session(user_id=USER, session_id=SESSION, sync=True, timeout_s=600)

        ctx = m.for_user(USER)
        print("What the image became:")
        for e in ctx.evidence.list(limit=10):
            if e.modality in ("image", "mixed"):
                print(f"  modality : {e.modality}")
                print(f"  stored at: {e.content_ref}")        # None if nothing keeps originals
                print(f"  described: {(e.content_inline or '')[:200]}")
                if e.content_ref:
                    from personos.config import get_config
                    on_disk = Path(get_config().media_root) / e.content_ref
                    print(f"  on disk  : {on_disk.exists()} ({on_disk})")

        print("\nAtoms extracted from the image and the surrounding turns:")
        for a in ctx.atoms.list(limit=20):
            print(f"  [{a.object_type}] {a.text}")

        print("\n" + "─" * 70)
        for question in ("What were the sprint goals on the whiteboard?",
                         "Who owns the sprint items?",
                         "What is the recall latency target?"):
            out = m.search(question, user_id=USER, session_id=SESSION)
            print(f"\nQ: {question}")
            print(f"A: {out.ans.answer if out.ans else '(no answer)'}")
            for w in out.warnings:
                print(f"   warning: {w}")

        m.reset(user_id=USER)
        print("\nDone (this user's data was removed so the example can be re-run).")


if __name__ == "__main__":
    main()
