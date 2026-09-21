"""Remember an image alongside the conversation it appeared in.

Needs a multimodal model on top of the quickstart setup:

    export PERSONOS_LLM_API_KEY=sk-...
    export PERSONOS_MLLM_API_KEY=sk-...   # may be the same key
    export PERSONOS_MLLM_MODEL=gpt-4o

Without those the image is still stored as evidence, and the result carries a
warning saying it contributed nothing to retrieval — the text path is
unaffected either way.
"""

import sys
from pathlib import Path

from personos import Memory

if len(sys.argv) < 2:
    raise SystemExit("usage: python examples/image_memory.py <path-to-image>")

image_path = Path(sys.argv[1])

with Memory() as m:
    result = m.add(
        [{"role": "user", "content": "This is the whiteboard from today's planning session.",
          "image": image_path}],
        user_id="alice", session_id="planning")
    for w in result.warnings:
        print(f"warning: {w}")

    m.end_session(user_id="alice", session_id="planning")
    out = m.search("what was on the whiteboard?", user_id="alice")
    print("Answer:", out.ans.answer if out.ans else "(none)")
