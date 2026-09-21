"""The smallest useful thing: remember a conversation, then ask about it.

    export PERSONOS_LLM_API_KEY=sk-...
    python examples/quickstart.py

Nothing else is configured. Storage is SQLite under ~/.personos, created on
first use; there is no server to start and no database to provision.
"""

from personos import Memory

USER = "alice"
SESSION = "quickstart"

CONVERSATION = [
    {"role": "user", "content": "I moved from Hangzhou to Shanghai in June for a new job."},
    {"role": "assistant", "content": "How is the new place?"},
    {"role": "user", "content": "Good, but the commute is four stops now instead of a ten minute walk."},
    {"role": "user", "content": "Actually I checked and it is six stops, I miscounted."},
]

with Memory() as m:
    # Capabilities are worth printing once: they tell you what this
    # configuration can and cannot do, and what to set for the rest.
    for cap in m.capabilities():
        if cap.required and not cap.available:
            raise SystemExit(f"{cap.name} is required: {cap.remedy}")

    m.add(CONVERSATION, user_id=USER, session_id=SESSION)
    # Closing the session builds memories from the trailing segment. Without
    # this the last segment stays open, waiting for turns that never come.
    m.end_session(user_id=USER, session_id=SESSION)

    out = m.search("how long is my commute?", user_id=USER)
    print("\nAnswer:", out.ans.answer if out.ans else "(none)")

    # The correction is kept as its own atom rather than overwriting the first
    # statement — the memory layer groups contradictions, the answer layer
    # decides which one is current.
    print("\nWhat was retrieved:")
    for hit in out.hits[:5]:
        print(f"  - {hit.atom.text}")
