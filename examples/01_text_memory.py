"""Text memory end to end: write, close the session, then ask.

    export PERSONOS_LLM_API_KEY=sk-...
    python examples/01_text_memory.py

Storage is SQLite under ~/.personos, created on first use. No server, no
database to provision.

What this shows, beyond "it answers":

- **add() is not a save button.** Turns accumulate into a segment; memories are
  built when the segment closes. That is why end_session() matters.
- **Contradictions are kept, not overwritten.** The fish count goes 15 -> 13 ->
  11. Asking "how many now" gives 11; asking "did it change" recovers all three.
  A store that overwrote on update could not do the second one.
- **Recall shows its work.** search() returns the retrieved atoms, the ranked
  materials and the adjudication verdict alongside the answer, so a wrong
  answer can be traced to the stage that produced it.
"""

from personos import Memory

USER, SESSION = "alice", "tank-talk"

CONVERSATION = [
    {"role": "user", "content": "I'm allergic to peanuts, so I always check menus."},
    {"role": "assistant", "content": "Noted — I'll keep that in mind for restaurant picks."},
    {"role": "user", "content": "I keep 15 neon tetras in my planted tank."},
    {"role": "assistant", "content": "That's a good school size for them."},
    {"role": "user", "content": "Two died last week, so I'm down to 13."},
    {"role": "assistant", "content": "Sorry to hear that. Any idea what happened?"},
    {"role": "user", "content": "Actually I recounted — it's 11. Two jumped out earlier and I missed it."},
    {"role": "assistant", "content": "Worth adding a lid, then."},
]


def main() -> None:
    with Memory() as m:
        blocked = [c for c in m.capabilities() if c.required and not c.available]
        if blocked:
            raise SystemExit(f"{blocked[0].name}: {blocked[0].remedy}")

        print("Writing the conversation...")
        m.add(CONVERSATION, user_id=USER, session_id=SESSION)

        # Until the segment closes, nothing has been distilled yet — the raw
        # turns are stored, but no episode and no atoms exist.
        print("Closing the session (this is where memories get built)...")
        cells = m.end_session(user_id=USER, session_id=SESSION)
        print(f"  built {len(cells)} episode(s)\n")

        # The three layers, so the structure is visible rather than implied.
        ctx = m.for_user(USER)
        print("Evidence — the raw turns, never edited:")
        for e in ctx.evidence.list(limit=3):
            print(f"  {e.holder}: {(e.content_inline or '')[:68]}")
        print("\nEpisode — the narrative unit that answers are written from:")
        for c in ctx.cells.iter_all():
            print(f"  topic: {c.topic}")
            print(f"  {c.episode[:220]}...")
        print("\nAtoms — the retrieval index; short, self-contained, embeddable:")
        for a in ctx.atoms.list(limit=20):
            print(f"  [{a.object_type}] {a.text}")

        print("\n" + "─" * 70)
        for question in ("How many neon tetras do I have now?",
                         "Did the number of fish change over time?",
                         "What am I allergic to?"):
            out = m.search(question, user_id=USER, session_id=SESSION)
            print(f"\nQ: {question}")
            print(f"A: {out.ans.answer if out.ans else '(no answer)'}")
            # How it got there — useful when the answer is wrong.
            print(f"   retrieved {len(out.hits)} atoms -> ranked {len(out.ranked)} materials"
                  f" -> verdict {[r.verdict for r in out.reviews]}"
                  f"{' -> escalated to the deep agent' if out.escalated else ''}")
            print(f"   cited: {out.ans.cited_cells if out.ans else []}")

        print("\n" + "─" * 70)
        print("Provenance — every memory can be traced back to the turn it came from:")
        atom = ctx.atoms.list(limit=1)[0]
        print(f"  atom {atom.id}: {atom.text}")
        print(f"  trace: {m.trace(atom.id, user_id=USER)}")

        m.reset(user_id=USER)           # keep the example re-runnable
        print("\nDone (this user's data was removed so the example can be re-run).")


if __name__ == "__main__":
    main()
