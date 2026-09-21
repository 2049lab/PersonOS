"""A person talking across several days: write, watch the profile fill in, ask.

    export PERSONOS_LLM_API_KEY=sk-...
    python examples/quickstart.py

Storage is SQLite under ~/.personos, created on first use. No server, no
database to provision.

This is the whole loop, in the order it actually happens.

**Writing.** add() is not a save button. Turns accumulate into a segment, and
memories are built when the segment closes — either because the topic moved on,
or because you called end_session().

**The profile builds itself.** Nothing here calls for it. Every time a segment
closes, the profile is reconsidered; whether it actually rebuilds is decided by
an annealing schedule that starts eager and slows down as the profile settles.
So it is sharp after the first conversation and stops churning once it knows
someone.

**Contradictions survive.** The fish count goes 15 -> 13 -> 11. "How many now"
answers 11; "did it change" recovers the whole sequence. Both are true at once,
which a store that overwrote on update could not manage.

**Recall shows its work.** search() returns the retrieved atoms, the ranked
materials and the adjudication verdict next to the answer, so a wrong answer
can be traced to the stage that produced it.
"""

from __future__ import annotations

import time

from personos import Memory

USER = "alice"

# Several days of conversation. Things are said once and never repeated, the
# way people actually talk — which is the situation memory has to handle.
DAYS = {
    "monday": [
        {"role": "user", "content": "I'm a backend engineer — mostly Go and Postgres these days."},
        {"role": "assistant", "content": "Nice. What are you working on?"},
        {"role": "user", "content": "Migrating a payments service off a monolith. Slow going."},
        {"role": "assistant", "content": "Those migrations usually are."},
        {"role": "user", "content": "Also, I'd rather have a blunt answer than a diplomatic one."},
    ],
    "wednesday": [
        {"role": "user", "content": "I'm vegetarian, so most of the team lunch places don't work."},
        {"role": "assistant", "content": "Anywhere you do like?"},
        {"role": "user", "content": "The Sichuan place near the office has good options."},
        {"role": "user", "content": "I moved to Shanghai from Hangzhou in June, still settling in."},
        {"role": "user", "content": "I keep 15 neon tetras in a planted tank at the new place."},
    ],
    "friday": [
        {"role": "user", "content": "Two of the tetras died this week, so I'm down to 13."},
        {"role": "assistant", "content": "Sorry to hear it. Any idea why?"},
        {"role": "user", "content": "Actually I recounted — it's 11. Two jumped out earlier and I missed it."},
        {"role": "assistant", "content": "Worth adding a lid, then."},
        {"role": "user", "content": "This quarter's goal is getting the payments migration to staging."},
    ],
}

QUESTIONS = [
    ("How many neon tetras do I have now?",
     "the correction wins — 11, not 15 or 13"),
    ("Did the number of fish change over time?",
     "and yet the whole sequence is still recoverable"),
    ("What am I working on?",
     "stated on Monday, refined on Friday, never repeated"),
    ("Suggest somewhere for lunch with the team tomorrow.",
     "nothing in any episode answers this — the profile does"),
]


def main() -> None:
    with Memory() as m:
        blocked = [c for c in m.capabilities() if c.required and not c.available]
        if blocked:
            raise SystemExit(f"{blocked[0].name}: {blocked[0].remedy}")

        # ── Write ──────────────────────────────────────────────────────
        for day, turns in DAYS.items():
            m.add(turns, user_id=USER, session_id=day)
            m.end_session(user_id=USER, session_id=day)
            print(f"wrote {day}: {len(turns)} turns")

        ctx = m.for_user(USER)
        print(f"\nStored: {len(ctx.evidence.list(limit=500))} evidence, "
              f"{len(ctx.cells.iter_all())} episodes, "
              f"{len(ctx.atoms.list(limit=500))} atoms")

        print("\nEpisodes — the narrative unit answers are written from:")
        for c in ctx.cells.iter_all():
            print(f"  {c.topic}")

        print("\nAtoms — the retrieval index; short, self-contained, embeddable:")
        for a in ctx.atoms.list(limit=30):
            print(f"  [{a.object_type:<5}] {a.text}")

        # ── The profile, which nothing above asked for ─────────────────
        # It is built on a background thread, so give it a moment to land.
        # (Call consolidate_profile() if you would rather not wait.)
        for _ in range(30):
            profile = m.profile(user_id=USER)
            if profile:
                break
            time.sleep(1)
        else:
            profile = m.consolidate_profile(user_id=USER) and m.profile(user_id=USER)

        print("\n" + "─" * 70)
        print("Profile — distilled from the episodes, not written by anyone:\n")
        for key, trait in (profile.get("traits") or {}).items():
            if trait and trait.get("text"):
                print(f"  {key:<20} {trait['text']}  ({trait.get('status')})")
        for band, facts in (profile.get("facts") or {}).items():
            for fact in facts:
                print(f"  [{band:<6}]{'':<14}{fact.get('text')}")

        # ── Ask ────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        for question, why in QUESTIONS:
            out = m.search(question, user_id=USER)
            print(f"\nQ: {question}")
            print(f"A: {out.ans.answer if out.ans else '(no answer)'}")
            print(f"   ({why})")
            print(f"   retrieved {len(out.hits)} atoms -> ranked {len(out.ranked)} materials"
                  f" -> verdict {[r.verdict for r in out.reviews]}"
                  f"{' -> escalated to the deep agent' if out.escalated else ''}")

        # ── Provenance ─────────────────────────────────────────────────
        atom = ctx.atoms.list(limit=1)[0]
        print("\n" + "─" * 70)
        print("Every memory traces back to the turn it came from:")
        print(f"  {atom.text}")
        print(f"  {m.trace(atom.id, user_id=USER)}")

        m.reset(user_id=USER)
        print("\nDone (this user's data was removed so the example can be re-run).")


if __name__ == "__main__":
    main()
