"""One week of Alice's life, remembered — the whole loop as a single story.

    export PERSONOS_LLM_API_KEY=sk-...
    python examples/quickstart.py

Storage is SQLite under ~/.personos, created on first use. No server, nothing
to provision.

The story is the point. Alice talks on three days. She says things once and
never repeats them; she corrects herself; she assumes you remember. Between
days the library distils a profile of her that nobody asked for. At the end
we ask questions, and every answer can be traced back to the exact turn it
came from.

What to watch:

- **The profile grows between days.** After Monday it knows almost nothing;
  after Friday it has an occupation, a diet, a city, a preference for blunt
  answers. Nobody called for it — every segment close reconsiders it, on an
  annealing schedule that starts eager and slows as the profile settles.
- **The fish count contradicts itself.** 15 -> 13 -> 11, said across two days.
  Both "how many now" and "did it change" are answerable, because the store
  keeps the sequence instead of overwriting it.
- **The last question has no answer in any episode.** "Where should the team
  eat" is never discussed. It is the *profile* (vegetarian, liked the Sichuan
  place, blunt) that answers it — injected into recall automatically.
- **Recall shows its work.** Each answer comes with the atoms retrieved, the
  materials ranked and the adjudication verdict, so a wrong answer can be
  traced to the stage that produced it.
"""

from __future__ import annotations

import time

from personos import Memory

USER = "alice"

# ── The story ────────────────────────────────────────────────────────────
# Three days of conversation. Things are said once and never repeated, the
# way people actually talk — which is exactly the situation memory has to
# handle.

WEEK = [
    ("Monday", [
        ("user", "I'm a backend engineer — mostly Go and Postgres these days."),
        ("assistant", "Nice. What are you working on?"),
        ("user", "Migrating a payments service off a monolith. Slow going."),
        ("assistant", "Those migrations usually are."),
        ("user", "Also — I'd rather have a blunt answer than a diplomatic one."),
    ]),
    ("Wednesday", [
        ("user", "I'm vegetarian, so most of the team lunch places don't work."),
        ("assistant", "Anywhere you do like?"),
        ("user", "The Sichuan place near the office has good options."),
        ("user", "I moved to Shanghai from Hangzhou in June, still settling in."),
        ("user", "I keep 15 neon tetras in a planted tank at the new place."),
    ]),
    ("Friday", [
        ("user", "Two of the tetras died this week, so I'm down to 13."),
        ("assistant", "Sorry to hear it. Any idea why?"),
        ("user", "Actually I recounted — it's 11. Two jumped out earlier and I missed it."),
        ("assistant", "Worth adding a lid, then."),
        ("user", "This quarter's goal is getting the payments migration to staging."),
    ]),
]

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


def show_profile(m: Memory, day: str, after_version: int) -> int:
    """Print the profile as it stands after a day — nobody asked for it.

    Returns the version printed, so the next day can tell whether the profile
    moved. The rebuild runs on a background thread after each segment close,
    on an annealing schedule — so we wait for the version to advance, and if
    it doesn't, that is the schedule choosing to skip a round (it starts
    eager and slows as the profile settles).
    """
    profile = m.profile(user_id=USER)
    for _ in range(60):
        if profile.get("version", 0) > after_version:
            break
        time.sleep(1)
        profile = m.profile(user_id=USER)

    lines = []
    for key, trait in (profile.get("traits") or {}).items():
        if trait and trait.get("text"):
            lines.append(f"  {key:<18} {trait['text']}  ({trait.get('status')})")
    for band, facts in (profile.get("facts") or {}).items():
        for fact in facts:
            lines.append(f"  [{band:<6}]{'':<9}{fact.get('text')}")

    version = profile.get("version", 0)
    note = "" if version > after_version else \
        " — unchanged: the annealing schedule skipped this round"
    print(f"── profile after {day} (version {version}{note}) "
          f"{'─' * max(4, 44 - len(day) - len(note))}")
    print("\n".join(lines) if lines else "  (nothing yet)")
    print()
    return version


def main() -> None:
    with Memory() as m:
        blocked = [c for c in m.capabilities() if c.required and not c.available]
        if blocked:
            raise SystemExit(f"{blocked[0].name}: {blocked[0].remedy}")

        # Idempotent: a previous run may have died before its final reset().
        m.reset(user_id=USER)

        # ── The week, written as it happens ────────────────────────────
        # add() is not a save button: turns accumulate into a segment, and
        # memories are built when the segment closes — here, at end_session().
        version = 0
        for day, turns in WEEK:
            print(f"══ {day} {'═' * (66 - len(day))}")
            for role, text in turns:
                print(f"  {role:<10} {text}")
            m.add([{"role": r, "content": t} for r, t in turns],
                  user_id=USER, session_id=day.lower())
            m.end_session(user_id=USER, session_id=day.lower())
            print()
            version = show_profile(m, day, version)

        # ── What the week became ───────────────────────────────────────
        ctx = m.for_user(USER)
        print("═" * 70)
        print("What the week became: raw turns kept verbatim, episodes written\n"
              "from them, atoms indexing the episodes.\n")
        for c in ctx.cells.iter_all():
            print(f"  episode · {c.topic}\n            {c.episode[:150]}")
        print()
        for a in ctx.atoms.list(limit=30):
            print(f"  atom [{a.object_type:<5}] {a.text}")

        # ── Questions ──────────────────────────────────────────────────
        print("\n" + "═" * 70)
        print("Questions — note where each answer actually comes from:\n")
        for question, why in QUESTIONS:
            out = m.search(question, user_id=USER)
            print(f"Q: {question}")
            print(f"A: {out.ans.answer if out.ans else '(no answer)'}")
            print(f"   ({why})")
            print(f"   retrieved {len(out.hits)} atoms -> ranked {len(out.ranked)} materials"
                  f" -> verdict {[r.verdict for r in out.reviews]}"
                  f"{' -> escalated to the deep agent' if out.escalated else ''}\n")

        # ── Provenance ─────────────────────────────────────────────────
        atom = ctx.atoms.list(limit=1)[0]
        print("═" * 70)
        print("Every memory traces back to the turn it came from:\n")
        print(f"  atom: {atom.text}")
        print(f"  chain: {m.trace(atom.id, user_id=USER)}")

        m.reset(user_id=USER)
        print("\nDone (this user's data was removed so the example can be re-run).")


if __name__ == "__main__":
    main()
