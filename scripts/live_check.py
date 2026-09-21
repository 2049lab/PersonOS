"""End-to-end check against real models, from a user's point of view.

Everything else in the test suite uses fakes, which proves the wiring but not
that the thing works. This drives the public API exactly as someone would after
`pip install personos`, against live endpoints, and inspects what each stage
produced rather than only whether the final answer reads well — a plausible
answer can come out of a pipeline that quietly skipped a stage.

    PERSONOS_ENV_FILE=.env.livetest python scripts/live_check.py

Storage is a throwaway SQLite file and a local media directory, so this never
touches shared infrastructure.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load configuration before importing personos, so the config module sees it.
_env_file = os.environ.get("PERSONOS_ENV_FILE", ".env.livetest")
if Path(_env_file).exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file, override=True)

_ok = True
_failures: list[str] = []


def check(name: str, condition, detail: str = "") -> bool:
    global _ok
    condition = bool(condition)
    _ok &= condition
    if not condition:
        _failures.append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


# A small conversation with the properties worth testing: a fact, an update
# that supersedes it, and a correction of the update. A flat store cannot tell
# these apart; a layered one should keep all three and still answer with the
# current value.
CONVERSATION = [
    {"role": "user", "content": "I'm allergic to peanuts, so I always check menus."},
    {"role": "assistant", "content": "Noted. I'll keep that in mind for restaurant suggestions."},
    {"role": "user", "content": "I keep 15 neon tetras in my planted tank."},
    {"role": "assistant", "content": "That's a good school size for them."},
    {"role": "user", "content": "Two of them died last week, so I'm down to 13."},
    {"role": "assistant", "content": "Sorry to hear that. Any idea what happened?"},
    {"role": "user", "content": "Actually I recounted and it's 11 — two jumped out earlier and I missed it."},
    {"role": "assistant", "content": "Worth adding a lid then."},
]

QUESTIONS = [
    ("how many neon tetras do I have now?", ["11", "eleven"],
     "the correction must win over both earlier counts"),
    ("what am I allergic to?", ["peanut"],
     "a plain fact, retrieved across segment boundaries"),
    ("did the number of fish change over time?", ["15", "13", "11"],
     "the history must survive, not be overwritten by the latest value"),
]


def main() -> int:
    from personos import Memory, MissingCapability

    user = f"live_{int(time.time())}"
    session = "live-session"

    section("0. Configuration")
    with Memory() as m:
        for cap in m.capabilities():
            mark = "on " if cap.available else ("MISSING" if cap.required else "off")
            print(f"  {mark:<8} {cap.name:<26} {cap.detail[:60]}")
        missing = [c.name for c in m.capabilities() if c.required and not c.available]
        if not check("required capabilities are configured", not missing, str(missing)):
            return 1

        section("1. Write: add() then end_session()")
        t0 = time.monotonic()
        receipt = m.add(CONVERSATION, user_id=user, session_id=session)
        add_s = time.monotonic() - t0
        check("add() accepted the batch and returned immediately",
              receipt.accepted and receipt.kind == "ingest" and add_s < 5,
              f"receipt seq={receipt.seq} in {add_s:.2f}s (queueing must not block)")
        check("no unexpected warnings", not receipt.warnings, str(receipt.warnings))

        t0 = time.monotonic()
        m.end_session(user_id=user, session_id=session, sync=True, timeout_s=600)
        end_s = time.monotonic() - t0
        print(f"  end_session(sync=True) drained the queue in {end_s:.1f}s")
        cells = m.for_user(user).cells.iter_all()
        check("end_session() built memories", cells, f"{len(cells)} cells")

        # Inspect what was actually built, not just that something was.
        ctx = m.for_user(user)
        all_cells = ctx.cells.iter_all()
        atoms = ctx.atoms.list(limit=200)
        evidence = ctx.evidence.list(limit=200)
        print(f"\n  stored: {len(evidence)} evidence, {len(all_cells)} cells, {len(atoms)} atoms")
        for c in all_cells:
            print(f"    cell  topic={c.topic[:70]!r}")
        for a in atoms:
            print(f"    atom  [{a.object_type}/{a.kind}] holder={a.holder!r} {a.text[:76]}")

        check("evidence covers every turn", len(evidence) >= len(CONVERSATION),
              f"{len(evidence)} >= {len(CONVERSATION)}")
        check("at least one episode was built", len(all_cells) >= 1)
        check("atoms were extracted", len(atoms) >= 3,
              "atoms are the only retrieval entry; none means the segment is unreachable")
        check("atoms carry a holder", all(a.holder for a in atoms))
        check("atoms are linked to their episode", all(a.memcell_id for a in atoms))

        # The contradiction must be preserved rather than collapsed. What that
        # looks like at the atom level is model-dependent: one extractor writes
        # an atom per count, another writes the *events* that changed it ("two
        # died", "two jumped out") and treats the intermediate count as derived.
        # Both are faithful, so assert the property rather than one shape of it —
        # the narrative must still contain the whole sequence, and question 3
        # below proves it is recoverable.
        episode = " ".join(c.episode for c in all_cells)
        check("the episode retains the full sequence, not just the final value",
              all(n in episode for n in ("15", "13", "11")),
              "a flat store would keep only the last value")
        superseded = {n for n in ("15", "13") if any(n in a.text for a in atoms)}
        events = [a for a in atoms if a.object_type == "event"]
        check("the change itself is recorded as atoms", superseded or events,
              f"superseded counts={sorted(superseded)}, change events={len(events)}")

        section("2. Read: search()")
        for question, expected, why in QUESTIONS:
            t0 = time.monotonic()
            out = m.search(question, user_id=user, session_id=session)
            secs = time.monotonic() - t0
            answer = (out.ans.answer if out.ans else "") or ""
            print(f"\n  Q: {question}")
            print(f"  A: {answer[:200]}")
            print(f"     rewritten={out.rw.resolved[:60]!r} subject={out.rw.subject!r}"
                  if out.rw else "     (no rewrite)")
            print(f"     retrieved={len(out.hits)} atoms, ranked={len(out.ranked)} materials, "
                  f"verdicts={[r.verdict for r in out.reviews]}, escalated={out.escalated}, "
                  f"{secs:.1f}s")

            check(f"answered: {question[:38]}", answer.strip(), why)
            check("  retrieval found candidates", out.hits,
                  f"{len(out.hits)} atoms — empty means embedding or search failed")
            check("  materials were ranked", out.ranked, f"{len(out.ranked)} units")
            check("  the answer cites its sources", out.ans and out.ans.cited_cells,
                  str(out.ans.cited_cells if out.ans else []))
            check("  content is right", any(e.lower() in answer.lower() for e in expected),
                  f"expected one of {expected}")

        section("3. to_public(): the shape the server also returns")
        out = m.search(QUESTIONS[0][0], user_id=user, session_id=session)
        pub = out.to_public(atoms=ctx.atoms, evidence=ctx.evidence)
        print(f"  keys: {sorted(pub)}")
        check("carries the answer and its citations",
              {"answer", "cited_cells", "memories", "verdict"} <= set(pub))
        check("hides internals", not ({"hits", "ranked", "rw", "secs"} & set(pub)),
              "scores and prompts must not become contract")
        check("memories carry provenance",
              all(mm.get("atom_id") and "evidence" in mm for mm in pub["memories"]) or not pub["memories"],
              f"{len(pub['memories'])} memories")

        section("4. trace(): provenance in both directions")
        if atoms:
            tr = m.trace(atoms[0].id, user_id=user)
            check("an atom traces back to evidence", tr,
                  f"{list(tr)[:6] if isinstance(tr, dict) else type(tr).__name__}")

        section("5. Capability contract: asking for what is not configured")
        try:
            m.search("anything", user_id=user, mode="deep")
            check("deep mode either works or explains itself", True, "deep recall is installed")
        except MissingCapability as e:
            check("deep mode explains itself", "pip install" in e.remedy, e.remedy)

        section("6. reset(): a user's data really goes away")
        m.reset(user_id=user)
        after = m.for_user(user)
        check("evidence removed", not after.evidence.list(limit=10))
        check("atoms removed", not after.atoms.list(limit=10))
        check("cells removed", not after.cells.iter_all())

    section("Result")
    print(f"  {'ALL CHECKS PASSED' if _ok else 'FAILURES: ' + ', '.join(_failures)}")
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
