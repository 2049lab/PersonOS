"""Physical consistency checks: find self-contradictions in the model's output
that are **impossible in the real world**.

A multimodal LLM hallucinates, and the hallucinations most worth blocking
mechanically are the physically impossible attribution errors — one person taken
for two characters, two people in the same frame taken for one person, one person
in two places at once. Code judges these far more reliably than asking the model
to check itself, so we detect them mechanically first and then ask the model to
fix exactly those (see repair.py).

The rules:
  1 voice_overlap         two people's voice ranges intersect, so their
                          voiceprints would be mixed into one template
  2 cont_conflict         two casts continue the same earlier member — one person
                          cannot become two
  3 nom_position_conflict one cast nominated at two positions at the same instant
                          — people do not split in two
  5 duplicate_cast        the same id declared more than once
  6 bind_collision        same-frame casts bound to the same character; judged
                          during arbitration, since it needs the proposed bindings
  7 wearer_visible        the wearer nominated as visible — a first-person camera
                          cannot film its own wearer
  8 name_unsupported      a name that appears in no speech line of this clip, yet
                          claims speech-based evidence
  9 name_self_address     a name that appears only in that person's own lines —
                          people do not call themselves by name

**Detection only: these are pure functions with no side effects.** Repair and
degrade live in repair.py, and rule 6's re-arbitration lives in chains.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from personos.identity.screenplay import ENV_WHO, WEARER_CAST_ID, ClipScript


@dataclass(frozen=True)
class Violation:
    """One physical contradiction: the rule name, the casts involved, and a human-readable detail."""

    rule: str
    cast_ids: tuple[str, ...]
    detail: str
    times: tuple[float, ...] = ()


def present_casts(script: ClipScript) -> set[str]:
    """The session casts actually present in this clip — those with a line, a
    nomination or a voice range.

    An empty cast declaration does not count as being in frame with anyone.
    """
    present: set[str] = set()
    for line in script.lines:
        if line.who != ENV_WHO:
            cast_id = script.cast_map.get(line.who)
            if cast_id:
                present.add(cast_id)
    for nom in script.nominations:
        cast_id = script.cast_map.get(nom.local_id)
        if cast_id:
            present.add(cast_id)
    for vr in script.voice_ranges:
        cast_id = script.cast_map.get(vr.local_id)
        if cast_id:
            present.add(cast_id)
    return present


def inspect_bind_collisions(script: ClipScript, proposed: dict[str, str]) -> list[Violation]:
    """Rule 6: casts in the same frame colliding on one character.

    proposed is {session cast_id -> character_id}, the tentative pre-commit
    bindings. NEW and empty values are excluded, since a fresh profile is
    independent and cannot collide. The wearer and empty cast shells are excluded
    too.
    """
    present = present_casts(script)
    groups: dict[str, list[str]] = {}
    for cast_id, character_id in proposed.items():
        if cast_id == WEARER_CAST_ID or not character_id or character_id == "NEW":
            continue
        if cast_id not in present:
            continue
        groups.setdefault(character_id, []).append(cast_id)
    return [
        Violation(
            rule="bind_collision", cast_ids=tuple(sorted(members)),
            detail=(f"co-occurring casts {', '.join(sorted(members))} all bound to "
                    f"{character_id} — people appearing together in one clip are"
                    " physically distinct"),
        )
        for character_id, members in groups.items()
        if len(members) > 1
    ]


# ── Rules 1/2/3/5/7: physical contradictions internal to the screenplay.
#    Checkable straight after parsing, with no dependency on cast_map ──────

_NOM_SAME_MOMENT_SEC = 0.5   # nominations closer together than this count as "the same instant"
_MIN_VOICE_SEC = 0.4         # matches harvest._MIN_VOICE_SEC: a leftover shorter than this is worthless for a voiceprint


def inspect_voice_overlap(script: ClipScript) -> list[Violation]:
    """Rule 1: the voice ranges of two **different** casts intersect.

    The prompt asks explicitly for ranges where that person speaks alone, so an
    intersection means the model broke its own contract and at least one of the
    two is wrong. Left unfixed, two people's voices get mixed into one voiceprint
    template, and that contamination is irreversible.
    """
    out: list[Violation] = []
    rs = [v for v in script.voice_ranges if v.local_id != ENV_WHO]
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            if a.local_id == b.local_id:
                continue
            lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
            if hi <= lo:
                continue
            out.append(Violation(
                rule="voice_overlap", cast_ids=tuple(sorted((a.local_id, b.local_id))),
                detail=(f"voice ranges overlap: {a.local_id} [{a.t0:.1f},{a.t1:.1f}] vs "
                        f"{b.local_id} [{b.t0:.1f},{b.t1:.1f}] — a voice range must cover "
                        f"only that person speaking ALONE"),
                times=(lo, hi)))
    return out


def inspect_cont_conflict(script: ClipScript) -> list[Violation]:
    """Rule 2: two casts both claim to continue the same earlier member — one person cannot become two."""
    by_prev: dict[str, list[str]] = {}
    for local_id, prev in (script.cont or {}).items():
        if prev and prev != "none":
            by_prev.setdefault(prev, []).append(local_id)
    return [
        Violation(rule="cont_conflict", cast_ids=tuple(sorted(ids)),
                  detail=(f"{', '.join(sorted(ids))} all continue roster member {prev} — "
                          f"one earlier person can continue as at most ONE cast here"))
        for prev, ids in by_prev.items() if len(ids) > 1
    ]


def inspect_nom_position(script: ClipScript) -> list[Violation]:
    """Rule 3: one cast nominated at two horizontal positions at the same instant —
    people do not split in two.

    At least one of those nominations points at the wrong person, and harvesting a
    face from it files **someone else's face** under this person. Once that is
    learned into the probability cloud it affects every later recognition, which
    makes this rule more serious than it looks.
    """
    out: list[Violation] = []
    by_cast: dict[str, list] = {}
    for nom in script.nominations:
        if nom.pos:
            by_cast.setdefault(nom.local_id, []).append(nom)
    for local_id, noms in by_cast.items():
        noms = sorted(noms, key=lambda n: n.t)
        for i, a in enumerate(noms):
            for b in noms[i + 1:]:
                if b.t - a.t > _NOM_SAME_MOMENT_SEC:
                    break
                if a.pos != b.pos:
                    out.append(Violation(
                        rule="nom_position_conflict", cast_ids=(local_id,),
                        detail=(f"{local_id} nominated at {a.pos} (t={a.t:.1f}) and "
                                f"{b.pos} (t={b.t:.1f}) within "
                                f"{_NOM_SAME_MOMENT_SEC}s — one person cannot be in two places"),
                        times=(a.t, b.t)))
    return out


def inspect_duplicate_cast(script: ClipScript) -> list[Violation]:
    """Rule 5: the same id declared more than once.

    The parser keeps only the first, and a repeated declaration usually means the
    model contradicted itself.
    """
    seen: dict[str, int] = {}
    for c in script.casts:
        seen[c.local_id] = seen.get(c.local_id, 0) + 1
    return [Violation(rule="duplicate_cast", cast_ids=(cid,),
                      detail=f"cast {cid} declared {n} times — declare each person once")
            for cid, n in seen.items() if n > 1]


def inspect_wearer_visible(script: ClipScript) -> list[Violation]:
    """Rule 7: the wearer nominated as visible in frame.

    A first-person camera cannot film its own wearer, so that face certainly
    belongs to somebody else.
    """
    times = tuple(n.t for n in script.nominations if n.local_id == WEARER_CAST_ID)
    if not times:
        return []
    return [Violation(
        rule="wearer_visible", cast_ids=(WEARER_CAST_ID,),
        detail=(f"wearer {WEARER_CAST_ID} nominated as visible at "
                f"{', '.join(f'{t:.1f}s' for t in times)} — the camera wearer is never"
                f" in frame; that face belongs to someone else"),
        times=times)]


def inspect_name_claims(script: ClipScript) -> list[Violation]:
    """Rules 8 and 9: reconcile names against the transcript. This is purely a
    defence against model hallucination.

    8 name_unsupported: the name is claimed to come from speech, but it appears in
      no speech line of this clip. The failure mode we actually observed is the
      model parroting a known name off the roster and fabricating
      explicit_dialogue for it.
    9 name_self_address: the name appears only in that person's own lines and is
      not a self-introduction — people do not call themselves by name.

    visible_text is exempt, since on-screen text cannot be reconciled against the
    transcript.
    """
    spoken_by: dict[str, str] = {}
    for line in script.lines:
        if line.kind == "speech" and line.text:
            spoken_by[line.who] = spoken_by.get(line.who, "") + " " + line.text
    all_speech = " ".join(spoken_by.values())
    out: list[Violation] = []
    for c in script.casts:
        name = (c.name or "").strip()
        ev = (c.name_evidence or "none").lower()
        if not name or ev in ("none", "visible_text"):
            continue
        if name.lower() not in all_speech.lower():
            out.append(Violation(
                rule="name_unsupported", cast_ids=(c.local_id,),
                detail=(f"{c.local_id} claims name {name!r} with evidence {ev!r}, but "
                        f"{name!r} never appears in any speech line of this clip")))
            continue
        if ev == "self_introduction":
            continue
        others = " ".join(t for who, t in spoken_by.items() if who != c.local_id)
        if name.lower() not in others.lower():
            out.append(Violation(
                rule="name_self_address", cast_ids=(c.local_id,),
                detail=(f"{name!r} appears only in {c.local_id}'s own lines and is not a"
                        f" self-introduction — people do not call themselves by name")))
    return out


# Rules repair can fix: logical contradictions, where a second look gives the
# model a real chance of correcting itself.
REPAIRABLE = ("voice_overlap", "cont_conflict", "nom_position_conflict",
              "duplicate_cast", "wearer_visible")
# Degrade only, never repair: whether a name is supported by speech is something
# code judges more reliably than the model, and re-asking mostly gets the original
# answer parroted back.
DEGRADE_ONLY = ("name_unsupported", "name_self_address")


def inspect_script(script: ClipScript) -> list[Violation]:
    """Run every screenplay-level check (rules 1/2/3/5/7/8/9).

    Rule 6 needs the binding results, so it is checked separately during
    arbitration.
    """
    return [*inspect_voice_overlap(script), *inspect_cont_conflict(script),
            *inspect_nom_position(script), *inspect_duplicate_cast(script),
            *inspect_wearer_visible(script), *inspect_name_claims(script)]
