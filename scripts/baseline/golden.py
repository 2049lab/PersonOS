"""Extract a **deterministic structural fingerprint** from a RecallOutcome, as
the reference point for "the reshaping did not change behaviour".

The central trade-off: **do not compare answer text**. Given the same model
responses the answer text ought to be identical word for word, but a single
stray space, or a change in the order in which materials are concatenated, will
move it — that is noise, not signal. What actually needs pinning down is the
**structure** of the pipeline: how the retrieval scope was decided, which
materials came back, in what order they were ranked, how many review rounds
ran, and whether it escalated. If any of those move, behaviour has moved.

By the same reasoning, this also skips: elapsed time (secs), raw prompt text
(system/user/raw — that is the cassette's job), and floating-point similarity
scores (floats wobble in their last digits across numpy/BLAS builds, so
comparing them only manufactures false alarms; compare the *order* instead).
"""

from __future__ import annotations


def fingerprint(o) -> dict:
    """RecallOutcome -> a JSON-comparable structural fingerprint."""
    rw, asm, ans = o.rw, o.asm, o.ans
    return {
        "query": o.query,
        "mode": o.mode,
        # R0 retrieval scope: subject / time window / domain decide the shape of
        # everything retrieved downstream.
        "rw": None if rw is None else {
            "resolved": rw.resolved,
            "subject": rw.subject,
            "expansions": list(rw.expansions or []),
            "time_start": str(rw.time_start or ""),
            "time_end": str(rw.time_end or ""),
            "domains": list(rw.domains or []),
        },
        # R1: which atoms are in the pool, **in order** (the RRF fusion order is
        # a core algorithmic output).
        "hit_atom_ids": [h.atom.id for h in o.hits],
        # R2: the order of material units after reranking, plus which atoms each
        # unit covers (the result of chain weaving).
        # Note that CellHit.atoms holds AtomHit (score included) rather than the
        # Atom itself, so reaching the id takes one extra hop.
        "ranked": [{"cell_id": c.cell.id, "atoms": sorted(a.atom.id for a in (c.atoms or []))}
                   for c in o.ranked],
        # Unit assembly breakdown: counts of pool / chain / woven / plain units,
        # plus the incompleteness hint.
        "asm": None if asm is None else {
            "n_pool": asm.n_pool, "n_chains": asm.n_chains,
            "n_woven": asm.n_woven, "n_plain": asm.n_plain,
            "boundary": bool(asm.boundary),
        },
        # Review history: how many rounds ran and what each concluded (the
        # critique text itself is excluded).
        "verdicts": [r.verdict for r in o.reviews],
        "retried": o.retried,
        "escalated": o.escalated,
        # Which cells the final answer cited - the structural expression of
        # "what the answer rests on", far more stable than the answer text.
        "cited_cells": sorted(ans.cited_cells or []) if ans else [],
        "has_answer": bool(ans and ans.answer.strip()),
        # Deep path: pin down the tool-call sequence only (it is the structural
        # trace of the agent's decisions).
        "deep_tools": _deep_tools(o.deep),
    }


def _deep_tools(deep) -> list[str]:
    if deep is None:
        return []
    out = []
    for step in getattr(deep, "trace", None) or []:
        if isinstance(step, dict):
            out.append(str(step.get("tool") or step.get("action") or ""))
        else:
            out.append(str(getattr(step, "tool", "")))
    return out


def diff(old: dict, new: dict) -> list[str]:
    """Compare two fingerprints field by field, returning differences in plain
    language. An empty list means behaviour is unchanged."""
    msgs: list[str] = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            only_old = [x for x in a if x not in b]
            only_new = [x for x in b if x not in a]
            if not only_old and not only_new:
                msgs.append(f"{k}: same elements but a different **order**\n    old={a}\n    new={b}")
            else:
                msgs.append(f"{k}: only in old={only_old} only in new={only_new}")
        else:
            msgs.append(f"{k}: {a!r} -> {b!r}")
    return msgs
