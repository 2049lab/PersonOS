"""The profile merge engine (deterministic, no LLM): merge the **patch** produced by consolidation
into the current profile.

The division of labour under this clean structure: **the band is decided explicitly by the LLM**
(given as `band` inside add/rewrite), the engine never assigns bands mechanically by date; the engine
only applies the patch, stamps last_confirmed, and **enforces the per-band item cap as a backstop**.

- traits: replaced wholesale per domain; domains not mentioned are kept; None clears explicitly;
  last_confirmed is stamped with today by the engine.
- facts: drop deletes by id / rewrite patches by id (fields not mentioned are kept, sources are
  unioned to preserve traceability) / add gets its f_id assigned by the engine; any fact that was
  touched gets last_confirmed stamped with today; which band it lands in is the LLM's call (the
  `band` field).
- Over cap: each band has an item cap, and going over evicts the oldest by last_confirmed
  (consolidate first gives the LLM a chance to converge semantically; this is the final backstop).

Before merging, profile_harness has already validated that the patch's fields are well-formed and
that sources have been mapped back from short to long handles (see profile_consolidate).
"""

from __future__ import annotations

from ..storage.profile_store import BANDS, ProfileFact, ProfileTrait, UserProfile

# Per-band item caps (see the design doc): going over evicts the oldest by last_confirmed
BAND_CAPS: dict[str, int] = {"today": 1, "week": 3, "month": 5, "long": 21}


def apply_patch(current: UserProfile | None, patch: dict, *, today_str: str,
                evict: bool = True) -> UserProfile:
    """Merge the patch into the current profile and return a new one (the argument is not mutated).

    today_str: YYYY-MM-DD, what the engine stamps as last_confirmed on every touched item.
    evict: True also runs the over-cap eviction backstop (the default); the consolidate loop passes
    False so the LLM gets to converge semantically first.
    """
    p = current.model_copy(deep=True) if current is not None else UserProfile.empty()
    for b in BANDS:
        p.facts.setdefault(b, [])

    _apply_traits(p, patch.get("traits") or {}, today_str)
    _apply_facts(p, patch.get("facts") or {}, today_str)
    if evict:
        enforce_caps(p)
    return p


def _apply_traits(p: UserProfile, traits_patch: dict, today_str: str) -> None:
    for dom, val in traits_patch.items():
        if val is None:
            p.traits[dom] = None                    # explicit clear (valid)
        else:
            p.traits[dom] = ProfileTrait(
                text=val.get("text", ""),
                status=val.get("status", "inferred"),
                last_confirmed=today_str,           # stamped by the engine; never trust what the LLM writes by hand
                sources=list(val.get("sources") or []),
            )


def _find(p: UserProfile, fid: str) -> tuple[str, ProfileFact] | None:
    for b in BANDS:
        for f in p.facts[b]:
            if f.id == fid:
                return b, f
    return None


def _apply_facts(p: UserProfile, facts_patch: dict, today_str: str) -> None:
    for fid in facts_patch.get("drop") or []:
        hit = _find(p, fid)
        if hit:
            b, f = hit
            p.facts[b].remove(f)

    for rw in facts_patch.get("rewrite") or []:
        hit = _find(p, rw.get("id"))
        if not hit:
            continue                                # unknown id: skip silently (the harness already validated this; the backstop just must not blow up)
        b, f = hit
        if "text" in rw:
            f.text = rw["text"]
        if "sources" in rw:
            f.sources = sorted(set(f.sources) | set(rw["sources"] or []))   # union, to preserve traceability
        f.last_confirmed = today_str                # touched -> stamp today
        new_band = rw.get("band")
        if new_band and new_band != b:              # the LLM asked to move bands (e.g. promote to long)
            p.facts[b].remove(f)
            p.facts[new_band].append(f)

    for a in facts_patch.get("add") or []:
        f = ProfileFact(text=a.get("text", ""), last_confirmed=today_str,
                        sources=list(a.get("sources") or []))
        p.facts[a.get("band", "today")].append(f)   # the band is the LLM's call (the harness already guaranteed it is valid)


def over_cap(profile: UserProfile) -> dict[str, int]:
    """Return the bands that are over cap -> their current item count (empty = none are over). Used by
    consolidate to decide whether to send the profile back to the LLM to converge."""
    return {b: len(profile.facts.get(b, [])) for b in BANDS
            if len(profile.facts.get(b, [])) > BAND_CAPS[b]}


def enforce_caps(profile: UserProfile) -> None:
    """Backstop: when a band is over cap, evict the oldest by last_confirmed and keep the newest `cap`
    items (preserving their original relative order)."""
    for b, cap in BAND_CAPS.items():
        lst = profile.facts.get(b, [])
        if len(lst) <= cap:
            continue
        # Take the first `cap` ids by descending last_confirmed (stable: same-date items keep the
        # earlier-inserted one), then filter the list in its original order
        keep = {f.id for f in sorted(lst, key=lambda f: f.last_confirmed, reverse=True)[:cap]}
        profile.facts[b] = [f for f in lst if f.id in keep]
