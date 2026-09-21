"""Profile rendering: a structured UserProfile -> the plain-text block injected internally (R0 / R5 /
deep track).

Different from the public /profile endpoint, which returns structured JSON only; the text block here
is fed into the R0 / R5 / deep-track prompts and is rendered deterministically from the structure
(the date prefix and the confirmed/inferred markers are products of rendering, never written by hand
by an agent). It carries a disclaimer: the profile may guide "where to look and how to answer", but
the factual basis must still come from the memory materials (the boundary it shares with R3'
adjudication).
"""

from __future__ import annotations

from datetime import datetime

from ..storage.profile_store import BANDS, PMO16, UserProfile

_DISCLAIMER = ("USER PROFILE (what is known about the person asking. Use it to interpret "
               "the question and shape the answer; never cite it as evidence):")

# Display labels for the profile dimensions. This text is injected into the
# rewriting and answering prompts, so it is part of what the model reads.
_KEY_LABELS = {
    "personality": "personality", "communication_style": "communication style",
    "social_style": "social style", "occupation": "occupation", "goals": "goals",
    "values": "values", "work_style": "work style", "learning_style": "learning style",
    "tech_environment": "tech environment", "lifestyle": "lifestyle",
    "health": "health", "finance": "finance",
    "identity": "identity", "location": "location", "family": "family",
    "interests": "interests",
}
_STATUS_LABELS = {"confirmed": "confirmed", "inferred": "inferred"}


def _fmt_date(s: str) -> str:
    """Normalise a date for display; returned unchanged when it cannot be parsed."""
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return d.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return s or "?"


def render(profile: UserProfile | None, *, mode: str = "full") -> str:
    """Render the injection block. mode='full' (traits + facts, for R0 and the deep track) |
    'traits' (traits only, for R5). An empty profile renders as ''."""
    if profile is None:
        return ""
    lines: list[str] = []

    trait_lines = []
    for key in PMO16:                                    # fixed order (L1 -> L3), stable and unit-testable
        t = profile.traits.get(key)
        if t and t.text:
            lc = f", last confirmed {_fmt_date(t.last_confirmed)}" if t.last_confirmed else ""
            trait_lines.append(f"{_KEY_LABELS.get(key, key)}: {t.text}({_STATUS_LABELS.get(t.status, t.status)}{lc})")
    if trait_lines:
        lines.append("[Traits]")
        lines.extend(trait_lines)

    if mode == "full":
        fact_lines = []
        for band in BANDS:                               # today -> long
            for f in profile.facts.get(band, []):
                if f.text:
                    fact_lines.append(f"- {f.text}")     # the date is already in the body text, so rendering adds no prefix
        if fact_lines:
            lines.append("[Recent facts]")
            lines.extend(fact_lines)

    if not lines:                                        # nothing at all -> empty string (the consumer's None path)
        return ""
    return _DISCLAIMER + "\n" + "\n".join(lines)
