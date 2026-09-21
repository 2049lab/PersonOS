"""Public view rendering (pure functions that import no runtime and call no model, so they are easy
to unit-test).

The "memory view" of the public contract: clean and traceable, deliberately free of any internal
quantity (scores, RRF scores, prompts, provenance).
"""

from __future__ import annotations

from ..models import atom_anchor
from .retrieval import _TYPE_LABELS
from .trust import evidence_entries


def memory_view(a, evidence_store=None, media_store=None) -> dict:
    """The public view of one memory: clean and traceable.

    Always carries evidence_refs (pointers); when evidence_store is given, the evidence is INLINED
    (the original text plus the Q<->A of the same turn) — so every memory carries its own supporting
    evidence and the public structure stays exactly the same. Evidence entries for image evidence
    carry media_url (a signed URL for the original image).
    """
    t = atom_anchor(a)   # representative time: occurrence -> recorded (same convention as the trust chain)
    return {
        "atom_id": a.id,
        "cell_id": a.memcell_id or None,     # the cell it belongs to (from which the segment narrative topic/episode can be looked up)
        "text": a.text,
        "type": _TYPE_LABELS.get(a.object_type, a.object_type),   # experience / fact / claim
        "holder": a.holder,                  # who said it / whose attribute it is (ownership)
        "domains": a.domains,
        "kind": a.kind,                      # memory type code on the K axis (aligned with the trust chain)
        "occurred_at": t.isoformat() if t else None,
        "evidence_refs": [r.evidence_id for r in a.evidence_refs],   # traceability pointers
        "evidence": (evidence_entries(a, evidence_store, media_store)
                     if evidence_store is not None else []),
    }
