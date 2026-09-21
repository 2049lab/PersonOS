"""Trust-chain assembly: the adopted atoms -> their ownership / epistemic status -> drilling down to
the original evidence content.

It is returned alongside the answer so the consumer can judge "is this answer trustworthy?"
(explainability and trust). The atom -> evidence pointers already exist (evidence_refs); all this
module does is assemble them into a readable structure and surface it.
"""

from __future__ import annotations

from ..storage.atom_store import AtomStore
from ..storage.evidence_store import EvidenceStore


def _media_url(ev, media_store) -> str | None:
    """Image evidence -> a signed URL for the original image (for the frontend to display); not an
    image, no stored copy, or no signer -> None."""
    if media_store is None or ev.modality not in ("image", "mixed") or not ev.content_ref:
        return None
    try:
        return media_store.sign_url(ev.content_ref)
    except Exception:   # noqa: BLE001  a signing failure must not affect the traceability payload itself
        return None


def evidence_entries(atom, evidence_store: EvidenceStore, media_store=None) -> list[dict]:
    """Assemble one atom's evidence_refs into a list of evidence entries: each carries the original
    text plus the assistant reply from the same turn (the Q<->A pairing).

    Shared by the public memory view (recall's `memories`) and the trust chain (trust chain / trace),
    which keeps the evidence sub-structure consistent. Image evidence additionally carries `modality`
    and `media_url` (a signed URL for the original image); without a media_store only `modality` is
    set.
    """
    evs = []
    for ref in atom.evidence_refs:
        ev = evidence_store.get(ref.evidence_id)
        if ev is None:
            continue
        entry = {
            "id": ev.id,
            "holder": ev.holder,
            "content": ev.content_inline,
            "captured_at": ev.captured_at.isoformat() if ev.captured_at else None,
        }
        if ev.modality != "text":
            entry["modality"] = ev.modality
            url = _media_url(ev, media_store)
            if url:
                entry["media_url"] = url
        reply = evidence_store.reply_for(ev.id)   # the assistant reply from the same turn (if any) -> a complete Q<->A
        if reply is not None:
            entry["reply"] = {
                "id": reply.id,
                "content": reply.content_inline,
                "captured_at": reply.captured_at.isoformat() if reply.captured_at else None,
            }
        evs.append(entry)
    return evs


def _ev_dict(ev, media_store=None) -> dict:
    d = {"id": ev.id, "holder": ev.holder, "content": ev.content_inline,
         "captured_at": ev.captured_at.isoformat() if ev.captured_at else None}
    if ev.modality != "text":
        d["modality"] = ev.modality
        url = _media_url(ev, media_store)
        if url:
            d["media_url"] = url
    return d


def trace_evidence(evidence_id: str, evidence_store: EvidenceStore, atom_store: AtomStore,
                   media_store=None) -> dict | None:
    """Trace backwards from an evidence_id: reconstruct the WHOLE dialogue pair (always in
    user -> assistant order) plus WHICH MEMORIES CITE IT.

    Whichever half you look up, the pair is completed:
    - found the user utterance -> attach the assistant reply from the same turn (the one whose
      reply_to points at it);
    - found the assistant utterance -> follow reply_to back to the user utterance it answered.
    cited_by is based on the USER half (memories cite user statements). This is the dual of
    build_trust_chain (which goes forward, atom -> evidence). Not found -> None.
    """
    ev = evidence_store.get(evidence_id)
    if ev is None:
        return None
    # Locate the user and assistant halves of this turn
    if ev.holder == "assistant":
        user_ev = evidence_store.get((ev.source or {}).get("reply_to", "") or "")
        asst_ev = ev
    else:
        user_ev = ev
        asst_ev = evidence_store.reply_for(ev.id)
    pair = [_ev_dict(e, media_store) for e in (user_ev, asst_ev) if e is not None]   # user -> assistant
    anchor_id = user_ev.id if user_ev is not None else evidence_id      # memories cite the user half
    cited_by = [a.id for a in atom_store.list(limit=10000)
                if any(r.evidence_id == anchor_id for r in a.evidence_refs)]
    return {"node": "evidence", "evidence_id": ev.id, "pair": pair, "cited_by": cited_by}


def build_trust_chain(
    atom_ids: list[str],
    atom_store: AtomStore,
    evidence_store: EvidenceStore,
    media_store=None,
) -> list[dict]:
    """Assemble a list of adopted atom ids into a trust chain: each entry carries ownership /
    epistemic status plus the evidence text it drills down to. Entries that cannot be found are
    marked missing (the fallback must not crash)."""
    chain: list[dict] = []
    seen: set[str] = set()
    for aid in atom_ids:
        if not aid or aid in seen:
            continue
        seen.add(aid)
        a = atom_store.get(aid)
        if a is None:
            chain.append({"atom_id": aid, "missing": True})
            continue
        evs = evidence_entries(a, evidence_store, media_store)   # evidence + Q<->A (the same assembly recall's memories use)
        chain.append({
            "atom_id": a.id,
            "text": a.text,
            "cell_id": a.memcell_id or None,  # the cell it belongs to (from which the segment narrative can be looked up)
            "object_type": a.object_type,
            "holder": a.holder,               # who said it / whose attribute it is (ownership)
            "domains": a.domains,
            "kind": a.kind,
            "evidence": evs,
            "evidence_count": len(evs),       # number of independent pieces of evidence -- an intuitive trust signal
        })
    return chain
