"""ChainBook unit tests backed by MemoryDraftStore, with no network and no model calls.

Covers the four refresh triggers of observe_clip, the co-presence merge guard in apply_verdict,
vote_summary/strong_majority, and pending_cards excluding co-present chains. Matched point by point against the
reference implementation's chain semantics.
"""

from __future__ import annotations

import numpy as np

from personos.identity.chains import (
    BETTER_FACE,
    FIRST_NAME,
    FIRST_SEEN,
    ChainBook,
)
from personos.identity.draft import MemoryDraftStore
from personos.identity.screenplay import CastDecl, ClipLine, ClipScript
from personos.identity.types import CastEvidence, FacePick

S = "sess1"


def _script(casts, cast_map, lines=()):
    sc = ClipScript(casts=list(casts), lines=list(lines))
    sc.cast_map = dict(cast_map)
    return sc


def _face(q, dim=4):
    return FacePick(t=0.0, embedding=np.ones(dim), q=q)


def _book():
    return ChainBook(MemoryDraftStore("u1"))


def test_observe_first_seen_then_better_face():
    book = _book()
    sc = _script([CastDecl(local_id="P1", desc="a man")], {"P1": "S1"})
    r1 = book.observe_clip(S, 0, sc, {"S1": CastEvidence("S1", faces=[_face(0.5)])})
    ref = book.store.chain_ref(S, "S1")
    assert r1[ref] == [FIRST_SEEN]                          # the first sighting reports first_seen only
    chain = book.store.get_chain(ref)
    assert chain["best_face_q"] == 0.5 and chain["presence"] == [0] and chain["desc_text"] == "a man"
    # A better face triggers better_face, and first_seen is no longer reported.
    r2 = book.observe_clip(S, 1, sc, {"S1": CastEvidence("S1", faces=[_face(0.8)])})
    assert r2[ref] == [BETTER_FACE]
    assert book.store.get_chain(ref)["best_face_q"] == 0.8 and book.store.get_chain(ref)["presence"] == [0, 1]
    # A worse face triggers no refresh at all.
    r3 = book.observe_clip(S, 2, sc, {"S1": CastEvidence("S1", faces=[_face(0.3)])})
    assert ref not in r3


def test_observe_first_name_trigger():
    book = _book()
    # A first sighting that already carries an introduced name still reports only first_seen, because first_seen
    # already covers the first evidence and first_name is not reported alongside it.
    sc = _script([CastDecl(local_id="P1", name="Bob", name_evidence="introduction", desc="d")],
                 {"P1": "S1"})
    r = book.observe_clip(S, 0, sc, {"S1": CastEvidence("S1")})
    ref = book.store.chain_ref(S, "S1")
    assert r[ref] == [FIRST_SEEN] and book.store.get_chain(ref)["named"] == 1
    # FIRST_NAME only fires when a name arrives for the first time on a previously unnamed chain, so build a chain
    # that starts out nameless.
    sc2 = _script([CastDecl(local_id="P2", desc="d2")], {"P2": "S2"})
    book.observe_clip(S, 0, sc2, {"S2": CastEvidence("S2")})
    sc2b = _script([CastDecl(local_id="P2", name="Al", name_evidence="explicit_dialogue", desc="d2")],
                   {"P2": "S2"})
    r2 = book.observe_clip(S, 1, sc2b, {"S2": CastEvidence("S2")})
    assert FIRST_NAME in r2[book.store.chain_ref(S, "S2")]


def test_apply_verdict_copresent_merge_rejected():
    book = _book()
    # The two chains are co-present (their presence sets intersect), so trying to merge S1 into S2 must be rejected.
    for c in ("S1", "S2"):
        book.store.ensure_chain(S, c)
        book.store.update_chain(book.store.chain_ref(S, c), presence=[0])
    r1, r2 = book.store.chain_ref(S, "S1"), book.store.chain_ref(S, "S2")
    out = book.apply_verdict(r1, r2, session_id=S, clip_index=0,
                             reason="collision_rearbitration", issues=[])
    assert out == book.store.canonical_chain(r1)            # the hypothesis is unchanged and no merge happened
    assert book.store.canonical_chain(r1) != book.store.canonical_chain(r2)


def test_apply_verdict_hypothesis_and_vote_majority():
    book = _book()
    book.store.ensure_chain(S, "S1")
    ref = book.store.chain_ref(S, "S1")
    # Three evidence-triggered evaluations all land on char_a, which is a strong majority.
    for i in range(3):
        book.apply_verdict(ref, "char_a", session_id=S, clip_index=i,
                           reason=FIRST_SEEN if i == 0 else BETTER_FACE, issues=[])
    assert book.store.get_chain(ref)["hypothesis"] == "char_a"
    summary = book.vote_summary(ref)
    assert summary["counts"]["char_a"] == 3 and summary["total"] == 3
    assert book.strong_majority(ref) == "char_a"


def test_pending_cards_excludes_copresent():
    book = _book()
    for c in ("S1", "S2", "S3"):
        book.store.ensure_chain(S, c)
    # S1 and S2 are co-present; S3 stands alone.
    book.store.update_chain(book.store.chain_ref(S, "S1"), presence=[0])
    book.store.update_chain(book.store.chain_ref(S, "S2"), presence=[0])
    book.store.update_chain(book.store.chain_ref(S, "S3"), presence=[1])
    cards = book.pending_cards(S, for_chain=book.store.chain_ref(S, "S1"))
    ids = {c.character_id for c in cards}
    assert book.store.chain_ref(S, "S2") not in ids         # excluded because it is co-present
    assert book.store.chain_ref(S, "S3") in ids
    assert book.store.chain_ref(S, "S1") not in ids         # the chain itself is excluded
