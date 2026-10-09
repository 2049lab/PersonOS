"""Unit tests for commit_session against the shared integration database (the db fixture wraps each test in a
rollback scope), plus MemoryDraft and MockOmni.

Covers: registering a NEW character and updating its prototype cloud, matching an existing character, the
SW -> wearer pointer, falling back to the chain hypothesis when the final adjudication call fails, and degrading
one chain of a co-present collision group.
"""

from __future__ import annotations

import numpy as np

from personos.identity.backends.mock import MockOmni
from personos.identity.chains import ChainBook
from personos.identity.cloud import CloudEngine
from personos.identity.commit import commit_session
from personos.identity.draft import MemoryDraftStore
from personos.identity.registry import AnchorRegistry
from personos.identity.store import CharacterStore
from personos.identity.types import CastEvidence, FacePick, VoiceSample

U = "vtest_commit"
S = "sess1"


class _FailOmni:
    def chat(self, *a, **k):
        raise RuntimeError("omni down")


def _e(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    v[i] = 1.0
    return v


def _setup(db):
    store = CharacterStore(db, U)
    cloud = CloudEngine(store, template_cap=8)
    draft = MemoryDraftStore(U)
    book = ChainBook(draft, media_store=None)
    registry = AnchorRegistry(store, cloud, draft, media_store=None)
    return store, cloud, draft, book, registry


def _seed_chain(draft, cast, *, desc="a person", presence=(0,), face_i=0, q=0.8, best_face_q=0.8):
    """Create a pending chain and stage one face evidence carrying an embedding. media is None, so the object-store
    key stays empty, which does not affect prototype learning.
    """
    draft.ensure_chain(S, cast)
    ref = draft.chain_ref(S, cast)
    draft.update_chain(ref, desc_text=desc, presence=list(presence), best_face_q=best_face_q)
    ev = CastEvidence(cast, faces=[FacePick(t=0.0, embedding=_e(face_i), q=q, crop_b64="")])
    draft.stage_evidence(ref, session_id=S, clip_index=0, evidence=ev, media_store=None)
    return ref


def test_commit_new_registration_and_learn(db):
    store, cloud, draft, book, registry = _setup(db)
    ref = _seed_chain(draft, "S1", face_i=1)
    omni = MockOmni("BIND|S1|NEW\nEND")                     # empty library, so NEW is the only possible verdict
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert len(report["registered"]) == 1
    cid = report["by_chain"][ref]
    assert cid == report["registered"][0]
    assert store.get_character(cid) is not None
    assert len(store.active_assets(cid, "face")) == 1      # the staged asset was persisted
    mean, tau, n = store.load_prototype(cid, "face")
    assert mean is not None and n == 1                      # the prototype cloud was updated
    assert draft.get_chain(ref)["status"] == "committed"


def test_commit_match_existing(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")             # character a already exists in the library
    ref = _seed_chain(draft, "S1", face_i=0)
    omni = MockOmni("BIND|S1|%s\nEND" % a)
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert report["by_chain"][ref] == a and report["registered"] == []
    assert len(store.active_assets(a, "face")) == 1         # the asset was appended to the existing character


def test_commit_wearer_pointer(db):
    store, cloud, draft, book, registry = _setup(db)
    draft.ensure_chain(S, "SW")
    ref = draft.chain_ref(S, "SW")
    draft.update_chain(ref, presence=[0])                   # SW has no face, only a voiceprint; this only checks the wearer pointer
    omni = MockOmni("BIND|SW|NEW\nEND")
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    wearer = report["wearer"]
    assert wearer and store.get_character(wearer)["is_wearer"] is True
    assert store.wearer_character() == wearer
    assert store.session_wearer(S) == wearer


def _seed_wearer(draft, session, voice_i, voice=None):
    """A pending SW chain with one staged voice sample (the wearer has no face)."""
    draft.ensure_chain(session, "SW")
    ref = draft.chain_ref(session, "SW")
    draft.update_chain(ref, presence=[0])
    draft.stage_evidence(ref, session_id=session, clip_index=0, media_store=None,
                         evidence=CastEvidence("SW", voices=[VoiceSample(t0=0.0, t1=2.0, embedding=_e(voice_i) if voice is None else voice, q=0.8)]))
    return ref


def test_second_session_wearer_binds_to_existing_wearer(db):
    """The wearer is the device: a new session's SW chain must land on the existing wearer
    even when the model votes NEW, and its voice samples are added to that character."""
    store, cloud, draft, book, registry = _setup(db)
    _seed_wearer(draft, S, voice_i=2)
    r1 = commit_session(store, cloud, registry, book, session_id=S, omni=MockOmni("END"))
    wearer = r1["wearer"]
    assert r1["registered"] == [wearer] and len(store.active_assets(wearer, "voice")) == 1

    ref2 = _seed_wearer(draft, "sess2", voice_i=3, voice=_e(2) * 0.9 + _e(3) * 0.4)      # the same voice, slightly off
    r2 = commit_session(store, cloud, registry, book, session_id="sess2", omni=MockOmni("BIND|SW|NEW\nEND"))
    assert r2["registered"] == [] and r2["by_chain"][ref2] == wearer and r2["wearer"] == wearer
    assert len(store.active_assets(wearer, "voice")) == 2
    assert [c["id"] for c in store.list_active_characters(include_wearer=True)] == [wearer]
    assert draft.get_chain(ref2)["status"] == "committed" and store.session_wearer("sess2") == wearer


def test_wearer_voice_from_a_different_speaker_is_not_enrolled(db):
    """A human's voice mislabelled as SW must not be learned into the wearer; the chain still binds to it."""
    store, cloud, draft, book, registry = _setup(db)
    _seed_wearer(draft, S, voice_i=2)
    wearer = commit_session(store, cloud, registry, book, session_id=S, omni=MockOmni("END"))["wearer"]
    ref2 = _seed_wearer(draft, "sess2", voice_i=3)                      # orthogonal to the wearer's voice: cos 0
    r2 = commit_session(store, cloud, registry, book, session_id="sess2", omni=MockOmni("END"))
    assert r2["by_chain"][ref2] == wearer and r2["registered"] == []
    assert len(store.active_assets(wearer, "voice")) == 1               # nothing added
    assert draft.get_chain(ref2)["status"] == "committed"


def test_wearer_is_never_sent_to_the_model(db):
    store, cloud, draft, book, registry = _setup(db)
    _seed_wearer(draft, S, voice_i=2)
    _seed_chain(draft, "S1", face_i=1)
    omni = MockOmni("BIND|S1|NEW\nEND")
    commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert "QUERY SW" not in omni.calls[0]["prompt"] and "QUERY S1" in omni.calls[0]["prompt"]


def test_wearer_only_session_makes_no_model_call(db):
    store, cloud, draft, book, registry = _setup(db)
    _seed_wearer(draft, S, voice_i=2)
    omni = MockOmni("END")
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert omni.calls == [] and report["wearer"] and not report["fallbacks"]


def test_legacy_multiple_wearers_bind_to_oldest(db):
    store, cloud, draft, book, registry = _setup(db)
    old = store.create_character(session_id="s0", is_wearer=True)
    new = store.create_character(session_id="s1", is_wearer=True)
    ref = _seed_wearer(draft, S, voice_i=2)
    report = commit_session(store, cloud, registry, book, session_id=S, omni=MockOmni("END"))
    assert old < new and report["by_chain"][ref] == old and report["registered"] == []
    assert len(store.active_assets(old, "voice")) == 1 and not store.active_assets(new, "voice")


def test_commit_call_failure_falls_back_to_hypothesis(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")
    ref = _seed_chain(draft, "S1")
    draft.update_chain(ref, hypothesis=a, hypo_method="omni_rerank")   # a hypothesis is already on the chain
    report = commit_session(store, cloud, registry, book, session_id=S, omni=_FailOmni())
    assert report["fallbacks"][ref] == "hypothesis"
    assert report["by_chain"][ref] == a                     # falls back to the chain hypothesis a


def test_name_merge_guardrails():
    """Name-based fallback merging: chains that share a name, never appear in the same clip, and are both NEW get
    merged into the longest chain. Chains that are co-present, or already bound to a registered character, are left
    alone by the guards. Pure draft state, no network.
    """
    from personos.identity.commit import _merge_same_name_chains
    draft = MemoryDraftStore(U)
    book = ChainBook(draft, media_store=None)

    def _chain(cast, name, presence, hyp="NEW"):
        draft.ensure_chain(S, cast)
        ref = draft.chain_ref(S, cast)
        draft.update_chain(ref, presence=list(presence), hypothesis=hyp)
        draft.save_roster_entry(S, cast, character_id=None, card={"name": name})
        return ref

    # Bob is split across two chains that never co-occur (disjoint presence), so they should merge and keep S1,
    # which has the longer presence.
    r1 = _chain("S1", "Bob", [0, 1, 2])
    r3 = _chain("S3", "Bob", [5])
    # Two Alice chains that are co-present (overlapping presence), so the second guard blocks the merge.
    r2 = _chain("S2", "Alice", [0])
    r4 = _chain("S4", "Alice", [0])
    # Two Mike chains that never co-occur but are already bound to different registered characters, so the first
    # guard blocks the merge -- the model deliberately told two real Mikes apart.
    r5 = _chain("S5", "Mike", [1], hyp="char_m1")
    r6 = _chain("S6", "Mike", [7], hyp="char_m2")

    report: dict = {}
    _merge_same_name_chains(book, S, report)

    assert draft.canonical_chain(r3) == r1                 # Bob merged into S1, the longest chain
    assert draft.canonical_chain(r2) != draft.canonical_chain(r4)   # Alice not merged: co-present
    assert draft.canonical_chain(r5) != draft.canonical_chain(r6)   # Mike not merged: already bound
    merges = report.get("name_merges", [])
    assert len(merges) == 1 and merges[0]["name"] == "Bob" and merges[0]["kept"] == r1


def test_commit_copresent_collision_degrades_one(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")
    # S1 and S2 are co-present (presence intersection [0]) and the final adjudication binds both to a, so one of
    # them must be degraded.
    r1 = _seed_chain(draft, "S1", presence=(0,), best_face_q=0.9, face_i=0)
    r2 = _seed_chain(draft, "S2", presence=(0,), best_face_q=0.5, face_i=1)
    omni = MockOmni("BIND|S1|%s\nBIND|S2|%s\nEND" % (a, a))
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    finals = {report["by_chain"][r1], report["by_chain"][r2]}
    assert a in finals and len(finals) == 2                 # one keeps a, the other is degraded into a new character
    assert report["by_chain"][r1] == a                      # the higher best_face_q (S1) keeps a
    assert len(report["registered"]) == 1                   # the degraded chain registered a new character
