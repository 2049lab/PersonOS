"""Unit tests for bridge C, video to memory: draft screenplay lines plus by_chain ownership
become evidence attributed to people, then build_cell.

Uses FakeLLM (two stages, episode then atoms) against the real shared database (the db fixture
rolls everything back). Verifies:
- Line evidence carries holder = the display name (Bob / user / env) and the right
  source.character_id;
- The raw_clip original media evidence is persisted (modality=video, no content_inline);
- build_cell produces a memcell and atoms, atom.holder keeps the display name, and
  evidence_refs link back to the line evidence;
- After "screenplay line to text evidence", everything goes through the existing build_cell
  with zero difference from plain text.
"""

from __future__ import annotations

from personos.identity.draft import MemoryDraftStore
from personos.identity.store import CharacterStore
from personos.online.video_memory import flush_session_to_memory
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder, FakeLLM

U = "vtest_vmem"
S = "s1"

_EPISODE = '{"topic":"hiking chat","episode":"user and Bob chat about hiking in the living room","domains":[]}'
_ATOMS = ('{"atoms":[{"text":"Bob loves hiking","holder":"Bob","object_type":"claim",'
          '"kind":"K06","quote":"I love hiking"}]}')


def test_flush_session_to_memory(db):
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    # Seed the store: Bob (named) plus a wearer record.
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "SW"):
        draft.ensure_chain(S, cast)
    draft.stage_lines(S, 0, [
        (0.0, 1.0, "S1", "speech", "I love hiking"),
        (1.0, 2.0, "SW", "speech", "Noted"),
        (2.0, 3.0, "ENV", "environment", "a bright living room with a green sofa"),
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "SW"): wearer}

    cb = flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(),
        clip_keys={0: "oss/clip0.mp4"})

    # Evidence persisted: 1 raw_clip plus 3 lines.
    recs = ev.by_session(S)
    by_kind = {(r.source or {}).get("kind"): r for r in recs}
    raw = by_kind["raw_clip"]
    assert raw.modality == "video" and raw.content_ref == "oss/clip0.mp4" and not raw.content_inline

    speech = [r for r in recs if (r.source or {}).get("kind") == "speech"]
    s1 = next(r for r in speech if r.source["cast_id"] == "S1")
    sw = next(r for r in speech if r.source["cast_id"] == "SW")
    assert s1.holder == "Bob" and s1.source["character_id"] == bob      # display name plus the exact id
    assert sw.holder == "user" and sw.source["character_id"] == wearer  # SW maps to user
    env = next(r for r in recs if (r.source or {}).get("kind") == "environment")
    assert env.holder == "env"
    # Every line of evidence points back at the raw_clip and carries modality=video.
    assert s1.source["raw_evidence_id"] == raw.id and s1.modality == "video"

    # What build_cell produced: a memcell, atoms keeping the display name, and links back to
    # the line evidence.
    assert cb is not None and cb.cell is not None
    assert len(cb.atoms) == 1 and cb.atoms[0].holder == "Bob"
    assert cb.atoms[0].evidence_refs and cb.atoms[0].evidence_refs[0].evidence_id == s1.id


def test_flush_no_lines_returns_none(db):
    store = CharacterStore(db, U)
    draft = MemoryDraftStore(U)
    out = flush_session_to_memory(
        draft, {}, session_id="empty", char_store=store,
        evidence_store=EvidenceStore(db, U), cell_store=CellStore(db, U),
        atom_store=AtomStore(db, U), chain_store=ChainStore(db, U),
        llm=FakeLLM([]), embedder=FakeEmbedder())
    assert out is None


def test_unnamed_gets_stable_handle(db):
    """An unnamed person gets a stable short handle of the form "person #N"."""
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    anon = store.create_character(session_id=S)          # no name
    draft = MemoryDraftStore(U)
    draft.ensure_chain(S, "S1")
    draft.stage_lines(S, 0, [(0.0, 1.0, "S1", "speech", "hello there")])
    by_chain = {draft.chain_ref(S, "S1"): anon}
    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, '{"atoms":[]}']), embedder=FakeEmbedder(),
        clip_keys={0: "oss/c.mp4"})
    line = next(r for r in ev.by_session(S) if (r.source or {}).get("kind") == "speech")
    assert line.holder == "人物#1" and line.source["character_id"] == anon


def test_rewrite_ids_word_boundary_and_longest_first():
    """Id rewriting: only whole tokens are touched, longer keys win, and ids absent from the
    mapping are left exactly as they are."""
    from personos.identity.screenplay import rewrite_ids

    m = {"P1": "Alice", "P12": "Bob", "SW": "user"}
    assert rewrite_ids("P1 enters; P12 sits; SW films", m) == "Alice enters; Bob sits; user films"
    assert rewrite_ids("P12 is not P1", m) == "Bob is not Alice"        # a longer key is not cut in half by P1
    assert rewrite_ids("SPAM P1X nothing", m) == "SPAM P1X nothing"     # word boundaries: P1X and SPAM are untouched
    assert rewrite_ids("P3 unknown", m) == "P3 unknown"                 # not in the table, so left as is
    assert rewrite_ids("", m) == "" and rewrite_ids("x", {}) == "x"


def test_flush_rewrites_ids_in_text_and_marks_actions(db):
    """Cast ids inside the line text become display names, action lines get a marker, and
    unnamed people are numbered by ORDER OF APPEARANCE, reproducibly.

    Regression for a real problem: episodes contained things like "P1 enters holding an orange
    basketball" and "where P2 is seated". The ownership rewrite only touched holder, so bare
    ids in the text leaked all the way through into episodes and atoms, and the same person
    ended up with three names (holder called them person #1, the text called them P1, and
    other people called them Alice).

    Separately, action lines and speech lines rendered into the same shape ("holder: text"),
    which made the episode LLM read the action as something that person said.
    """
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    anon = store.create_character(session_id=S)          # unnamed: should get person #N
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "S2", "SW"):
        draft.ensure_chain(S, cast)
    draft.stage_lines(S, 0, [
        # S2 (unnamed) appears first, so it should become person #1; every S1/S2/SW in the text
        # must be replaced.
        (0.0, 1.0, "S2", "action", "S2 enters holding a basketball while SW films"),
        (1.0, 2.0, "S1", "speech", "S2, you are so stinky"),
        (2.0, 3.0, "ENV", "environment", "S1 is seated at the table"),
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "S2"): anon,
                draft.chain_ref(S, "SW"): wearer}

    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(), clip_keys={})

    texts = {(r.source or {}).get("cast_id"): (r.content_inline or "")
             for r in ev.by_session(S) if (r.source or {}).get("kind") != "raw_clip"}
    assert "S1" not in texts["S2"] and "S2" not in texts["S2"] and "SW" not in texts["S2"], \
        f"bare ids still left in the text: {texts['S2']!r}"
    assert texts["S2"] == "(action) 人物#1 enters holding a basketball while user films", texts["S2"]
    # S2 appeared first, so it is person #1.
    assert texts["S1"] == "人物#1, you are so stinky", texts["S1"]
    # An env line gets no action marker.
    assert texts["ENV"] == "Bob is seated at the table", texts["ENV"]


def test_anon_person_gets_one_intro_line_with_description(db):
    """An unnamed person gets ONE SEPARATE LINE describing their appearance, placed before the
    dialogue, and only once per person; people who have a name get no such line.

    Why this matters: without it, "person #1" in the memory is a hollow number — an episode or
    atom reading it has no idea who that is, so it can neither judge whether this is the same
    person across sessions nor say anything better than the bare number when answering.
    """
    store = CharacterStore(db, U)
    ev, cells, atoms, chains = (EvidenceStore(db, U), CellStore(db, U),
                                AtomStore(db, U), ChainStore(db, U))
    bob = store.create_character(session_id=S)
    store.add_name_claim(bob, "Bob", "explicit_dialogue")
    anon = store.create_character(session_id=S)
    wearer = store.create_character(session_id=S, is_wearer=True)
    store.mark_wearer(wearer)
    store.set_session_wearer(S, wearer)

    draft = MemoryDraftStore(U)
    for cast in ("S1", "S2", "SW"):
        draft.ensure_chain(S, cast)
    # The appearance description carried on the chain (screenplay cast.desc, written into
    # desc_text by chains.observe_clip).
    draft.update_chain(draft.chain_ref(S, "S2"),
                       desc_text="a woman in a pink dress with a ponytail")
    draft.update_chain(draft.chain_ref(S, "S1"), desc_text="a man in a grey hoodie")
    draft.stage_lines(S, 0, [
        (0.0, 1.0, "S2", "speech", "You ruined it"),
        (1.0, 2.0, "S1", "speech", "Sorry"),
        (2.0, 3.0, "S2", "speech", "Again!"),          # the same person again: no second intro line
    ])
    by_chain = {draft.chain_ref(S, "S1"): bob, draft.chain_ref(S, "S2"): anon,
                draft.chain_ref(S, "SW"): wearer}

    flush_session_to_memory(
        draft, by_chain, session_id=S, char_store=store, evidence_store=ev,
        cell_store=cells, atom_store=atoms, chain_store=chains,
        llm=FakeLLM([_EPISODE, _ATOMS]), embedder=FakeEmbedder(), clip_keys={})

    intros = [r for r in ev.by_session(S) if (r.source or {}).get("kind") == "cast_intro"]
    assert len(intros) == 1, \
        f"an unnamed person should get exactly one intro line (Bob has a name, so none): {[r.content_inline for r in intros]}"
    assert intros[0].content_inline == "人物#1 is a woman in a pink dress with a ponytail"
    assert intros[0].holder == "env", \
        "the description is narration and must not be attributed to the person, or it reads as something they said"
    assert (intros[0].source or {}).get("character_id") == anon
