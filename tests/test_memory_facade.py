"""The Memory facade: the surface a library user actually touches.

This file exists because its absence cost three bugs. The facade re-implemented
calls that the HTTP layer already made correctly, and nothing checked it:

- ``trace()`` passed a bare string where a list was expected, so it iterated the
  id **character by character** and returned 29 "missing" entries. It looked
  like it worked — truthy, non-empty — which is why a live smoke check passed.
- ``search(with_profile=True)`` passed the version wrapper where the profile
  belonged. Fine on a fresh install, then throwing the moment the background
  job published v1: a failure that arrives with usage, not with a deploy.
- ``profile()`` returned a dataclass while declaring ``dict``, which crashed the
  shipped quickstart on ``.get()``.

The common thread is that all three are *wiring*, not logic — and wiring is
exactly what type checkers and downstream tests do not catch. So these tests
call the public methods against a seeded store and assert on the shape that
comes back.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from personos.models import EvidenceRecord, MemCell, MemoryAtom
from personos.storage.profile_store import ProfileStore, ProfileTrait, UserProfile

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def seeded(db):
    """One user with a complete chain: evidence -> cell -> atom."""
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.evidence_store import EvidenceStore

    user = "facade_user"
    ev = EvidenceRecord(id="ev_facade_1", modality="text", holder="user",
                        content_inline="I am allergic to peanuts.", captured_at=NOW,
                        source={"session_id": "s1"})
    EvidenceStore(db, user).append(ev)
    cell = MemCell(id="cell_facade_1", session_id="s1", topic="allergies",
                   episode="The user said they are allergic to peanuts.",
                   t_start=NOW, t_end=NOW, evidence_refs=[{"evidence_id": ev.id}])
    CellStore(db, user).upsert(cell)
    atom = MemoryAtom(id="atom_facade_1", memcell_id=cell.id, object_type="fact",
                text="The user is allergic to peanuts.", holder="user",
                evidence_refs=[{"evidence_id": ev.id}], recorded_at=NOW)
    AtomStore(db, user).upsert_many([(atom, None)])
    return user, ev, cell, atom


@pytest.fixture
def memory(db, monkeypatch):
    """A Memory bound to the test database, with no model clients built."""
    from personos import Memory

    m = Memory.__new__(Memory)          # skip __init__: no pools, no providers
    m.db = db
    m._uctx = {}
    m._uctx_guard = __import__("threading").Lock()
    m._media_store = False
    m._media_guard = __import__("threading").Lock()
    m._seg_store = None
    m._session_lock = None
    m._state_guard = __import__("threading").Lock()
    return m


# ── trace(): the bug was an argument-order mistake, not a logic error ────

def test_trace_of_an_atom_returns_one_node_with_its_evidence(memory, seeded):
    user, ev, _cell, atom = seeded
    out = memory.trace(atom.id, user_id=user)

    assert isinstance(out, dict), "a single id must yield a single node, not a list"
    assert out.get("node") == "memory"
    assert out.get("atom_id") == atom.id
    assert not out.get("missing")
    # The whole point of provenance: the original turn is reachable from here.
    cited = str(out.get("evidence") or out)
    assert ev.id in cited


def test_trace_does_not_iterate_the_id_as_characters(memory, seeded):
    """The original defect. A bare string where a list was expected produced one
    'missing' entry per character — truthy, non-empty, and completely wrong."""
    user, _ev, _cell, atom = seeded
    out = memory.trace(atom.id, user_id=user)
    assert not isinstance(out, list), "returned a list -> the id was iterated"


def test_trace_of_evidence_returns_the_backward_chain(memory, seeded):
    user, ev, _cell, _atom = seeded
    out = memory.trace(ev.id, user_id=user)
    assert out and out.get("node") == "evidence"


def test_trace_of_an_unknown_id_is_none(memory, seeded):
    user, *_ = seeded
    assert memory.trace("atom_does_not_exist", user_id=user) is None


# ── profile(): declared dict, returned a dataclass ──────────────────────

def test_profile_is_a_dict_even_when_empty(memory, seeded):
    user, *_ = seeded
    out = memory.profile(user_id=user)

    assert isinstance(out, dict)
    assert out["exists"] is False, "no profile yet is a normal state, not None"
    assert out["traits"] == {} and out["version"] == 0


def test_profile_renders_traits_and_hides_internals(memory, db, seeded):
    user, _ev, cell, _atom = seeded
    profile = UserProfile(traits={"occupation": ProfileTrait(
        text="backend engineer", status="confirmed", last_confirmed="2026-09-21")})
    ProfileStore(db, user).save_version(profile, up_to_cell_id=cell.id)

    out = memory.profile(user_id=user)
    assert out["exists"] is True and out["version"] >= 1
    assert out["traits"]["occupation"]["text"] == "backend engineer"
    # Fact ids and the cells behind each claim are how the profile is
    # maintained, not what it means; publishing them makes them contract.
    assert "f_id" not in str(out) and cell.id not in str(out)


def test_library_and_server_render_a_profile_identically(memory, db, seeded):
    """Both go through one renderer, so they cannot drift apart."""
    from personos.online.views import profile_view

    user, _ev, cell, _atom = seeded
    ProfileStore(db, user).save_version(
        UserProfile(traits={"values": ProfileTrait(text="direct", status="inferred")}),
        up_to_cell_id=cell.id)

    assert memory.profile(user_id=user) == profile_view(ProfileStore(db, user).current())


# ── search(with_profile=True): broke only once a profile existed ─────────

def test_profile_strings_render_once_a_profile_exists(memory, db, seeded):
    """The nastiest of the three: correct on a fresh install, then failing as
    soon as the background job published a version."""
    user, _ev, cell, _atom = seeded
    ProfileStore(db, user).save_version(
        UserProfile(traits={"occupation": ProfileTrait(text="backend engineer",
                                                status="confirmed")}),
        up_to_cell_id=cell.id)

    full, traits = memory._profile_strings(user)
    assert "backend engineer" in full
    assert "backend engineer" in traits
    assert len(traits) <= len(full), "the traits-only view must be the narrower one"


def test_profile_strings_are_empty_before_any_profile(memory, seeded):
    user, *_ = seeded
    assert memory._profile_strings(user) == ("", "")


# ── input handling ──────────────────────────────────────────────────────

@pytest.mark.parametrize("make,label", [
    (lambda p: p.read_bytes(), "raw bytes"),
    (lambda p: str(p), "a path"),
    (lambda p: __import__("base64").b64encode(p.read_bytes()).decode(), "bare base64"),
    (lambda p: "data:image/png;base64," + __import__("base64").b64encode(
        p.read_bytes()).decode(), "a data URL"),
])
def test_images_are_accepted_in_every_form_people_pass_them(tmp_path, make, label):
    """The HTTP API takes base64 for this field, so someone will pass base64 to
    the library too. Treating it as a filename fails confusingly."""
    from personos.memory import _as_image_bytes

    png = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 32
    path = tmp_path / "shot.png"
    path.write_bytes(png)

    assert _as_image_bytes(make(path)) == png, label


def test_an_unreadable_image_says_so():
    from personos.memory import _as_image_bytes

    with pytest.raises(ValueError, match="neither an existing path nor valid base64"):
        _as_image_bytes("this is not base64 !!!")


def test_an_image_recall_without_identity_backend_degrades_instead_of_raising(
        memory, seeded, monkeypatch):
    """mllm configured but identity backend at its default (none): building the
    face-matching deps raises. Recall is a read path — it must degrade to text
    with a warning, never fail the question. It did fail (RuntimeError), which
    is how this test came to exist."""
    from types import SimpleNamespace

    import personos.online.recall_flow as rf

    user, *_ = seeded
    memory.llm = type("L", (), {"available": True})()
    memory.embedder = type("E", (), {"available": True})()
    memory.mllm = type("M", (), {"available": True})()
    memory.reranker = None
    monkeypatch.setattr(memory, "visual_deps",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("backend=none")))
    captured = {}
    monkeypatch.setattr(rf, "run_recall",
                        lambda *a, **kw: captured.update(kw) or SimpleNamespace(
                            warnings=[], image=kw.get("image")))

    out = memory.search("who is this?", user_id=user, image=b"\xff\xd8fake")
    assert captured["image"] is None, "the image must be dropped, not crash the recall"
    assert any("face matching" in w for w in out.warnings)


def test_video_and_turns_cannot_share_one_call(memory, seeded):
    """A clip is a recording to watch, a turn is text to append to a segment.
    Mixing them in one call would make the batch non-atomic."""
    user, *_ = seeded
    memory.llm = type("L", (), {"available": True})()
    memory.embedder = type("E", (), {"available": True})()
    with pytest.raises(ValueError, match="not both"):
        memory.add([{"role": "user", "content": "look"}, {"video": "/tmp/x.mp4"}],
                   user_id=user, session_id="s1")
