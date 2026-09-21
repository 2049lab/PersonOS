"""Unit tests for profile storage: version increment / current returns the latest / lookup by
version / per-user isolation / the cells_after cursor.

These run against a shared database; the autouse db fixture already wraps each test in a
rollback scope (an unconditional rollback at the end), which makes fixed user_ids and ids safe
and leaves no residue.
"""

from __future__ import annotations

from datetime import timedelta

from personos.models import MemCell, now
from personos.storage.cell_store import CellStore
from personos.storage.profile_store import (
    ProfileFact,
    ProfileStore,
    ProfileTrait,
    UserProfile,
)


def _profile(trait_text: str = "prefers terse feedback", fact_text: str = "ate ramen") -> UserProfile:
    p = UserProfile.empty()
    p.traits["communication_style"] = ProfileTrait(
        text=trait_text, status="confirmed", last_confirmed="2026-09-14", sources=["cell_x"]
    )
    p.traits["finance"] = None                      # an empty dimension is legal
    p.facts["today"].append(ProfileFact(
        id="f_1", text=fact_text, last_confirmed="2026-09-14", sources=["cell_y"],
    ))
    return p


def test_current_none_when_never_consolidated(db):
    ps = ProfileStore(db, user_id="u_prof_1")
    assert ps.current() is None
    assert ps.version_count() == 0


def test_save_version_monotonic_and_current_latest(db):
    ps = ProfileStore(db, user_id="u_prof_2")
    v1 = ps.save_version(_profile(trait_text="v1"), up_to_cell_id="cell_a")
    v2 = ps.save_version(_profile(trait_text="v2"), up_to_cell_id="cell_b")
    assert (v1, v2) == (1, 2)                        # monotonically increasing
    assert ps.version_count() == 2
    cur = ps.current()
    assert cur.version == 2 and cur.up_to_cell_id == "cell_b"
    assert cur.profile.traits["communication_style"].text == "v2"   # the latest version wins


def test_profile_json_round_trip(db):
    """The structure survives a full round trip: traits (including a None empty dimension) and
    facts (including f_id) come back unchanged and complete."""
    ps = ProfileStore(db, user_id="u_prof_3")
    ps.save_version(_profile(), up_to_cell_id="")
    p = ps.current().profile
    assert p.traits["finance"] is None
    t = p.traits["communication_style"]
    assert t.status == "confirmed" and t.sources == ["cell_x"]
    f = p.facts["today"][0]
    assert f.id == "f_1" and f.text and f.sources == ["cell_y"]


def test_get_version(db):
    ps = ProfileStore(db, user_id="u_prof_4")
    ps.save_version(_profile(trait_text="first"), up_to_cell_id="c1")
    ps.save_version(_profile(trait_text="second"), up_to_cell_id="c2")
    assert ps.get_version(1).profile.traits["communication_style"].text == "first"
    assert ps.get_version(2).profile.traits["communication_style"].text == "second"
    assert ps.get_version(99) is None


def test_user_isolation(db):
    """Profiles must never cross between users: after u_a publishes a version, u_b still has no
    profile at all — each sees only their own."""
    pa = ProfileStore(db, user_id="u_iso_a")
    pb = ProfileStore(db, user_id="u_iso_b")
    pa.save_version(_profile(trait_text="a's profile"), up_to_cell_id="ca")
    assert pb.current() is None                      # b cannot see a's profile
    assert pb.version_count() == 0
    assert pa.current().profile.traits["communication_style"].text == "a's profile"


def test_cells_after_cursor(db):
    """cells_after returns the cells newer than the cursor (oldest to newest); an empty cursor
    means everything, and a cursor that does not exist also means everything (re-distill)."""
    cs = CellStore(db, user_id="u_cells")
    base = now()
    ids = []
    for i in range(3):
        c = MemCell(session_id="s", topic=f"t{i}", episode=f"episode {i}",
                    t_start=base + timedelta(minutes=i), t_end=base + timedelta(minutes=i, seconds=30))
        cs.upsert(c)
        ids.append(c.id)

    assert [c.id for c in cs.cells_after("")] == ids                 # empty cursor -> everything
    assert [c.id for c in cs.cells_after(ids[0])] == ids[1:]         # the two after the cursor
    assert cs.cells_after(ids[2]) == []                              # nothing after the last one
    # Cursor does not exist -> everything (re-distill).
    assert [c.id for c in cs.cells_after("cell_nonexistent")] == ids


def test_cells_after_isolation(db):
    """cells_after only sees cells belonging to this user."""
    ca = CellStore(db, user_id="u_ca")
    cb = CellStore(db, user_id="u_cb")
    base = now()
    a = MemCell(session_id="s", topic="a", episode="a", t_start=base)
    ca.upsert(a)
    b = MemCell(session_id="s", topic="b", episode="b", t_start=base + timedelta(minutes=1))
    cb.upsert(b)
    assert [c.id for c in ca.cells_after("")] == [a.id]              # each sees only their own
    assert [c.id for c in cb.cells_after("")] == [b.id]
