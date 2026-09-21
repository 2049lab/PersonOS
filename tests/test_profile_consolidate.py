"""Unit tests for consolidate (clean structure + short/long id mapping): success / success
after a bounce / keeping the old version / retry on non-JSON / no-op when there are no cells /
bounce on exceeding the limit / short labels resolved back to real cell_ids."""

from __future__ import annotations

import json
from datetime import date, datetime

from personos.models import MemCell, MemoryAtom
from personos.online.profile_consolidate import consolidate
from personos.storage.profile_store import UserProfile

from .fakes import FakeLLM

TODAY = date(2026, 9, 14)


def _cells():
    c = MemCell(id="cell_LONG_ULID_1", session_id="s", topic="lunch",
                episode="user had ramen and dislikes cilantro",
                t_start=datetime(2026, 9, 14, 12, 0))
    atoms = {"cell_LONG_ULID_1": [MemoryAtom(text="user had ramen on 2026-09-14"),
                                  MemoryAtom(text="user dislikes cilantro")]}
    return [c], atoms


def _patch(**kw) -> str:
    return json.dumps(kw)


def test_consolidate_success_and_source_remap():
    cells, atoms = _cells()
    patch = _patch(
        traits={"communication_style": {"text": "direct and terse", "status": "inferred",
                                        "sources": ["c1"]}},
        facts={"add": [{"band": "today", "text": "2026-09-14 had ramen, dislikes cilantro",
                        "sources": ["c1"]}]},
    )
    p = consolidate(FakeLLM([patch]), current=None, cells=cells, atoms_by_cell=atoms, today=TODAY)
    assert p is not None
    t = p.traits["communication_style"]
    assert t.text == "direct and terse" and t.last_confirmed == "2026-09-14"
    # The short label c1 is resolved back to the real long id.
    assert t.sources == ["cell_LONG_ULID_1"]
    assert [f.text for f in p.facts["today"]] == ["2026-09-14 had ramen, dislikes cilantro"]
    assert p.facts["today"][0].sources == ["cell_LONG_ULID_1"]


def test_consolidate_bounces_then_succeeds():
    """The first version has a bad status and gets bounced; the second is legal and succeeds."""
    cells, atoms = _cells()
    bad = _patch(traits={"goals": {"text": "x", "status": "maybe", "sources": ["c1"]}})
    good = _patch(traits={"goals": {"text": "ship the memory service", "status": "confirmed",
                                   "sources": ["c1"]}})
    p = consolidate(FakeLLM([bad, good]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is not None and p.traits["goals"].text == "ship the memory service"


def test_consolidate_field_invalid_keeps_old():
    """Still failing field-level validation after the retry budget is spent (a hallucinated
    source) returns None, which keeps the old version."""
    cells, atoms = _cells()
    bad = _patch(facts={"add": [{"band": "today", "text": "x", "sources": ["c_ghost"]}]})
    p = consolidate(FakeLLM([bad, bad, bad]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is None


def test_consolidate_overflow_bounces_then_backstop():
    """The LLM keeps stuffing 2 entries into today (the limit is 1), so it gets bounced. When it
    still does not converge, the engine drops the oldest as a backstop and publishes anyway
    (so the result is not None)."""
    cells, atoms = _cells()
    over = _patch(facts={"add": [{"band": "today", "text": "a", "sources": ["c1"]},
                                 {"band": "today", "text": "b", "sources": ["c1"]}]})
    p = consolidate(FakeLLM([over, over, over]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is not None and len(p.facts["today"]) == 1   # the backstop converged to the limit


def test_consolidate_non_json_then_valid():
    cells, atoms = _cells()
    good = _patch(traits={"identity": {"text": "software engineer in SG", "status": "confirmed",
                                      "sources": ["c1"]}})
    p = consolidate(FakeLLM(["这不是 JSON,只是闲聊", good]), current=None, cells=cells,
                    atoms_by_cell=atoms, today=TODAY, max_retries=2)
    assert p is not None and p.traits["identity"].text == "software engineer in SG"


def test_consolidate_no_cells_returns_none():
    p = consolidate(FakeLLM([_patch(traits={})]), current=None, cells=[], atoms_by_cell={}, today=TODAY)
    assert p is None


def test_consolidate_preserves_existing_via_patch():
    """The patch touches only one domain and everything else in the current profile survives —
    the incremental semantics, end to end."""
    cells, atoms = _cells()
    from personos.storage.profile_store import ProfileTrait
    cur = UserProfile.empty()
    cur.traits["location"] = ProfileTrait(text="Singapore", status="confirmed",
                                          last_confirmed="2026-01-01", sources=["cell_0"])
    patch = _patch(traits={"occupation": {"text": "engineer", "status": "confirmed", "sources": ["c1"]}})
    p = consolidate(FakeLLM([patch]), current=cur, cells=cells, atoms_by_cell=atoms, today=TODAY)
    assert p.traits["location"].text == "Singapore"       # an unmentioned domain is preserved
    assert p.traits["occupation"].text == "engineer"


# Kept at the bottom: an import at the top of the file would be order-dependent with conftest,
# so it is referenced close to where it is used.
from .fakes import FakeLLM  # noqa: E402
