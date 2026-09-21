"""Unit tests for the profile trigger's annealing schedule and the single-user orchestration:
anneal_step and should_consolidate are pure logic, while run_user_consolidation runs against
real stores with a FakeLLM."""

from __future__ import annotations

import json
import time

from personos.models import MemCell, MemoryAtom, now
from personos.online.profile_consolidate import (
    anneal_step,
    run_user_consolidation,
    should_consolidate,
)
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.profile_store import ProfileStore

from .fakes import FakeLLM


def test_anneal_step_ramps_and_caps():
    assert [anneal_step(v) for v in range(8)] == [1, 2, 2, 3, 4, 5, 5, 5]   # starts cold at 1 and rises to a cap of 5


def test_should_consolidate_either_condition():
    # annealing: at version 0 the step is 1, so a single new cell already triggers
    assert should_consolidate(n_new=1, ep_chars=10, version_count=0, ep_chars_trigger=15000)
    # at version 1 the step is 2, so one cell is not enough and neither is the character count
    assert not should_consolidate(n_new=1, ep_chars=10, version_count=1, ep_chars_trigger=15000)
    # the character count reaches its bound, which triggers even though the cell count is below the step
    assert should_consolidate(n_new=1, ep_chars=15000, version_count=1, ep_chars_trigger=15000)


def _seed_cell(db, user_id, episode, atom_text):
    cs, ats = CellStore(db, user_id), AtomStore(db, user_id)
    c = MemCell(session_id="s", topic="t", episode=episode, t_start=now())
    cs.upsert(c)
    ats.upsert(MemoryAtom(memcell_id=c.id, text=atom_text))
    return cs, ats, c


def test_run_user_consolidation_first_version(db):
    cs, ats, c = _seed_cell(db, "u_cons_1", "user likes ramen", "user likes ramen (2026-09-14)")
    ps = ProfileStore(db, "u_cons_1")
    patch = json.dumps({"traits": {"interests": {"text": "likes ramen", "status": "confirmed",
                                                 "sources": ["c1"]}}})
    v = run_user_consolidation(FakeLLM([patch]), cells_store=cs, atoms_store=ats,
                               profile_store=ps, today=now().date())
    assert v == 1
    cur = ps.current()
    assert cur.up_to_cell_id == c.id                       # the cursor advances to the last cell
    t = cur.profile.traits["interests"]
    assert t.text == "likes ramen" and t.sources == [c.id]  # the short label c1 is filled back in with the real cell_id


def test_run_user_consolidation_no_new_cells_returns_none(db):
    cs, ats, c = _seed_cell(db, "u_cons_2", "x", "y (2026-09-14)")
    ps = ProfileStore(db, "u_cons_2")
    patch = json.dumps({"traits": {"identity": {"text": "engineer", "status": "confirmed",
                                               "sources": ["c1"]}}})
    assert run_user_consolidation(FakeLLM([patch]), cells_store=cs, atoms_store=ats,
                                  profile_store=ps, today=now().date()) == 1
    # run again: the cursor already sits at c and there are no new cells, so it returns None.
    # The call is idempotent and does not publish a second version.
    assert run_user_consolidation(FakeLLM([patch]), cells_store=cs, atoms_store=ats,
                                  profile_store=ps, today=now().date()) is None


def test_single_flight_lock_serializes_two_triggers():
    """Two concurrent triggers, standing in for two service instances or two segment closes,
    share one lock: only one wins and does the real work, the other returns immediately
    instead of consolidating a second time.

    This covers lock semantics only; it does not touch the pinned database connection, since
    database-level concurrency is exercised by the end-to-end scripts against the real path.
    What is checked here is the single-flight skeleton of runtime._run_user_profile: if
    try_acquire fails, return.
    """
    import threading

    from personos.storage.session_lock import MemorySessionLock

    lock = MemorySessionLock()
    ran = {"n": 0}
    start = threading.Barrier(2)

    def worker():
        start.wait()
        token = lock.try_acquire("u", "profile")
        if not token:
            return                                     # lost the race -> leave it to the other worker, which heals idempotently
        try:
            ran["n"] += 1
            time.sleep(0.05)                           # widen the critical section window
        finally:
            lock.release("u", "profile", token)

    ts = [threading.Thread(target=worker) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(2)
    assert ran["n"] == 1                               # single-flight: only one consolidation runs at a time
