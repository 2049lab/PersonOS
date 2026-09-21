"""Backfill tests for build_chains (§4.3): clear-and-replay, idempotence (a re-run creates no
duplicates), and correct statistics.

FakeLLM dequeues one chain-assignment JSON per cell; the replay order is deterministic
(cells by time, atoms by occurrence time within a cell), so the results are assertable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from scripts.build_chains import rebuild_user

from .fakes import FakeLLM
from .test_deep_recall import Env
from .test_retrieval import _v


def _scene(db):
    """Two cells, three atoms: c1 (08-10, gym + coach) and c2 (08-24, relocation).

    FakeLLM returns two responses: the first cell (no candidates) splits into two new chains;
    the second cell appends to candidate c1, the location chain."""
    env = Env(db)
    env.add_cell(topic="瑜伽一", episode="叙事一",
                 t_start=datetime(2026, 8, 10, tzinfo=timezone.utc), atoms=[
                     {"text": "馆在健身房", "vec": _v(1, 0, 0, 0),
                      "when": datetime(2026, 8, 10, 1, tzinfo=timezone.utc)},
                     {"text": "教练叫Mira", "vec": _v(0, 1, 0, 0),
                      "when": datetime(2026, 8, 10, 2, tzinfo=timezone.utc)},
                 ])
    env.add_cell(topic="瑜伽二", episode="叙事二",
                 t_start=datetime(2026, 8, 24, tzinfo=timezone.utc), atoms=[
                     {"text": "馆搬到了MBS", "vec": _v(0.9, 0.1, 0, 0),
                      "when": datetime(2026, 8, 24, tzinfo=timezone.utc)},
                 ])
    responses = [
        '{"assignments":[{"chain":"new","title":"地点链","atoms":[1]},'
        '{"chain":"new","title":"教练链","atoms":[2]}]}',
        # c1 is the pre-filtered candidate (the location chain); append to it.
        '{"assignments":[{"chain":"c1","atoms":[1]}]}',
    ]
    return env, responses


def test_rebuild_clears_and_replays_in_time_order(db):
    """Backfill: first clear this user's chains, then replay W2.5 in cell time order (a chain
    may be appended to across cells); the statistics must add up."""
    env, responses = _scene(db)
    stats = rebuild_user(db, FakeLLM(responses), "")

    assert stats["cleared_chains"] == 0                       # there were no chains to begin with
    assert stats["rebuilt_assigned"] == 3                     # all three atoms got assigned
    assert stats["n_chains"] == 2 and stats["chained_atoms"] == 3
    assert stats["free_rate"] == 0.0
    # The relocation atom was appended to the location chain.
    assert (2, "地点链") in stats["top_chains"]
    assert (1, "教练链") in stats["top_chains"]


def test_rebuild_is_idempotent(db):
    """Idempotence: a re-run clears before rebuilding, so the chain count and the number of
    chained atoms stay the same with no duplicate chain rows; the atoms themselves are untouched."""
    env, responses = _scene(db)
    s1 = rebuild_user(db, FakeLLM(responses), "")
    # Re-run with a fresh LLM queue holding the same responses.
    s2 = rebuild_user(db, FakeLLM(list(responses)), "")

    assert s2["cleared_chains"] == s1["n_chains"]             # the previous round's chains were cleared
    assert s2["n_chains"] == s1["n_chains"]                   # same count after the rebuild, no duplicates
    assert s2["chained_atoms"] == s1["chained_atoms"] == 3
    assert s2["n_atoms"] == s1["n_atoms"]                     # atoms are not re-extracted
