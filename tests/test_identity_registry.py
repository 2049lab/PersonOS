"""⑥a/⑥b 单测:inspect 同框约束 + registry map_casts/build_candidates(fake store,无网)。"""

from __future__ import annotations

from personos.identity.draft import MemoryDraftStore
from personos.identity.inspect import inspect_bind_collisions, present_casts
from personos.identity.registry import AnchorRegistry
from personos.identity.screenplay import CastDecl, ClipLine, ClipScript, Nomination
from personos.identity.types import CandidateCard, CastEvidence

S = "sess1"


def _script(casts, cast_map, *, lines=(), noms=(), cont=None):
    sc = ClipScript(casts=list(casts), lines=list(lines), nominations=list(noms),
                    cont=dict(cont or {}))
    sc.cast_map = dict(cast_map)
    return sc


# ── inspect ──────────────────────────────────────────────────────────
def test_present_casts_ignores_env_and_empty_shell():
    sc = _script([CastDecl(local_id="P1"), CastDecl(local_id="P2")],
                 {"P1": "S1", "P2": "S2"},
                 lines=[ClipLine(0, 1, "P1", "speech", "hi"),
                        ClipLine(1, 2, "ENV", "environment", "a room")],
                 noms=[Nomination(local_id="P2", t=1.0)])
    assert present_casts(sc) == {"S1", "S2"}               # ENV 不计,P2 靠提名在场


def test_bind_collision_detects_copresent_same_char():
    sc = _script([CastDecl(local_id="P1"), CastDecl(local_id="P2"), CastDecl(local_id="P3")],
                 {"P1": "S1", "P2": "S2", "P3": "S3"},
                 lines=[ClipLine(0, 1, "P1", "speech", "a"), ClipLine(1, 2, "P2", "speech", "b"),
                        ClipLine(2, 3, "P3", "speech", "c")])
    proposed = {"S1": "char_a", "S2": "char_a", "S3": "char_b"}   # S1/S2 撞 char_a
    vs = inspect_bind_collisions(sc, proposed)
    assert len(vs) == 1 and vs[0].cast_ids == ("S1", "S2") and vs[0].rule == "bind_collision"


def test_bind_collision_excludes_new_and_wearer():
    sc = _script([CastDecl(local_id="P1"), CastDecl(local_id="SW", is_wearer=True)],
                 {"P1": "S1", "SW": "SW"},
                 lines=[ClipLine(0, 1, "P1", "speech", "a"), ClipLine(1, 2, "SW", "speech", "b")])
    assert inspect_bind_collisions(sc, {"S1": "NEW", "SW": "char_a"}) == []


# ── registry.map_casts ───────────────────────────────────────────────
class _FakeCloud:
    def coarse_recall(self, evidence, ids, k=3):
        return [(i, 0.0) for i in ids[:k]]


class _FakeStore:
    def __init__(self, chars):
        self._chars = chars   # [{id, primary_name, payload, names:[...]}]

    def list_active_characters(self, *, include_wearer=False):
        return self._chars

    def names_for(self, cid):
        return next((c.get("names", []) for c in self._chars if c["id"] == cid), [])

    def best_asset(self, cid, kind):
        return None


def _registry(chars=()):
    return AnchorRegistry(_FakeStore(list(chars)), _FakeCloud(), MemoryDraftStore("u1"),
                          media_store=None)


def test_map_casts_mint_and_continuation():
    reg = _registry()
    # 首 clip:P1/P2 新人 → 铸 S1/S2;SW 恒 SW
    sc = _script([CastDecl(local_id="P1"), CastDecl(local_id="P2"),
                  CastDecl(local_id="SW", is_wearer=True)], {})
    m = reg.map_casts(S, sc)
    assert m == {"P1": "S1", "P2": "S2", "SW": "SW"}
    # 写 roster 后,第二 clip 用 CONT prev=S1 续接
    reg.draft.save_roster_entry(S, "S1", character_id=None, card={"name": "Bob"})
    sc2 = _script([CastDecl(local_id="P1")], {}, cont={"P1": "S1"})
    assert reg.map_casts(S, sc2) == {"P1": "S1"}


def test_map_casts_unknown_cont_mints_new():
    reg = _registry()
    sc = _script([CastDecl(local_id="P1")], {}, cont={"P1": "S9"})   # S9 不在 roster
    assert reg.map_casts(S, sc) == {"P1": "S1"}
    assert any("S9" in i for i in sc.issues)


# ── registry.build_candidates ─────────────────────────────────────────
def test_build_candidates_small_library_passthrough_and_extras():
    chars = [{"id": "char_a", "primary_name": "A", "payload": {}, "names": ["A"]},
             {"id": "char_b", "primary_name": "B", "payload": {}, "names": ["B"]}]
    reg = _registry(chars)
    sc = _script([CastDecl(local_id="P1")], {"P1": "S1"})
    extra = {"S1": [CandidateCard(character_id="chain:sess1:S2", name="pend")]}
    out = reg.build_candidates(["S1"], {"S1": CastEvidence("S1")}, sc, extra_cards=extra)
    ids = [c.character_id for c in out["S1"]]
    assert ids == ["char_a", "char_b", "chain:sess1:S2"]   # 小库全量 + extra 附尾


def test_build_candidates_empty_library_only_extras():
    reg = _registry([])
    sc = _script([CastDecl(local_id="P1")], {"P1": "S1"})
    out = reg.build_candidates(["S1"], {"S1": CastEvidence("S1")}, sc)
    assert out["S1"] == []
