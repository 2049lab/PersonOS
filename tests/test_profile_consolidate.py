"""consolidate 单测(干净结构 + 短/长 id 映射):成功 / 打回后成功 / 保留旧版 /
非 JSON 重试 / 无 cell 空转 / 超限打回 / 短标回填成真实 cell_id。"""

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
    assert t.sources == ["cell_LONG_ULID_1"]              # 短标 c1 → 真实长 id 回填
    assert [f.text for f in p.facts["today"]] == ["2026-09-14 had ramen, dislikes cilantro"]
    assert p.facts["today"][0].sources == ["cell_LONG_ULID_1"]


def test_consolidate_bounces_then_succeeds():
    """首版坏 status → 打回;次版合法 → 成功。"""
    cells, atoms = _cells()
    bad = _patch(traits={"goals": {"text": "x", "status": "maybe", "sources": ["c1"]}})
    good = _patch(traits={"goals": {"text": "ship the memory service", "status": "confirmed",
                                   "sources": ["c1"]}})
    p = consolidate(FakeLLM([bad, good]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is not None and p.traits["goals"].text == "ship the memory service"


def test_consolidate_field_invalid_keeps_old():
    """字段级打回上限仍不过(幻觉出处)→ None(保留旧版)。"""
    cells, atoms = _cells()
    bad = _patch(facts={"add": [{"band": "today", "text": "x", "sources": ["c_ghost"]}]})
    p = consolidate(FakeLLM([bad, bad, bad]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is None


def test_consolidate_overflow_bounces_then_backstop():
    """LLM 反复往 today 塞 2 条(超上限 1)→ 打回;仍不收敛 → 引擎踢最旧兜底后出版(非 None)。"""
    cells, atoms = _cells()
    over = _patch(facts={"add": [{"band": "today", "text": "a", "sources": ["c1"]},
                                 {"band": "today", "text": "b", "sources": ["c1"]}]})
    p = consolidate(FakeLLM([over, over, over]), current=None, cells=cells, atoms_by_cell=atoms,
                    today=TODAY, max_retries=2)
    assert p is not None and len(p.facts["today"]) == 1   # 兜底收敛到上限


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
    """补丁只改一个域,当前画像其余内容保留(增量语义端到端)。"""
    cells, atoms = _cells()
    from personos.storage.profile_store import ProfileTrait
    cur = UserProfile.empty()
    cur.traits["location"] = ProfileTrait(text="Singapore", status="confirmed",
                                          last_confirmed="2026-01-01", sources=["cell_0"])
    patch = _patch(traits={"occupation": {"text": "engineer", "status": "confirmed", "sources": ["c1"]}})
    p = consolidate(FakeLLM([patch]), current=cur, cells=cells, atoms_by_cell=atoms, today=TODAY)
    assert p.traits["location"].text == "Singapore"       # 未提域保留
    assert p.traits["occupation"].text == "engineer"


# 放最后:import 在文件头会与 conftest 顺序无关,这里就近引用
from .fakes import FakeLLM  # noqa: E402
