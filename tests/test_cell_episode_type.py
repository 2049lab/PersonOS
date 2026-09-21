"""episode_type 存储 + 分页检索单测:写列往返 / 按类型过滤 / 时间范围 / 分页 / 倒序 / user 隔离。

共享 SIT 库,autouse db fixture 已把每个测试包进 rollback_scope(零污染)。
"""

from __future__ import annotations

from datetime import timedelta

from personos.models import MemCell, now
from personos.storage.cell_store import CellStore


def _cell(cs, sid, topic, ep_type, t):
    c = MemCell(session_id=sid, topic=topic, episode=f"episode of {topic}", episode_type=ep_type,
                t_start=t, t_end=t + timedelta(minutes=1))
    cs.upsert(c)
    return c


def test_episode_type_round_trip(db):
    cs = CellStore(db, user_id="u_et_rt")
    c = _cell(cs, "s", "lunch", "food", now())
    got = cs.get(c.id)
    assert got.episode_type == "food"


def test_default_unknown(db):
    cs = CellStore(db, user_id="u_et_def")
    c = MemCell(session_id="s", topic="t", episode="e", t_start=now())   # 不给 episode_type
    cs.upsert(c)
    assert cs.get(c.id).episode_type == "unknown"


def test_list_by_type_filters_and_desc(db):
    cs = CellStore(db, user_id="u_et_list")
    base = now()
    _cell(cs, "s", "a", "work", base)
    _cell(cs, "s", "b", "work", base + timedelta(minutes=5))
    _cell(cs, "s", "c", "health", base + timedelta(minutes=10))
    work = cs.list_by_type(episode_type="work", limit=20, offset=0)
    assert [c.topic for c in work] == ["b", "a"]                # 只 work,且 t_start 倒序
    assert cs.count_by_type(episode_type="work") == 2
    assert cs.count_by_type(episode_type="health") == 1
    assert cs.count_by_type() == 3                              # 不传类型 = 全部


def test_time_range_filter(db):
    cs = CellStore(db, user_id="u_et_time")
    base = now()
    _cell(cs, "s", "old", "t", base)
    _cell(cs, "s", "mid", "t", base + timedelta(hours=1))
    _cell(cs, "s", "new", "t", base + timedelta(hours=2))
    lo = (base + timedelta(minutes=30)).isoformat()
    hi = (base + timedelta(minutes=90)).isoformat()
    got = cs.list_by_type(start=lo, end=hi, limit=20, offset=0)
    assert [c.topic for c in got] == ["mid"]                   # 只落在 [lo,hi] 的
    assert cs.count_by_type(start=lo, end=hi) == 1


def test_pagination(db):
    cs = CellStore(db, user_id="u_et_page")
    base = now()
    for i in range(5):
        _cell(cs, "s", f"m{i}", "t", base + timedelta(minutes=i))   # m4 最新
    page1 = cs.list_by_type(episode_type="t", limit=2, offset=0)
    page2 = cs.list_by_type(episode_type="t", limit=2, offset=2)
    assert [c.topic for c in page1] == ["m4", "m3"]            # 倒序第 1 页
    assert [c.topic for c in page2] == ["m2", "m1"]            # 第 2 页
    assert cs.count_by_type(episode_type="t") == 5


def test_episode_vo_shape():
    """VO 只暴露段粒度对外字段(id/会话/起止/主题/叙事/分类),不含 payload/atoms/向量。"""
    from personos.app.service_api import _episode_vo
    base = now()
    c = MemCell(session_id="s1", topic="lunch", episode="had ramen", episode_type="food",
                t_start=base, t_end=base + timedelta(minutes=2))
    vo = _episode_vo(c)
    assert vo == {"memcell_id": c.id, "session_id": "s1",
                  "start_time": base.isoformat(), "end_time": (base + timedelta(minutes=2)).isoformat(),
                  "topic": "lunch", "episode": "had ramen", "episode_type": "food"}
    assert set(vo) == {"memcell_id", "session_id", "start_time", "end_time",
                       "topic", "episode", "episode_type"}   # 不多下发


def test_episode_vo_null_times():
    from personos.app.service_api import _episode_vo
    vo = _episode_vo(MemCell(session_id="s", topic="t", episode="e"))
    assert vo["start_time"] is None and vo["end_time"] is None


def test_user_isolation(db):
    ca = CellStore(db, user_id="u_et_a")
    cb = CellStore(db, user_id="u_et_b")
    _cell(ca, "s", "a-cell", "shared", now())
    _cell(cb, "s", "b-cell", "shared", now())
    assert [c.topic for c in ca.list_by_type(episode_type="shared")] == ["a-cell"]   # 各看各的
    assert cb.count_by_type(episode_type="shared") == 1
