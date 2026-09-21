"""build_chains 回填测试(§4.3):清链重放 + 幂等(重跑无重复)+ 统计口径。

FakeLLM 按格出队判链 JSON;重放顺序确定性(格时间序 × 格内发生时间),可断言。
"""

from __future__ import annotations

from datetime import datetime, timezone

from scripts.build_chains import rebuild_user

from .fakes import FakeLLM
from .test_deep_recall import Env
from .test_retrieval import _v


def _scene(db):
    """两格三 atom:c1(08-10,健身房+教练)、c2(08-24,搬迁)。
    FakeLLM 两响应:第 1 格(无候选)拆两条新链;第 2 格追加到候选 c1(地点链)。"""
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
        '{"assignments":[{"chain":"c1","atoms":[1]}]}',       # c1=预筛候选(地点链),追加
    ]
    return env, responses


def test_rebuild_clears_and_replays_in_time_order(db):
    """回填:先清本 user 链,再按格时间序重放 W2.5(跨格可追加);统计口径正确。"""
    env, responses = _scene(db)
    stats = rebuild_user(db, FakeLLM(responses), "")

    assert stats["cleared_chains"] == 0                       # 原本无链
    assert stats["rebuilt_assigned"] == 3                     # 三 atom 全分配
    assert stats["n_chains"] == 2 and stats["chained_atoms"] == 3
    assert stats["free_rate"] == 0.0
    assert (2, "地点链") in stats["top_chains"]               # 搬迁 atom 追加进了地点链
    assert (1, "教练链") in stats["top_chains"]


def test_rebuild_is_idempotent(db):
    """幂等:重跑先清后建,链数/挂链 atom 数不变,无重复链行;atom 本身不动。"""
    env, responses = _scene(db)
    s1 = rebuild_user(db, FakeLLM(responses), "")
    s2 = rebuild_user(db, FakeLLM(list(responses)), "")       # 重跑(新 LLM 队列同响应)

    assert s2["cleared_chains"] == s1["n_chains"]             # 清掉了上一轮的链
    assert s2["n_chains"] == s1["n_chains"]                   # 重建后数量一致,无重复
    assert s2["chained_atoms"] == s1["chained_atoms"] == 3
    assert s2["n_atoms"] == s1["n_atoms"]                     # 不重抽 atom
