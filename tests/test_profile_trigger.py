"""画像触发退火 + 单 user 编排单测:anneal/should(纯逻辑)+ run_user_consolidation(真 store + FakeLLM)。"""

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
    assert [anneal_step(v) for v in range(8)] == [1, 2, 2, 3, 4, 5, 5, 5]   # 冷启动1,涨到5封顶


def test_should_consolidate_either_condition():
    # 退火:第 0 版 step=1,1 个新 cell 即触发
    assert should_consolidate(n_new=1, ep_chars=10, version_count=0, ep_chars_trigger=15000)
    # 第 1 版 step=2,1 个 cell 不够 & 字数不够 → 不触发
    assert not should_consolidate(n_new=1, ep_chars=10, version_count=1, ep_chars_trigger=15000)
    # 字数达上界 → 触发(哪怕 cell 数没到 step)
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
    assert cur.up_to_cell_id == c.id                       # 游标记到最后一个 cell
    t = cur.profile.traits["interests"]
    assert t.text == "likes ramen" and t.sources == [c.id]  # 短标 c1 → 真实 cell_id 回填


def test_run_user_consolidation_no_new_cells_returns_none(db):
    cs, ats, c = _seed_cell(db, "u_cons_2", "x", "y (2026-09-14)")
    ps = ProfileStore(db, "u_cons_2")
    patch = json.dumps({"traits": {"identity": {"text": "engineer", "status": "confirmed",
                                               "sources": ["c1"]}}})
    assert run_user_consolidation(FakeLLM([patch]), cells_store=cs, atoms_store=ats,
                                  profile_store=ps, today=now().date()) == 1
    # 再跑:游标已到 c,无新 cell → None(幂等,不重复出版)
    assert run_user_consolidation(FakeLLM([patch]), cells_store=cs, atoms_store=ats,
                                  profile_store=ps, today=now().date()) is None


def test_single_flight_lock_serializes_two_triggers():
    """并发两次触发(模拟两 pod/两次关段)共享一把锁:只有一个抢到真跑,另一个直接返回(不重复整理)。

    纯锁语义(不碰 pin 的 db 连接,DB 并发由 E2E 脚本真链路覆盖):验证 runtime._run_user_profile
    的单飞骨架——try_acquire 抢不到即返回。
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
            return                                     # 抢不到 → 交给对方(幂等自愈)
        try:
            ran["n"] += 1
            time.sleep(0.05)                           # 放大临界区窗口
        finally:
            lock.release("u", "profile", token)

    ts = [threading.Thread(target=worker) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(2)
    assert ran["n"] == 1                               # 单飞:同一时刻只有一个整理在跑
