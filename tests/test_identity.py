"""V0 身份层:CharacterStore(3 表存取)+ CloudEngine(概率云)+ per-user 隔离。

用 conftest 的 autouse `db`(rollback_scope,写入测试结束回退,共享 SIT 库零污染)。
向量 roundtrip 走真 MySQL 的 UNHEX/HEX hex 通道(float32)。
"""

from __future__ import annotations

import numpy as np
import pytest

from personos.identity.cloud import CloudEngine
from personos.identity.store import CharacterStore
from personos.identity.types import CastEvidence, FacePick, VoiceSample

U = "vtest_user_a"
U2 = "vtest_user_b"


def _e(i: int, dim: int = 8) -> np.ndarray:
    """第 i 个坐标轴的单位向量(彼此正交,便于构造"像/不像")。"""
    v = np.zeros(dim, dtype=np.float64)
    v[i] = 1.0
    return v


@pytest.fixture
def store(db):
    return CharacterStore(db, U)


@pytest.fixture
def cloud(store):
    return CloudEngine(store, template_cap=3)


# ── CharacterStore ───────────────────────────────────────────────────

def test_create_and_get(store):
    cid = store.create_character(session_id="s1")
    ch = store.get_character(cid)
    assert ch and ch["id"] == cid and ch["user_id"] == U
    assert ch["is_wearer"] is False and ch["status"] == "active"
    assert ch["payload"]["name_claims"] == [] and ch["payload"]["first_session"] == "s1"


def test_list_active_excludes_wearer_by_default(store):
    normal = store.create_character()
    wearer = store.create_character(is_wearer=True)
    ids = {c["id"] for c in store.list_active_characters()}
    assert normal in ids and wearer not in ids
    ids_all = {c["id"] for c in store.list_active_characters(include_wearer=True)}
    assert wearer in ids_all


def test_name_claim_count_and_primary(store):
    cid = store.create_character()
    store.add_name_claim(cid, "李四", "被叫到")
    store.add_name_claim(cid, "老李", "别称")
    store.add_name_claim(cid, "李四", "又被叫")           # 李四 cnt=2
    assert store.names_for(cid) == ["李四", "老李"]
    assert store.get_character(cid)["primary_name"] == "李四"


def test_asset_roundtrip_and_retire(store):
    cid = store.create_character()
    emb = _e(3)
    aid = store.add_asset(cid, "face", quality=0.8, embedding=emb,
                          payload={"oss_key": "personos/u/face/x.jpg", "clip_index": 0})
    got = store.active_assets(cid, "face")
    assert len(got) == 1
    assert np.allclose(got[0]["embedding"].astype(np.float64), emb, atol=1e-6)
    assert got[0]["payload"]["oss_key"].endswith("x.jpg")
    store.retire_asset(aid)
    assert store.active_assets(cid, "face") == []


def test_prototype_roundtrip_and_tau_mirror(store):
    cid = store.create_character()
    mean = _e(2)
    store.save_prototype(cid, "face", mean, tau=3.5, n_obs=4)
    m, tau, n = store.load_prototype(cid, "face")
    assert np.allclose(m.astype(np.float64), mean, atol=1e-6) and tau == 3.5 and n == 4
    assert store.get_character(cid)["face_tau"] == 3.5   # τ 镜像到 characters


def test_template_add_list_remove(store):
    cid = store.create_character()
    t1 = store.add_template(cid, "face", _e(0), 0.5, payload={"clip_index": 1})
    store.add_template(cid, "face", _e(1), 0.9)
    assert len(store.templates(cid, "face")) == 2
    store.remove_template(t1)
    rest = store.templates(cid, "face")
    assert len(rest) == 1 and rest[0]["q"] == 0.9


def test_per_user_isolation(db, store):
    """A 建的角色,B 一律查不到——user_id 是墙。"""
    cid = store.create_character()
    other = CharacterStore(db, U2)
    assert other.get_character(cid) is None
    assert other.list_active_characters() == []
    assert other.load_prototype(cid, "face") == (None, 0.0, 0)


# ── CloudEngine ──────────────────────────────────────────────────────

def test_learn_grows_prototype(store, cloud):
    cid = store.create_character()
    assert cloud.learn(cid, "face", _e(0), q=1.0) is True
    assert cloud.learn(cid, "face", _e(0), q=0.5) is True
    _m, tau, n = store.load_prototype(cid, "face")
    assert tau == pytest.approx(1.5) and n == 2


def test_zero_quality_not_learned(store, cloud):
    cid = store.create_character()
    assert cloud.learn(cid, "face", _e(0), q=0.0) is False
    assert store.load_prototype(cid, "face") == (None, 0.0, 0)


def test_score_similar_beats_dissimilar(store, cloud):
    cid = store.create_character()
    cloud.learn(cid, "face", _e(0), q=1.0)
    sim = cloud.score_observation(cid, "face", _e(0), 1.0)
    dis = cloud.score_observation(cid, "face", _e(1), 1.0)
    assert sim is not None and dis is not None and sim > dis


def test_score_none_when_no_cloud(store, cloud):
    cid = store.create_character()
    assert cloud.score_observation(cid, "voice", _e(0), 1.0) is None   # 该模态无云


def test_template_cap_enforced(store, cloud):
    cid = store.create_character()
    for i in range(5):                       # cap=3,喂 5 个不同向量
        cloud.learn(cid, "face", _e(i), q=0.5 + 0.1 * i)
    assert len(store.templates(cid, "face")) == 3


def test_coarse_recall_ranks_match_first(store, cloud):
    a = store.create_character()
    b = store.create_character()
    cloud.learn(a, "face", _e(0), q=1.0)
    cloud.learn(b, "face", _e(1), q=1.0)
    ev = CastEvidence(cast_id="P1", faces=[FacePick(t=0.0, embedding=_e(0), q=1.0)])
    ranked = cloud.coarse_recall(ev, [a, b], k=2)
    assert ranked[0][0] == a and ranked[0][1] > ranked[1][1]


def test_score_evidence_fuses_modalities(store, cloud):
    cid = store.create_character()
    cloud.learn(cid, "face", _e(0), q=1.0)
    cloud.learn(cid, "voice", _e(2), q=1.0)
    ev = CastEvidence(cast_id="P1",
                      faces=[FacePick(t=0.0, embedding=_e(0), q=1.0)],
                      voices=[VoiceSample(t0=0.0, t1=1.0, embedding=_e(2), q=1.0)])
    only_face = CastEvidence(cast_id="P1", faces=[FacePick(t=0.0, embedding=_e(0), q=1.0)])
    assert cloud.score_evidence(cid, ev) > cloud.score_evidence(cid, only_face)  # 双模态相加更高
