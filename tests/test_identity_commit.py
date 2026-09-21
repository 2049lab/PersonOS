"""⑦ commit_session 单测:真 SIT(db fixture=rollback_scope 原子回退)+ MemoryDraft + MockOmni。

覆盖:NEW 建档+学云、命中既有档、SW→wearer 指针、终审调用失败回退链假设、同框碰撞组降级。
"""

from __future__ import annotations

import numpy as np

from personos.identity.backends.mock import MockOmni
from personos.identity.chains import FIRST_SEEN, ChainBook
from personos.identity.cloud import CloudEngine
from personos.identity.commit import commit_session
from personos.identity.draft import MemoryDraftStore
from personos.identity.registry import AnchorRegistry
from personos.identity.store import CharacterStore
from personos.identity.types import CastEvidence, FacePick

U = "vtest_commit"
S = "sess1"


class _FailOmni:
    def chat(self, *a, **k):
        raise RuntimeError("omni down")


def _e(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64); v[i] = 1.0
    return v


def _setup(db):
    store = CharacterStore(db, U)
    cloud = CloudEngine(store, template_cap=8)
    draft = MemoryDraftStore(U)
    book = ChainBook(draft, media_store=None)
    registry = AnchorRegistry(store, cloud, draft, media_store=None)
    return store, cloud, draft, book, registry


def _seed_chain(draft, cast, *, desc="a person", presence=(0,), face_i=0, q=0.8, best_face_q=0.8):
    """建一条 pending 链 + 暂存一份带向量的脸证据(media None → oss_key 空,不影响学云)。"""
    draft.ensure_chain(S, cast)
    ref = draft.chain_ref(S, cast)
    draft.update_chain(ref, desc_text=desc, presence=list(presence), best_face_q=best_face_q)
    ev = CastEvidence(cast, faces=[FacePick(t=0.0, embedding=_e(face_i), q=q, crop_b64="")])
    draft.stage_evidence(ref, session_id=S, clip_index=0, evidence=ev, media_store=None)
    return ref


def test_commit_new_registration_and_learn(db):
    store, cloud, draft, book, registry = _setup(db)
    ref = _seed_chain(draft, "S1", face_i=1)
    omni = MockOmni("BIND|S1|NEW\nEND")                     # 空库 → 只能 NEW
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert len(report["registered"]) == 1
    cid = report["by_chain"][ref]
    assert cid == report["registered"][0]
    assert store.get_character(cid) is not None
    assert len(store.active_assets(cid, "face")) == 1      # 暂存素材已落库
    mean, tau, n = store.load_prototype(cid, "face")
    assert mean is not None and n == 1                      # 学云生效
    assert draft.get_chain(ref)["status"] == "committed"


def test_commit_match_existing(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")             # 库里已有 char a
    ref = _seed_chain(draft, "S1", face_i=0)
    omni = MockOmni("BIND|S1|%s\nEND" % a)
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    assert report["by_chain"][ref] == a and report["registered"] == []
    assert len(store.active_assets(a, "face")) == 1         # 素材追加到既有档


def test_commit_wearer_pointer(db):
    store, cloud, draft, book, registry = _setup(db)
    draft.ensure_chain(S, "SW")
    ref = draft.chain_ref(S, "SW")
    draft.update_chain(ref, presence=[0])                   # SW 无脸,仅声纹场景:这里只验 wearer 指针
    omni = MockOmni("BIND|SW|NEW\nEND")
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    wearer = report["wearer"]
    assert wearer and store.get_character(wearer)["is_wearer"] is True
    assert store.wearer_character() == wearer
    assert store.session_wearer(S) == wearer


def test_commit_call_failure_falls_back_to_hypothesis(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")
    ref = _seed_chain(draft, "S1")
    draft.update_chain(ref, hypothesis=a, hypo_method="omni_rerank")   # 已有假设
    report = commit_session(store, cloud, registry, book, session_id=S, omni=_FailOmni())
    assert report["fallbacks"][ref] == "hypothesis"
    assert report["by_chain"][ref] == a                     # 回退到链假设 a


def test_name_merge_guardrails():
    """名字兜底:同名+非同框+都NEW → 并到最长链;同框/已绑档 → 不并(护栏)。纯 draft,无网。"""
    from personos.identity.commit import _merge_same_name_chains
    draft = MemoryDraftStore(U)
    book = ChainBook(draft, media_store=None)

    def _chain(cast, name, presence, hyp="NEW"):
        draft.ensure_chain(S, cast)
        ref = draft.chain_ref(S, cast)
        draft.update_chain(ref, presence=list(presence), hypothesis=hyp)
        draft.save_roster_entry(S, cast, character_id=None, card={"name": name})
        return ref

    # Bob 分裂成两条非同框链(presence 不交)→ 应合并,保留 presence 更长的 S1
    r1 = _chain("S1", "Bob", [0, 1, 2])
    r3 = _chain("S3", "Bob", [5])
    # Alice 两条但同框(presence 交)→ 护栏②不并
    r2 = _chain("S2", "Alice", [0])
    r4 = _chain("S4", "Alice", [0])
    # Mike 两条非同框但已绑不同注册档 → 护栏①不并(尊重模型区分两个真 Mike)
    r5 = _chain("S5", "Mike", [1], hyp="char_m1")
    r6 = _chain("S6", "Mike", [7], hyp="char_m2")

    report: dict = {}
    _merge_same_name_chains(book, S, report)

    assert draft.canonical_chain(r3) == r1                 # Bob 并到最长的 S1
    assert draft.canonical_chain(r2) != draft.canonical_chain(r4)   # Alice 同框未并
    assert draft.canonical_chain(r5) != draft.canonical_chain(r6)   # Mike 已绑档未并
    merges = report.get("name_merges", [])
    assert len(merges) == 1 and merges[0]["name"] == "Bob" and merges[0]["kept"] == r1


def test_commit_copresent_collision_degrades_one(db):
    store, cloud, draft, book, registry = _setup(db)
    a = store.create_character(session_id="s0")
    # S1、S2 同框(presence 交集 [0]),终审都判 a → 应降级一条
    r1 = _seed_chain(draft, "S1", presence=(0,), best_face_q=0.9, face_i=0)
    r2 = _seed_chain(draft, "S2", presence=(0,), best_face_q=0.5, face_i=1)
    omni = MockOmni("BIND|S1|%s\nBIND|S2|%s\nEND" % (a, a))
    report = commit_session(store, cloud, registry, book, session_id=S, omni=omni)
    finals = {report["by_chain"][r1], report["by_chain"][r2]}
    assert a in finals and len(finals) == 2                 # 一条留 a,一条降级建新档
    assert report["by_chain"][r1] == a                      # best_face_q 高者(S1)留
    assert len(report["registered"]) == 1                   # 降级的那条建了新档
