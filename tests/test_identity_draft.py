"""⑤a 会话草稿(draft.py)纯逻辑单测:MemoryDraftStore,无网。

覆盖 union-find canonical / merge 压平 / presence 累积 / 评估台账 / staged best_pair·best_voice /
commit_chain 标记 / pending 过滤。逐处对照 mneme AnchorStore 语义。
"""

from __future__ import annotations

import numpy as np

from personos.identity.draft import MemoryDraftStore
from personos.identity.types import CastEvidence, FacePick, VoiceSample

S = "sess1"


def _draft() -> MemoryDraftStore:
    return MemoryDraftStore("u1")


def test_ensure_chain_defaults_and_idempotent():
    d = _draft()
    row, created = d.ensure_chain(S, "S1")
    assert created and row["hypothesis"] == "NEW" and row["status"] == "pending"
    assert row["canonical"] is None and row["best_face_q"] == -1.0 and row["presence"] == []
    row2, created2 = d.ensure_chain(S, "S1")
    assert not created2 and row2["chain_ref"] == d.chain_ref(S, "S1")


def test_update_field_whitelist():
    d = _draft()
    d.ensure_chain(S, "S1")
    ref = d.chain_ref(S, "S1")
    d.update_chain(ref, hypothesis="char_x", best_face_q=0.7, presence=[0, 1])
    row = d.get_chain(ref)
    assert row["hypothesis"] == "char_x" and row["best_face_q"] == 0.7 and row["presence"] == [0, 1]
    try:
        d.update_chain(ref, bogus=1)
        assert False, "应拒未知字段"
    except ValueError:
        pass


def test_merge_canonical_and_aliases_flatten():
    d = _draft()
    for c in ("S1", "S2", "S3"):
        d.ensure_chain(S, c)
    r1, r2, r3 = (d.chain_ref(S, c) for c in ("S1", "S2", "S3"))
    d.update_chain(r1, best_face_q=0.5, presence=[0])
    d.update_chain(r2, best_face_q=0.8, presence=[1], named=1)
    # S1 先并入 S2
    d.merge_chain(r1, r2)
    assert d.canonical_chain(r1) == r2
    assert d.aliases_of(r2) == [r1]
    dst = d.get_chain(r2)
    assert dst["best_face_q"] == 0.8 and dst["named"] == 1 and dst["presence"] == [0, 1]
    # S3 并入 S1(别名)→ 应压平到根 S2,保持一层深
    d.merge_chain(r3, r1)
    assert d.canonical_chain(r3) == r2
    assert set(d.aliases_of(r2)) == {r1, r3}
    # pending 只剩链根 S2
    assert [r["chain_ref"] for r in d.pending_chains(S)] == [r2]


def test_merge_same_chain_noop():
    d = _draft()
    d.ensure_chain(S, "S1"); d.ensure_chain(S, "S2")
    r1, r2 = d.chain_ref(S, "S1"), d.chain_ref(S, "S2")
    d.merge_chain(r1, r2)
    d.merge_chain(r1, r2)   # 已同链,幂等
    assert d.aliases_of(r2) == [r1]


def test_evaluations_append():
    d = _draft()
    d.ensure_chain(S, "S1")
    ref = d.chain_ref(S, "S1")
    d.add_chain_evaluation(ref, session_id=S, clip_index=0, reason="first_seen",
                           verdict="NEW", evidence={"best_face_q": 0.5})
    d.add_chain_evaluation(ref, session_id=S, clip_index=1, reason="better_face",
                           verdict="char_a", issues=["x"])
    evals = d.evaluations_for(ref)
    assert len(evals) == 2 and evals[0]["reason"] == "first_seen"
    assert evals[1]["verdict"] == "char_a" and evals[1]["issues"] == ["x"]


def test_roster_roundtrip():
    d = _draft()
    d.save_roster_entry(S, "S1", character_id="char_a", card={"name": "Bob", "key_lines": ["hi"]})
    roster = d.load_roster(S)
    assert roster["S1"]["name"] == "Bob" and roster["S1"]["character_id"] == "char_a"


def test_stage_and_best(monkeypatch):
    d = _draft()
    ref = d.chain_ref(S, "S1")
    d.ensure_chain(S, "S1")

    class _MS:   # 假 media_store:save_image 返回带 key 的对象,save_audio 返回 key 串
        def save_image(self, b, owner, content_type):
            return type("O", (), {"key": f"img/{len(b)}"})()
        def save_audio(self, b, owner):
            return f"aud/{len(b)}"

    ev = CastEvidence(cast_id="S1",
                      faces=[FacePick(t=1.0, embedding=np.ones(4), q=0.6, crop_b64="AAAA",
                                      body_crop_b64="BBBB"),
                             FacePick(t=2.0, embedding=np.ones(4) * 2, q=0.9, crop_b64="CCCC")],
                      voices=[VoiceSample(t0=0, t1=1, embedding=np.ones(3), q=0.7,
                                          wav_bytes=b"wavwav")])
    d.stage_evidence(ref, session_id=S, clip_index=0, evidence=ev, media_store=_MS())
    pair = d.best_pair(ref)
    assert pair["quality"] == 0.9                          # 最高质量脸
    assert pair["body_oss_key"] and pair["face_oss_key"]
    voice = d.best_voice(ref)
    assert voice["quality"] == 0.7
    faces = d.active_staged(ref, "face")
    assert len(faces) == 2 and faces[0]["q"] == 0.9        # 按 q 降序
    assert isinstance(faces[0]["embedding"], np.ndarray)


def test_stage_and_all_lines_time_order():
    d = _draft()
    d.stage_lines(S, 1, [(2.0, 3.0, "S1", "speech", "later")])
    d.stage_lines(S, 0, [(1.0, 2.0, "S1", "speech", "hi"), (0.5, 1.0, "ENV", "environment", "a room")])
    rows = d.all_lines(S)
    # 按 (clip, t0) 时序:clip0 的 ENV(0.5) → clip0 的 speech(1.0) → clip1(2.0)
    assert [r["text"] for r in rows] == ["a room", "hi", "later"]
    assert rows[0]["who"] == "ENV" and rows[1]["who"] == "S1"


def test_next_clip_seq_monotonic():
    d = _draft()
    assert [d.next_clip_seq(S) for _ in range(4)] == [0, 1, 2, 3]   # 会话内单调
    assert d.next_clip_seq("sess2") == 0                            # 别的会话独立从 0


def test_commit_chain_marks_aliases():
    d = _draft()
    d.ensure_chain(S, "S1"); d.ensure_chain(S, "S2")
    r1, r2 = d.chain_ref(S, "S1"), d.chain_ref(S, "S2")
    d.merge_chain(r1, r2)
    d.commit_chain(r2, "char_final")
    assert d.get_chain(r1)["status"] == "committed"
    assert d.get_chain(r2)["final_character_id"] == "char_final"
    assert d.pending_chains(S) == []                       # 提交后无 pending
