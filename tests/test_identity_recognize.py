"""⑥c 批量仲裁 + enroll 单测:parse_verdicts / build_arbitration_prompt(纯逻辑),enroll(真 SIT)。

仲裁改批量 BIND 协议(break change):逐处对齐 mneme arbiter.py。
"""

from __future__ import annotations

import numpy as np

from personos.identity.cloud import CloudEngine
from personos.identity.recognize import (
    NEW,
    build_arbitration_prompt,
    enroll_evidence,
    parse_verdicts,
    parse_verdicts_detailed,
)
from personos.identity.store import CharacterStore
from personos.identity.types import CandidateCard, CastEvidence, FacePick

U = "vtest_recog"


def _e(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64); v[i] = 1.0
    return v


def _ev(i: int, *, crop: str = "") -> CastEvidence:
    return CastEvidence(cast_id="P1", faces=[FacePick(t=0.0, embedding=_e(i), q=0.8,
                                                      crop_b64=crop, descriptor="a person")])


# ── parse_verdicts(BIND 协议)─────────────────────────────────────────
def test_parse_verdicts_bind_and_new():
    raw = "BIND|S1|char_a\nBIND|S2|NEW\nEND"
    verdicts, issues = parse_verdicts(raw, cast_ids=["S1", "S2"], candidate_ids=["char_a"])
    assert verdicts == {"S1": "char_a", "S2": NEW} and not issues


def test_parse_verdicts_unknown_target_defaults_new():
    raw = "BIND|S1|ghost\nEND"                               # 越白名单
    verdicts, issues, defaulted = parse_verdicts_detailed(
        raw, cast_ids=["S1"], candidate_ids=["char_a"])
    assert verdicts["S1"] == NEW and "S1" in defaulted and issues


def test_parse_verdicts_missing_line_defaults_new():
    verdicts, issues, defaulted = parse_verdicts_detailed(
        "BIND|S1|char_a\nEND", cast_ids=["S1", "S2"], candidate_ids=["char_a"])
    assert verdicts["S2"] == NEW and "S2" in defaulted        # 缺行回退
    assert "S1" not in defaulted                              # 显式命中不算 defaulted


def test_parse_verdicts_explicit_new_not_defaulted():
    _, _, defaulted = parse_verdicts_detailed(
        "BIND|S1|NEW\nEND", cast_ids=["S1"], candidate_ids=["char_a"])
    assert "S1" not in defaulted                              # 显式 NEW 是真 verdict


def test_parse_verdicts_chain_target_allowed_if_in_candidates():
    raw = "BIND|S1|chain:sess:S2\nEND"
    verdicts, _ = parse_verdicts(raw, cast_ids=["S1"], candidate_ids=["chain:sess:S2"])
    assert verdicts["S1"] == "chain:sess:S2"                  # 链 id 在候选内即合法(跨段同人)


# ── build_arbitration_prompt(白名单 + 图/音编号)─────────────────────
def test_build_prompt_whitelist_and_media_numbering():
    queries = [{"cast_id": "S1", "desc": "a man", "name": "Bob", "key_lines": ["hi"],
                "face_b64": "QF", "voice_b64": "QV"}]
    cands = [CandidateCard(character_id="char_a", name="A", desc="denim",
                           face_b64="CF", body_b64="CB")]
    prompt, images, audios = build_arbitration_prompt(queries, cands)
    assert "ALLOWED TARGETS" in prompt and "char_a" in prompt
    assert images == ["CF", "CB", "QF"] and audios == ["QV"]  # 候选先、query 后,独立编号
    assert "BIND|<query_id>|" in prompt


def test_build_prompt_no_candidates_new_only():
    prompt, images, audios = build_arbitration_prompt(
        [{"cast_id": "S1", "desc": "x"}], [])
    assert "NEW is the only valid answer" in prompt and not images and not audios


# ── enroll(真 SIT,rollback 回退)──────────────────────────────────────
def test_enroll_adds_asset_and_learns(db):
    s = CharacterStore(db, U)
    cl = CloudEngine(s, template_cap=8)
    cid = s.create_character()
    enroll_evidence(s, cl, None, cid, _ev(3))
    assert len(s.active_assets(cid, "face")) == 1
    mean, tau, n = s.load_prototype(cid, "face")
    assert mean is not None and tau > 0 and n == 1
