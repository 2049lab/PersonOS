"""剧本物理一致性守护:检测 8 条规则 → 统一重修 → 保守降级。

为什么值得单独一组测试:这层的职责是**兜住 MLLM 的幻觉**,所以它自己必须极其可靠——
误判会白白丢素材,漏判会让错误归属进概率云(不可逆),而它挂了不能拖垮整条 clip。
三件事各自锁死:检得出 / 改得对 / 改不动时安全降级且不影响后续流程。
"""

from __future__ import annotations

import json

import pytest

from personos.identity import repair
from personos.identity.inspect import (
    inspect_cont_conflict, inspect_duplicate_cast, inspect_name_claims,
    inspect_nom_position, inspect_script, inspect_voice_overlap, inspect_wearer_visible,
)
from personos.identity.screenplay import (
    CastDecl, ClipLine, ClipScript, Nomination, VoiceRange,
)


def _script(**kw) -> ClipScript:
    base = dict(casts=[CastDecl(local_id="P1", desc="a"), CastDecl(local_id="P2", desc="b")],
                lines=[ClipLine(t0=0.0, t1=1.0, who="P1", kind="speech", text="hi"),
                       ClipLine(t0=1.0, t1=2.0, who="P2", kind="speech", text="yo")],
                nominations=[], voice_ranges=[], cont={}, cont_evidence={}, parsed_ok=True)
    base.update(kw)
    return ClipScript(**base)


# ── 检测 ────────────────────────────────────────────────────────────────

def test_rule1_voice_overlap():
    """两人语音区间相交 —— 不挡住的话两个人的声音会被混进同一个声纹模板。"""
    s = _script(voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    v = inspect_voice_overlap(s)
    assert len(v) == 1 and v[0].rule == "voice_overlap"
    assert v[0].times == (4.0, 5.0) and set(v[0].cast_ids) == {"P1", "P2"}
    # 同一个人的区间相邻/重叠不算矛盾(他本来就一直在说)
    assert inspect_voice_overlap(_script(voice_ranges=[
        VoiceRange(local_id="P1", t0=1.0, t1=5.0),
        VoiceRange(local_id="P1", t0=4.0, t1=8.0)])) == []


def test_rule2_cont_conflict():
    """两个 cast 都续接同一昔日成员 = 一个人变成两个人。"""
    v = inspect_cont_conflict(_script(cont={"P1": "S1", "P2": "S1"}))
    assert len(v) == 1 and set(v[0].cast_ids) == {"P1", "P2"}
    assert inspect_cont_conflict(_script(cont={"P1": "S1", "P2": "none"})) == []


def test_rule3_nom_position_conflict():
    """同一 cast 瞬时被提名在两个位置 —— 至少一个提名指错了人,照它抽脸会污染档案。"""
    s = _script(nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.3, pos="right")])
    v = inspect_nom_position(s)
    assert len(v) == 1 and v[0].cast_ids == ("P1",) and v[0].times == (6.0, 6.3)
    # 间隔够久 = 人走动了,合法
    assert inspect_nom_position(_script(nominations=[
        Nomination(local_id="P1", t=6.0, pos="left"),
        Nomination(local_id="P1", t=9.0, pos="right")])) == []


def test_rule5_duplicate_cast():
    s = _script(casts=[CastDecl(local_id="P1"), CastDecl(local_id="P1"), CastDecl(local_id="P2")])
    v = inspect_duplicate_cast(s)
    assert len(v) == 1 and v[0].cast_ids == ("P1",)


def test_rule7_wearer_visible():
    """佩戴者被提名出现在画面 —— 第一人称拍不到自己,那张脸一定是别人的。"""
    v = inspect_wearer_visible(_script(nominations=[Nomination(local_id="SW", t=3.0, pos="center")]))
    assert len(v) == 1 and v[0].cast_ids == ("SW",)


def test_rule8_9_name_claims():
    """名字对账:防模型复读 roster 名字 + 伪造台词证据。"""
    # 8:声称来自台词,但台词里根本没这个名字
    s = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")])
    v = inspect_name_claims(s)
    assert len(v) == 1 and v[0].rule == "name_unsupported"

    # 9:名字只出现在他自己的台词里(人不喊自己名字),且非自我介绍
    s2 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P1", kind="speech", text="Bob is here")])
    v2 = inspect_name_claims(s2)
    assert len(v2) == 1 and v2[0].rule == "name_self_address"

    # 自我介绍豁免
    s3 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="self_introduction")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P1", kind="speech", text="I am Bob")])
    assert inspect_name_claims(s3) == []

    # 别人喊了他 → 合法,不该误杀
    s4 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P2", kind="speech", text="Bob, come here")])
    assert inspect_name_claims(s4) == []

    # visible_text 豁免(画面文字无法与转写对账)
    s5 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="visible_text")])
    assert inspect_name_claims(s5) == []


def test_clean_script_has_no_violations():
    """干净剧本不得误报 —— 误报会白白丢素材。"""
    s = _script(nominations=[Nomination(local_id="P1", t=1.0, pos="left"),
                             Nomination(local_id="P2", t=1.0, pos="right")],
                voice_ranges=[VoiceRange(local_id="P1", t0=0.0, t1=1.0),
                              VoiceRange(local_id="P2", t0=1.0, t1=2.0)],
                cont={"P1": "S1", "P2": "S2"})
    assert inspect_script(s) == []


# ── 重修 ────────────────────────────────────────────────────────────────

class _Omni:
    """记录收到的 prompt;按队列返回预设响应。"""

    def __init__(self, outs):
        self.outs, self.prompts = list(outs), []

    def chat(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.outs.pop(0) if self.outs else "{}"


_FIXED = json.dumps({                       # 模型改对了:P2 不再续接 S1
    "casts": [{"id": "P1", "desc": "a"}, {"id": "P2", "desc": "b"}],
    "noms": [], "voices": [],
    "conts": [{"id": "P1", "prev": "S1"}, {"id": "P2", "prev": "none"}]})


def test_repair_fixes_violation_and_feeds_contradictions(monkeypatch):
    """触发规则 → prompt 里带上矛盾清单和上次输出 → 模型改对 → 采纳,不再降级。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni([_FIXED])
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)

    assert rep["found"] == {"cont_conflict": 1}
    assert rep["attempts"] == 1 and rep["degraded"] is False and rep["remaining"] == {}
    assert out.cont == {"P1": "S1", "P2": "none"}, "修复结果要被采纳"
    p = omni.prompts[0]
    assert "CONTRADICTIONS TO FIX" in p and "cont_conflict" in p
    assert "YOUR PREVIOUS RECORDS" in p, "要把模型上次的输出回灌,让它在自己的输出上改"
    assert "KEPT LINE RECORDS" in p and "do NOT re-output" in p, "台词保留不重出"
    assert out.lines == s.lines, "台词必须原样保留"


def test_repair_batches_all_violations_in_one_call(monkeypatch):
    """多条规则同时触发 → **一次调用**把所有矛盾一起交出去,不是一条改一条。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["{}", "{}"])              # 故意让它改不动,看调用次数与内容
    s = _script(cont={"P1": "S1", "P2": "S1"},
                nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.2, pos="right")],
                voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    _out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert set(rep["found"]) == {"cont_conflict", "nom_position_conflict", "voice_overlap"}
    p = omni.prompts[0]
    for rule in ("cont_conflict", "nom_position_conflict", "voice_overlap"):
        assert rule in p, f"{rule} 没进同一个 prompt"


def test_repair_rejected_when_reference_broken(monkeypatch):
    """重修把台词引用的 cast 删了 → 整次作废,退回降级(台词归属不能悬空)。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    broken = json.dumps({"casts": [{"id": "P1"}], "conts": []})   # P2 没了,但台词还引用它
    omni = _Omni([broken, broken])
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert rep["degraded"] is True
    assert {c.local_id for c in out.casts} == {"P1", "P2"}, "作废后应保留原 casts"


def test_degrade_when_repair_keeps_failing(monkeypatch):
    """两轮都改不动 → 保守降级,且降级只做减法。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["{}", "{}"])
    s = _script(cont={"P1": "S1", "P2": "S1"},
                nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.2, pos="right")])
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert rep["attempts"] == 2 and rep["degraded"] is True
    assert out.cont == {}, "续接冲突 → 断开续接"
    assert out.nominations == [], "分身提名 → 弃掉"
    assert out.lines == s.lines, "降级不动台词"


def test_degrade_voice_overlap_keeps_clean_remainder():
    """语音重叠 → 重叠段双方都不采,只留各自不重叠且 ≥0.4s 的残段。"""
    s = _script(voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    out = repair.degrade(s, inspect_voice_overlap(s))
    got = sorted((v.local_id, round(v.t0, 1), round(v.t1, 1)) for v in out.voice_ranges)
    assert got == [("P1", 1.0, 4.0), ("P2", 5.0, 8.0)], got


def test_degrade_strips_unsupported_name():
    s = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")])
    out = repair.degrade(s, inspect_name_claims(s))
    assert out.casts[0].name is None and out.casts[0].name_evidence == "none"


# ── 不影响后续流程 ──────────────────────────────────────────────────────

def test_clean_script_makes_no_mllm_call(monkeypatch):
    """无矛盾 → 一次模型都不调,剧本原样返回(不给正常路径加成本)。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["不该被调用"])
    s = _script()
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4")
    assert omni.prompts == [] and rep["found"] == {} and out is s


def test_detect_mode_logs_but_does_not_touch_script(monkeypatch):
    """detect 档:只检测留痕,不改剧本也不调模型(上线初期收触发率用)。"""
    monkeypatch.setattr(repair, "MODE", "detect")
    omni = _Omni(["不该被调用"])
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4")
    assert omni.prompts == [] and out is s
    assert rep["found"] == {"cont_conflict": 1} and rep["degraded"] is False


def test_off_mode_is_a_noop(monkeypatch):
    monkeypatch.setattr(repair, "MODE", "off")
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=_Omni([]), clip_url="https://x/c.mp4")
    assert out is s and rep["found"] == {}


def test_guard_never_raises(monkeypatch):
    """守护层自己坏了,最坏退化成"不守护",绝不能把整条 clip 拖垮。"""
    monkeypatch.setattr(repair, "MODE", "degrade")

    def _boom(_s):
        raise RuntimeError("检测器炸了")

    monkeypatch.setattr(repair, "inspect_script", _boom)
    s = _script()
    out, rep = repair.enforce(s, omni=_Omni([]), clip_url="https://x/c.mp4")
    assert out is s and "error" in rep


def test_mllm_exception_falls_back_to_degrade(monkeypatch):
    """重修时模型挂了 → 转降级,不抛。"""
    monkeypatch.setattr(repair, "MODE", "degrade")

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("上游 500")

    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=_Boom(), clip_url="https://x/c.mp4")
    assert rep["degraded"] is True and out.cont == {}


def test_no_omni_still_degrades(monkeypatch):
    """没装配模型(纯本地/单测场景)→ 跳过重修直接降级,不崩。"""
    monkeypatch.setattr(repair, "MODE", "degrade")
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=None, clip_url="")
    assert rep["attempts"] == 0 and rep["degraded"] is True and out.cont == {}
