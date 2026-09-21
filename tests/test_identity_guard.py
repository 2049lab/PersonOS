"""Screenplay physical-consistency guard: detect 8 rules -> batch repair -> conservative degrade.

Why this deserves its own test group: this layer exists to **contain multimodal LLM
hallucinations**, so it has to be extremely reliable itself -- a false positive throws away
usable assets, a missed violation lets a wrong attribution into the probability cloud
(irreversible), and if the layer itself crashes it must not take the whole clip down with it.
Three things are pinned separately: it detects, it repairs correctly, and when it cannot
repair it degrades safely without disturbing the rest of the pipeline.
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


# ── Detection ───────────────────────────────────────────────────────────

def test_rule1_voice_overlap():
    """Two people's voice ranges intersect -- if this is not blocked, two people's voices
    end up mixed into the same voiceprint template."""
    s = _script(voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    v = inspect_voice_overlap(s)
    assert len(v) == 1 and v[0].rule == "voice_overlap"
    assert v[0].times == (4.0, 5.0) and set(v[0].cast_ids) == {"P1", "P2"}
    # Adjacent or overlapping ranges for the same person are not a contradiction
    # (that person was simply talking continuously)
    assert inspect_voice_overlap(_script(voice_ranges=[
        VoiceRange(local_id="P1", t0=1.0, t1=5.0),
        VoiceRange(local_id="P1", t0=4.0, t1=8.0)])) == []


def test_rule2_cont_conflict():
    """Two casts continuing the same earlier member means one person has become two."""
    v = inspect_cont_conflict(_script(cont={"P1": "S1", "P2": "S1"}))
    assert len(v) == 1 and set(v[0].cast_ids) == {"P1", "P2"}
    assert inspect_cont_conflict(_script(cont={"P1": "S1", "P2": "none"})) == []


def test_rule3_nom_position_conflict():
    """The same cast is nominated at two positions at nearly the same instant -- at least one
    nomination points at the wrong person, and cropping a face from it would pollute the profile."""
    s = _script(nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.3, pos="right")])
    v = inspect_nom_position(s)
    assert len(v) == 1 and v[0].cast_ids == ("P1",) and v[0].times == (6.0, 6.3)
    # A long enough gap means the person simply moved, which is legitimate
    assert inspect_nom_position(_script(nominations=[
        Nomination(local_id="P1", t=6.0, pos="left"),
        Nomination(local_id="P1", t=9.0, pos="right")])) == []


def test_rule5_duplicate_cast():
    s = _script(casts=[CastDecl(local_id="P1"), CastDecl(local_id="P1"), CastDecl(local_id="P2")])
    v = inspect_duplicate_cast(s)
    assert len(v) == 1 and v[0].cast_ids == ("P1",)


def test_rule7_wearer_visible():
    """The wearer is nominated as visible on screen -- a first-person camera cannot film its own
    wearer, so that face must belong to someone else."""
    v = inspect_wearer_visible(_script(nominations=[Nomination(local_id="SW", t=3.0, pos="center")]))
    assert len(v) == 1 and v[0].cast_ids == ("SW",)


def test_rule8_9_name_claims():
    """Name reconciliation: stop the model from parroting roster names and from fabricating
    dialogue evidence for them."""
    # Rule 8: the name claims to come from dialogue, but no line contains that name
    s = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")])
    v = inspect_name_claims(s)
    assert len(v) == 1 and v[0].rule == "name_unsupported"

    # Rule 9: the name appears only in that person's own lines (people do not call out their own
    # name), and it is not a self-introduction
    s2 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P1", kind="speech", text="Bob is here")])
    v2 = inspect_name_claims(s2)
    assert len(v2) == 1 and v2[0].rule == "name_self_address"

    # Self-introduction is exempt
    s3 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="self_introduction")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P1", kind="speech", text="I am Bob")])
    assert inspect_name_claims(s3) == []

    # Someone else called him by name -> legitimate, must not be flagged
    s4 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")],
                 lines=[ClipLine(t0=0.0, t1=1.0, who="P2", kind="speech", text="Bob, come here")])
    assert inspect_name_claims(s4) == []

    # visible_text is exempt (on-screen text cannot be reconciled against the transcript)
    s5 = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="visible_text")])
    assert inspect_name_claims(s5) == []


def test_clean_script_has_no_violations():
    """A clean screenplay must not produce false positives -- they throw away usable assets."""
    s = _script(nominations=[Nomination(local_id="P1", t=1.0, pos="left"),
                             Nomination(local_id="P2", t=1.0, pos="right")],
                voice_ranges=[VoiceRange(local_id="P1", t0=0.0, t1=1.0),
                              VoiceRange(local_id="P2", t0=1.0, t1=2.0)],
                cont={"P1": "S1", "P2": "S2"})
    assert inspect_script(s) == []


# ── Repair ──────────────────────────────────────────────────────────────

class _Omni:
    """Records the prompts it receives; returns canned responses from a queue."""

    def __init__(self, outs):
        self.outs, self.prompts = list(outs), []

    def chat(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.outs.pop(0) if self.outs else "{}"


_FIXED = json.dumps({                       # The model fixed it: P2 no longer continues S1
    "casts": [{"id": "P1", "desc": "a"}, {"id": "P2", "desc": "b"}],
    "noms": [], "voices": [],
    "conts": [{"id": "P1", "prev": "S1"}, {"id": "P2", "prev": "none"}]})


def test_repair_fixes_violation_and_feeds_contradictions(monkeypatch):
    """A rule fires -> the prompt carries the contradiction list plus the previous output ->
    the model fixes it -> the fix is adopted and no degrade happens."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni([_FIXED])
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)

    assert rep["found"] == {"cont_conflict": 1}
    assert rep["attempts"] == 1 and rep["degraded"] is False and rep["remaining"] == {}
    assert out.cont == {"P1": "S1", "P2": "none"}, "the repaired result must be adopted"
    p = omni.prompts[0]
    assert "CONTRADICTIONS TO FIX" in p and "cont_conflict" in p
    assert "YOUR PREVIOUS RECORDS" in p, "feed the model's previous output back so it edits its own output"
    assert "KEPT LINE RECORDS" in p and "do NOT re-output" in p, "lines are kept, not re-emitted"
    assert out.lines == s.lines, "lines must be preserved exactly"


def test_repair_batches_all_violations_in_one_call(monkeypatch):
    """Several rules fire at once -> **a single call** hands over all contradictions together,
    rather than fixing them one at a time."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["{}", "{}"])              # Deliberately unable to fix, so we can inspect call count and content
    s = _script(cont={"P1": "S1", "P2": "S1"},
                nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.2, pos="right")],
                voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    _out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert set(rep["found"]) == {"cont_conflict", "nom_position_conflict", "voice_overlap"}
    p = omni.prompts[0]
    for rule in ("cont_conflict", "nom_position_conflict", "voice_overlap"):
        assert rule in p, f"{rule} did not make it into the same prompt"


def test_repair_rejected_when_reference_broken(monkeypatch):
    """The repair deleted a cast that lines still reference -> the whole attempt is discarded and
    we fall back to degrade (line attribution must never dangle)."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    broken = json.dumps({"casts": [{"id": "P1"}], "conts": []})   # P2 is gone, but lines still reference it
    omni = _Omni([broken, broken])
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert rep["degraded"] is True
    assert {c.local_id for c in out.casts} == {"P1", "P2"}, "the original casts must survive the discard"


def test_degrade_when_repair_keeps_failing(monkeypatch):
    """Neither round fixes anything -> conservative degrade, and the degrade only removes things."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["{}", "{}"])
    s = _script(cont={"P1": "S1", "P2": "S1"},
                nominations=[Nomination(local_id="P1", t=6.0, pos="left"),
                             Nomination(local_id="P1", t=6.2, pos="right")])
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4", duration_sec=60.0)
    assert rep["attempts"] == 2 and rep["degraded"] is True
    assert out.cont == {}, "continuation conflict -> drop the continuation"
    assert out.nominations == [], "cloned nominations -> discard them"
    assert out.lines == s.lines, "degrade never touches lines"


def test_degrade_voice_overlap_keeps_clean_remainder():
    """Voice overlap -> neither side keeps the overlapping segment; only the non-overlapping
    remainders of at least 0.4s survive."""
    s = _script(voice_ranges=[VoiceRange(local_id="P1", t0=1.0, t1=5.0),
                              VoiceRange(local_id="P2", t0=4.0, t1=8.0)])
    out = repair.degrade(s, inspect_voice_overlap(s))
    got = sorted((v.local_id, round(v.t0, 1), round(v.t1, 1)) for v in out.voice_ranges)
    assert got == [("P1", 1.0, 4.0), ("P2", 5.0, 8.0)], got


def test_degrade_strips_unsupported_name():
    s = _script(casts=[CastDecl(local_id="P1", name="Bob", name_evidence="explicit_dialogue")])
    out = repair.degrade(s, inspect_name_claims(s))
    assert out.casts[0].name is None and out.casts[0].name_evidence == "none"


# ── Must not disturb the rest of the pipeline ───────────────────────────

def test_clean_script_makes_no_mllm_call(monkeypatch):
    """No contradictions -> not a single model call, and the screenplay is returned unchanged
    (the normal path pays no extra cost)."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    omni = _Omni(["should never be called"])
    s = _script()
    out, rep = repair.enforce(s, omni=omni, clip_url="https://x/c.mp4")
    assert omni.prompts == [] and rep["found"] == {} and out is s


def test_detect_mode_logs_but_does_not_touch_script(monkeypatch):
    """detect mode: only record detections; never edit the screenplay and never call the model
    (used early after rollout to collect trigger rates)."""
    monkeypatch.setattr(repair, "MODE", "detect")
    omni = _Omni(["should never be called"])
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
    """If the guard layer itself breaks, the worst case is "no guarding" -- it must never take the
    whole clip down with it."""
    monkeypatch.setattr(repair, "MODE", "degrade")

    def _boom(_s):
        raise RuntimeError("inspector blew up")

    monkeypatch.setattr(repair, "inspect_script", _boom)
    s = _script()
    out, rep = repair.enforce(s, omni=_Omni([]), clip_url="https://x/c.mp4")
    assert out is s and "error" in rep


def test_mllm_exception_falls_back_to_degrade(monkeypatch):
    """The model fails during repair -> switch to degrade instead of raising."""
    monkeypatch.setattr(repair, "MODE", "degrade")

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("upstream 500")

    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=_Boom(), clip_url="https://x/c.mp4")
    assert rep["degraded"] is True and out.cont == {}


def test_no_omni_still_degrades(monkeypatch):
    """No model wired in (pure local or unit-test setup) -> skip repair, degrade directly, do not
    crash."""
    monkeypatch.setattr(repair, "MODE", "degrade")
    s = _script(cont={"P1": "S1", "P2": "S1"})
    out, rep = repair.enforce(s, omni=None, clip_url="")
    assert rep["attempts"] == 0 and rep["degraded"] is True and out.cont == {}
