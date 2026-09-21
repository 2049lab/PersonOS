"""V2 single-pass screenplay (JSON protocol): parse_clip_output validation plus
build_clip_prompt (offline, no network)."""

from __future__ import annotations

import json

from personos.identity.screenplay import (
    ENV_WHO,
    build_clip_prompt,
    parse_clip_output,
)

GOOD = json.dumps({
    "casts": [
        {"id": "P1", "wearer": False, "name": "David", "name_evidence": "explicit_dialogue",
         "desc": "middle-aged man in red plaid shirt"},
        {"id": "SW", "wearer": True, "name": "-", "name_evidence": "none", "desc": "behind camera"},
    ],
    "lines": [
        {"t0": 1.58, "t1": 3.18, "who": "P1", "kind": "speech", "text": "Hey robot, what juice?"},
        {"t0": 13.7, "t1": 19.7, "who": "P1", "kind": "action", "text": "walks to the desk"},
        {"t0": 0.0, "t1": 13.7, "who": "ENV", "kind": "environment", "text": "Bright attic room"},
        {"t0": 5.0, "t1": 6.0, "who": "SW", "kind": "action", "text": "hands the cup over"},
    ],
    "noms": [{"id": "P1", "t": 4.1, "pos": "left", "desc": "red plaid man, left"}],
    "voices": [{"id": "P1", "t0": 3.3, "t1": 6.7}],
    "conts": [{"id": "P1", "prev": "S2", "evidence": "same red shirt"}],
})


def test_parse_full_json_screenplay():
    s = parse_clip_output(GOOD, duration_sec=30.0)
    assert s.parsed_ok and not s.issues
    assert {c.local_id for c in s.casts} == {"P1", "SW"}
    assert s.cast_decl("P1").name == "David"
    assert s.cast_decl("SW").is_wearer is True
    kinds = {(l.who, l.kind) for l in s.lines}
    assert ("P1", "speech") in kinds and ("P1", "action") in kinds
    assert (ENV_WHO, "environment") in kinds and ("SW", "action") in kinds
    assert s.nominations[0].pos == "left"
    assert s.voice_ranges[0].t0 == 3.3 and s.voice_ranges[0].t1 == 6.7
    assert s.cont["P1"] == "S2"


def test_markdown_fenced_json_tolerated():
    s = parse_clip_output("```json\n" + GOOD + "\n```", duration_sec=30.0)
    assert s.parsed_ok and len(s.casts) == 2


def test_invalid_json_flagged_not_crash():
    s = parse_clip_output("not json at all", duration_sec=30.0)
    assert not s.parsed_ok and any("JSON parse failed" in i for i in s.issues)


def test_bad_records_skipped_to_issues():
    raw = json.dumps({
        "casts": [{"id": "P1", "desc": "x"}],
        "lines": [
            {"t0": 0, "t1": 1, "who": "P9", "kind": "speech", "text": "hi"},   # cast never declared
            {"t0": 0, "t1": 1, "who": "P1", "kind": "bogus", "text": "hi"},    # bad kind
            {"t0": 0, "t1": 1, "who": "P1", "kind": "speech", "text": ""},     # empty text
        ],
        "noms": [{"id": "P1", "t": 1, "pos": "weird"}],
    })
    s = parse_clip_output(raw)
    assert s.parsed_ok
    assert len(s.lines) == 0 and len(s.issues) == 3
    assert s.nominations[0].pos == ""                     # an illegal pos is blanked out


def test_env_kind_forces_who_env():
    raw = json.dumps({"casts": [{"id": "P1", "desc": "x"}],
                      "lines": [{"t0": 0, "t1": 5, "who": "whatever", "kind": "environment",
                                 "text": "a dim hallway"}]})
    s = parse_clip_output(raw)
    assert s.lines[0].who == ENV_WHO


def test_env_who_with_wrong_kind_coerced_to_environment():
    raw = json.dumps({"casts": [{"id": "P1", "desc": "x"}],
                      "lines": [{"t0": 44, "t1": 51, "who": "ENV", "kind": "action",
                                 "text": "camera moves back to the entrance"}]})
    s = parse_clip_output(raw)
    assert len(s.lines) == 1 and s.lines[0].who == ENV_WHO and s.lines[0].kind == "environment"


def test_sw_referenced_but_undeclared_auto_added():
    raw = json.dumps({"casts": [{"id": "P1", "desc": "x"}],
                      "lines": [{"t0": 0, "t1": 1, "who": "SW", "kind": "speech", "text": "Okay."}],
                      "voices": [{"id": "SW", "t0": 0, "t1": 1}]})
    s = parse_clip_output(raw)
    sw = s.cast_decl("SW")
    assert sw is not None and sw.is_wearer is True
    assert len(s.lines) == 1 and s.lines[0].who == "SW" and len(s.voice_ranges) == 1


def test_left_center_right_still_accepted():
    raw = json.dumps({"casts": [{"id": "P1", "desc": "x"}],
                      "noms": [{"id": "P1", "t": 1, "pos": "center"}]})
    assert parse_clip_output(raw).nominations[0].pos == "center"


def test_build_prompt_json_schema_and_grid():
    prompt, images = build_clip_prompt(scene_setting="robot home assistant")
    assert images == []
    for token in ('"casts"', '"lines"', '"kind"', "left\" / \"center\" / \"right", "robot home assistant",
                  "SALIENCE", "JSON object"):
        assert token in prompt


def test_build_prompt_roster_images():
    cards = [{"cast_id": "S1", "name": "David", "desc": "red shirt",
              "face_b64": "AAAA", "body_b64": "BBBB"}]
    prompt, images = build_clip_prompt(roster_cards=cards)
    assert images == ["AAAA", "BBBB"] and "ROSTER" in prompt and "David" in prompt
