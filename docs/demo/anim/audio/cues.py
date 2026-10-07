"""Export the film's cue sheet from the story timeline (docs/demo/anim/story*.js, deterministic).

    python docs/demo/anim/audio/cues.py            # -> cues.json

Dialogue lines come straight from SAYS + story.json; SFX events from the screenplay, notebook and stage timeline constants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

JS = r"""
() => {
  const says = SAYS.map(s => ({ id: s.id, speaker: s.who, also: s.also || [], label: s.label || null, text: STORY.lines[s.id].en, start: s.t0, end: s.t0 + s.dur, offscreen: !!s.offscreen }));
  const ev = [];
  const add = (t, name, extra = {}) => ev.push({ t: +t.toFixed(3), name, ...extra });
  for (const [t, kind, who, text] of SCRIPT2) if (kind === 'act') add(t, 'script_act', { text });
  for (const cd of CARDS) { add(cd.t, 'card_appear', { card: cd.id }); for (const n of cd.name) add(n[0], n[1] === 'ink' ? 'name_sure' : 'name_guess', { card: cd.id, name: n[2] }); }
  for (const ch of CHIPS) for (const sg of ch.segs) { if (sg.from) add(sg.t0, 'chip_fly', { chip: ch.id, to: sg.card }); add(sg.t, 'chip_land', { chip: ch.id, card: sg.card }); }
  add(CONFLICT_T[0], 'alert_red'); add(REPAIR_T[0], 'repair_start'); add(REPAIR_T[1], 'repair_land'); add(FOLD_T[0], 'card_fold');
  add(SAVE_T, 'save_stamp'); add(WIPE0, 'wipe_start'); add(WIPE_END, 'wipe_end');
  add(T_DOOR, 'door_burst'); add(65.6, 'thunder'); add(66.15, 'thunder2'); add(72.6, 'lights_back'); add(T_FLIP[0], 'page_turn_cover'); add(T_FLIP2[0], 'page_turn_morning');
  add(59.0, 'ketchup_squirt'); add(47.8, 'sheets_toss'); add(49.4, 'ghost_poof'); add(146.0, 'wrapper_drop'); add(T_END, 'end_card');
  for (const h of HEIST2) add(h.grab, 'candy_grab'); add(94.9, 'antenna_flash'); add(80.2, 'lens_zoom');
  add(69.0, 'night_view_on'); add(72.5, 'night_view_off'); add(64.0, 'idea_spark'); add(60.3, 'sheet_pull_off'); add(63.9, 'sheet_hang'); add(70.0, 'sheet_drop'); add(70.9, 'sheet_pull_on'); add(72.4, 'sheet_pull_on2'); add(117.7, 'sheet_lift'); add(119.2, 'wrapper_peel');
  for (const t of [9.6, 22.5, 28.0, 36.4, 52.2, 77.2, 113.4, 124.8]) add(t, 'bell');
  for (const t of [30.5, 64.5, 84.6, 88.6, 92.0, 112.0, 120.0, 127.0]) add(t, 'snore');
  return { says, events: ev.sort((a, b) => a.t - b.t), duration: DUR, flips: { cover: T_FLIP, morning: T_FLIP2 } };
}
"""


def main() -> None:
    srv, port = render.serve()
    with sync_playwright() as p:
        b = p.chromium.launch(args=render.FLAGS)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(f"http://127.0.0.1:{port}/demo/anim/index.html")
        page.wait_for_function("window.ready === true", timeout=120000)
        data = page.evaluate(JS)
        b.close()
    (HERE / "cues.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(len(data["says"]), "lines,", len(data["events"]), "events")


if __name__ == "__main__":
    main()
