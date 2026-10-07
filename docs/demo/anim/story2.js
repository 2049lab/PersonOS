/* story2.js - everything after clip 1 (t >= 36): Mum's rule + costumes + ketchup, the blackout and the correction, the heist and
 * the end-of-night rewrite, the morning recall. Loaded after story.js; it overrides overlayStage / worldNight / drawPanel for t >= 36
 * and leaves clip 1 untouched (the old versions stay as ...Old). */
const T2 = 36.0;
const SAVE_T = 95.6, WIPE0 = 98.0, WIPE_DT = .15;
const FOLD_T = [56.0, 56.9], REPAIR_T = [81.4, 82.2], CONFLICT_T = [76.9, 81.4];
{
  for (let i = SAYS.length - 1; i >= 0; i--) if (SAYS[i].t0 >= 35.5) SAYS.splice(i, 1);
  SAYS.push(
    { id: 'mumrule', who: 'mum', t0: 37.0, dur: 3.4 }, { id: 'groan', who: 'mia', also: ['leo', 'sam'], label: 'Mia, Leo, Sam', t0: 41.0, dur: 2.6 }, { id: 'costume', who: 'mia', t0: 45.8, dur: 2.6 },
    { id: 'boo', who: 'sam', t0: 52.0, dur: 2.6 }, { id: 'arf', who: 'biscuit', t0: 55.2, dur: 2.5 }, { id: 'ketchup', who: 'leo', t0: 58.0, dur: 3.2 }, { id: 'sticky', who: 'leo', t0: 61.8, dur: 2.8 },
    { id: 'eek', who: 'sam', t0: 66.2, dur: 2.6 }, { id: 'oj', who: 'leo', t0: 76.2, dur: 2.6, offscreen: true }, { id: 'giggle', who: 'mia', t0: 79.4, dur: 2.6 },
    { id: 'oneforme', who: 'mia', label: 'Mia (whisper)', t0: 84.2, dur: 3.0 },
    { id: 'who1', who: 'mum', t0: 113.8, dur: 2.6 }, { id: 'sheet', who: 'mum', t0: 117.0, dur: 3.4 }, { id: 'wasnt', who: 'leo', t0: 121.0, dur: 2.5 }, { id: 'notmeeither', who: 'sam', t0: 124.5, dur: 2.6 },
    { id: 'biscuit?', who: 'mia', t0: 127.7, dur: 2.8 }, { id: 'woof?', who: 'biscuit', t0: 131.1, dur: 2.5 }, { id: 'mum2', who: 'mum', t0: 134.2, dur: 2.8 }, { id: 'answer2', who: 'pebble', t0: 137.6, dur: 7.0 },
    { id: 'happy', who: 'mia', t0: 148.6, dur: 2.8 }, { id: 'woof', who: 'biscuit', t0: 152.0, dur: 2.5 });
}
const H2 = { mia: [670, 800], leo: [820, 880], sam: [990, 850], dogBasket: [255, 935], bowl: [560, 795], pile: 480 };
const MORN = { mum: [430, 900], leo: [620, 850], sam: [790, 872], mia: [955, 785], pebble: [1165, 915], dog: [255, 935] };
const SOFA_BOX = [300, 440, 800, 704];
const walkPh = (t, t0, rate = 1.6) => (t - t0) * rate;
const pulseK = (t, t0, d = 6) => t >= t0 ? Math.exp(-(t - t0) * d) : 0;
const OLD2 = { mia: STATES.mia, leo: STATES.leo, sam: STATES.sam, pebble: STATES.pebble, biscuit: STATES.biscuit };
const sneak = (o) => { o.stepH = .95; o.stride = 1.15; o.lift = .05; };

/* ───────── Mum ───────── */
function plateHook(s, sw) { push(); translate(.1 * s, -.35 * s); paint(ellPts(0, 0, 1.5 * s, .32 * s, 14), { wash: '#FFFDF5', ink: PAL.ink, sw: sw * .6 });
  for (const dx of [-.5, .5]) { paint(rrPts(dx * s - .6 * s, -.55 * s, 1.2 * s, .42 * s, .2 * s), { wash: '#D9A066', ink: PAL.ink, sw: sw * .5 }); inkLine([[dx * s - .45 * s, -.38 * s], [dx * s + .45 * s, -.38 * s]], sw * .5, '#E8C22C', 'ink', .4); }
  pop(); }
function sheetHook(s, sw) { push(); translate(.2 * s, .2 * s); paint([[-.7 * s, -.2 * s], [.5 * s, -.7 * s], [1.0 * s, .3 * s], [.4 * s, 1.6 * s], [-.4 * s, 2.5 * s], [-.9 * s, 1.2 * s]], { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw * .7 });
  paint(ellPts(.1 * s, 1.2 * s, .65 * s, .55 * s, 12), { wash: PAL.ketchup, ink: '#9c2f25', sw: sw * .4 });
  if (T >= 116.5) { const peeled = T >= 119.2, n = peeled ? 2 : 3; for (let i = 0; i < n; i++) wrapperShape(((i - 1) * .5 + .1) * s + Math.sin(T * 5 + i) * .08 * s, (1.2 + .9 + .12 * i) * s + Math.abs(Math.sin(T * 6 + i)) * .1 * s, Math.sin(T * 4 + i) * .5, [PAL.pink, PAL.candle, PAL.mint][i], .55 * s / 34);
    if (peeled) { const u = seg(T, 119.2, 120.4); wrapperShape(-1.6 * s - u * .9 * s, (1.4 - u * .6) * s, u * 3, PAL.pink, .6 * s / 34); } }
  pop(); }
function mumS(t) {
  if (t < T2 || (t >= 43.2 && t < T_C5)) return null; const o = { who: 'mum', blush: true }; let x, y = 805;
  if (t < 43.2) {
    if (t < 37.2) { x = lerp(110, 330, ease(seg(t, T2 + .1, 37.2))); y = 803; o.walk = walkPh(t, T2, 1.2); o.handR = plateHook; o.aR = .9; o.eyes = 'dot'; o.mouth = 'smile'; o.turn = .6; }
    else { x = 330; o.turn = 1; o.eyes = 'dot'; o.mouth = 'smile'; const place = seg(t, 38.2, 38.9);
      if (place < 1) { o.handR = plateHook; o.aR = lerp(.9, .4, easeOut(place)); } else o.aR = lerp(.4, -1.1, seg(t, 38.9, 39.4));
      if (t > 38.2 && t < 38.7) o.sq = -.05 * Math.sin(seg(t, 38.2, 38.7) * Math.PI);
      if (t >= 41.3) { x = lerp(330, 110, ease(seg(t, 41.6, 43.0))); o.walk = walkPh(t, 41.6, 1.2); o.turn = -.7; o.eyes = 'dot'; o.mouth = 'flat'; if (t > 42.9) return null; } }
  } else {
    const inT = seg(t, 112.0, 113.4); x = lerp(110, MORN.mum[0], ease(inT)); y = MORN.mum[1] - 6 * (1 - inT); o.handL = cauldronHand; o.aL = .7; o.eyes = 'dot'; o.mouth = 'flat'; o.turn = .5;
    if (inT < 1) o.walk = walkPh(t, 112.0, 1.2);
    else {
      o.aR = -.55 + .05 * Math.sin(t * 2); o.aL = .5; o.brows = 'angry'; o.turn = .6;
      if (t >= 113.8 && t < 116.4) { o.rot = -.02 * Math.sin(t * 22); o.brows = 'up'; o.emote = t > 113.9 && t < 114.7 ? 'anger' : null; o.emoteK = seg(t, 113.9, 114.1); }
      if (t >= 116.4 && t < 117.9) { const k = seg(t, 116.5, 117.5); o.sq = -.1 * Math.sin(k * Math.PI); o.aR = lerp(.15, -1.0, easeOut(k)); }                    // bends to pick the stained sheet up
      if (t >= 117.7 && t < 122.6) { o.handR = sheetHook; o.aR = lerp(.5, 1.15, easeOut(seg(t, 117.7, 118.5))); o.turn = 1; o.brows = 'angry'; o.mouth = 'O'; }
      if (t >= 122.6 && t < 134.0) { o.handR = sheetHook; o.aR = .85; o.turn = .7; o.brows = 'angry'; o.eyes = 'dot'; }
      if (t >= 134.0 && t < 146.5) { o.handR = sheetHook; o.aR = .6; o.turn = lerp(.7, .3, easeOut(seg(t, 134.0, 134.6))); o.look = [1, 0]; o.brows = 'up'; }
      if (t >= 146.6) { o.handR = sheetHook; o.aR = .6; const k = easeOut(seg(t, 147.0, 147.5)); o.turn = lerp(.3, 1, k); o.eyes = 'wide'; o.look = [1, .1]; o.brows = 'up'; o.mouth = 'o'; } }
  }
  talk(o, 'mum', t); return { form: 'kid', x, y, o, s: sFor(y) };
}

/* ───────── Mia ───────── */
function pocketHook(bulge) { return (s, sw) => { const px = .95 * s, py = -(KID.mia.legH + KID.mia.torsoH * .32) * s; paint(ellPts(px, py, .85 * s, .75 * s * (.8 + .2 * bulge), 12), { wash: '#F6A257', ink: PAL.ink, sw: sw * .6 });
  for (const [dx, c_] of [[-.2, PAL.pink], [.2, PAL.mint], [0, PAL.candle]]) blob(px + dx * s, py - .55 * s * bulge, .3 * s, c_);
  paint([[px - .1 * s, py - .6 * s], [px + .35 * s, py - 1.0 * s], [px + .6 * s, py - .55 * s]], { wash: PAL.pink, ink: PAL.ink, sw: sw * .4 }); }; }
function miaGhostS(t) {                                                              // costumes, blackout, swap, heist
  const land = t - 49.4, o = { kind: 'mia', ph: 0, lift: .22 + .2 * Math.sin(t * 2.6), walk: undefined, look: [Math.sin(t * 1.1) * .9, -.2], rot: .04 * Math.sin(t * 2.1), eyes: land < .5 ? 'wide' : 'dot' };
  o.sq = -.18 * Math.exp(-Math.max(0, land) * 7) * Math.cos(Math.max(0, land) * 22);
  let x = H2.mia[0] - 60, y = H2.mia[1];
  if (t < 52.0) { const k = ease(seg(t, 49.8, 50.8)); x = lerp(520, H2.mia[0], k); if (k > 0 && k < 1) { o.walk = walkPh(t, 49.8, 1.7); o.lift = .1; } o.eyes = land < .5 ? 'wide' : 'dot'; }
  else if (t < 65.6) { x = H2.mia[0] + Math.sin(t * .8) * 6; if (t > 58.2) o.look = [lerp(.5, -1, seg(t, 59.4, 61.2)), -.1]; if (t > 60.6) { o.eyes = 'sly'; o.look = [-1, Math.sin((t - 60.6) * 3.4) > 0 ? -.35 : .45]; } if (t > 64.0) o.look = [-1, .45]; }
  else if (t < 72.6) { x = H2.mia[0]; o.eyes = 'wide'; o.lift = .15; o.look = [Math.sin(t * 3), -.3];                    // blackout: eyes only, then a night-view tiptoe to the sofa arm
    if (t >= 68.9 && t < 70.0) { x = lerp(H2.mia[0], 450, ease(seg(t, 68.9, 70.0))); o.walk = walkPh(t, 68.9, 1.7); o.lift = .1; }
    if (t >= 70.0) { x = t < 70.8 ? lerp(450, 340, ease(seg(t, 70.0, 70.8))) : t < 71.6 ? 340 : lerp(340, 560, ease(seg(t, 71.6, 72.5))); if (t < 70.8 || t >= 71.6) { o.walk = walkPh(t, 70.0, 1.8); o.lift = .08; } o.eyes = 'sly'; if (t >= 71.0) { o.stain = true; o.sheetK = lerp(1, 1.28, seg(t, 71.0, 71.6)); } } }
  else { o.stain = true; o.sheetK = 1.28; o.eyes = 'sly'; sneak(o);
    if (t < 73.4) { x = 560; o.eyes = 'sly'; }
    else if (t < 75.2) { const k = ease(seg(t, 73.4, 75.2)); x = lerp(560, H2.mia[0], k); y = lerp(H2.mia[1] + 10, H2.bowl[1], k); o.walk = walkPh(t, 73.4, 1.4); o.rot = -.06 * Math.sin(k * 9); o.look = [-1, 0]; }
    else if (t < 82.4) { x = H2.mia[0]; y = H2.bowl[1]; o.reach = [1.0 + .15 * Math.sin(t * 8), 2.3]; o.lift = .06; o.eyes = 'happy';
      if (t > 76.6 && t < 79.4) { o.reach = [.65, 1.7]; o.eyes = 'wide'; o.emote = '!'; o.emoteK = seg(t, 76.6, 76.8); o.sq = .1 * pulseK(t, 76.6, 8) * Math.cos((t - 76.6) * 24); o.look = [-1, 0]; }
      if (t >= 79.4) { o.reach = undefined; o.eyes = 'happy'; o.sq = .1 * Math.sin(t * 24) * .6; o.dy = .15 * Math.abs(Math.sin(t * 12)); o.look = [-.6, 0]; } }
    else if (t < 83.8) { const k = ease(seg(t, 82.4, 83.6)); x = lerp(H2.mia[0], 800, k); y = lerp(H2.bowl[1], 870, k); o.walk = walkPh(t, 82.4, 1.6); o.look = [1, 0]; }
    else { const h = HEIST2.find(h => t < h.done + .02) || HEIST2[2]; x = 800; y = 870; o.eyes = 'happy';
      if (t < h.grab - .05 && t >= h.go) { const u = easeOut(seg(t, h.go, h.grab - .05)); x = lerp(800, H2.mia[0], u); y = lerp(870, H2.bowl[1], u); o.walk = walkPh(t, h.go, 1.4); o.rot = -.07 * Math.sin(u * Math.PI * 3); o.look = [-1, 0]; }
      else if (t >= h.grab - .05 && t < h.back) { x = H2.mia[0]; y = H2.bowl[1]; o.reach = [1.0 + .2 * Math.sin(t * 20), 2.3]; o.sq = -.08 * Math.sin(seg(t, h.grab - .05, h.back) * Math.PI) + .06 * Math.sin(t * 25); }
      else if (t >= h.back && t < h.done) { const u = ease(seg(t, h.back, h.done)); x = lerp(H2.mia[0], 800, u); y = lerp(H2.bowl[1], 870, u); o.walk = walkPh(t, h.back, 1.4); o.rot = .07 * Math.sin(u * Math.PI * 3); o.look = [1, 0]; }
      if (t >= 92.6) { x = 800; y = 870; o.eyes = 'sly'; }
      o.wrap = HEIST2.filter(h_ => t > h_.grab + .45).length; } }
  if (speakingNow('mia', t)) o.sq = (o.sq || 0) + .04 * Math.sin(t * 22);
  return { form: 'ghost', x, y, o, s: sFor(y) * .96 };
}
const HEIST2 = [{ go: 84.0, grab: 85.2, back: 85.6, done: 87.2 }, { go: 87.0, grab: 87.9, back: 88.3, done: 89.8 }, { go: 89.8, grab: 90.6, back: 91.0, done: 92.4 }];
function miaMorning(t) {
  const o = { who: 'mia', blush: true, draw: t >= 146.6 ? undefined : null }; const inT = seg(t, 112.8, 114.4), x = lerp(1560, MORN.mia[0], ease(inT)), y = MORN.mia[1];
  o.draw = pocketHook(1); o.aL = -.9; o.aR = -.5; o.turn = .8; o.eyes = 'dot'; o.mouth = 'wobble'; o.look = [1, .35];
  if (inT < 1) { o.walk = walkPh(t, 112.8, 1.4); o.mouth = 'wobble'; }
  else { o.sq = -.02 * Math.sin(t * 5);
    if (t >= 117.0 && t < 121.4) { o.eyes = 'dot'; o.look = [1, .4]; o.mouth = 'wobble'; o.emote = t > 119 && t < 120.2 ? 'sweat' : null; o.emoteK = seg(t, 119, 119.2); }
    if (t >= 127.7 && t < 130.6) { const k = seg(t, 127.7, 128.1); o.aL = lerp(-.9, 1.1, easeOut(k)); o.eyes = 'wide'; o.mouth = 'grin'; o.look = [-1, .2]; o.turn = -.8; }
    if (t >= 131.0 && t < 134) { o.eyes = 'sly'; o.emote = t > 131.3 && t < 132.4 ? 'sweat' : null; o.emoteK = seg(t, 131.3, 131.5); }
    if (t >= 146.0) { const k = seg(t, 146.0, 146.5); o.eyes = 'wide'; o.mouth = 'wobble'; o.brows = 'worried'; o.turn = lerp(.8, 0, k); o.look = [0, 0]; o.emote = 'sweat'; o.emoteK = k; o.sq = .1 * pulseK(t, 146.0, 6) * Math.cos((t - 146) * 20);
      if (t > 148.6) { o.eyes = 'happy'; o.mouth = 'grin'; o.brows = null; o.emote = null; o.aR = .9 + Math.sin(t * 12) * .3; } } }
  talk(o, 'mia', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function miaKidHeist(t) {                                                            // she drops the sheet and tiptoes off with a bulging pocket
  const k = seg(t, 93.2, 96.0), o = { who: 'mia', blush: true, draw: pocketHook(1) }; let x = 800 - 340 * ease(seg(t, 94.0, 96.4)), y = 870 - 70 * ease(seg(t, 94.0, 96.4));
  o.walk = t > 94.0 ? walkPh(t, 94.0, 1.1) : undefined; o.stepH = 1; o.turn = -.7; o.eyes = 'sly'; o.mouth = 'grin'; o.look = [-.6, 0]; o.aL = -.9; o.aR = -.4; o.sq = t < 93.8 ? .1 * pulseK(t, 93.2, 7) * Math.cos((t - 93.2) * 20) : 0;
  if (t > 96.4) return null; return { form: 'kid', x, y, o, s: sFor(y) };
}
const miaS = (t) => t < T2 ? OLD2.mia(t) : t < 45.4 ? miaS1(t) : t < 49.4 ? miaS2(t) : t < 93.2 ? miaGhostS(t) : t < T_C5 ? miaKidHeist(t) : miaMorning(t);
function miaS1(t) {                                                                  // clip 2 opening: the same Mia, still at the bowl
  const o = { who: 'mia', blush: true, turn: -.7, aL: 1.0 + .1 * Math.sin(t * 14), eyes: 'happy', look: [-.8, -.2], mouth: 'grin' }; o.sq = -.03 * Math.sin(t * 9);
  if (t < 37.0) { o.turn = -.9; } else { o.turn = lerp(-.9, -1, .5); o.eyes = 'dot'; o.aL = -1.1; o.look = [-1, 0]; o.mouth = 'grin'; }
  if (t >= 37.0 && t < 40.6) { o.look = [-1, 0]; o.mouth = 'smile'; }
  if (t >= 41.0 && t < 43.6) { o.eyes = 'closed'; o.mouth = 'wobble'; o.brows = 'worried'; o.aL = o.aR = -.9; o.rot = .03 * Math.sin(t * 10); }
  if (t >= 43.6) { o.eyes = 'sly'; o.look = [-1, .4]; o.mouth = 'smirk'; o.brows = null; }
  talk(o, 'mia', t); return { form: 'kid', x: 560, y: H2.mia[1], o, s: sFor(H2.mia[1]) };
}
function miaS2(t) {                                                                  // "Costume time!": pulls the sheets out from behind the sofa and tosses them up
  const o = { who: 'mia', blush: true }; let x = 560;
  if (t < 46.4) { x = lerp(560, H2.pile + 40, ease(seg(t, 45.4, 46.4))); o.run = walkPh(t, 45.4, 2.8); o.eyes = 'happy'; o.mouth = 'grin'; o.turn = -.9; }
  else if (t < 47.6) { x = H2.pile + 40; const k = easeOut(seg(t, 46.5, 47.2)); o.aL = o.aR = lerp(-1.25, 1.5, k); o.draw = t > 47.0 ? bundleDraw : undefined; o.eyes = 'happy'; o.mouth = 'grin'; o.sq = t < 47.0 ? -.1 * seg(t, 46.5, 47.0) : .1 * pulseK(t, 47.0, 8); o.turn = 0; }
  else { x = H2.pile + 40; const u = seg(t, 47.6, 48.2); o.aL = o.aR = lerp(1.4, 1.65, u); o.draw = t < 48.0 ? bundleDraw : undefined; o.eyes = 'happy'; o.mouth = 'laugh'; o.dy = Math.sin(seg(t, 47.8, 48.9) * Math.PI) * .9; o.turn = 0; if (t > 49.0) o.eyes = 'closed'; }
  talk(o, 'mia', t); return { form: 'kid', x, y: 800, o, s: sFor(800) };
}

/* ───────── Leo ───────── */
function leoKid(t) {
  const o = { who: 'leo', blush: true }; let x = lerp(720, H2.leo[0], ease(seg(t, T2, T2 + 1.2))), y = H2.leo[1]; o.turn = -.4; o.eyes = 'dot'; o.look = [-.4, 0]; o.mouth = 'smile'; o.sq = -.02 * Math.sin(t * 4);
  if (t >= 37.0 && t < 40.6) { o.look = [-1, 0]; o.eyes = 'dot'; o.mouth = 'flat'; }
  if (t >= 41.0 && t < 43.6) { o.eyes = 'closed'; o.mouth = 'wobble'; o.brows = 'worried'; o.aL = o.aR = -.8; o.rot = -.03 * Math.sin(t * 10); }
  if (t >= 43.6) { o.look = [-.3, .2]; o.mouth = 'smile'; }
  if (t >= 44.0 && t < 45.4) o.look = [-1, .4];
  if (t >= 47.0) { o.look = [-.6, 0]; o.eyes = 'wide'; if (t > 48.2) { o.eyes = 'closed'; o.aL = o.aR = lerp(-1.25, 1.6, easeOut(seg(t, 48.3, 48.8))); o.mouth = 'grin'; o.dy = .7 * Math.sin(seg(t, 48.8, 49.4) * Math.PI); } }
  talk(o, 'leo', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function leoGhostS(t) {                                                              // ghost: the ketchup scene (49.4-59.8) and after the lights come back (72.5-75.4)
  const land = t - 49.4, o = { kind: 'leo', ph: 1.7, lift: .22 + .2 * Math.sin(t * 2.6 + 1.7), walk: undefined, look: [Math.sin(t * 1.1 + 1) * .9, -.2], rot: .04 * Math.sin(t * 2.1 + 1.7), eyes: land < .5 ? 'wide' : 'happy' };
  o.sq = -.18 * Math.exp(-Math.max(0, land) * 7) * Math.cos(Math.max(0, land) * 22);
  let x = H2.leo[0] + Math.sin(t * .8 + 1) * 6, y = H2.leo[1];
  if (t < 72.5) { const k = ease(seg(t, 58.0, 59.4)); x = lerp(H2.leo[0], 470, k); y = lerp(H2.leo[1], 830, k); if (k > 0 && k < 1) { o.walk = walkPh(t, 58.0, 1.7); o.stride = .9; o.lift = .1; }
    if (t >= 59.4) { o.sq = .16 * pulseK(t, 59.4, 7) * Math.cos((t - 59.4) * 22); o.rot = -.12 * pulseK(t, 59.4, 6); o.eyes = 'wide'; o.emote = t < 60.2 ? '!' : null; o.emoteK = seg(t, 59.4, 59.6); o.stain = true; } }
  else { o.stain = false; o.sheetK = .88; x = 450; y = 860; o.eyes = 'happy'; o.look = [-1, 0];
    if (t < 73.2) { o.armsUp = 1.3 * seg(t, 72.7, 73.0); o.wave = .1; o.eyes = 'closed'; }
    if (t >= 73.2) { const k = ease(seg(t, 73.2, 75.6)); x = lerp(450, 110, k); y = lerp(860, 770, k); o.walk = walkPh(t, 73.2, 1.9); o.stride = .8; o.lift = 0; o.eyes = 'happy'; if (t >= 75.4) return null; } }
  if (speakingNow('leo', t)) o.sq = (o.sq || 0) + .04 * Math.sin(t * 22);
  return { form: 'ghost', x, y, o, s: sFor(y) * .96 };
}
function leoOffS(t) {                                                                // Leo without his sheet (59.8-72.5): hangs it on the sofa arm, then gropes for it in the dark
  const o = { who: 'leo', blush: true, turn: -.4, eyes: 'dot', mouth: 'flat', look: [-.6, .2] }; let x = 470, y = 830;
  if (t < 62.6) { const k = easeOut(seg(t, 59.8, 60.6)); o.aR = lerp(-1.25, 1.0, k); o.handR = sheetHook; o.eyes = t < 60.8 ? 'wide' : 'dot'; o.mouth = t < 60.8 ? 'O' : 'wobble'; o.brows = 'worried'; o.sq = .1 * pulseK(t, 59.8, 6) * Math.cos((t - 59.8) * 20); o.look = [.2, .4]; if (t > 61.0) o.aR = 1.0 + .05 * Math.sin(t * 6); }
  else if (t < 63.8) { const k = ease(seg(t, 62.6, 63.7)); x = lerp(470, 410, k); y = lerp(830, 860, k); o.walk = walkPh(t, 62.6, 1.7); o.handR = sheetHook; o.aR = 1.0; o.turn = -.8; o.brows = 'worried'; o.mouth = 'O'; }
  else if (t < 64.6) { x = 410; y = 860; o.turn = -.9; const k = seg(t, 63.8, 64.2); o.aR = lerp(1.0, 1.55, k); if (t < 64.2) o.handR = sheetHook; o.eyes = 'dot'; o.mouth = 'o'; }
  else if (t < 70.9) { x = 410; y = 860; o.turn = -.5; o.aR = .4 + .25 * Math.sin(t * 5); o.mouth = 'smile'; o.look = [-.8, 0]; }
  else if (t < 72.0) { x = 410 - 60 * Math.sin(seg(t, 70.9, 72) * Math.PI * .9); y = 860; o.turn = -.9; o.aR = 1.3 + .15 * Math.sin(t * 9); o.aL = .9; o.eyes = 'wide'; o.mouth = 'o'; o.look = [-1, 0]; }
  else { const k = ease(seg(t, 72.0, 72.5)); x = lerp(412, 450, k); y = 860; o.walk = walkPh(t, 72.0, 2.0); o.turn = .6; o.aR = .8; o.aL = .8; }
  talk(o, 'leo', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function leoMorning(t) {
  const o = { who: 'leo', blush: true }; const inT = seg(t, 112.4, 114.0), x = lerp(1560, MORN.leo[0], ease(inT)), y = MORN.leo[1]; o.turn = -.5;
  if (inT < 1) { o.walk = walkPh(t, 112.4, 1.3); o.eyes = 'closed'; o.mouth = 'O'; o.aL = o.aR = 1.0 + Math.sin(t * 5) * .1; }
  else { o.eyes = 'dot'; o.mouth = 'flat'; o.look = [-.6, 0]; o.sq = -.02 * Math.sin(t * 4.2); o.turn = -.3;
    if (t >= 113.8 && t < 117) { o.eyes = 'wide'; o.look = [-1, 0]; }
    if (t >= 118.0 && t < 121.0) { o.eyes = 'wide'; o.mouth = 'o'; o.brows = 'worried'; o.aL = o.aR = lerp(-1.25, 1.3, easeOut(seg(t, 118.2, 118.6))); o.sq = .04 * Math.sin(t * 12); o.look = [-1, 0]; }
    if (t >= 121.0 && t < 124.0) { o.eyes = 'closed'; o.mouth = 'O'; o.brows = 'angry'; o.rot = .05 * Math.sin(t * 14); o.aL = o.aR = 1.3; }
    if (t >= 124.0) { o.aL = o.aR = -1.1; o.eyes = 'dot'; }
    if (t >= 146.5) { const k = easeOut(seg(t, 147.4, 147.9)); o.turn = lerp(-.3, 1, k); o.eyes = 'wide'; o.mouth = 'o'; o.look = [1, 0]; } }
  talk(o, 'leo', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
/* ───────── Sam ───────── */
function samGhostS(t) {
  const land = t - 49.4, o = { kind: 'sam', ph: 3.4, lift: .22 + .2 * Math.sin(t * 2.6 + 3.4), walk: undefined, look: [Math.sin(t * 1.1 + 2) * .9, -.2], rot: .04 * Math.sin(t * 2.1 + 3.4), eyes: land < .5 ? 'wide' : 'happy' };
  o.sq = -.18 * Math.exp(-Math.max(0, land) * 7) * Math.cos(Math.max(0, land) * 22);
  let x = H2.sam[0] + Math.sin(t * .8 + 2) * 6, y = H2.sam[1];
  if (t >= 51.2 && t < 54.9) { o.look = [-1, .2]; } if (t >= 52.0 && t < 54.6) { o.armsUp = 1.2; o.wave = .5; o.eyes = 'wide'; o.lift += Math.abs(Math.sin((t - 52.0) * 5)) * .6; }
  if (t >= 55.2 && t < 58) { o.eyes = 'happy'; o.look = [-1, 0]; o.sq += .05 * Math.sin(t * 12); }
  if (t >= 65.6) { o.eyes = 'wide'; o.lift = .2; o.look = [Math.sin(t * 3), -.3]; }
  if (t >= 72.6) { o.eyes = 'happy'; o.look = [-1, 0]; }
  if (t >= 77.0) { const k = ease(seg(t, 77.0, 79.2)); x = lerp(H2.sam[0], 1480, k); o.walk = walkPh(t, 77.0, 1.4); o.eyes = 'happy'; o.look = [1, 0]; if (k >= 1) return null; }
  if (speakingNow('sam', t)) o.sq = (o.sq || 0) + .04 * Math.sin(t * 22);
  return { form: 'ghost', x, y, o, s: sFor(y) * .96 };
}
function samKid(t) {
  const o = { who: 'sam', blush: true }; let x = lerp(980, H2.sam[0], ease(seg(t, T2, T2 + 1.0))), y = H2.sam[1]; o.turn = -.6; o.eyes = 'closed'; o.mouth = 'laugh'; o.dy = .2 * Math.abs(Math.sin(t * 7)); o.sq = -.04 * Math.sin(t * 7); o.rot = .03 * Math.sin(t * 7);
  if (frac(t * 2.2) < .1) { o.emote = 'note'; o.emoteK = frac(t * 2.2) * 10; }
  if (t >= 37.0 && t < 40.6) { o.eyes = 'dot'; o.mouth = 'smile'; o.dy = 0; o.look = [-1, 0]; o.rot = 0; o.sq = 0; o.emote = null; }
  if (t >= 41.0 && t < 43.6) { o.eyes = 'closed'; o.mouth = 'wobble'; o.brows = 'worried'; o.aL = o.aR = -.9; o.dy = 0; o.emote = null; }
  if (t >= 48.0) { o.eyes = 'closed'; o.aL = o.aR = lerp(-1.25, 1.6, easeOut(seg(t, 48.4, 48.9))); o.mouth = 'laugh'; o.dy = .7 * Math.sin(seg(t, 48.9, 49.4) * Math.PI); }
  talk(o, 'sam', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function samMorning(t) {
  const o = { who: 'sam', blush: true }; const inT = seg(t, 113.0, 114.6), x = lerp(1560, MORN.sam[0], ease(inT)), y = MORN.sam[1]; o.turn = -.5;
  if (inT < 1) { o.walk = walkPh(t, 113.0, 1.3); o.eyes = 'closed'; o.mouth = 'flat'; o.aL = o.aR = .5 + Math.sin(t * 9) * .2; }
  else { o.eyes = 'dot'; o.mouth = 'smile'; o.look = [-.8, 0]; o.dy = .08 * Math.abs(Math.sin(t * 4));
    if (t >= 124.5 && t < 127.1) { o.eyes = 'closed'; o.brows = 'angry'; o.rot = .05 * Math.sin(t * 14); o.dy = 0; o.mouth = 'O'; if (frac(t * 3) < .12) { o.emote = 'note'; o.emoteK = frac(t * 3) * 8; } }
    if (t >= 147.0 && t < 160) { const k = easeOut(seg(t, 147.8, 148.3)); o.turn = lerp(-.5, 1, k); o.eyes = 'wide'; o.mouth = 'o'; o.look = [1, 0]; o.dy = 0; } }
  talk(o, 'sam', t); return { form: 'kid', x, y, o, s: sFor(y) };
}

/* ───────── Biscuit ───────── */
function biscuitS(t) {
  const [bx, by] = H2.dogBasket;
  if (t < T2) return OLD2.biscuit(t);
  if (t < 44.0) return { form: 'dog', x: bx, y: by - 10, o: { sleep: true }, s: 36 };
  if (t < 45.7) { const u = seg(t, 44.0, 45.7), wake = t < 44.7; return { form: 'dog', x: lerp(bx, H2.pile, easeOut(seg(t, 44.6, 45.7))), y: lerp(by - 10, 716, easeOut(seg(t, 44.6, 45.7))), o: wake ? { sleep: false, eyes: t < 44.3 ? 'closed' : 'dot', wag: 0, sq: .12 * pulseK(t, 44.0, 6) * Math.cos((t - 44) * 22), tongue: t > 44.4, earUp: t > 44.2 } : { trot: (t - 44.6) * 2.4, wag: .8, tongue: true, eyes: 'dot', flip: false, earUp: false }, s: 36 }; }
  if (t < 51.0) return { form: 'dog', x: H2.pile, y: 716, o: { sleep: false, eyes: 'dot', wag: 1.0, tongue: true, hidden: true }, s: 36 };
  if (t < 63.0) { const k = ease(seg(t, 51.0, 53.4)), x = lerp(H2.pile, 300, k), y = lerp(716, 905, k); const pop = pulseK(t, 55.2, 5);
    return { form: 'dogghost', x, y, o: { kind: 'dog', walk: k > 0 && k < 1 ? (t - 51) * 2.2 : undefined, wag: 1.0 + pop, lift: .18 + .1 * Math.sin(t * 3) + .5 * pop, flip: false, ph: 3, eyes: t < 52 ? 'wide' : 'happy', look: [-1, 0], sq: t > 55.2 ? .18 * pulseK(t, 55.2, 7) * Math.cos((t - 55.2) * 26) : 0, emote: t > 55.2 && t < 56.1 ? '!' : null, emoteK: seg(t, 55.2, 55.4) }, s: 36 }; }
  if (t < 71.0) { const k = ease(seg(t, 63.0, 64.4)); return { form: 'dogghost', x: lerp(300, bx + 20, k), y: lerp(905, by - 10, k), o: { kind: 'dog', walk: k > 0 && k < 1 ? (t - 63) * 2.2 : undefined, wag: k < 1 ? .8 : 0, lift: k < 1 ? .15 : 0, flip: false, ph: 3, eyes: k < 1 ? 'happy' : 'closed', sleep: k >= 1, sq: 0 }, s: 36 }; }
  if (t < 97.5) return { form: 'dog', x: bx, y: by - 10, o: { sleep: true }, s: 36 };
  if (t < 131.0) return { form: 'dog', x: bx, y: by - 10, o: { sleep: true }, s: 36 };
  if (t < 134.0) { const up = t < 133.4; return { form: 'dog', x: bx, y: by - 10, o: { sleep: false, eyes: 'wide', wag: 0, earUp: up, tongue: false, dy: t < 131.6 ? .35 * Math.abs(Math.sin((t - 131.1) * 14)) : 0 }, s: 36 }; }
  const wagK = t >= 152.0 ? 1.9 : t >= 148 ? 1.4 : .9;
  return { form: 'dog', x: bx, y: by - 10, o: { sleep: false, wag: wagK, tongue: true, eyes: t >= 146.6 ? 'wide' : 'dot', earUp: t >= 147.8 && t < 148.8, flip: false }, s: 36 };
}

/* ───────── Pebble ───────── */
const GL2 = [10.0, 10.3, 10.6, 17.0, 23.4, 26.9, 30.0, 33.3, 33.6, 37.3, 49.7, 51.1, 56.0, 59.8, 70.0, 73.9, 76.2, 81.4, 95.6];
function pebbleS(t) {
  if (t < T2) return OLD2.pebble(t);
  let x = 1175, y = 915; const o = { armL: .4, armR: .4, rolling: false, roll: 0, pupil: .5, glow: .45 };
  const tk = SAYS.find(s_ => s_.who !== 'pebble' && !s_.offscreen && t >= s_.t0 && t <= s_.t0 + s_.dur + .5);
  let tx = 900; if (tk) { const st = STATES[tk.who](t); if (st) tx = st.x; } o.look = [clamp((tx - x) / 350, -1, 1), -.1];
  if (t >= T_FLIP2[0] - .2) { x = MORN.pebble[0]; y = MORN.pebble[1] - 15; o.look = [Math.sin(t * 1.2) * .4, 0]; }
  if (GL2.some(g => t > g && t < g + 1.1) && t < T_FLIP2[0]) o.look = [1, -.35];
  if (t > 65.6 && t < 72.6) { o.glow = .95; o.pupil = .78; o.look = [Math.sin(t * 2) * .6, -.1]; }
  const pu = [37.3, 56.0, 81.4].find(v => t > v && t < v + 1.2); if (pu) o.glow = .5 + .6 * Math.exp(-(t - pu) * 3.5);
  if (t >= 94.9 && t < 97.0) { o.glow = 1; o.pupil = .8; o.emote = t < 95.6 ? '!' : null; o.emoteK = seg(t, 94.9, 95.1); o.dy = .25 * pulseK(t, 94.9, 6) * Math.abs(Math.sin((t - 94.9) * 16)); }
  if (t >= 137.0 && t < 146.6) { const u = easeOut(seg(t, 137.2, 138.4)); x = lerp(MORN.pebble[0], MORN.pebble[0] - 25, u); o.armR = .9 + .15 * Math.sin(t * 8); o.handR = penHand; o.glow = .6 + .4 * Math.sin(t * 5); o.pupil = .66; o.look = [-.3, -.4]; o.rolling = u > 0 && u < 1; o.roll = (t - 137.2) * 2.4; }
  if (t >= 146.6) { o.look = [-.5, -.1]; x = MORN.pebble[0] - 25; }
  if (speakingNow('pebble', t)) { o.glow = .9 + .1 * Math.sin(t * 20); o.pupil = .62 + .1 * Math.sin(t * 16); o.dy = (o.dy || 0) + .1 * Math.abs(Math.sin(t * 13)); }
  return { form: 'pebble', x, y, o, s: 34 };
}
STATES.mia = miaS; STATES.mum = mumS; STATES.pebble = pebbleS; STATES.biscuit = biscuitS;
STATES.leo = (t) => t < T2 ? OLD2.leo(t) : t < 49.4 ? leoKid(t) : t < T_C5 ? (t < 59.8 || (t >= 72.5 && t < 75.4) ? leoGhostS(t) : (t >= 59.8 && t < 72.5 ? leoOffS(t) : null)) : leoMorning(t);
STATES.sam = (t) => t < T2 ? OLD2.sam(t) : t < 49.4 ? samKid(t) : t < T_C5 ? (t < 79.2 ? samGhostS(t) : null) : samMorning(t);

/* ───────── stage: tags, ticks, props ───────── */
function tagFor(id, t) {
  const PEN = '#8a8a92', G_ = '#a9a39a';
  if (id === 'mum') { if (t < 37.9) return { text: 'P5', pencil: false, stroke: G_ }; return { text: 'Mum', stroke: CARD_OF.mum.ink }; }
  if (id === 'biscuit') { const st = STATES.biscuit(t); if (!st || st.o?.hidden) return null; if (t < 51.0) return { text: 'Biscuit?', pencil: true, stroke: CARD_OF.biscuit.ink };
    if (t < FOLD_T[1]) return { text: 'P6', stroke: G_ }; return { text: 'Biscuit', stroke: CARD_OF.biscuit.ink }; }
  if (id === 'sam') return { text: 'Sam', stroke: CARD_OF.sam.ink };
  if (id === 'leo') { if (t >= 72.6 && t < T_C5) return t < 75.4 ? { text: 'P7', stroke: G_ } : null; return { text: 'Leo', stroke: CARD_OF.leo.ink }; }
  if (id === 'mia') { if (t >= 72.6 && t < 93.2) { if (t < 74.7) return { text: 'P8', stroke: G_ }; if (t < REPAIR_T[1]) return { text: 'Leo?', pencil: true, stroke: CARD_OF.leo.ink }; return { text: 'Mia', stroke: CARD_OF.mia.ink }; } return { text: 'Mia', stroke: CARD_OF.mia.ink }; }
  return null;
}
function overlayStage(t) {
  if (t < T2) return overlayStageOld(t);
  assertOnStage(t); const dk = darkK(t);
  for (const char of ['mia', 'leo', 'sam', 'mum', 'biscuit']) { const tg = tagFor(char, t); if (!tg || dk > .2) continue; const an = anchorOf(char, t); if (!an) continue; assertIds('tag', char, an.id);
    const [x, y] = sp(an.x, an.top - 40 * (an.s / 36)), pop = backOut(seg(t, char === 'mum' ? 36.3 : T2, char === 'mum' ? 36.7 : T2 + .4)), key = tg.text;
    const flash = (char === 'mia' && t > REPAIR_T[1] && t < REPAIR_T[1] + .6) || (char === 'leo' && t > FOLD_T[1] && false);
    OVER.push((c) => { UI.tag(x, y, tg.text, { a: Math.min(1, pop), s: (flash ? 1 + .12 * Math.sin((t - REPAIR_T[1]) * 22) : 1) * .84, size: 38, rot: -.04 + hash(char.length) * .06, color: tg.pencil ? '#8a8a92' : PAL.ink, stroke: tg.stroke });
      if (window.DEBUG_IDS) { c.save(); c.fillStyle = '#d00'; c.font = '700 20px sans-serif'; c.fillText(`${char} (${key})`, x - 50, y - 38); c.restore(); } }); }
  for (const s of SAYS) { const speakers = [s.who, ...(s.also || [])]; for (const who of speakers) { const cur = speakingNow(who, t); if (!cur || cur.id !== s.id || cur.offscreen) continue; const an = anchorOf(who, t); if (!an) continue; assertIds('tick', who, an.id);
    const dir = (STATES[who](t).o.turn ?? 0) > .05 ? 1 : -1, [x, y] = sp(an.x + dir * an.w * .95, an.top + 1.6 * an.s), k = .5 + .5 * Math.sin(t * 16);
    OVER.push((c) => { c.save(); c.strokeStyle = INKC[who]; c.lineWidth = 4; c.lineCap = 'round'; for (let i = -1; i <= 1; i++) { const a = i * .5 - (dir < 0 ? Math.PI : 0), r0 = 12 + 3 * k, r1 = 26 + 6 * k; c.beginPath(); c.moveTo(x + Math.cos(a) * r0, y + Math.sin(a) * r0); c.lineTo(x + Math.cos(a) * r1, y + Math.sin(a) * r1); c.stroke(); } c.restore(); }); } }
  const off = SAYS.find(s => s.offscreen && t >= s.t0 + .1 && t <= s.t0 + s.dur - .35);
  if (off) { const [x, y] = sp(150, 540); OVER.push((c) => { c.save(); c.strokeStyle = INKC[off.who]; c.lineWidth = 5; c.lineCap = 'round'; for (let i = 0; i < 3; i++) { c.globalAlpha = .9 - i * .22; c.beginPath(); c.arc(x - 18, y, 26 + i * 18 + ((t * 40) % 18), -.8, .8); c.stroke(); } c.fillStyle = INKC[off.who]; for (let i = 0; i < 5; i++) { const h = [8, 18, 28, 16, 10][i] * (.6 + .4 * Math.sin(t * 14 + i)); c.fillRect(x - 74 + i * 9, y - h / 2, 5, h); } c.restore(); }); }
  if (t > 65.4 && t < 65.8) OVER.push((c) => { c.save(); c.globalAlpha = 1 - seg(t, 65.6, 65.8); c.font = '700 130px Caveat'; c.textAlign = 'center'; c.fillStyle = '#FFF4B0'; c.strokeStyle = PAL.ink; c.lineWidth = 9; c.lineJoin = 'round'; c.translate(STAGE_CX, 280); c.rotate(-.06); c.strokeText('CRACK!', 0, 0); c.fillText('CRACK!', 0, 0); c.restore(); });
  if (t > 83.8 && t < 93) { const mm = Math.round(lerp(16, 31, seg(t, 84.2, 92.4))), a = seg(t, 83.8, 84.3) * (1 - seg(t, 92.4, 93)); OVER.push(() => UI.tag(300, 215, `21:${String(mm).padStart(2, '0')}`, { a, size: 56, fill: '#FFF8EC', rot: -.03 })); }
}
function perceiveFrames(t) {
  const out = []; const wins = { mia: [[10.0, 13.0], [49.9, 52.4], [72.8, 74.6]], leo: [[10.3, 13.3], [49.9, 52.4], [72.8, 74.6]], sam: [[10.6, 13.6], [49.9, 52.4]], biscuit: [[33.3, 36.0], [51.1, 54.0]], mum: [[36.4, 38.4]] };
  for (const [id, ws] of Object.entries(wins)) for (const [a, b] of ws) if (t >= a && t < b) { const an = anchorOf(id, t); if (an) out.push({ id, an, p: backOut(seg(t, a, a + .3)) * (1 - seg(t, b - .3, b)) }); }
  return out;
}
function drawPlateProps(t) {
  if (t < 36.5) return; const hold = t < 39.0, drop = seg(t, 38.9, 39.3);
  if (!hold) { const y = 736 + (1 - easeOut(drop)) * -14, bx = 495; paint(ellPts(430, y, 66, 14, 14), { wash: '#FFFDF5', ink: PAL.ink, sw: .8 });
    for (const dx of [-24, 20]) { paint(rrPts(430 + dx - 26, y - 22, 52, 20, 9), { wash: '#D9A066', ink: PAL.ink, sw: .6 }); inkLine([[430 + dx - 18, y - 12], [430 + dx + 18, y - 12]], .8, '#E8C22C', 'ink', .3); }
    const tilt = t > 59.0 && t < 59.9 ? -.6 * Math.sin(seg(t, 59.0, 59.9) * Math.PI) : 0; push(); translate(bx, y + 4); rotate(tilt);
    paint(rrPts(-13, -58, 26, 58, 9), { wash: '#D0382B', ink: PAL.ink, sw: .8 }); paint(rrPts(-7, -72, 14, 16, 4), { wash: '#FFF4E0', ink: PAL.ink, sw: .6 }); paint(rrPts(-9, -38, 18, 20, 3), { wash: '#FFF4E0', ink: null }); pop(); }
  if (t > 59.2 && t < 59.9) { const an = anchorOf('leo', t); if (an) for (let k = 0; k < 9; k++) { const u = clamp((t - 59.2 - k * .03) / .35); if (u <= 0 || u >= 1) continue; const tx = an.x - 30 + hash(k) * 70, ty = an.top + 3.6 * an.s + hash(k * 2) * 40; blob(lerp(480, tx, u), lerp(676, ty, u) - Math.sin(u * Math.PI) * 60, 7 - 2 * u, PAL.ketchup); } }
}
function stainedSheetProp(t) {
  if (t < 93.2 || t >= 117.7) return; const k = easeOut(seg(t, 93.2, 93.7)), x = 650, y = 818 - (1 - k) * 90;
  paint([[x - 52, y + 4], [x - 36, y - 18], [x - 6, y - 10], [x + 20, y - 24], [x + 52, y - 6], [x + 58, y + 12], [x + 20, y + 22], [x - 30, y + 20]], { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: .8 });
  paint(ellPts(x + 6, y - 2, 20, 12, 12), { wash: PAL.ketchup, ink: '#9c2f25', sw: .5 }); blob(x - 20, y + 6, 6, PAL.ketchup); blob(x + 34, y + 2, 5, PAL.ketchup);
}
const levelAt2 = t => t < 85.3 ? 1 : t < 88.0 ? lerp(1, .66, seg(t, 85.3, 85.7)) : t < 90.7 ? lerp(.66, .33, seg(t, 88.0, 88.4)) : t < 90.9 ? lerp(.33, 0, seg(t, 90.7, 91.1)) : 0;
function wrappers2(t) {
  const spots = [[480, 835, 0], [620, 862, 1], [700, 828, 2], [440, 880, 1], [800, 840, 0], [540, 910, 2]], times = [85.7, 88.3, 91.0, 86.3, 89.0, 91.4];
  spots.forEach(([x, y, k], i) => { const d = t - times[i]; if (d < 0) return; const f = Math.min(1, d / .45), yy = y - (1 - easeOut(f)) * 150 - Math.sin(f * Math.PI) * 30; paint([[x - 14, yy], [x - 5, yy - 13], [x + 9, yy - 9], [x + 16, yy + 4], [x + 3, yy + 11], [x - 11, yy + 9]], { wash: [PAL.pink, PAL.candle, PAL.mint][k], ink: PAL.ink, sw: .5 }); });
}
function worldNight(t) {
  if (t < T2) return worldNightOld(t);
  const dk = darkK(t); setSky(t, false); setSpider(t); setClockHands(t < 65 ? 8 : t < 84 ? 14 : Math.round(lerp(16, 31, seg(t, 84.2, 92.4)))); setCauldron(levelAt2(t), t, false); setDoor(t, T_DOOR);
  paint(rectPts(-40, -40, W + 80, H + 80, 0), { wash: PAL.paper, washOp: 58, ink: null });
  if (t > 65.4 && t < 65.8) { const k = 1 - seg(t, 65.4, 65.8); paint([[1000, 120], [1050, 250], [1015, 255], [1075, 400], [980, 290], [1015, 285], [960, 140]], { wash: '#FFFFFF', washOp: 255 * k, ink: null }); }
  drawPlateProps(t); stainedSheetProp(t); sheetArmProp(t); ideaSpark(t);
  const items = [];
  for (const id of ['mia', 'leo', 'sam', 'biscuit', 'pebble', 'mum']) { const st = STATES[id](t); if (st) items.push([st.y, () => { if (st.o?.hidden) { const w = Math.sin(t * 14) * 14; paint([[H2.pile - 8, 470], [H2.pile + 10 + w * .3, 436], [H2.pile + 24 + w, 404], [H2.pile + 30 + w, 398], [H2.pile + 16 + w * .6, 440], [H2.pile + 8, 472]], { wash: PAL.brown, ink: PAL.ink, sw: .8 }); } else drawState(id, t, dk); }]); }
  items.sort((a, b) => a[0] - b[0]); for (const [, fn] of items) fn();
  if (t > 47.6 && t < 49.6) for (const [who, t0] of [['mia', 47.8], ['leo', 47.86], ['sam', 47.92]]) { const u = seg(t, t0, 49.4); if (u > 0 && u < 1) { const px = who === 'mia' ? H2.pile + 40 : who === 'leo' ? H2.leo[0] : H2.sam[0], py = who === 'mia' ? 800 : who === 'leo' ? H2.leo[1] : H2.sam[1]; drawSheet(px, py + 20, u, who); } }
  if (t > 49.35 && t < 50.0) for (const [who, px, py] of [['mia', H2.pile + 40, 800], ['leo', H2.leo[0], H2.leo[1]], ['sam', H2.sam[0], H2.sam[1]]]) puff(px, py + 20, seg(t, 49.4, 50.0));
  if (t > 49.6 && t < 51.0) { const u = seg(t, 49.7, 50.7); if (u < 1) { push(); translate(H2.pile + Math.sin(u * 7) * 24 * (1 - u), 520); rotate(Math.sin(u * 9) * .25 * (1 - u)); scale(.62); drawSheet(0, 0, u, 'dog'); pop(); }
    if (t > 50.7 && t < 51.3) puff(H2.pile, 650, seg(t, 50.7, 51.3)); }
  if (t > 49.4 && t < 50.2) puff(H2.pile, 716, seg(t, 49.4, 50.2));
  if (t > 63.8 && t < 64.6) puff(255, 940, seg(t, 63.8, 64.6));
  if (t > 92.9 && t < 93.7) puff(800, 880, seg(t, 92.9, 93.7));
  setBats(t); setLights(t, 1 - dk, false); zzzBubbles(t); flushGlowQ(); sparkles(t, 8);
}
function worldMorning(t) {
  setSky(t, true); setSpider(t); setDoor(t, -99); setClockHands(40); setMorningExtras(t);
  paint(rectPts(-40, -40, W + 80, H + 80, 0), { wash: PAL.paper, washOp: 40, ink: null });
  stainedSheetProp(t);
  const items = []; for (const id of ['mia', 'leo', 'sam', 'biscuit', 'pebble', 'mum']) { const st = STATES[id](t); if (st) items.push([st.y, () => drawState(id, t)]); }
  items.sort((a, b) => a[0] - b[0]); for (const [, fn] of items) fn();
  if (t > 146.0) { const m = STATES.mia(t); if (m) { const d = seg(t, 146.0, 148.0), f = ease(d), sx = m.x + .95 * m.s, sy = m.y - (KID.mia.legH + KID.mia.torsoH * .55) * m.s, yy = lerp(sy, m.y - 4, f * f), xx = sx + 20 * f + Math.sin(d * 6) * 10 * (1 - f);
      push(); translate(xx, yy); rotate(d * 5); paint([[-22, 0], [-8, -20], [14, -12], [24, 6], [4, 18], [-16, 12]], { wash: PAL.pink, ink: PAL.ink, sw: .8 }); paint([[-6, -4], [6, -8], [8, 4], [-4, 6]], { wash: PAL.candle, ink: null }); pop(); } }
  dust(t, 36); flushGlowQ();
}
function lensOverlay(t) {
  const pi = seg(t, 80.2, 80.8) * (1 - seg(t, 82.6, 83.2)); if (pi <= .01) return; const m = STATES.mia(t); if (!m) return; const [lx, ly] = sp(m.x, m.y - 20), r = lerp(1100, 330, easeOut(pi)), pw = seg(t, 80.8, 81.3) * (1 - seg(t, 82.4, 82.9)), match = seg(t, 81.0, 81.6);
  OVER.push((c) => { c.save(); c.globalAlpha = .7 * pi; c.fillStyle = '#10131c'; c.beginPath(); c.rect(0, 0, W, H); c.arc(lx, ly, r, 0, TAU, true); c.fill('evenodd'); c.restore();
    c.save(); c.globalAlpha = pi; c.strokeStyle = '#7fe3ff'; c.lineWidth = 8; c.beginPath(); c.arc(lx, ly, r, 0, TAU); c.stroke(); c.lineWidth = 3; for (let i = 0; i < 12; i++) { const a = i / 12 * TAU + t * .3; c.beginPath(); c.moveTo(lx + Math.cos(a) * (r + 6), ly + Math.sin(a) * (r + 6)); c.lineTo(lx + Math.cos(a) * (r + 24), ly + Math.sin(a) * (r + 24)); c.stroke(); } c.restore();
    if (pw > .01) { c.save(); c.globalAlpha = pw; const px = 300, py = 230; c.fillStyle = 'rgba(255,251,240,.96)'; roundRect(c, px - 200, py - 120, 400, 250, 20); c.fill(); c.strokeStyle = PAL.ink; c.lineWidth = 3; c.stroke(); c.font = '600 32px Caveat'; c.fillStyle = PAL.ink; c.textAlign = 'center'; c.fillText('voice heard', px, py - 82); c.fillText("high voice · pink sneakers", px, py + 18);
      for (let row = 0; row < 2; row++) { c.strokeStyle = row ? PAL.pink : PAL.violet; c.lineWidth = 5; c.lineCap = 'round'; c.beginPath(); for (let i = 0; i <= 60; i++) { const xx = px - 150 + i * 5, base = Math.sin(i * .5 + t * 6) * 18 * Math.sin(i / 60 * Math.PI), alt = Math.sin(i * .5 + t * 6 + (1 - match) * 2.4 * (row ? 0 : 1)) * (18 - (1 - match) * 6) * Math.sin(i / 60 * Math.PI), yy = py + (row ? 62 : -36) + (row ? base : alt); if (i === 0) c.moveTo(xx, yy); else c.lineTo(xx, yy); } c.stroke(); }
      if (match > .9) { c.fillStyle = '#2a9d6f'; c.font = '700 40px Caveat'; c.fillText('match ✓', px, py + 112); } c.restore(); } });
}

/* ───────── notebook: Who's who + Screenplay (later clips) ───────── */
CARDS.find(c => c.id === 'biscuit').name.push([56.9, 'ink', 'Biscuit', 1, 'ears, tail, bark']);
CARDS.push({ id: 'p6', char: 'biscuit', t: 51.1, col: '#c98a4b', ink: '#9a6a36', chips: [], name: [], ev0: 'small ghost — new person?', cue: 'ears', voiceAt: 1e9, ghostOnly: true, fold: FOLD_T });
CARDS.push({ id: 'p8', char: 'mia', t: 69.9, col: '#FF8FC4', ink: '#d2468a', chips: [], name: [], ev0: 'small silhouette', mini: true, sub: 0, fold: [74.0, 74.7] }, { id: 'p7', char: 'leo', t: 72.4, col: '#9d8cff', ink: '#5d4bd0', chips: [], name: [], ev0: 'tall silhouette', mini: true, sub: 1, fold: [76.2, 76.9] });
const EVX = {
  mum: [[37.9, 'recognised: face + voice']], mia: [[50.3, 'same pink sneakers'], [70.0, 'lost track in the dark'], [82.4, "high voice + pink sneakers"], [111.8, 'wore the stained sheet']],
  leo: [[50.3, 'tallest, striped socks'], [59.8, '+ ketchup stain'], [60.6, 'stained sheet on the sofa arm'], [70.0, 'lost track in the dark'], [74.7, 'P8: his ketchup stain?'], [76.9, 'P7: kitchen voice'], [82.4, 'kitchen voice; stain was not his']],
  sam: [[50.3, 'bell on his shoe']], biscuit: [[56.9, 'ears, tail, bark']],
};
const PULSE = { mum: [[0, 37.9], [2, 37.9]], mia: [[2, 79.4], [3, 81.6]], leo: [[2, 58.0], [2, 76.2], [3, 59.8]], sam: [[3, 52.0]], biscuit: [[3, 56.0]] };
const CHIPS = [
  { id: 'P1', segs: [{ card: 'mia', t: 10.0 }], sure: 33.6 }, { id: 'P2', segs: [{ card: 'leo', t: 10.3 }], sure: 26.9 }, { id: 'P3', segs: [{ card: 'sam', t: 10.6 }], sure: 30.0 }, { id: 'P4', segs: [{ card: 'biscuit', t: 33.3 }], sure: 56.9 },
  { id: 'P5', segs: [{ card: 'mum', t: 37.9, from: 'origin', t0: 37.2 }], sure: 37.9 }, { id: 'P6', segs: [{ card: 'p6', t: 51.1 }, { card: 'biscuit', t: 56.8, from: 'p6', t0: 56.0 }], sure: 56.8 },
  { id: 'P7', segs: [{ card: 'p7', t: 72.4 }, { card: 'leo', t: 76.9, from: 'p7', t0: 76.2 }], sure: 76.9 },
  { id: 'P8', segs: [{ card: 'p8', t: 69.9 }, { card: 'leo', t: 74.7, from: 'p8', t0: 74.0 }, { card: 'mia', t: 82.2, from: 'leo', t0: 81.4 }], sure: 82.2 },
];
const NAMED = { P1: 'Mia', P8: 'Mia', P2: 'Leo', P7: 'Leo', P3: 'Sam', P4: 'Biscuit', P6: 'Biscuit', P5: 'Mum' };
const toNamed = s => s.replace(/\bP[1-8]\b/g, m => NAMED[m]);
const SCRIPT2 = SCRIPT1.filter(l => l[1] !== 'head').map(l => [...l]).concat([
  [36.0, 'head', null, '21:08'], [37.0, 'say', 'P5', 'One candy each tonight — the rest is for tomorrow!'], [38.6, 'act', null, '(P5 puts hot dogs and ketchup on the table)'], [41.0, 'act', null, '(P1 P2 P3 groan)'], [42.3, 'act', null, '(P5 leaves)'], [44.4, 'act', null, '(P4 leaves the basket)'],
  [45.8, 'say', 'P1', 'Costume time!'], [48.9, 'act', null, '(P1 P2 P3 put on white sheets)'], [51.0, 'act', null, '(P6, a small ghost, appears)'], [52.0, 'say', 'P3', 'Boooo!'], [55.2, 'say', 'P6', 'Arf!'],
  [58.0, 'say', 'P2', "Aw, ketchup! I can't see a thing in this."], [59.6, 'act', null, "(ketchup splats on P2's sheet)"], [61.8, 'say', 'P2', "Ew, it's all sticky! I'll put it back on later."], [63.7, 'act', null, '(P2 hangs the stained sheet on the sofa arm)'], [64.8, 'act', null, '(P6 falls asleep in the basket)'],
  [65.4, 'head', null, '21:14'], [65.7, 'act', null, '(lights go out)'], [66.2, 'say', 'P3', "Eek! It's dark!"], [69.6, 'act', null, '(P8, a small silhouette, drops a plain sheet on the floor)'], [70.9, 'act', null, '(P8 takes the stained sheet from the sofa arm)', { cite: 1 }], [72.3, 'act', null, '(P7, a tall silhouette, finds a plain sheet on the floor and puts it on)'], [72.9, 'act', null, '(lights come back)'],
  [73.6, 'act', null, '(P7, plain sheet, walks into the kitchen)'], [73.9, 'act', null, '21:15 (P8, ketchup-stained sheet, goes to the candy bowl)', { cite: 1 }], [76.2, 'say', 'P7', '(kitchen) Do we have orange juice?'], [79.4, 'say', 'P8', 'Hehehe…'],
  [83.8, 'head', null, '21:16'], [84.2, 'say', 'P8', 'One for me… one more for me…'], [85.2, 'act', null, '21:16 (P8 takes a candy)', { cite: 1 }], [87.9, 'act', null, '21:22 (P8 takes a candy)', { cite: 1 }], [90.6, 'act', null, '21:29 (P8 takes a candy)', { cite: 1 }], [92.4, 'act', null, '21:31 (the candy bowl is empty)', { cite: 1 }],
  [111.6, 'head', null, '08:30'],
].map(l => l)).sort((a, b) => a[0] - b[0]);
SCRIPT2.forEach(([t, kind, who, text]) => { if (kind === 'act' && NAMEWORDS.test(text)) throw new Error('screenplay action contains a character name before the save stamp: ' + text); if (kind === 'say' && who !== 'Pebble' && !/^P\d$/.test(who)) throw new Error('screenplay speaker must be a code'); });
// every chip move must match the script: the chip's flights must be preceded by a line that mentions its code
CHIPS.forEach(ch => ch.segs.forEach(sg => { if (sg.from && !SCRIPT2.some(l => (l[2] === ch.id || (l[3] || '').includes(ch.id)) && l[0] <= sg.t0 + 1.9)) throw new Error(`chip ${ch.id} moves at ${sg.t0} without a script line`); }));
const WIPE_LINES = SCRIPT2.filter(l => l[1] !== 'head');
const WIPE_END = WIPE0 + WIPE_LINES.length * WIPE_DT + .6;
const FOCUS = [[127.7, 131.9, l => /falls asleep in the basket/.test(l[3])], [137.6, 146.9, l => l[4] && l[4].cite]];

function clockAt(t) { return t < 65.4 ? '21:08' : t < 83.8 ? '21:14' : t < 111.6 ? '21:16' : '08:30'; }
function chipPositions(t, x0, ry, cw, cardIdx) {
  const holdersAt = {}; const pos = {};
  const slotXY = (cid, j) => cid === 'hover' ? [x0 + cw - 66 - 2 * 56, ry(cardIdx.leo) - 16] : cid === 'origin' ? [x0 + 220, y0P + 6 * 86 + 60] : [x0 + cw - 66 - j * 56, ry(cardIdx[cid]) + 8];
  const y0P = 20 + 124 - 40;
  const holders = {};
  for (const ch of CHIPS) { let cur = null; for (const sg of ch.segs) { if (t >= sg.t) cur = sg; } if (!cur) { const first = ch.segs[0]; if (first.from && t >= first.t0) cur = null; }
    const nextFlight = ch.segs.find(sg => sg.from && t >= sg.t0 && t < sg.t); const leaving = ch.segs.find(sg => sg.from && sg.t0 <= t && cur && sg !== cur && ch.segs.indexOf(sg) > ch.segs.indexOf(cur));
    ch._cur = cur; ch._flight = nextFlight; if (cur && !(leaving && t >= leaving.t0)) { (holders[cur.card ?? 'hover'] ||= []).push([cur.t, ch]); } }
  for (const k of Object.keys(holders)) holders[k].sort((a, b) => a[0] - b[0]);
  const place = (ch, x, y, extra = {}) => { pos[ch.id] = { x, y, ...extra }; };
  for (const [cid, arr] of Object.entries(holders)) arr.forEach(([tt, ch], j) => { const [x, y] = slotXY(cid, j); const bounce = t < tt + .45 ? -12 * Math.abs(Math.sin((t - tt) * 14)) * Math.exp(-(t - tt) * 5) : 0; place(ch, x, y + (cid === 'hover' ? 0 : bounce), { card: cid }); });
  for (const ch of CHIPS) if (ch._flight) { const sg = ch._flight, u = ease(seg(t, sg.t0, sg.t)); const fromId = sg.from; let fx, fy;
    if (fromId === 'origin' || fromId === 'hover') [fx, fy] = slotXY(fromId, 0); else { const prev = holders[fromId] || []; const jj = Math.max(0, prev.length); const [sx_, sy_] = slotXY(fromId, Math.max(0, prev.length)); fx = sx_; fy = sy_; if (ch.segs.length > 1) { const [ax, ay] = slotXY(fromId, 0); fx = ax; fy = ay; } }
    const dest = holders[sg.card ?? 'hover'] ? holders[sg.card ?? 'hover'].length : 0; const [dx, dy] = slotXY(sg.card ?? 'hover', dest);
    place(ch, lerp(fx, dx, u), lerp(fy, dy, u) - Math.sin(u * Math.PI) * 70, { flying: true, card: sg.card }); }
  return pos;
}
function drawPanel(c, t) {
  if (t < T2) return drawPanelOld(c, t);
  const x0 = PX0, y0 = 20, w = PW, h = 1040, ink = '#3a2f2a'; ctx = c;
  c.save(); c.shadowColor = 'rgba(30,20,10,.35)'; c.shadowBlur = 24; c.shadowOffsetX = -6; c.shadowOffsetY = 8; c.fillStyle = '#FFF6E0'; roundRect(c, x0, y0, w, h, 26); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = PAL.ink; c.lineWidth = 4; c.stroke();
  c.strokeStyle = 'rgba(120,150,200,.22)'; c.lineWidth = 2; for (let y = y0 + 90; y < y0 + h - 20; y += 38) { c.beginPath(); c.moveTo(x0 + 50, y); c.lineTo(x0 + w - 20, y); c.stroke(); }
  for (let y = y0 + 40; y < y0 + h - 20; y += 54) { c.strokeStyle = PAL.ink; c.lineWidth = 5; c.lineCap = 'round'; c.beginPath(); c.moveTo(x0 - 8, y); c.lineTo(x0 + 34, y); c.stroke(); c.fillStyle = '#d8d3c8'; c.beginPath(); c.arc(x0 + 30, y, 6, 0, TAU); c.fill(); }
  c.fillStyle = PAL.ink; c.font = '700 44px Caveat'; c.textBaseline = 'alphabetic'; c.textAlign = 'left'; c.fillText("Pebble's memory", x0 + 64, y0 + 54);
  const sec = (label, y, tech) => { c.fillStyle = PAL.violet; c.font = '700 36px Caveat'; c.textAlign = 'left'; c.fillText(label, x0 + 64, y); if (tech) { const lw = c.measureText(label).width; c.fillStyle = '#9a8e82'; c.font = '600 26px Caveat'; c.fillText(tech, x0 + 82 + lw, y); } c.strokeStyle = PAL.violet; c.lineWidth = 3; c.beginPath(); c.moveTo(x0 + 64, y + 10); c.lineTo(x0 + w - 34, y + 10); c.stroke(); };
  const swipe = (x, y, ww, hh, t0) => { const k = ease(seg(t, t0, t0 + .45)), a = 1 - seg(t, t0 + 1.0, t0 + 1.7); if (k > 0 && a > 0) { c.save(); c.fillStyle = `rgba(255,226,90,${.55 * a})`; roundRect(c, x, y, ww * k, hh, 10); c.fill(); c.restore(); } };
  sec("Who's who", y0 + 104);
  if (t > 49.9 && t < 111.3) { const a = sstep(49.9, 50.4, t) * (1 - sstep(110.8, 111.3, t)); c.save(); c.globalAlpha = a; c.fillStyle = '#FFF0E0'; c.strokeStyle = PAL.ketchup; c.lineWidth = 2.6; roundRect(c, x0 + w - 270, y0 + 72, 232, 38, 12); c.fill(); c.stroke(); c.fillStyle = PAL.ketchup; c.font = '700 26px Caveat'; c.textAlign = 'center'; c.fillText('faces visible: 0', x0 + w - 154, y0 + 99); c.restore(); }
  const cardY = i => y0 + 124 + i * 86, cw = w - 90, idx = Object.fromEntries(CARDS.map((cd, i) => [cd.id, i])); idx.p8 = 5; idx.p7 = 5 + 42 / 86;
  const chipPos = chipPositions(t, x0, cardY, cw, idx), glowOf = {};
  WIPE_LINES.forEach((l, i) => { const tw = WIPE0 + i * WIPE_DT; if (t >= tw && t < tw + .9) for (const id of Object.keys(NAMED)) if (l[2] === id || (l[3] || '').match(new RegExp('\\b' + id + '\\b'))) glowOf[id] = Math.max(glowOf[id] || 0, 1 - (t - tw) / .9); });
  CARDS.forEach((cd, i) => {
    let k = backOut(seg(t, cd.t, cd.t + .4)); if (k <= .01) return; let ry = cardY(idx[cd.id]), fold = 0;
    if (cd.fold) { fold = ease(seg(t, cd.fold[0], cd.fold[1])); if (fold >= 1) return; }
    if (cd.mini) { swipe(x0 + 52, ry - 2, cw + 6, 42, cd.t); const faintM = 1; c.save(); c.translate(0, (1 - Math.min(1, k)) * 10 - fold * 30); c.globalAlpha = Math.min(1, k) * (1 - fold); c.fillStyle = '#ECE7DC'; roundRect(c, x0 + 56, ry, cw, 38, 12); c.fill(); c.strokeStyle = '#bdb6a8'; c.lineWidth = 2.4; c.setLineDash([6, 5]); c.stroke(); c.setLineDash([]);
      c.fillStyle = 'rgba(120,135,170,.65)'; c.beginPath(); c.arc(x0 + 84, ry + 12, cd.id === 'p7' ? 8 : 6, 0, TAU); c.fill(); c.beginPath(); c.moveTo(x0 + 70, ry + 34); c.quadraticCurveTo(x0 + 84, ry + (cd.id === 'p7' ? 12 : 16), x0 + 98, ry + 34); c.closePath(); c.fill();
      c.fillStyle = '#b3ab9d'; c.font = '700 34px Caveat'; c.textAlign = 'left'; c.fillText('?', x0 + 112, ry + 29); c.fillStyle = '#7a6b60'; c.font = '500 22px Caveat'; c.fillText(cd.ev0, x0 + 138, ry + 27, 200); c.restore(); return; }
    const h_ = cardName(cd, t), known = !!h_, col = cd.col, faint = (cd.id === 'mia' || cd.id === 'leo') && t >= 69.4 && t < 73.9 ? .5 : 1;
    const conflict = cd.id === 'leo' && t >= CONFLICT_T[0] && t < REPAIR_T[1] ? Math.min(1, seg(t, CONFLICT_T[0], CONFLICT_T[0] + .3)) * (1 - seg(t, REPAIR_T[0] + .4, REPAIR_T[1])) : 0, pulse = .5 + .5 * Math.sin(t * 9);
    const evt = [cd.t, ...cd.name.map(n => n[0]), ...(EVX[cd.id] || []).map(e => e[0])]; evt.forEach(tt => swipe(x0 + 52, ry - 2, cw + 6, 80, tt));
    c.save(); c.translate(0, (1 - Math.min(1, k)) * 12 - fold * 70); c.globalAlpha = Math.min(1, k) * faint * (1 - fold); if (fold > 0) { c.translate(x0 + 56 + cw / 2, ry + 38); c.scale(1, 1 - fold * .6); c.translate(-(x0 + 56 + cw / 2), -(ry + 38)); }
    c.fillStyle = conflict > 0 ? mixCol('#FFD9D2', '#FF9C8F', pulse * conflict) : known ? mixCol(col, '#FFF6E0', .78) : '#ECE7DC'; roundRect(c, x0 + 56, ry, cw, 76, 16); c.fill(); c.strokeStyle = conflict > 0 ? PAL.ketchup : known ? cd.ink : '#bdb6a8'; c.lineWidth = conflict > 0 ? 3.4 : 2.6; c.stroke();
    const ghostForm = (cd.id !== 'biscuit' && cd.id !== 'mum' && cd.id !== 'p6' && (cd.id === 'leo' ? ((t >= 49.9 && t < 60.4) || (t >= 72.6 && t < 111.3)) : (t >= 49.9 && t < 111.3))) || (cd.id === 'biscuit' && t >= 51.0 && t < 71.0) || cd.id === 'p6';
    doodle(c, cd.char, x0 + 92, ry + 38, cd.id === 'p6' ? 22 : 26, t, ghostForm);
    let nm = '?', pencil = false, nw = 0; c.textAlign = 'left'; if (h_) { nm = h_.tx; pencil = h_.st === 'pencil'; } c.font = '700 40px Caveat'; nw = c.measureText(nm).width;
    c.fillStyle = h_ ? (pencil ? '#8f8f98' : ink) : '#b3ab9d'; c.fillText(nm, x0 + 134, ry + 38);
    if (pencil) { const bx = x0 + 134, by = ry + 46; c.strokeStyle = '#8f8f98'; c.lineWidth = 2; roundRect(c, bx, by, 84, 8, 4); c.stroke(); c.fillStyle = '#a9a9b2'; roundRect(c, bx, by, 84 * h_.cf * seg(t, h_.t, h_.t + .5), 8, 4); c.fill(); }
    if (h_ && !pencil && !cd.known) rowSeal(c, x0 + 134 + nw + 62, ry + 24, ease(seg(t, h_.t + .1, h_.t + .45)));
    if (cd.known) { c.fillStyle = '#2a9d6f'; c.font = '700 26px Caveat'; c.fillText('✓ known', x0 + 134 + nw + 14, ry + 38); }
    let ev = h_ ? h_.ev : cd.ev0; for (const [tt, e] of (EVX[cd.id] || [])) if (t >= tt) ev = e; if (faint < 1) ev = 'lost track in the dark'; if (conflict > .3) ev = '⚠ Leo in two places? (P7 + P8)';
    c.fillStyle = conflict > 0 ? PAL.ketchup : '#7a6b60'; c.font = '500 22px Caveat'; c.fillText(ev, x0 + 134, ry + 69, 250);
    const cues = cd.known ? cd.cues : ['face', 'body', 'voice', cd.cue], lits = cd.known ? cd.lit : cd.id === 'p6' ? [1e9, cd.t + .3, 1e9, 1e9] : [cd.t + .2, cd.t + .45, cd.voiceAt, cd.t + .8];
    cues.forEach((kk, j) => { const l = lits[j], on = t >= l; let pu = on ? 1 + .25 * Math.exp(-(t - l) * 6) : 1; for (const [pj, pt] of (PULSE[cd.id] || [])) if (pj === j && t >= pt && t < pt + 1.2) pu = Math.max(pu, 1 + .35 * Math.exp(-(t - pt) * 5)); UI.icon(kk, x0 + cw - 138 + j * 40, ry + 58, .72 * pu, 1, on ? cd.ink : '#cdc5b8', on ? PAL.ink : '#b9b1a4'); });
    c.restore();
  });
  for (const ch of CHIPS) { const p = chipPos[ch.id]; if (!p) continue; const cd = CARDS.find(c_ => c_.id === p.card), known = cd && (cd.known || !!cardName(cd, t)), sure = t >= ch.sure && !p.flying; const g = glowOf[ch.id] || 0;
    if (cd && cd.fold && t >= cd.fold[0]) continue; c.save(); c.globalAlpha = 1; c.fillStyle = p.flying ? '#FFE9A8' : known ? cd.col : '#cfc8bc'; if (g > 0) { c.shadowColor = '#FFD34A'; c.shadowBlur = 22 * g; }
    roundRect(c, p.x, p.y, 52, 30, 9); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = PAL.ink; c.lineWidth = 2.6; if (!sure) c.setLineDash([5, 4]); c.stroke(); c.setLineDash([]); c.fillStyle = PAL.ink; c.font = '700 24px Caveat'; c.textAlign = 'center'; c.fillText(ch.id, p.x + 26, p.y + 23); c.restore(); }
  if (t >= REPAIR_T[1] && t < REPAIR_T[1] + 1.0) { const ry = cardY(idx.mia); c.save(); c.globalAlpha = 1 - seg(t, REPAIR_T[1] + .3, REPAIR_T[1] + 1.0); c.strokeStyle = '#2a9d6f'; c.lineWidth = 5; roundRect(c, x0 + 52, ry - 3, cw + 8, 82, 18); c.stroke(); c.restore(); }
  // ── Screenplay
  const sy = y0 + 124 + 6 * 86 + 34; sec('Screenplay', sy, clockAt(t));
  const top = sy + 28, cap = 6 * 47 + 8, W2 = w - 130, size = (t >= 137.0 && t < 147.5) ? 30 : 33;
  const focus = FOCUS.find(f => t >= f[0] - .5 && t < f[1] + .5), fk = focus ? seg(t, focus[0] - .5, focus[0] + .1) * (1 - seg(t, focus[1], focus[1] + .5)) : 0;
  const lines = []; ctx = c;
  for (const L of SCRIPT2) { const [tt, kind, who, text] = L; if (t < tt) continue; const pre = kind === 'say' ? who + ': ' : '', wrapped = UI.wrap(pre + text, W2, `600 ${size}px Caveat`); const hh = kind === 'head' ? 40 : wrapped.length * (size * 1.12) + 8; lines.push({ L, tt, kind, who, text, pre, wrapped, h: hh }); }
  let total = 0; lines.forEach((l, i) => { l.y = total; const inFocus = focus && focus[2](l.L); l.collapse = focus && !inFocus && l.kind !== 'head' ? fk : (focus && l.kind === 'head' ? fk : 0); total += l.h * (1 - l.collapse) * (i === lines.length - 1 ? ease(seg(t, l.tt, l.tt + .4)) : 1); });
  const maxOff = Math.max(0, total - cap), baseOff = maxOff;
  const yOfIdx = i => { let yy = 0; for (let k = 0; k < i; k++) yy += lines[k].h; return yy; };
  let off = baseOff;
  if (t >= WIPE0 - .8 && lines.length) { const full = lines.reduce((a, l) => a + l.h, 0), mx = Math.max(0, full - cap);
    const nonHead = lines.map((l, i) => [l, i]).filter(([l]) => l.kind !== 'head'); const fIdx = clamp((t - WIPE0) / WIPE_DT, 0, nonHead.length - 1), a_ = Math.floor(fIdx), b_ = Math.min(nonHead.length - 1, a_ + 1), yy = lerp(yOfIdx(nonHead[a_][1]), yOfIdx(nonHead[b_][1]), fIdx - a_);
    if (t < WIPE0) off = lerp(baseOff, 0, ease(seg(t, WIPE0 - .8, WIPE0))); else if (t < WIPE0 + nonHead.length * WIPE_DT) off = clamp(yy - 70, 0, mx);
    else { const k1 = yOfIdx(nonHead[1][1]) - 20, k2 = mx; off = kf(t, [[WIPE0 + nonHead.length * WIPE_DT, clamp(yy - 70, 0, mx)], [WIPE_END + .3, clamp(k1 - 10, 0, mx)], [WIPE_END + 2.3, clamp(k1 - 10, 0, mx)], [WIPE_END + 3.1, k2]], ease); }
    if (t >= 111.0) off = mx; }
  if (focus && fk > 0) off = lerp(off, 0, fk);
  c.save(); c.beginPath(); c.rect(x0 + 40, top - 4, w - 56, cap + 8); c.clip();
  const named = (l, wp) => ({ pre: l.kind === 'say' ? NAMED[l.who] + ': ' : '', text: toNamed(l.text) });
  const drawLine = (l, yb, version, alpha, clipX) => {
    const pre = version === 'named' ? (l.kind === 'say' ? (NAMED[l.who] || l.who) + ': ' : '') : l.pre, txt = version === 'named' ? toNamed(l.text) : l.text; const wrapped = UI.wrap(pre + txt, W2, `600 ${size}px Caveat`);
    const owner = CHIPS.find(ch => ch.id === l.who), od = owner ? CARDS.find(cd => cd.id === (owner.segs.filter(sg => sg.card && sg.t <= Math.max(l.tt, t >= 82.2 ? 99 : 0)).pop() || owner.segs[0]).card) : null;
    const chipCol = l.who === 'Pebble' ? COLS.pebble : (l.who && NAMED[l.who] ? CARDS.find(cd => cd.char === ({ Mia: 'mia', Leo: 'leo', Sam: 'sam', Biscuit: 'biscuit', Mum: 'mum' })[NAMED[l.who]]).col : '#cfc8bc');
    const tot = pre.length + txt.length, dur = l.kind === 'say' ? (SAYS.find(s => Math.abs(s.t0 - l.tt) < .06)?.dur ?? 2.4) * .8 : 1.1, p = version === 'named' || t > WIPE0 - 1 ? 1 : clamp((t - l.tt) / dur); let done = 0;
    c.save(); if (clipX) { c.beginPath(); c.rect(clipX[0], yb - 8, clipX[1] - clipX[0], l.h + 20); c.clip(); } c.globalAlpha *= alpha;
    wrapped.forEach((ln, i) => { const a = clamp((p * tot - done) / ln.length), yy = yb + (i + 1) * size * 1.12; c.font = `600 ${size}px Caveat`;
      if (l.kind === 'say' && i === 0) { const wl = c.measureText(pre).width; c.fillStyle = (version === 'raw' && t < 10) ? '#cfc8bc' : chipFill(l, t, version); roundRect(c, x0 + 62, yy - size * .82, wl - 2, size * 1.08, 8); c.fill(); c.strokeStyle = PAL.ink; c.lineWidth = 2; c.stroke(); c.fillStyle = PAL.ink; c.textAlign = 'left'; c.fillText(pre.slice(0, -2), x0 + 68, yy); UI.writeOn(ln.slice(pre.length), x0 + 66 + wl, yy, size, clamp((a * ln.length - pre.length) / Math.max(1, ln.length - pre.length)), { color: ink, nopen: a >= 1 }); }
      else UI.writeOn(ln, x0 + 66, yy, size, a, { color: l.kind === 'act' ? '#7a6b60' : ink, nopen: a >= 1 });
      done += ln.length + 1; });
    c.restore();
  };
  const chipFill = (l, tt, version) => { if (l.who === 'Pebble') return COLS.pebble; const nm = NAMED[l.who]; if (!nm) return '#cfc8bc'; const key = { Mia: 'mia', Leo: 'leo', Sam: 'sam', Biscuit: 'biscuit', Mum: 'mum' }[nm]; const cd = CARDS.find(c_ => c_.char === key && c_.id !== 'p6'); if (version === 'named') return cd.col;
    const ch = CHIPS.find(c_ => c_.id === l.who); const held = ch && ch.segs.some(sg => sg.card && sg.t <= tt); if (!held && l.who !== 'P1' && l.who !== 'P2' && l.who !== 'P3') return '#cfc8bc'; const known = cardName(cd, tt) || cd.known; return known ? cd.col : '#cfc8bc'; };
  let ii = 0;
  for (const l of lines) {
    const yb = top + l.y - off, vis = 1 - l.collapse; if (vis <= .02) { if (l.kind !== 'head') ii++; continue; } c.globalAlpha = clamp((yb + l.h - top + 4) / (l.h * 1.1)) * vis;
    if (l.kind === 'head') { c.fillStyle = PAL.violet; c.font = '700 30px Caveat'; c.textAlign = 'left'; const a = clamp((t - l.tt) / .5); c.globalAlpha *= a; c.fillText(l.text, x0 + 66, yb + 30); c.strokeStyle = 'rgba(120,100,200,.45)'; c.lineWidth = 2; c.beginPath(); c.moveTo(x0 + 66 + 80, yb + 22); c.lineTo(x0 + w - 60, yb + 22); c.stroke(); c.globalAlpha = 1; continue; }
    const tw = WIPE0 + ii * WIPE_DT, pw = clamp((t - tw) / .5); ii++;
    const hl = focus && fk > .5 && focus[2](l.L) && focus === FOCUS[1]; if (hl || (focus === FOCUS[0] && focus[2](l.L) && fk > .5)) { c.fillStyle = 'rgba(255,226,90,.6)'; roundRect(c, x0 + 56, yb - 2, w - 100, l.wrapped.length * size * 1.12 + 8, 8); c.fill(); }
    if (pw <= 0) drawLine(l, yb, 'raw', 1);
    else if (pw >= 1) drawLine(l, yb, 'named', 1);
    else { const wx = x0 + 60 + pw * (w - 90); drawLine(l, yb, 'raw', 1, [wx, x0 + w]); drawLine(l, yb, 'named', 1, [x0 + 40, wx]);
      const g = c.createLinearGradient(wx - 30, 0, wx + 6, 0); g.addColorStop(0, 'rgba(60,45,35,0)'); g.addColorStop(1, 'rgba(60,45,35,.55)'); c.fillStyle = g; c.fillRect(wx - 30, yb + 2, 36, l.h - 6); }
    c.globalAlpha = 1;
  }
  c.restore();
  if (t >= 138.4 && t < 146.9) { const k = backOut(seg(t, 138.4, 139.0)) * (1 - seg(t, 146.4, 146.9)); c.save(); c.translate(x0 + w - 120, top + 190); c.rotate(.06); c.scale(.4 * k, .4 * k); polaroidArt(c, 'feet', 'pink sneakers'); c.restore(); }
  if (t >= SAVE_T && t < SAVE_T + 2.4) { c.save(); ctx = c; c.globalAlpha = 1 - seg(t, SAVE_T + 1.8, SAVE_T + 2.4); UI.stamp(x0 + w / 2, y0 + 598, "tonight's memory saved", SAVE_T, t, { size: 52, rot: -.06, hold: true }); c.restore(); }
  c.restore();
}

function polaroidArt(c, kind, cap) {                                                // 300x350 polaroid at the origin: pink sneakers under the ketchup-stained sheet hem
  c.save(); c.shadowColor = 'rgba(40,25,10,.35)'; c.shadowBlur = 14; c.shadowOffsetY = 7; c.fillStyle = '#FFFBF0'; roundRect(c, -150, -180, 300, 350, 8); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = PAL.ink; c.lineWidth = 3; c.stroke();
  c.save(); c.beginPath(); c.rect(-132, -162, 264, 240); c.clip();
  const bg = c.createLinearGradient(0, -162, 0, 78); bg.addColorStop(0, '#A9AFD2'); bg.addColorStop(.7, '#C5CAE6'); bg.addColorStop(.71, '#D8C5A0'); bg.addColorStop(1, '#C9B48C'); c.fillStyle = bg; c.fillRect(-132, -162, 264, 240);
  c.lineJoin = 'round'; c.lineWidth = 3.4; c.strokeStyle = PAL.ink;
  c.save(); c.translate(0, 38); c.scale(1.32, 1.32); c.translate(0, -38);
  for (const sx of [-58, 62]) { c.fillStyle = '#FFFFFF'; c.beginPath(); c.rect(sx - 14, -20, 28, 52); c.fill(); c.stroke(); }                          // white socks
  for (const sx of [-58, 62]) { const fl = sx < 0 ? -1 : 1;
    c.fillStyle = '#FFFFFF'; c.beginPath(); c.ellipse(sx + fl * 8, 56, 56, 14, 0, 0, TAU); c.fill(); c.stroke();                                                   // sole
    c.fillStyle = '#FF7EB6'; c.beginPath(); c.moveTo(sx - 34, 52); c.quadraticCurveTo(sx - 36, 14, sx - 10, 12); c.lineTo(sx + 16, 12); c.quadraticCurveTo(sx + 62, 22, sx + 68, 52); c.quadraticCurveTo(sx + 16, 60, sx - 34, 52); c.closePath(); c.fill(); c.stroke();
    c.fillStyle = '#FFB6D8'; c.beginPath(); c.ellipse(sx + 46, 44, 20, 12, .2, 0, TAU); c.fill(); c.stroke();                                                          // toe cap
    c.strokeStyle = '#FFFFFF'; c.lineWidth = 4; for (let k = 0; k < 3; k++) { c.beginPath(); c.moveTo(sx - 4 + k * 12, 18 + k * 2); c.lineTo(sx + 6 + k * 12, 26 + k * 2); c.stroke(); } c.strokeStyle = PAL.ink; c.lineWidth = 3.4; }
  c.restore();
  c.fillStyle = '#FFFFFF'; c.beginPath(); c.moveTo(-132, -162); c.lineTo(132, -162); c.lineTo(132, -22);
  for (let i = 0; i < 6; i++) { const x1 = 132 - (i + 1) * 44; c.quadraticCurveTo(132 - i * 44 - 22, -22 + 26, x1, -22 + ((i + 1) % 2 ? 0 : 2)); }
  c.lineTo(-132, -162); c.closePath(); c.fill(); c.stroke();                                                                                                        // sheet hem
  c.fillStyle = PAL.ketchup; c.strokeStyle = '#9c2f25'; c.lineWidth = 2.6; c.beginPath(); for (let i = 0; i < 14; i++) { const a = i / 14 * TAU, r = (i % 2 ? 15 : 30) * (.9 + .25 * Math.sin(i * 2.3)); const px = -28 + Math.cos(a) * r * 1.2, py = -30 + Math.sin(a) * r * .8; if (i) c.lineTo(px, py); else c.moveTo(px, py); } c.closePath(); c.fill(); c.stroke();
  for (const [dx, h_] of [[-40, 30], [-14, 44], [10, 22]]) { c.beginPath(); c.moveTo(dx - 4, -26); c.lineTo(dx - 4, -26 + h_); c.arc(dx, -26 + h_, 4, Math.PI, 0, true); c.lineTo(dx + 4, -26); c.closePath(); c.fill(); }
  for (const [dx, dy, r] of [[34, -52, 6], [48, -38, 4], [-70, -30, 5]]) { c.beginPath(); c.arc(dx, dy, r, 0, TAU); c.fill(); }
  c.restore(); c.strokeStyle = PAL.ink; c.lineWidth = 3; c.strokeRect(-132, -162, 264, 240); c.fillStyle = 'rgba(205,170,110,.8)'; c.fillRect(-30, -198, 60, 26);
  c.fillStyle = PAL.ink; c.font = '600 34px Caveat'; c.textAlign = 'center'; c.fillText(cap, 0, 140); c.restore();
}
/* ZH subtitle wrapping: break at clause boundaries, keep Latin words (names) whole, never leave a short orphan */
function wrapZH(text, maxW, font) {
  ctx.font = font; const mw = s => ctx.measureText(s).width, out = [], NOSTART = /^[，。！？；：、…—）”’]/;
  for (const para of text.split('\n')) {
    const clauses = para.split(/(?<=[，。！？；：、]|——|……)/).filter(Boolean); let line = '';
    const breakLong = (cl) => { const toks = cl.match(/[A-Za-z0-9'’]+|[^A-Za-z0-9]/g) || []; let cur = ''; for (const tk of toks) { if (mw(cur + tk) > maxW && cur && !NOSTART.test(tk)) { out.push(cur); cur = tk; } else cur += tk; } return cur; };
    const lines = []; for (const cl of clauses) { if (mw(line + cl) <= maxW) line += cl; else { if (line) lines.push(line); line = ''; if (mw(cl) > maxW) line = breakLong(cl); else line = cl; } } if (line) lines.push(line);
    const plain = s => s.replace(/[，。！？；：、…—\s]/g, '').length;
    if (lines.length >= 2 && plain(lines[lines.length - 1]) < 4) { const j = lines.splice(-2).join(''), cut = [...j.matchAll(/(?<=[，。！？；：、]|——|……)/g)].map(m => m.index).filter(i => i > 0 && i < j.length).sort((a, b) => Math.abs(i_(a, j) - j.length / 2) - Math.abs(i_(b, j) - j.length / 2))[0]; if (cut) lines.push(j.slice(0, cut), j.slice(cut)); else { const m = Math.ceil(j.length / 2); lines.push(j.slice(0, m), j.slice(m)); } }
    out.push(...lines);
  } return out;
}
const i_ = (i) => i;

ZOOMKEYS.length = 0;
{ const K = [[0, [735, 540, 1]], [79.8, [735, 540, 1]], [80.7, [570, 760, 1.9]], [82.2, [570, 760, 1.9]], [83.3, [735, 540, 1]]];
  for (const h of HEIST2) K.push([h.grab - .55, [735, 540, 1]], [h.grab - .15, [672, 650, 1.75]], [h.back + .5, [672, 650, 1.75]], [h.back + 1.0, [735, 540, 1]]);
  K.push([170, [735, 540, 1]]); ZOOMKEYS.push(...K); }
/* ───────── new props: the sticky sheet on the sofa arm, wrappers stuck to the ketchup, the idea sparkle ───────── */
function wrapperShape(x, y, rot, col, s = 1) { push(); translate(x, y); rotate(rot); scale(s); paint([[-14, -2], [-6, -12], [10, -8], [15, 3], [4, 11], [-10, 8]], { wash: col, ink: PAL.ink, sw: .55 }); paint([[-4, -3], [5, -6], [7, 3], [-3, 4]], { wash: '#FFF4B0', washOp: 190, ink: null }); pop(); }
function sheetArmProp(t) {
  if (t < 63.9 || t >= 71.2) return; const k = easeOut(seg(t, 63.9, 64.3)), x = 336, y = 640 - (1 - k) * 18;
  paint([[x - 36, y - 56], [x + 6, y - 66], [x + 44, y - 50], [x + 40, y + 36], [x + 24, y + 62], [x + 4, y + 44], [x - 14, y + 68], [x - 34, y + 44]], { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: .9, curv: .3 });
  paint(ellPts(x + 4, y + 8, 22, 16, 12), { wash: PAL.ketchup, ink: '#9c2f25', sw: .5 }); blob(x - 14, y + 30, 5, PAL.ketchup); blob(x + 20, y + 28, 4, PAL.ketchup);
}
function ideaSpark(t) {
  if (t < 63.9 || t > 65.4) return; const an = anchorOf('mia', t); if (!an) return; const k = seg(t, 63.9, 64.2) * (1 - seg(t, 64.9, 65.4)), x = an.x + an.w * .9, y = an.top - 20;
  paint(ellPts(x, y, 15 * k, 17 * k, 12), { wash: '#FFE98A', ink: PAL.ink, sw: .8 }); paint(rectPts(x - 5 * k, y + 16 * k, 10 * k, 7 * k, 0), { wash: '#9a8a70', ink: PAL.ink, sw: .5 });
  for (let i = 0; i < 6; i++) { const a = i / 6 * TAU + t * 2; inkLine([[x + Math.cos(a) * 24 * k, y + Math.sin(a) * 26 * k], [x + Math.cos(a) * 38 * k, y + Math.sin(a) * 40 * k]], .9, '#F2C14A', 'ink', 0); }
}
function stainedSheetProp(t) {
  if (t < 93.2 || t >= 117.7) return; const k = easeOut(seg(t, 93.2, 93.7)), x = 650, y = 818 - (1 - k) * 90;
  paint([[x - 52, y + 4], [x - 36, y - 18], [x - 6, y - 10], [x + 20, y - 24], [x + 52, y - 6], [x + 58, y + 12], [x + 20, y + 22], [x - 30, y + 20]], { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: .8 });
  paint(ellPts(x + 6, y - 2, 20, 12, 12), { wash: PAL.ketchup, ink: '#9c2f25', sw: .5 }); blob(x - 20, y + 6, 6, PAL.ketchup); blob(x + 34, y + 2, 5, PAL.ketchup);
  wrapperShape(x - 4, y - 8, -.4, PAL.pink, .9); wrapperShape(x + 14, y - 2, .5, PAL.candle, .85); wrapperShape(x + 2, y + 4, 1.1, PAL.mint, .8);
}
function silhouettes(t) {
  const out = []; const mia = STATES.mia(t), leo = STATES.leo(t);
  if (t < 71.0) out.push({ k: 'arm', x: 336, y: 640, s: 34 }); if (t >= 70.0 && t < 72.4) out.push({ k: 'floor', x: 450, y: 885, s: 34 });
  if (mia) out.push({ k: t < 70.0 ? 'ghostS' : t < 71.0 ? 'kidS' : 'ghostStain', x: mia.x, y: mia.y, s: mia.s });
  if (leo) out.push({ k: t < 72.45 ? 'kidT' : 'ghostT', x: leo.x, y: leo.y, s: leo.s });
  return out;
}
function drawSilhouette(c, it) {
  const [x, y] = sp(it.x, it.y), s = it.s; c.beginPath();
  const ghost = (Wd, H, wave) => { const r = Wd * s / 2, top = y - H * s; c.moveTo(x - r * 1.1, y); c.lineTo(x - r * 1.1, top + r); c.arc(x, top + r, r * 1.1, Math.PI, 0); c.lineTo(x + r * 1.1, y); for (let i = 6; i >= 0; i--) c.lineTo(x - r * 1.1 + (i / 6) * r * 2.2, y - (i % 2 ? .35 : 0) * s - Math.sin(i + wave) * .1 * s); c.closePath(); };
  if (it.k === 'ghostS') ghost(3.6, 8.2, 0); else if (it.k === 'ghostStain') ghost(4.1, 10.5, 1); else if (it.k === 'ghostT') ghost(4.3, 9.7, 2);
  else if (it.k === 'kidS') { c.arc(x, y - 7.2 * s, 2.6 * s, 0, TAU); c.moveTo(x - 1.5 * s, y - 2.1 * s); c.rect(x - 1.5 * s, y - 4.6 * s, 3 * s, 2.5 * s); c.rect(x - 1.0 * s, y - 2.1 * s, .8 * s, 2.1 * s); c.rect(x + .2 * s, y - 2.1 * s, .8 * s, 2.1 * s); c.moveTo(x - 3.2 * s, y - 6 * s); c.ellipse(x - 3.3 * s, y - 5.2 * s, .8 * s, 1.8 * s, .3, 0, TAU); c.moveTo(x + 4.1 * s, y - 6 * s); c.ellipse(x + 3.3 * s, y - 5.2 * s, .8 * s, 1.8 * s, -.3, 0, TAU); }
  else if (it.k === 'kidT') { c.arc(x, y - 9.6 * s, 2.4 * s, 0, TAU); c.rect(x - 1.7 * s, y - 7.2 * s, 3.4 * s, 3.3 * s); c.rect(x - 1.2 * s, y - 3.9 * s, 1.0 * s, 3.9 * s); c.rect(x + .2 * s, y - 3.9 * s, 1.0 * s, 3.9 * s); }
  else if (it.k === 'floor') c.ellipse(x, y, 2.0 * s, .4 * s, 0, 0, TAU);
  else if (it.k === 'arm') { c.moveTo(x - 1.1 * s, y - 1.8 * s); c.quadraticCurveTo(x, y - 2.3 * s, x + 1.3 * s, y - 1.6 * s); c.lineTo(x + 1.1 * s, y + 1.2 * s); c.lineTo(x + .6 * s, y + 2.0 * s); c.lineTo(x - .2 * s, y + 1.4 * s); c.lineTo(x - 1.0 * s, y + 2.0 * s); c.closePath(); }
  c.fill();
}
function nightView(t) {
  const a = seg(t, 69.0, 69.4) * (1 - seg(t, 72.45, 72.62)); if (a <= .01) return; const its = silhouettes(t), SW = PX0 - 40;
  OVER.push((c) => { c.save(); c.beginPath(); c.rect(0, 0, SW, H); c.clip(); c.globalAlpha = a;
    c.fillStyle = '#071226'; c.fillRect(0, 0, SW, H); const g = c.createRadialGradient(SW / 2, H / 2, 200, SW / 2, H / 2, 760); g.addColorStop(0, 'rgba(40,70,120,.22)'); g.addColorStop(1, 'rgba(0,0,0,.65)'); c.fillStyle = g; c.fillRect(0, 0, SW, H);
    c.filter = 'blur(5px)'; c.fillStyle = 'rgba(120,150,205,.5)'; for (const it of its) drawSilhouette(c, it); c.filter = 'none';
    const f = Math.floor(t * 12); for (let i = 0; i < 700; i++) { const x = hash(i * 1.7 + f * 3.1) * SW, y = hash(i * 2.3 + f * 5.3) * H; c.fillStyle = `rgba(160,190,235,${.05 + hash(i + f) * .12})`; c.fillRect(x, y, 2 + hash(i) * 2, 2); }
    c.strokeStyle = 'rgba(0,0,0,.18)'; c.lineWidth = 2; for (let y = (f % 3) * 2; y < H; y += 6) { c.beginPath(); c.moveTo(0, y); c.lineTo(SW, y); c.stroke(); }
    c.strokeStyle = 'rgba(110,230,255,.55)'; c.lineWidth = 6; roundRect(c, 24, 24, SW - 48, H - 48, 40); c.stroke(); c.fillStyle = 'rgba(110,230,255,.7)'; c.font = '600 30px Caveat'; c.textAlign = 'left'; c.fillText('night view', 52, 70);
    c.restore(); });
}
function blackoutOverlay(t) { blackoutOverlayBase(t); nightView(t); }
