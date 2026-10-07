/* story.js - the whole film (cover, clips 1-5, end card) as pure functions of t.
 * The stage is the left 66% of the frame; Pebble's notebook is a permanent panel on the right that mirrors the memory pipeline (Characters rows,
 * Screenplay lines, inspect / repair / commit). Subtitles are dialogue only. Every label, stamp and notebook row is tied to a character id
 * (anchorOf / ROWS) and assertIds() throws if an event would land on anything but its target. */
const U = 36;
const SCY = y => lerp(.86, 1.04, clamp((y - 780) / 130));
const sFor = y => U * SCY(y);
const STAGE_CX = 610, STAGE_W = 1220, PX0 = 1232, PW = 668;
const T_FLIP = [2.35, 3.2], T_DOOR = 8.5, T_FLIP2 = [110.4, 111.3], T_C5 = 111.6, T_END = 155.0, DUR = 160.4, T_COMMIT = 80.8;
const NAMES = { mia: 'Mia', leo: 'Leo', sam: 'Sam', biscuit: 'Biscuit', pebble: 'Pebble', mum: 'Mum' };
const COLS = { mia: '#FF8FC4', leo: '#9d8cff', sam: '#43c99b', biscuit: '#c98a4b', pebble: '#58c7e8', mum: '#c79bff' };
const INKC = { mia: '#d2468a', leo: '#5d4bd0', sam: '#1b9a6c', biscuit: '#9a6a36', pebble: '#1f7fa8', mum: '#8a4fc0' };
const sp = (x, y) => toScreen(x, y);
const PEB = [1340, 915], PEB0 = [860, 915], MIA_C = [560, 800];                     // Pebble's rest spot (clear of every kid) and the spot in front of the cauldron
const HOME = { leo: [850, 880], mia: [1040, 815], sam: [1200, 855] };

/* ───────── camera ───────── */
const CAMKEYS = [[0, 900], [3.4, 900], [7.0, 1090], [11.0, 950], [17.6, 950], [30.2, 950], [31.4, 731], [34.6, 735], [36.4, 735], [110.3, 735], [111.3, 700], [165, 700]];
const ZOOMKEYS = [[0, [735, 540, 1]], [79.8, [735, 540, 1]], [80.7, [570, 760, 1.9]], [82.2, [570, 760, 1.9]], [83.3, [735, 540, 1]], [170, [735, 540, 1]]];
function camAt(t) {
  let cx = kf(t, CAMKEYS, ease), cy = 540, zoom = 1, sx = 0, sy = 0;
  if (t > 79) { const [zx, zy, zz] = kf(t, ZOOMKEYS, ease); if (zz > 1.001) { cx = zx; cy = zy; zoom = zz; } }
  for (const [t0, a] of [[T_DOOR, 12], [65.6, 22], [66.1, 14], [72.6, 8]]) if (t > t0 && t < t0 + .6) { const k = Math.exp(-(t - t0) * 6) * a, f = shakeXY(t, 1); sx = f[0] * k; sy = f[1] * k; }
  const hw = STAGE_W / (2 * zoom), hh = H / (2 * zoom);
  return { cx: clamp(cx, hw, W - hw) + sx, cy: clamp(cy, hh, H - hh) + sy, zoom, rot: 0, sx: STAGE_CX };
}
const lerpRun = (t, t0, t1, x0, y0, x1, y1, e = easeOut) => { const u = clamp((t - t0) / (t1 - t0)); return { x: lerp(x0, x1, e(u)), y: lerp(y0, y1, u), u }; };
const kidTop = (who, y, s, o = {}) => y - (KID[who].legH + KID[who].torsoH + KID[who].headR * 1.8) * s - (o.dy || 0) * s;

/* ───────── the script ───────── */
const SAYS = [
  { id: 'coming', who: 'pebble', t0: 5.3, dur: 2.6 }, { id: 'trick', who: 'mia', t0: 11.2, dur: 2.6 }, { id: 'whoa', who: 'leo', t0: 14.4, dur: 3.2 }, { id: 'mine', who: 'mia', t0: 18.2, dur: 2.6 },
  { id: 'catch', who: 'sam', t0: 21.4, dur: 2.6 }, { id: 'oof', who: 'leo', t0: 24.6, dur: 2.6 }, { id: 'hehe', who: 'sam', t0: 27.8, dur: 3.0 }, { id: 'shh', who: 'mia', t0: 31.2, dur: 2.8 },
  { id: 'costume', who: 'mia', t0: 36.4, dur: 2.6 }, { id: 'boo', who: 'sam', t0: 41.8, dur: 2.6 }, { id: 'cantsee', who: 'leo', t0: 45.0, dur: 2.7 }, { id: 'ghosttoo', who: 'mia', t0: 48.2, dur: 2.8 },
  { id: 'eek', who: 'sam', t0: 53.0, dur: 2.6 }, { id: 'phew', who: 'leo', t0: 59.2, dur: 2.9 }, { id: 'oj', who: 'leo', t0: 63.4, dur: 2.6, offscreen: true }, { id: 'giggle', who: 'mia', t0: 67.2, dur: 2.6 },
  { id: 'oneforme', who: 'mia', t0: 72.0, dur: 3.0 }, { id: 'whoate', who: 'mum', t0: 87.8, dur: 2.8 }, { id: 'notme', who: 'leo', t0: 91.0, dur: 2.5 }, { id: 'notmeeither', who: 'sam', t0: 94.2, dur: 2.6 },
  { id: 'biscuit?', who: 'mia', t0: 97.4, dur: 2.8 }, { id: 'woof?', who: 'biscuit', t0: 100.6, dur: 2.5 }, { id: 'answer', who: 'pebble', t0: 103.8, dur: 6.4 }, { id: 'happy', who: 'mia', t0: 112.4, dur: 2.8 }, { id: 'woof', who: 'biscuit', t0: 115.6, dur: 2.5 },
];
const speakingNow = (who, t) => SAYS.find(s => (s.who === who || (s.also || []).includes(who)) && t >= s.t0 + .1 && t <= s.t0 + s.dur - .35);
/* Who's who: one card per person Pebble knows about. Screenplay codes (P1..) sit on the card as chips; the name field goes ? -> guess (pencil) -> sure (ink + seal).
 * The screenplay text itself is never edited while the story runs. */
const CARDS = [
  { id: 'mum', char: 'mum', t: 3.9, known: true, col: '#c79bff', ink: '#8a4fc0', chips: [], name: [[3.9, 'ink', 'Mum', 1, 'known from before']], cues: ['face', 'voice'], lit: [3.9, 3.9] },
  { id: 'mia', char: 'mia', t: 10.0, col: '#FF8FC4', ink: '#d2468a', chips: [{ id: 'P1', t: 10.0, sure: 33.6 }], ev0: 'pink sneakers', cue: 'shoe', voiceAt: 11.2,
    name: [[17.0, 'pencil', 'Mia?', .45, 'called "Mia" by P2'], [33.6, 'ink', 'Mia', 1, 'called "Mia" again, same voice']] },
  { id: 'leo', char: 'leo', t: 10.3, col: '#9d8cff', ink: '#5d4bd0', chips: [{ id: 'P2', t: 10.3, sure: 26.9 }], ev0: 'tall, striped socks', cue: 'sock', voiceAt: 14.4,
    name: [[23.4, 'pencil', 'Leo?', .45, 'called "Leo" by P3'], [26.9, 'ink', 'Leo', 1, 'answered to "Leo"']] },
  { id: 'sam', char: 'sam', t: 10.6, col: '#43c99b', ink: '#1b9a6c', chips: [{ id: 'P3', t: 10.6, sure: 30.0 }], ev0: 'bell on his shoe', cue: 'bell', voiceAt: 21.4,
    name: [[30.0, 'ink', 'Sam', 1, 'self-introduction']] },
  { id: 'biscuit', char: 'biscuit', t: 33.3, col: '#c98a4b', ink: '#9a6a36', chips: [{ id: 'P4', t: 33.3, sure: 1e9 }], ev0: 'a dog', cue: 'ears', voiceAt: 1e9,
    name: [[33.3, 'pencil', 'Biscuit?', .6, 'named by P1']] },
];
const CARD_OF = Object.fromEntries(CARDS.map(c => [c.char, c]));
const CODE_OF = { mia: 'P1', leo: 'P2', sam: 'P3', biscuit: 'P4' };
const cardName = (card, t) => { let h = null; for (const [tt, st, tx, cf, ev] of card.name) if (t >= tt) h = { st, tx, cf, ev, t: tt }; return h; };
const chipOwner = code => CARDS.find(c => c.chips.some(ch => ch.id === code));
const SCRIPT1 = Object.freeze([   // [t, kind, code/label, text]: immutable while the story runs; codes only, names only inside quoted dialogue
  [3.8, 'head', null, 'Screenplay · 21:02'], [4.1, 'act', null, '(doorbell rings)'], [5.3, 'say', 'Pebble', 'Coming!'], [10.0, 'act', null, '(three kids burst in)'], [11.2, 'say', 'P1', 'Trick or treat!'], [14.4, 'say', 'P2', "Whoa… Mia, the candy's over there!"],
  [18.2, 'say', 'P1', 'Mine!'], [18.5, 'act', null, '(P1 runs to the candy bowl)'], [21.4, 'say', 'P3', 'Leo, catch!'], [22.1, 'act', null, '(P3 tosses a candy to P2)'], [24.6, 'say', 'P2', 'Oof!'], [24.9, 'act', null, '(P2 trips over the rug)'],
  [27.8, 'say', 'P3', "Hehe — I'm Sam, by the way."], [31.2, 'say', 'P1', "Shh — Biscuit's sleeping."], [33.6, 'act', null, '(a dog sleeps in the basket)'],
].map(Object.freeze));
const GLANCES = [10.0, 10.3, 10.6, 17.0, 23.4, 26.9, 30.0, 33.3, 33.6];
const NAMEWORDS = /\b(Mia|Leo|Sam|Biscuit|Mum)\b/;
SCRIPT1.forEach(([t, kind, who, text]) => { if (kind === 'act' && NAMEWORDS.test(text)) throw new Error('screenplay action contains a character name: ' + text); if (kind === 'say' && who !== 'Pebble' && !/^P\d$/.test(who)) throw new Error('screenplay speaker must be a code: ' + who); });
const beliefTag = (char, t) => {               // what the head tag says: the code, then the card's current name
  const card = CARD_OF[char]; if (!card || char === 'mum') return null; if (t < card.t + .1) return null; const h = cardName(card, t); return h ? h.tx : (CODE_OF[char]);
};
/* ───────── characters ───────── */
const MOUTHS = ['O', 'o', 'grin', 'smile', 'o', 'O'];
function talk(o, who, t) { if (speakingNow(who, t)) o.mouth = MOUTHS[Math.floor(t * 9) % MOUTHS.length]; return o; }
const bundleDraw = (s, sw) => { for (let i = 0; i < 3; i++) paint(ellPts((i - 1) * .55 * s, -(KID.mia.legH + KID.mia.torsoH + KID.mia.headR * 2 + .7 + i * .12) * s, 1.1 * s, .65 * s, 12), { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw * .7 }); };
function candyHand(s, sw) { candyArt(0, -.2 * s, s * .6, '#FF8FC4'); }
function pebbleState(t) {
  let x = PEB0[0], y = PEB0[1], o = { armL: .4, armR: .4, rolling: false, roll: 0 };
  if (t < 4.3) { o.look = [t > 3.9 ? .9 : .2, -.1]; }
  else if (t < 6.4) { const u = ease(seg(t, 4.5, 6.3)); x = lerp(PEB0[0], 1340, u); y = lerp(PEB0[1], 880, u); o.rolling = u > 0 && u < 1; o.roll = (t - 4.5) * 2.4; o.rot = .06 * Math.sin(t * 9) * (u > 0 && u < 1 ? 1 : 0); o.look = [1, -.1]; if (t >= 4.1 && t < 4.5) { o.emote = '?'; o.emoteK = seg(t, 4.1, 4.3); } if (u >= 1) o.sq = .1 * Math.exp(-(t - 6.3) * 10) * Math.cos((t - 6.3) * 28); }
  else if (t < T_DOOR) { x = 1340; y = 880; o.look = [1, -.2]; o.lid = .2 * seg(t, 7.8, 8.4); }
  else { x = lerp(1340, PEB[0], ease(seg(t, 9.3, 10.0))); y = lerp(880, PEB[1], ease(seg(t, 9.3, 10.0)));
    if (t < 9.2) { const st = Math.exp(-(t - T_DOOR) * 9); o.sq = .15 * st * Math.cos((t - T_DOOR) * 28); o.dy = st * .8 * Math.abs(Math.sin((t - T_DOOR) * 12)); o.emote = '!'; o.emoteK = seg(t, T_DOOR, T_DOOR + .15); o.pupil = .25; x = 1340 - 30 * st; }
    else { o.pupil = .5; const talker = SAYS.find(s => !s.offscreen && t >= s.t0 && t <= s.t0 + s.dur + .6); const tx = talker ? (talker.who === 'pebble' ? x : (STATES[talker.who](t) || { x: 1000 }).x) : 1100; o.look = [clamp((tx - x) / 350, -1, 1), -.1]; }
    if (GLANCES.some(g => t > g && t < g + 1.1) && !SAYS.some(s_ => s_.who === 'pebble' && t >= s_.t0 && t <= s_.t0 + s_.dur)) o.look = [1, -.35];
    const pulse = [17.0, 23.4, 30.0, 69.9, T_COMMIT].find(v => t > v && t < v + 1.2); o.glow = .4 + (pulse ? .7 * Math.exp(-(t - pulse) * 3.5) : 0); }
  if (t < 36) x = lerp(x, 1175, ease(seg(t, 29.8, 30.8)));
  if (t >= 36 && t < T_FLIP2[0]) { const tk = SAYS.find(s_ => s_.who !== 'pebble' && t >= s_.t0 && t <= s_.t0 + s_.dur + .5), tx = tk ? ((STATES[tk.who](t) || { x: 900 }).x) : 900; o.look = [clamp((tx - x) / 350, -1, 1), -.1]; if (t > 52.4 && t < 58.4) { o.glow = .95; o.pupil = .78; o.lid = 0; } }
  if (t >= T_FLIP2[0]) { x = PEB[0]; y = 900; o.look = [Math.sin(t * 1.2) * .5, 0]; o.glow = .45;
    if (t >= 103.2 && t < 111) { const u = easeOut(seg(t, 103.0, 104.2)); x = lerp(PEB[0], PEB[0] - 40, u); o.look = [-.2, -.4]; o.armR = .9 + .15 * Math.sin(t * 8); o.handR = penHand; o.glow = .6 + .5 * Math.exp(-((t - 103.4) % 9) * 3); o.pupil = .66; o.lid = 0; if (t > 103.2 && t < 104.0) { o.emote = '!'; o.emoteK = seg(t, 103.2, 103.4); } }
    if (t >= 111) { o.look = [.8, -.1]; x = PEB[0] - 40; o.glow = .5; o.pupil = .45; } }
  if (speakingNow('pebble', t)) { o.glow = .9 + .1 * Math.sin(t * 20); o.pupil = .62 + .1 * Math.sin(t * 16); o.dy = (o.dy || 0) + .12 * Math.abs(Math.sin(t * 13)); }
  return { form: 'pebble', x, y, o, s: 34 };
}
function penHand(s, sw) { push(); rotate(.3); paint(rectPts(0, -.18 * s, 1.1 * s, .3 * s, 0), { wash: '#6b4a14', ink: PAL.ink, sw: sw * .4 }); paint([[0, -.18 * s], [-.35 * s, .0], [0, .12 * s]], { wash: PAL.ink, ink: null }); pop(); }

function miaState(t) {
  if (t < T_DOOR + .05) return null;
  const o = { who: 'mia', blush: true }; let x = 890, y = 815;
  if (t < 36.4) {
    if (t < 9.7) { const r = lerpRun(t, T_DOOR + .05, 9.7, 1700, 796, 890, 815); x = r.x; y = r.y; o.run = (t - T_DOOR) * 2.7; o.eyes = 'happy'; o.mouth = 'laugh'; o.turn = -.6; o.rot = -.04; }
    else if (t < 11.2) { o.sq = .12 * Math.exp(-(t - 9.7) * 7) * Math.cos((t - 9.7) * 22); o.eyes = 'dot'; o.mouth = 'grin'; o.turn = -.3; o.look = [-.5, 0]; }
    else if (t < 13.8) { const u = seg(t, 11.2, 11.9); o.aL = o.aR = lerp(-1.25, 1.6, easeOut(u)); o.eyes = 'happy'; o.mouth = 'grin'; o.dy = Math.sin(seg(t, 11.5, 12.1) * Math.PI) * .8; o.sq = t < 11.5 ? -.09 * seg(t, 11.2, 11.5) : 0; }
    else if (t < 18.2) { o.turn = .5; o.eyes = t < 17.0 ? 'dot' : 'wide'; o.look = [-.9, -.2]; o.mouth = t > 17.0 ? 'o' : 'smile'; if (t > 17.0 && t < 17.5) { o.emote = '!'; o.emoteK = seg(t, 17.0, 17.2); } }
    else if (t < 19.7) { const r = lerpRun(t, 18.2, 19.7, 890, 815, MIA_C[0], MIA_C[1], ease); x = r.x; y = r.y; o.run = (t - 18.2) * 2.8; o.eyes = 'happy'; o.mouth = 'O'; o.turn = -.9; o.rot = -.05; }
    else { x = MIA_C[0]; y = MIA_C[1]; o.turn = -.7; o.aL = 1.0 + .1 * Math.sin(t * 14); o.eyes = 'happy'; o.look = [-.8, -.2]; o.mouth = 'grin'; o.sq = -.03 * Math.sin(t * 9); o.dy = .06 * Math.abs(Math.sin(t * 9));
      if (t < 20.0) o.sq = .12 * Math.exp(-(t - 19.7) * 8) * Math.cos((t - 19.7) * 22);
      if (t >= 31.2) { const k = seg(t, 31.1, 31.5); o.turn = lerp(-.7, -.9, k); o.look = [-.6, .5]; o.eyes = 'sly'; o.aR = lerp(-1.25, 1.2, easeOut(k)); o.aL = -.9; o.mouth = 'smile'; o.dy = 0; } }
  } else if (t < 40.7) {                                                            // clip 2: fetch the sheets from behind the sofa, carry them back, throw them up
    if (t < 37.6) { const r = lerpRun(t, 36.4, 37.6, MIA_C[0], MIA_C[1], 340, 795, ease); x = r.x; y = r.y; o.run = (t - 36.4) * 2.8; o.eyes = 'happy'; o.mouth = 'O'; o.turn = -.9; }
    else if (t < 38.2) { x = 340; y = 795; o.aL = o.aR = lerp(-1.25, 1.5, easeOut(seg(t, 37.6, 38.0))); o.draw = t > 37.9 ? bundleDraw : undefined; o.eyes = 'happy'; o.mouth = 'grin'; o.sq = t < 37.9 ? -.1 * seg(t, 37.6, 37.9) : .1 * Math.exp(-(t - 37.9) * 8); o.turn = 0; }
    else if (t < 39.5) { const r = lerpRun(t, 38.2, 39.5, 340, 795, HOME.mia[0], HOME.mia[1], ease); x = r.x; y = r.y; o.run = (t - 38.2) * 2.8; o.aL = o.aR = 1.4; o.draw = bundleDraw; o.eyes = 'happy'; o.mouth = 'grin'; o.turn = .6; }
    else { x = HOME.mia[0]; y = HOME.mia[1]; const u = seg(t, 39.5, 40.0); o.aL = o.aR = lerp(1.4, 1.65, u); o.draw = t < 39.8 ? bundleDraw : undefined; o.eyes = 'happy'; o.mouth = 'laugh'; o.dy = Math.sin(seg(t, 39.6, 40.3) * Math.PI) * .9; o.sq = t < 39.6 ? -.1 * seg(t, 39.5, 39.6) : 0; o.turn = 0; }
    talk(o, 'mia', t);
  } else return null;
  talk(o, 'mia', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function leoState(t) {
  if (t < T_DOOR + .45 || (t >= 40.7 && t < T_C5 + .1)) return null; const o = { who: 'leo', blush: true }; let x = 710, y = 875;
  if (t < 10.1) { const r = lerpRun(t, T_DOOR + .45, 10.1, 1780, 880, 710, 875, ease); x = r.x; y = r.y; o.run = (t - T_DOOR) * 2.3; o.eyes = 'wide'; o.mouth = 'o'; o.turn = -.7; o.sq = t > 9.9 ? .1 * Math.exp(-(t - 10.1) * 6) : 0; }
  else if (t < 14.4) { o.turn = -.4; o.eyes = 'dot'; o.look = [.6, 0]; o.mouth = 'smile'; o.sq = -.02 * Math.sin(t * 4); }
  else if (t < 17.8) { const k = seg(t, 14.4, 14.9); o.turn = -.8; o.aL = lerp(-1.25, .75, easeOut(k)); o.eyes = 'wide'; o.look = [-1, -.2]; o.mouth = 'O'; }
  else if (t < 21.4) { o.turn = -.3; o.aL = lerp(.75, -1.25, seg(t, 17.8, 18.3)); o.eyes = 'dot'; o.look = [-1, 0]; o.mouth = 'flat'; o.sq = -.02 * Math.sin(t * 4); }
  else if (t < 24.6) { const k = seg(t, 21.4, 22.0); o.turn = .8; o.look = [1, -.3]; o.eyes = 'wide'; o.mouth = 'o'; o.aL = o.aR = lerp(-1.25, 1.0, easeOut(k)); if (t > 22.9) { o.aL = lerp(1.0, 1.6, seg(t, 22.9, 23.2)); o.sq = t > 23.0 ? -.1 * Math.exp(-(t - 23.0) * 9) * Math.cos((t - 23.0) * 25) : 0; o.eyes = 'happy'; o.mouth = 'grin'; o.handL = candyHand; } }
  else if (t < 28.5) { const u = seg(t, 24.6, 26.4); x = 710 - 10 * easeOut(u); o.rot = Math.sin(Math.min(1, u * 1.2) * Math.PI) * .55 * (u < .65 ? 1 : 1 - (u - .65) * .5); o.dy = Math.max(0, Math.sin(u * Math.PI * 1.6)) * .35; o.aL = 1.2 + Math.sin(t * 28) * .7; o.aR = 1.2 - Math.sin(t * 28) * .7; o.walk = (t - 24.6) * 3.2; o.eyes = u < .6 ? 'wide' : 'closed'; o.brows = 'worried'; o.mouth = 'O'; o.turn = .4; if (u > .1 && u < .5) o.emote = '!';
    if (u >= 1) { x = 720; o.rot = .04; o.walk = undefined; o.aL = o.aR = -1.1; o.eyes = 'dot'; o.mouth = 'wobble'; o.turn = -.4; o.sq = .1 * Math.exp(-(t - 26.4) * 6) * Math.cos((t - 26.4) * 20); o.emote = 'sweat'; o.emoteK = seg(t, 26.4, 26.6); } }
  else if (t < 40.7) { x = 720 + 130 * ease(seg(t, 36.0, 37.0)); o.turn = .6; o.look = [1, 0]; o.eyes = 'dot'; o.mouth = 'smile'; o.sq = 0;
    if (t >= 38.6 && t < 39.5) { o.eyes = 'wide'; o.aR = lerp(-1.25, .8, seg(t, 38.6, 39.0)); } if (t >= 39.5) { const u = seg(t, 39.9, 40.4); o.aL = o.aR = lerp(-1.25, 1.6, easeOut(u)); o.eyes = 'closed'; o.mouth = 'grin'; if (t > 40.3) o.dy = .7 * Math.sin(seg(t, 40.3, 40.7) * Math.PI); } }
  else { const r = lerpRun(t, 85.6, 87.0, 1500, 872, 810, 872); x = r.x; y = r.y; o.turn = -.5;                       // morning: a big yawn on the way in
    if (t < 87.0) { o.walk = (t - 85.6) * 1.3; o.eyes = 'closed'; o.mouth = 'O'; o.aL = o.aR = 1.0 + Math.sin(t * 5) * .1; }
    else { o.eyes = 'dot'; o.mouth = 'flat'; o.look = [-.6, 0]; o.sq = -.02 * Math.sin(t * 4.2);
      if (t >= 91.0 && t < 93.5) { o.turn = lerp(-.5, -.2, easeOut(seg(t, 91, 91.3))); o.rot = .05 * Math.sin(t * 14); o.eyes = 'closed'; o.mouth = 'O'; o.brows = 'angry'; }          // shaking his head
      if (t >= 110.9) { const k = easeOut(seg(t, 110.9, 111.4)); o.turn = lerp(-.5, 1, k); o.eyes = 'wide'; o.mouth = 'o'; o.look = [1, 0]; } } }
  talk(o, 'leo', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function samState(t) {
  if (t < T_DOOR + .8 || (t >= 40.7 && t < T_C5 + .1)) return null; const o = { who: 'sam', blush: true }; let x = 1100, y = 850;
  if (t < 10.4) { const r = lerpRun(t, T_DOOR + .8, 10.4, 1850, 805, 1100, 850, ease); x = r.x; y = r.y; const ph = (t - T_DOOR - .8) * 1.3; o.walk = ph * 1.5; o.dy = Math.abs(Math.sin(ph * Math.PI * 3)) * .9; o.eyes = 'closed'; o.mouth = 'laugh'; o.aL = .9 + Math.sin(t * 14) * .3; o.aR = .9 - Math.sin(t * 14) * .3; o.turn = -.6; o.emote = frac(ph * 3) < .25 ? 'note' : null; o.emoteK = frac(ph * 3) * 4; }
  else if (t < 40.7) { x = t < 36.2 ? 1100 : lerp(980, 1200, ease(seg(t, 36.2, 37.8))); o.turn = -.6; o.eyes = 'closed'; o.mouth = 'laugh'; o.dy = .25 * Math.abs(Math.sin(t * 7)); o.sq = -.04 * Math.sin(t * 7); o.rot = .04 * Math.sin(t * 7);
    if (frac(t * 2.2) < .1) { o.emote = 'note'; o.emoteK = frac(t * 2.2) * 10; }
    if (t >= 21.4 && t < 24.0) { const k = seg(t, 21.4, 21.9); o.mouth = 'O'; o.eyes = 'happy'; o.turn = -.9; o.dy = 0; o.rot = 0; o.emote = null; o.aR = t < 22.1 ? lerp(-1.25, 1.7, easeOut(k)) : lerp(1.7, -.4, easeOut(seg(t, 22.1, 22.35))); o.handR = t < 22.15 ? candyHand : null; o.sq = t > 22.1 && t < 22.5 ? .1 * Math.sin((t - 22.1) * 18) : 0; }
    if (t >= 27.8 && t < 30.8) { o.aR = lerp(-1.25, .3, easeOut(seg(t, 27.8, 28.2))); o.eyes = 'happy'; o.turn = 0; o.dy = .35 * Math.abs(Math.sin(t * 8)); }
    if (t >= 39.7) { const u = seg(t, 39.9, 40.4); o.aL = o.aR = lerp(-1.25, 1.6, easeOut(u)); o.eyes = 'closed'; o.mouth = 'laugh'; if (t > 40.3) o.dy = .7 * Math.sin(seg(t, 40.3, 40.7) * Math.PI); } }
  else { const r = lerpRun(t, 86.0, 87.4, 1540, 880, 1170, 880); x = r.x; y = r.y; o.turn = -.5;
    if (t < 87.4) { o.walk = (t - 86.0) * 1.3; o.eyes = 'closed'; o.mouth = 'flat'; o.aL = o.aR = .5 + Math.sin(t * 9) * .2; }
    else { o.eyes = 'dot'; o.mouth = 'smile'; o.look = [-.8, 0]; o.dy = .1 * Math.abs(Math.sin(t * 4));
      if (t >= 94.2 && t < 96.7) { o.eyes = 'closed'; o.brows = 'angry'; o.rot = .05 * Math.sin(t * 14); o.dy = 0; o.mouth = 'O'; }
      if (t >= 111.2) { const k = easeOut(seg(t, 111.2, 111.7)); o.turn = lerp(-.5, -1, k); o.eyes = 'wide'; o.mouth = 'o'; o.look = [-1, 0]; o.dy = 0; } } }
  if (t >= 22.6 && t < 36.2) x -= 120 * ease(seg(t, 22.6, 23.6));
  talk(o, 'sam', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function mumState(t) {
  if (t < T_C5 + .2) return null; const o = { who: 'mum', blush: true }; const r = lerpRun(t, 85.4, 86.8, 150, 760, 420, 885, ease); let x = r.x, y = r.y;
  o.handL = cauldronHand; o.aL = .7;
  if (t < 86.8) { o.walk = (t - 85.4) * 1.2; o.eyes = 'dot'; o.mouth = 'flat'; o.turn = .6; }
  else { o.aR = .15 + .1 * Math.sin(t * 2); o.eyes = 'dot'; o.mouth = 'flat'; o.brows = 'angry'; o.turn = .7; o.sq = .05 * Math.exp(-(t - 86.8) * 6) * Math.cos((t - 86.8) * 20); o.emote = t > 87.0 && t < 87.8 ? 'anger' : null; o.emoteK = seg(t, 87, 87.2);
    if (t > 87.7 && t < 90.7) { o.brows = 'up'; o.rot = -.02 * Math.sin(t * 22); }
    if (t >= 110.6) { const k = easeOut(seg(t, 110.6, 111.1)); o.turn = lerp(.7, 1, k); o.eyes = 'wide'; o.look = [1, 0]; o.mouth = 'o'; o.brows = 'up'; } }
  talk(o, 'mum', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
function cauldronHand(s, sw) { push(); translate(-.2 * s, .5 * s); paint(ellPts(0, .4 * s, 1.5 * s, 1.15 * s, 18), { wash: '#2a2733', ink: PAL.ink, sw: sw * .9 }); paint(ellPts(0, -.45 * s, 1.35 * s, .35 * s, 16), { wash: '#12101a', ink: PAL.ink, sw: sw * .8 }); inkLine([[-1.4 * s, .1 * s], [-1.9 * s, .5 * s], [-1.4 * s, .8 * s]], sw * .8, PAL.ink, 'ink', .5); pop(); }
function morningMia(t) {
  const ball = (s, sw) => { const bx = -2.25 * s, by = -(KID.mia.legH + 1.5) * s; paint(ellPts(bx, by, 1.15 * s, 1.0 * s, 16), { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw * .8 }); inkLine([[bx - .5 * s, by - .3 * s], [bx + .3 * s, by + .2 * s]], sw * .4, '#b7c0da', 'inkfine', .5); };
  const o = { who: 'mia', blush: true, draw: ball }; const r = lerpRun(t, 86.4, 87.8, 1520, 800, 1000, 805); let x = r.x, y = r.y;
  o.aL = -.8; o.aR = -1.2; o.turn = -.6;
  if (t < 87.8) { o.walk = (t - 86.4) * 1.5; o.eyes = 'sly'; o.mouth = 'wobble'; o.look = [-1, 0]; }
  else { o.eyes = 'sly'; o.mouth = 'smirk'; o.look = [-.8, 0]; o.sq = -.02 * Math.sin(t * 5);
    if (t >= 97.4 && t < 100.2) { const k = seg(t, 97.4, 97.8); o.aR = lerp(-1.2, .9, easeOut(k)); o.eyes = 'wide'; o.mouth = 'grin'; o.look = [-1, .2]; }                     // points at the dog
    if (t >= 101 && t < 102) { o.emote = 'sweat'; o.emoteK = seg(t, 101, 101.2); } }
  if (t >= 110.4) { const k = seg(t, 110.4, 110.8); o.eyes = 'wide'; o.mouth = 'wobble'; o.brows = 'worried'; o.turn = lerp(-.6, 0, k); o.look = [0, 0]; o.emote = 'sweat'; o.emoteK = k; o.sq = .1 * Math.exp(-(t - 110.4) * 6) * Math.cos((t - 110.4) * 20);
    if (t > 112.4) { o.eyes = 'sad'; o.mouth = 'wobble'; o.emote = null; o.brows = 'worried'; } if (t > 113.5) { o.eyes = 'happy'; o.mouth = 'grin'; o.brows = null; o.aR = .9 + Math.sin(t * 12) * .35; } }
  talk(o, 'mia', t); return { form: 'kid', x, y, o, s: sFor(y) };
}
const HEIST = [{ go: 72.0, grab: 73.0, back: 73.4, done: 74.8 }, { go: 74.8, grab: 75.7, back: 76.1, done: 77.6 }, { go: 77.6, grab: 78.5, back: 78.9, done: 80.4 }];
function ghostMiaState(t) {                                                         // Mia as a ghost: plain sheet in clip 2, the stained sheet after the swap in clip 3, then the heist
  const land = t - 40.7, o = { kind: 'mia', ph: 0, lift: .25 + .22 * Math.sin(t * 2.6), walk: t * .9, look: [Math.sin(t * 1.1) * .9, -.2], rot: .04 * Math.sin(t * 2.1) };
  o.sq = -.18 * Math.exp(-land * 7) * Math.cos(land * 22) + (land < .15 ? .2 : 0); o.eyes = land < .45 ? 'wide' : (t < 44 ? 'dot' : 'sly');
  let [x, y] = HOME.mia; x += Math.sin(t * .8) * 8;
  if (t >= 47.6 && t < 52.4) {                                                      // runs to Biscuit's basket, drapes the mini sheet, comes back
    const go = ease(seg(t, 47.6, 48.6)), back = ease(seg(t, 50.8, 52.2)); x = lerp(HOME.mia[0], 370, go - back * go); y = lerp(HOME.mia[1], 935, go - back * go); o.walk = (go > 0 && go < 1 || back > 0 && back < 1) ? t * 2.2 : undefined; o.stride = 1; o.eyes = 'happy'; if (t > 49.4 && t < 50.2) o.reach = [1.1, 2.0];
  }
  if (t >= 52.4 && t < 56.2) { o.eyes = 'wide'; o.lift = .15; o.look = [Math.sin(t * 3), -.2]; }
  if (t >= 56.2 && t < 58.4) { const u = ease(seg(t, 56.2, 57.9)); x = lerp(HOME.mia[0], HOME.leo[0], u); y = lerp(HOME.mia[1], HOME.leo[1], u) - Math.sin(u * Math.PI) * 30; o.walk = (t - 56.2) * 1.8; o.stride = 1; o.lift = .1; o.eyes = 'sly'; o.rot = -.06 * Math.sin(u * Math.PI); if (t > 57.0) { o.stain = true; o.sheetK = 1.28; } }
  if (t >= 58.4) { o.stain = true; o.sheetK = 1.28; x = HOME.leo[0]; y = HOME.leo[1]; o.lift = .12 + .08 * Math.sin(t * 3); o.eyes = 'happy';
    if (t < 62.4) { o.look = [1, 0]; o.eyes = 'sly'; }
    else if (t < 63.8) { const u = ease(seg(t, 62.4, 63.8)); x = lerp(HOME.leo[0], MIA_C[0], u); y = lerp(HOME.leo[1], MIA_C[1] - 5, u); o.walk = (t - 62.4) * 1.8; o.stride = .9; o.stepH = .8; o.lift = .05; o.eyes = 'sly'; o.look = [-1, 0]; }
    else if (t < 71.0) { x = MIA_C[0]; y = MIA_C[1] - 5; o.walk = undefined; o.reach = [1.0 + .2 * Math.sin(t * 8), 2.2]; o.lift = .08; o.eyes = 'happy';
      if (t > 66.3 && t < 67.2) { o.reach = [.5, 1.2]; o.eyes = 'wide'; o.emote = '!'; o.emoteK = seg(t, 66.3, 66.5); o.sq = .12 * Math.exp(-(t - 66.3) * 8) * Math.cos((t - 66.3) * 24); }
      if (t >= 67.2) { o.reach = undefined; o.eyes = 'happy'; o.sq = .08 * Math.sin(t * 22); o.walk = (t - 67.2) * 2.4; o.stride = .45; o.lift = 0; o.look = [-1, 0]; } }
    else { const h = HEIST.find(h => t >= h.go - .1 && t < h.done + .6); heist(o, t, h); x = o.__x ?? x; y = o.__y ?? y; }
  }
  talk(o, 'mia', t); return { form: 'ghost', x, y, o, s: sFor(y) * .96 };
}
function heist(o, t, h) {
  let x = 1030, y = 880; o.stain = true; o.sheetK = 1.28; o.eyes = 'sly'; o.stepH = .95; o.stride = 1.15; o.lift = .05; o.look = [-1, 0];
  if (h) { if (t < h.grab - .05) { const u = easeOut(seg(t, h.go, h.grab - .05)); x = lerp(1030, MIA_C[0], u); y = lerp(880, MIA_C[1] - 5, u); o.walk = (t - h.go) * 1.6; o.rot = -.07 * Math.sin(u * Math.PI * 3); }
    else if (t < h.back) { x = MIA_C[0]; y = MIA_C[1] - 5; o.walk = undefined; o.reach = [1.0 + .2 * Math.sin(t * 20), 2.3]; o.sq = -.08 * Math.sin(seg(t, h.grab - .05, h.back) * Math.PI) + .06 * Math.sin(t * 25); o.eyes = 'happy'; }
    else { const u = ease(seg(t, h.back, h.done)); x = lerp(MIA_C[0], 1030, u); y = lerp(MIA_C[1] - 5, 880, u); o.walk = (t - h.back) * 1.6; o.rot = .07 * Math.sin(u * Math.PI * 3); o.look = [1, 0]; } }
  else o.eyes = 'happy';
  o.__x = x; o.__y = y;
}
function ghostLeoState(t) {
  const land = t - 40.7, o = { kind: 'leo', ph: 1.7, stain: true, lift: .25 + .22 * Math.sin(t * 2.6 + 1.7), walk: t * .9 + .3, look: [Math.sin(t * 1.1 + 1) * .9, -.2], rot: .04 * Math.sin(t * 2.1 + 1.7) };
  o.sq = -.18 * Math.exp(-land * 7) * Math.cos(land * 22) + (land < .15 ? .2 : 0); o.eyes = land < .45 ? 'wide' : (t < 44 ? 'dot' : 'happy');
  let [x, y] = HOME.leo; x += Math.sin(t * .8 + 1) * 8;
  if (t >= 45.0 && t < 48.6) { const go = ease(seg(t, 45.0, 46.3)), back = ease(seg(t, 47.4, 48.6)), k = go - back * go; x = lerp(HOME.leo[0], 690, k); y = lerp(HOME.leo[1], 800, k); o.walk = (go > 0 && go < 1 || back > 0 && back < 1) ? (t - 45) * 1.8 : undefined; o.stride = .9; o.lift = .1;      // bumps into the sofa, bounces off
    if (t > 46.3 && t < 47.4) { o.sq = .15 * Math.exp(-(t - 46.3) * 7) * Math.cos((t - 46.3) * 22); o.rot = -.12 * Math.exp(-(t - 46.3) * 6); o.eyes = 'wide'; o.emote = '!'; o.emoteK = seg(t, 46.3, 46.5); } }
  if (t >= 52.4 && t < 56.2) { o.eyes = 'wide'; o.lift = .15; }
  if (t >= 56.2) { o.stain = t < 57.0; o.sheetK = t < 57.0 ? 1 : .88; const u = ease(seg(t, 56.2, 57.9)); x = lerp(HOME.leo[0], HOME.mia[0], u); y = lerp(HOME.leo[1], HOME.mia[1], u) + Math.sin(u * Math.PI) * 24; o.walk = (t - 56.2) * 1.8; o.stride = 1; o.lift = .1; o.eyes = 'sly'; o.rot = .06 * Math.sin(u * Math.PI);
    if (t >= 58.4) { o.stain = false; o.sheetK = .88; x = HOME.mia[0]; y = HOME.mia[1]; o.eyes = 'happy'; o.walk = undefined; o.lift = .2 + .1 * Math.sin(t * 3); }
    if (t >= 58.4 && t < 59.2) { o.aU = 1; o.armsUp = 1.3 * seg(t, 58.5, 58.9); o.wave = .1; o.eyes = 'closed'; }                        // a stretch
    if (t >= 59.4) { const u2 = ease(seg(t, 59.4, 62.6)); x = lerp(HOME.mia[0], 175, u2); y = lerp(HOME.mia[1], 770, u2); o.walk = (t - 59.4) * 2.0; o.stride = .8; o.lift = 0; o.eyes = 'happy'; const sk = lerp(1, .62, u2 * u2); if (t >= 62.7) return null; return { form: 'ghost', x, y, o, s: sFor(y) * .96 * sk }; } }
  talk(o, 'leo', t); return { form: 'ghost', x, y, o, s: sFor(y) * .96 };
}
function ghostSamState(t) {
  const land = t - 40.7, o = { kind: 'sam', ph: 3.4, lift: .25 + .22 * Math.sin(t * 2.6 + 3.4), walk: t * .9 + .6, look: [Math.sin(t * 1.1 + 2) * .9, -.2], rot: .04 * Math.sin(t * 2.1 + 3.4) };
  o.sq = -.18 * Math.exp(-land * 7) * Math.cos(land * 22) + (land < .15 ? .2 : 0); o.eyes = land < .45 ? 'wide' : 'happy';
  if (t >= 41.8 && t < 44.5) { o.armsUp = 1.2; o.wave = .5; o.eyes = 'wide'; o.lift += Math.abs(Math.sin((t - 41.8) * 5)) * .6; }                // "Boooo!"
  if (t >= 52.4 && t < 58.4) { o.eyes = 'wide'; o.lift = .2; o.look = [Math.sin(t * 3), -.3]; }
  if (t >= 58.4) { o.eyes = t < 62 ? 'wide' : 'happy'; o.look = [-1, 0]; }
  talk(o, 'sam', t); return { form: 'ghost', x: HOME.sam[0] + Math.sin(t * .8 + 2) * 8, y: HOME.sam[1], o, s: sFor(HOME.sam[1]) * .96 };
}
function biscuitState(t) {
  const [bx, by] = [LAYOUT.basket[0], LAYOUT.basket[1] - 10];
  if (t >= T_C5 + .1) { const u = easeOut(seg(t, 85.4, 86.6)); const x = lerp(bx, 600, u), y = lerp(by, 915, u); const up = t >= 100.6 && t < 103.5; const wagK = t >= 115.0 ? 1.8 : t >= 111.5 ? 1.3 : .8; return { form: 'dog', x, y, o: { trot: u < 1 ? (t - 85.4) * 2.4 : undefined, wag: wagK, tongue: true, eyes: up || t >= 111.5 ? 'wide' : 'dot', earUp: up || (t >= 111.5 && t < 112.2), flip: false, dy: up && t < 101.0 ? .35 * Math.abs(Math.sin((t - 100.6) * 14)) : 0 }, s: 36 }; }
  if (t < 49.4 || t >= 70.9) { if (t >= 70.9 && t < 71.5) { } return { form: 'dog', x: bx, y: by, o: { sleep: true }, s: 36 }; }
  if (t < 50.4) return { form: 'dog', x: bx, y: by, o: { sleep: false, eyes: t < 49.7 ? 'closed' : 'dot', wag: 0, sq: .15 * Math.exp(-(t - 49.4) * 6) * Math.cos((t - 49.4) * 24), tongue: t > 49.9 }, s: 36 };
  return { form: 'dogghost', x: bx + 20, y: by, o: { kind: 'dog', wag: .8, lift: .22 + .2 * Math.sin(t * 3), flip: false, ph: 3, eyes: t < 52 ? 'wide' : 'happy', look: [1, 0], sq: t < 50.8 ? .2 * Math.exp(-(t - 50.4) * 7) * Math.cos((t - 50.4) * 26) : 0 }, s: 36 };
}
const STATES = { pebble: pebbleState, biscuit: biscuitState, mum: mumState,
  mia: (t) => t >= 40.7 && t < T_C5 + .1 ? ghostMiaState(t) : (t >= T_C5 + .1 ? morningMia(t) : miaState(t)),
  leo: (t) => t >= 40.7 && t < T_C5 + .1 ? (t < 63.0 ? ghostLeoState(t) : null) : leoState(t),
  sam: (t) => t >= 40.7 && t < T_C5 + .1 ? ghostSamState(t) : samState(t) };

function drawState(id, t, dark = 0) {
  const st = STATES[id](t); if (!st) return null; const { form, x, y, o, s } = st;
  if (form === 'kid') kid(x, y, s, o); else if (form === 'ghost' || form === 'dogghost') ghost(x, y, s, { ...o, dark }); else if (form === 'pebble') pebble(x, y, s, o); else if (form === 'dog') biscuit(x, y, s, o);
  return st;
}
function anchorOf(id, t) {                                                           // the head anchor every tag / frame / tick hangs from
  const st = STATES[id](t); if (!st) return null; const { form, x, y, o, s } = st; let top, w;
  if (form === 'kid') { top = kidTop(o.who, y, s, o); w = (KID[o.who].hw + 1.5) * s; }
  else if (form === 'ghost') { top = ghostTop(o.kind, y, s, o.lift ?? 0, o.sheetK ?? 1) - .3 * s; w = GHOST[o.kind].Wd * s * .65; }
  else if (form === 'dogghost') { top = y - GHOST.dog.H * s - (o.lift ?? 0) * s - 1.1 * s; w = 3.2 * s; }
  else if (form === 'dog') { top = y - 4.9 * s; w = 3.3 * s; } else { top = y - 7.8 * s; w = 2.7 * s; }
  return { id, x, top, bottom: y, w, s, form };
}
const clockOf = t => t < 36 ? '21:02' : t < 52 ? '21:08' : t < 62.5 ? '21:14' : t < 71.4 ? '21:15' : '21:16';
const darkK = t => seg(t, 65.7, 66.0) * (1 - seg(t, 72.5, 72.65)) * .93 + .5 * seg(t, 94.6, 97.0) * (1 - seg(t, 110.0, 110.4));
function tagText(id, t) {                                                            // what the small tag over a head says (the stained sheet misleads it, then the repair fixes it)
  if (id === 'mia' && t >= 62.8 && t < 69.9) return { text: 'Leo?', color: PAL.pumpkin, stroke: PAL.pumpkin };
  return { text: id };
}
function assertOnStage(t) {                                           // every character that should be on stage lies fully inside the stage rect
  const req = { mia: [10.6, 35], leo: [10.6, 35], sam: [10.6, 35], pebble: [6.5, 35], biscuit: [31.4, 34.6] };
  if (t >= 114.8 && t <= 155) for (const id of ['mia', 'leo', 'sam', 'mum', 'biscuit', 'pebble']) req[id] = [114.8, 155];
  for (const [id, [a, b]] of Object.entries(req)) { if (t < a || t > b) continue; const an = anchorOf(id, t); if (!an) continue; const [x0] = sp(an.x - an.w, an.top), [x1] = sp(an.x + an.w, an.top);
    if (x0 < 0 || x1 > PX0 - 50) throw new Error(`${id} is outside the stage at t=${t.toFixed(2)} (${x0.toFixed(0)}..${x1.toFixed(0)} of ${STAGE_W})`); }
}
function overlayStageOld(t) {
  assertOnStage(t); const dk = darkK(t);
  for (const char of ['mia', 'leo', 'sam', 'biscuit']) { const txt = beliefTag(char, t); if (!txt || dk > .2) continue; const an = anchorOf(char, t); if (!an) continue; assertIds('tag', char, an.id);
    const card = CARD_OF[char], [x, y] = sp(an.x, an.top - 40 * (an.s / 36)), pop = backOut(seg(t, card.t + .1, card.t + .5)), h = cardName(card, t), pencil = !h || h.st === 'pencil', known = !!h;
    OVER.push((c) => { UI.tag(x, y, txt, { a: Math.min(1, pop), s: .84, size: 38, rot: -.04 + hash(char.length) * .06, color: pencil ? '#8a8a92' : PAL.ink, stroke: known ? card.ink : '#a9a39a' });
      if (window.DEBUG_IDS) { c.save(); c.fillStyle = '#d00'; c.font = '700 20px sans-serif'; c.fillText(`${char} (card ${card.id})`, x - 50, y - 38); c.restore(); } }); }
  for (const s of SAYS) { const cur = speakingNow(s.who, t); if (!cur || cur.id !== s.id || cur.offscreen) continue; const an = anchorOf(s.who, t); if (!an) continue; assertIds('tick', s.who, an.id);
    const dir = (STATES[s.who](t).o.turn ?? 0) > .05 ? 1 : -1, [x, y] = sp(an.x + dir * an.w * .95, an.top + 1.6 * an.s), k = .5 + .5 * Math.sin(t * 16);
    OVER.push((c) => { c.save(); c.strokeStyle = INKC[s.who]; c.lineWidth = 4; c.lineCap = 'round'; for (let i = -1; i <= 1; i++) { const a = i * .5 - (dir < 0 ? Math.PI : 0), r0 = 12 + 3 * k, r1 = 26 + 6 * k; c.beginPath(); c.moveTo(x + Math.cos(a) * r0, y + Math.sin(a) * r0); c.lineTo(x + Math.cos(a) * r1, y + Math.sin(a) * r1); c.stroke(); } c.restore(); }); }
  const off = SAYS.find(s => s.offscreen && t >= s.t0 + .1 && t <= s.t0 + s.dur - .35);
  if (off) { const [x, y] = sp(150, 540); OVER.push((c) => { c.save(); c.strokeStyle = INKC[off.who]; c.lineWidth = 5; c.lineCap = 'round'; for (let i = 0; i < 3; i++) { c.globalAlpha = .9 - i * .22; c.beginPath(); c.arc(x - 18, y, 26 + i * 18 + ((t * 40) % 18), -.8, .8); c.stroke(); } c.fillStyle = INKC[off.who]; for (let i = 0; i < 5; i++) { const h = [8, 18, 28, 16, 10][i] * (.6 + .4 * Math.sin(t * 14 + i)); c.fillRect(x - 74 + i * 9, y - h / 2, 5, h); } c.restore(); }); }
  if (t > 4.0 && t < 4.7) OVER.push((c) => { const [x, y] = sp(1650, 430); c.save(); c.globalAlpha = 1 - seg(t, 4.4, 4.7); c.translate(x, y); c.rotate(-.1); c.font = '700 64px Caveat'; c.fillStyle = PAL.candle; c.strokeStyle = PAL.ink; c.lineWidth = 7; c.lineJoin = 'round'; c.textAlign = 'center'; const sc = 1 + .1 * Math.sin(t * 40); c.scale(sc, sc); c.strokeText('DING-DONG', 0, 0); c.fillText('DING-DONG', 0, 0); c.restore(); });
  if (t > 52.2 && t < 52.6) OVER.push((c) => { c.save(); c.globalAlpha = 1 - seg(t, 52.4, 52.6); c.font = '700 130px Caveat'; c.textAlign = 'center'; c.fillStyle = '#FFF4B0'; c.strokeStyle = PAL.ink; c.lineWidth = 9; c.lineJoin = 'round'; c.translate(STAGE_CX, 280); c.rotate(-.06); c.strokeText('CRACK!', 0, 0); c.fillText('CRACK!', 0, 0); c.restore(); });
  if (t > 40.8 && t < 85) OVER.push((c) => { const a = sstep(40.8, 41.3, t) * (1 - sstep(84.2, 84.8, t)); UI.tag(150, 120, 'faces visible: 0', { a, size: 34, fill: '#FFF0E0', stroke: PAL.ketchup, color: PAL.ketchup, rot: -.03, s: 1 + .03 * Math.sin(t * 6) }); });
  if (t > 71.8 && t < 81) { const mm = Math.round(lerp(16, 31, seg(t, 72.0, 80.4))), a = seg(t, 71.8, 72.3) * (1 - seg(t, 80.6, 81)); OVER.push(() => UI.tag(310, 215, `21:${String(mm).padStart(2, '0')}`, { a, size: 56, fill: '#FFF8EC', rot: -.03 })); }
}
function perceiveFrames(t) {
  const out = []; const wins = { mia: [[10.0, 13.0], [40.7, 43.0], [62.8, 64.6]], leo: [[10.3, 13.3], [40.7, 43.0]], sam: [[10.6, 13.6], [40.7, 43.0]], biscuit: [[33.3, 36.0], [50.4, 52.4]] };
  for (const [id, ws] of Object.entries(wins)) for (const [a, b] of ws) if (t >= a && t < b) { const an = anchorOf(id, t); if (an) out.push({ id, an, p: backOut(seg(t, a, a + .3)) * (1 - seg(t, b - .3, b)) }); }
  return out;
}
function percBoxes(t) {
  if (darkK(t) > .2) return;
  for (const f of perceiveFrames(t)) { const { an, p, id } = f; assertIds('frame', id, an.id); const x0 = an.x - an.w, x1 = an.x + an.w, y0 = an.top - 8, y1 = an.bottom + 10, col = COLS[id], ink = mixCol(col, '#2B2A33', .6), g = lerp(1.18, 1, clamp(p));
    dashedBox(x0, y0, x1, y1, 1.15, ink, g);
    for (const [cx, cy, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x1, y1, -1, -1], [x0, y1, 1, -1]]) { const ex = (cx - an.x) * (g - 1) / g; inkLine([[cx + ex + dx * 34, cy], [cx + ex, cy], [cx + ex, cy + dy * 34]], 1.7, col, 'ink', 0); } }
}

/* ───────── Pebble's notebook panel ───────── */
function doodle(c, who, x, y, r, t, ghostForm) {                                      // a little face (or a ghost doodle) for each row
  c.save(); c.translate(x, y); c.lineWidth = 3; c.strokeStyle = PAL.ink; c.lineJoin = 'round';
  if (ghostForm && who !== 'pebble') { c.fillStyle = '#FBF8F0'; c.beginPath(); c.moveTo(-r, r); c.lineTo(-r, -r * .1); c.quadraticCurveTo(-r, -r * 1.15, 0, -r * 1.15); c.quadraticCurveTo(r, -r * 1.15, r, -r * .1); c.lineTo(r, r); for (let i = 3; i >= 0; i--) c.lineTo(-r + (i + .5) * r * .5, r - (i % 2 ? .0 : .22) * r); c.closePath(); c.fill(); c.stroke();
    c.fillStyle = PAL.ink; c.beginPath(); c.ellipse(-r * .35, -r * .3, r * .12, r * .18, 0, 0, TAU); c.ellipse(r * .35, -r * .3, r * .12, r * .18, 0, 0, TAU); c.fill();
    if (who === 'leo') { c.fillStyle = PAL.ketchup; c.beginPath(); c.arc(r * .1, r * .25, r * .22, 0, TAU); c.fill(); }
    c.fillStyle = who === 'mia' ? '#FF7EB6' : who === 'sam' ? PAL.gold : who === 'leo' ? PAL.green : PAL.brown; c.beginPath(); c.ellipse(0, r * 1.05, r * .55, r * .18, 0, 0, TAU); c.fill(); c.restore(); return; }
  const face = (fill) => { c.fillStyle = fill; c.beginPath(); c.arc(0, 0, r, 0, TAU); c.fill(); c.stroke(); };
  const eyes = () => { c.fillStyle = PAL.ink; c.beginPath(); c.arc(-r * .35, -r * .05, r * .1, 0, TAU); c.arc(r * .35, -r * .05, r * .1, 0, TAU); c.fill(); c.fillStyle = 'rgba(255,126,182,.55)'; c.beginPath(); c.ellipse(-r * .55, r * .25, r * .2, r * .12, 0, 0, TAU); c.ellipse(r * .55, r * .25, r * .2, r * .12, 0, 0, TAU); c.fill(); c.strokeStyle = PAL.ink; c.beginPath(); c.arc(0, r * .2, r * .25, .2, Math.PI - .2); c.stroke(); };
  if (who === 'pebble') { c.fillStyle = '#FBF4E6'; c.beginPath(); c.ellipse(0, r * .1, r * 1.0, r * .95, 0, 0, TAU); c.fill(); c.stroke(); c.fillStyle = '#2b2d44'; c.beginPath(); c.ellipse(0, -r * .05, r * .7, r * .55, 0, 0, TAU); c.fill(); c.fillStyle = '#58c7e8'; c.beginPath(); c.arc(0, -r * .05, r * .38, 0, TAU); c.fill(); c.fillStyle = '#12101c'; c.beginPath(); c.arc(0, -r * .05, r * .17, 0, TAU); c.fill(); c.fillStyle = '#fff'; c.beginPath(); c.arc(-r * .12, -r * .17, r * .07, 0, TAU); c.fill(); c.fillStyle = '#3a2d59'; c.beginPath(); c.moveTo(-r * .7, -r * .62); c.lineTo(-r * .2, -r * 1.3); c.lineTo(r * .1, -r * .62); c.closePath(); c.fill(); c.stroke(); }
  else if (who === 'biscuit') { c.fillStyle = PAL.brown; c.beginPath(); c.ellipse(-r * .85, r * .15, r * .35, r * .75, .3, 0, TAU); c.ellipse(r * .85, r * .15, r * .35, r * .75, -.3, 0, TAU); c.fill(); c.stroke(); face(PAL.brown); c.fillStyle = '#d9a86f'; c.beginPath(); c.ellipse(0, r * .3, r * .5, r * .38, 0, 0, TAU); c.fill(); c.fillStyle = PAL.ink; c.beginPath(); c.arc(-r * .35, -r * .15, r * .1, 0, TAU); c.arc(r * .35, -r * .15, r * .1, 0, TAU); c.arc(0, r * .2, r * .12, 0, TAU); c.fill(); c.fillStyle = PAL.pumpkin; c.beginPath(); c.moveTo(-r * .6, r * .75); c.lineTo(r * .6, r * .75); c.lineTo(0, r * 1.15); c.closePath(); c.fill(); }
  else { if (who === 'mia') { c.fillStyle = '#8a5530'; for (const sd of [-1, 1]) { c.beginPath(); c.ellipse(sd * r * .95, r * .35, r * .3, r * .6, sd * .3, 0, TAU); c.fill(); c.stroke(); } }
    face(PAL.skin);
    if (who === 'mia') { c.fillStyle = '#8a5530'; c.beginPath(); c.arc(0, -r * .1, r * 1.02, Math.PI, 0); c.lineTo(r * .7, -r * .35); c.lineTo(-r * .7, -r * .35); c.closePath(); c.fill(); c.stroke(); c.fillStyle = PAL.pumpkin; c.beginPath(); c.arc(-r * .98, r * .25, r * .14, 0, TAU); c.arc(r * .98, r * .25, r * .14, 0, TAU); c.fill(); }
    if (who === 'leo') { c.fillStyle = '#6a4a35'; c.beginPath(); c.arc(0, -r * .1, r * 1.02, Math.PI, 0); c.closePath(); c.fill(); c.stroke(); for (const k of [-1, 0, 1]) { c.beginPath(); c.moveTo(k * r * .5 - r * .3, -r * .45); c.quadraticCurveTo(k * r * .55, -r * .1, k * r * .5 + r * .35, -r * .05); c.lineTo(k * r * .5 + r * .3, -r * .5); c.closePath(); c.fill(); c.stroke(); } }
    if (who === 'sam') { c.fillStyle = '#a9825a'; for (let k = 0; k < 7; k++) { const a = Math.PI * (1.08 + k / 6 * .84); c.beginPath(); c.arc(Math.cos(a) * r * 1.0, Math.sin(a) * r * 1.0, r * .3, 0, TAU); c.fill(); c.stroke(); } face(PAL.skin); }
    if (who === 'mum') { c.fillStyle = '#8a5a3a'; c.beginPath(); c.arc(0, -r * .1, r * 1.04, Math.PI, 0); c.lineTo(r * .7, -r * .3); c.lineTo(-r * .7, -r * .3); c.closePath(); c.fill(); c.stroke(); c.beginPath(); c.arc(0, -r * 1.12, r * .42, 0, TAU); c.fill(); c.stroke();
      c.fillStyle = '#FF9CCB'; for (const sd of [-1, 1]) for (let k = 0; k < 2; k++) { c.beginPath(); c.roundRect ? c.roundRect(sd * r * .95 - r * .22, -r * .75 + k * r * .4, r * .44, r * .26, r * .1) : c.rect(sd * r * .95 - r * .22, -r * .75 + k * r * .4, r * .44, r * .26); c.fill(); c.stroke(); } }
    eyes(); }
  c.restore();
}
function rowSeal(c, x, y, k) { c.save(); c.translate(x, y); c.rotate(-.08); c.globalAlpha = k; c.strokeStyle = PAL.ketchup; c.fillStyle = PAL.ketchup; c.lineWidth = 3; roundRect(c, -48, -17, 96, 34, 9); c.stroke(); c.font = '700 24px Caveat'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('✓ named', 0, 2); c.restore(); }
function rowStamp(c, x, y, text, t0, t) {                                           // lands over empty row space, then shrinks into a small seal
  const d = t - t0; if (d < 0) return; if (d >= .85) { rowSeal(c, x + 8, y, 1); return; }
  if (d < .55) { UI.stamp(x, y, text, t0, t, { size: 44, rot: -.08, hold: true }); return; }
  const k = ease((d - .55) / .3); c.save(); c.translate(x + 8 * k, y); c.scale(lerp(1, .42, k), lerp(1, .42, k)); c.globalAlpha = 1 - k; UI.stamp(0, 0, text, t0, t, { size: 44, rot: -.08, hold: true }); c.restore(); rowSeal(c, x + 8, y, k);
}
function drawPanelOld(c, t) {
  const x0 = PX0, y0 = 20, w = PW, h = 1040, ink = '#3a2f2a'; ctx = c;
  c.save(); c.shadowColor = 'rgba(30,20,10,.35)'; c.shadowBlur = 24; c.shadowOffsetX = -6; c.shadowOffsetY = 8; c.fillStyle = '#FFF6E0'; roundRect(c, x0, y0, w, h, 26); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = PAL.ink; c.lineWidth = 4; c.stroke();
  c.strokeStyle = 'rgba(120,150,200,.22)'; c.lineWidth = 2; for (let y = y0 + 90; y < y0 + h - 20; y += 38) { c.beginPath(); c.moveTo(x0 + 50, y); c.lineTo(x0 + w - 20, y); c.stroke(); }
  for (let y = y0 + 40; y < y0 + h - 20; y += 54) { c.strokeStyle = PAL.ink; c.lineWidth = 5; c.lineCap = 'round'; c.beginPath(); c.moveTo(x0 - 8, y); c.lineTo(x0 + 34, y); c.stroke(); c.fillStyle = '#d8d3c8'; c.beginPath(); c.arc(x0 + 30, y, 6, 0, TAU); c.fill(); }
  c.fillStyle = PAL.ink; c.font = '700 44px Caveat'; c.textBaseline = 'alphabetic'; c.textAlign = 'left'; c.fillText("Pebble's memory", x0 + 64, y0 + 54);
  const sec = (label, y, tech) => { c.fillStyle = PAL.violet; c.font = '700 36px Caveat'; c.textAlign = 'left'; c.fillText(label, x0 + 64, y); if (tech) { const lw = c.measureText(label).width; c.fillStyle = '#9a8e82'; c.font = '600 26px Caveat'; c.fillText(tech, x0 + 82 + lw, y); } c.strokeStyle = PAL.violet; c.lineWidth = 3; c.beginPath(); c.moveTo(x0 + 64, y + 10); c.lineTo(x0 + w - 34, y + 10); c.stroke(); };
  const swipe = (x, y, ww, hh, t0) => { const k = ease(seg(t, t0, t0 + .45)), a = 1 - seg(t, t0 + 1.0, t0 + 1.7); if (k > 0 && a > 0) { c.save(); c.fillStyle = `rgba(255,226,90,${.55 * a})`; roundRect(c, x, y, ww * k, hh, 10); c.fill(); c.restore(); } };
  sec("Who's who", y0 + 104);
  const cardY = i => y0 + 124 + i * 86, cw = w - 90;
  CARDS.forEach((cd, i) => {
    const k = backOut(seg(t, cd.t, cd.t + .4)); if (k <= .01) return; const ry = cardY(i), h_ = cardName(cd, t), known = !!h_, col = cd.col;
    swipe(x0 + 52, ry - 2, cw + 6, 80, cd.t); cd.name.forEach(([tt]) => swipe(x0 + 52, ry - 2, cw + 6, 80, tt));
    c.save(); c.translate(0, (1 - Math.min(1, k)) * 12); c.globalAlpha = Math.min(1, k);
    c.fillStyle = known ? mixCol(col, '#FFF6E0', .78) : '#ECE7DC'; roundRect(c, x0 + 56, ry, cw, 76, 16); c.fill(); c.strokeStyle = known ? cd.ink : '#bdb6a8'; c.lineWidth = 2.6; c.stroke();
    doodle(c, cd.char, x0 + 92, ry + 38, 26, t, false);
    let nm = '?', pencil = false, nw = 0; c.textAlign = 'left';
    if (h_) { nm = h_.tx; pencil = h_.st === 'pencil'; } c.font = '700 40px Caveat'; nw = c.measureText(nm).width;
    c.fillStyle = h_ ? (pencil ? '#8f8f98' : ink) : '#b3ab9d'; c.fillText(nm, x0 + 134, ry + 38);
    if (pencil) { const bx = x0 + 134, by = ry + 46; c.strokeStyle = '#8f8f98'; c.lineWidth = 2; roundRect(c, bx, by, 84, 8, 4); c.stroke(); c.fillStyle = '#a9a9b2'; roundRect(c, bx, by, 84 * h_.cf * seg(t, h_.t, h_.t + .5), 8, 4); c.fill(); }
    if (h_ && !pencil && !cd.known) rowSeal(c, x0 + 134 + nw + 62, ry + 24, ease(seg(t, h_.t + .1, h_.t + .45)));
    if (cd.known) { c.fillStyle = '#2a9d6f'; c.font = '700 26px Caveat'; c.fillText('✓ known', x0 + 134 + nw + 14, ry + 38); }
    const ev = h_ ? h_.ev : cd.ev0; c.fillStyle = '#7a6b60'; c.font = '500 22px Caveat'; c.fillText(ev, x0 + 134, ry + 69, 250);
    cd.chips.forEach((ch, j) => { const cx_ = x0 + cw - 20 + 0 - j * 56, sure = t >= ch.sure, kc = t >= ch.t + .01; if (!kc) return; c.save(); c.fillStyle = known ? col : '#cfc8bc'; roundRect(c, cx_ - 46, ry + 8, 52, 30, 9); c.fill(); c.strokeStyle = PAL.ink; c.lineWidth = 2.6; if (!sure) c.setLineDash([5, 4]); c.stroke(); c.setLineDash([]); c.fillStyle = PAL.ink; c.font = '700 24px Caveat'; c.textAlign = 'center'; c.fillText(ch.id, cx_ - 20, ry + 31); c.restore(); });
    const cues = cd.known ? cd.cues : ['face', 'body', 'voice', cd.cue], lits = cd.known ? cd.lit : [cd.t + .2, cd.t + .45, cd.voiceAt, cd.t + .8];
    cues.forEach((kk, j) => { const l = lits[j], on = t >= l, pu = on ? 1 + .25 * Math.exp(-(t - l) * 6) : 1; UI.icon(kk, x0 + cw - 138 + j * 40, ry + 58, .72 * pu, 1, on ? cd.ink : '#cdc5b8', on ? PAL.ink : '#b9b1a4'); });
    c.restore();
  });
  const sy = y0 + 124 + 6 * 86 + 34; sec('Screenplay', sy, '21:02');
  const top = sy + 28, cap = 6 * 47 + 8, lines = [];
  for (const [tt, kind, who, text] of SCRIPT1) { if (kind === 'head' || t < tt) continue; const pre = kind === 'say' ? who + ': ' : '', size = 33; ctx = c; const wrapped = UI.wrap(pre + text, w - 130, `600 ${size}px Caveat`); lines.push({ tt, kind, who, text, pre, size, wrapped, h: wrapped.length * (size * 1.12) + 8 }); }
  let total = 0; lines.forEach((l, i) => { l.y = total; total += l.h * (i === lines.length - 1 ? ease(seg(t, l.tt, l.tt + .4)) : 1); });
  const off = Math.max(0, total - cap);
  c.save(); c.beginPath(); c.rect(x0 + 40, top - 4, w - 56, cap + 8); c.clip();
  for (const l of lines) {
    const yb = top + l.y - off; c.globalAlpha = clamp((yb + l.h - top + 4) / (l.h * 1.1)); const sayDur = l.kind === 'say' ? (SAYS.find(s => Math.abs(s.t0 - l.tt) < .05)?.dur ?? 2.4) * .8 : 1.1, p = clamp((t - l.tt) / sayDur), tot = l.pre.length + l.text.length; let done = 0;
    const owner = chipOwner(l.who), chipCol = l.who === 'Pebble' ? COLS.pebble : (owner && cardName(owner, t) ? owner.col : '#cfc8bc');
    l.wrapped.forEach((ln, i) => { const a = clamp((p * tot - done) / ln.length), yy = yb + (i + 1) * l.size * 1.12; c.font = `600 ${l.size}px Caveat`;
      if (l.kind === 'say' && i === 0) { const wl = c.measureText(l.pre).width; c.fillStyle = chipCol; roundRect(c, x0 + 62, yy - l.size * .82, wl - 2, l.size * 1.08, 8); c.fill(); c.strokeStyle = PAL.ink; c.lineWidth = 2; c.stroke(); c.fillStyle = PAL.ink; c.textAlign = 'left'; c.fillText(l.pre.slice(0, -2), x0 + 68, yy); UI.writeOn(ln.slice(l.pre.length), x0 + 66 + wl, yy, l.size, clamp((a * ln.length - l.pre.length) / Math.max(1, ln.length - l.pre.length)), { color: ink, nopen: a >= 1 }); }
      else UI.writeOn(ln, x0 + 66, yy, l.size, a, { color: l.kind === 'act' ? '#7a6b60' : ink, nopen: a >= 1 });
      done += ln.length + 1; });
    c.globalAlpha = 1;
  }
  c.restore(); c.restore();
}
function polaroidArt(c, kind, cap) {                                                // a 300x350 polaroid at the origin
  c.save(); c.shadowColor = 'rgba(40,25,10,.35)'; c.shadowBlur = 14; c.shadowOffsetY = 7; c.fillStyle = '#FFFBF0'; roundRect(c, -150, -180, 300, 350, 8); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = PAL.ink; c.lineWidth = 3; c.stroke();
  c.save(); c.beginPath(); c.rect(-132, -162, 264, 240); c.clip();
  c.fillStyle = '#3a3552'; c.fillRect(-132, -162, 264, 240); c.fillStyle = '#FFF5F8'; c.fillRect(-60, -40, 40, 100); c.fillRect(20, -40, 40, 100); c.fillStyle = '#FF7EB6'; c.strokeStyle = PAL.ink; c.lineWidth = 3; for (const sx of [-95, 55]) { c.beginPath(); c.ellipse(sx + 40, 66, 62, 30, 0, 0, TAU); c.fill(); c.stroke(); } c.strokeStyle = '#7fe3ff'; c.lineWidth = 4; c.beginPath(); c.arc(0, 20, 108, 0, TAU); c.stroke();
  c.restore(); c.strokeStyle = PAL.ink; c.lineWidth = 3; c.strokeRect(-132, -162, 264, 240); c.fillStyle = 'rgba(205,170,110,.8)'; c.fillRect(-30, -198, 60, 26);
  c.fillStyle = PAL.ink; c.font = '600 34px Caveat'; c.textAlign = 'center'; c.fillText(cap, 0, 140); c.restore();
}

/* ───────── subtitles: dialogue only, under the stage, opaque while the line is active ───────── */
function subtitleRect(s, c) {
  const ln = STORY.lines[s.id], who = s.label ?? NAMES[s.who]; c.font = '600 44px Caveat'; const one = c.measureText(who + ': ' + ln.en).width, maxW = 1090;
  const size = one > maxW ? 36 : 44; ctx = c; const enLines = UI.wrap(who + ': ' + ln.en, maxW, `600 ${size}px Caveat`); c.font = "400 26px 'LXGW WenKai'";
  const zhLines = wrapZH(who + '：' + ln.zh, maxW, "400 26px 'LXGW WenKai'"); let wmax = 0; c.font = `600 ${size}px Caveat`; for (const l of enLines) wmax = Math.max(wmax, c.measureText(l).width); c.font = "400 26px 'LXGW WenKai'"; for (const l of zhLines) wmax = Math.max(wmax, c.measureText(l).width);
  const enH = size * 1.0, zhH = 31, h = 18 + enLines.length * enH + zhLines.length * zhH + 12, w = Math.min(1180, wmax + 80);
  return { ln, who, size, enLines, zhLines, enH, zhH, w, h, x0: STAGE_CX - w / 2, y0: 1062 - h };
}
function assertSubtitleClear(s, t, R) {
  for (const id of ['mia', 'leo', 'sam', 'mum', 'biscuit', 'pebble']) { const an = anchorOf(id, t); if (!an) continue; const [x0, y0] = sp(an.x - 2.2 * an.s, an.top), [x1, y1] = sp(an.x + 2.2 * an.s, an.top + (id === 'biscuit' ? 2.4 : 5.0) * an.s);
    if (x0 < R.x0 + R.w && x1 > R.x0 && y0 < R.y0 + R.h && y1 > R.y0) throw new Error(`subtitle covers ${id}'s face at t=${t.toFixed(2)}`); }
}
function subtitle(c, t) {
  const s = SAYS.find(s => t >= s.t0 && t <= s.t0 + s.dur + .12); if (!s) return; const R = subtitleRect(s, c); assertSubtitleClear(s, t, R);
  const a = sstep(s.t0, s.t0 + .2, t) * (1 - sstep(s.t0 + s.dur, s.t0 + s.dur + .12, t));
  c.save(); c.globalAlpha = a; c.fillStyle = 'rgb(250,243,228)'; c.shadowColor = 'rgba(40,25,10,.18)'; c.shadowBlur = 12; roundRect(c, R.x0, R.y0, R.w, R.h, 26); c.fill(); c.shadowColor = 'transparent'; c.strokeStyle = 'rgba(43,42,51,.55)'; c.lineWidth = 2; c.stroke();
  c.textAlign = 'center'; c.textBaseline = 'alphabetic'; let y = R.y0 + 12;
  R.enLines.forEach((l, i) => { y += R.enH; c.font = `600 ${R.size}px Caveat`; const first = i === 0 && l.startsWith(R.who + ': '); if (first) { const wl = c.measureText(R.who + ': ').width, wa = c.measureText(l).width, sx = STAGE_CX - wa / 2; c.textAlign = 'left'; c.fillStyle = INKC[s.who]; c.fillText(R.who + ': ', sx, y - 6); c.fillStyle = PAL.ink; c.fillText(l.slice(R.who.length + 2), sx + wl, y - 6); c.textAlign = 'center'; } else { c.fillStyle = PAL.ink; c.fillText(l, STAGE_CX, y - 6); } });
  R.zhLines.forEach((l, i) => { y += R.zhH; c.font = "400 26px 'LXGW WenKai'"; const first = i === 0 && l.startsWith(R.who + '：'); if (first) { const wl = c.measureText(R.who + '：').width, wa = c.measureText(l).width, sx = STAGE_CX - wa / 2; c.textAlign = 'left'; c.fillStyle = INKC[s.who]; c.fillText(R.who + '：', sx, y - 6); c.fillStyle = '#5b5560'; c.fillText(l.slice(R.who.length + 1), sx + wl, y - 6); c.textAlign = 'center'; } else { c.fillStyle = '#5b5560'; c.fillText(l, STAGE_CX, y - 6); } });
  c.restore();
}

/* ───────── the world ───────── */
const levelAt = t => t < 73.1 ? 1 : t < 75.7 ? lerp(1, .66, seg(t, 73.1, 73.5)) : t < 78.6 ? lerp(.66, .33, seg(t, 75.8, 76.2)) : lerp(.33, 0, seg(t, 78.6, 79.0));
const sheetFlight = [['mia', 39.7], ['leo', 39.76], ['sam', 39.82]];
function worldNightOld(t) {
  const dk = darkK(t); setSky(t, false); setSpider(t); const mm = t < 71 ? 2 : Math.round(lerp(16, 31, seg(t, 72, 80.4))) ; setClockHands(t >= 52 && t < 71 ? 14 : mm); setCauldron(levelAt(t), t, false); setDoor(t, T_DOOR);
  paint(rectPts(-40, -40, W + 80, H + 80, 0), { wash: PAL.paper, washOp: 58, ink: null });
  if (t > 52.4 && t < 52.8) { const k = 1 - seg(t, 52.4, 52.8); paint([[1000, 120], [1050, 250], [1015, 255], [1075, 400], [980, 290], [1015, 285], [960, 140]], { wash: '#FFFFFF', washOp: 255 * k, ink: null }); }
  const items = [];
  for (const id of ['mia', 'leo', 'sam', 'biscuit', 'pebble']) { const st = STATES[id](t); if (st) items.push([st.y, () => drawState(id, t, dk)]); }
  items.sort((a, b) => a[0] - b[0]); wrappers(t);
  for (const [, fn] of items) fn();
  if (t > 39.6 && t < 40.8) for (const [who, t0] of sheetFlight) { const u = seg(t, t0, 40.7); if (u > 0 && u < 1) { const [hx, hy] = HOME[who] ?? HOME.mia; drawSheet(who === 'mia' ? HOME.mia[0] : HOME[who][0], HOME[who][1] + 20, u, who); } }
  if (t > 40.65 && t < 41.3) for (const who of ['mia', 'leo', 'sam']) puff(HOME[who][0], HOME[who][1] + 20, seg(t, 40.7, 41.3));
  if (t > 49.4 && t < 50.9) { const [bx, by] = [LAYOUT.basket[0], LAYOUT.basket[1] - 10], u = seg(t, 49.5, 50.3); if (u < 1) drawSheet(bx, by + 10, u, 'dog'); puff(bx, by + 20, seg(t, 50.3, 50.9)); }
  if (t > 70.8 && t < 71.5) puff(LAYOUT.basket[0], LAYOUT.basket[1], seg(t, 70.85, 71.5));
  if (t > 22.1 && t < 22.95) { const u = seg(t, 22.15, 22.9), sx = 1100 - 40, sy = 850 - 250, lx = 710 + 60, ly = 875 - 440; candyArt(lerp(sx, lx, u), lerp(sy, ly, u) - Math.sin(u * Math.PI) * 140, 18, '#FF8FC4'); }
  HEIST.forEach((h, i) => { const d = t - h.grab; if (d < 0 || d > .6) return; const n = i === 0 ? 3 : 2; for (let k = 0; k < n; k++) { const u = clamp((d - k * .08) / .45); if (u <= 0 || u >= 1) continue; candyArt(lerp(580 + k * 40, 700, u), lerp(640, 660, u) - Math.sin(u * Math.PI) * 120, 16 * (1 - .3 * u), [PAL.pink, PAL.candle, PAL.mint][(i + k) % 3]); } });
  setBats(t); setLights(t, 1 - dk, false); zzzBubbles(t); flushGlowQ(); sparkles(t, 8);
}
function wrappers(t) {
  const spots = [[860, 835, 0], [980, 862, 1], [1090, 828, 2], [800, 878, 1], [1150, 880, 0], [940, 818, 2]], times = [73.6, 76.3, 79.2, 74.6, 77.4, 79.8];
  spots.forEach(([x, y, k], i) => { const d = t - times[i]; if (d < 0) return; const f = Math.min(1, d / .45), yy = y - (1 - easeOut(f)) * 150 - Math.sin(f * Math.PI) * 30; paint([[x - 14, yy], [x - 5, yy - 13], [x + 9, yy - 9], [x + 16, yy + 4], [x + 3, yy + 11], [x - 11, yy + 9]], { wash: [PAL.pink, PAL.candle, PAL.mint][k], ink: PAL.ink, sw: .5 }); });
}
function worldMorning(t) {
  setSky(t, true); setSpider(t); setDoor(t, -99); setClockHands(40); setMorningExtras(t);
  paint(rectPts(-40, -40, W + 80, H + 80, 0), { wash: PAL.paper, washOp: 40, ink: null });
  const items = []; for (const id of ['mia', 'leo', 'sam', 'biscuit', 'pebble', 'mum']) { const st = STATES[id](t); if (st) items.push([st.y, () => drawState(id, t)]); }
  items.sort((a, b) => a[0] - b[0]); for (const [, fn] of items) fn();
  if (t > 110.35) { const m = STATES.mia(t), d = seg(t, 110.4, 111.9), f = ease(d); if (m) { const sx = m.x - 2.25 * m.s * .88, sy = m.y - (KID.mia.legH + 1.5) * m.s, yy = lerp(sy, m.y - 6, f * f), xx = sx - 70 * f; paint([[xx - 28, yy], [xx - 9, yy - 27 + 10 * Math.sin(d * 9)], [xx + 20, yy - 18], [xx + 32, yy + 9], [xx + 5, yy + 24], [xx - 22, yy + 18]], { wash: PAL.pink, ink: PAL.ink, sw: .8 }); paint([[xx - 8, yy - 6], [xx + 8, yy - 10], [xx + 10, yy + 6], [xx - 6, yy + 8]], { wash: PAL.candle, ink: null }); } }
  dust(t, 36); flushGlowQ();
}
function blackoutOverlayBase(t) {
  const d = darkK(t); if (d <= .01) return; const eyes = [];
  for (const id of ['mia', 'leo', 'sam']) { const st = STATES[id](t); if (st && st.form === 'ghost') { const { x, y, s, o } = st, g = GHOST[o.kind], sk = o.sheetK ?? 1, domeR = g.Wd * s * .5 * (sk > 1 ? 1 + (sk - 1) * .5 : 1), topY = y - (o.lift ?? 0) * s - g.H * s * sk + domeR * .9; for (const side of [-1, 1]) eyes.push(sp(x + side * domeR * .38, topY + domeR * .45)); } }
  { const st = STATES.biscuit(t); if (st.form === 'dogghost') eyes.push(sp(st.x + 1.0 * st.s + .55 * st.s * .8, st.y - GHOST.dog.H * st.s + 1.25 * st.s * .8)); }
  const pb = STATES.pebble(t); const [lx, ly] = sp(pb.x, pb.y - 3.15 * pb.s), [ax, ay] = sp(pb.x + 1.65 * pb.s, pb.y - 6.1 * pb.s);
  OVER.push((c) => { c.save(); c.globalCompositeOperation = 'multiply'; c.fillStyle = `rgba(${70 - 50 * d},${82 - 50 * d},${150 - 70 * d},${.5 + .45 * d})`; c.fillRect(0, 0, W, H); c.restore(); c.save(); c.fillStyle = `rgba(4,6,24,${.78 * d})`; c.fillRect(0, 0, W, H); c.restore();
    c.save(); c.globalCompositeOperation = 'lighter'; for (const [x, y] of eyes) { radial(c, x, y, 70, '220,240,255', .55 * d); c.fillStyle = `rgba(255,255,255,${d})`; c.beginPath(); c.ellipse(x, y, 12, 17, 0, 0, TAU); c.fill(); }
    radial(c, lx, ly, 190 * (1 + .06 * Math.sin(t * 6)), '110,230,255', .75 * d); radial(c, lx, ly, 60, '230,255,255', .9 * d); radial(c, ax, ay, 90, '255,225,110', .7 * d * (.6 + .4 * Math.sin(t * 7))); c.restore(); });
}
function lensOverlay(t) {                                                           // clip 3: the lens zooms to the stained ghost's feet while the voice is matched
  const pi = seg(t, 67.6, 68.2) * (1 - seg(t, 69.9, 70.5)); if (pi <= .01) return; const m = STATES.mia(t), [lx, ly] = sp(m.x, m.y - 20), r = lerp(1100, 330, easeOut(pi)), pw = seg(t, 68.4, 69.0) * (1 - seg(t, 69.6, 70.1)), match = seg(t, 68.6, 69.2);
  OVER.push((c) => { c.save(); c.globalAlpha = .7 * pi; c.fillStyle = '#10131c'; c.beginPath(); c.rect(0, 0, W, H); c.arc(lx, ly, r, 0, TAU, true); c.fill('evenodd'); c.restore();
    c.save(); c.globalAlpha = pi; c.strokeStyle = '#7fe3ff'; c.lineWidth = 8; c.beginPath(); c.arc(lx, ly, r, 0, TAU); c.stroke(); c.lineWidth = 3; for (let i = 0; i < 12; i++) { const a = i / 12 * TAU + t * .3; c.beginPath(); c.moveTo(lx + Math.cos(a) * (r + 6), ly + Math.sin(a) * (r + 6)); c.lineTo(lx + Math.cos(a) * (r + 24), ly + Math.sin(a) * (r + 24)); c.stroke(); } c.restore();
    if (pw > .01) { c.save(); c.globalAlpha = pw; const px = 960, py = 300; c.fillStyle = 'rgba(255,251,240,.96)'; roundRect(c, px - 200, py - 120, 400, 250, 20); c.fill(); c.strokeStyle = PAL.ink; c.lineWidth = 3; c.stroke(); c.font = '600 32px Caveat'; c.fillStyle = PAL.ink; c.textAlign = 'center'; c.fillText('voice heard', px, py - 82); c.fillText("Mia's voiceprint", px, py + 18);
      for (let row = 0; row < 2; row++) { c.strokeStyle = row ? PAL.pink : PAL.violet; c.lineWidth = 5; c.lineCap = 'round'; c.beginPath(); for (let i = 0; i <= 60; i++) { const xx = px - 150 + i * 5, base = Math.sin(i * .5 + t * 6) * 18 * Math.sin(i / 60 * Math.PI), alt = Math.sin(i * .5 + t * 6 + (1 - match) * 2.4 * (row ? 0 : 1)) * (18 - (1 - match) * 6) * Math.sin(i / 60 * Math.PI), yy = py + (row ? 62 : -36) + (row ? base : alt); if (i === 0) c.moveTo(xx, yy); else c.lineTo(xx, yy); } c.stroke(); }
      if (match > .9) { c.fillStyle = '#2a9d6f'; c.font = '700 40px Caveat'; c.fillText('match ✓', px, py + 112); } c.restore(); } });
}
function drawScene(t, pass) {
  if (pass === 'cover') { FCAM = { cx: W / 2, cy: H / 2, zoom: 1, rot: 0, sx: W / 2 }; image(paperG, 0, 0); camBegin(FCAM); drawCover(t); camEnd(); return; }
  if (pass === 'end') { FCAM = { cx: W / 2, cy: H / 2, zoom: 1, rot: 0, sx: W / 2 }; image(paperG, 0, 0); drawEnd(t); return; }
  const morning = pass === 'morning'; FCAM = camAt(t); if (!morning) lightOverlay(t, 1);
  overlayStage(t);
  camBegin(); image(BG[morning ? 'morning' : 'night'], 0, 0); if (morning) worldMorning(t); else worldNight(t); percBoxes(t); camEnd();
  if (morning) OVER.push((c) => radial(c, 600, 140, 800, '255,236,170', .12));
  if (!morning) { blackoutOverlay(t); lensOverlay(t); }
  const f = flashAt(t); if (f > 0) OVER.push((c) => { c.save(); c.fillStyle = `rgba(255,255,255,${f})`; c.fillRect(0, 0, W, H); c.restore(); });
  OVER.push((c) => { c.save(); c.fillStyle = PAL.paper; c.fillRect(PX0 - 40, 0, W - PX0 + 40, H); c.restore(); });
  OVER.push((c) => drawPanel(c, t)); OVER.push((c) => subtitle(c, t));
}
function flashAt(t) { let f = 0; for (const [t0, a, d] of [[65.6, .95, .4], [66.15, .7, .3], [72.6, .85, .5]]) { const dd = t - t0; if (dd >= 0 && dd < d) f = Math.max(f, a * (1 - dd / d)); } return f; }
function drawEnd(t) {
  const u = seg(t, T_END + .1, T_END + .8), p2 = seg(t, T_END + .2, T_END + .6), k = backOut(seg(t, T_END + .4, T_END + 1.0));
  const wink = t - T_END > 1.0 && frac((t - T_END) * .8) < .35;
  pebble(1450, 720, 52 * Math.max(.01, k), { wink, look: [-.8, .1], glow: .55 + .35 * Math.sin(t * 5), rot: .05 * Math.sin(t * 3), armR: .9 + .4 * Math.sin(t * 6), roll: 0 });
  sparkles(t, 14); flushGlowQ();
  OVER.push((c) => {
    c.save(); c.globalAlpha = u; const lw = 980, lh = lw * (LOCKUP.height / LOCKUP.width); c.drawImage(LOCKUP, 330, 230 + (1 - u) * 30, lw, lh); c.restore();
    c.save(); c.globalAlpha = p2; c.fillStyle = '#FFFBF0'; c.strokeStyle = PAL.ink; c.lineWidth = 3; roundRect(c, 330, 640, 640, 96, 22); c.fill(); c.stroke(); c.fillStyle = PAL.ink; c.font = '600 60px Caveat'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('$ ' + STORY.ui.pip, 650, 690);
    c.font = '500 44px Caveat'; c.fillStyle = '#6a5f55'; c.fillText(STORY.ui.gh, 650, 790); c.font = '500 28px Caveat'; c.fillStyle = '#9a8f84'; c.fillText('Voices & SFX: ElevenLabs', 650, 852); c.restore(); });
}
/* ───────── compositing ───────── */
let layerA = null, layerB = null, LOCKUP = null;
async function layer(t, pass, dst) {
  await glPass(t, pass);
  const c = dst.getContext('2d'); c.clearRect(0, 0, W, H); c.drawImage(drawingContext.canvas, 0, 0, W, H);
  const prev = ctx; ctx = c; paintOver(t); ctx = prev;
}
async function composeFrame(t) {
  layerA ??= Object.assign(document.createElement('canvas'), { width: W, height: H }); layerB ??= Object.assign(document.createElement('canvas'), { width: W, height: H });
  if (t < T_FLIP[0]) { await layer(t, 'cover', layerA); ctx.drawImage(layerA, 0, 0); return; }
  if (t < T_FLIP[1]) { await layer(Math.max(t, 3.0), 'world', layerA); await layer(t, 'cover', layerB); ctx.drawImage(layerA, 0, 0); pageFlip(layerB, clamp((t - T_FLIP[0]) / (T_FLIP[1] - T_FLIP[0]))); return; }
  if (t < T_FLIP2[0]) { await layer(t, 'world', layerA); ctx.drawImage(layerA, 0, 0); return; }
  if (t < T_FLIP2[1]) { await layer(Math.max(t, T_C5 + .3), 'morning', layerA); await layer(Math.min(t, T_FLIP2[0]), 'world', layerB); ctx.drawImage(layerA, 0, 0); pageFlip(layerB, clamp((t - T_FLIP2[0]) / (T_FLIP2[1] - T_FLIP2[0]))); return; }
  if (t < T_END - .6) { await layer(t, 'morning', layerA); ctx.drawImage(layerA, 0, 0); return; }
  if (t < T_END) { await layer(t, 'morning', layerA); ctx.drawImage(layerA, 0, 0); ctx.save(); ctx.fillStyle = PAL.paper; ctx.globalAlpha = seg(t, T_END - .6, T_END); ctx.fillRect(0, 0, W, H); ctx.restore(); return; }
  await layer(t, 'end', layerA); ctx.drawImage(layerA, 0, 0);
}

/* ───────── carried over: cover, lighting, snore bubbles, page turn, sheets ───────── */
/* ───────── pieces carried over: cover, lighting, snore bubbles, page turn ───────── */
/* ───────── cover (shot 0) ───────── */
function drawCover(t) {
  const rise = kf(t, [[0, 360], [.35, 360], [.95, 226], [1.15, 238]], easeOut), look = [Math.sin(t * 2.3) * .9, -.3 + Math.sin(t * 1.7) * .3];
  const flash = Math.max(0, Math.sin((t - 1.6) * 9)) * seg(t, 1.6, 2.0) * (1 - seg(t, 2.0, 2.2));
  pebble(960 - 330, rise + Math.sin(t * 6) * 1, 30, { look, pupil: .5 + .2 * Math.sin(t * 3), lid: blinkAt(.2) ? .9 : 0, glow: .4 + flash, antSway: Math.sin(t * 5) * .3, noShadow: true, armR: kf(t, [[1.3, .2], [1.6, 1.0], [1.9, .5], [2.2, 1.0]], ease) });
  paint(rrPts(405, 150, 1110, 900, 40, 3), { wash: '#3F3470', fill: '#6a5aa8', fillOp: 70, tex: .6, ink: PAL.ink, sw: 1.3 });
  paint(rrPts(440, 185, 1040, 830, 28, 2), { wash: '#FFF3DC', fill: '#f3e1bd', fillOp: 70, tex: .5, ink: PAL.ink, sw: .9 });
  for (let i = 0; i < 14; i++) {
    const u = i / 14, x = lerp(480, 1440, u), s = .9 + .15 * Math.sin(t * 2 + i);
    if (i % 3 === 0) { paint(ellPts(x, 222, 20 * s, 17 * s, 12), { wash: PAL.pumpkin, ink: PAL.ink, sw: .6 }); paint(rectPts(x - 2, 198, 4, 8, 0), { wash: '#4c7a3a', ink: null }); }
    else if (i % 3 === 1) bat(x, 224, .6, t * 2 + i); else paint(starPts(x, 222, 14 * s, .4, 4), { wash: PAL.candle, ink: PAL.ink, sw: .4 });
    const x2 = lerp(480, 1440, 1 - u); if (i % 3 === 0) paint(ellPts(x2, 980, 20, 17, 12), { wash: PAL.pumpkin, ink: PAL.ink, sw: .6 }); else paint(starPts(x2, 980, 12, .4, 4), { wash: PAL.violet, ink: PAL.ink, sw: .4 });
  }
  blob(1330, 330, 52, '#FFF3C4'); blob(1352, 314, 46, '#FFF3DC');
  jack(700, 905, 170, 150, false);
  paint(ellPts(960, 880, 110, 78, 22), { wash: '#2a2733', fill: '#555062', fillOp: 70, tex: .5, ink: PAL.ink, sw: 1.0 }); paint(ellPts(960, 820, 100, 22, 20), { wash: '#12101a', ink: PAL.ink, sw: .9 });
  [PAL.pink, PAL.candle, PAL.mint, PAL.violet, PAL.pumpkin, '#E2476E', PAL.pink, PAL.candle].forEach((cl, i) => paint(ellPts(900 + i * 17 + (i % 2) * 6, 806 - (i % 3) * 8, 13, 9, 8, 0, i), { wash: cl, ink: PAL.ink, sw: .4 }));
  ghost(1240, 925, 27, { kind: 'sam', lift: .5 + .4 * Math.sin(t * 2.6), look: [Math.sin(t), -.2], ph: 1, eyes: 'happy' });
  const lines = ['Who ate the', 'Halloween', 'candy?'];
  OVER.push((c) => lines.forEach((ln, i) => {
    const p = backOut(seg(t, .55 + i * .3, 1.0 + i * .3)); if (p <= .01) return;
    c.save(); c.translate(960, 395 + (i - 1) * 138); c.rotate(-.03 + i * .02); c.scale(p, p); c.font = '700 138px Caveat'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.lineJoin = 'round';
    c.lineWidth = 20; c.strokeStyle = '#FFF3DC'; c.strokeText(ln, 0, 0); c.lineWidth = 6; c.strokeStyle = PAL.ink; c.fillStyle = i === 2 ? PAL.ketchup : PAL.pumpkin; c.strokeText(ln, 0, 0); c.fillText(ln, 0, 0); c.restore();
  }));
  sparkles(t, 10); flushGlowQ();
}


/* ───────── lighting overlay (soft glows composited in 2D) ───────── */
function radial(c, x, y, r, col, a) { const g = c.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, `rgba(${col},${a})`); g.addColorStop(1, `rgba(${col},0)`); c.fillStyle = g; c.fillRect(x - r, y - r, r * 2, r * 2); }
function lightOverlay(t, k = 1) {
  if (k <= .01) return;
  OVER.push((c) => {
    c.save(); c.globalCompositeOperation = 'lighter'; const z = FCAM.zoom;
    const S = (x, y) => toScreen(x, y);
    for (const [x, y, r, a] of [[660, 700, 190, .42], [716, 720, 120, .3], [1285, 380, 110, .28], [1342, 600, 120, .35], [95, 320, 300, .22]]) { const [sx, sy] = S(x, y); radial(c, sx, sy, r * z * (1 + .04 * Math.sin(t * 9 + x)), '255,170,70', a * k * flick(t, x)); }
    { const [sx, sy] = S(660, 800); c.save(); c.translate(sx, sy); c.scale(1, .32); radial(c, 0, 0, 340 * z, '255,170,80', .3 * k); c.restore(); }          // warm pool on the floor
    for (let i = 0; i < BULBS; i++) { const [x, y] = bulbPos(i), [sx, sy] = S(x, y), tw = .5 + .5 * Math.sin(t * (1.6 + (i % 5) * .37) + i * 1.9), col = ['255,215,106', '255,156,203', '143,211,182', '255,176,100'][i % 4]; radial(c, sx, sy, 70 * z, col, (.25 + .4 * tw) * k); }
    const [mx, my] = S(1010, 240); radial(c, mx, my, 220 * z, '255,242,184', .22 * k);
    const [fx, fy] = S(1080, 760); c.save(); c.translate(fx, fy); c.scale(1.6, .5); radial(c, 0, 0, 260 * z, '150,170,255', .1 * k); c.restore();                  // moonlight on the rug
    c.restore();
  });
}

function flushGlowQ() {                                                               // Pebble's antenna bulb: soft glow composited in 2D
  const q = GLOWQ.splice(0);
  for (const [x, y, flip, [bx, by], k, s, dy] of q) { const [sx, sy] = sp(x + (flip ? -1 : 1) * bx, y + dy * s + by); OVER.push((c) => { c.save(); c.globalCompositeOperation = 'lighter'; radial(c, sx, sy, 110 * FCAM.zoom * (.6 + k * .7), '255,214,110', .25 + .5 * Math.min(1, k)); c.restore(); }); }
}

function zzzBubbles(t) {                                                              // Biscuit's snore: bubbles that grow, then pop with a tiny burst
  const [bx, by] = LAYOUT.basket; if (STATES.biscuit(t).o.sleep !== true) return;
  const cyc = 2.4, ph = frac(t / cyc), x = bx + 150, y = by - 60;
  if (ph < .78) { const r = 8 + ph * 46; paint(ellPts(x + 20 * ph, y - ph * 70, r, r, 14), { wash: '#DCE6FF', washOp: 150, ink: PAL.ink, sw: .5 }); paint(ellPts(x + 20 * ph - r * .3, y - ph * 70 - r * .3, r * .25, r * .18, 8), { wash: '#ffffff', washOp: 220, ink: null }); if (ph > .45) zzz(x + 20 * ph - 6, y - ph * 70 + 8, 14, 0); }
  else { const k = (ph - .78) / .22; for (let i = 0; i < 8; i++) { const a = i / 8 * TAU; inkLine([[x + 16 + Math.cos(a) * (50 + k * 10), y - 54 + Math.sin(a) * (50 + k * 10)], [x + 16 + Math.cos(a) * (50 + k * 32), y - 54 + Math.sin(a) * (50 + k * 32)]], .6, '#8E7CC3', 'inkfine', 0); } }
}


function pageFlip(page, u) {
  const ang = ease(u) * Math.PI * .5, c = Math.cos(ang), s = Math.sin(ang); if (c < .01) return;
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W * c, H); ctx.clip(); ctx.drawImage(page, 0, 0, W * c, H);
  const g = ctx.createLinearGradient(0, 0, W * c, 0); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(40,25,10,${.5 * s})`); ctx.fillStyle = g; ctx.fillRect(0, 0, W * c, H); ctx.restore();
  const g2 = ctx.createLinearGradient(W * c, 0, W * c + 170, 0); g2.addColorStop(0, `rgba(30,20,10,${.5 * s})`); g2.addColorStop(1, 'rgba(30,20,10,0)'); ctx.save(); ctx.fillStyle = g2; ctx.fillRect(W * c, 0, 170, H); ctx.restore();
}

function candyArt(x, y, r, col) { paint(ellPts(x, y, r * .9, r * .6, 10), { wash: col, ink: PAL.ink, sw: .5 }); paint([[x - r * .9, y], [x - r * 1.5, y - r * .45], [x - r * 1.5, y + r * .45]], { wash: col, ink: PAL.ink, sw: .4 }); paint([[x + r * .9, y], [x + r * 1.5, y - r * .45], [x + r * 1.5, y + r * .45]], { wash: col, ink: PAL.ink, sw: .4 }); }


function assertIds(kind, evtId, drawnOn) { if (evtId !== drawnOn) throw new Error(`label anchor mismatch: ${kind} for ${evtId} was drawn on ${drawnOn}`); }


function drawSheet(x, y, u, who) {
  const dog = who === 'dog', w = lerp(1.2, dog ? 3.4 : 4.4, easeOut(u)) * U * (who === 'leo' ? 1.0 : .88), hh = dog ? 3.8 : (who === 'leo' ? 11 : 8.5), top = lerp(y - (hh + 5) * U, y - hh * U, easeIn(u)) - Math.sin(u * Math.PI) * 60, bot = lerp(top + 60, y - 1.5 * U, u), pts = [], n = 10;
  for (let i = 0; i <= n; i++) { const a = i / n; pts.push([x - w / 2 + w * a + Math.sin(a * 6 + T * 18) * 10 * (1 - u), top + Math.sin(a * Math.PI) * -26 - 8 * Math.sin(a * 9 + T * 14)]); }
  for (let i = n; i >= 0; i--) { const a = i / n; pts.push([x - w * .56 + w * 1.12 * a + Math.sin(a * 5 + T * 12) * 8, bot + Math.sin(a * 7 + T * 16) * 10]); }
  paint(pts, { wash: PAL.sheet, fill: '#c9d6f2', fillOp: 70, tex: .5, ink: PAL.ink, sw: .9, curv: .3 });
}
function puff(x, y, k) { for (let i = 0; i < 9; i++) { const a = i / 9 * TAU, r = (30 + k * 120) * (.7 + hash(i) * .5); paint(ellPts(x + Math.cos(a) * r, y - 70 + Math.sin(a) * r * .55, 26 * (1 - k * .5), 22 * (1 - k * .5), 10), { wash: '#ffffff', washOp: 200 * (1 - k), ink: null }); } }

