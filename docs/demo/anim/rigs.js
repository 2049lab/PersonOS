/* rigs.js - procedural characters. Every rig is char(x, y, s, o): (x, y) is the ground point between the feet, s the unit
 * (a kid is ~9-12 s tall), o the pose: walk/run phase (cycles), aL/aR arm angles (0 = out sideways, + up, -1.25 hanging),
 * eyes, mouth, brows, blush, look [x,y], sq (squash), dy (lift, in s), rot, flip, emote, hooks handL/handR. */
let T = 0;                                   // current time, set by the frame hook
const SK = PAL.skin, SKS = '#E9A98A';

const KID = {
  mia: { legH: 2.1, torsoH: 2.5, hw: 1.5, headR: 2.6, hair: '#8a5530', top: '#F28C38', bottom: '#2B2A33', skirt: true, shoe: '#FF7EB6', sole: '#FFFFFF', sock: '#FFF5F8' },
  leo: { legH: 3.9, torsoH: 3.3, hw: 1.7, headR: 2.4, hair: '#6a4a35', top: '#8E7CC3', bottom: '#3b3a4a', striped: true, shoe: '#2B2A33', sole: '#FFFFFF' },
  sam: { legH: 1.9, torsoH: 3.0, hw: 2.25, headR: 2.65, hair: '#a9825a', top: '#8FD3B6', bottom: '#9aa0aa', shoe: '#2B2A33', sole: '#FFFFFF', bell: true },
  mum: { legH: 3.5, torsoH: 3.9, hw: 2.0, headR: 2.4, hair: '#8a5a3a', top: '#9b7bc8', bottom: '#6b5a9a', apron: true, shoe: '#FFB6D2', sole: '#FFFFFF', slippers: true },
};
const blinkAt = (ph = 0) => ((T * .8 + ph) % 3.7) < .13;

function kid(x, y, s, o = {}) {
  const c = KID[o.who || 'mia'], sw = clamp(s / 14, .5, 2.2), J = s * .035, sq = o.sq || 0;
  const legH = c.legH, torsoH = c.torsoH, hw = c.hw, R = c.headR;
  const hipY = -legH, shY = -(legH + torsoH * .9), headY = -(legH + torsoH + R * .78);
  if (!o.noShadow) paint(ellPts(x, y + s * .12, s * (hw + 1.6), s * .5, 16), { wash: PAL.ink, washOp: 55, ink: null });
  push(); translate(x, y + (o.dy || 0) * s);
  const tn = o.turn || 0;                                 // body turn toward a target: narrower torso, a lean, the head swings round
  if (o.rot || tn) rotate((o.rot || 0) + tn * .015);
  scale((o.flip ? -1 : 1) * (1 + sq * .5) * (1 - .2 * Math.abs(tn)), 1 - sq);
  const ph = o.walk ?? o.run, moving = ph != null, run = o.run != null;
  const bounce = moving ? Math.abs(Math.sin(ph * TAU)) * (run ? .5 : .28) : 0;
  translate(0, -bounce * s);
  const lean = run ? .16 : 0; if (lean) rotate(lean);

  // pigtails / back hair behind the head
  const swing = (o.swing ?? (moving ? Math.sin(ph * TAU * 2) * .35 : Math.sin(T * 2.2) * .06));
  if (o.who === 'mia') for (const side of [-1, 1]) {
    push(); translate(side * R * .95 * s, (headY + R * .1) * s); rotate(side * (.35 + swing) );
    paint(ellPts(side * .35 * s, 1.4 * s, .95 * s, 1.9 * s, 14, J), { wash: c.hair, fill: '#c98a55', fillOp: 70, tex: .5, ink: PAL.ink, sw: sw * .7 });
    pop();
  }
  if (o.who === 'mum') paint(ellPts(0, (headY - R * .9) * s, 1.15 * s, 1.0 * s, 14, J), { wash: c.hair, ink: PAL.ink, sw: sw * .7 });   // bun

  // legs
  const leg = (i) => {
    const side = i ? 1 : -1, u = (ph ?? 0) + (i ? .5 : 0), stride = run ? 1.5 : .85, lift = moving ? Math.max(0, -Math.cos(u * TAU)) * (run ? 1.1 : .55) : 0;
    let fx = side * hw * .48 + (moving ? Math.sin(u * TAU) * stride : 0);
    if (o.tip) { fx = side * hw * .4 + (i ? .5 : -.2); }
    let fy = -lift - (o.tip && i ? .8 : 0);
    if (o.legsOut) { fx = side * (hw * .5 + o.legsOut); }
    const hx = side * hw * .5, hy = hipY;
    const kneeU = .42;
    const skinLeg = c.skirt || c.slippers || o.who === 'sam';
    if (c.striped) {                                    // Leo: shorts then striped socks
      const kx = lerp(hx, fx, kneeU), ky = lerp(hy, fy, kneeU);
      paint(limbPts(hx * s, hy * s, kx * s, ky * s, .95 * s), { wash: SK, ink: PAL.ink, sw: sw * .6 });
      const n = 6, x0 = kx, y0 = ky;
      for (let k = 0; k < n; k++) {
        const a0 = k / n, a1 = (k + 1.02) / n;
        paint(limbPts(lerp(x0, fx, a0) * s, lerp(y0, fy, a0) * s, lerp(x0, fx, a1) * s, lerp(y0, fy, a1) * s, .98 * s), { wash: k % 2 ? '#FFFFFF' : PAL.green, ink: null });
      }
      inkLine([[lerp(x0, fx, 0) * s - .5 * s, lerp(y0, fy, 0) * s], [fx * s - .5 * s, fy * s]], sw * .6, PAL.ink, 'inkfine', 0);
      inkLine([[lerp(x0, fx, 0) * s + .5 * s, lerp(y0, fy, 0) * s], [fx * s + .5 * s, fy * s]], sw * .6, PAL.ink, 'inkfine', 0);
    } else {
      paint(limbPts(hx * s, hy * s, fx * s, fy * s, .9 * s), { wash: skinLeg ? SK : c.bottom, ink: PAL.ink, sw: sw * .6 });
      if (c.sock) paint(limbPts(lerp(hx, fx, .6) * s, lerp(hy, fy, .6) * s, fx * s, fy * s, .95 * s), { wash: c.sock, ink: PAL.ink, sw: sw * .5 });
    }
    const dirX = (fx - hx) * .25;
    paint(ellPts((fx + (o.who === 'mum' ? 0 : .2)) * s, (fy - .05) * s, (c.slippers ? 1.0 : .95) * s, .5 * s, 12, J), { wash: c.shoe, ink: PAL.ink, sw: sw * .7 });
    if (c.sole && !c.slippers) paint(rectPts((fx - .75) * s, (fy + .15) * s, 1.9 * s, .22 * s, 0), { wash: c.sole, ink: null });
    if (c.bell && !i) { const jig = (moving ? Math.sin(T * 30) : Math.sin(T * 3) * .15) * .12; blob((fx - .6 + jig) * s, (fy - .35) * s, .38 * s, PAL.gold); inkLine([[(fx - .6) * s, (fy - .2) * s], [(fx - .6 + jig) * s, (fy - .55) * s]], sw * .4, PAL.ink, 'inkfine', 0); }
  };
  leg(0); leg(1);

  // skirt / shorts hint
  if (c.skirt) paint([[-hw * s * 1.0, hipY * s - .2 * s], [hw * s * 1.0, hipY * s - .2 * s], [hw * s * 1.7, (hipY + .9) * s], [-hw * s * 1.7, (hipY + .9) * s]], { wash: c.bottom, ink: PAL.ink, sw: sw * .7 });

  // arms
  const arm = (side, a, hook) => {
    const sx0 = side * hw * .98, sy0 = shY, L = 2.3, ex = sx0 + side * Math.cos(a) * L, ey = sy0 - Math.sin(a) * L;
    paint(limbPts(sx0 * s, sy0 * s, ex * s, ey * s, .95 * s, .8 * s), { wash: o.who === 'mum' ? c.top : c.top, ink: PAL.ink, sw: sw * .7 });
    paint(ellPts(ex * s, ey * s, .52 * s, .52 * s, 10), { wash: SK, ink: PAL.ink, sw: sw * .6 });
    if (hook) { push(); translate(ex * s, ey * s); hook(s, sw, side); pop(); }
  };
  const aL = o.aL ?? (moving ? -1.2 + Math.sin((ph + .5) * TAU) * (run ? .7 : .25) : -1.25), aR = o.aR ?? (moving ? -1.2 + Math.sin(ph * TAU) * (run ? .7 : .25) : -1.25);
  arm(-1, aL, o.handL);

  // torso
  const tp = [[-hw * .9 * s, shY * s - .2 * s], [hw * .9 * s, shY * s - .2 * s], [hw * 1.05 * s, hipY * s], [-hw * 1.05 * s, hipY * s]];
  paint(tp, { wash: c.top, fill: '#ffffff', fillOp: 30, tex: .5, ink: PAL.ink, sw: sw * .8, curv: .15 });
  if (o.who === 'mia') {                                   // pumpkin face on the sweater
    paint([[-.55 * s, (shY + 1.0) * s], [-.15 * s, (shY + 1.0) * s], [-.35 * s, (shY + .6) * s]], { wash: PAL.ink, ink: null });
    paint([[.15 * s, (shY + 1.0) * s], [.55 * s, (shY + 1.0) * s], [.35 * s, (shY + .6) * s]], { wash: PAL.ink, ink: null });
    inkLine([[-.6 * s, (shY + 1.6) * s], [-.2 * s, (shY + 1.9) * s], [.2 * s, (shY + 1.6) * s], [.6 * s, (shY + 1.9) * s]], sw * .5, PAL.ink, 'inkfine', 0);
  }
  if (o.who === 'leo') {
    inkLine([[-.35 * s, (shY + .3) * s], [-.4 * s, (shY + 1.5) * s]], sw * .6, '#fff', 'inkfine', 0); inkLine([[.35 * s, (shY + .3) * s], [.4 * s, (shY + 1.5) * s]], sw * .6, '#fff', 'inkfine', 0);
    paint(rectPts(-hw * .6 * s, (hipY - 1.3) * s, hw * 1.2 * s, 1.0 * s, 0), { ink: PAL.ink, sw: sw * .5 });
  }
  if (o.who === 'sam') {
    paint(ellPts(0, (shY + 1.4) * s, 1.0 * s, .9 * s, 14), { wash: PAL.ink, ink: null });
    paint([[-.85 * s, (shY + .8) * s], [-.55 * s, (shY + .35) * s], [-.3 * s, (shY + .6) * s]], { wash: PAL.ink, ink: null }); paint([[.85 * s, (shY + .8) * s], [.55 * s, (shY + .35) * s], [.3 * s, (shY + .6) * s]], { wash: PAL.ink, ink: null });
    blob(-.35 * s, (shY + 1.3) * s, .12 * s, PAL.mint); blob(.35 * s, (shY + 1.3) * s, .12 * s, PAL.mint);
  }
  if (c.apron) {
    paint([[-hw * .7 * s, shY * s + .2 * s], [hw * .7 * s, shY * s + .2 * s], [hw * .95 * s, (hipY + .6) * s], [-hw * .95 * s, (hipY + .6) * s]], { wash: '#FFF3DA', ink: PAL.ink, sw: sw * .6 });
    paint(ellPts(0, (hipY - 1.0) * s, .6 * s, .55 * s, 10), { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .5 });
  }
  arm(1, aR, o.handR);

  // head
  const hx = (o.headX || 0) + tn * .45;
  paint(ellPts(hx * s, headY * s, R * s, R * .96 * s, 26, J * .6), { wash: SK, fill: SKS, fillOp: 50, tex: .5, border: .4, ink: PAL.ink, sw: sw * .9 });
  // hair front
  const hr = c.hair;
  if (o.who === 'mia') {
    paint(hairCap(hx, headY, R, s, .55), { wash: hr, ink: PAL.ink, sw: sw * .7 });
    for (const side of [-1, 1]) { blob(side * R * .95 * s, (headY + R * .1) * s, .38 * s, PAL.pumpkin); }
  } else if (o.who === 'leo') {
    paint(hairCap(hx, headY, R, s, .5), { wash: hr, ink: PAL.ink, sw: sw * .7 });
    for (const [bx, sw_, drop] of [[-1.15, -.9, 1.5], [-.45, -.4, 1.75], [.25, .3, 1.9], [.95, .85, 1.55]]) {                   // a messy floppy fringe over the forehead
      const x0 = bx * s, y0 = (headY - R * .92) * s, tipx = x0 + sw_ * s * .5, tipy = y0 + drop * s * .9 + Math.sin(T * 2 + bx) * .04 * s;
      paint([[x0 - .55 * s, y0], [x0 + .55 * s, y0], [tipx + .25 * s, tipy], [tipx - .12 * s, tipy + .18 * s]], { wash: hr, ink: PAL.ink, sw: sw * .5 });
    }
    inkLine([[.2 * s, (headY - R * 1.02) * s], [.55 * s, (headY - R * 1.32) * s], [.95 * s, (headY - R * 1.2) * s], [.7 * s, (headY - R * 1.0) * s]], sw * 1.1, hr, 'ink', .6);   // one small cowlick
  } else if (o.who === 'sam') {
    paint(hairCap(hx, headY, R, s, .42), { wash: hr, ink: null });
    for (let k = 0; k < 7; k++) { const a = Math.PI * (1.08 + k / 6 * .84); paint(ellPts(Math.cos(a) * R * 1.0 * s, (headY + Math.sin(a) * R * 1.0) * s, .75 * s, .7 * s, 10), { wash: hr, ink: PAL.ink, sw: sw * .6 }); }
  } else if (o.who === 'mum') {
    paint(hairCap(hx, headY, R, s, .5), { wash: hr, ink: PAL.ink, sw: sw * .7 });
    for (const side of [-1, 1]) for (let k = 0; k < 2; k++) paint(rrPts((side * (R * .98) - .5 + (side < 0 ? -.2 : 0)) * s, (headY - R * .55 + k * 1.15) * s, 1.0 * s, .62 * s, .3 * s), { wash: '#FF9CCB', ink: PAL.ink, sw: sw * .5 });
  }
  faceRig(c, o, hx, headY, R, s, sw);
  if (o.draw) o.draw(s, sw);
  pop();
  if (o.emote) emote(o.emote, x + (o.flip ? -1 : 1) * 2.4 * s, y + (o.dy || 0) * s + (headY - R - 1.2) * s, s, o.emoteK ?? 1);
}
function hairCap(hx, hy, R, s, fringe) {
  const p = []; for (let i = 0; i <= 14; i++) { const a = Math.PI * 1.0 + i / 14 * Math.PI; p.push([hx * s + Math.cos(a) * R * 1.06 * s, hy * s + Math.sin(a) * R * 1.02 * s]); }
  p.push([hx * s + R * .85 * s, (hy - R * fringe * .15) * s], [hx * s + R * .35 * s, (hy - R * fringe * .55) * s], [hx * s - R * .15 * s, (hy - R * fringe * .2) * s], [hx * s - R * .6 * s, (hy - R * fringe * .62) * s], [hx * s - R * .98 * s, (hy - R * .1) * s]);
  return p;
}

function faceRig(c, o, hx, hy, R, s, sw) {
  const e = (o.squint || 0) > .5 ? 'closed' : (o.eyes || 'dot'), blink = (e === 'dot' || e === 'look') && blinkAt(o.bph || 0), lx = (o.look?.[0] || 0) * .22 * s, ly = (o.look?.[1] || 0) * .18 * s;
  const ang = (o.ang ?? o.turn) || 0, ax = ang * R * .36 * s;
  const ey = (hy + R * .1) * s, ex = R * .42 * s * (1 - Math.abs(ang) * .18), er = .3 * s;
  for (const side of [-1, 1]) {
    const cx = hx * s + ax + side * ex;
    if (e === 'dot' || e === 'look') {
      if (blink) inkLine([[cx - er, ey], [cx + er, ey]], sw * .8, PAL.ink, 'ink', 0);
      else { paint(ellPts(cx + lx, ey + ly, er * .8, er * 1.1, 10), { wash: PAL.ink, ink: null }); blob(cx + lx + er * .25, ey + ly - er * .35, er * .28, '#fff'); }
    } else if (e === 'wide') {
      paint(ellPts(cx, ey, er * 1.45, er * 1.6, 12), { wash: '#fff', ink: PAL.ink, sw: sw * .5 });
      paint(ellPts(cx + lx, ey + ly, er * .85, er * 1.0, 10), { wash: PAL.ink, ink: null }); blob(cx + lx + er * .25, ey + ly - er * .3, er * .26, '#fff');
    } else if (e === 'closed' || e === 'happy') inkLine([[cx - er * 1.1, ey + er * .35], [cx, ey - er * .75], [cx + er * 1.1, ey + er * .35]], sw * .9, PAL.ink, 'ink', .6);
    else if (e === 'sly') { paint([[cx - er * 1.1, ey - er * .1], [cx + er * 1.1, ey - er * (side > 0 ? .5 : .1)], [cx + er * .9, ey + er * .9], [cx - er * .9, ey + er * .9]], { wash: PAL.ink, ink: null, curv: .4 }); }
    else if (e === 'sad') { paint(ellPts(cx + lx, ey + ly + er * .3, er * .8, er * 1.0, 10), { wash: PAL.ink, ink: null }); blob(cx + er * .25, ey - er * .1, er * .3, '#fff'); }
    else if (e === 'star') paint(starPts(cx, ey, er * 1.9 * (1 + .12 * Math.sin(T * 13 + side))), { wash: PAL.candle, ink: PAL.ink, sw: sw * .4 });
    else if (e === 'x') { inkLine([[cx - er, ey - er], [cx + er, ey + er]], sw * .8, PAL.ink, 'ink', 0); inkLine([[cx + er, ey - er], [cx - er, ey + er]], sw * .8, PAL.ink, 'ink', 0); }
    else if (e === 'heart') paint(heartPts(cx, ey, er * 1.5), { wash: '#E2476E', ink: null });
    else if (e === 'swirl') { const sp = []; for (let k = 0; k < 14; k++) { const a = k * .8 + T * 7 * side, r = k * .05 * s; sp.push([cx + Math.cos(a) * r, ey + Math.sin(a) * r]); } inkLine(sp, sw * .5, PAL.ink, 'inkfine', .6); }
  }
  if (o.blush ?? true) for (const side of [-1, 1]) paint(ellPts(hx * s + ax + side * R * .68 * s, (hy + R * .42) * s, .5 * s, .28 * s, 12), { wash: PAL.pink, washOp: 120, ink: null });
  const b = o.brows || (e === 'sad' ? 'worried' : null);
  if (b) for (const side of [-1, 1]) {
    const bx = hx * s + ax + side * ex, by = ey - .75 * s, tilt = b === 'worried' ? -side * .3 : b === 'angry' ? side * .38 : 0, lift = b === 'up' ? -.35 * s : 0;
    inkLine([[bx - .45 * s, by + lift + tilt * s * .6], [bx + .45 * s, by + lift - tilt * s * .6]], sw * .8, PAL.ink, 'ink', 0);
  }
  const m = o.mouth || 'smile', my = (hy + R * .58) * s, mx = hx * s + ax * 1.1;
  if (m === 'smile') inkLine([[mx - .5 * s, my - .08 * s], [mx, my + .28 * s], [mx + .5 * s, my - .08 * s]], sw * .75, PAL.ink, 'ink', .6);
  else if (m === 'o') paint(ellPts(mx, my + .08 * s, .26 * s, .32 * s, 10), { wash: '#6A2A35', ink: PAL.ink, sw: sw * .4 });
  else if (m === 'O') paint(ellPts(mx, my + .2 * s, .5 * s, .66 * s, 14), { wash: '#6A2A35', ink: PAL.ink, sw: sw * .5 });
  else if (m === 'flat') inkLine([[mx - .4 * s, my], [mx + .4 * s, my]], sw * .75, PAL.ink, 'ink', 0);
  else if (m === 'wobble') inkLine([[mx - .65 * s, my], [mx - .32 * s, my - .18 * s], [mx, my], [mx + .32 * s, my - .18 * s], [mx + .65 * s, my]], sw * .6, PAL.ink, 'ink', .3);
  else if (m === 'grin') paint([[mx - .75 * s, my - .12 * s], [mx + .75 * s, my - .12 * s], [mx + .45 * s, my + .5 * s], [mx - .45 * s, my + .5 * s]], { wash: '#6A2A35', ink: PAL.ink, sw: sw * .5, curv: .4 });
  else if (m === 'laugh') { const k = .6 + .4 * Math.abs(Math.sin(T * 14)); paint([[mx - .7 * s, my - .1 * s], [mx + .7 * s, my - .1 * s], [mx + .5 * s, my + .75 * s * k], [mx - .5 * s, my + .75 * s * k]], { wash: '#6A2A35', ink: PAL.ink, sw: sw * .5, curv: .5 }); }
  else if (m === 'smirk') inkLine([[mx - .5 * s, my + .05 * s], [mx + .1 * s, my + .22 * s], [mx + .6 * s, my - .15 * s]], sw * .75, PAL.ink, 'ink', .6);
}

function emote(kind, x, y, s, k = 1) {
  const sc = s * (1 + .12 * Math.sin(T * 12)) * backOut(k);
  if (kind === '!') { paint(rrPts(x - .22 * sc, y - 1.6 * sc, .44 * sc, 1.3 * sc, .2 * sc), { wash: PAL.ketchup, ink: PAL.ink, sw: .6 }); blob(x, y + .1 * sc, .26 * sc, PAL.ketchup); }
  else if (kind === '?') { inkLine([[x - .5 * sc, y - 1.3 * sc], [x, y - 1.8 * sc], [x + .55 * sc, y - 1.3 * sc], [x + .2 * sc, y - .7 * sc], [x, y - .35 * sc]], 1.3, PAL.violet, 'ink', .6); blob(x, y + .1 * sc, .24 * sc, PAL.violet); }
  else if (kind === 'note') { paint(ellPts(x, y, .45 * sc, .32 * sc, 10, 0, -.4), { wash: PAL.ink, ink: null }); inkLine([[x + .4 * sc, y - .1 * sc], [x + .4 * sc, y - 1.9 * sc], [x + 1.0 * sc, y - 1.4 * sc]], .9, PAL.ink, 'ink', 0); }
  else if (kind === 'heart') paint(heartPts(x, y - sc, .9 * sc), { wash: '#E2476E', ink: PAL.ink, sw: .5 });
  else if (kind === 'sweat') { paint([[x, y - 1.5 * sc], [x - .4 * sc, y - .8 * sc], [x + .4 * sc, y - .8 * sc]], { wash: PAL.sky, ink: PAL.ink, sw: .4 }); paint(ellPts(x, y - .75 * sc, .4 * sc, .4 * sc, 10), { wash: PAL.sky, ink: PAL.ink, sw: .4 }); }
  else if (kind === 'star') paint(starPts(x, y - sc, .9 * sc), { wash: PAL.candle, ink: PAL.ink, sw: .5 });
  else if (kind === 'anger') { for (let i = 0; i < 4; i++) { const a = i * Math.PI / 2 + .8; inkLine([[x + Math.cos(a) * .35 * sc, y - sc + Math.sin(a) * .35 * sc], [x + Math.cos(a) * .95 * sc, y - sc + Math.sin(a) * .95 * sc]], 1.4, PAL.ketchup, 'ink', 0); } }
}
function zzz(x, y, s, t0 = 0) {                                         // Biscuit's snore
  for (let k = 0; k < 3; k++) {
    const ph = frac((T - t0) * .5 + k / 3), sz = (.5 + ph * 1.0) * s, px = x + ph * 1.4 * s + Math.sin(ph * 8) * .3 * s, py = y - ph * 3.2 * s;
    const a = Math.sin(ph * Math.PI);
    inkLine([[px - sz * .5, py - sz * .5], [px + sz * .5, py - sz * .5], [px - sz * .5, py + sz * .5], [px + sz * .5, py + sz * .5]], 1.3 * Math.max(.4, a), PAL.violet, 'ink', 0);
  }
}

/* ───── Pebble: egg-shaped cream shell, a dark visor holding one big camera lens, stubby arms, a witch hat, a glowing antenna ───── */
function pebble(x, y, s, o = {}) {
  const sw = clamp(s / 14, .5, 2.2), J = s * .03;
  if (!o.noShadow) paint(ellPts(x, y + s * .1, s * 3.0, s * .45, 16), { wash: PAL.ink, washOp: 55, ink: null });
  push(); translate(x, y + (o.dy || 0) * s); rotate(o.rot || 0); const sq = o.sq || 0; scale((o.flip ? -1 : 1) * (1 + sq * .5), 1 - sq);
  const roll = o.roll || 0, bob = Math.abs(Math.sin(roll * TAU * 2)) * (o.rolling ? .12 : 0);
  translate(0, -bob * s);
  for (const wx of [-1.5, 0, 1.5]) {                                              // wheels
    push(); translate(wx * s, -.5 * s);
    paint(ellPts(0, 0, .52 * s, .52 * s, 12), { wash: PAL.ink, ink: null });
    paint(ellPts(0, 0, .22 * s, .22 * s, 8), { wash: '#8c8a99', ink: null });
    for (let k = 0; k < 2; k++) { const a = roll * TAU + k * Math.PI; inkLine([[Math.cos(a) * .4 * s, Math.sin(a) * .4 * s], [-Math.cos(a) * .4 * s, -Math.sin(a) * .4 * s]], sw * .4, '#8c8a99', 'inkfine', 0); }
    pop();
  }
  // arms (behind the shell): stubby, with an optional hand hook
  const arm = (side, a, hook) => {
    push(); translate(side * 2.35 * s, -2.1 * s); rotate(side * -a);
    paint(limbPts(0, 0, side * 1.15 * s, 0, .75 * s, .6 * s), { wash: '#F3E9D3', ink: PAL.ink, sw: sw * .7 }); translate(side * 1.25 * s, 0);
    paint(ellPts(0, 0, .45 * s, .45 * s, 10), { wash: PEBBLE.palette.shell, ink: PAL.ink, sw: sw * .7 }); if (hook) hook(s, sw, side); pop();
  };
  arm(-1, o.armL ?? .55, o.handL);
  // shell: an egg, wider at the bottom, with soft shading and a sheen
  const bp = []; for (let i = 0; i < 32; i++) { const a = i / 32 * TAU, c = Math.cos(a), sn = Math.sin(a), wdt = 2.45 + .45 * Math.max(0, sn); bp.push([c * wdt * s + jit(J), -3.0 * s + sn * 2.65 * s * (sn < 0 ? 1.05 : .85)]); }
  paint(bp, { wash: PEBBLE.palette.shell, fill: '#cfdcf0', fillOp: 70, tex: .5, border: .45, ink: PAL.ink, sw: sw * 1.0 });
  paint(ellPts(.9 * s, -1.3 * s, 1.9 * s, .8 * s, 14, 0, -.25), { wash: PEBBLE.palette.shellShade, washOp: 150, ink: null });
  paint(ellPts(-1.5 * s, -4.6 * s, .55 * s, .25 * s, 10, 0, -.6), { wash: '#ffffff', washOp: 200, ink: null });
  paint(ellPts(1.45 * s, -1.0 * s, .52 * s, .5 * s, 12), { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .6 });                    // a pumpkin sticker
  paint([[1.25 * s, -1.15 * s], [1.4 * s, -1.15 * s], [1.32 * s, -1.3 * s]], { wash: PAL.ink, ink: null }); paint([[1.5 * s, -1.15 * s], [1.65 * s, -1.15 * s], [1.57 * s, -1.3 * s]], { wash: PAL.ink, ink: null });
  inkLine([[1.2 * s, -.85 * s], [1.45 * s, -.7 * s], [1.7 * s, -.85 * s]], sw * .4, PAL.ink, 'inkfine', .5);
  arm(1, o.armR ?? o.arm ?? .55, o.handR ?? o.hand);
  // visor with one big lens
  push(); translate(0, -3.15 * s);
  paint(ellPts(0, 0, 2.05 * s, 1.55 * s, 26), { wash: PEBBLE.palette.visor, fill: PEBBLE.palette.visorLight, fillOp: 70, tex: .4, ink: PAL.ink, sw: sw * .9 });
  paint(ellPts(-.9 * s, -.75 * s, .7 * s, .22 * s, 10, 0, -.4), { wash: '#ffffff', washOp: 70, ink: null });
  const lx = (o.look?.[0] || 0) * .32 * s, ly = (o.look?.[1] || 0) * .22 * s, lr = 1.12 * s, pup = o.pupil ?? .5, lid = o.lid ?? 0;
  push(); translate(lx, ly);
  paint(ellPts(0, 0, lr, lr, 22), { wash: PEBBLE.palette.irisRing, fill: '#7e86a8', fillOp: 90, tex: .3, ink: PAL.ink, sw: sw * .8 });
  paint(ellPts(0, 0, lr * .82, lr * .82, 22), { wash: o.iris ?? PEBBLE.palette.iris, ink: null });
  paint(ellPts(0, 0, lr * .6, lr * .6, 18), { wash: PEBBLE.palette.irisDeep, ink: null });
  for (let k = 0; k < 6; k++) { const a = T * .25 + k * TAU / 6; inkLine([[Math.cos(a) * lr * .25, Math.sin(a) * lr * .25], [Math.cos(a + .6) * lr * .78, Math.sin(a + .6) * lr * .78]], sw * .35, '#1d6e9c', 'inkfine', 0); }   // aperture blades
  paint(ellPts(0, 0, lr * .62 * pup * 1.4, lr * .62 * pup * 1.4, 16), { wash: PEBBLE.palette.pupil, ink: null });
  blob(-lr * .34, -lr * .36, lr * .2, '#ffffff'); blob(lr * .3, lr * .28, lr * .1, '#ffffff', 220);
  pop();
  if (lid > .02) { const t = lid * lr * 1.2, col = PEBBLE.palette.visor;
    paint([[-lr * 1.4, -lr * 1.4], [lr * 1.4, -lr * 1.4], [lr * 1.4, -lr + t], [0, -lr + t * 1.25], [-lr * 1.4, -lr + t]], { wash: col, ink: null });
    paint([[-lr * 1.4, lr * 1.4], [lr * 1.4, lr * 1.4], [lr * 1.4, lr - t * .8], [0, lr - t], [-lr * 1.4, lr - t * .8]], { wash: col, ink: null });
    inkLine([[-lr * .95, -lr + t * 1.0], [0, -lr + t * 1.3], [lr * .95, -lr + t * 1.0]], sw * .8, PAL.ink, 'ink', .6); }
  if (o.wink) { paint(ellPts(lx, ly, lr * 1.05, lr * 1.05, 18), { wash: PEBBLE.palette.visor, ink: null }); inkLine([[-lr * .9, 0], [0, -lr * .45], [lr * .9, 0]], sw * 1.3, PEBBLE.palette.shell, 'ink', .7); }
  pop();
  for (const side of [-1, 1]) paint(ellPts(side * 1.95 * s, -2.1 * s, .38 * s, .22 * s, 10), { wash: PAL.pink, washOp: 140, ink: null });
  arm(-1, -99, null);                                                                          // (no-op placeholder keeps arm order explicit)
  // witch hat, tilted, with a band and a buckle
  push(); translate(-.9 * s, -5.45 * s); rotate(-.28 + Math.sin(T * 2.4) * .04);
  paint(ellPts(0, 0, 1.45 * s, .36 * s, 14), { wash: '#2f2447', ink: PAL.ink, sw: sw * .7 });
  paint([[-.8 * s, -.05 * s], [.8 * s, -.05 * s], [.35 * s, -1.9 * s], [.75 * s, -2.6 * s + Math.sin(T * 3) * .1 * s], [-.1 * s, -2.0 * s]], { wash: '#3a2d59', ink: PAL.ink, sw: sw * .8 });
  paint(rectPts(-.78 * s, -.55 * s, 1.56 * s, .34 * s, 0), { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .5 });
  paint(rectPts(-.14 * s, -.57 * s, .28 * s, .38 * s, 0), { wash: PAL.gold, ink: PAL.ink, sw: sw * .4 });
  pop();
  // antenna with a glowing bulb
  const sway = Math.sin(T * 2.6) * .25 + (o.antSway || 0), glowK = o.glow ?? .5, bx = 1.65 * s + sway * s, by = -6.1 * s;
  inkLine([[1.0 * s, -5.2 * s], [1.35 * s, -5.7 * s], [bx, by + .45 * s]], sw * .8, PAL.ink, 'ink', .6);
  paint(ellPts(bx, by, .5 * s, .56 * s, 12), { wash: mixCol(PEBBLE.palette.bulb, PEBBLE.palette.bulbGlow, glowK), ink: PAL.ink, sw: sw * .7 });
  blob(bx - .15 * s, by - .2 * s, .13 * s, '#ffffff', 220);
  const wp = [bx, by]; GLOWQ.push([x, y, o.flip, wp, glowK, s, o.dy || 0, o.rot || 0]);
  pop();
  if (o.emote) emote(o.emote, x + 2.8 * s, y + (o.dy || 0) * s - 6.2 * s, s, o.emoteK ?? 1);
}
const GLOWQ = [];

/* ───── Biscuit the sausage dog (side view, faces right unless flipped) ───── */
function biscuit(x, y, s, o = {}) {
  const sw = clamp(s / 14, .5, 2.2), B = PAL.brown, BD = '#8d5a31';
  if (!o.noShadow) paint(ellPts(x, y + s * .1, s * 3.6, s * .45, 16), { wash: PAL.ink, washOp: 55, ink: null });
  push(); translate(x, y + (o.dy || 0) * s); scale((o.flip ? -1 : 1) * (1 + (o.sq || 0) * .5), 1 - (o.sq || 0));
  if (o.sleep) {
    const br = 1 + .035 * Math.sin(T * 2.2);
    push(); scale(1, br);
    paint(ellPts(0, -1.2 * s, 3.1 * s, 1.35 * s, 24), { wash: B, fill: BD, fillOp: 60, tex: .5, ink: PAL.ink, sw: sw * .9 });
    pop();
    paint(ellPts(2.5 * s, -1.0 * s, 1.35 * s, 1.1 * s, 16), { wash: B, ink: PAL.ink, sw: sw * .9 });
    paint(ellPts(3.4 * s, -.7 * s, .7 * s, .5 * s, 12), { wash: '#d9a86f', ink: PAL.ink, sw: sw * .6 });
    blob(3.9 * s, -.85 * s, .2 * s, PAL.ink);
    inkLine([[2.0 * s, -1.4 * s], [2.4 * s, -1.2 * s], [2.8 * s, -1.4 * s]], sw * .7, PAL.ink, 'ink', .5);
    paint(ellPts(1.95 * s, -1.0 * s, .55 * s, 1.0 * s, 10, 0, .3), { wash: BD, ink: PAL.ink, sw: sw * .7 });         // ear
    paint([[1.4 * s, -1.5 * s], [.8 * s, -1.9 * s], [1.0 * s, -.9 * s]], { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .6 });
    inkLine([[-2.8 * s, -1.0 * s], [-3.4 * s, -.6 * s], [-3.2 * s, -.2 * s]], sw * .8, PAL.ink, 'ink', .6);           // tail tucked
    pop(); return;
  }
  const trot = o.trot, wag = o.wag ?? 0;
  const leg = (lx, i) => { const u = (trot ?? 0) + (i % 2 ? .5 : 0), sx = trot != null ? Math.sin(u * TAU) * .8 : 0, lift = trot != null ? Math.max(0, -Math.cos(u * TAU)) * .5 : 0;
    paint(limbPts(lx * s, -1.4 * s, (lx + sx * .5) * s, -(.2 + lift) * s, .75 * s), { wash: BD, ink: PAL.ink, sw: sw * .7 }); paint(ellPts((lx + sx * .5 + .15) * s, -(.15 + lift) * s, .55 * s, .28 * s, 8), { wash: BD, ink: PAL.ink, sw: sw * .6 }); };
  leg(-2.2, 0); leg(-1.5, 1);
  paint(ellPts(0, -1.9 * s, 3.2 * s, 1.35 * s, 24), { wash: B, fill: BD, fillOp: 60, tex: .5, ink: PAL.ink, sw: sw * .9 });
  leg(1.4, 1); leg(2.1, 0);
  const wg = Math.sin(T * 14) * wag;                                                                                  // tail
  inkLine([[-3.0 * s, -2.1 * s], [(-3.8 + wg * .3) * s, (-3.0 - wag * .2) * s], [(-4.1 + wg * .8) * s, (-3.9 + Math.abs(wg) * .1) * s]], sw * 1.5, B, 'ink', .6);
  paint([[1.9 * s, -3.0 * s], [2.5 * s, -3.4 * s], [2.9 * s, -2.6 * s]], { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .6 });                       // bandana
  const hb = (trot != null ? Math.abs(Math.sin(trot * TAU * 2)) * .15 : 0) * s;
  paint(ellPts(3.1 * s, -2.9 * s - hb, 1.35 * s, 1.2 * s, 18), { wash: B, ink: PAL.ink, sw: sw * .9 });
  paint(ellPts(4.1 * s, -2.55 * s - hb, .9 * s, .6 * s, 12), { wash: '#d9a86f', ink: PAL.ink, sw: sw * .7 }); blob(4.75 * s, -2.8 * s - hb, .24 * s, PAL.ink);
  const earA = o.earUp ? -.7 + Math.sin(T * 20) * .08 : .5 + Math.sin(T * 7) * (wag ? .15 : .05) + (trot != null ? Math.sin(trot * TAU * 2) * .25 : 0);
  push(); translate(2.6 * s, -3.5 * s - hb); rotate(earA); paint(ellPts(-.2 * s, .9 * s, .55 * s, 1.15 * s, 10), { wash: BD, ink: PAL.ink, sw: sw * .7 }); pop();
  const eyes = o.eyes || 'dot';
  if (eyes === 'dot' && !blinkAt(.7)) { blob(3.55 * s, -3.1 * s - hb, .22 * s, PAL.ink); blob(3.62 * s, -3.18 * s - hb, .07 * s, '#fff'); }
  else inkLine([[3.3 * s, -3.1 * s - hb], [3.8 * s, -3.1 * s - hb]], sw * .7, PAL.ink, 'ink', 0);
  if (o.tongue) paint(ellPts(4.2 * s, -1.85 * s - hb, .28 * s, .5 * s, 8), { wash: '#f06a8a', ink: PAL.ink, sw: sw * .5 });
  pop();
}

/* ───── Ghost: a bedsheet over a wearer. The cue (shoes / socks / bell / dog ears+tail) peeks out and walks with the wearer ───── */
const GHOST = { mia: { H: 8.2, Wd: 3.6 }, leo: { H: 11.0, Wd: 4.3 }, sam: { H: 8.8, Wd: 4.6 }, dog: { H: 3.5, Wd: 5.2 } };
function ghostTop(kind, y, s, lift, sheetK = 1) { return y - lift * s - GHOST[kind].H * s * sheetK; }
function ghostEyes(c, ex, ey, s, sw, shape, dark, look = [0, 0]) {                               // eye holes that change shape with emotion
  const lk = look;
  for (const side of [-1, 1]) {
    const cx = ex * side + lk[0] * .12 * s, cy = ey + lk[1] * .1 * s;
    if (shape === 'happy') inkLine([[cx - .42 * s, cy + .22 * s], [cx, cy - .42 * s], [cx + .42 * s, cy + .22 * s]], sw * 2.6, PAL.ink, 'ink', .7);
    else if (shape === 'sly') { paint([[cx - .5 * s, cy - .12 * s], [cx + .5 * s, cy - (side > 0 ? .22 : .02) * s], [cx + .42 * s, cy + .28 * s], [cx - .42 * s, cy + .3 * s]], { wash: PAL.ink, ink: null, curv: .4 }); }
    else if (shape === 'closed') inkLine([[cx - .38 * s, cy], [cx + .38 * s, cy]], sw * .9, PAL.ink, 'ink', 0);
    else if (shape === 'wide') { paint(ellPts(cx, cy, .56 * s, .72 * s, 12), { wash: PAL.ink, ink: null }); blob(cx + .14 * s, cy - .24 * s, .12 * s, '#ffffff'); }
    else { paint(ellPts(cx, cy, .4 * s, .55 * s, 12), { wash: PAL.ink, ink: null }); }
    if (dark > .1) { glowAt(cx, cy, 1.0 * s, '#EAF6FF', 140 * dark); blob(cx, cy, .3 * s, '#ffffff', 255 * dark); }
  }
  for (const side of [-1, 1]) paint(ellPts(ex * 1.55 * side, ey + .75 * s, .42 * s, .24 * s, 10), { wash: PAL.pink, washOp: 105, ink: null });          // a blush under the sheet
}
function ghost(x, y, s, o = {}) {
  const kind = o.kind || 'mia', g = GHOST[kind], sw = clamp(s / 14, .5, 2.2), lift = (o.lift ?? 0), walk = o.walk, dark = o.dark ?? 0, sk = o.sheetK ?? 1;
  const H_ = g.H * s * sk, Wd = g.Wd * s * (sk > 1 ? 1 + (sk - 1) * .5 : 1), hemY = -1.5 * s, moving = walk != null;
  if (!o.noShadow) paint(ellPts(x, y + s * .1, Wd * .55, s * .4, 14), { wash: PAL.ink, washOp: 55 * (1 - lift * .1), ink: null });
  push(); translate(x, y + (o.dy || 0) * s - lift * s); rotate(o.rot || 0); scale((o.flip ? -1 : 1) * (1 + (o.sq || 0) * .5), 1 - (o.sq || 0));
  const sheetCol = o.sheet || PAL.sheet;
  if (kind === 'dog') {                                                    // a sheet-covered dog in side view: ear holes, a wagging tail, eye holes, paws
    const trot = walk ?? 0, wag = o.wag ?? 0, top = -g.H * s, wg = Math.sin(T * 14) * wag;
    for (const [lx, i] of [[-1.7, 0], [-.9, 1], [1.0, 1], [1.8, 0]]) { const u = trot + (i ? .5 : 0), sx = moving ? Math.sin(u * TAU) * .5 : 0; paint(limbPts((lx + sx) * s, -1.3 * s, (lx + sx * .8) * s, -.25 * s, .6 * s), { wash: '#8d5a31', ink: PAL.ink, sw: sw * .5 }); paint(ellPts((lx + sx + .12) * s, -.2 * s, .52 * s, .27 * s, 8), { wash: PAL.brown, ink: PAL.ink, sw: sw * .6 }); }
    inkLine([[-2.3 * s, -1.9 * s], [(-3.1 + wg * .25) * s, -2.9 * s], [(-3.4 + wg * .8) * s, -3.9 * s + Math.abs(wg) * .1 * s]], sw * 1.9, PAL.brown, 'ink', .6);       // tail out the back
    const pts = []; for (let i = 0; i <= 16; i++) { const a = Math.PI + i / 16 * Math.PI; pts.push([Math.cos(a) * Wd * .5, -1.4 * s + Math.sin(a) * (g.H - 1.2) * s]); }
    for (let i = 8; i >= 0; i--) { const u = i / 8; pts.push([lerp(-Wd * .5, Wd * .5, 1 - u) * -1 * -1, -1.2 * s + Math.sin(u * 5 * Math.PI + T * 4 + (o.ph || 0)) * .22 * s + (i % 2 ? .35 * s : 0)]); }
    paint(pts, { wash: sheetCol, fill: '#c9d6f2', fillOp: 70, tex: .6, border: .5, ink: PAL.ink, sw: sw * .9 });
    for (const ex of [-.35, .55]) { paint(ellPts(ex * s, top + .75 * s, .46 * s, .26 * s, 10), { wash: PAL.ink, ink: null }); push(); translate(ex * s, top + .7 * s); rotate((ex < 0 ? -.7 : .55) + Math.sin(T * 5 + ex) * .12); paint(ellPts(0, -.55 * s, .5 * s, 1.05 * s, 10), { wash: '#8d5a31', fill: '#a8703f', fillOp: 70, tex: .4, ink: PAL.ink, sw: sw * .7 }); pop(); }   // floppy ears through two holes
    const sn = Wd * .5; paint(ellPts(sn + .1 * s, -2.0 * s, .85 * s, .62 * s, 12), { wash: '#d9a86f', ink: PAL.ink, sw: sw * .7 }); paint(ellPts(sn + .75 * s, -2.1 * s, .24 * s, .2 * s, 8), { wash: PAL.ink, ink: null });   // a snout poking out the front
    push(); translate(1.0 * s, 0); ghostEyes(null, .55 * s, top + 1.25 * s, s * .8, sw, o.eyes || 'dot', dark, o.look || [0, 0]); pop();
    pop(); if (o.emote) emote(o.emote, x + 1.5 * s, y - lift * s - H_ - 1.0 * s, s, o.emoteK ?? 1); return;
  }
  for (const i of [0, 1]) {                                                                               // the wearer's cue under the hem
    const side = i ? 1 : -1, u = (walk ?? 0) + (i ? .5 : 0), stride = o.stride ?? .8, fx = side * .9 * s + (moving ? Math.sin(u * TAU) * stride * s : 0), lf = moving ? Math.max(0, -Math.cos(u * TAU)) * (o.stepH ?? .5) * s : 0, fy = -lf, lower = -1.5 * s;
    if (kind === 'mia') { paint(limbPts(fx, lower, fx, fy - .4 * s, .85 * s), { wash: '#FFF5F8', ink: PAL.ink, sw: sw * .5 }); paint(ellPts(fx + .15 * s, fy - .1 * s, .95 * s, .5 * s, 12), { wash: PAL.pink, ink: PAL.ink, sw: sw * .7 }); paint(rectPts(fx - .75 * s, fy + .1 * s, 1.85 * s, .2 * s, 0), { wash: '#fff', ink: null }); }
    else if (kind === 'leo') { for (let k = 0; k < 5; k++) { const a0 = k / 5, a1 = (k + 1.05) / 5; paint(limbPts(fx, lerp(lower, fy - .4 * s, a0), fx, lerp(lower, fy - .4 * s, a1), .95 * s), { wash: k % 2 ? '#fff' : PAL.green, ink: null }); } inkLine([[fx - .48 * s, lower], [fx - .48 * s, fy - .4 * s]], sw * .5, PAL.ink, 'inkfine', 0); inkLine([[fx + .48 * s, lower], [fx + .48 * s, fy - .4 * s]], sw * .5, PAL.ink, 'inkfine', 0); paint(ellPts(fx + .1 * s, fy - .1 * s, 1.0 * s, .5 * s, 12), { wash: PAL.ink, ink: PAL.ink, sw: sw * .5 }); }
    else { paint(ellPts(fx + .1 * s, fy - .1 * s, .95 * s, .5 * s, 12), { wash: PAL.ink, ink: PAL.ink, sw: sw * .5 }); paint(limbPts(fx, lower, fx, fy - .4 * s, .8 * s), { wash: SK, ink: PAL.ink, sw: sw * .5 }); if (!i) { const jig = (moving ? Math.sin(T * 30) : Math.sin(T * 3) * .15) * .12; blob(fx - .75 * s + jig * s, fy - .35 * s, .4 * s, PAL.gold); inkLine([[fx - .75 * s, fy - .2 * s], [fx - .75 * s + jig * s, fy - .55 * s]], sw * .4, PAL.ink, 'inkfine', 0); } }
  }
  const sway = (o.sway ?? Math.sin(T * 2.0 + (o.ph || 0)) * .12) * s, domeR = Wd * .5, top = -H_;
  const pts = [], topY = top + domeR * .9;
  for (let i = 0; i <= 18; i++) { const a = Math.PI + i / 18 * Math.PI; pts.push([Math.cos(a) * domeR + sway, topY + Math.sin(a) * domeR * .95]); }
  const nHem = 10;
  for (let i = 0; i <= nHem; i++) { const u = i / nHem, hx_ = lerp(domeR * 1.18, -domeR * 1.18, u), w = Math.sin((u * 5.0) * Math.PI + T * 4.2 + (o.ph || 0)) * .28 * s * (1 + (moving ? .6 : 0)); pts.push([hx_ + sway * .3, hemY + w + (i % 2 ? .5 * s : -.1 * s)]); }
  // a sheet-covered arm reaching out (an arm-shaped bump)
  if (o.reach) { const [ra, rl] = o.reach, ax0 = domeR * .8, ay0 = topY + domeR * 1.4, ax1 = ax0 + Math.cos(ra) * rl * s, ay1 = ay0 - Math.sin(ra) * rl * s;
    paint(limbPts(ax0 + sway * .5, ay0, ax1, ay1, .95 * s, .8 * s), { wash: sheetCol, ink: PAL.ink, sw: sw * .8 }); paint(ellPts(ax1, ay1, .5 * s, .46 * s, 10), { wash: sheetCol, ink: PAL.ink, sw: sw * .8 }); }
  paint(pts, { wash: sheetCol, fill: '#c9d6f2', fillOp: 70, tex: .6, border: .5, ink: PAL.ink, sw: sw * .9, curv: .35 });
  if (o.armsUp !== undefined) for (const sd of [-1, 1]) { const ax0 = sd * domeR * .85 + sway * .6, ay0 = topY + domeR * 1.25, a = o.armsUp + sd * .15 + Math.sin(T * 11 + sd) * (o.wave ?? .35), ax1 = ax0 + sd * Math.cos(a) * 2.1 * s, ay1 = ay0 - Math.sin(a) * 2.1 * s;
    paint(limbPts(ax0, ay0, ax1, ay1, 1.0 * s, .85 * s), { wash: sheetCol, ink: PAL.ink, sw: sw * .8 }); paint(ellPts(ax1, ay1, .52 * s, .48 * s, 10), { wash: sheetCol, ink: PAL.ink, sw: sw * .8 }); }
  inkLine([[-domeR * .5 + sway * .6, topY + domeR * .5], [-domeR * .6, hemY - .6 * s]], sw * .4, '#b7c0da', 'inkfine', .6);
  inkLine([[domeR * .45 + sway * .6, topY + domeR * .6], [domeR * .55, hemY - .5 * s]], sw * .4, '#b7c0da', 'inkfine', .6);
  if (o.stain) { const sx = domeR * .12 + sway * .5, sy = topY + domeR * 1.9, sp = [], k = g.H > 10 ? 1.0 : .8;
    for (let i = 0; i < 16; i++) { const a = i / 16 * TAU, r = (i % 2 ? .55 : 1.0) * s * (.9 + .35 * Math.sin(i * 2.3)) * k; sp.push([sx + Math.cos(a) * r * 1.05, sy + Math.sin(a) * r * .85]); }
    paint(sp, { wash: PAL.ketchup, ink: '#9c2f25', sw: sw * .5, curv: .4 });
    for (const [dx, dy, r] of [[1.4, .6, .22], [-1.3, 1.2, .18], [.2, 1.5, .2]]) blob(sx + dx * s, sy + dy * s, r * s, PAL.ketchup);
    for (let i = 0; i < (o.wrap || 0); i++) { const wx = sx + [-.5, .7, .1][i] * s, wy = sy + [-.2, .2, .55][i] * s, col = [PAL.pink, PAL.candle, PAL.mint][i]; paint([[wx - .5 * s, wy], [wx - .2 * s, wy - .45 * s], [wx + .3 * s, wy - .3 * s], [wx + .55 * s, wy + .1 * s], [wx + .1 * s, wy + .4 * s]], { wash: col, ink: PAL.ink, sw: sw * .45 }); } }
  push(); translate(sway, 0); ghostEyes(null, domeR * .38, topY + domeR * .45, s, sw, (blinkAt(o.bph || .3) && !o.eyes) ? 'closed' : (o.eyes || 'dot'), dark, o.look || [0, 0]); pop();
  pop();
  if (o.emote) emote(o.emote, x + 1.5 * s, y - lift * s - H_ - 1.0 * s, s, o.emoteK ?? 1);
}
