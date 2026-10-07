/* set.js - the Halloween living room, drawn in code. The static furniture is painted once per variant with p5.brush and cached
 * as an image; everything that moves (sky, door, glows, bats, spider, cauldron level, bulbs) is painted live on top. */
const BG = {};                                         // variant -> p5.Graphics
const LAYOUT = {
  window: [830, 110, 1190, 490], door: [1530, 140, 1800, 712], kitchen: [40, 250, 250, 712], shelf: [1215, 130, 1455, 712],
  sofa: [300, 470, 800, 702], table: [350, 740, 740, 770], floorY: 702, basket: [255, 940],
  cauldron: [560, 724], moon: [1010, 240],
};

function drawSetStatic(v) {
  const morning = v === 'morning', J = 3, sw = 1.0;
  const wallC = morning ? '#E9C8A0' : '#5B4C8C', wallD = morning ? '#DDB68A' : '#4a3d78', floorC = morning ? '#C98F5E' : '#7a5340', floorD = morning ? '#B97C4D' : '#694634';
  // wall + wainscot + floor
  paint(rectPts(-40, -40, W + 80, 760, 0), { wash: wallC, fill: wallD, fillOp: 50, tex: .5, ink: null });
  for (let i = 0; i < 24; i++) paint(rectPts(i * 86 - 20, -10, 14, 640, 0), { wash: wallD, washOp: 70, ink: null });                       // wallpaper stripes
  for (let r = 0; r < 4; r++) for (let i = 0; i < 24; i++) { const x = i * 86 + 22 + (r % 2) * 43, y = 70 + r * 150; paint(starPts(x, y, 9, .45, 4), { wash: morning ? '#F2D9B5' : '#7a6bb0', washOp: 190, ink: null }); }
  paint(rectPts(-40, 560, W + 80, 150, 0), { wash: wallD, ink: PAL.ink, sw: sw * .8 });
  for (let i = 0; i < 14; i++) paint(rrPts(i * 150 + 20, 585, 120, 100, 10), { ink: PAL.ink, sw: sw * .5, wash: mixCol(wallD, '#000000', .08) });
  paint(rectPts(-40, 690, W + 80, 24, 0), { wash: morning ? '#F6EAD2' : '#d9cdb0', ink: PAL.ink, sw: sw * .9 });
  paint(rectPts(-40, 712, W + 80, 400, 0), { wash: floorC, fill: floorD, fillOp: 60, tex: .5, ink: null });
  for (let i = 0; i < 9; i++) inkLine([[-20, 730 + i * 46], [W + 20, 730 + i * 46 + (i % 2 ? 4 : -3)]], sw * .5, floorD, 'inkfine', .3);
  for (let i = 0; i < 24; i++) inkLine([[i * 90 + (i % 3) * 20, 712 + (i % 5) * 72], [i * 90 + (i % 3) * 20 + 2, 712 + (i % 5) * 72 + 72]], sw * .4, floorD, 'inkfine', 0);

  // striped rug (a perspective trapezoid)
  const rugTop = 770, rugBot = 1040, rc = [PAL.pumpkin, '#F6E3B8', PAL.violet, '#F6E3B8'];
  const rug = [[560, rugTop], [1560, rugTop], [1760, rugBot], [380, rugBot]];
  paint(rug, { wash: '#F6E3B8', ink: PAL.ink, sw: sw * .9 });
  for (let k = 0; k < 8; k++) { const a = k / 8, b = (k + 1) / 8, xl = (u, y) => lerp(560, 380, y) + 0, p = (u, yy) => [lerp(lerp(560, 380, yy), lerp(1560, 1760, yy), u), lerp(rugTop, rugBot, yy)];
    if (k % 2 === 0) paint([p(a, 0), p(b, 0), p(b, 1), p(a, 1)], { wash: rc[(k / 2) % 4], washOp: morning ? 200 : 170, ink: null }); }
  paint(rug, { ink: PAL.ink, sw: sw * .9 });

  // kitchen doorway, left: a warm room beyond
  const [kx0, ky0, kx1, ky1] = LAYOUT.kitchen;
  paint(rectPts(kx0 - 14, ky0 - 14, kx1 - kx0 + 28, ky1 - ky0 + 14, 0), { wash: '#6b3b2a', ink: PAL.ink, sw: sw });
  paint(rectPts(kx0, ky0, kx1 - kx0, ky1 - ky0, 0), { wash: morning ? '#FFEFC4' : '#F7CF7E', fill: '#F2A04A', fillOp: 70, tex: .4, ink: PAL.ink, sw: sw * .8 });
  paint(rectPts(kx0 + 20, ky0 + 60, 70, 120, 0), { wash: '#e9edf2', ink: PAL.ink, sw: sw * .7 }); paint(ellPts(kx0 + 55, ky0 + 40, 36, 40, 12), { wash: '#d9a86f', ink: PAL.ink, sw: sw * .6 });   // fridge-ish + lamp
  for (let i = 0; i < 4; i++) inkLine([[kx0, ky0 + 250 + i * 40], [kx1, ky0 + 250 + i * 40]], sw * .4, '#c98a3a', 'inkfine', 0);

  // window with curtains
  const [wx0, wy0, wx1, wy1] = LAYOUT.window;
  paint(rectPts(wx0 - 22, wy0 - 22, wx1 - wx0 + 44, wy1 - wy0 + 44, 0), { wash: '#EFE3C6', ink: PAL.ink, sw: sw });
  paint(rectPts(wx0, wy0, wx1 - wx0, wy1 - wy0, 0), { wash: morning ? '#9ED2F2' : '#1E2A5A', fill: morning ? '#dff1ff' : '#3a4a8a', fillOp: 80, tex: .4, ink: PAL.ink, sw: sw * .8 });
  inkLine([[(wx0 + wx1) / 2, wy0], [(wx0 + wx1) / 2, wy1]], sw * 1.1, '#EFE3C6', 'ink', 0); inkLine([[wx0, (wy0 + wy1) / 2], [wx1, (wy0 + wy1) / 2]], sw * 1.1, '#EFE3C6', 'ink', 0);
  paint(rectPts(wx0 - 40, wy1 + 14, wx1 - wx0 + 80, 26, 0), { wash: '#EFE3C6', ink: PAL.ink, sw: sw * .9 });
  for (const side of [-1, 1]) {
    const cx = side < 0 ? wx0 - 60 : wx1 + 60;
    paint([[cx - 50, wy0 - 40], [cx + 50, wy0 - 40], [cx + 70 * side * -1 + side * 20, wy0 + 200], [cx + 40 * side * -1, wy1 + 20], [cx - 70 * side * -1 - side * 20, wy1 + 20], [cx - 50, wy0 + 200]], { wash: morning ? '#C98FD6' : '#8E5BB5', fill: '#d8b0f0', fillOp: 50, tex: .5, ink: PAL.ink, sw: sw * .9, curv: .3 });
    inkLine([[cx - 30, wy0], [cx - 38, wy1]], sw * .5, '#6a3f8a', 'inkfine', 0); inkLine([[cx + 30, wy0], [cx + 38, wy1]], sw * .5, '#6a3f8a', 'inkfine', 0);
  }
  paint(rectPts(wx0 - 120, wy0 - 52, wx1 - wx0 + 240, 18, 0), { wash: '#6b3b2a', ink: PAL.ink, sw: sw * .8 });
  // cobweb in the window's top-left corner
  const cwx = wx0 - 22, cwy = wy0 - 22;
  for (let a = 0; a < 5; a++) inkLine([[cwx, cwy], [cwx + Math.cos(a * .4 + .1) * 130, cwy + Math.sin(a * .4 + .1) * 130]], sw * .4, '#e8e4f0', 'inkfine', 0);
  for (let r = 1; r <= 3; r++) { const pts = []; for (let a = 0; a < 5; a++) pts.push([cwx + Math.cos(a * .4 + .1) * r * 42 * (a % 2 ? .9 : 1), cwy + Math.sin(a * .4 + .1) * r * 42 * (a % 2 ? .9 : 1)]); inkLine(pts, sw * .35, '#e8e4f0', 'inkfine', .3); }

  // door frame + porch (the door leaf is dynamic)
  const [dx0, dy0, dx1, dy1] = LAYOUT.door;
  paint(rectPts(dx0 - 18, dy0 - 18, dx1 - dx0 + 36, dy1 - dy0 + 18, 0), { wash: '#6b3b2a', ink: PAL.ink, sw: sw });
  paint(rectPts(dx0, dy0, dx1 - dx0, dy1 - dy0, 0), { wash: morning ? '#BFE3F7' : '#10163a', fill: morning ? '#fff' : '#26306a', fillOp: 70, tex: .3, ink: PAL.ink, sw: sw * .7 });
  paint(rectPts(dx0, dy1 - 120, dx1 - dx0, 120, 0), { wash: morning ? '#9fcf8a' : '#1a2448', ink: null });
  if (!morning) for (let i = 0; i < 9; i++) paint(starPts(dx0 + 30 + hash(i * 3) * 220, dy0 + 20 + hash(i * 5) * 200, 7, .4, 4), { wash: '#FFF4C0', ink: null });

  // bookshelf
  const [bx0, by0, bx1, by1] = LAYOUT.shelf;
  paint(rectPts(bx0, by0, bx1 - bx0, by1 - by0, 0), { wash: '#7b4a30', fill: '#a8683c', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw });
  const bcols = [PAL.pumpkin, PAL.violet, PAL.mint, '#E2476E', PAL.candle, PAL.sky, PAL.green];
  for (let r = 0; r < 4; r++) {
    const ry = by0 + 20 + r * 142; paint(rectPts(bx0 + 10, ry, bx1 - bx0 - 20, 118, 0), { wash: '#3d2a24', ink: PAL.ink, sw: sw * .5 });
    let x = bx0 + 18; for (let i = 0; i < 9 && x < bx1 - 30; i++) { const w = 18 + hash(r * 10 + i) * 14, h = 72 + hash(r * 7 + i * 3) * 38; paint(rectPts(x, ry + 118 - h, w, h, 0), { wash: bcols[(r * 3 + i) % 7], ink: PAL.ink, sw: sw * .5 }); x += w + 3; }
    paint(rectPts(bx0 + 4, ry + 118, bx1 - bx0 - 8, 12, 0), { wash: '#a8683c', ink: PAL.ink, sw: sw * .5 });
  }
  paint(ellPts(bx0 + 70, by0 + 20 + 142 - 10 + 118 - 150, 28, 24, 12), { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .7 });                       // shelf pumpkin
  paint(rrPts(bx0 + 150, by0 + 20 + 142 + 118 - 62, 26, 50, 10), { wash: PAL.sheet, ink: PAL.ink, sw: sw * .6 });                              // tiny ghost figurine

  // sofa
  const [sx0, sy0, sx1, sy1] = LAYOUT.sofa;
  paint(rrPts(sx0 + 20, sy0, sx1 - sx0 - 40, 150, 50), { wash: '#C8553D', fill: '#f08a6a', fillOp: 55, tex: .6, ink: PAL.ink, sw: sw });
  paint(rrPts(sx0, sy0 + 100, sx1 - sx0, 100, 30), { wash: '#D9674B', fill: '#f5a085', fillOp: 55, tex: .6, ink: PAL.ink, sw: sw });
  for (const side of [-1, 1]) paint(rrPts(side < 0 ? sx0 - 26 : sx1 - 34, sy0 + 70, 60, 150, 26), { wash: '#B8472F', ink: PAL.ink, sw: sw });
  inkLine([[sx0 + 250, sy0 + 108], [sx0 + 250, sy0 + 190]], sw * .6, '#8d3320', 'inkfine', 0);
  paint(rrPts(sx0 + 40, sy0 + 20, 120, 110, 30), { wash: PAL.pumpkin, ink: PAL.ink, sw: sw * .8 }); paint([[sx0 + 78, sy0 + 62], [sx0 + 98, sy0 + 62], [sx0 + 88, sy0 + 44]], { wash: PAL.ink, ink: null }); paint([[sx0 + 108, sy0 + 62], [sx0 + 128, sy0 + 62], [sx0 + 118, sy0 + 44]], { wash: PAL.ink, ink: null });
  paint([[sx0 + 70, sy0 + 92], [sx0 + 135, sy0 + 92], [sx0 + 125, sy0 + 108], [sx0 + 80, sy0 + 108]], { wash: PAL.ink, ink: null });
  paint(rrPts(sx1 - 180, sy0 + 14, 120, 120, 36), { wash: PAL.sheet, ink: PAL.ink, sw: sw * .8 }); blob(sx1 - 138, sy0 + 62, 9, PAL.ink); blob(sx1 - 104, sy0 + 62, 9, PAL.ink);
  if (morning) paint([[sx0 + 280, sy0 + 100], [sx0 + 380, sy0 + 80], [sx0 + 410, sy0 + 130], [sx0 + 330, sy0 + 170], [sx0 + 270, sy0 + 150]], { wash: PAL.sheet, ink: PAL.ink, sw: sw * .8, curv: .4 });   // crumpled sheet

  // coffee table, jack-o-lanterns, candle, dog basket
  const [tx0, ty0, tx1, ty1] = LAYOUT.table;
  paint(rectPts(tx0, ty0, tx1 - tx0, 28, 0), { wash: '#8a5a3a', fill: '#b57c52', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw });
  for (const x of [tx0 + 24, tx1 - 40]) paint(rectPts(x, ty0 + 26, 16, 78, 0), { wash: '#6b3f27', ink: PAL.ink, sw: sw * .8 });
  jack(660, 716, 74, 66, morning); jack(716, 730, 50, 44, morning);
  // candle on the sill-side table
  paint(rectPts(1330, 620, 24, 70, 0), { wash: '#FFF0D0', ink: PAL.ink, sw: sw * .7 });
  const [bkx, bky] = LAYOUT.basket;
  paint(ellPts(bkx, bky + 10, 150, 52, 22), { wash: '#B78250', fill: '#d9a86f', fillOp: 60, tex: .5, ink: PAL.ink, sw: sw });
  paint(ellPts(bkx, bky - 4, 118, 34, 20), { wash: '#E8685A', ink: PAL.ink, sw: sw * .7 });
  if (v === 'night') {                                                                 // garland wire (bulbs are live)
    const wire = []; for (let i = 0; i <= 28; i++) { const x = i / 28 * (W + 40) - 20; wire.push([x, 40 + 30 * Math.sin(i / 28 * Math.PI * 4 + .4) + (i % 2) * 2]); }
    inkLine(wire, sw * .7, '#2B2A33', 'inkfine', .5);
  }
  // wall clock
  paint(ellPts(610, 190, 54, 54, 22), { wash: '#FFF5E2', ink: PAL.ink, sw: sw * 1.0 });
  for (let i = 0; i < 12; i++) { const a = i / 12 * TAU; inkLine([[610 + Math.cos(a) * 42, 190 + Math.sin(a) * 42], [610 + Math.cos(a) * 49, 190 + Math.sin(a) * 49]], sw * .6, PAL.ink, 'inkfine', 0); }
}
function jack(x, y, w, h, morning) {                                                   // static carved pumpkin (glow is live)
  paint(ellPts(x, y - h / 2, w / 2, h / 2, 18), { wash: PAL.pumpkin, fill: '#F7B15A', fillOp: 70, tex: .5, ink: PAL.ink, sw: .9 });
  inkLine([[x, y - h + 4], [x - w * .1, y - h / 2], [x, y - 2]], .5, '#c4661d', 'inkfine', .4);
  paint(rectPts(x - 4, y - h - 8, 8, 14, 0), { wash: '#4c7a3a', ink: PAL.ink, sw: .6 });
  const lit = morning ? '#E9A23A' : '#FFD76A';
  paint([[x - w * .28, y - h * .62], [x - w * .08, y - h * .62], [x - w * .18, y - h * .8]], { wash: lit, ink: PAL.ink, sw: .5 });
  paint([[x + w * .08, y - h * .62], [x + w * .28, y - h * .62], [x + w * .18, y - h * .8]], { wash: lit, ink: PAL.ink, sw: .5 });
  paint([[x - w * .3, y - h * .42], [x + w * .3, y - h * .42], [x + w * .2, y - h * .2], [x + w * .08, y - h * .3], [x - w * .08, y - h * .2], [x - w * .2, y - h * .3]], { wash: lit, ink: PAL.ink, sw: .5 });
}

/* ───── live parts ───── */
function flick(t, seed) { return .78 + .22 * Math.sin(t * 9 + seed * 3.1) * Math.sin(t * 5.3 + seed) + .1 * Math.sin(t * 23 + seed); }
function setSky(t, morning) {
  const [wx0, wy0, wx1, wy1] = LAYOUT.window;
  if (morning) { blob(1060, 190, 52, '#FFE9A0');
    for (let i = 0; i < 2; i++) { const u = frac((t * .01 + i * .5)), cx = lerp(wx0 - 40, wx1 + 40, u), a = Math.sin(u * Math.PI); if (a > .15) { for (const [dx, dy, r] of [[0, 0, 36], [34, 6, 30], [-32, 10, 26]]) paint(ellPts(cx + dx, 150 + i * 120 + dy, r, r * .7, 14), { wash: '#ffffff', washOp: 230 * a, ink: null }); } }
    return; }
  const [mx, my] = LAYOUT.moon;
  blob(mx, my, 62, '#FFF3C4'); blob(mx - 18, my - 12, 11, '#EBD9A0'); blob(mx + 22, my + 16, 8, '#EBD9A0'); blob(mx + 10, my - 26, 6, '#EBD9A0');
  for (let i = 0; i < 7; i++) { const x = wx0 + 20 + hash(i * 2.1) * (wx1 - wx0 - 40), y = wy0 + 20 + hash(i * 3.7) * (wy1 - wy0 - 40), tw = .5 + .5 * Math.sin(t * 2 + i * 1.9); if (Math.hypot(x - mx, y - my) > 90) paint(starPts(x, y, 6 + 4 * tw, .4, 4), { wash: '#FFF4C0', washOp: 120 + 120 * tw, ink: null }); }
  for (let i = 0; i < 3; i++) {                                                           // clouds drift over the moon
    const u = frac(t * (.018 + i * .006) + i * .37), cx = lerp(wx0 - 60, wx1 + 60, u), a = Math.sin(u * Math.PI), cy = 170 + i * 105;
    if (a > .08) for (const [dx, dy, r] of [[0, 0, 40], [38, 8, 32], [-36, 12, 28], [8, 14, 34]]) paint(ellPts(cx + dx, cy + dy, r * 1.2, r * .7, 14), { wash: '#7f90c8', washOp: 160 * a, ink: null });
  }
  if (t > 12.2 && t < 17.4) { const u = (t - 12.2) / 5.2, wx = lerp(wx0 - 30, wx1 + 30, u), wy = 340 - 120 * Math.sin(u * Math.PI) + 6 * Math.sin(t * 7), a = Math.sin(u * Math.PI) > .1 ? 1 : 0;   // the witch crosses once
    if (wx > wx0 + 10 && wx < wx1 - 10) { push(); translate(wx, wy); rotate(-.12); const k = 1.0; const dk = '#150f2a';
      paint([[-40, 4], [40, 0], [40, 8], [-40, 10]], { wash: dk, ink: null }); paint([[-40, 4], [-62, -2], [-62, 16], [-40, 10]], { wash: dk, ink: null });
      paint(ellPts(6, -10, 8, 13, 10), { wash: dk, ink: null }); blob(8, -28, 7, dk); paint([[-2, -31], [8, -58], [19, -31]], { wash: dk, ink: null }); paint(rectPts(-5, -32, 27, 4, 0), { wash: dk, ink: null }); pop(); } }
}
function setDoor(t, openT) {                                                              // front door leaf; opens at openT with a bounce
  const [dx0, dy0, dx1, dy1] = LAYOUT.door, u = clamp((t - openT) / .32), o = openT == null ? 0 : (u < 1 ? easeOut(u) : 1) + (u >= 1 ? Math.sin((t - openT - .32) * 16) * Math.exp(-(t - openT - .32) * 7) * .04 : 0);
  const w = (dx1 - dx0) * (1 - clamp(o) * .86);
  paint(rectPts(dx1 - w, dy0, w, dy1 - dy0, 0), { wash: '#B5472E', fill: '#d9694a', fillOp: 60, tex: .5, ink: PAL.ink, sw: 1.0 });
  if (w > 60) { paint(rrPts(dx1 - w + 24, dy0 + 40, w - 48, 150, 18), { ink: PAL.ink, sw: .6, wash: '#9c3d27' }); paint(rrPts(dx1 - w + 24, dy0 + 230, w - 48, 200, 18), { ink: PAL.ink, sw: .6, wash: '#9c3d27' }); blob(dx1 - w + 22, dy0 + 330, 12, PAL.gold);
    paint(ellPts(dx1 - w / 2, dy0 + 90, 30, 28, 14), { wash: PAL.pumpkin, ink: PAL.ink, sw: .6 }); }
}
const BULBS = 12;
const bulbPos = i => [(i + .5) / BULBS * (W + 40) - 20, 40 + 30 * Math.sin((i + .5) / BULBS * Math.PI * 4 + .4) + 20];
const BULB_COL = ['#FFD76A', '#FF9CCB', '#8FD3B6', '#FFB064'];
function setLights(t, k = 1, morning = false) {                                          // flames and drawn bulbs; the soft glow itself is composited in 2D (see lightOverlay)
  if (morning) return;
  for (const [x, y] of [[1342, 612]]) { const f = flick(t, x); paint([[x - 7, y + 4], [x + 7, y + 4], [x + 2 * Math.sin(t * 8), y - 22 * f]], { wash: '#FFC44A', ink: null }); }
  for (let i = 0; i < BULBS; i++) { const [x, y] = bulbPos(i), tw = .5 + .5 * Math.sin(t * (1.6 + (i % 5) * .37) + i * 1.9), col = mixCol('#8a7a5a', BULB_COL[i % 4], .35 + .65 * tw * k);
    paint(rectPts(x - 3, y - 13, 6, 6, 0), { wash: '#3a3340', ink: null }); paint([[x - 3, y - 7], [x + 3, y - 7], [x + 6, y + 2], [x, y + 9], [x - 6, y + 2]], { wash: col, ink: PAL.ink, sw: .45 }); paint(ellPts(x - 2, y - 1, 1.6, 2.4, 8), { wash: '#ffffff', washOp: 160, ink: null }); }
}
const BATS = [[2.0, 7.5, -120, 190, 2050, 120, 1.0], [7.0, 12.5, 2050, 330, -150, 230, .8], [14.0, 19.6, -100, 420, 2000, 260, 1.1], [23.0, 29.0, 2050, 150, -120, 380, .9], [30.0, 35.4, -120, 260, 2050, 90, 1.0]];
function bat(x, y, sz, ph, rot = 0) {
  const wing = .5 + .5 * Math.sin(ph * TAU); push(); translate(x, y); rotate(rot);
  for (const sd of [-1, 1]) { push(); scale(sd, 1); const wy = -22 * sz * (.3 + wing);
    paint([[4 * sz, -2 * sz], [18 * sz, wy], [34 * sz, wy * .55], [26 * sz, 4 * sz], [18 * sz, 0], [12 * sz, 9 * sz], [5 * sz, 2 * sz]], { wash: '#2a2040', ink: PAL.ink, sw: .5, curv: .25 }); pop(); }
  paint(ellPts(0, 2 * sz, 7 * sz, 10 * sz, 10), { wash: '#2a2040', ink: PAL.ink, sw: .5 }); paint([[-5 * sz, -7 * sz], [-4 * sz, -15 * sz], [0, -8 * sz], [4 * sz, -15 * sz], [5 * sz, -7 * sz]], { wash: '#2a2040', ink: null });
  blob(-2.5 * sz, -2 * sz, 1.6 * sz, '#FFD76A'); blob(2.5 * sz, -2 * sz, 1.6 * sz, '#FFD76A'); pop();
}
function setBats(t) {
  for (const [t0, t1, x0, y0, x1, y1, sz] of BATS) if (t >= t0 && t <= t1) { const u = (t - t0) / (t1 - t0); bat(lerp(x0, x1, u), lerp(y0, y1, u) + 40 * Math.sin(u * 14 + t0), sz * 1.1, t * 3 + t0, .12 * Math.sin(u * 14 + t0) * (x1 > x0 ? 1 : -1)); }
  // paper bats hanging on strings from the garland, swaying
  for (const [x, len, ph] of [[300, 120, 0], [1100, 90, 1.4], [1650, 140, 2.7]]) { const sw = Math.sin(t * 1.3 + ph) * .25, bx = x + Math.sin(sw) * len, by = 60 + Math.cos(sw) * len; inkLine([[x, 50], [bx, by - 10]], .5, PAL.ink, 'inkfine', 0); bat(bx, by + 8, .7, .5 + .02 * Math.sin(t * 2 + ph), sw); }
}
function setSpider(t) {
  const [cx, top] = [1075, 130], y = 215 + 24 * Math.sin(t * 2.1) + 8 * Math.sin(t * 5.3);
  inkLine([[cx, top], [cx, y - 16]], .5, '#e8e4f0', 'inkfine', 0);
  paint(ellPts(cx, y, 16, 14, 12), { wash: '#2a2036', ink: PAL.ink, sw: .5 }); paint(ellPts(cx, y - 14, 10, 9, 10), { wash: '#2a2036', ink: PAL.ink, sw: .5 });
  for (let i = 0; i < 4; i++) for (const sd of [-1, 1]) { const a = (i - 1.5) * .45, k = Math.sin(t * 6 + i) * 4; inkLine([[cx + sd * 8, y - 3 + i * 3], [cx + sd * 28, y - 14 + i * 8 + k], [cx + sd * 34, y + 10 + i * 8]], .5, '#2a2036', 'inkfine', .5); }
  blob(cx - 4, y - 15, 2.4, '#fff'); blob(cx + 4, y - 15, 2.4, '#fff'); inkLine([[cx - 4, y - 10], [cx, y - 7], [cx + 4, y - 10]], .4, '#fff', 'inkfine', .5);
}
function setClockHands(min) {                                                            // 21:xx on the wall clock
  const m = min / 60 * TAU - Math.PI / 2, h = (9 + min / 60) / 12 * TAU - Math.PI / 2;
  inkLine([[610, 190], [610 + Math.cos(m) * 38, 190 + Math.sin(m) * 38]], .9, PAL.ink, 'ink', 0); inkLine([[610, 190], [610 + Math.cos(h) * 26, 190 + Math.sin(h) * 26]], 1.2, PAL.ink, 'ink', 0);
}
function setCauldron(level, t, morning) {                                                // level 1 = heaped with candy, 0 = empty
  const [cx, cy] = LAYOUT.cauldron;
  paint(ellPts(cx, cy - 36, 92, 66, 22), { wash: '#2a2733', fill: '#555062', fillOp: 70, tex: .5, ink: PAL.ink, sw: 1.0 });
  paint(ellPts(cx, cy - 78, 86, 20, 20), { wash: '#12101a', ink: PAL.ink, sw: .9 });
  if (level > .02) {
    const n = Math.round(6 + 14 * level), cols = [PAL.pink, PAL.candle, PAL.mint, PAL.violet, PAL.pumpkin, '#E2476E'];
    for (let i = 0; i < n; i++) { const a = hash(i * 3.3) * TAU, r = Math.sqrt(hash(i * 7.1)) * 74 * (.4 + .6 * level), px = cx + Math.cos(a) * r, py = cy - 82 - level * 16 + Math.sin(a) * r * .22 - hash(i) * 12 * level;
      paint(ellPts(px, py, 11, 8, 8, 0, hash(i) * 3), { wash: cols[i % 6], ink: PAL.ink, sw: .4 }); }
  }
  paint([[cx - 92, cy - 52], [cx - 112, cy - 28], [cx - 92, cy - 20]], { ink: PAL.ink, sw: .8 });
  paint(rectPts(cx - 62, cy + 18, 22, 14, 0), { wash: '#1a1822', ink: null }); paint(rectPts(cx + 40, cy + 18, 22, 14, 0), { wash: '#1a1822', ink: null });
}
function setMorningExtras(t) {
  for (const [x, y, k, r] of [[820, 800, 0, .3], [980, 860, 1, -.5], [1140, 790, 2, .8], [760, 880, 1, 1.2], [1260, 880, 0, -.9], [900, 780, 2, .1], [1420, 820, 1, .6], [600, 960, 0, -.2]]) paint([[x - 12, y], [x - 4, y - 12 + r * 2], [x + 8, y - 8], [x + 14, y + 4], [x + 2, y + 10], [x - 10, y + 8]], { wash: [PAL.pink, PAL.candle, PAL.mint][k], ink: PAL.ink, sw: .5 });
  // sun shafts
  for (const [ax, bx, cx2, dx2, a] of [[840, 1180, 520, 1260, 40], [1540, 1790, 1180, 1850, 50]]) paint([[ax, 140], [bx, 140], [dx2, 1000], [cx2, 1000]], { wash: '#FFF0B0', washOp: a + 8 * Math.sin(t * .8), ink: null });
}
function dust(t, n = 40) { for (let i = 0; i < n; i++) { const x = (hash(i * 3.1) * W + t * (6 + hash(i) * 14)) % W, y = (hash(i * 7.7) * H - t * (4 + hash(i * 2) * 8) + H * 4) % H, tw = .5 + .5 * Math.sin(t * (1 + hash(i) * 2) + i); blob(x, y, 2 + hash(i * 2.2) * 4, '#FFF4D0', 140 * tw); } }

function sparkles(t, n = 10) { for (let i = 0; i < n; i++) { const x = hash(i * 3.1) * W, y = 100 + hash(i * 7.7) * 700, tw = Math.max(0, Math.sin(t * (1 + hash(i) * 1.5) + i * 2)); if (tw > .4) paint(starPts(x + Math.sin(t + i) * 12, y + Math.cos(t * .8 + i) * 8, 5 + tw * 6, .35, 4), { wash: '#FFE9A8', washOp: 200 * tw, ink: null }); } }
