/* core.js - constants, helpers, paint wrappers over p5.brush, paper/grain, camera, frame hooks.
 * Everything is a pure function of t: nothing survives between frames, so the renderer can seek anywhere. */
const W = 1920, H = 1080, TAU = Math.PI * 2, BOIL = 12;
const PAL = {
  paper: '#F4EBDD', ink: '#2B2A33', pumpkin: '#F28C38', candle: '#FFC861', violet: '#8E7CC3', night: '#2E3A63',
  mint: '#8FD3B6', ketchup: '#D9483B', pink: '#FF7EB6', cream: '#FFF5E2', skin: '#F6C9A5', wall: '#5A4B86',
  sky: '#8EC3E6', sheet: '#FBF8F0', gold: '#F5C542', brown: '#B07A4A', green: '#6bc28a', teal: '#3A9C98',
};
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, x) => a + (b - a) * x;
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const easeOut = x => 1 - Math.pow(1 - clamp(x), 3);
const easeIn = x => Math.pow(clamp(x), 3);
const backOut = x => { x = clamp(x); const s = 1.9; return 1 + (s + 1) * Math.pow(x - 1, 3) + s * Math.pow(x - 1, 2); };
const elasticOut = x => { x = clamp(x); return x === 0 || x === 1 ? x : Math.pow(2, -10 * x) * Math.sin((x * 10 - .75) * (TAU / 3)) + 1; };
const hash = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const jit = a => (random() * 2 - 1) * a;                      // seeded by the boil frame
const seg = (t, a, b) => clamp((t - a) / (b - a));
const wob = (t, f = 1, ph = 0) => Math.sin((t * f + ph) * TAU);
const frac = x => x - Math.floor(x);
function kf(t, keys, e = ease) {                               // keyframes [[t,v],...]; v number or array
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) if (t < keys[i][0]) {
    const [a, va] = keys[i - 1], [b, vb] = keys[i], k = e((t - a) / (b - a));
    return Array.isArray(va) ? va.map((v, j) => lerp(v, vb[j], k)) : lerp(va, vb, k);
  }
  return keys[keys.length - 1][1];
}
function mixCol(a, b, k) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16), c = i => Math.round(lerp((pa >> i) & 255, (pb >> i) & 255, clamp(k)));
  return '#' + ((1 << 24) + (c(16) << 16) + (c(8) << 8) + c(0)).toString(16).slice(1);
}
const shakeXY = (t, amt) => { const f = Math.floor(t * 24); return [(hash(f * 1.7) - .5) * 2 * amt, (hash(f * 2.3 + 9) - .5) * 2 * amt]; };

/* ───── camera: world point (cx,cy) lands at screen centre ───── */
let FCAM = { cx: W / 2, cy: H / 2, zoom: 1, rot: 0 };
function camBegin(c = FCAM) { push(); translate(c.sx ?? W / 2, H / 2); rotate(c.rot); scale(c.zoom); translate(-c.cx, -c.cy); }
function camEnd() { pop(); }
function toScreen(x, y, c = FCAM) {
  const co = Math.cos(c.rot), si = Math.sin(c.rot), dx = (x - c.cx) * c.zoom, dy = (y - c.cy) * c.zoom;
  return [(c.sx ?? W / 2) + dx * co - dy * si, H / 2 + dx * si + dy * co];
}

/* ───── geometry ───── */
function rectPts(x, y, w, h, j = 0) {
  return [[x + jit(j), y + jit(j)], [x + w / 2 + jit(j), y + jit(j) * .5], [x + w + jit(j), y + jit(j)], [x + w + jit(j) * .5, y + h / 2],
    [x + w + jit(j), y + h + jit(j)], [x + w / 2 + jit(j), y + h + jit(j) * .5], [x + jit(j), y + h + jit(j)], [x + jit(j) * .5, y + h / 2]];
}
function ellPts(cx, cy, rx, ry, n = 24, j = 0, rot = 0) {
  const p = []; for (let i = 0; i < n; i++) { const a = rot + i / n * TAU; p.push([cx + Math.cos(a) * rx + jit(j), cy + Math.sin(a) * ry + jit(j)]); } return p;
}
function rrPts(x, y, w, h, r, j = 0) {
  const p = [], corner = (cx, cy, a0) => { for (let i = 0; i <= 5; i++) { const a = a0 + i / 5 * Math.PI / 2; p.push([cx + Math.cos(a) * r + jit(j), cy + Math.sin(a) * r + jit(j)]); } };
  corner(x + w - r, y + r, -Math.PI / 2); corner(x + w - r, y + h - r, 0); corner(x + r, y + h - r, Math.PI / 2); corner(x + r, y + r, Math.PI); return p;
}
function starPts(cx, cy, r, inner = .4, n = 4, rot = -Math.PI / 2) {
  const p = []; for (let i = 0; i < n * 2; i++) { const a = rot + i * Math.PI / n, q = i % 2 ? r * inner : r; p.push([cx + Math.cos(a) * q, cy + Math.sin(a) * q]); } return p;
}
function heartPts(cx, cy, r) {
  const p = []; for (let i = 0; i < 24; i++) { const t = i / 24 * TAU; p.push([cx + r * 0.0625 * 16 * Math.pow(Math.sin(t), 3), cy - r * 0.0625 * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t))]); } return p;
}
function limbPts(x0, y0, x1, y1, w0, w1 = w0) {                // a tapered quad from (x0,y0) to (x1,y1)
  const dx = x1 - x0, dy = y1 - y0, d = Math.hypot(dx, dy) || 1, nx = -dy / d, ny = dx / d;
  return [[x0 + nx * w0 / 2, y0 + ny * w0 / 2], [x1 + nx * w1 / 2, y1 + ny * w1 / 2], [x1 - nx * w1 / 2, y1 - ny * w1 / 2], [x0 - nx * w0 / 2, y0 - ny * w0 / 2]];
}

/* ───── painting: one call = one shape (flat wash, optional watercolor fill, hatch, ink outline) ───── */
function paint(pts, o = {}) {
  if (o.wash || o.fill || o.hatch) {
    if (o.wash) brush.wash(o.wash, o.washOp ?? 255); else brush.noWash();
    if (o.fill) { brush.fill(o.fill, o.fillOp ?? 170); brush.fillBleed(o.bleed ?? .1); brush.fillTexture(o.tex ?? .4, o.border ?? .35); } else brush.noFill();
    if (o.hatch) { brush.hatch(o.hatch.d, o.hatch.a, o.hatch.o || { rand: .15 }); brush.hatchStyle(o.hatch.b || 'HB', o.hatch.c || PAL.ink, o.hatch.w || 1); } else brush.noHatch();
    brush.noStroke();
    if (o.curv) { brush.beginShape(o.curv); for (const p of pts) brush.vertex(p[0], p[1]); brush.endShape(true); } else brush.polygon(pts);
  }
  if (o.ink !== null) {
    brush.noWash(); brush.noFill(); brush.noHatch(); brush.set(o.br || 'ink', o.ink || PAL.ink, o.sw ?? 1);
    brush.beginShape(o.curv || 0); for (const p of pts) brush.vertex(p[0], p[1]); brush.endShape(true);
  }
}
function inkLine(pts, sw = 1, col = PAL.ink, br = 'ink', curv = .5) { brush.noFill(); brush.noWash(); brush.noHatch(); brush.set(br, col, sw); brush.spline(pts, curv); }
function blob(x, y, r, col, op = 255) { paint(ellPts(x, y, r, r, 10), { wash: col, washOp: op, ink: null }); }
function glowAt(x, y, r, col, a) { paint(ellPts(x, y, r, r, 14), { wash: col, washOp: a * .45, ink: null }); paint(ellPts(x, y, r * .5, r * .5, 12), { wash: col, washOp: a * .6, ink: null }); }
function dashedLine(x0, y0, x1, y1, sw, col, dash = 36, gap = .45) {
  const L = Math.hypot(x1 - x0, y1 - y0), n = Math.max(1, Math.round(L / dash));
  for (let i = 0; i < n; i++) { const u0 = i / n, u1 = (i + 1 - gap) / n; inkLine([[lerp(x0, x1, u0), lerp(y0, y1, u0)], [lerp(x0, x1, u1), lerp(y0, y1, u1)]], sw, col, 'inkfine', 0); }
}
function dashedBox(x0, y0, x1, y1, sw, col, grow = 1) {
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hw = (x1 - x0) / 2 * grow, hh = (y1 - y0) / 2 * grow;
  dashedLine(cx - hw, cy - hh, cx + hw, cy - hh, sw, col); dashedLine(cx + hw, cy - hh, cx + hw, cy + hh, sw, col);
  dashedLine(cx + hw, cy + hh, cx - hw, cy + hh, sw, col); dashedLine(cx - hw, cy + hh, cx - hw, cy - hh, sw, col);
}

/* ───── paper and grain ───── */
function lcg(seed) { let s = seed; return () => (s = (s * 16807) % 2147483647) / 2147483647; }
function makePaper() {
  const g = createGraphics(W, H); g.pixelDensity(1); const c = g.drawingContext, rnd = lcg(11);
  c.fillStyle = PAL.paper; c.fillRect(0, 0, W, H);
  for (let i = 0; i < 60; i++) { const x = rnd() * W, y = rnd() * H, r = 120 + rnd() * 380, gr = c.createRadialGradient(x, y, 0, x, y, r), a = .045 * rnd(); gr.addColorStop(0, `rgba(160,125,80,${a})`); gr.addColorStop(1, 'rgba(160,125,80,0)'); c.fillStyle = gr; c.fillRect(x - r, y - r, r * 2, r * 2); }
  c.lineWidth = 1;
  for (let i = 0; i < 1200; i++) { const x = rnd() * W, y = rnd() * H, l = 6 + rnd() * 26, a = rnd() * TAU; c.strokeStyle = `rgba(110,88,60,${.03 + rnd() * .05})`; c.beginPath(); c.moveTo(x, y); c.quadraticCurveTo(x + Math.cos(a + .6) * l * .5, y + Math.sin(a + .6) * l * .5, x + Math.cos(a) * l, y + Math.sin(a) * l); c.stroke(); }
  return g;
}
function makeGrain() {
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H; const c = cv.getContext('2d'), rnd = lcg(5), id = c.createImageData(W, H), d = id.data;
  for (let i = 0; i < d.length; i += 4) { const v = 255 - (rnd() < .55 ? rnd() * rnd() * 34 : 0); d[i] = v; d[i + 1] = v - 1; d[i + 2] = v - 3; d[i + 3] = 255; }
  c.putImageData(id, 0, 0);
  const g = c.createRadialGradient(W / 2, H / 2, H * .45, W / 2, H / 2, H * 1.05); g.addColorStop(0, 'rgba(255,255,255,0)'); g.addColorStop(1, 'rgba(120,95,70,.32)'); c.fillStyle = g; c.fillRect(0, 0, W, H);
  return cv;
}
function defineBrushes() {
  brush.add('ink', { type: 'default', weight: 5, scatter: .25, sharpness: .8, grain: 40, opacity: 235, spacing: .2, pressure: [1.15, .75], rotate: 'natural', noise: .15 });
  brush.add('inkfine', { type: 'default', weight: 2.6, scatter: .15, sharpness: .85, grain: 40, opacity: 230, spacing: .2, pressure: [1.1, .8], rotate: 'natural', noise: .1 });
  brush.add('dry', { type: 'default', weight: 14, scatter: 3, sharpness: .3, grain: 6, opacity: 90, spacing: .6, pressure: [1, .6], rotate: 'natural', noise: .4 });
}
