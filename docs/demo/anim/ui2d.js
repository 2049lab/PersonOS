/* ui2d.js - 2D overlay drawn on the compositor canvas, in screen space (after the p5.brush scene): paper name tags, rubber stamps,
 * speech bubbles, handwriting that writes itself on, polaroids, the bilingual subtitle strip. */
let ctx;                                              // the compositor's 2D context, set by main.js
const UI = {}; const PI2 = Math.PI * 2;
const sstep = (a, b, x) => ease((x - a) / (b - a));
function roundRect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
UI.icon = function (kind, x, y, s = 1, a = 1, mono = null, lineCol = null) {
  const F = (col) => mono || col;
  ctx.save(); ctx.globalAlpha *= a; ctx.translate(x, y); ctx.scale(s, s); ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.strokeStyle = lineCol || PAL.ink;
  if (kind === 'voice') {
    ctx.fillStyle = F(PAL.violet); for (let i = 0; i < 5; i++) { const h = [8, 18, 28, 16, 10][i]; ctx.fillRect(-18 + i * 8, -h / 2, 5, h); }
  } else if (kind === 'body') {
    ctx.fillStyle = F(PAL.mint); ctx.beginPath(); ctx.arc(0, -15, 7, 0, PI2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -8); ctx.lineTo(0, 10); ctx.moveTo(-10, -2); ctx.lineTo(10, -2); ctx.moveTo(0, 10); ctx.lineTo(-8, 22); ctx.moveTo(0, 10); ctx.lineTo(8, 22); ctx.stroke();
  } else if (kind === 'face') {
    ctx.fillStyle = F('#FFE3B8'); ctx.beginPath(); ctx.arc(0, 0, 15, 0, PI2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = F(PAL.ink); ctx.beginPath(); ctx.arc(-5, -3, 2, 0, PI2); ctx.arc(5, -3, 2, 0, PI2); ctx.fill();
    ctx.beginPath(); ctx.arc(0, 3, 6, 0.2, Math.PI - 0.2); ctx.stroke();
  } else if (kind === 'shoe') {
    ctx.fillStyle = F(PAL.pink); ctx.beginPath(); ctx.moveTo(-18, 8); ctx.lineTo(-18, -8); ctx.quadraticCurveTo(-6, -12, 0, -4); ctx.quadraticCurveTo(16, -2, 20, 6); ctx.lineTo(20, 10); ctx.lineTo(-18, 10); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = F('#fff'); ctx.fillRect(-18, 8, 38, 4);
  } else if (kind === 'sock') {
    ctx.save(); for (let i = 0; i < 5; i++) { ctx.fillStyle = F(i % 2 ? '#fff' : '#6bc28a'); ctx.fillRect(-9, -22 + i * 9, 18, 9); }
    ctx.restore(); ctx.strokeRect(-9, -22, 18, 45);
    ctx.fillStyle = F('#e9e1cf'); ctx.beginPath(); ctx.moveTo(-26, -20); ctx.lineTo(-18, -20); ctx.lineTo(-18, 24); ctx.lineTo(-26, 24); ctx.closePath(); ctx.fill(); ctx.stroke();
    for (let k = 0; k < 5; k++) { ctx.beginPath(); ctx.moveTo(-26, -14 + k * 9); ctx.lineTo(-21, -14 + k * 9); ctx.stroke(); }
  } else if (kind === 'bell') {
    ctx.fillStyle = F('#F5C542'); ctx.beginPath(); ctx.arc(0, 2, 14, Math.PI, 0); ctx.lineTo(16, 12); ctx.lineTo(-16, 12); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 15, 3.5, 0, PI2); ctx.fill(); ctx.stroke();
  } else if (kind === 'ears') {
    ctx.fillStyle = F('#9a5b34'); for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(sx * 12, 2, 7, 15, sx * 0.3, 0, PI2); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = F('#b9763f'); ctx.beginPath(); ctx.arc(0, 8, 9, 0, PI2); ctx.fill(); ctx.stroke();
  }
  ctx.restore();
};

UI.tag = function (x, y, text, o = {}) {              // a paper name tag hanging at (x,y) centre
  const a = o.a ?? 1; if (a <= 0.01) return;
  const size = o.size ?? 40, icons = o.icons ?? [], rot = o.rot ?? -0.04, sc = (o.s ?? 1);
  ctx.save(); ctx.globalAlpha *= a; ctx.translate(x, y); ctx.rotate(rot); ctx.scale(sc, sc);
  ctx.font = `600 ${size}px Caveat`; const tw = ctx.measureText(text).width;
  const iw = icons.length * 40, w = Math.max(tw + 36, iw + 24), h = size + 18 + (icons.length ? 36 : 0);
  ctx.shadowColor = 'rgba(40,25,10,.28)'; ctx.shadowBlur = 8; ctx.shadowOffsetY = 4;
  ctx.fillStyle = o.fill ?? '#FFFBF0'; roundRect(ctx, -w / 2, -h / 2, w, h, 12); ctx.fill(); ctx.shadowColor = 'transparent';
  ctx.strokeStyle = o.stroke ?? PAL.ink; ctx.lineWidth = 3; ctx.stroke();
  ctx.fillStyle = o.color ?? PAL.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 0, -h / 2 + size / 2 + 10);
  icons.forEach((ic, i) => UI.icon(ic.kind, -iw / 2 + 20 + i * 40, h / 2 - 22, 0.78, ic.a ?? 1));
  ctx.restore();
};

UI.stamp = function (x, y, text, t0, t, o = {}) {      // rubber stamp thunk + ink splat
  const d = t - t0; if (d < 0) return;
  const land = 0.14, u = clamp(d / land), s = d < land ? lerp(2.4, 1, easeIn(u)) : 1 + 0.12 * Math.exp(-(d - land) * 9) * Math.cos((d - land) * 38);
  const a = (d < land ? u : 1) * (o.hold ? 1 : 1 - sstep(land + 0.55, land + 1.0, d));
  ctx.save(); ctx.translate(x, y); ctx.rotate(o.rot ?? -0.1); ctx.scale(s, s); ctx.globalAlpha *= a;
  const size = o.size ?? 64; ctx.font = `700 ${size}px Caveat`; const tw = ctx.measureText(text).width, w = tw + 44, h = size + 26;
  const col = o.color ?? PAL.ketchup;
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 6; roundRect(ctx, -w / 2, -h / 2, w, h, 16); ctx.stroke();
  ctx.lineWidth = 2; roundRect(ctx, -w / 2 + 8, -h / 2 + 8, w - 16, h - 16, 10); ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 0, 4);
  for (let i = 0; i < 60; i++) { ctx.globalCompositeOperation = 'destination-out'; ctx.globalAlpha = 0.55 * a; ctx.fillRect((hash(i * 3.3) - 0.5) * w, (hash(i * 5.7) - 0.5) * h, 2 + hash(i) * 3, 2 + hash(i + 9) * 3); }
  ctx.restore();
  if (d >= land && d < land + 0.5) {                 // ink drops
    const k = (d - land) / 0.5; ctx.save(); ctx.globalAlpha = 0.9 * (1 - k); ctx.fillStyle = col;
    for (let i = 0; i < 9; i++) { const ang = i * 0.7 + 0.3, r = 10 + k * (60 + hash(i * 2.1) * 50); ctx.beginPath(); ctx.arc(x + Math.cos(ang) * r, y + Math.sin(ang) * r * 0.7, 5 * (1 - k) + 1, 0, PI2); ctx.fill(); }
    ctx.restore();
  }
};

UI.wrap = function (text, maxW, font) {
  ctx.font = font; const words = text.split(' '), lines = []; let line = '';
  for (const w of words) { const test = line ? line + ' ' + w : w; if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = w; } else line = test; }
  lines.push(line); return lines;
};

UI.bubbleLayout = function (x, y, text, o = {}) {      // geometry of a bubble at its final length; asserts that the text fits inside it
  const size = o.size ?? 46, maxW = o.maxW ?? 560, font = `600 ${size}px Caveat`;
  const full = UI.wrap(text, maxW, font); ctx.font = font; const lw = Math.max(...full.map(l => ctx.measureText(l).width));
  const w = lw + 50, h = full.length * size * 1.05 + 36;
  let bx, by;
  if (o.fixed) { bx = o.fixed.cx - w / 2; by = o.fixed.y0; } else { bx = clamp(x - w * (o.anchor ?? 0.5), 20, W - w - 20); by = y - h - 64; }
  for (const l of full) if (ctx.measureText(l).width + 50 > w + .5) throw new Error('bubble text does not fit: ' + l);
  if (full.length * size * 1.05 + 36 > h + .5) throw new Error('bubble text is taller than its bubble');
  return { bx, by, w, h, full, size, font };
};
UI.bubble = function (x, y, text, p, o = {}) {        // speech bubble; (x,y) = tail tip, typewriter progress p
  if (p <= 0) return;
  const L = UI.bubbleLayout(x, y, text, o), { bx, by, w, h, size, font } = L;
  const shown = text.slice(0, Math.ceil(text.length * clamp(p))), lines = UI.wrap(shown, o.maxW ?? 560, font);
  const pop = o.pop === false ? 1 : backOut(clamp(p * 6));
  ctx.save(); ctx.translate(x, y); ctx.scale(pop, pop); ctx.translate(-x, -y); ctx.globalAlpha *= o.a ?? 1;
  ctx.shadowColor = 'rgba(40,25,10,.25)'; ctx.shadowBlur = 10; ctx.shadowOffsetY = 5;
  ctx.fillStyle = '#FFFDF6'; ctx.strokeStyle = PAL.ink; ctx.lineWidth = 4; roundRect(ctx, bx, by, w, h, 26); ctx.fill(); ctx.shadowColor = 'transparent'; ctx.stroke();
  const tx = clamp(x, bx + 40, bx + w - 40);
  ctx.beginPath(); ctx.moveTo(tx - 20, by + h - 1); ctx.quadraticCurveTo(tx - 6, by + h + 30, x, y); ctx.quadraticCurveTo(tx + 8, by + h + 24, tx + 22, by + h - 1);
  ctx.fillStyle = '#FFFDF6'; ctx.fill(); ctx.stroke();
  ctx.fillStyle = o.color ?? PAL.ink; ctx.font = font; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  lines.forEach((l, i) => ctx.fillText(l, bx + 25, by + 18 + i * size * 1.05));
  ctx.restore();
};

/* handwriting that writes itself on, left to right with a little pen wobble */
UI.writeOn = function (text, x, y, size, p, o = {}) {
  if (p <= 0) return 0;
  ctx.save(); ctx.font = `${o.weight ?? 600} ${size}px Caveat`; const tw = ctx.measureText(text).width;
  const rw = tw * clamp(p);
  ctx.beginPath(); ctx.rect(x - 8, y - size * 1.2, rw + 8, size * 1.8); ctx.clip();
  ctx.fillStyle = o.color ?? '#3a2f2a'; ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.globalAlpha *= o.a ?? 1;
  ctx.fillText(text, x, y);
  ctx.restore();
  if (p < 1 && !o.nopen) {                             // pen tip
    const px = x + rw, py = y - size * 0.25 + 3 * Math.sin(p * 90);
    ctx.save(); ctx.translate(px, py); ctx.rotate(0.5); ctx.fillStyle = '#6b4a14'; ctx.fillRect(0, -3, 30, 6); ctx.fillStyle = PAL.ink; ctx.beginPath(); ctx.moveTo(0, -3); ctx.lineTo(-9, 0); ctx.lineTo(0, 3); ctx.fill(); ctx.restore();
  }
  return tw;
};
UI.strike = function (x, y, w, p, color = PAL.ketchup) {       // one clean hand-drawn line through the old text (the text stays readable)
  if (p <= 0) return;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 3.2; ctx.lineCap = 'round'; ctx.beginPath(); const n = 18, end = Math.round(n * clamp(p));
  for (let i = 0; i <= end; i++) { const px = x + w * i / n, py = y + 1.2 * Math.sin(i * 1.1) - .6; if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
  ctx.stroke(); ctx.restore();
};
UI.polaroid = function (img, x, y, w, rot, a = 1) {  // img: canvas or image; blank paper frame with a photo on it
  ctx.save(); ctx.globalAlpha *= a; ctx.translate(x, y); ctx.rotate(rot);
  const h = w * 1.18; ctx.shadowColor = 'rgba(40,25,10,.35)'; ctx.shadowBlur = 16; ctx.shadowOffsetY = 8;
  ctx.fillStyle = '#FFFBF0'; roundRect(ctx, -w / 2, -h / 2, w, h, 6); ctx.fill(); ctx.shadowColor = 'transparent'; ctx.strokeStyle = PAL.ink; ctx.lineWidth = 3; ctx.stroke();
  const pw = w - 34, ph = w - 34; ctx.drawImage(img, -pw / 2, -h / 2 + 16, pw, ph); ctx.strokeRect(-pw / 2, -h / 2 + 16, pw, ph);
  ctx.fillStyle = 'rgba(190,160,110,.55)'; ctx.fillRect(-26, -h / 2 - 10, 52, 22);                   // tape
  ctx.restore();
};

UI.subtitle = function (story, t) {
  const s = story.subs.find(s => t >= s.t0 && t <= s.t1); if (!s) return;
  const a = sstep(s.t0, s.t0 + .35, t) * (1 - sstep(s.t1 - .3, s.t1, t));
  ctx.save(); ctx.globalAlpha = a;
  ctx.font = '600 44px Caveat'; const wEn = ctx.measureText(s.en).width; ctx.font = "400 28px 'LXGW WenKai'"; const wZh = ctx.measureText(s.zh).width;
  const w = Math.min(1380, Math.max(wEn, wZh) + 90), h = 94, x0 = W / 2 - w / 2, y0 = 960;
  ctx.fillStyle = 'rgba(250,243,228,.78)'; ctx.shadowColor = 'rgba(40,25,10,.16)'; ctx.shadowBlur = 12; roundRect(ctx, x0, y0, w, h, 26); ctx.fill(); ctx.shadowColor = 'transparent';
  ctx.strokeStyle = 'rgba(43,42,51,.45)'; ctx.lineWidth = 2; ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = PAL.ink;
  ctx.font = '600 44px Caveat'; ctx.fillText(s.en, W / 2, y0 + 43, w - 50);
  ctx.font = "400 28px 'LXGW WenKai'"; ctx.fillStyle = '#5b5560'; ctx.fillText(s.zh, W / 2, y0 + 80, w - 50);
  ctx.restore();
};

/* Pebble's notebook, drawn in code: two cream pages on a spine with rings and a ribbon. (cx,cy) = centre, w x h = open size. */
UI.notebook = function (cx, cy, w, h, a = 1, rot = 0) {
  if (a <= .01) return;
  ctx.save(); ctx.globalAlpha *= a; ctx.translate(cx, cy); ctx.rotate(rot);
  ctx.shadowColor = 'rgba(30,20,10,.38)'; ctx.shadowBlur = 26; ctx.shadowOffsetY = 14;
  ctx.fillStyle = '#9b5b3a'; roundRect(ctx, -w / 2 - 16, -h / 2 - 14, w + 32, h + 28, 22); ctx.fill(); ctx.shadowColor = 'transparent';
  ctx.strokeStyle = PAL.ink; ctx.lineWidth = 4; ctx.stroke();
  for (const side of [-1, 1]) {
    ctx.fillStyle = '#FFF6E0'; ctx.beginPath(); ctx.moveTo(0, -h / 2); ctx.quadraticCurveTo(side * w / 4, -h / 2 - 12, side * w / 2, -h / 2 + 4); ctx.lineTo(side * w / 2, h / 2 + 2);
    ctx.quadraticCurveTo(side * w / 4, h / 2 + 14, 0, h / 2); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = 'rgba(120,150,200,.35)'; ctx.lineWidth = 2;
    for (let i = 0; i < 9; i++) { const y = -h / 2 + 90 + i * (h - 130) / 8; ctx.beginPath(); ctx.moveTo(side * 22, y); ctx.lineTo(side * (w / 2 - 22), y); ctx.stroke(); }
    ctx.strokeStyle = PAL.ink; ctx.lineWidth = 4;
  }
  const g = ctx.createLinearGradient(-30, 0, 30, 0); g.addColorStop(0, 'rgba(120,80,40,0)'); g.addColorStop(.5, 'rgba(120,80,40,.35)'); g.addColorStop(1, 'rgba(120,80,40,0)'); ctx.fillStyle = g; ctx.fillRect(-30, -h / 2, 60, h);
  ctx.strokeStyle = PAL.ink; ctx.lineWidth = 5; ctx.lineCap = 'round';
  for (let i = 0; i < 7; i++) { const y = -h / 2 + 28 + i * (h - 56) / 6; ctx.beginPath(); ctx.moveTo(-26, y); ctx.lineTo(26, y); ctx.stroke(); ctx.fillStyle = '#d8d3c8'; ctx.beginPath(); ctx.arc(-24, y, 6, 0, PI2); ctx.arc(24, y, 6, 0, PI2); ctx.fill(); }
  ctx.fillStyle = PAL.ketchup; ctx.beginPath(); ctx.moveTo(w / 2 - 70, h / 2 - 4); ctx.lineTo(w / 2 - 50, h / 2 - 4); ctx.lineTo(w / 2 - 50, h / 2 + 46); ctx.lineTo(w / 2 - 60, h / 2 + 36); ctx.lineTo(w / 2 - 70, h / 2 + 46); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.restore();
};
