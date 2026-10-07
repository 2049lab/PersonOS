/* main.js - p5 owns a WEBGL canvas that p5.brush paints on; each frame is composited onto a plain 2D canvas, where the in-world UI,
 * lettering and the paper grain are drawn. renderAt(t) is the only entry point the recorder needs. */
let paperG, grainC, outC, STORY = null, BUILDING = null, PASS = 'world';
const OVER = [];                                          // overlay draw calls queued by the scene for the 2D compositor

async function setup() {
  createCanvas(W, H, WEBGL); pixelDensity(1); noLoop();
  brush.scaleBrushes(5); defineBrushes();
  paperG = makePaper(); grainC = makeGrain();
  outC = document.getElementById('out'); ctx = outC.getContext('2d');
  STORY = await (await fetch('story.json')).json();
  LOCKUP = await new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = '../../assets/brand/lockup-tagline.svg?v=' + Date.now(); });
  window.DEBUG_IDS = new URLSearchParams(location.search).has('debug');
  await Promise.all([document.fonts.load("600 40px 'Caveat'"), document.fonts.load("700 40px 'Caveat'"), document.fonts.load("400 30px 'LXGW WenKai'", '谁吃了糖果')]);
  await document.fonts.ready;
  window.ready = true;
}
function draw() {
  if (!BUILDING && !BG.ready) return;
  OVER.length = 0;
  randomSeed(1000 + Math.floor(T * BOIL) + (BUILDING ? 0 : 0)); noiseSeed(77);
  push(); translate(-W / 2, -H / 2);
  if (BUILDING) { image(paperG, 0, 0); drawSetStatic(BUILDING); }
  else drawScene(T, PASS);
  pop();
}
function paintOver(t) { for (const f of OVER) f(ctx, t); OVER.length = 0; }
async function glPass(t, pass) { T = t; PASS = pass; await redraw(); }
function copyGL(dst) { const c = dst.getContext('2d'); c.clearRect(0, 0, W, H); c.drawImage(drawingContext.canvas, 0, 0, W, H); }
async function ensureBG() {                              // paint the static furniture once per variant (after setup, so p5.brush flushes normally)
  if (BG.ready) return;
  for (const v of ['night', 'morning']) {
    BUILDING = v; T = 0; await redraw();
    const g = createGraphics(W, H); g.pixelDensity(1); g.drawingContext.drawImage(drawingContext.canvas, 0, 0, W, H); BG[v] = g;
  }
  BUILDING = null; BG.ready = true;
}
window.renderAt = async (t) => {
  await ensureBG();
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  await composeFrame(t);
  ctx.globalCompositeOperation = 'multiply'; ctx.drawImage(grainC, 0, 0); ctx.globalCompositeOperation = 'source-over';
  return true;
};
window.gpuInfo = () => { const gl = drawingContext, e = gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); };
