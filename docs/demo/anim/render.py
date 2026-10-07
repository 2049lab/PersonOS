"""Render the film frame by frame (deterministic: every frame is a pure function of t), several browsers in parallel, then encode.

    python docs/demo/anim/render.py --range 0 10.8 --name halloween_s0-2         # -> docs/demo/out/halloween_s0-2.mp4
    python docs/demo/anim/render.py                                              # the whole film -> docs/demo/out/halloween.mp4
    python docs/demo/anim/render.py --stills 3,5.5,9 --prefix anim_shot1_        # PNG stills only

The recorder steps window.renderAt(t) in headless chromium (software GL) and screenshots the compositor canvas.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
OUT = HERE.parent / "out"
FLAGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--force-color-profile=srgb"]


def serve():
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(DOCS)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def worker(jobs: list[tuple[float, str, str]], debug: bool = False) -> int:
    """jobs: (t, path, fmt). One browser, one page, many frames."""
    srv, port = serve()
    with sync_playwright() as p:
        b = p.chromium.launch(args=FLAGS)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        page.on("pageerror", lambda e: print("page error:", e, flush=True))
        page.goto(f"http://127.0.0.1:{port}/demo/anim/index.html{'?debug=1' if debug else ''}")
        page.wait_for_function("window.ready === true", timeout=120000)
        for t, path, fmt in jobs:
            page.evaluate("t => window.renderAt(t)", t)
            mime = "image/png" if fmt == "png" else "image/jpeg"
            data = page.evaluate("m => document.getElementById('out').toDataURL(m, 0.96)", mime)
            Path(path).write_bytes(base64.b64decode(data.split(",", 1)[1]))
        b.close()
    return len(jobs)


def run_parallel(jobs, workers, debug=False):
    chunks = [jobs[i::workers] for i in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [pool.apply_async(worker, (c, debug)) for c in chunks if c]
        while not all(r.ready() for r in res):
            time.sleep(15)
            done = sum(1 for _, p, _ in jobs if Path(p).exists())
            if done:
                print(f"{done}/{len(jobs)} frames  {(time.time() - t0) / done * (len(jobs) - done):.0f}s left", flush=True)
        for r in res:
            r.get()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", nargs=2, type=float, default=[0, 50.0])
    ap.add_argument("--fps", type=int, default=24, help="output frame rate")
    ap.add_argument("--draw-fps", type=int, default=12, help="distinct frames drawn per second (animation on twos at 12; each is held for fps/draw-fps output frames)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="halloween")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir", default="/tmp/pstills")
    ap.add_argument("--prefix", default="anim_t")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--debug", action="store_true", help="draw anchor ids on every tag, stamp and bubble")
    ap.add_argument("--keep-frames", default="", help="write frames here and keep them (resumable: existing frames are skipped)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stills:
        Path(args.stills_dir).mkdir(parents=True, exist_ok=True)
        jobs = [(float(s), str(Path(args.stills_dir) / f"{args.prefix}{s.replace('.', '_')}.png"), "png") for s in args.stills.split(",")]
        run_parallel(jobs, min(args.workers, len(jobs)), args.debug)
        for _, p, _ in jobs:
            print(p)
        return 0
    t0, t1 = args.range
    n = int(round((t1 - t0) * args.draw_fps))
    tmp = Path(args.keep_frames) if args.keep_frames else Path(tempfile.mkdtemp(prefix="halloween_"))
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(t0 + i / args.draw_fps, str(tmp / f"f{i:05d}.jpg"), "jpg") for i in range(n)]
    jobs = [j for j in jobs if not Path(j[1]).exists()]
    if jobs:
        run_parallel(jobs, args.workers)
    mp4 = OUT / f"{args.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.draw_fps), "-i", str(tmp / "f%05d.jpg"), "-r", str(args.fps), "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    if not args.keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{mp4}  {mp4.stat().st_size / 1e6:.1f} MB  {n / args.draw_fps:.1f}s, drawn at {args.draw_fps} fps, encoded at {args.fps} fps")
    if not args.no_preview:
        gif = OUT / f"{args.name}.gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=True)
        print(gif, f"{gif.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
