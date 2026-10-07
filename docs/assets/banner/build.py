"""Build docs/assets/banner.png: Pebble is drawn by the film's own rig (docs/demo/anim) so the banner and the
short share one character; the layout is banner/index.html, screenshotted at 2x.

    python docs/assets/banner/build.py
"""
from __future__ import annotations

import base64
import functools
import http.server
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
FLAGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--force-color-profile=srgb"]


def serve():
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(DOCS)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def grain(path: Path, n=320, seed=3):
    rng = np.random.default_rng(seed)
    g = (255 - rng.integers(0, 28, (n, n))).astype(np.uint8)
    Image.fromarray(g, "L").save(path)


def main():
    port = serve()
    grain(HERE / "grain.png")
    with sync_playwright() as p:
        b = p.chromium.launch(args=FLAGS)
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.goto(f"http://127.0.0.1:{port}/demo/anim/index.html")
        pg.wait_for_function("window.ready === true", timeout=120_000)
        data = pg.evaluate("renderPebble(105, { noShadow: true, plain: true, look: [-.35, -.1], glow: .9, armR: .9, armL: .35, antSway: .2 })")
        im = Image.open(__import__("io").BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGBA")
        bbox = im.getbbox()
        im.crop(bbox).save(HERE / "pebble.png")
        print("pebble", im.crop(bbox).size)

        pg2 = b.new_page(viewport={"width": 1600, "height": 640}, device_scale_factor=2)
        pg2.goto(f"http://127.0.0.1:{port}/assets/banner/index.html")
        pg2.wait_for_timeout(600)
        pg2.evaluate("document.fonts.ready")
        pg2.locator("#banner").screenshot(path=str(DOCS / "assets" / "banner.png"), omit_background=True)
        b.close()
    out = Image.open(DOCS / "assets" / "banner.png").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
    out.save(DOCS / "assets" / "banner.png", optimize=True)
    print("banner", out.size, (DOCS / "assets" / "banner.png").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
