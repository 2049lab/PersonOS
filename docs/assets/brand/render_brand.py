"""Render brand PNGs (favicons, social preview) and, optionally, a review board from the SVGs.

    python docs/assets/brand/render_brand.py
    python docs/assets/brand/render_brand.py --board /tmp/pstills/logo_board_v2.png \\
        --reference /tmp/pgen/gpt_logo_2.png --previous /tmp/brand_v1
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
PAPER = "#F7F7F5"
DARK = "#111318"


def _uri(p: Path) -> str:
    return p.resolve().as_uri()


def _data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def raster(page, svg: Path, out: Path, w: int, h: int, bg: str | None = None) -> None:
    html = HERE / "_r.html"
    html.write_text(f'<body style="margin:0;background:{bg or "transparent"}"><img src="{_uri(svg)}" '
                    f'style="display:block;width:{w}px;height:{h}px"></body>')
    page.set_viewport_size({"width": w, "height": h})
    page.goto(_uri(html))
    page.wait_for_timeout(100)
    page.screenshot(path=str(out), omit_background=bg is None)
    html.unlink()


def board(page, out_path: str, reference: Path, previous: Path | None) -> None:
    tmp = Path("/tmp/brand_r")
    tmp.mkdir(exist_ok=True)

    def mark_cell(svg: Path, bg: str, h: int, ar: float = 19.4 / 26.2) -> str:
        return f'<div style="background:{bg};padding:24px 28px;display:flex;align-items:flex-end"><img src="{_uri(svg)}" style="height:{h}px"></div>'

    def zoom16(svg: Path, bg: str) -> str:
        out = tmp / f"z_{svg.parent.name}_{svg.stem}_{bg.strip('#')}.png"
        raster(page, svg, out, 24, 32, bg)
        im = Image.open(out).convert("RGB").resize((24 * 6, 32 * 6), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    ref_mark = tmp / "ref_mark.png"
    Image.open(reference).convert("RGB").crop((250, 300, 950, 1050)).save(ref_mark)
    rows = []
    for label, d, files in (("PREVIOUS (v5)", previous, ("mark.svg", "mark-dark.svg", "lockup-tagline.svg", "lockup-tagline-dark.svg", "favicon.svg")),
                            ("V6 (this revision)", HERE, ("mark.svg", "mark-dark.svg", "lockup-tagline.svg", "lockup-tagline-dark.svg", "favicon.svg"))):
        if d is None:
            continue
        mark, mark_d, lock, lock_d, fav = (d / f for f in files)
        cells = (
            f'<div style="display:flex;gap:6px;align-items:stretch">'
            f'{mark_cell(mark, PAPER, 256)}{mark_cell(mark_d, DARK, 256)}'
            f'{mark_cell(mark, PAPER, 32)}{mark_cell(mark_d, DARK, 32)}'
            f'<div style="background:{PAPER};padding:24px 28px;display:flex;align-items:flex-end;gap:16px"><img src="{_uri(fav)}" style="width:32px"><img src="{_uri(fav)}" style="width:16px">'
            f'<img src="{zoom16(fav, PAPER)}" style="image-rendering:pixelated"></div></div>'
            f'<div style="display:flex;gap:6px;margin-top:6px"><div style="background:{PAPER};padding:36px 40px;flex:1"><img src="{_uri(lock)}" style="width:100%"></div>'
            f'<div style="background:{DARK};padding:36px 40px;flex:1"><img src="{_uri(lock_d)}" style="width:100%"></div></div>')
        rows.append(f'<div style="margin:18px 0 6px;font:600 13px sans-serif;color:#888;letter-spacing:.08em">{label}</div>{cells}')
    board_html = HERE / "_board.html"
    board_html.write_text(f"""<meta charset="utf-8"><body style="margin:0;background:#fff;font-family:Inter,sans-serif;width:2600px;padding:24px;box-sizing:border-box">
<div style="display:flex;gap:28px">
 <div style="width:760px"><div style="font:600 13px sans-serif;color:#888;letter-spacing:.08em;margin-bottom:6px">REFERENCE</div>
   <img src="{_data(reference)}" style="width:100%"><img src="{_data(ref_mark)}" style="height:420px;margin-top:8px"></div>
 <div style="flex:1">{''.join(rows)}</div>
</div></body>""")
    page.set_viewport_size({"width": 2600, "height": 1000})
    page.goto(_uri(board_html))
    page.wait_for_timeout(400)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=out_path, full_page=True)
    board_html.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="")
    ap.add_argument("--reference", default="/tmp/pgen/gpt_logo_2.png")
    ap.add_argument("--previous", default="", help="directory with the previous revision's SVGs for the board")
    args = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        for n in (16, 32):
            raster(pg, HERE / "favicon.svg", HERE / f"favicon-{n}.png", n, n)
        for n in (180, 512):                      # large sizes can afford the crescent and the long stem
            raster(pg, HERE / "icon.svg", HERE / f"favicon-{n}.png", n, n)
        lock = HERE / "lockup-tagline.svg"
        html = HERE / "_social.html"
        html.write_text(f'<body style="margin:0;width:1280px;height:640px;background:{PAPER};display:flex;align-items:center;'
                        f'justify-content:center"><img src="{_uri(lock)}" style="width:700px"></body>')
        pg.set_viewport_size({"width": 1280, "height": 640})
        pg.goto(_uri(html))
        pg.wait_for_timeout(100)
        pg.screenshot(path=str(HERE / "social-preview.png"))
        html.unlink()
        if args.board:
            board(pg, args.board, Path(args.reference), Path(args.previous) if args.previous else None)
        b.close()


if __name__ == "__main__":
    main()
