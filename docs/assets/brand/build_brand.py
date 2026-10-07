"""Build the PersonOS brand mark, wordmark, lockups and PNGs from geometry.

    python docs/assets/brand/build_brand.py

The mark is a P whose negative space is a person: a white head, a white neck and white shoulders that widen downward
and run out through the open bottom of the bowl, hugging the stem's inner edge down to the stem's foot. A lighter
crescent behind the lower right of the bowl is the same person remembered over time. Everything is derived from a
24-unit grid with exact circles (combined with shapely and written as a polygon), and text is converted to outlines
(Inter, via fontTools + HarfBuzz kerning), so the SVGs carry no font dependency.

Geometry (units, y down; the mark is 24 tall):
  W  = 4.0     stem width = bowl ring weight
  R  = 8       bowl radius; bowl centre (8, CY). Its left edge is tangent to the stem's outer edge.
  CY = 8 - 0.12  the bowl rises 1.5% of R above the cap line (round overshoot)
  head: white disc, `CLEAR` inside the ring's inner circle at the top
  neck: white, 0.5 x head diameter wide, concave corners rounded (fillet 0.7)
  shoulders: white circle of radius R - W + 0.7 whose left extreme is tangent to the stem's inner edge x = W, continued as a
             strip down the stem; where it leaves the bowl the ring ends in a blunt cut that follows the shoulder circle
  crescent: the bowl disc shifted right/down, behind everything, minus the person
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from shapely.geometry import Point, box
from shapely.ops import unary_union

HERE = Path(__file__).parent
FONT = HERE.parent / "fonts" / "Inter-Variable.woff2"

# ── geometry ────────────────────────────────────────────────────────────
W = 4.0                      # stem width / ring weight
R = 8.0                      # bowl radius
OVERSHOOT = 0.015 * R
CY = R - OVERSHOOT           # bowl centre y (top of bowl sits OVERSHOOT above the cap line y=0)
BOTTOM = 24.0                # stem bottom (square)
CX = R                       # bowl centre x: bowl's left edge x=0 is the stem's outer edge
HEAD_R = 2.9                 # head diameter ~0.36 of the bowl's outer diameter, as in the reference
CLEAR = 1.0                  # white clearance between the head and the ring
NECK_FRAC = 0.5             # neck width / head diameter
NECK_LEN = 0.3
FILLET = 0.8
SHOULDER_EXTRA = 0.7         # shoulder circle radius = R - W + this
CRES = (2.7, 1.57)           # crescent offset (right, down): v5's (2.4, 1.4) enlarged 12%, as in the reference
RES = 128                    # segments per quarter circle


def _disc(cx: float, cy: float, r: float):
    return Point(cx, cy).buffer(r, RES)


def person(w: float = W, head_r: float = HEAD_R, clear: float = CLEAR, neck_len: float = NECK_LEN,
           neck_frac: float = NECK_FRAC, fillet: float = FILLET, bottom: float = BOTTOM):
    """The white person: head + neck + shoulders (+ the strip down the stem), concave corners rounded."""
    r_i = R - w
    hy = CY - r_i + clear + head_r
    apex = hy + head_r + neck_len
    sh_r = R - w + SHOULDER_EXTRA          # a little wider than the counter, so the body flares like the reference
    sh_cx = w + sh_r                       # left extreme stays tangent to the stem's inner edge
    sh_y = apex + sh_r
    neck_w = neck_frac * 2 * head_r
    parts = [_disc(CX, hy, head_r), box(CX - neck_w / 2, hy, CX + neck_w / 2, apex + 1.0), _disc(sh_cx, sh_y, sh_r), box(w, sh_y, sh_cx, bottom + 2)]
    u = unary_union(parts)
    return u.buffer(fillet, RES).buffer(-fillet, RES) if fillet else u


def _poly_path(poly) -> str:
    geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
    out = []
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            pts = list(ring.coords)[:-1]
            out.append("M" + "L".join(f"{x:.3f} {y:.3f}" for x, y in pts) + "Z")
    return "".join(out)


def black_shape(w: float = W, bottom: float = BOTTOM, **kw):
    solid = unary_union([_disc(CX, CY, R), box(0, CY, w, bottom)])
    return solid.difference(person(w=w, bottom=bottom, **kw))


def p_path(w: float = W, bottom: float = BOTTOM, **kw) -> str:
    return _poly_path(black_shape(w, bottom, **kw))


def crescent_path(off=CRES, w: float = W, **kw) -> str:
    shifted = _disc(CX + off[0], CY + off[1], R)
    return _poly_path(shifted.difference(person(w=w, **kw)))


MARK_BOX = (-0.6, CY - R - 0.6, CX + R + CRES[0] + 0.6, BOTTOM + 0.6)   # x0, y0, x1, y1 (padded)

INK, CRESCENT = "#111318", "#9AA3B2"
INK_D, CRESCENT_D = "#F5F6F8", "#5B6475"
MUTED, MUTED_D = "#8A8F98", "#8A8F98"
PAPER = "#F7F7F5"


def mark_group(ink: str, crescent: str | None, *, tx: float = 0, ty: float = 0, **kw) -> str:
    parts = []
    if crescent:
        parts.append(f'<path d="{crescent_path(w=kw.get("w", W), **{k: v for k, v in kw.items() if k not in ("w", "bottom")})}" fill="{crescent}"/>')
    parts.append(f'<path d="{p_path(**kw)}" fill="{ink}"/>')
    return f'<g transform="translate({tx} {ty})">' + "".join(parts) + "</g>"


def svg(w: float, h: float, body: str, vb: tuple | None = None, extra: str = "", label: str = "") -> str:
    x0, y0, x1, y1 = vb or (0, 0, w, h)
    lab = f' role="img" aria-label="{label}"' if label else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{h:g}" '
            f'viewBox="{x0:g} {y0:g} {x1 - x0:g} {y1 - y0:g}"{lab}>{extra}{body}</svg>\n')


# ── text as outlines ────────────────────────────────────────────────────
_FONTS: dict[int, tuple[TTFont, bytes]] = {}


def _font(wght: int):
    if wght not in _FONTS:
        f = TTFont(str(FONT))
        f = instancer.instantiateVariableFont(f, {"wght": wght})
        buf = io.BytesIO()
        f.flavor = None
        f.save(buf)
        _FONTS[wght] = (TTFont(io.BytesIO(buf.getvalue())), buf.getvalue())
    return _FONTS[wght]


def text_path(text: str, *, size: float, wght: int, tracking: float, x: float, baseline: float) -> tuple[str, float, float]:
    """Outline path for `text`; returns (d, ink_left_offset_applied, advance_width). x is the ink-left of the first glyph."""
    font, raw = _font(wght)
    upem = font["head"].unitsPerEm
    scale = size / upem
    face = hb.Face(raw)
    hbfont = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbfont, buf, {"kern": True, "liga": False})
    gs = font.getGlyphSet()
    order = font.getGlyphOrder()
    pen = SVGPathPen(gs, ntos=lambda v: f"{v:.3f}".rstrip("0").rstrip("."))
    # ink-left of the first glyph (its left side bearing) so the text starts exactly at x
    first = order[buf.glyph_infos[0].codepoint]
    lsb = font["hmtx"][first][1] * scale
    cursor = x - lsb
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = order[info.codepoint]
        tp = TransformPen(pen, (scale, 0, 0, -scale, cursor + pos.x_offset * scale, baseline))
        gs[name].draw(tp)
        cursor += pos.x_advance * scale + tracking * size
    return pen.getCommands(), lsb, cursor - (x - lsb)


CAP = 0.7275                 # Inter cap height / em
CAP_H = 0.45 * 2 * R         # wordmark cap height: 0.45 of the bowl diameter
BASELINE = CY + CAP_H / 2    # caps centred on the bowl's centre
WM_SIZE = CAP_H / CAP
GAP_MARK = W                 # one stroke between mark and text
TEXT_X = CX + R + CRES[0] + GAP_MARK
TAG_CAP = 0.42 * CAP_H
TAG_SIZE = TAG_CAP / CAP
TAG_BASELINE = BASELINE + 0.35 * CAP_H + TAG_CAP     # the tagline sits 0.35 cap-heights below the wordmark


def wordmark_body(ink: str, crescent: str | None, text: str, tag: str | None = None, tag_color: str = MUTED):
    d, _, adv = text_path("PersonOS", size=WM_SIZE, wght=500, tracking=-0.015, x=TEXT_X, baseline=BASELINE)
    body = mark_group(ink, crescent) + f'<path d="{d}" fill="{text}"/>'
    right = TEXT_X + adv
    if tag:
        td, _, tadv = text_path(tag, size=TAG_SIZE, wght=400, tracking=0.06, x=TEXT_X, baseline=TAG_BASELINE)
        body += f'<path d="{td}" fill="{tag_color}"/>'
        right = max(right, TEXT_X + tadv)
    return body, right, BOTTOM


def build() -> dict[str, str]:
    out: dict[str, str] = {}
    x0, y0, x1, y1 = MARK_BOX
    mw, mh = x1 - x0, y1 - y0
    out["mark.svg"] = svg(mw * 10, mh * 10, mark_group(INK, CRESCENT), MARK_BOX, label="PersonOS")
    out["mark-dark.svg"] = svg(mw * 10, mh * 10, mark_group(INK_D, CRESCENT_D), MARK_BOX, label="PersonOS")
    out["mark-mono.svg"] = svg(mw * 10, mh * 10, mark_group("#000", None), MARK_BOX, label="PersonOS")
    out["_mark-mono-tint.svg"] = svg(mw * 10, mh * 10,
                                    f'<path d="{crescent_path()}" fill="#000" fill-opacity=".32"/><path d="{p_path()}" fill="#000"/>',
                                    MARK_BOX, label="PersonOS")

    def lockup(ink, cres, text, tag=None, tag_color=MUTED):
        body, right, bottom = wordmark_body(ink, cres, text, tag, tag_color)
        box = (x0, y0, right + 0.6, y1)
        return svg((box[2] - box[0]) * 10, (box[3] - box[1]) * 10, body, box, label="PersonOS"), box

    out["wordmark.svg"], _ = lockup(INK, CRESCENT, INK)
    out["wordmark-dark.svg"], _ = lockup(INK_D, CRESCENT_D, INK_D)
    out["lockup-tagline.svg"], _ = lockup(INK, CRESCENT, INK, "Memory that knows who.", MUTED)
    out["lockup-tagline-dark.svg"], _ = lockup(INK_D, CRESCENT_D, INK_D, "Memory that knows who.", MUTED_D)

    # favicon: the mark alone on a rounded paper tile. At 16px a 1.5px counter vanishes, so the small-size
    # drawing is heavier (ring 4.8), shorter in the stem and drops the crescent; same construction otherwise.
    def tile(inner: str, bw: float, bh: float, fill: float, top: float) -> str:
        sc = fill / bh
        mx = (32 - bw * sc) / 2
        my = (32 - bh * sc) / 2 - top * sc
        return (f'<rect width="32" height="32" rx="7" fill="{INK}"/>'
                f'<g transform="translate({mx:.3f} {my:.3f}) scale({sc:.4f})">{inner}</g>')

    small = mark_group(INK_D, None, w=4.0, head_r=3.0, clear=0.8, neck_len=0.2, neck_frac=0.62, fillet=0.7, bottom=20.5)
    out["favicon.svg"] = svg(32, 32, tile(small, R * 2, 20.5 - (CY - R), 25.0, CY - R), (0, 0, 32, 32), label="PersonOS")
    big = mark_group(INK_D, CRESCENT_D)
    out["icon.svg"] = svg(32, 32, tile(big, R * 2 + CRES[0], BOTTOM - (CY - R + CRES[1]), 24.0, CY - R + CRES[1]), (0, 0, 32, 32), label="PersonOS")
    return out


def main() -> None:
    for name, content in build().items():
        if name.startswith("_"):      # candidates that were tested and not shipped (kept for the review board)
            Path("/tmp/brand_r").mkdir(exist_ok=True)
            (Path("/tmp/brand_r") / name).write_text(content)
            continue
        (HERE / name).write_text(content)
        print(name, len(content))


if __name__ == "__main__":
    main()
