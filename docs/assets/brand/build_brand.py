"""Generate the canonical PersonOS identity from a 64-unit geometry.

Requires fonttools[woff] and uharfbuzz. All lettering is outlined Inter.
Run this file, then render_brand.py. See README.md in this directory.
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

HERE = Path(__file__).parent
FONT = HERE.parent / "fonts" / "Inter-Variable.woff2"
INK, WHITE, PAPER = "#171816", "#FFFFFF", "#F4F1E9"
COLORS = ("#225CFF", "#111719", "#79C8FF")
CENTER_COLOR = "#FFB344"
ARC = "M32 8 A24 24 0 0 1 52.78461 20"


def mark_group(ink: str = INK, *, color: bool = False, dark: bool = False) -> str:
    """Three equal 60-degree arcs, 120 degrees apart, around one solid center."""
    palette = (COLORS[0], WHITE if dark else COLORS[1], COLORS[2])
    arcs = "".join(
        f'<path d="{ARC}" transform="rotate({angle} 32 32)" '
        f'fill="none" stroke="{palette[i] if color else ink}" '
        'stroke-width="6" stroke-linecap="round"/>'
        for i, angle in enumerate((0, 120, 240))
    )
    return arcs + f'<circle cx="32" cy="32" r="8" fill="{CENTER_COLOR if color else ink}"/>'


def svg(w: float, h: float, body: str, label: str = "PersonOS") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{h:g}" '
            f'viewBox="0 0 {w:g} {h:g}" role="img" aria-label="{label}">'
            f'{body}</svg>\n')


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


def wordmark(ink: str, tagline: bool = False) -> str:
    # 30-unit cap height, vertically centered against the 64-unit mark.
    d, _, advance = text_path("PersonOS", size=30 / .7275, wght=600,
                             tracking=-.025, x=80, baseline=47)
    body = mark_group(color=True, dark=ink == WHITE) + f'<path d="{d}" fill="{ink}"/>'
    width = math.ceil(80 + advance + 2)
    height = 64
    if tagline:
        tag, _, tag_advance = text_path("Memory that knows who.", size=13,
                                       wght=400, tracking=0, x=80, baseline=70)
        body += f'<path d="{tag}" fill="{ink}"/>'
        width = max(width, math.ceil(80 + tag_advance + 2))
        height = 80
    return svg(width, height, body)


def build() -> dict[str, str]:
    out = {
        "mark.svg": svg(64, 64, mark_group()),
        "mark-dark.svg": svg(64, 64, mark_group(WHITE)),
        "mark-mono.svg": svg(64, 64, mark_group("#000000")),
        "mark-color.svg": svg(64, 64, mark_group(color=True)),
        "mark-color-dark.svg": svg(64, 64, mark_group(color=True, dark=True)),
        "wordmark.svg": wordmark(INK),
        "wordmark-dark.svg": wordmark(WHITE),
        "lockup-tagline.svg": wordmark(INK, True),
        "lockup-tagline-dark.svg": wordmark(WHITE, True),
    }
    # Same silhouette in every context; browser theme changes color only.
    theme = f'<style>svg{{color:{COLORS[1]}}}@media(prefers-color-scheme:dark){{svg{{color:{WHITE}}}}}</style>'
    adaptive_mark = mark_group(color=True).replace(f'stroke="{COLORS[1]}"', 'stroke="currentColor"')
    out["favicon.svg"] = svg(64, 64, theme + adaptive_mark)
    out["icon.svg"] = svg(64, 64,
        f'<rect width="64" height="64" rx="14" fill="{PAPER}"/>'
        f'<g transform="translate(4 4) scale(.875)">{mark_group(color=True)}</g>')
    wm = out["wordmark.svg"]
    import xml.etree.ElementTree as ET
    wm_width = float(ET.fromstring(wm).attrib["width"])
    scale = 720 / wm_width
    content = wm[wm.index(">") + 1:wm.rindex("</svg>")]
    out["social-preview.svg"] = svg(1280, 640,
        f'<rect width="1280" height="640" fill="{PAPER}"/>'
        f'<g transform="translate(280 {(640 - 64 * scale) / 2:g}) scale({scale:g})">{content}</g>')
    return out


def main() -> None:
    for name, content in build().items():
        (HERE / name).write_text(content)
        print(name, len(content))


if __name__ == "__main__":
    main()
