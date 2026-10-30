"""Generate PersonOS assets from the shared particle vector master.

Requires fonttools[woff] and uharfbuzz. Latin lettering is outlined Inter;
localized labels use native CJK text.
Run this file, then render_brand.py. See README.md in this directory.
"""
from __future__ import annotations

import io
import math
import re
from html import escape
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

HERE = Path(__file__).parent
FONT = HERE.parent / "fonts" / "Inter-Variable.woff2"
INK, WHITE, PAPER = "#17191C", "#F1F3F5", "#F6F7F8"
MASTER = HERE / "particle-master.svg"


def mark_group(ink: str = INK, *, color: bool = False, dark: bool = False) -> str:
    """Normalize the existing 420-unit particle mark without redrawing its paths.

    The legacy color argument remains for callers; the shared identity is now
    monochrome. Clip paths and particle positions come from one master.
    """
    source = MASTER.read_text()
    body = source[source.index(">") + 1:source.rindex("</svg>")]
    body = re.sub(r"<(title|desc)\b[^>]*>.*?</\1>", "", body, flags=re.S)
    tone = WHITE if dark else ink
    body = body.replace('fill="#000000"', f'fill="{tone}"')
    return f'<g transform="scale({64 / 420:.12f}) translate(-558 -231)">{body}</g>'


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
        tag, _, tag_advance = text_path("Long-term memory for personal agents.", size=11,
                                       wght=400, tracking=0, x=80, baseline=70)
        body += f'<path d="{tag}" fill="{ink}"/>'
        width = max(width, math.ceil(80 + tag_advance + 2))
        height = 80
    return svg(width, height, body)


def lettering(text: str, x: float, baseline: float, size: float, ink: str,
              weight: int = 500, tracking: float = -.02) -> str:
    """Outline Latin type; leave localized labels as accessible native CJK text."""
    if any(ord(c) > 127 for c in text):
        return (f'<text x="{x}" y="{baseline}" fill="{ink}" font-size="{size}" '
                f'font-weight="{weight}" font-family="Noto Sans CJK SC,PingFang SC,'
                f'Microsoft YaHei,sans-serif">{escape(text)}</text>')
    path, _, _ = text_path(text, x=x, baseline=baseline, size=size,
                           wght=weight, tracking=tracking)
    return f'<path d="{path}" fill="{ink}"/>'


def hero(*, dark: bool = False, mobile: bool = False, chinese: bool = False) -> str:
    """A quiet editorial header: product name, actual purpose, shared particle mark."""
    ink, secondary = (WHITE, "#ABB3BE") if dark else (INK, "#535C68")
    background, stroke = ("#151B23", "#303842") if dark else (PAPER, "#E2E5E9")
    width, height = (480, 280) if mobile else (1040, 300)
    body = (f'<rect x=".5" y=".5" width="{width-1}" height="{height-1}" rx="18" '
            f'fill="{background}" stroke="{stroke}"/>')
    if mobile:
        body += lettering("PersonOS", 28, 76, 52, ink, 600, -.035)
        lines = ("面向个人 Agent", "的长期记忆框架") if chinese else (
            "Long-term memory", "for personal agents.")
        for index, line in enumerate(lines):
            body += lettering(line, 30, 122 + index * 35, 24, secondary, 450)
        body += f'<g transform="translate(286 100) scale({190/64})">{mark_group(ink)}</g>'
    else:
        body += lettering("PersonOS", 52, 117, 76, ink, 600, -.035)
        lines = ("面向个人 Agent", "的长期记忆框架") if chinese else (
            "Long-term memory", "for personal agents.")
        for index, line in enumerate(lines):
            body += lettering(line, 55, 181 + index * 40, 30, secondary, 450)
        body += f'<g transform="translate(692 8) scale({284/64})">{mark_group(ink)}</g>'
    label = "PersonOS — 面向个人 Agent 的长期记忆框架" if chinese else (
        "PersonOS — Long-term memory for personal agents")
    return svg(width, height, body, label)


NAV = {
    "start": ("Quick start", "快速开始", 124),
    "docs": ("Docs", "使用文档", 124),
    "examples": ("Examples", "示例", 124),
    "language": ("简体中文", "English", 124),
}


def navigation(key: str, *, dark: bool = False, chinese: bool = False) -> str:
    """Keep a readable action area and dissolve only the terminal into particles."""
    en, zh, width = NAV[key]
    label = zh if chinese else en
    primary = key == "start"
    ink = (INK if dark else WHITE) if primary else (WHITE if dark else INK)
    fill = (WHITE if dark else INK) if primary else ("#1C2430" if dark else "#F1F3F5")
    stroke = fill if primary else ("#3B4654" if dark else "#CDD3DB")
    particle = fill if primary else ("#6D7B8E" if dark else "#A7AFBA")
    # A fixed, sparse continuation of the mark's rightward square fragments.
    # The first 100 units remain intact so decoration cannot cross the label.
    cutouts = [(105, 5, 3), (102, 10, 2), (107, 15, 3), (104, 20, 2),
               (108, 25, 3), (102, 30, 2), (106, 35, 3), (104, 40, 2)]
    fragments = [(111, 7, 2), (115, 11, 1.5), (110, 15, 2.5),
                 (114, 19, 2), (120, 21, 1.5), (111, 24, 2),
                 (117, 28, 2), (111, 31, 2.5), (121, 33, 1),
                 (115, 36, 1.5), (110, 39, 2), (118, 41, 1)]
    holes = ''.join(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="#000"/>'
                    for x, y, s in cutouts)
    body = ('<defs><mask id="edge" maskUnits="userSpaceOnUse" x="0" y="0" width="124" height="44">'
            f'<rect width="124" height="44" fill="#fff"/>{holes}</mask></defs>'
            '<rect x=".5" y=".5" width="108" height="43" rx="7" '
            f'fill="{fill}" stroke="{stroke}" mask="url(#edge)"/>')
    body += ''.join(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="{particle}"/>'
                    for x, y, s in fragments)
    if any(ord(c) > 127 for c in label):
        text_width = len(label) * 14
    else:
        _, _, text_width = text_path(label, x=0, baseline=0, size=14,
                                    wght=500, tracking=-.01)
    body += lettering(label, (104-text_width)/2, 27, 14, ink, 500, -.01)
    return svg(width, 44, body, label)


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
    theme = f'<style>svg{{color:{INK}}}@media(prefers-color-scheme:dark){{svg{{color:{WHITE}}}}}</style>'
    out["favicon.svg"] = svg(64, 64, theme + mark_group("currentColor"))
    out["icon.svg"] = svg(64, 64,
        f'<rect width="64" height="64" rx="14" fill="{PAPER}"/>'
        f'<g transform="translate(4 4) scale(.875)">{mark_group(color=True)}</g>')
    for chinese in (False, True):
        language = "-zh" if chinese else ""
        for dark in (False, True):
            tone = "-dark" if dark else ""
            for mobile in (False, True):
                size = "-mobile" if mobile else ""
                out[f"readme-hero{language}{size}{tone}.svg"] = hero(
                    dark=dark, mobile=mobile, chinese=chinese)
            for key in NAV:
                out[f"nav-{key}{language}{tone}.svg"] = navigation(
                    key, dark=dark, chinese=chinese)
    out["social-preview.svg"] = svg(1280, 640,
        f'<rect width="1280" height="640" fill="{PAPER}"/>'
        + lettering("PersonOS", 80, 285, 98, INK, 600, -.035)
        + lettering("Long-term memory", 84, 363, 32, "#535C68", 450)
        + lettering("for personal agents.", 84, 409, 32, "#535C68", 450)
        + f'<g transform="translate(820 140) scale(5.5)">{mark_group()}</g>')
    return out


def main() -> None:
    for name, content in build().items():
        (HERE / name).write_text(content)
        print(name, len(content))


if __name__ == "__main__":
    main()
