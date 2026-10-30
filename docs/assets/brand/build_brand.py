"""Generate PersonOS assets from the shared particle vector master.

Requires fonttools[woff] and uharfbuzz. Latin lettering is outlined Inter;
localized labels use native CJK text.
Run this file, then render_brand.py. See README.md in this directory.
"""
from __future__ import annotations

import hashlib
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


def centered(text: str, cx: float, baseline: float, size: float, ink: str,
             weight: int = 500, tracking: float = -.02) -> str:
    if any(ord(c) > 127 for c in text):
        return (f'<text x="{cx}" y="{baseline}" text-anchor="middle" fill="{ink}" '
                f'font-size="{size}" font-weight="{weight}" font-family="Noto Sans CJK SC,'
                f'PingFang SC,Microsoft YaHei,sans-serif">{escape(text)}</text>')
    _, _, advance = text_path(text, x=0, baseline=0, size=size,
                              wght=weight, tracking=tracking)
    return lettering(text, cx-advance/2, baseline, size, ink, weight, tracking)


def hero(*, dark: bool = False, mobile: bool = False, chinese: bool = False) -> str:
    """A centered masthead; one visible mark-and-name group, one message."""
    width, height = (480, 320) if mobile else (1040, 348)
    background, foreground, secondary = "#191B1D", "#F4F2ED", "#C1C1BD"
    body = f'<rect width="{width}" height="{height}" rx="4" fill="{background}"/>'
    size, mark_width, gap = (74, 58, 20) if mobile else (110, 88, 26)
    baseline, mark_top = (112, 55) if mobile else (162, 80)
    _, _, advance = text_path("PersonOS", x=0, baseline=0, size=size,
                              wght=650, tracking=-.035)
    start = (width-mark_width-gap-advance)/2
    # Master visible bounds: x=6.125..58, y=7.25..56.75 in normalized units.
    # Center the visible ink, not the SVG's transparent padding.
    scale = mark_width / 51.875
    body += (f'<g transform="translate({start-6.125*scale} {mark_top-7.25*scale}) '
             f'scale({scale})">{mark_group(foreground)}</g>')
    body += lettering("PersonOS", start+mark_width+gap, baseline, size,
                      foreground, 650, -.035)
    if mobile:
        lines = ("让 Agent 记住你，", "也记住你的变化。") if chinese else (
            "Memory that grows", "with a person.")
        for index, line in enumerate(lines):
            body += centered(line, width/2, 178+index*36, 28, foreground, 450)
        lines = ("面向个人 Agent", "的开源长期记忆框架") if chinese else (
            "An open-source memory framework", "for personal agents.")
        for index, line in enumerate(lines):
            body += centered(line, width/2, 260+index*32, 24, secondary, 400)
    else:
        message = "让 Agent 记住你，也记住你的变化。" if chinese else (
            "Memory that grows with a person.")
        subtitle = "面向个人 Agent 的开源长期记忆框架" if chinese else (
            "An open-source memory framework for personal agents.")
        body += centered(message, width/2, 239, 28, foreground, 450)
        body += centered(subtitle, width/2, 276, 18, secondary, 400)
    label = "PersonOS — 面向个人 Agent 的开源长期记忆框架" if chinese else (
        "PersonOS — An open-source memory framework for personal agents")
    return svg(width, height, body, label)


NAV = {
    "start": ("Quick start", "快速开始", 124),
    "docs": ("Docs", "使用文档", 124),
    "examples": ("Examples", "示例", 124),
    "language": ("简体中文", "English", 124),
}


def navigation(key: str, *, dark: bool = False, chinese: bool = False) -> str:
    """Quiet typographic links, grouped on the masthead's center axis."""
    en, zh, width = NAV[key]
    label = zh if chinese else en
    ink = WHITE if dark else INK
    body = centered(label, width/2, 27, 14, ink, 600 if key == "start" else 500, -.01)
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
    cover = hero()
    out["social-preview.svg"] = svg(1280, 640,
        '<rect width="1280" height="640" fill="#191B1D"/>'
        + '<g transform="translate(0 106) scale(1.230769)">'
        + cover[cover.index('>')+1:cover.rindex('</svg>')] + '</g>')
    return out


def publish_readme_assets(assets: dict[str, str]) -> dict[str, str]:
    """Give changed README artwork new URLs instead of reusing cached paths."""
    published = HERE / "readme"
    published.mkdir(exist_ok=True)
    paths = {}
    for name, content in assets.items():
        if not name.startswith(("readme-hero", "nav-")):
            continue
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()[:12]
        filename = f"{Path(name).stem}.{digest}.svg"
        (published / filename).write_bytes(data)
        paths[Path(name).stem] = f"docs/assets/brand/readme/{filename}"

    # Older fingerprints remain available for readers with cached README HTML.
    image_ref = re.compile(
        r'((?:src|srcset)=")docs/assets/brand/(?:readme/)?'
        r'([a-z-]+)(?:\.[0-9a-f]{12})?\.svg(")'
    )
    for name in ("README.md", "README.zh-CN.md"):
        readme = HERE.parents[2] / name
        source = readme.read_text(encoding="utf-8")
        updated = image_ref.sub(
            lambda match: (match[1] + paths[match[2]] + match[3]
                           if match[2] in paths else match[0]), source
        )
        if updated != source:
            readme.write_text(updated, encoding="utf-8")
    return paths


def main() -> None:
    assets = build()
    for name, content in assets.items():
        (HERE / name).write_text(content)
        print(name, len(content))
    publish_readme_assets(assets)


if __name__ == "__main__":
    main()
