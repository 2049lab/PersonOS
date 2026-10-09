"""Build the PersonOS architecture diagrams (SVG + PNG + contact sheet).
Brand assets are generated only by brand/build_brand.py.

    python docs/assets/build_assets.py

Needs: fonttools + brotli (font subsetting) and playwright with chromium (PNG rendering).
SVGs are fully self-contained: a subset of Inter / JetBrains Mono is embedded as base64 woff2,
so they render the same as <img> on GitHub, in browsers and in the replay page.
All artwork is original vector art (Apache-2.0); fonts are SIL OFL 1.1 (see fonts/).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

HERE = Path(__file__).parent
FONTS = HERE / "fonts"

VIOLET, CYAN = "#7C5CFF", "#22D3EE"
WARN, OK = "#FF5C7A", "#34D399"
DARK = dict(bg="#0B0D12", surface="#12151C", border="#1F2430", text="#E6E8EE", muted="#8A93A6",
            surface2="#181C26", dots="#E6E8EE", dot_op=0.07)
LIGHT = dict(bg="#FFFFFF", surface="#F6F7FB", border="#DDE1EA", text="#0B0D12", muted="#5B6478",
             surface2="#ECEEF5", dots="#0B0D12", dot_op=0.07)

_UNICODE = "".join(chr(c) for c in range(0x20, 0x7F)) + "·→←—…’“”×✓"


def _font_b64(name: str, chars: str = _UNICODE) -> str:
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga", "calt"]
    opts.notdef_outline = False
    opts.hinting = False
    # Preserve the source font metadata so identical diagrams produce identical bytes.
    opts.recalc_timestamp = False
    font = TTFont(str(FONTS / name), recalcTimestamp=False)
    limits = {"wght": (400, 700)}
    font = instancer.instantiateVariableFont(
        font, {k: v for k, v in limits.items() if any(a.axisTag == k for a in font["fvar"].axes)})
    tmp = io.BytesIO()
    font.save(tmp)
    tmp.seek(0)
    font = TTFont(tmp, recalcTimestamp=False)
    sub = subset.Subsetter(opts)
    sub.populate(text=chars)
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


_FONT_CACHE: dict[tuple, str] = {}


def font_css(sans_chars: str = _UNICODE, mono: bool = True) -> str:
    key = (sans_chars, mono)
    if key not in _FONT_CACHE:
        css = ("@font-face{font-family:'Inter';font-weight:400 700;"
               f"src:url(data:font/woff2;base64,{_font_b64('Inter-Variable.woff2', sans_chars)}) format('woff2');}}")
        if mono:
            css += ("@font-face{font-family:'JetBrains Mono';font-weight:400 700;"
                    f"src:url(data:font/woff2;base64,{_font_b64('JetBrainsMono-Variable.woff2')}) format('woff2');}}")
        _FONT_CACHE[key] = css
    return _FONT_CACHE[key]


SANS = "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\""
MONO = "font-family=\"'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace\""


def defs_grad(uid: str = "g", x1=0, y1=0, x2=1, y2=1, units: str = "objectBoundingBox") -> str:
    return (f'<linearGradient id="{uid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'gradientUnits="{units}"><stop offset="0" stop-color="{VIOLET}"/>'
            f'<stop offset="1" stop-color="{CYAN}"/></linearGradient>')


def svg_doc(w: int, h: int, body: str, defs: str = "", font: bool = True, label: str = "",
            css: str | None = None) -> str:
    style = f"<style>{css if css is not None else font_css()}</style>" if font else ""
    aria = f' role="img" aria-label="{label}"' if label else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"{aria}><defs>{style}{defs}</defs>{body}</svg>\n')


# ───────────────────────── architecture ─────────────────────────

class Arch:
    def __init__(self, t: dict) -> None:
        self.t = t
        self.o: list[str] = []
        self.checks: list[tuple[str, float]] = []

    def text(self, x, y, s, size=14, weight=400, fill=None, mono=False, anchor="start", maxw=None, ls=0):
        fam = MONO if mono else SANS
        fill = fill or self.t["text"]
        cid = ""
        if maxw:
            cid = f' data-maxw="{maxw}"'
        self.o.append(f'<text x="{x}" y="{y}" {fam} font-size="{size}" font-weight="{weight}" fill="{fill}" '
                      f'text-anchor="{anchor}" letter-spacing="{ls}"{cid}>{s.replace("&", "&amp;")}</text>')

    def rect(self, x, y, w, h, rx=14, fill=None, stroke=None, dash=None, sw=1):
        fill = fill or self.t["surface2"]
        stroke = stroke or self.t["border"]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>')

    def arrow(self, x1, y1, x2, y2, color="url(#gl)", sw=1.6):
        self.o.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>')
        if y1 == y2:
            self.o.append(f'<path d="M{x2 - 7} {y2 - 5} L{x2} {y2} L{x2 - 7} {y2 + 5}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
        else:
            self.o.append(f'<path d="M{x2 - 5} {y2 - 7} L{x2} {y2} L{x2 + 5} {y2 - 7}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')

    def column(self, x, w, n, title, sub):
        t = self.t
        self.rect(x, 60, w, 580, rx=22, fill=t["surface"], stroke=t["border"])
        self.o.append(f'<circle cx="{x + 38}" cy="102" r="15" fill="url(#ga)"/>')
        self.text(x + 38, 108, str(n), 15, 700, fill="#0B0D12", anchor="middle")
        self.text(x + 64, 108, title, 22, 600, maxw=w - 80)
        self.text(x + 28, 144, sub, 12, 400, fill=t["muted"], mono=True, maxw=w - 56)

    def step(self, x, y, w, h, title, sub, accent=None, icon=None):
        t = self.t
        self.rect(x, y, w, h, rx=12, stroke=accent or t["border"])
        tx = x + 16
        if icon:
            self.o.append(icon)
            tx = x + 64
        self.text(tx, y + (h / 2 - 3 if sub else h / 2 + 5), title, 16, 600, maxw=w - (tx - x) - 12)
        if sub:
            self.text(tx, y + h / 2 + 17, sub, 11, 400, fill=t["muted"], mono=True, maxw=w - (tx - x) - 12)

    def pill(self, x, y, label, fill=None, color=None, stroke=None, mono=True, size=12, h=30, pad=12):
        w = round(pad * 2 + (7.2 if mono else 7.6) * len(label) * size / 12)
        self.rect(x, y, w, h, rx=h / 2, fill=fill or "none", stroke=stroke or self.t["border"])
        self.text(x + w / 2, y + h / 2 + 4, label, size, 500, fill=color or self.t["muted"], mono=mono, anchor="middle")
        return w


def architecture_svg(t: dict) -> str:
    a = Arch(t)
    W, H = 1600, 700
    xs = [35, 550, 1065]
    ws = [480, 480, 500]
    stroke = t["muted"]
    defs = (defs_grad("ga", 0, 0, 1, 1) + defs_grad("gl", 0, 0, 1600, 0, "userSpaceOnUse")
            + f'<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="1.1" fill="{t["dots"]}" opacity="{t["dot_op"]}"/></pattern>')
    a.o.append(f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/><rect width="{W}" height="{H}" fill="url(#dots)"/>')
    a.text(35, 38, "PERSONOS \u00b7 ARCHITECTURE", 12, 500, fill=t["muted"], mono=True, ls=1.5)

    # arrows between columns
    for i in range(2):
        a.arrow(xs[i] + ws[i] + 6, 350, xs[i + 1] - 6, 350)

    def ic(path, x, y, color=None):
        return (f'<g transform="translate({x} {y})" fill="none" stroke="{color or stroke}" stroke-width="1.8" '
                f'stroke-linecap="round" stroke-linejoin="round">{path}</g>')

    film = ic('<rect x="2" y="6" width="32" height="24" rx="4"/><path d="M10 6v24M26 6v24M2 14h8M2 22h8M26 14h8M26 22h8"/>', 14, 22)
    image = ic('<rect x="2" y="5" width="32" height="26" rx="4"/><circle cx="12" cy="14" r="3"/><path d="M3 27l9-8 7 6 5-4 9 8"/>', 14, 22)
    bubble = ic('<path d="M4 8a4 4 0 0 1 4-4h20a4 4 0 0 1 4 4v14a4 4 0 0 1-4 4H16l-8 6v-6a4 4 0 0 1-4-4z"/><path d="M11 12h14M11 18h9"/>', 14, 22)

    # 1 Perceive
    x, w = xs[0], ws[0]
    a.column(x, w, 1, "Perceive", "text, images and video")
    for i, (ti, su, icon) in enumerate([("Text", "conversation turns", bubble),
                                       ("Images", "text + visual content", image),
                                       ("Video clips", "frames \u00b7 speech \u00b7 actions", film)]):
        y = 176 + i * 94
        a.step(x + 24, y, w - 48, 76, ti, su,
               icon=icon.replace("translate(14 22)", f"translate({x + 38} {y + 16})"))
    a.arrow(x + w / 2, 444, x + w / 2, 466, color=t["muted"], sw=1.3)
    a.rect(x + 24, 472, w - 48, 144, rx=14, fill=t["surface2"], stroke=VIOLET, dash="4 4")
    a.text(x + 40, 497, "OPTIONAL VIDEO PATH", 10, 500, fill=t["muted"], mono=True, ls=1)
    a.text(x + 40, 526, "Associate people with events", 16, 600, maxw=w - 80)
    a.text(x + 40, 550, "Revise identity matches across clips.", 12, fill=t["muted"], maxw=w - 80)
    a.text(x + 40, 582, "Requires multimodal + identity backends;", 11, fill=t["muted"], mono=True, maxw=w - 80)
    a.text(x + 40, 600, "commits person links at session close.", 11, fill=t["muted"], mono=True, maxw=w - 80)

    # 2 Remember: all input paths converge on evidence-linked memory.
    x, w = xs[1], ws[1]
    a.column(x, w, 2, "Remember", "evidence-linked long-term memory")
    for i, (ti, su) in enumerate([("evidence", "source text, media and timestamps"),
                                  ("memcell", "episodic records linked to sources"),
                                  ("atom", "retrieval anchors with evidence references")]):
        y = 172 + i * 76
        a.step(x + 24, y, w - 48, 56, "", None)
        a.text(x + 40, y + 24, ti, 15, 600, mono=True, maxw=w - 80)
        a.text(x + 40, y + 43, su, 11, 400, fill=t["muted"], mono=True, maxw=w - 80)
        if i < 2:
            a.arrow(x + w / 2, y + 58, x + w / 2, y + 74, color=t["muted"], sw=1.3)
    a.text(x + 28, 415, "ACROSS SESSIONS", 10, 500, fill=t["muted"], mono=True, ls=1)
    half = (w - 64) / 2
    for dx, title, lines in [(24, "atom_chain", ("Related statements", "across records")),
                             (40 + half, "profile", ("Versioned user", "profile summary"))]:
        a.rect(x + dx, 432, half, 88, rx=12)
        a.text(x + dx + 16, 458, title, 15, 600, mono=True, maxw=half - 32)
        for j, line in enumerate(lines):
            a.text(x + dx + 16, 480 + 18 * j, line, 11, fill=t["muted"], mono=True, maxw=half - 32)
    a.text(x + 28, 567, "Source evidence stays linked.", 12, 500, maxw=w - 56)
    a.text(x + 28, 593, "Related statements can remain in conflict.", 11, fill=t["muted"], mono=True, maxw=w - 56)

    # 3 Recall
    x, w = xs[2], ws[2]
    a.column(x, w, 3, "Recall", "evidence-backed answers")
    px = x + 24
    px += a.pill(px, 168, "fast", color=t["text"], h=34, pad=22) + 10
    px += a.pill(px, 168, "deep (agent)", color=t["text"], h=34, pad=16, stroke=VIOLET) + 10
    a.pill(px, 168, "auto", color=t["text"], h=34, pad=22)
    a.step(x + 24, 224, w - 48, 68, "Text or image query", "scoped to a user")
    a.arrow(x + w / 2, 298, x + w / 2, 314, color=t["muted"], sw=1.3)
    a.step(x + 24, 320, w - 48, 76, "Retrieve & rerank", "atoms \u2192 source episodes and evidence")
    a.arrow(x + w / 2, 402, x + w / 2, 418, color=t["muted"], sw=1.3)
    a.rect(x + 24, 424, w - 48, 100, rx=12, stroke=VIOLET)
    a.text(x + 40, 452, "Answer with references", 16, 600, maxw=w - 80)
    a.text(x + 40, 477, "supporting material + review information", 11, fill=t["muted"], mono=True, maxw=w - 80)
    a.text(x + 40, 506, "trace() follows source evidence", 11, fill=t["muted"], mono=True, maxw=w - 80)
    a.text(x + 28, 567, "auto starts fast; deep follows if needed.", 12, 500, maxw=w - 56)
    a.text(x + 28, 593, "Deep recall requires optional dependencies.", 11, fill=t["muted"], mono=True, maxw=w - 56)

    return svg_doc(W, H, "".join(a.o), defs,
                   label="PersonOS architecture: perceive, remember, recall; optional video identity path")


# ───────────────────────── render ─────────────────────────

def write(name: str, content: str) -> Path:
    p = HERE / name
    p.write_text(content)
    return p


def build_svgs() -> dict[str, Path]:
    return {
        "architecture.svg": write("architecture.svg", architecture_svg(DARK)),
        "architecture-light.svg": write("architecture-light.svg", architecture_svg(LIGHT)),
    }


def render_pngs(svgs: dict[str, Path]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()

        def shot(svg: Path, out: str, w: int, h: int, scale: float) -> None:
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=scale)
            page = ctx.new_page()
            page.goto(svg.resolve().as_uri())
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(150)
            over = page.evaluate("""() => [...document.querySelectorAll('text[data-maxw]')].map(t =>
                [t.textContent, t.getComputedTextLength(), +t.dataset.maxw]).filter(a => a[1] > a[2])""")
            for txt, got, mx in over:
                print(f"  OVERFLOW {svg.name}: {txt!r} {got:.0f} > {mx:.0f}")
            page.screenshot(path=str(HERE / out), omit_background=True)
            ctx.close()

        shot(svgs["architecture.svg"], "architecture.png", 1600, 700, 2)
        shot(svgs["architecture-light.svg"], "architecture-light.png", 1600, 700, 2)

        sheet = HERE / "_preview.html"
        def panel(bg, fg, arch, label):
            return (f'<section style="background:{bg};color:{fg}"><h2>{label}</h2>'
                    f'<img src="{arch}" width="1280"></section>')
        sheet.write_text(
            '<!doctype html><meta charset="utf-8"><style>body{margin:0;display:flex;font-family:sans-serif}'
            'section{padding:32px;width:1280px}h2{font-size:16px}</style>'
            + panel("#0B0D12", "#E6E8EE", "architecture.svg", "On dark")
            + panel("#FFFFFF", "#0B0D12", "architecture-light.svg", "On white"))
        page = browser.new_page(viewport={"width": 2688, "height": 1500})
        page.goto(sheet.resolve().as_uri())
        page.wait_for_timeout(300)
        page.screenshot(path=str(HERE / "_preview.png"), full_page=True)
        sheet.unlink()
        browser.close()


if __name__ == "__main__":
    render_pngs(build_svgs())
    for f in sorted(HERE.glob("*")):
        if f.is_file() and f.suffix in (".svg", ".png"):
            print(f"{f.name:28s}{f.stat().st_size / 1024:8.1f} KB")
