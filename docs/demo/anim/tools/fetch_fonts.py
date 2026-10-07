"""Vendor the two OFL fonts the animation needs, locally, for offline rendering.

    python docs/demo/anim/tools/fetch_fonts.py

- Caveat (variable, latin) for the English handwriting - @fontsource-variable/caveat, OFL.
- LXGW WenKai for the Chinese subtitles: only the unicode-range slices that contain characters used in
  story.json are downloaded (lxgw-wenkai-webfont, OFL), with a generated fonts.css pointing at them.
"""
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "fonts"
OUT.mkdir(exist_ok=True)
CDN = "https://cdn.jsdelivr.net/npm/"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> None:
    (OUT / "Caveat-Variable.woff2").write_bytes(get(CDN + "@fontsource-variable/caveat/files/caveat-latin-wght-normal.woff2"))
    (OUT / "OFL-Caveat.txt").write_bytes(get(CDN + "@fontsource-variable/caveat/LICENSE"))
    story = json.loads((HERE / "story.json").read_text())
    def zh_of(o):
        if isinstance(o, dict):
            return "".join((v if k == "zh" else zh_of(v)) for k, v in o.items() if isinstance(v, (str, dict, list)))
        if isinstance(o, list):
            return "".join(zh_of(v) for v in o)
        return ""
    text = zh_of(story) + story.get("title_zh", "")
    chars = {ord(c) for c in text if ord(c) > 0x2000}
    css = get(CDN + "lxgw-wenkai-webfont@1.7.0/lxgwwenkai-regular.css").decode()
    blocks = re.findall(r"@font-face\s*\{[^}]*\}", css)
    rules, seen = [], set()
    for b in blocks:
        m = re.search(r"unicode-range:\s*([^;}]+)", b)
        u = re.search(r"url\(['\"]?([^)'\"]+)", b)
        if not m or not u:
            continue
        ranges = []
        for part in m.group(1).split(","):
            part = part.strip().lstrip("U+")
            lo, _, hi = part.partition("-")
            ranges.append((int(lo, 16), int(hi or lo, 16)))
        if any(lo <= c <= hi for c in chars for lo, hi in ranges):
            rel = u.group(1).lstrip("./")
            name = Path(rel).name
            if name not in seen:
                seen.add(name)
                (OUT / name).write_bytes(get(CDN + "lxgw-wenkai-webfont@1.7.0/" + rel))
            rules.append(b.replace(u.group(1), name))
    (OUT / "fonts.css").write_text(
        "@font-face{font-family:'Caveat';font-weight:400 700;src:url(Caveat-Variable.woff2) format('woff2');}\n" + "\n".join(rules))
    (OUT / "OFL-LXGWWenKai.txt").write_text("LXGW WenKai is licensed under the SIL Open Font License 1.1 (https://github.com/lxgw/LxgwWenKai).\n")
    print(f"{len(rules)} WenKai slices, {sum(f.stat().st_size for f in OUT.iterdir()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
