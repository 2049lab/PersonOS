"""Contact sheet of every sprite on a mid-grey checker, labelled, for review."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent.parent
meta = json.loads((HERE / "assets" / "sprites.json").read_text())
names = sorted(meta)
cell = 300
cols = 6
rows = (len(names) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 22)), (150, 150, 150))
d = ImageDraw.Draw(sheet)
for i, n in enumerate(names):
    im = Image.open(HERE / "assets" / "sprites" / f"{n}.png")
    im.thumbnail((cell - 10, cell - 10))
    x, y = (i % cols) * cell, (i // cols) * (cell + 22)
    for cy in range(0, cell, 20):
        for cx in range(0, cell, 20):
            if (cx // 20 + cy // 20) % 2: d.rectangle([x + cx, y + cy, x + cx + 19, y + cy + 19], fill=(170, 170, 170))
    sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height)), im)
    d.text((x + 6, y + cell + 4), n, fill=(255, 255, 255))
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pstills/anim_sprites.png"
Path(out).parent.mkdir(parents=True, exist_ok=True)
sheet.save(out); print(out, sheet.size)
