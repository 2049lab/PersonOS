"""Render canonical SVG assets with librsvg (rsvg-convert).

    python docs/assets/brand/render_brand.py

No browser or network access is required. Install librsvg separately.
"""
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).parent


def main() -> None:
    renderer = shutil.which('rsvg-convert')
    if not renderer:
        raise SystemExit('Install librsvg to provide rsvg-convert.')
    for size in (16, 32, 180, 512):
        source = 'favicon.svg' if size < 180 else 'icon.svg'
        output = HERE / f'favicon-{size}.png'
        subprocess.run([renderer, '-w', str(size), '-h', str(size),
                        '-o', str(output), str(HERE / source)], check=True)
    subprocess.run([renderer, '-o', str(HERE / 'social-preview.png'),
                    str(HERE / 'social-preview.svg')], check=True)


if __name__ == '__main__':
    main()
