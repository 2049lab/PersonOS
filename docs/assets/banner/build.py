"""Export the current README cover from the canonical vector asset."""

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    assets = Path(__file__).resolve().parent.parent
    renderer = shutil.which("rsvg-convert")
    if not renderer:
        raise SystemExit("Install librsvg to provide rsvg-convert.")
    subprocess.run([
        renderer, "-w", "1600", "-o", str(assets / "banner.png"),
        str(assets / "brand" / "readme-hero.svg"),
    ], check=True)


if __name__ == "__main__":
    main()
