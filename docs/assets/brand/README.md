# PersonOS identity

This directory is the canonical source for the PersonOS project mark. The
2049lab organization avatar is a separate laboratory identity.

The mark has one solid center and three equal circular arcs. Every SVG variant
uses the same 64 × 64 geometry: center (32, 32), center radius 8, arc radius
24, stroke width 6, 60-degree sweeps and rotations of 0, 120 and 240 degrees.
The lettering is Inter SemiBold, converted to outlines.

| Asset | Use |
|---|---|
| `mark-color.svg`, `wordmark.svg` | Primary color identity on light backgrounds |
| `mark-color-dark.svg`, `wordmark-dark.svg` | Color identity with a white neutral arc and lettering on dark backgrounds |
| `mark-color.png` | Transparent 512 px export of the primary mark |
| `mark.svg`, `mark-dark.svg`, `mark-mono.svg` | Monochrome alternatives for single-color use |
| `favicon.svg` | Browser icon; adapts to the browser's light/dark preference |
| `icon.svg`, `favicon-180.png`, `favicon-512.png` | Color mark on a warm-white application tile |
| `favicon-16.png`, `favicon-32.png` | Raster fallback on light backgrounds |
| `lockup-tagline*.svg` | Animation end cards |
| `social-preview.*` | Share artwork |

Keep the mark square, leave at least one center radius of clear space around
it, and display it at 16 pixels or larger. Use the explicit light/dark assets;
do not recolor them with CSS filters, stretch them, add shadows, or rotate
individual arcs. The symbol uses cobalt blue, cyan and a neutral arc around
an amber center; the lettering remains black or white for legibility.

## Color identity

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="mark-color-dark.svg">
  <img src="mark-color.svg" alt="PersonOS: three colored arcs around an amber center" width="160" height="160">
</picture>

Use this flat color mark in navigation, project introductions and avatars.
Keep its background transparent except in application tiles. Do not add glass
effects, gradients, lighting or dimensional backgrounds to the mark.

## Rebuild

```bash
uv run --no-project --with 'fonttools[woff]' --with uharfbuzz python docs/assets/brand/build_brand.py
python docs/assets/brand/render_brand.py  # requires rsvg-convert from librsvg
```

Copy the generated SVG files into website/workbench asset directories, keeping
their contents identical. Do not maintain separate drawings in components.
README and animation source already reference this directory. Previously
published video files are historical renders; their end cards only change
when the animation is rendered and published again.

Font license: `../fonts/OFL-Inter.txt`. Project license: Apache-2.0.
