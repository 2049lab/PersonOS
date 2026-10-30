# PersonOS identity

PersonOS uses the black-and-white particle mark already present on the
[website](https://personos-ai.com/) and native clients: a rounded-square ring
whose right side opens into square particles. `particle-master.svg` preserves
the original paths and clipping geometry. Other assets normalize that master
to a 64-unit square; they do not redraw or randomly resample the particles.

| Asset | Use |
|---|---|
| `particle-master.svg` | Canonical geometry and accessible description |
| `mark.svg`, `mark-dark.svg`, `mark-mono.svg` | Transparent standalone mark |
| `wordmark.svg`, `wordmark-dark.svg` | Mark with outlined Inter lettering |
| `readme-hero*.svg` | English/Chinese, desktop/mobile, light/dark README headers |
| `nav-*.svg` | Native GitHub image links with readable action labels |
| `favicon.svg`, `favicon-*.png`, `icon.svg` | Browser and application-size exports |
| `lockup-tagline*.svg` | Animation end cards |
| `social-preview.*` | Share artwork; uploading repository settings is separate |

`mark-color.svg`, `mark-color-dark.svg` and `mark-color.png` remain as
compatibility filenames. They now carry the same monochrome particle mark.
Use the dedicated light/dark variants, keep the complete silhouette, and avoid
stretching, recoloring individual particles, adding glow or rotating fragments.

## README layout

The header uses one composition with product name, purpose and the shared mark.
The narrow version rearranges that composition. Navigation remains real HTML
links; the linked SVGs provide styling because GitHub strips custom page CSS.
The four equal-width controls wrap into two rows on narrow screens. Their right
edges dissolve into a small fixed set of square fragments; text stays in a solid,
undecorated area. The compact English label “Docs” keeps that space readable. Every link
has an accessible text label, and every image is hosted in this repo.

The cover-and-navigation structure was informed by the
[LanceDB](https://github.com/lancedb/lancedb) and
[Supabase](https://github.com/supabase/supabase) READMEs. Their artwork is not
included. Localized labels use native CJK text; Latin display lettering is
outlined from the bundled Inter font.

## Rebuild

```bash
uv run --no-project --with 'fonttools[woff]' --with uharfbuzz python docs/assets/brand/build_brand.py
python docs/assets/brand/render_brand.py  # requires rsvg-convert from librsvg
python docs/assets/banner/build.py       # exports the current README cover
```

The animation source uses the canonical mark and lockup. Previously rendered
video files remain historical until explicitly rendered again.

Font license: `../fonts/OFL-Inter.txt`. Project license: Apache-2.0.
