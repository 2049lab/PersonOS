# PersonOS brand assets

The canonical monochrome identity is a rounded-square ring whose right side
opens into square particles. `particle-master.svg` preserves its original
paths and clipping geometry. Derived assets normalize that master to a
64-unit square without redrawing or resampling the particles.

[DESIGN.md](DESIGN.md) records the implemented README visual system;
[PRODUCT.md](PRODUCT.md) records the product context and claim boundaries.

| Asset | Use |
|---|---|
| `particle-master.svg` | Canonical geometry and accessible description |
| `mark.svg`, `mark-dark.svg`, `mark-mono.svg` | Transparent standalone mark |
| `wordmark.svg`, `wordmark-dark.svg` | Mark with outlined Inter lettering |
| `readme-hero*.svg` | English/Chinese, desktop/mobile mastheads; charcoal in both themes |
| `nav-*.svg` | Text-only labels inside real HTML links |
| `favicon.svg`, `favicon-*.png`, `icon.svg` | Browser and application-size exports |
| `lockup-tagline*.svg` | Animation end cards |
| `social-preview.*` | Share artwork; uploading repository settings is separate |

`mark-color.svg`, `mark-color-dark.svg` and `mark-color.png` remain as
compatibility filenames. They now carry the same monochrome particle mark.
Use the dedicated light/dark variants where provided. Keep the complete
silhouette; do not stretch, recolor individual particles, add glow or rotate
fragments.

## README layout

The masthead centers the particle mark and large PersonOS name as one horizontal
lockup, followed by a benefit line and subtitle. It keeps its charcoal background
in light and dark themes. Desktop artwork is 1040 × 348; mobile is 480 × 320.

A compact centered badge row links to the CI workflow, Python requirement,
Apache-2.0 license and alpha release status. CI uses the workflow's live badge;
the remaining badges are static labels linked to their repository sources.

Four centered navigation links use transparent 124 × 44 labels. They have no
button backgrounds, borders or particle decoration, and wrap naturally on
narrow screens. The README supplies real `<a>` targets and image alt text;
GitHub does not need custom CSS to make them work.

The body returns to GitHub's normal left-aligned layout: two short project
introduction paragraphs, four core capability bullets, quick start and technical
documentation. The introduction is a project overview. Tutorial scenarios and
code belong in the quick-start and example documentation.

Both READMEs select mobile artwork at 600 px and below through `<picture>`.
ASCII text in generated SVGs is outlined from bundled Inter; strings containing
non-ASCII characters use native CJK text. Mobile supporting text is at least
24 source units, which becomes 12.7 px at a 254 px image width inside a 320 px
GitHub viewport. Check the rendered English and Chinese pages in both themes
after changing copy, dimensions or navigation.

## Rebuild

```bash
uv run --no-project --with 'fonttools[woff]' --with uharfbuzz python docs/assets/brand/build_brand.py
python3 docs/assets/brand/render_brand.py
python3 docs/assets/banner/build.py
```

Run from the repository root. The two raster export commands require
`rsvg-convert` from librsvg. Edit `build_brand.py` and regenerate SVGs; raster
exports are rendered from those canonical SVGs. `render_brand.py` exports icons,
the compatibility mark and social preview; `banner/build.py` exports
`docs/assets/banner.png` from `readme-hero.svg`.

The animation source uses the canonical mark and lockup. Existing rendered
videos remain separate artifacts until explicitly rendered again. Updating
`social-preview.png` does not upload it to repository settings.

Font license: `../fonts/OFL-Inter.txt`. Project license: Apache-2.0.
