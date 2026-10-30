---
name: PersonOS README
description: Centered monochrome introduction for the public Python framework.
colors:
  masthead: "#191B1D"
  masthead-ink: "#F4F2ED"
  masthead-secondary: "#C1C1BD"
  ink-light: "#17191C"
  ink-dark: "#F1F3F5"
  badge-python: "#3776AB"
  badge-neutral: "#59636e"
typography:
  display:
    fontFamily: "Inter"
    fontSize: "110px"
    fontWeight: 650
    letterSpacing: "-0.035em"
  display-mobile:
    fontFamily: "Inter"
    fontSize: "74px"
    fontWeight: 650
    letterSpacing: "-0.035em"
  benefit:
    fontFamily: "Inter, Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"
    fontSize: "28px"
    fontWeight: 450
  supporting:
    fontFamily: "Inter, Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"
    fontSize: "18px"
    fontWeight: 400
  supporting-mobile:
    fontFamily: "Inter, Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"
    fontSize: "24px"
    fontWeight: 400
  navigation:
    fontFamily: "Inter, Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    letterSpacing: "-0.01em"
rounded:
  masthead: "4px"
components:
  navigation-link:
    typography: "{typography.navigation}"
    width: "124px"
    height: "44px"
  masthead:
    backgroundColor: "{colors.masthead}"
    textColor: "{colors.masthead-ink}"
    rounded: "{rounded.masthead}"
---

# Design System: PersonOS README

## Overview

**Creative North Star: "The centered masthead"**

This document describes the public repository's README artwork and composition,
not a runtime application theme. A compact, monochrome masthead introduces the
framework, followed by project status, navigation and an ordinary GitHub project
overview. Typography and alignment provide the hierarchy.

## Colors

The palette is neutral. The masthead uses its fixed charcoal, warm white and
secondary gray in both themes. Transparent navigation switches between the light
and dark ink tokens. Project badges retain their standard status treatment:
Python blue, neutral license and alpha labels, and the live CI badge's current
color. README body colors remain GitHub-native.

## Typography

Generated ASCII text is outlined from the bundled Inter variable font. A string
containing non-ASCII characters stays native text with the CJK fallback stack;
its appearance therefore depends on the renderer's installed fonts. Preserve
alt text in the README and accessible labels in the SVGs.

Token sizes are source SVG units. The PersonOS name uses the display roles; the
benefit stays prominent and the subtitle quieter. Quick start navigation uses
weight 600, while the other links use the navigation role's normal weight.
The mobile hero splits the benefit and subtitle into two lines each.

The mobile subtitle uses the supporting-mobile role. At a 254 px rendered image
width, its 24-unit type becomes 12.7 px.
Use that narrow rendering as a readability check when editing copy.

## Layout

The mark and PersonOS name form one horizontal group centered by visible ink,
including the mark's transparent padding correction. The benefit, subtitle,
badge row and navigation share the center axis. The project introduction and
all subsequent sections use GitHub's normal left-aligned document layout.

| Artwork | Desktop source | Mobile source |
|---|---|---|
| Masthead | 1040 × 348 | 480 × 320 |
| Each navigation label | 124 × 44 | 124 × 44 |

The README selects mobile artwork at viewport widths of 600 px and below.
Navigation and badges wrap as native inline links. After the header, two short
introduction paragraphs and four core capability bullets lead to quick start,
memory organization, integration and documentation links.

## Elevation & Depth

The surface is flat: no shadows, glows or raised cards. The charcoal masthead
provides the main tonal contrast. The project overview sits directly on the
native page background.

## Shapes

Keep the canonical particle mark's paths, clipping and complete silhouette.
The masthead has only a small corner radius. Navigation has no enclosing shape
or particle ornament; the particle geometry belongs to the identity mark.

## Components

- **Masthead:** canonical mark plus outlined PersonOS name, then centered benefit
  and subtitle. Light and dark filenames intentionally share the same palette.
- **Project badges:** compact linked images for live CI status, Python 3.10+,
  Apache-2.0 and alpha status. Keep each link pointed at its repository source.
- **Navigation:** four text-only SVG labels inside real HTML anchors, with image
  alt text and GitHub's native link interaction. Do not embed navigation targets
  inside the artwork.
- **Project overview:** ordinary Markdown paragraphs with a bold opening
  definition, followed by a short capability list. Keep detailed scenarios and
  code in the technical sections.

## Do's and Don'ts

- **Do** regenerate assets from `build_brand.py` and rasterize the canonical SVGs
  with librsvg, as described in [README.md](README.md).
- **Do** inspect both languages at desktop and narrow widths in light and dark
  themes, including navigation targets, wrapping and native CJK text.
- **Don't** redraw or randomly resample the particle mark.
- **Don't** add decorated button shapes, accent colors or shadows to the generated
  masthead and navigation.
- **Don't** use tutorial figures or unverified benchmark claims as the project
  introduction.
