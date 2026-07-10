# Fonts

These are embedded (subset) into the EPUB so Chinese renders in Kai and IPA
renders correctly on any device. They are **not committed** — run
`bash fonts/fetch_fonts.sh` to download them.

- `kai.ttf` — LXGW WenKai (霞鹜文楷), SIL OFL 1.1. Chinese characters.
- `CharisSIL.ttf` — Charis SIL, SIL OFL 1.1. IPA readings (optional).

Map which font plays which role in `FONT_ROLES` at the top of
`src/build_epub.py`. Any `.ttf` you drop here can be wired up the same way.
