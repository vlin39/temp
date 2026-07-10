#!/usr/bin/env bash
# Download fonts to embed in the EPUB. Both are free to embed (SIL OFL 1.1).
# Failures are non-fatal: the EPUB builder skips fonts that aren't present.
set -e
cd "$(dirname "$0")"

fetch() { # fetch <file> <label> <url>
  [ -f "$1" ] && return 0
  echo "Downloading $2…"
  # -f keeps HTTP error pages from being saved as a "font"; remove any partial
  # file on failure so the next run retries instead of embedding garbage
  curl -fL --retry 3 -o "$1" "$3" \
    || { rm -f "$1"; echo "  ($2 download failed — the EPUB will fall back to system fonts.)"; }
}

# LXGW WenKai (霞鹜文楷) — an open-source Kai/Kaiti font for the Chinese text.
fetch kai.ttf "LXGW WenKai (Kai font)" \
  "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKai-Regular.ttf"

# Charis SIL — excellent IPA coverage for the fr/it/de readings (optional).
# If this URL ever changes, grab the TTF from https://software.sil.org/charis/
fetch CharisSIL.ttf "Charis SIL (IPA font)" \
  "https://github.com/silnrsi/font-charis/releases/download/v7.000/CharisSIL-Regular.ttf"

echo "Done. Fonts in $(pwd)"
