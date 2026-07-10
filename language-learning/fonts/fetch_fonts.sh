#!/usr/bin/env bash
# Download fonts to embed in the EPUB. Both are free to embed (SIL OFL 1.1).
set -e
cd "$(dirname "$0")"

# LXGW WenKai (霞鹜文楷) — an open-source Kai/Kaiti font for the Chinese text.
if [ ! -f kai.ttf ]; then
  echo "Downloading LXGW WenKai (Kai font)…"
  curl -L -o kai.ttf \
    "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKai-Regular.ttf"
fi

# Charis SIL — excellent IPA coverage for the fr/it/de readings (optional).
# If this URL ever changes, grab the TTF from https://software.sil.org/charis/
if [ ! -f CharisSIL.ttf ]; then
  echo "Downloading Charis SIL (IPA font)…"
  curl -L -o CharisSIL.ttf \
    "https://github.com/silnrsi/font-charis/releases/download/v7.000/CharisSIL-Regular.ttf" \
    || echo "  (Charis download failed — EPUB will fall back to a system IPA font.)"
fi

echo "Done. Fonts in $(pwd)"
