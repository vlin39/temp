# Parallel Reader

A small toolkit for turning paragraph-aligned translations of the same book into
study material for language learning. From one JSON file of aligned paragraphs it
builds:

- **`reader.html`** — an interactive side-by-side reader. One column per language,
  one row per paragraph. Toggle any language on/off, toggle phonetics, and hover a
  paragraph to highlight the *same* paragraph across every language (click to pin).
- **`reader.epub`** — a paragraph-aligned multilingual e-book with embedded fonts,
  for reading on a phone or e-reader.

Phonetics are generated automatically:

| Language | Reading shown above each unit |
|----------|-------------------------------|
| `zh` Chinese | pinyin (per character) |
| `fr` French  | IPA (per word) |
| `it` Italian | IPA (per word) |
| `de` German  | IPA (per word) |
| `en` English | — (plain) |

Translation is *not* done here — you supply the aligned text yourself.

## Setup

```bash
pip install -r requirements.txt        # python deps
# IPA needs the espeak-ng binary on PATH:
#   macOS:  brew install espeak-ng
#   Debian: sudo apt-get install espeak-ng
bash fonts/fetch_fonts.sh              # (optional) fonts to embed in the EPUB
```

## Build

```bash
make html          # -> output/reader.html
make epub          # -> output/reader.epub
make all

# or directly, e.g. an EPUB with just English + Chinese:
python3 src/build_epub.py data/percy-jackson.json -l en zh -o output/en-zh.epub
```

## Adding text

Edit `data/percy-jackson.json`. Each entry in `paragraphs` is one aligned
paragraph, keyed by language code. Keep the paragraph order identical across
languages — that alignment is what powers the row highlighting and the EPUB.

```json
{
  "en": "…", "zh": "…", "fr": "…", "it": "…", "de": "…"
}
```

To add a language: add its code to `languages`, a display name to
`language_names`, and its text to every paragraph. If it needs IPA, add its
espeak-ng voice code to `ESPEAK_VOICE` in `src/phonetics.py`.

## Layout

```
data/         aligned source text (JSON)
src/
  phonetics.py    pinyin + IPA annotation (the shared core)
  build_html.py   interactive HTML reader
  build_epub.py   multilingual EPUB
fonts/        fonts to embed in the EPUB (fetched, not committed)
output/       generated reader.html / reader.epub
```
