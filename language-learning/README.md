# Parallel Reader

A small toolkit for turning paragraph-aligned translations of the same book into
study material for language learning. From plain text files of the translations
(one `.txt` file per language) it builds:

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
python3 src/build_epub.py data/percy-jackson.*.txt -l en zh -o output/en-zh.epub
```

## Adding text

The input is one plain-text `.txt` file per language. Paragraphs are separated
by blank lines, and paragraph N must be the *same* paragraph in every file —
that alignment is what powers the row highlighting and the EPUB.

```
# title: Percy Jackson and the Lightning Thief
# author: Rick Riordan

Look, I didn’t want to be a half-blood.

If you’re reading this because you think you might be one, …
```

The language is taken from the file name (`percy-jackson.en.txt` or `en.txt`
→ `en`) or from an optional `# language: en` header line. Column order in the
reader follows the order the files are passed on the command line. The `# title:`,
`# author:` and `# language-name:` header lines are all optional; display
names for en/zh/fr/it/de are built in.

To add a language: add its `.txt` file to the build. If it needs IPA, add its
espeak-ng voice code to `ESPEAK_VOICE` in `src/phonetics.py`, and (optionally)
a display name to `DEFAULT_LANGUAGE_NAMES` in `src/load_input.py`.

Alternatively a single aligned JSON file (the internal schema) is accepted —
see `data/percy-jackson.json`.

## Layout

```
data/         aligned source text (one .txt per language; JSON also accepted)
src/
  load_input.py   input loading (.txt per language, or aligned JSON)
  phonetics.py    pinyin + IPA annotation (the shared core)
  build_html.py   interactive HTML reader
  build_epub.py   multilingual EPUB
fonts/        fonts to embed in the EPUB (fetched, not committed)
output/       generated reader.html / reader.epub
```
