# Project context for Claude Code

## What this is
A generator for language-learning study material. Input: one JSON file of the
same book's paragraphs aligned across languages. Output: an interactive HTML
parallel reader and a multilingual EPUB. Phonetics (pinyin for Chinese, IPA for
French/Italian/German) are generated at build time.

The user supplies the translations; this repo does **not** translate. Only the
text the user provides should live in `data/` — do not fetch or paste additional
book text.

## Architecture
- `src/phonetics.py` is the shared core. `annotate(text, lang)` returns HTML where
  each unit is a `<ruby>` with its reading in `<rt>`. Both builders call it, so any
  improvement to readings benefits HTML and EPUB at once.
  - `zh`: `jieba` segments words, `pypinyin` gives per-character tone-marked pinyin.
  - `fr/it/de`: tokenized into words, each phonemized to IPA via the `espeak-ng`
    binary (subprocess, `lru_cache`d).
- `src/build_html.py`: renders a `<table>` — columns = languages, rows =
  paragraphs. Alignment is structural (a `<tr>` is one paragraph in every
  language), so `tr:hover` highlights across all columns for free; JS adds
  click-to-pin. Column/phonetics visibility is pure CSS via `body.hide-*` classes.
- `src/build_epub.py`: paragraph-aligned stacking (one section per paragraph, all
  languages inside). Fonts in `fonts/` are subset with `fonttools` and embedded.

## Conventions
- Language codes are ISO 639-1 (`en`, `zh`, `fr`, `it`, `de`).
- Paragraph order MUST match across languages — it is the alignment key.
- Keep `phonetics.py` free of HTML/EPUB specifics beyond the ruby markup so both
  builders stay in sync.

## Build / verify
```bash
make all
python3 -c "import ebooklib"                 # deps present?
# quick EPUB structure check:
python3 -m zipfile -l output/reader.epub
```
`espeak-ng` must be installed for IPA; without it, `annotate` degrades to plain
text for fr/it/de (no crash).

## Good next steps (roadmap)
- **Click-to-lookup**: click a word in the HTML reader to show a gloss/dictionary
  entry (CC-CEDICT for zh; Wiktionary dumps for fr/it/de).
- **Sentence-level alignment** inside a paragraph for tighter side-by-side.
- **Facing-pages EPUB** for a chosen language *pair* (page-break-per-language) as
  an alternative to stacking — nice on tablets in two-up view.
- **Tone-sandhi option** for pinyin, and a switch between IPA broad/narrow.
- **Per-user saved state** in the HTML (which languages/phonetics are on).
- **Import helper** that takes plain per-language text files and aligns by
  paragraph into the JSON schema.

## Watch out for
- IPA fonts: many e-readers lack full IPA coverage. Embedding Charis SIL
  (`fonts/CharisSIL.ttf`) fixes it; the EPUB builder already wires it up if present.
- espeak-ng IPA is rule-based and imperfect (esp. French liaison); treat readings
  as a study aid, not ground truth. A dictionary-based backend could replace it.
