# -*- coding: utf-8 -*-
"""
Input loading for the parallel reader.

Two accepted input shapes, both normalized to the dict the builders consume
({title, author, languages, language_names, paragraphs}):

  * One or more plain-text (.txt) files, one per language — the usual case.
    Paragraphs are separated by blank lines; paragraph N in every file is the
    *same* paragraph (that alignment powers the row highlighting and the EPUB).
    The language comes from a `# language: xx` header line or from the file
    name (percy-jackson.en.txt / en.txt -> en). Optional `# title:`,
    `# author:` and `# language-name:` header lines fill in metadata; the
    first file that defines title/author wins.

  * A single .json file with the aligned schema (see data/percy-jackson.json).
"""

import json
import re
import sys
from pathlib import Path

DEFAULT_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "中文",
    "fr": "Français",
    "it": "Italiano",
    "de": "Deutsch",
}

_HEADER = re.compile(r"^#\s*(title|author|language|language[-_ ]name)\s*:\s*(.+?)\s*$", re.I)
_LANG_CODE = re.compile(r"^[a-z]{2,3}$")
# CJK ideographs, kana, and CJK punctuation/fullwidth forms — scripts written
# without spaces, so hard-wrapped lines must be joined without inserting one
_CJK = re.compile(r"[⺀-鿿豈-﫿　-〿＀-￯"
                  r"\U00020000-\U0003134f]")


def _join_wrapped(lines):
    out = lines[0]
    for l in lines[1:]:
        joint = "" if _CJK.search(out[-1:]) and _CJK.search(l[:1]) else " "
        out += joint + l
    return out


def _parse_txt(path: Path):
    """-> (meta dict, [paragraph, ...]) for one plain-text file."""
    meta = {}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    # header: a contiguous run of leading `# key: value` lines; the first
    # blank or non-header line ends it, so body text is never eaten as metadata
    while i < len(lines):
        m = _HEADER.match(lines[i])
        if not m:
            break
        meta[re.sub(r"[-_ ]", "-", m.group(1).lower())] = m.group(2)
        i += 1
    paras = []
    for block in re.split(r"\n\s*\n", "\n".join(lines[i:])):
        stripped = [l.strip() for l in block.splitlines() if l.strip()]
        if stripped:
            paras.append(_join_wrapped(stripped))
    return meta, paras


def _lang_of(path: Path, meta: dict) -> str:
    if "language" in meta:
        lang = meta["language"].lower()
        if not _LANG_CODE.match(lang):
            sys.exit(f"{path}: '# language: {meta['language']}' is not a "
                     f"2-3 letter language code")
        return lang
    cand = path.stem.rsplit(".", 1)[-1].lower()  # percy-jackson.en -> en, en -> en
    if _LANG_CODE.match(cand):
        return cand
    sys.exit(f"{path}: cannot tell the language — name the file like "
             f"book.en.txt or add a '# language: en' header line")


def load(paths) -> dict:
    """Load one aligned .json file, or one .txt file per language."""
    paths = [Path(p) for p in paths]
    exts = {p.suffix.lower() for p in paths}
    if exts == {".json"}:
        if len(paths) > 1:
            sys.exit("pass a single aligned .json file (or one .txt file per language)")
        try:
            return json.loads(paths[0].read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            sys.exit(f"{paths[0]}: invalid JSON — {e}")
    if exts != {".txt"}:
        sys.exit("expected .txt input, one file per language "
                 "(or a single aligned .json file)")

    langs, names, texts = [], {}, {}
    title = author = None
    for p in paths:
        meta, paras = _parse_txt(p)
        lang = _lang_of(p, meta)
        if lang in texts:
            sys.exit(f"{p}: language '{lang}' given twice")
        langs.append(lang)
        texts[lang] = paras
        names[lang] = meta.get("language-name", DEFAULT_LANGUAGE_NAMES.get(lang, lang))
        title = title or meta.get("title")
        author = author or meta.get("author")

    counts = {l: len(texts[l]) for l in langs}
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{p.name}: {counts[l]}" for p, l in zip(paths, langs))
        sys.exit("paragraph counts differ between languages — every file must have "
                 f"the same number of blank-line-separated paragraphs ({detail})")

    data = {
        "languages": langs,
        "language_names": names,
        "paragraphs": [{l: texts[l][i] for l in langs} for i in range(counts[langs[0]])],
    }
    if title:
        data["title"] = title
    if author:
        data["author"] = author
    return data
