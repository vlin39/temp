# -*- coding: utf-8 -*-
"""
Phonetic annotation for the parallel reader.

  annotate(text, lang) -> HTML string where each unit is wrapped in <ruby>,
  with the phonetic reading in the <rt>:

    zh  ->  per-character pinyin  (pypinyin, tone marks)
    fr  ->  per-word IPA          (espeak-ng)
    it  ->  per-word IPA          (espeak-ng)
    de  ->  per-word IPA          (espeak-ng)
    en  ->  no annotation (plain, escaped text)

Phonetics can be shown/hidden purely in CSS by toggling <rt> visibility, so we
always emit the ruby markup and let the reader decide what to display.

Requires: pypinyin, jieba (Chinese word segmentation), and the `espeak-ng`
binary on PATH for IPA. Install espeak-ng with your package manager, e.g.
`apt-get install espeak-ng` or `brew install espeak-ng`.
"""

import re
import html
import shutil
import functools
import subprocess

from pypinyin import pinyin, Style
import jieba

HANZI = re.compile(r"[\u4e00-\u9fff]")
# A "word" for Latin scripts: letters incl. accents, plus internal apostrophes
LATIN_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'\u2019]+")

ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")
ESPEAK_VOICE = {"fr": "fr", "it": "it", "de": "de"}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# --- IPA (espeak-ng) --------------------------------------------------------
@functools.lru_cache(maxsize=50000)
def ipa_word(word: str, lang: str) -> str:
    """IPA for a single word via espeak-ng. Cached. Returns '' on failure."""
    if not ESPEAK:
        return ""
    voice = ESPEAK_VOICE.get(lang)
    if not voice:
        return ""
    try:
        r = subprocess.run(
            [ESPEAK, "-q", "--ipa", "-v", voice, word],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return ""
    ipa = r.stdout.strip().replace("\n", " ")
    # espeak marks some liaisons/word-links with a trailing hyphen; drop edges
    return ipa.strip(" -\u200d")


def ruby(base: str, top: str) -> str:
    if top:
        return f"<ruby>{esc(base)}<rt>{esc(top)}</rt></ruby>"
    return esc(base)


# --- Per-language annotation ------------------------------------------------
def annotate_zh(text: str) -> str:
    out = []
    for word in jieba.cut(text, HMM=True):
        if not any(HANZI.match(c) for c in word):
            out.append(f'<span class="punct">{esc(word)}</span>')
            continue
        syls = pinyin(word, style=Style.TONE, heteronym=False)
        chars = []
        for i, ch in enumerate(word):
            top = syls[i][0] if i < len(syls) and syls[i] else ""
            chars.append(ruby(ch, top))
        out.append(f'<span class="word">{"".join(chars)}</span>')
    return "".join(out)


def annotate_latin(text: str, lang: str) -> str:
    out = []
    pos = 0
    for m in LATIN_WORD.finditer(text):
        if m.start() > pos:
            out.append(f'<span class="punct">{esc(text[pos:m.start()])}</span>')
        w = m.group(0)
        out.append(f'<span class="word">{ruby(w, ipa_word(w, lang))}</span>')
        pos = m.end()
    if pos < len(text):
        out.append(f'<span class="punct">{esc(text[pos:])}</span>')
    return "".join(out)


def annotate(text: str, lang: str) -> str:
    if lang == "zh":
        return annotate_zh(text)
    if lang in ESPEAK_VOICE:
        return annotate_latin(text, lang)
    # English or anything without a phonetic scheme
    return esc(text)


if __name__ == "__main__":
    for lang, s in [
        ("zh", "我不愿意当一个混血者。"),
        ("fr", "je n'ai jamais souhaité"),
        ("it", "chiudetelo all'istante"),
        ("de", "ich hab nicht darum gebeten"),
    ]:
        print(lang, "->", annotate(s, lang)[:200])
