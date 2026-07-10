# -*- coding: utf-8 -*-
"""
Build a multilingual EPUB from plain-text input (one .txt file per language,
paragraphs separated by blank lines) or from a single aligned JSON file.

EPUB is reflowable, so reliable "parallel" reading across many languages is done
by *paragraph alignment*: each paragraph becomes one section, and inside it every
selected language is stacked in order with a label and its phonetics (pinyin for
zh, IPA for fr/it/de). One screen = one paragraph in all chosen languages.

Fonts placed in ./fonts are subset to the glyphs used and embedded, so Chinese
renders in Kai and IPA renders correctly even on e-readers without those fonts.
Map which font serves which role in FONT_ROLES below.

Usage:
    python3 src/build_epub.py data/percy-jackson.*.txt -o output/reader.epub
    python3 src/build_epub.py data/percy-jackson.*.txt -l en zh  # subset of langs
    python3 src/build_epub.py data/percy-jackson.json  -o output/reader.epub
"""

import io
import sys
import argparse
from html import escape
from pathlib import Path

from ebooklib import epub
from fontTools import subset
from fontTools.ttLib import TTFont

import load_input
from phonetics import annotate

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"

# role -> (filename in ./fonts, css font-family name). Add IPA font (e.g. Charis
# SIL) as "CharisSIL.ttf" and it will be embedded and used for the IPA readings.
FONT_ROLES = {
    "han": ("kai.ttf", "ReaderHan"),      # Chinese characters (a Kai/Kaiti font)
    "ipa": ("CharisSIL.ttf", "ReaderIPA"),  # IPA readings (optional)
}


def subset_font(path: Path, charset: str) -> bytes:
    ss = subset.Subsetter(options=subset.Options(layout_features="*", glyph_names=False))
    font = TTFont(str(path))
    ss.populate(text=charset)
    ss.subset(font)
    buf = io.BytesIO(); font.save(buf)
    return buf.getvalue()


def build(data: dict, langs, out_path: str):
    names = data.get("language_names", {})
    paras = data["paragraphs"]
    ipa_langs = {"fr", "it", "de"}

    # gather everything we might render, for font subsetting
    all_text = data.get("title", "") + "".join(names.get(l, l) for l in langs)
    for p in paras:
        for l in langs:
            all_text += p.get(l, "")
    # add pinyin + IPA characters by rendering annotations once
    ann_blob = "".join(annotate(p.get(l, ""), l) for p in paras for l in langs)
    all_text += ann_blob

    book = epub.EpubBook()
    book.set_identifier("parallel-reader-001")
    book.set_title(data.get("title", "Parallel Reader"))
    book.set_language(langs[0] if langs else "en")
    if data.get("author"):
        book.add_author(data["author"])

    # --- fonts ---
    font_css = []
    for role, (fname, family) in FONT_ROLES.items():
        fpath = FONTS_DIR / fname
        if not fpath.exists():
            continue
        content = subset_font(fpath, all_text)
        book.add_item(epub.EpubItem(uid=f"font-{role}", file_name=f"fonts/{fname}",
                                    media_type="application/vnd.ms-opentype",
                                    content=content))
        font_css.append(f'@font-face{{font-family:"{family}";src:url("../fonts/{fname}");}}')

    han = FONT_ROLES["han"][1]
    ipa = FONT_ROLES["ipa"][1]
    css = "\n".join(font_css) + f"""
html,body{{margin:0;padding:0}}
body{{font-family:"Iowan Old Style",Georgia,serif;line-height:1.5;color:#111}}
.para{{padding:1.1em 1.1em}}
.block{{margin:0 0 1.3em;padding-bottom:1.0em;border-bottom:1px solid #e6e0d4}}
.block:last-child{{border-bottom:none}}
.lbl{{font-size:.62em;letter-spacing:.22em;text-transform:uppercase;color:#8a8a8a;
  font-family:sans-serif;margin:0 0 .35em}}
.txt{{font-size:1.02em;line-height:1.65}}
.txt.zh{{font-family:"{han}","Kaiti SC","STKaiti",serif;font-size:1.35em;line-height:2.25}}
ruby{{ruby-align:center}}
rt{{font-size:.5em;font-weight:normal;line-height:1;
  font-family:"{ipa}","Charis SIL","Gentium Plus",sans-serif;color:#666}}
.zh rt{{font-family:sans-serif;font-size:.42em;color:#7a4a33}}
.word{{margin-right:.14em}} .punct{{margin:0}}
.title-wrap{{text-align:center;padding:3.5em 1.2em}}
.title-wrap h1{{font-weight:normal;font-size:1.6em;margin:0 0 .4em}}
.title-wrap .tag{{margin-top:2em;color:#999;font-size:.78em;font-family:sans-serif;
  letter-spacing:.16em;text-transform:uppercase}}
"""
    book.add_item(epub.EpubItem(uid="css", file_name="style/main.css",
                                media_type="text/css", content=css))

    def page(title, lang, body):
        # ebooklib regenerates each document's <head> from the item's title and
        # links, so the stylesheet must be attached with add_link (see stylize)
        return (f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}" lang="{lang}">'
                f'<head><meta charset="utf-8"/><title>{escape(title, quote=False)}</title></head>'
                f'<body>{body}</body></html>')

    def stylize(item):
        item.add_link(href="../style/main.css", rel="stylesheet", type="text/css")

    # title page
    tp = epub.EpubHtml(title="Title", file_name="text/title.xhtml", lang="en")
    stylize(tp)
    title = escape(data.get("title", ""), quote=False)
    tag = escape(" · ".join(names.get(l, l) for l in langs), quote=False)
    tp.content = page(data.get("title", ""), "en",
        f'<div class="title-wrap"><h1>{title}</h1>'
        f'<div class="tag">{tag}</div></div>')
    book.add_item(tp)

    spine = [tp]
    toc = []
    for i, p in enumerate(paras, 1):
        blocks = []
        for l in langs:
            cls = "zh" if l == "zh" else ("ipa-lang" if l in ipa_langs else "en")
            blocks.append(
                f'<div class="block" lang="{l}" xml:lang="{l}">'
                f'<p class="lbl">{escape(names.get(l, l), quote=False)}</p>'
                f'<p class="txt {cls}">{annotate(p.get(l, ""), l)}</p></div>')
        pg = epub.EpubHtml(title=f"¶ {i}", file_name=f"text/p{i:03d}.xhtml", lang=langs[0])
        stylize(pg)
        pg.content = page(f"Paragraph {i}", langs[0], f'<div class="para">{"".join(blocks)}</div>')
        book.add_item(pg)
        spine.append(pg)
        toc.append(pg)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + spine

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(out_path, book)
    print("wrote", out_path, "| languages:", ",".join(langs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+",
                    help="one .txt file per language (paragraphs separated by "
                         "blank lines), or a single aligned .json file")
    ap.add_argument("-o", "--out", default="output/reader.epub")
    ap.add_argument("-l", "--langs", nargs="+", default=None,
                    help="subset/order of languages (default: all in input)")
    args = ap.parse_args()

    data = load_input.load(args.data)
    langs = args.langs or data["languages"]
    unknown = [l for l in langs if l not in data["languages"]]
    if unknown:
        sys.exit("language(s) not in the input: " + ", ".join(unknown)
                 + " (have: " + ", ".join(data["languages"]) + ")")
    build(data, langs, args.out)


if __name__ == "__main__":
    main()
