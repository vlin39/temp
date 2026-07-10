# -*- coding: utf-8 -*-
"""
Build a single-file interactive HTML parallel reader from an aligned JSON file.

Layout: one column per language, one row per paragraph. Because paragraphs are
aligned across translations, each table row is the "same" paragraph in every
language. Features:

  * Toggle any language column on/off.
  * Toggle phonetics (pinyin for zh, IPA for fr/it/de) on/off.
  * Hover a paragraph to highlight it across every visible language; click to pin.
  * Adjust text size.

Usage:
    python3 src/build_html.py data/percy-jackson.json -o output/reader.html
"""

import json
import argparse
from pathlib import Path

from phonetics import annotate

CSS = """
:root{
  --paper:#fbfaf7; --ink:#1b1a17; --soft:#6c665b; --rule:#e6e0d4;
  --hl:#fff3d0; --pin:#ffe49c; --accent:#3a5a78; --pinyin:#9a5b3d; --ipa:#3f6d5a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif;line-height:1.5}
.wrap{max-width:1500px;margin:0 auto;padding:28px 22px 120px}
header h1{font-size:22px;margin:0 0 2px;font-weight:600}
header .by{color:var(--soft);font-size:13px;margin:0 0 18px}

.controls{position:sticky;top:0;z-index:20;display:flex;gap:16px;flex-wrap:wrap;
  align-items:center;padding:12px 4px;margin-bottom:14px;
  background:linear-gradient(var(--paper) 80%,transparent);border-bottom:1px solid var(--rule)}
.group{display:flex;gap:6px;align-items:center}
.group .lbl{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--soft);margin-right:2px}
.chip{font:inherit;font-size:13px;padding:6px 12px;border-radius:999px;cursor:pointer;
  border:1px solid var(--rule);background:#fff;color:var(--ink);transition:all .12s ease}
.chip[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

table.reader{border-collapse:collapse;width:100%;table-layout:fixed}
table.reader col{width:1fr}
th.lang{position:sticky;top:56px;background:var(--paper);text-align:left;
  font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);
  padding:8px 14px;border-bottom:1px solid var(--rule);z-index:5}
td{vertical-align:top;padding:14px 14px;border-bottom:1px solid var(--rule)}
tbody tr{transition:background .1s ease}
tbody tr:hover{background:var(--hl)}
tbody tr.pinned{background:var(--pin)}
td .num{display:none}

/* text cells */
.cell{font-size:var(--fs,17px)}
.cell.zh{font-family:"Kaiti SC","STKaiti","KaiTi","LXGW WenKai","Songti SC",serif;
  font-size:calc(var(--fs,17px) * 1.28);line-height:2.15}
.cell.ipa-lang{line-height:2.0}
ruby{ruby-align:center;margin:0 .01em}
rt{font-size:.5em;font-weight:500;line-height:1.05;
  font-family:"Charis SIL","Gentium Plus","Doulos SIL",ui-sans-serif,system-ui,sans-serif;
  -webkit-user-select:none;user-select:none}
.zh rt{color:var(--pinyin);font-family:ui-sans-serif,"Helvetica Neue",Arial,sans-serif;font-size:.42em}
.ipa-lang rt{color:var(--ipa)}
.word{margin-right:.14em}
.punct{margin:0}

/* toggles */
body.hide-phon rt{display:none}
body.hide-en td.c-en, body.hide-en th.c-en{display:none}
body.hide-zh td.c-zh, body.hide-zh th.c-zh{display:none}
body.hide-fr td.c-fr, body.hide-fr th.c-fr{display:none}
body.hide-it td.c-it, body.hide-it th.c-it{display:none}
body.hide-de td.c-de, body.hide-de th.c-de{display:none}

.hint{color:var(--soft);font-size:12.5px;margin:10px 2px 0}
@media (max-width:700px){ .cell{--fs:15px} .wrap{padding:18px 12px 90px} }
"""

JS = """
const body=document.body;
document.querySelectorAll('.chip[data-lang]').forEach(c=>{
  c.addEventListener('click',()=>{
    const on=body.classList.toggle('hide-'+c.dataset.lang);
    c.setAttribute('aria-pressed',String(!on));
  });
});
const ph=document.getElementById('phon');
ph.addEventListener('click',()=>{
  const on=body.classList.toggle('hide-phon');
  ph.setAttribute('aria-pressed',String(!on));
});
document.querySelectorAll('.chip[data-fs]').forEach(c=>{
  c.addEventListener('click',()=>{
    document.documentElement.style.setProperty('--fs',c.dataset.fs+'px');
    document.querySelectorAll('.chip[data-fs]').forEach(x=>x.setAttribute('aria-pressed','false'));
    c.setAttribute('aria-pressed','true');
  });
});
// click a row to pin/unpin its highlight across all languages
document.querySelectorAll('tbody tr').forEach(tr=>{
  tr.addEventListener('click',e=>{
    if(window.getSelection().toString())return; // don't fight text selection
    tr.classList.toggle('pinned');
  });
});
"""


def build(data: dict) -> str:
    langs = data["languages"]
    names = data.get("language_names", {})
    paras = data["paragraphs"]
    ipa_langs = {"fr", "it", "de"}

    # header controls
    lang_chips = "".join(
        f'<button class="chip" data-lang="{l}" aria-pressed="true">{names.get(l, l)}</button>'
        for l in langs
    )
    fs_chips = "".join(
        f'<button class="chip" data-fs="{s}" aria-pressed="{"true" if s==17 else "false"}">{lbl}</button>'
        for s, lbl in [(15, "A-"), (17, "A"), (20, "A+")]
    )

    cols = "".join(f'<col class="c-{l}">' for l in langs)
    thead = "".join(f'<th class="lang c-{l}">{names.get(l, l)}</th>' for l in langs)

    rows = []
    for i, p in enumerate(paras):
        cells = []
        for l in langs:
            text = p.get(l, "")
            cls = "cell " + ("zh" if l == "zh" else ("ipa-lang" if l in ipa_langs else "en"))
            cells.append(f'<td class="c-{l}"><div class="{cls}">{annotate(text, l)}</div></td>')
        rows.append(f'<tr data-idx="{i}">{"".join(cells)}</tr>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{data.get('title','Parallel Reader')}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{data.get('title','Parallel Reader')}</h1>
    <p class="by">{data.get('author','')}</p>
  </header>

  <div class="controls">
    <div class="group"><span class="lbl">Languages</span>{lang_chips}</div>
    <div class="group"><span class="lbl">Show</span>
      <button class="chip" id="phon" aria-pressed="true">Phonetics</button>
    </div>
    <div class="group"><span class="lbl">Size</span>{fs_chips}</div>
  </div>

  <table class="reader"><colgroup>{cols}</colgroup>
    <thead><tr>{thead}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <p class="hint">Hover a paragraph to highlight it across every language &middot; click to pin &middot; toggle any language or the phonetics above.</p>
</div>
<script>{JS}</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="aligned JSON file")
    ap.add_argument("-o", "--out", default="output/reader.html")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(data), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
