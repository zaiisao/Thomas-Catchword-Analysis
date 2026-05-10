#!/usr/bin/env python3
"""
Parse Coptic SCRIPTORIUM TreeTagger (.tt) files for the Sahidica NT.

The Sahidica NT corpus (CopticScriptorium/corpora/sahidica.nt/sahidica.nt_TT.zip)
contains all 27 NT books fully annotated. The CoNLL-U release of the same corpus
is incomplete (Mark + 1 Cor only) — so this parser is the path to a complete
Sahidic NT for parallel alignment with the Peshitta.

Format: XML-like, rooted in <meta>. Each verse is a <verse_n verse_n="N" translation="...">
element containing <entity>, <norm_group>, and <norm xml:id pos lemma func ...>token</norm>
descendants. We flatten to a list of <norm> tokens per verse.

Output JSONL (one record per verse):
  {"book", "chapter", "verse", "text", "translation",
   "tokens": [{"id", "form", "lemma", "pos", "func", "norm"}, ...]}

Run from repo root:
  python scripts/parse_coptic_tt.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parent.parent
TT_DIR = REPO_ROOT / "data" / "raw" / "coptic_repo" / "sahidica.nt" / "sahidica.nt_TT"
OUTPUT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"

FILENAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)\.tt$")


def parse_tt_file(path: Path):
    """Yield one record per <verse_n>."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return
    _, book, chapter_s = m.group(1), m.group(2), m.group(3)
    chapter = int(chapter_s)

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(str(path), parser=parser)
    root = tree.getroot()
    if root is None:
        return

    for verse_n in root.iter("verse_n"):
        v_attr = verse_n.get("verse_n")
        if v_attr is None:
            continue
        try:
            verse = int(v_attr)
        except ValueError:
            continue
        translation = verse_n.get("translation", "")

        tokens = []
        forms = []
        for norm in verse_n.iter("norm"):
            form = (norm.text or "").strip()
            if not form:
                continue
            tokens.append({
                "id": norm.get("{http://www.w3.org/XML/1998/namespace}id") or norm.get("xml:id") or "",
                "form": form,
                "lemma": norm.get("lemma"),
                "pos": norm.get("pos"),
                "func": norm.get("func"),
                "norm": norm.get("norm"),
            })
            forms.append(form)

        yield {
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "text": " ".join(forms),
            "translation": translation,
            "tokens": tokens,
        }


def main():
    if not TT_DIR.exists():
        sys.exit(f"Missing: {TT_DIR}\nRun: bash scripts/fetch_data.sh && unzip ...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    book_stats: dict[str, dict] = {}
    n_verses_total = 0
    with OUTPUT.open("w", encoding="utf-8") as out:
        for path in sorted(TT_DIR.glob("*.tt")):
            n_verses_book = 0
            n_tokens_book = 0
            for rec in parse_tt_file(path):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_verses_book += 1
                n_tokens_book += len(rec["tokens"])
                book = rec["book"]
                stats = book_stats.setdefault(book, {"verses": 0, "tokens": 0, "chapters": set()})
                stats["verses"] += 1
                stats["tokens"] += len(rec["tokens"])
                stats["chapters"].add(rec["chapter"])
            n_verses_total += n_verses_book

    print(f"Per-book summary:")
    print(f"  {'BOOK':<18s} {'CHAPTERS':>9s} {'VERSES':>8s} {'TOKENS':>8s}")
    for book, stats in sorted(book_stats.items()):
        print(f"  {book:<18s} {len(stats['chapters']):>9d} "
              f"{stats['verses']:>8d} {stats['tokens']:>8d}")
    print(f"\n{n_verses_total} verses total -> {OUTPUT}")


if __name__ == "__main__":
    main()
