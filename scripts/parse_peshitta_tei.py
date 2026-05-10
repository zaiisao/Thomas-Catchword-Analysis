#!/usr/bin/env python3
"""
Parse Peshitta NT TEI/XML files from the Digital Syriac Corpus into a normalized JSONL.

Input:  data/raw/peshitta_repo/data/tei/{100,119,120,121,...}.xml
        data/raw/peshitta/peshitta_nt_manifest.csv
Output: data/processed/parallel_corpus/peshitta_nt.jsonl
        One record per verse:
          {"book": "Matt", "chapter": 1, "verse": 1, "syriac": "..."}

Run:
  python scripts/parse_peshitta_tei.py --gospels-only
  python scripts/parse_peshitta_tei.py            # full NT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

from lxml import etree

TEI_NS = "{http://www.tei-c.org/ns/1.0}"
REPO_ROOT = Path(__file__).resolve().parent.parent
TEI_DIR = REPO_ROOT / "data" / "raw" / "peshitta_repo" / "data" / "tei"
MANIFEST = REPO_ROOT / "data" / "raw" / "peshitta" / "peshitta_nt_manifest.csv"
OUTPUT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt.jsonl"


def strip_vocalization(text: str) -> str:
    """Strip Syriac diacritical marks, leaving the consonantal skeleton."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def parse_book(xml_path: Path, book_code: str):
    tree = etree.parse(str(xml_path))
    body = tree.find(f".//{TEI_NS}body")
    if body is None:
        return

    for chapter_div in body.findall(f"{TEI_NS}div[@type='chapter']"):
        chapter_n = chapter_div.get("n")
        if chapter_n is None:
            continue
        try:
            chapter_int = int(chapter_n)
        except ValueError:
            continue

        for verse in chapter_div.findall(f"{TEI_NS}ab[@type='verse']"):
            verse_n = verse.get("n")
            text = "".join(verse.itertext()).strip()
            text = " ".join(text.split())
            if not text or not verse_n:
                continue
            try:
                verse_int = int(verse_n)
            except ValueError:
                continue

            yield {
                "book": book_code,
                "chapter": chapter_int,
                "verse": verse_int,
                "syriac": text,
                "syriac_consonantal": strip_vocalization(text),
            }


def load_manifest(gospels_only: bool):
    rows = []
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            if gospels_only and row["is_gospel"] != "1":
                continue
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gospels-only", action="store_true", help="Parse only the four Gospels")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.gospels_only)
    if not rows:
        sys.exit(f"No books selected from manifest: {MANIFEST}")

    n_verses = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            xml_path = TEI_DIR / row["corpus_file"]
            if not xml_path.exists():
                print(f"  skip (missing): {xml_path}", file=sys.stderr)
                continue
            book_n = 0
            for record in parse_book(xml_path, row["book_code"]):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                book_n += 1
            print(f"  {row['book_code']:8s} {book_n:>5d} verses  ({row['corpus_file']})")
            n_verses += book_n

    print(f"\nWrote {n_verses} verses -> {args.output}")


if __name__ == "__main__":
    main()
