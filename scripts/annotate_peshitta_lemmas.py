#!/usr/bin/env python3
"""
Annotate the parsed Peshitta NT JSONL with SEDRA-3 lemmas, glosses, and parse codes.

Inputs:
  data/processed/parallel_corpus/peshitta_nt.jsonl       — verse-level Peshitta from TEI
  data/external/sedra/peshitta_list.txt                  — SEDRA-3 word-level data

Output:
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl
    {"book", "chapter", "verse", "syriac", "tokens": [
       {"form", "pointed", "lemma", "gloss", "parse"}, ...]}

The SEDRA reference encoding is BB CC VVV WW (book/chapter/verse/word).
We attach lemmas at the verse level by aggregating SEDRA records whose ref
matches our (book, chapter, verse) tuple.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEDRA_TXT = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
PESHITTA_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt.jsonl"
PESHITTA_OUT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"

# SEDRA book code → Peshitta NT manifest book_code
SEDRA_TO_MANIFEST = {
    "52": "Matt", "53": "Mark", "54": "Luke", "55": "John", "56": "Acts",
    "57": "Rom", "58": "1Cor", "59": "2Cor", "60": "Gal", "61": "Eph",
    "62": "Phil", "63": "Col", "64": "1Thess", "65": "2Thess",
    "66": "1Tim", "67": "2Tim", "68": "Titus", "69": "Phlm", "70": "Heb",
    "71": "Jas", "72": "1Pet", "73": "2Pet", "74": "1John", "75": "2John",
    "76": "3John", "77": "Jude", "78": "Rev",
}


def load_sedra_index():
    """Build {(book_code, chapter, verse): [token_dicts...]} from peshitta_list.txt."""
    index: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    with SEDRA_TXT.open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            ref = fields[0]
            if len(ref) != 9:
                continue
            sedra_bk, ch_s, vs_s, w_s = ref[:2], ref[2:4], ref[4:7], ref[7:9]
            book_code = SEDRA_TO_MANIFEST.get(sedra_bk)
            if not book_code:
                continue
            try:
                chapter = int(ch_s)
                verse = int(vs_s)
                word_index = int(w_s)
            except ValueError:
                continue

            unpointed, pointed, lemma, gloss = fields[1:5]
            parse = fields[5] if len(fields) >= 6 else ""

            index[(book_code, chapter, verse)].append({
                "word_index": word_index,
                "form": unpointed,
                "pointed": pointed,
                "lemma": lemma,
                "gloss": gloss,
                "parse": parse,
            })
    # Order tokens by word index within each verse
    for k in index:
        index[k].sort(key=lambda t: t["word_index"])
    return index


def main():
    if not SEDRA_TXT.exists():
        sys.exit(f"Missing: {SEDRA_TXT}\nRun: bash scripts/fetch_data.sh")
    if not PESHITTA_IN.exists():
        sys.exit(f"Missing: {PESHITTA_IN}\nRun: python scripts/parse_peshitta_tei.py")

    print(f"Indexing SEDRA-3 word data from {SEDRA_TXT}…")
    sedra = load_sedra_index()
    print(f"  {sum(len(v) for v in sedra.values())} word records across "
          f"{len(sedra)} verses.")

    matched = unmatched = 0
    book_match: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    PESHITTA_OUT.parent.mkdir(parents=True, exist_ok=True)

    with PESHITTA_IN.open(encoding="utf-8") as fin, \
         PESHITTA_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            key = (rec["book"], rec["chapter"], rec["verse"])
            tokens = sedra.get(key, [])
            if tokens:
                matched += 1
                m, u = book_match[rec["book"]]
                book_match[rec["book"]] = (m + 1, u)
            else:
                unmatched += 1
                m, u = book_match[rec["book"]]
                book_match[rec["book"]] = (m, u + 1)
            rec["tokens"] = [
                {"form": t["form"], "pointed": t["pointed"], "lemma": t["lemma"],
                 "gloss": t["gloss"], "parse": t["parse"]}
                for t in tokens
            ]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = matched + unmatched
    print(f"\nVerse-level match rate: {matched}/{total} = {100*matched/total:.1f}%")
    print(f"\nPer-book:")
    for bk in sorted(book_match.keys()):
        m, u = book_match[bk]
        tot = m + u
        rate = 100 * m / tot if tot else 0.0
        print(f"  {bk:8s}  {m:>5d} matched / {tot:>5d} total  ({rate:5.1f}%)")
    print(f"\nWrote {PESHITTA_OUT}")


if __name__ == "__main__":
    main()
