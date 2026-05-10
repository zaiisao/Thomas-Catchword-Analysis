#!/usr/bin/env python3
"""
Build a lemma → root mapping using the SEDRA-3 ROOTS.TXT + LEXEMES.TXT
data, joined to our existing peshitta_list.txt lemmas.

This unblocks etymological catchword detection (Perrin's link type where
two different lemmas share a triliteral root).

Inputs:
  data/external/sedra/ROOTS.TXT        — root_id → consonantal-root string
  data/external/sedra/LEXEMES.TXT      — lexeme_id → root_id + lemma string
  data/external/sedra/peshitta_list.txt — our existing per-word data

Output:
  data/processed/syriac_lemma_to_root.json
    {
      "ܢܘܪܐ": {"root_id": "0:N", "root": "ܢܘܪ",  "transliteration": "NWR"},
      "ܢܘܗܪܐ": {"root_id": "0:M", "root": "ܢܗܪ",  "transliteration": "NHR"},
      ...
    }

Usage:
  python scripts/build_sedra_root_map.py
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEDRA_DIR = REPO_ROOT / "data" / "external" / "sedra"
ROOTS_TXT = SEDRA_DIR / "ROOTS.TXT"
LEXEMES_TXT = SEDRA_DIR / "LEXEMES.TXT"
PESH_LIST = SEDRA_DIR / "peshitta_list.txt"
OUT = REPO_ROOT / "data" / "processed" / "syriac_lemma_to_root.json"

# SEDRA-3 ASCII transliteration → Syriac Unicode (consonantal letters only)
SEDRA_CONSONANTS = {
    "A": "ܐ", "B": "ܒ", "G": "ܓ", "D": "ܕ", "H": "ܗ",
    "O": "ܘ", "Z": "ܙ", "K": "ܚ", "Y": "ܛ", ";": "ܝ",
    "C": "ܟ", "L": "ܠ", "M": "ܡ", "N": "ܢ", "S": "ܣ",
    "E": "ܥ", "I": "ܦ", "/": "ܨ", "X": "ܩ", "R": "ܪ",
    "W": "ܫ", "T": "ܬ",
}
# Vowels and diacritics are stripped before transliteration.
VOWELS_AND_DIA = set("aoeiu',_*")


def sedra_to_syriac(s: str) -> str:
    """Map SEDRA-3 ASCII root/lemma string to consonantal Syriac Unicode.
    Vowels (lowercase a o e i u) and diacritics (' , _ *) are stripped.
    """
    out = []
    for c in s:
        if c in VOWELS_AND_DIA or c.isspace():
            continue
        u = SEDRA_CONSONANTS.get(c)
        if u is None:
            # Unknown character — skip (some roots have markers like |, 0)
            continue
        out.append(u)
    return "".join(out)


# ROOTS.TXT format: 0:N,"ROOT","other_field|flag",num
ROOT_LINE_RE = re.compile(r'^(\d+:\d+),"([^"]*)",.*')

# LEXEMES.TXT format: 1:N,0:M,"LEXEME",num,num
LEX_LINE_RE = re.compile(r'^(\d+:\d+),(\d+:\d+),"([^"]*)",.*')


def load_roots() -> dict[str, dict]:
    """Return root_id (e.g. '0:1') → {transliteration, syriac}."""
    roots = {}
    with ROOTS_TXT.open(encoding="utf-8") as f:
        for line in f:
            m = ROOT_LINE_RE.match(line)
            if not m:
                continue
            root_id = m.group(1)
            translit = m.group(2)
            roots[root_id] = {
                "transliteration": translit,
                "syriac": sedra_to_syriac(translit),
            }
    return roots


def load_lexemes() -> list[dict]:
    """Return list of {lex_id, root_id, transliteration, syriac}."""
    lexemes = []
    with LEXEMES_TXT.open(encoding="utf-8") as f:
        for line in f:
            m = LEX_LINE_RE.match(line)
            if not m:
                continue
            lex_id = m.group(1)
            root_id = m.group(2)
            translit = m.group(3)
            lexemes.append({
                "lex_id": lex_id,
                "root_id": root_id,
                "transliteration": translit,
                "syriac": sedra_to_syriac(translit),
            })
    return lexemes


def load_peshitta_lemmas() -> set[str]:
    """All distinct lemma strings (consonantal Syriac) from our SEDRA peshitta_list export."""
    lemmas = set()
    with PESH_LIST.open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                lemmas.add(fields[3])
    return lemmas


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def main():
    print("Loading SEDRA ROOTS.TXT…")
    roots = load_roots()
    print(f"  {len(roots)} roots")

    print("Loading SEDRA LEXEMES.TXT…")
    lexemes = load_lexemes()
    print(f"  {len(lexemes)} lexemes")

    print("Loading peshitta_list.txt lemmas…")
    pesh_lemmas = load_peshitta_lemmas()
    pesh_consonantal = {strip_marks(l): l for l in pesh_lemmas}
    print(f"  {len(pesh_lemmas)} distinct lemmas")

    # Build lemma → root by matching consonantal Syriac strings
    lemma_to_root: dict[str, dict] = {}
    matched = 0
    for lex in lexemes:
        lex_syr = lex["syriac"]
        # Match by consonantal-only form (peshitta lemmas already are
        # consonantal but may have an extra final ܐ that the lex transliteration
        # also has — both are stripped of vowels)
        if lex_syr in pesh_consonantal:
            lemma_unicode = pesh_consonantal[lex_syr]
            root_info = roots.get(lex["root_id"], {})
            if lemma_unicode not in lemma_to_root:
                lemma_to_root[lemma_unicode] = {
                    "root_id": lex["root_id"],
                    "root_syriac": root_info.get("syriac", ""),
                    "root_transliteration": root_info.get("transliteration", ""),
                    "lex_transliteration": lex["transliteration"],
                }
                matched += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(lemma_to_root, f, ensure_ascii=False, indent=2)

    print(f"\nMatched {matched} lemmas to roots out of {len(pesh_lemmas)} "
          f"({100*matched/len(pesh_lemmas):.1f}%)")
    print(f"Saved {OUT}")

    # Verify Perrin's specific examples
    print("\nVerifying Perrin's lexemes:")
    test_lemmas = ["ܢܘܪܐ", "ܢܘܗܪܐ", "ܐܬܪܐ", "ܥܘܬܪܐ", "ܐܢܫ", "ܦܢܝ"]
    for lem in test_lemmas:
        info = lemma_to_root.get(lem)
        if info:
            print(f"  {lem} ({info['lex_transliteration']:>10s}) → root "
                  f"{info['root_syriac']!r} = {info['root_transliteration']!r}")
        else:
            print(f"  {lem}: NOT MATCHED")


if __name__ == "__main__":
    main()
