#!/usr/bin/env python3
"""
Parse Coptic SCRIPTORIUM CoNLL-U files into a normalized JSONL with lemmas + glosses.

Two entry points:
  - parse_thomas:    one document, the Coptic Gospel of Thomas (Nag Hammadi II,2)
  - parse_sahidica:  Sahidica NT books (in this corpus only Mark + 1 Cor are
                     fully annotated; Matt/Luke/John are 2-byte placeholders)

Output JSONL records:
  {"book", "doc_id", "sent_id", "sent_index", "text", "text_en",
   "tokens": [{"id", "form", "lemma", "upos", "xpos", "feats"}, ...]}

Run:
  python scripts/parse_coptic_conllu.py thomas
  python scripts/parse_coptic_conllu.py sahidica
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
COPTIC_REPO = REPO_ROOT / "data" / "raw" / "coptic_repo"
THOMAS_FILE = COPTIC_REPO / "thomas-gospel" / "thomas.gospel_CONLLU" / "thomas_gospel.conllu"
SAHIDICA_DIR = COPTIC_REPO / "sahidica.nt" / "sahidica.nt_CONLLU"

OUT_DIR = REPO_ROOT / "data" / "processed"
THOMAS_OUT = OUT_DIR / "got_logia" / "thomas_coptic.jsonl"
SAHIDICA_OUT = OUT_DIR / "parallel_corpus" / "sahidica_nt_coptic.jsonl"

# Empty-stub threshold: real chapters are kilobytes; placeholders are 2 bytes.
MIN_REAL_BYTES = 100


def parse_conllu(path: Path) -> Iterator[dict]:
    """Yield one record per sentence."""
    sent_meta: dict = {}
    tokens: list[dict] = []
    sent_index = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if "=" in line:
                    key, _, val = line[1:].strip().partition("=")
                    sent_meta[key.strip()] = val.strip()
            elif line == "":
                if tokens:
                    yield {
                        "doc_id": sent_meta.get("newdoc id", ""),
                        "sent_id": sent_meta.get("sent_id", ""),
                        "sent_index": sent_index,
                        "text": sent_meta.get("text", ""),
                        "text_en": sent_meta.get("text_en", ""),
                        "tokens": tokens,
                    }
                    sent_index += 1
                    tokens = []
            else:
                fields = line.split("\t")
                if len(fields) < 6:
                    continue
                tok_id, form, lemma, upos, xpos, feats = fields[:6]
                # Skip multi-token range markers (e.g. "1-2"); keep only atomic tokens
                if "-" in tok_id or "." in tok_id:
                    continue
                tokens.append({
                    "id": tok_id,
                    "form": form,
                    "lemma": lemma if lemma != "_" else None,
                    "upos": upos if upos != "_" else None,
                    "xpos": xpos if xpos != "_" else None,
                    "feats": feats if feats != "_" else None,
                })
        if tokens:
            yield {
                "doc_id": sent_meta.get("newdoc id", ""),
                "sent_id": sent_meta.get("sent_id", ""),
                "sent_index": sent_index,
                "text": sent_meta.get("text", ""),
                "text_en": sent_meta.get("text_en", ""),
                "tokens": tokens,
            }


def parse_thomas():
    THOMAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not THOMAS_FILE.exists():
        sys.exit(f"Missing: {THOMAS_FILE}")
    n_sents = n_tokens = 0
    with THOMAS_OUT.open("w", encoding="utf-8") as out:
        for rec in parse_conllu(THOMAS_FILE):
            rec["book"] = "GoT"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_sents += 1
            n_tokens += len(rec["tokens"])
    print(f"Coptic Thomas: {n_sents} sentences, {n_tokens} tokens -> {THOMAS_OUT}")


def parse_sahidica():
    SAHIDICA_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not SAHIDICA_DIR.exists():
        sys.exit(f"Missing: {SAHIDICA_DIR}")

    book_stats: dict[str, tuple[int, int]] = {}
    n_files_real = n_files_empty = 0
    with SAHIDICA_OUT.open("w", encoding="utf-8") as out:
        for path in sorted(SAHIDICA_DIR.glob("*.conllu")):
            if path.stat().st_size < MIN_REAL_BYTES:
                n_files_empty += 1
                continue
            n_files_real += 1
            # Filename pattern: 41_Mark_03.conllu  → book=Mark, chapter=03
            stem = path.stem  # e.g. "41_Mark_03"
            parts = stem.split("_")
            if len(parts) >= 3:
                book = "_".join(parts[1:-1])
                try:
                    chapter = int(parts[-1])
                except ValueError:
                    chapter = -1
            else:
                book, chapter = stem, -1

            for rec in parse_conllu(path):
                rec["book"] = book
                rec["chapter"] = chapter
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                s, t = book_stats.get(book, (0, 0))
                book_stats[book] = (s + 1, t + len(rec["tokens"]))

    print(f"Real annotated files:    {n_files_real}")
    print(f"Empty placeholder files: {n_files_empty}")
    for book, (sents, toks) in sorted(book_stats.items()):
        print(f"  {book:18s} {sents:>5d} sents  {toks:>6d} tokens")
    print(f"\nWrote -> {SAHIDICA_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["thomas", "sahidica"])
    args = ap.parse_args()
    if args.target == "thomas":
        parse_thomas()
    else:
        parse_sahidica()


if __name__ == "__main__":
    main()
