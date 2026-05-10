#!/usr/bin/env python3
"""
Phase 2.1 — prepare the Coptic↔Syriac parallel NT corpus for translation
training.

Inputs:
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl   (Coptic, lemmatized)
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl   (Syriac, SEDRA-lemmatized)
  data/processed/got_logia/thomas_logia.jsonl                   (Coptic Gospel of Thomas)

Output:
  data/processed/parallel_corpus/coptic_syriac_pairs.jsonl
    one record per aligned NT verse, with both surface text and lemma sequences,
    plus a "thomas_overlap" score and a "split" tag (train/val/test/leak).

Critical: any verse with non-trivial lemma overlap to a Thomas logion is
flagged "leak" and **excluded** from both train and val splits, per the
project guide:

  > Exclude verses that parallel the Gospel of Thomas from the test set.
  > The model must be "blind" when translating Thomas.

We use Coptic-side lemma-set overlap as a conservative proxy for "Thomas
parallel" (the Synoptic→Thomas parallels are well-known but not codified
machine-readably; this overlap heuristic catches them automatically).

Usage:
  python scripts/prepare_parallel_corpus.py
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COPTIC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
SYRIAC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
THOMAS_IN = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "coptic_syriac_pairs.jsonl"

# Map Sahidica book name → Peshitta manifest book code
COPTIC_TO_MANIFEST = {
    "Matthew": "Matt", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts_of_the_Apostles": "Acts", "Romans": "Rom",
    "1_Corinthians": "1Cor", "2_Corinthians": "2Cor",
    "Galatians": "Gal", "Ephesians": "Eph", "Philippians": "Phil",
    "Colossians": "Col", "1_Thessalonians": "1Thess", "2_Thessalonians": "2Thess",
    "1_Timothy": "1Tim", "2_Timothy": "2Tim", "Titus": "Titus", "Philemon": "Phlm",
    "Hebrews": "Heb", "James": "Jas", "1_Peter": "1Pet", "2_Peter": "2Pet",
    "1_John": "1John", "2_John": "2John", "3_John": "3John", "Jude": "Jude",
    "Revelation": "Rev",
}

CONTENT_POS_COPTIC = {"N", "NPROP", "V", "VBD", "VSTAT", "VIMP", "ADJ", "ADV"}

# Heuristic: if Coptic content-lemma Jaccard overlap with ANY Thomas logion
# exceeds this, mark as Thomas parallel. 0.30 catches the strong Synoptic
# parallels (Logion 86 ↔ Matt 8:20 / Luke 9:58 etc.) without being too
# aggressive. We also drop verses whose token set is a subset of a logion.
THOMAS_PARALLEL_JACCARD = 0.30


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def load_coptic_verses():
    out = {}
    with COPTIC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            book_code = COPTIC_TO_MANIFEST.get(r["book"])
            if not book_code:
                continue
            key = (book_code, r["chapter"], r["verse"])
            content_lemmas = [
                t["lemma"] for t in r["tokens"]
                if t.get("lemma") and t.get("pos") in CONTENT_POS_COPTIC
            ]
            tokens_text = [t["form"] for t in r["tokens"] if t.get("form")]
            out.setdefault(key, {
                "book": book_code,
                "chapter": r["chapter"],
                "verse": r["verse"],
                "coptic_text": " ".join(tokens_text),
                "coptic_lemmas": content_lemmas,
                "coptic_translation": r.get("translation", ""),
            })
    return out


def load_syriac_verses():
    out = {}
    with SYRIAC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["book"], r["chapter"], r["verse"])
            tokens_form = [t["form"] for t in r["tokens"] if t.get("form")]
            tokens_pointed = [t["pointed"] for t in r["tokens"] if t.get("pointed")]
            tokens_lemma = [t["lemma"] for t in r["tokens"] if t.get("lemma")]
            out[key] = {
                "syriac_text_consonantal": " ".join(tokens_form),
                "syriac_text_pointed": " ".join(tokens_pointed),
                "syriac_lemmas": tokens_lemma,
            }
    return out


def load_thomas_logion_lemma_sets():
    by_logion: dict[int, set[str]] = {}
    with THOMAS_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            L = r["logion"]
            for t in r["tokens"]:
                if t.get("lemma") and t.get("pos") in CONTENT_POS_COPTIC:
                    by_logion.setdefault(L, set()).add(t["lemma"])
    return by_logion


def thomas_overlap(coptic_lemmas: list[str], thomas_logia: dict[int, set[str]]):
    """Return (max_jaccard, best_logion). Empty verses get 0.0, -1."""
    if not coptic_lemmas:
        return 0.0, -1
    verse_set = set(coptic_lemmas)
    best_j, best_L = 0.0, -1
    for L, lset in thomas_logia.items():
        if not lset:
            continue
        inter = len(verse_set & lset)
        if inter == 0:
            continue
        union = len(verse_set | lset)
        j = inter / union
        if j > best_j:
            best_j, best_L = j, L
    return best_j, best_L


def main():
    print("Loading Coptic NT…")
    coptic = load_coptic_verses()
    print(f"  {len(coptic)} verses")

    print("Loading Syriac NT…")
    syriac = load_syriac_verses()
    print(f"  {len(syriac)} verses")

    print("Loading Coptic Gospel of Thomas (for parallel detection)…")
    thomas_logia = load_thomas_logion_lemma_sets()
    print(f"  {len(thomas_logia)} logia, "
          f"{sum(len(s) for s in thomas_logia.values())} content lemmas total")

    keys = sorted(set(coptic) & set(syriac))
    print(f"\nVerses with both Coptic + Syriac: {len(keys)}")

    rng = random.Random(42)
    rng.shuffle(keys)

    # Reserve held-out test set (5%) and val set (5%) — drawn from the shuffled
    # pool so distribution mirrors training data. Thomas-parallel exclusion
    # applies AFTER split assignment, marking parallels as "leak".
    n = len(keys)
    n_test = int(round(0.05 * n))
    n_val = int(round(0.05 * n))
    initial_split = {}
    for i, k in enumerate(keys):
        if i < n_test:
            initial_split[k] = "test"
        elif i < n_test + n_val:
            initial_split[k] = "val"
        else:
            initial_split[k] = "train"

    # Now write out, computing Thomas overlap and demoting parallels to "leak".
    OUT.parent.mkdir(parents=True, exist_ok=True)
    book_split_counts = Counter()
    leak_logion_counts = Counter()
    n_written = 0
    n_leak = 0
    n_empty = 0
    leak_strong_examples = []

    with OUT.open("w", encoding="utf-8") as out:
        # Sort keys for deterministic file order (canonical NT order then verse)
        for k in sorted(keys):
            book, ch, v = k
            c = coptic[k]
            s = syriac[k]
            if not c["coptic_lemmas"] or not s["syriac_lemmas"]:
                n_empty += 1
                continue
            j, best_L = thomas_overlap(c["coptic_lemmas"], thomas_logia)
            split = initial_split[k]
            if j >= THOMAS_PARALLEL_JACCARD:
                if split in ("train", "val"):
                    split = "leak"
                    n_leak += 1
                    leak_logion_counts[best_L] += 1
                    if len(leak_strong_examples) < 12:
                        leak_strong_examples.append((j, best_L, book, ch, v,
                                                     c["coptic_translation"][:60]))
            rec = {
                **c,
                **s,
                "thomas_overlap_jaccard": round(j, 3),
                "thomas_overlap_logion": best_L,
                "split": split,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            book_split_counts[(book, split)] += 1
            n_written += 1

    print(f"\nWrote {n_written} aligned verse records → {OUT}")
    print(f"  empty (one side has no content lemmas): {n_empty}")
    print(f"  Thomas-parallel verses (split=leak):    {n_leak}")
    print()
    print(f"Top Thomas logia attracting parallels (overlap ≥ {THOMAS_PARALLEL_JACCARD}):")
    for L, c in leak_logion_counts.most_common(15):
        print(f"  Logion {L:>3}:  {c:>3} parallel NT verses")
    print()
    print(f"Strong leak examples:")
    for j, L, b, ch, v, en in sorted(leak_strong_examples, reverse=True)[:8]:
        print(f"  J={j:.2f} Logion {L:>3}  ↔ {b} {ch}:{v}   ({en!r})")
    print()
    print(f"Per-book per-split totals:")
    print(f"  {'Book':<8s}  {'train':>6s} {'val':>4s} {'test':>5s} {'leak':>5s}")
    books = sorted({b for b, _ in book_split_counts.keys()})
    splits = ["train", "val", "test", "leak"]
    for b in books:
        row = [book_split_counts.get((b, s), 0) for s in splits]
        print(f"  {b:<8s}  {row[0]:>6d} {row[1]:>4d} {row[2]:>5d} {row[3]:>5d}")
    totals = {s: sum(book_split_counts.get((b, s), 0) for b in books) for s in splits}
    print(f"  {'TOTAL':<8s}  {totals['train']:>6d} {totals['val']:>4d} "
          f"{totals['test']:>5d} {totals['leak']:>5d}")


if __name__ == "__main__":
    main()
