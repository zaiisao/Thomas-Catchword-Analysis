#!/usr/bin/env python3
"""
Build a Syriac→Coptic lexical translation map by IBM Model 1 EM on the same
~7,800-verse parallel NT corpus used for the forward map.

NOT a Bayes inversion of the forward map (would require marginal frequencies);
this is a fresh EM run with source/target swapped.

Inputs:
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl

Output:
  data/processed/lexical_mapping/syriac_to_coptic.jsonl
    {"syriac_lemma": str, "syriac_parse": str|null,
     "candidates": [{"coptic_lemma": str, "prob": float}, ...],
     "support_verses": int}

Usage:
  python scripts/build_reverse_lexical_map.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COPTIC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
SYRIAC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "syriac_to_coptic.jsonl"

NULL_TOKEN = "__NULL__"
N_ITERATIONS = 12
MIN_TRANSLATION_PROB = 1e-4
TOP_K_PER_LEMMA = 25

# Syriac parse-code prefixes considered content words (matches the SYRIAC
# language profile in phase1_montecarlo/language_data.py).
SYRIAC_CONTENT_PREFIXES = (
    "MS-", "FS-", "MP-", "FP-", "CS-", "CP-",          # noun states
    "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",      # verb stems
    "SHAPHEL", "ESTAPHAL", "ETHTAPHAL",
)


def is_syriac_content(parse: str) -> bool:
    if not parse:
        return False
    return any(parse.startswith(p) for p in SYRIAC_CONTENT_PREFIXES)


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


def load_coptic_verses():
    verses = {}
    with COPTIC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["book"], r["chapter"], r["verse"])
            lemmas = []
            for t in r["tokens"]:
                lem = t.get("lemma")
                if lem and lem != "_":
                    lemmas.append(lem)
            if lemmas:
                verses[key] = lemmas
    return verses


def load_syriac_verses():
    verses = {}
    with SYRIAC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["book"], r["chapter"], r["verse"])
            lemmas = []
            parse_for = {}
            for t in r["tokens"]:
                lem = t.get("lemma")
                if lem and lem != "_":
                    lemmas.append(lem)
                    parse_for[lem] = t.get("parse", "") or parse_for.get(lem) or ""
            if lemmas:
                verses[key] = {"lemmas": lemmas, "parse_for": parse_for}
    return verses


def pair_verses(coptic, syriac):
    """Return [(syriac_lemmas, coptic_lemmas, syriac_parse_for_lemma), ...]"""
    pairs = []
    for c_key, c_lemmas in coptic.items():
        c_book, c_ch, c_vs = c_key
        s_book = COPTIC_TO_MANIFEST.get(c_book)
        if not s_book:
            continue
        s_data = syriac.get((s_book, c_ch, c_vs))
        if not s_data:
            continue
        pairs.append((s_data["lemmas"], c_lemmas, s_data["parse_for"]))
    return pairs


def ibm1_em(pairs, n_iter):
    """Source = Syriac, target = Coptic. NULL inserted on source side."""
    src_vocab = {NULL_TOKEN}
    tgt_vocab = set()
    for src_lemmas, tgt_lemmas, _ in pairs:
        src_vocab.update(src_lemmas)
        tgt_vocab.update(tgt_lemmas)
    print(f"  IBM-1: src(Syr)={len(src_vocab)}, tgt(Cop)={len(tgt_vocab)}, "
          f"pairs={len(pairs)}")

    t = defaultdict(lambda: defaultdict(lambda: 1.0 / len(src_vocab)))
    for src_lemmas, tgt_lemmas, _ in pairs:
        for tg in tgt_lemmas:
            t_tg = t[tg]
            t_tg[NULL_TOKEN]
            for sr in src_lemmas:
                t_tg[sr]

    for it in range(n_iter):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        log_lik = 0.0
        for src_lemmas, tgt_lemmas, _ in pairs:
            src_with_null = [NULL_TOKEN] + src_lemmas
            for tg in tgt_lemmas:
                t_tg = t[tg]
                z = sum(t_tg[sr] for sr in src_with_null)
                if z <= 0.0:
                    continue
                log_lik += math.log(z) - math.log(len(src_with_null))
                for sr in src_with_null:
                    delta = t_tg[sr] / z
                    count[tg][sr] += delta
                    total[sr] += delta
        new_t = defaultdict(dict)
        for tg, sr_counts in count.items():
            for sr, val in sr_counts.items():
                if total[sr] > 0:
                    new_t[tg][sr] = val / total[sr]
        t = defaultdict(lambda: defaultdict(float),
                         {tg: defaultdict(float, d) for tg, d in new_t.items()})
        print(f"  iter {it+1:2d}: log-lik = {log_lik:.0f}")
    return t  # t[coptic][syriac] = P(coptic | syriac)


def main():
    if not COPTIC_IN.exists() or not SYRIAC_IN.exists():
        sys.exit("Missing inputs. Run parse_coptic_tt.py and "
                  "annotate_peshitta_lemmas.py first.")

    print("Loading Coptic verses…")
    coptic = load_coptic_verses()
    print(f"  {len(coptic)} verses")

    print("Loading Syriac verses…")
    syriac = load_syriac_verses()
    print(f"  {len(syriac)} verses")

    print("Joining on (book, chapter, verse)…")
    pairs = pair_verses(coptic, syriac)
    print(f"  {len(pairs)} aligned verse pairs")
    if not pairs:
        sys.exit("No aligned verses.")

    print(f"\nRunning IBM-1 EM ({N_ITERATIONS} iterations)…")
    t = ibm1_em(pairs, N_ITERATIONS)

    # Re-key: t[coptic][syriac] → by_syriac[s] = [(coptic, prob), ...]
    by_syriac = defaultdict(list)
    for tg_coptic, sr_dict in t.items():
        for sr_syriac, p in sr_dict.items():
            if sr_syriac == NULL_TOKEN:
                continue
            if p < MIN_TRANSLATION_PROB:
                continue
            by_syriac[sr_syriac].append((tg_coptic, p))

    # Verse support per Syriac lemma + capture parse code
    support = Counter()
    parse_for_lemma = {}
    for src_lemmas, _, parse_for in pairs:
        for sr in set(src_lemmas):
            support[sr] += 1
            if sr in parse_for and parse_for[sr]:
                parse_for_lemma[sr] = parse_for[sr]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_function = 0
    with OUT.open("w", encoding="utf-8") as fout:
        for sr, candidates in sorted(by_syriac.items(),
                                      key=lambda kv: -support.get(kv[0], 0)):
            parse = parse_for_lemma.get(sr, "")
            # Filter on the source (Syriac) side: emit only content words.
            # If we have no parse info for this lemma, keep it (defensive).
            if parse and not is_syriac_content(parse):
                skipped_function += 1
                continue
            ranked = sorted(candidates, key=lambda x: -x[1])[:TOP_K_PER_LEMMA]
            rec = {
                "syriac_lemma": sr,
                "syriac_parse": parse,
                "support_verses": support.get(sr, 0),
                "candidates": [
                    {"coptic_lemma": c, "prob": round(p, 6)}
                    for c, p in ranked
                ],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"\nWrote {written} entries (skipped {skipped_function} function-word lemmas)")
    print(f"  → {OUT}")

    # Spot-check Perrin's key examples
    print()
    print("=== Spot-check: top Coptic candidates for Perrin's example Syriac words ===")
    samples = {
        "ܢܘܪܐ": "fire (nūrā)",
        "ܢܘܗܪܐ": "light (nuhrā)",
        "ܥܝܢܐ": "eye (ʿaynā)",
        "ܐܢܬܬܐ": "woman (ʾanttā)",
        "ܐܬܪܐ": "place (ʾatar)",
    }
    for syr, gloss in samples.items():
        c = by_syriac.get(syr, [])
        c = sorted(c, key=lambda x: -x[1])[:5]
        print(f"  {syr} ({gloss}):")
        for cop, p in c:
            print(f"      {cop}  P={p:.3f}")
        if not c:
            print("      (no candidates above threshold)")


if __name__ == "__main__":
    main()
