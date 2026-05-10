#!/usr/bin/env python3
"""
Build a Coptic→Syriac lexical translation map by IBM Model 1 EM on the NT
parallel corpus.

Inputs (joined on (book, chapter, verse)):
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl

For each Coptic lemma c, output a probability distribution P(s | c) over
candidate Syriac lemmas s, learned by IBM-1 EM.

Output:
  data/processed/lexical_mapping/coptic_to_syriac.jsonl
    {"coptic_lemma": str, "coptic_pos": str|null,
     "candidates": [{"syriac_lemma": str, "prob": float, "count": int}, ...],
     "support_verses": int}

Why IBM Model 1?
  - For Phase 1 we need a P(s|c) translation distribution to sample from.
  - We don't need positional alignments — only lexical correspondences.
  - IBM-1 EM converges in 5-10 iterations on small parallel corpora.
  - No external deps (pure numpy + python).

Usage:
  python scripts/build_lexical_map.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COPTIC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
SYRIAC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"

NULL_TOKEN = "__NULL__"
N_ITERATIONS = 12
MIN_TRANSLATION_PROB = 1e-4   # prune candidates below this in output
TOP_K_PER_LEMMA = 25          # keep at most this many candidates per Coptic lemma

# Coptic POS tags worth keeping in the lexical map (content words).
# Function-word lemmas (ART, PREP, CONJ, …) still participate in EM training
# because they anchor the alignment; we just don't emit them.
KEEP_POS = {"N", "V", "VBD", "VSTAT", "NPROP", "ADJ", "ADV"}


def load_coptic_verses():
    """Yield (key, lemmas, pos_for_each_lemma) per Coptic verse."""
    verses: dict[tuple, dict] = {}
    with COPTIC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["book"], r["chapter"], r["verse"])
            lemmas = []
            pos_for = {}
            for t in r["tokens"]:
                lem = t.get("lemma")
                if not lem or lem == "_":
                    continue
                lemmas.append(lem)
                pos_for[lem] = t.get("pos") or pos_for.get(lem) or ""
            if lemmas:
                verses[key] = {"lemmas": lemmas, "pos_for": pos_for}
    return verses


# Map Coptic SCRIPTORIUM TT book name → Peshitta manifest book_code
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


def load_syriac_verses():
    """{(book_code, chapter, verse): [lemmas]}"""
    verses = {}
    with SYRIAC_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["book"], r["chapter"], r["verse"])
            lemmas = [t["lemma"] for t in r["tokens"] if t.get("lemma")]
            if lemmas:
                verses[key] = lemmas
    return verses


def pair_verses(coptic, syriac):
    pairs = []
    for c_key, c_data in coptic.items():
        c_book, c_ch, c_vs = c_key
        s_book = COPTIC_TO_MANIFEST.get(c_book)
        if not s_book:
            continue
        s_lemmas = syriac.get((s_book, c_ch, c_vs))
        if not s_lemmas:
            continue
        pairs.append((c_data["lemmas"], s_lemmas, c_data["pos_for"]))
    return pairs


def ibm1_em(pairs, n_iter):
    """Standard IBM Model 1 with NULL on the source (Coptic) side.
       Returns t[s][c] = P(syriac s | coptic c).
    """
    src_vocab: set[str] = {NULL_TOKEN}
    tgt_vocab: set[str] = set()
    for cs, ss, _ in pairs:
        src_vocab.update(cs)
        tgt_vocab.update(ss)
    print(f"  IBM-1: src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}, "
          f"pairs={len(pairs)}")

    # t[s][c] starts uniform: 1/|src_vocab|
    t = defaultdict(lambda: defaultdict(lambda: 1.0 / len(src_vocab)))
    # Pre-fill so defaultdict iteration is deterministic
    for cs, ss, _ in pairs:
        for s in ss:
            t_s = t[s]
            t_s[NULL_TOKEN]
            for c in cs:
                t_s[c]

    for it in range(n_iter):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        log_lik = 0.0
        for cs, ss, _ in pairs:
            cs_with_null = [NULL_TOKEN] + cs
            for s in ss:
                t_s = t[s]
                # P(s | sentence)
                z = sum(t_s[c] for c in cs_with_null)
                if z <= 0.0:
                    continue
                log_lik += __import__("math").log(z) - __import__("math").log(len(cs_with_null))
                for c in cs_with_null:
                    delta = t_s[c] / z
                    count[s][c] += delta
                    total[c] += delta
        # M-step
        new_t = defaultdict(dict)
        for s, c_counts in count.items():
            for c, val in c_counts.items():
                if total[c] > 0:
                    new_t[s][c] = val / total[c]
        t = defaultdict(lambda: defaultdict(float), {s: defaultdict(float, d) for s, d in new_t.items()})
        print(f"  iter {it+1:2d}: log-lik = {log_lik:.0f}")

    return t


def main():
    if not COPTIC_IN.exists() or not SYRIAC_IN.exists():
        sys.exit("Missing inputs. Run parse_coptic_tt.py and annotate_peshitta_lemmas.py first.")

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
        sys.exit("No aligned verses — check book-name mapping.")

    print(f"\nRunning IBM Model 1 EM ({N_ITERATIONS} iterations)…")
    t = ibm1_em(pairs, N_ITERATIONS)

    # Re-key from t[s][c] to {c: [(s, p), ...]}
    by_coptic: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s, c_dict in t.items():
        for c, p in c_dict.items():
            if c == NULL_TOKEN:
                continue
            if p < MIN_TRANSLATION_PROB:
                continue
            by_coptic[c].append((s, p))

    # Count verse support per Coptic lemma + capture POS
    support: dict[str, int] = Counter()
    pos_for_lemma: dict[str, str] = {}
    for cs, _, pos_for in pairs:
        for c in set(cs):
            support[c] += 1
            if c in pos_for:
                pos_for_lemma[c] = pos_for[c]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as fout:
        for c, candidates in sorted(by_coptic.items(), key=lambda kv: -support.get(kv[0], 0)):
            pos = pos_for_lemma.get(c, "")
            if pos and pos not in KEEP_POS:
                continue
            ranked = sorted(candidates, key=lambda x: -x[1])[:TOP_K_PER_LEMMA]
            rec = {
                "coptic_lemma": c,
                "coptic_pos": pos,
                "support_verses": support.get(c, 0),
                "candidates": [
                    {"syriac_lemma": s, "prob": round(p, 6)}
                    for s, p in ranked
                ],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} content-word lexical entries -> {OUT}")


if __name__ == "__main__":
    main()
