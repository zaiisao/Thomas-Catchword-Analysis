#!/usr/bin/env python3
"""
Build a Coptic→Syriac lexical translation map using BinaryAlign on the NT
parallel corpus.

Migrated from IBM Model 1 EM (Brown et al. 1993) to BinaryAlign (Yan et al.
2024) — word alignment as per-pair binary classification on top of a
multilingual encoder (default mDeBERTa-v3-base).  No EM, no translation-
probability tables; we instead obtain positional alignments per verse and
aggregate them into a soft co-occurrence distribution P(s | c).

Inputs (joined on (book, chapter, verse)):
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl

For each Coptic lemma c, we output a probability distribution P(s | c) over
candidate Syriac lemmas s, computed by:
  - running BinaryAlign on every verse pair (symmetrised, threshold 0.5)
  - for every (c, s) pair where Coptic word i is aligned to Syriac word j,
    accumulating the soft alignment score  p_sym(i, j)
  - normalising:   P(s | c) ∝ Σ_verses Σ_aligned-pairs p_sym(c, s)

Output (schema identical to the old IBM-1 output — Phase 1 Monte Carlo
expects these field names):
  data/processed/lexical_mapping/coptic_to_syriac.jsonl
    {"coptic_lemma": str, "coptic_pos": str|null,
     "candidates": [{"syriac_lemma": str, "prob": float}, ...],
     "support_verses": int}

Usage:
  python scripts/build_lexical_map.py                          # mDeBERTa default
  python scripts/build_lexical_map.py --model xlm-roberta-large
  python scripts/build_lexical_map.py --head-ckpt path/to/head.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from align_binary import BinaryAligner, BinaryAlignerConfig  # noqa: E402

COPTIC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
SYRIAC_IN = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"

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


def binary_align_corpus(pairs, aligner: BinaryAligner):
    """Run BinaryAlign on every verse pair and aggregate into a soft
    co-occurrence table.

    For every aligned word pair (c, s) above the threshold, accumulate
        score[s][c] += p_sym(c, s)
    Then normalise per Coptic lemma:
        P(s | c) = score[s][c] / Σ_{s'} score[s'][c]
    so the output schema matches the old IBM-1 t[s][c] table.

    Note: BinaryAlign emits a hard 'unaligned' decision when no target word
    exceeds the threshold — this is the BinaryAlign equivalent of an IBM
    NULL alignment.  Unlike IBM-1 we don't track a NULL probability mass.
    """
    raw_score: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    print(f"  BinaryAlign: pairs={len(pairs)}  "
          f"model={aligner.cfg.model_name}  "
          f"threshold={aligner.cfg.threshold}  "
          f"symmetrize={aligner.cfg.symmetrize}")
    n_aligned_pairs = 0
    for idx, (cs, ss, _) in enumerate(pairs):
        if not cs or not ss:
            continue
        result = aligner.align(cs, ss)
        for i, j, p in result["pairs"]:
            raw_score[ss[j]][cs[i]] += p
            n_aligned_pairs += 1
        if (idx + 1) % 200 == 0:
            print(f"    aligned {idx+1}/{len(pairs)} verse pairs "
                  f"({n_aligned_pairs} word-pair hits)")
    print(f"  BinaryAlign: total aligned word-pairs = {n_aligned_pairs}")

    # Column-normalise to recover P(s | c).
    col_totals: dict[str, float] = defaultdict(float)
    for s, c_dict in raw_score.items():
        for c, v in c_dict.items():
            col_totals[c] += v
    t: dict[str, dict[str, float]] = defaultdict(dict)
    for s, c_dict in raw_score.items():
        for c, v in c_dict.items():
            if col_totals[c] > 0:
                t[s][c] = v / col_totals[c]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base",
                    help="HF model name for the BinaryAlign backbone.")
    ap.add_argument("--head-ckpt", default=None,
                    help="Path to a trained BinaryAlign linear head.  "
                         "If omitted, falls back to a cosine-similarity proxy.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Symmetrised-probability threshold for emitting an "
                         "alignment edge.")
    ap.add_argument("--no-symmetrize", action="store_true",
                    help="Skip the reverse pass + averaging (faster but lossier).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N verse pairs (for smoke-testing).")
    ap.add_argument("--out", default=str(OUT),
                    help="Output JSONL path (defaults to the canonical "
                         "data/processed/lexical_mapping/coptic_to_syriac.jsonl).")
    args = ap.parse_args()
    out_path = Path(args.out)

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
    if args.limit is not None:
        pairs = pairs[: args.limit]
        print(f"  --limit {args.limit}: truncated to {len(pairs)} pairs")

    if not pairs:
        sys.exit("No aligned verses — check book-name mapping.")

    print(f"\nInitialising BinaryAlign aligner ({args.model})…")
    cfg = BinaryAlignerConfig(
        model_name=args.model,
        head_ckpt=args.head_ckpt,
        threshold=args.threshold,
        symmetrize=not args.no_symmetrize,
    )
    aligner = BinaryAligner(cfg)
    if not aligner._head_trained:
        print("  [no trained head — using cosine-similarity proxy "
              f"(T={cfg.cos_temperature}).  Supply --head-ckpt for full "
              "BinaryAlign behaviour.]")

    print(f"\nRunning BinaryAlign on parallel corpus…")
    t = binary_align_corpus(pairs, aligner)

    # Re-key from t[s][c] to {c: [(s, p), ...]}
    by_coptic: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s, c_dict in t.items():
        for c, p in c_dict.items():
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
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

    print(f"\nWrote {written} content-word lexical entries -> {out_path}")


if __name__ == "__main__":
    main()
