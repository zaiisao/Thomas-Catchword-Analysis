#!/usr/bin/env python3
"""
Build a Syriac→Coptic lexical translation map using BinaryAlign on the same
~7,800-verse parallel NT corpus used for the forward map.

Migrated from IBM Model 1 EM to BinaryAlign (binary-classification word
alignment on top of a multilingual encoder).  Source/target are swapped
relative to build_lexical_map.py — this is a fresh BinaryAlign pass rather
than a Bayes inversion of the forward map.

Inputs:
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl

Output (schema unchanged from the IBM-1 version — kept stable for
phase1_montecarlo and the round-trip scripts):
  data/processed/lexical_mapping/syriac_to_coptic.jsonl
    {"syriac_lemma": str, "syriac_parse": str|null,
     "candidates": [{"coptic_lemma": str, "prob": float}, ...],
     "support_verses": int}

Usage:
  python scripts/build_reverse_lexical_map.py
  python scripts/build_reverse_lexical_map.py --model xlm-roberta-large
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
OUT = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "syriac_to_coptic.jsonl"

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


def binary_align_corpus_raw(pairs, aligner: BinaryAligner):
    """BinaryAlign on every verse pair, source = Syriac, target = Coptic.
    Returns the raw co-occurrence count dict raw_score[coptic][syriac]
    accumulated from soft alignment probabilities above threshold.
    Normalisation is deferred to `normalize_raw` so that shard outputs can
    be merged additively before the column-normalisation step.
    """
    raw_score: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    print(f"  BinaryAlign: pairs={len(pairs)}  "
          f"model={aligner.cfg.model_name}  "
          f"threshold={aligner.cfg.threshold}  "
          f"symmetrize={aligner.cfg.symmetrize}  "
          f"batch_size={aligner.cfg.batch_size}", flush=True)
    n_aligned_pairs = 0
    for idx, (src_lemmas, tgt_lemmas, _) in enumerate(pairs):
        if not src_lemmas or not tgt_lemmas:
            continue
        result = aligner.align(src_lemmas, tgt_lemmas)
        for i, j, p in result["pairs"]:
            raw_score[tgt_lemmas[j]][src_lemmas[i]] += p
            n_aligned_pairs += 1
        if (idx + 1) % 200 == 0:
            print(f"    aligned {idx+1}/{len(pairs)} verse pairs "
                  f"({n_aligned_pairs} word-pair hits)", flush=True)
    print(f"  BinaryAlign: total aligned word-pairs = {n_aligned_pairs}", flush=True)
    return raw_score


def normalize_raw(raw_score):
    """Column-normalise raw_score[tg][src] -> t[tg][src] = P(tg | src)."""
    col_totals: dict[str, float] = defaultdict(float)
    for tg, sr_dict in raw_score.items():
        for sr, v in sr_dict.items():
            col_totals[sr] += v
    t: dict[str, dict[str, float]] = defaultdict(dict)
    for tg, sr_dict in raw_score.items():
        for sr, v in sr_dict.items():
            if col_totals[sr] > 0:
                t[tg][sr] = v / col_totals[sr]
    return t


def binary_align_corpus(pairs, aligner: BinaryAligner):
    """Convenience wrapper that raw-aligns and immediately normalises."""
    return normalize_raw(binary_align_corpus_raw(pairs, aligner))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base",
                    help="HF model name for the BinaryAlign backbone.")
    ap.add_argument("--head-ckpt", default=None,
                    help="Path to a trained BinaryAlign linear head.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-symmetrize", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Source-word queries per GPU forward pass.")
    ap.add_argument("--shard", default="0:1",
                    help="ID:N — process verse pairs where idx %% N == ID. "
                         "Use for multi-GPU sharding.")
    ap.add_argument("--raw-out", default=None,
                    help="If set, dump raw co-occurrence counts here as JSON "
                         "(skip normalisation + JSONL emission); use "
                         "merge_alignment_shards.py to aggregate shard "
                         "outputs into the final lexical map.")
    ap.add_argument("--out", default=str(OUT),
                    help="Final JSONL output path (when --raw-out is unset).")
    args = ap.parse_args()
    out_path = Path(args.out)

    shard_id, n_shards = map(int, args.shard.split(":"))
    if not (0 <= shard_id < n_shards):
        sys.exit(f"bad --shard {args.shard}")

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
    if args.limit is not None:
        pairs = pairs[: args.limit]
        print(f"  --limit {args.limit}: truncated to {len(pairs)} pairs")
    if n_shards > 1:
        sharded = [p for i, p in enumerate(pairs) if i % n_shards == shard_id]
        print(f"  --shard {shard_id}:{n_shards}: keeping {len(sharded)}/{len(pairs)} pairs")
        pairs = sharded
    if not pairs:
        sys.exit("No aligned verses.")

    print(f"\nInitialising BinaryAlign aligner ({args.model})…")
    cfg = BinaryAlignerConfig(
        model_name=args.model,
        head_ckpt=args.head_ckpt,
        threshold=args.threshold,
        symmetrize=not args.no_symmetrize,
        batch_size=args.batch_size,
    )
    aligner = BinaryAligner(cfg)
    if not aligner._head_trained:
        print("  [no trained head — using cosine-similarity proxy "
              f"(T={cfg.cos_temperature}).  Supply --head-ckpt for full "
              "BinaryAlign behaviour.]")

    print(f"\nRunning BinaryAlign on parallel corpus…")
    raw_score = binary_align_corpus_raw(pairs, aligner)

    # If sharding, just dump raw counts + per-Syriac-lemma supports; the
    # merger will aggregate across shards and normalise.
    if args.raw_out:
        support = Counter()
        parse_for_lemma = {}
        for src_lemmas, _, parse_for in pairs:
            for sr in set(src_lemmas):
                support[sr] += 1
                if sr in parse_for and parse_for[sr]:
                    parse_for_lemma[sr] = parse_for[sr]
        raw_out = Path(args.raw_out)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        with raw_out.open("w", encoding="utf-8") as f:
            json.dump({
                "raw_score": {tg: dict(sr_dict) for tg, sr_dict in raw_score.items()},
                "support": dict(support),
                "parse_for_lemma": parse_for_lemma,
                "n_pairs": len(pairs),
                "shard": f"{shard_id}:{n_shards}",
            }, f, ensure_ascii=False)
        print(f"\nWrote raw counts -> {raw_out}")
        return

    t = normalize_raw(raw_score)

    # Re-key: t[coptic][syriac] → by_syriac[s] = [(coptic, prob), ...]
    by_syriac = defaultdict(list)
    for tg_coptic, sr_dict in t.items():
        for sr_syriac, p in sr_dict.items():
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_function = 0
    with out_path.open("w", encoding="utf-8") as fout:
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
    print(f"  → {out_path}")

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
