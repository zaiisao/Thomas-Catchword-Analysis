#!/usr/bin/env python3
"""
Merge per-shard raw co-occurrence counts produced by
build_lexical_map.py / build_reverse_lexical_map.py with --raw-out, then
column-normalise and emit the final lexical map JSONL.

Usage:
  python scripts/merge_alignment_shards.py \
      --shards /tmp/rev_shard0.json /tmp/rev_shard1.json /tmp/rev_shard2.json \
      --direction reverse \
      --out data/processed/lexical_mapping/syriac_to_coptic.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_TRANSLATION_PROB = 1e-4
TOP_K_PER_LEMMA = 25

KEEP_COPTIC_POS = {"N", "V", "VBD", "VSTAT", "NPROP", "ADJ", "ADV"}
SYRIAC_CONTENT_PREFIXES = (
    "MS-", "FS-", "MP-", "FP-", "CS-", "CP-",
    "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",
    "SHAPHEL", "ESTAPHAL", "ETHTAPHAL",
)


def is_syriac_content(parse: str) -> bool:
    return bool(parse) and any(parse.startswith(p) for p in SYRIAC_CONTENT_PREFIXES)


def merge_shards(shard_paths):
    raw_score: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    support = Counter()
    parse_for_lemma: dict[str, str] = {}
    total_pairs = 0
    for sp in shard_paths:
        with open(sp) as f:
            d = json.load(f)
        for tg, sr_dict in d["raw_score"].items():
            for sr, v in sr_dict.items():
                raw_score[tg][sr] += v
        for k, v in d["support"].items():
            support[k] += v
        for k, v in d.get("parse_for_lemma", {}).items():
            if v and k not in parse_for_lemma:
                parse_for_lemma[k] = v
        total_pairs += d.get("n_pairs", 0)
    return raw_score, support, parse_for_lemma, total_pairs


def normalize(raw_score):
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


def write_forward(t, support, parse_for_lemma, out_path):
    """Coptic -> Syriac: raw_score is t[syriac][coptic]; emit per Coptic lemma."""
    by_coptic = defaultdict(list)
    for s, c_dict in t.items():
        for c, p in c_dict.items():
            if p < MIN_TRANSLATION_PROB:
                continue
            by_coptic[c].append((s, p))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c, candidates in sorted(by_coptic.items(), key=lambda kv: -support.get(kv[0], 0)):
            pos = parse_for_lemma.get(c, "")
            if pos and pos not in KEEP_COPTIC_POS:
                continue
            ranked = sorted(candidates, key=lambda x: -x[1])[:TOP_K_PER_LEMMA]
            rec = {
                "coptic_lemma": c,
                "coptic_pos": pos,
                "support_verses": support.get(c, 0),
                "candidates": [{"syriac_lemma": s, "prob": round(p, 6)} for s, p in ranked],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def write_reverse(t, support, parse_for_lemma, out_path):
    """Syriac -> Coptic: raw_score is t[coptic][syriac]; emit per Syriac lemma."""
    by_syriac = defaultdict(list)
    for tg_coptic, sr_dict in t.items():
        for sr_syriac, p in sr_dict.items():
            if p < MIN_TRANSLATION_PROB:
                continue
            by_syriac[sr_syriac].append((tg_coptic, p))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sr, candidates in sorted(by_syriac.items(),
                                       key=lambda kv: -support.get(kv[0], 0)):
            parse = parse_for_lemma.get(sr, "")
            if parse and not is_syriac_content(parse):
                skipped += 1
                continue
            ranked = sorted(candidates, key=lambda x: -x[1])[:TOP_K_PER_LEMMA]
            rec = {
                "syriac_lemma": sr,
                "syriac_parse": parse,
                "support_verses": support.get(sr, 0),
                "candidates": [{"coptic_lemma": c, "prob": round(p, 6)} for c, p in ranked],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Per-shard raw-count JSON files.")
    ap.add_argument("--direction", choices=["forward", "reverse"], required=True,
                    help="forward = Coptic->Syriac; reverse = Syriac->Coptic.")
    ap.add_argument("--out", required=True, help="Final lexical-map JSONL.")
    args = ap.parse_args()

    print(f"Merging {len(args.shards)} shards…")
    raw, support, parse_for_lemma, n_pairs = merge_shards(args.shards)
    print(f"  total pairs across shards: {n_pairs}")
    print(f"  raw_score: {sum(len(v) for v in raw.values())} (tg, src) pairs")

    print("Normalising…")
    t = normalize(raw)

    out_path = Path(args.out)
    if args.direction == "forward":
        written = write_forward(t, support, parse_for_lemma, out_path)
        print(f"Wrote {written} Coptic-lemma entries -> {out_path}")
    else:
        written, skipped = write_reverse(t, support, parse_for_lemma, out_path)
        print(f"Wrote {written} entries (skipped {skipped} function-word lemmas) "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
