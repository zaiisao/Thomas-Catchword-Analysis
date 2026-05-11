#!/usr/bin/env python3
"""
Cross-linguistic variant robustness sweep.

For a single language, repeat the permutation test using each of the 10
LLM translation variants (1,000 permutations each). Saves z-score and
p-value per variant.

Usage:
  python scripts/crossling_variant_robustness.py --lang syriac
  python scripts/crossling_variant_robustness.py --lang hebrew
  python scripts/crossling_variant_robustness.py --lang arabic
  python scripts/crossling_variant_robustness.py --lang greek

Writes:
  data/processed/crossling_variant_robustness/{lang}.json

Designed to run as one of four parallel processes (one per language) so
the whole sweep finishes in ~25 min instead of ~100 min serial.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse all the loaders / detectors / matrix builders from the main test.
from scripts.crossling_permutation_test import (  # noqa: E402
    N_LOGIA, FILTER_PCT, PHON_THRESHOLD,
    load_translations, compute_blocked, precompute_matrix,
    stats_for_order, run_permutation,
)

OUT_DIR = REPO_ROOT / "data" / "processed" / "crossling_variant_robustness"
N_PERMUTATIONS = 1000
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True,
                     choices=["syriac", "hebrew", "arabic", "greek"])
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--variants", default="0,1,2,3,4,5,6,7,8,9",
                     help="Comma-separated variant indices to run")
    args = ap.parse_args()

    variants = [int(x) for x in args.variants.split(",") if x.strip() != ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.lang}.json"

    print(f"=== Variant robustness — {args.lang} ===")
    print(f"  N_PERMUTATIONS per variant: {args.n_perms}")
    print(f"  threshold={PHON_THRESHOLD}, filter_pct={FILTER_PCT}")
    print()

    per_v = []
    t_start = time.time()
    for v in variants:
        print(f"  Variant {v}:", flush=True)
        t0 = time.time()
        translations = load_translations(args.lang, variant_idx=v)
        n_loaded = sum(1 for t in translations.values() if t)
        if n_loaded < N_LOGIA * 0.5:
            print(f"    SKIP — only {n_loaded}/{N_LOGIA} logia loaded")
            per_v.append({"variant": v, "skipped": True,
                          "n_loaded": n_loaded})
            continue
        blocked = compute_blocked(translations, FILTER_PCT)
        matrix = precompute_matrix(translations, blocked, args.lang)
        true_order = list(range(N_LOGIA))
        st = stats_for_order(true_order, matrix, [2])
        nl = run_permutation(matrix, args.n_perms, SEED, min_freqs=(2,))
        true_rec = st["recurring_2plus"]
        null_mean = float(nl["recurring_2plus"].mean())
        null_std = float(nl["recurring_2plus"].std())
        z = (true_rec - null_mean) / null_std if null_std > 0 else 0.0
        p = float((nl["recurring_2plus"] >= true_rec).mean())
        elapsed = time.time() - t0
        per_v.append({
            "variant": v,
            "skipped": False,
            "n_loaded": n_loaded,
            "n_blocked": len(blocked),
            "true_recurring_2plus": int(true_rec),
            "null_mean": null_mean,
            "null_std":  null_std,
            "z_score":   float(z),
            "p_value":   p,
            "elapsed_s": elapsed,
        })
        print(f"    true={true_rec}, null={null_mean:.1f}±{null_std:.1f}, "
              f"z={z:.2f}, p={p:.4f}  ({elapsed:.0f}s)", flush=True)

        # Save progressively so a kill doesn't lose work
        out_path.write_text(json.dumps({
            "language": args.lang,
            "n_permutations_per_variant": args.n_perms,
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "results": per_v,
        }, indent=2), encoding="utf-8")

    z_scores = [r["z_score"] for r in per_v if not r.get("skipped")]
    p_values = [r["p_value"] for r in per_v if not r.get("skipped")]
    elapsed_total = time.time() - t_start
    print()
    print(f"=== {args.lang} summary ({elapsed_total:.0f}s total) ===")
    if z_scores:
        print(f"  z-scores:   min={min(z_scores):.2f}, "
              f"median={np.median(z_scores):.2f}, max={max(z_scores):.2f}")
        print(f"  p-values:   min={min(p_values):.4f}, "
              f"median={np.median(p_values):.4f}, "
              f"max={max(p_values):.4f}")
        print(f"  all p<0.05?  {all(p < 0.05 for p in p_values)}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
