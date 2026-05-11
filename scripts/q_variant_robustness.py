#!/usr/bin/env python3
"""
Q variant robustness sweep — for a single language, repeat the permutation
test using each of the 10 LLM translation variants (1,000 perms each).

Designed to run as one of five parallel processes (greek/aramaic/syriac/
hebrew/arabic) so the whole sweep finishes in roughly 1× the slowest-
language time.

Usage:
  python scripts/q_variant_robustness.py --lang aramaic

Output:
  data/q_source/permutation/variant_{lang}.json
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

from scripts.q_permutation_test import (  # noqa: E402
    PHON_THRESHOLD, FILTER_PCT,
    load_q_translations, compute_blocked, precompute_matrix,
    stats_for_order, run_permutation,
    GREEK_FILE,
)

OUT_DIR = REPO_ROOT / "data" / "q_source" / "permutation"
N_PERMUTATIONS = 1000
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True,
                     choices=["greek", "aramaic", "syriac", "hebrew", "arabic"])
    ap.add_argument("--variants", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args()

    variants = [int(x) for x in args.variants.split(",") if x.strip() != ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"variant_{args.lang}.json"

    # Use the Greek pericope-id list as the canonical baseline ordering
    gdata = json.loads(GREEK_FILE.read_text(encoding="utf-8"))
    ids_master = [r["pericope_id"] for r in gdata if r.get("greek_text")]
    print(f"=== Q variant robustness — {args.lang} ===")
    print(f"  pericopes: {len(ids_master)}")
    print(f"  variants: {variants}")
    print(f"  N_PERMUTATIONS per variant: {args.n_perms}")
    print()

    per_v = []
    t_start = time.time()
    # For Greek, variant_idx is ignored — Greek is the source, not translated
    for v in variants:
        if args.lang == "greek" and v > 0:
            break  # only one "variant" of the source text
        print(f"  Variant {v}:", flush=True)
        t0 = time.time()
        translations = load_q_translations(args.lang, variant_idx=v)
        n_loaded = sum(1 for t in translations.values() if t)
        usable_ids = [i for i in ids_master if translations.get(i)]
        if len(usable_ids) < len(ids_master) * 0.5:
            print(f"    SKIP — only {n_loaded}/{len(ids_master)} loaded")
            per_v.append({"variant": v, "skipped": True,
                          "n_loaded": n_loaded})
            continue
        blocked = compute_blocked(translations, FILTER_PCT)
        matrix = precompute_matrix(translations, blocked, args.lang, usable_ids)
        st = stats_for_order(usable_ids, matrix, [2])
        nl = run_permutation(matrix, usable_ids, args.n_perms, SEED,
                                min_freqs=(2,))
        true_rec = st["recurring_2plus"]
        null_mean = float(nl["recurring_2plus"].mean())
        null_std  = float(nl["recurring_2plus"].std())
        z = (true_rec - null_mean) / null_std if null_std > 0 else 0.0
        p = float((nl["recurring_2plus"] >= true_rec).mean())
        elapsed = time.time() - t0
        per_v.append({
            "variant": v, "skipped": False,
            "n_loaded": n_loaded,
            "n_pericopes_used": len(usable_ids),
            "n_blocked": len(blocked),
            "true_recurring_2plus": int(true_rec),
            "null_mean": null_mean, "null_std": null_std,
            "z_score": float(z), "p_value": p,
            "elapsed_s": elapsed,
        })
        print(f"    true={true_rec}, null={null_mean:.1f}±{null_std:.1f}, "
              f"z={z:.2f}, p={p:.4f}  ({elapsed:.0f}s)", flush=True)
        # Save progressively
        out_path.write_text(json.dumps({
            "language": args.lang,
            "n_permutations_per_variant": args.n_perms,
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "n_pericopes": len(ids_master),
            "results": per_v,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    zs = [r["z_score"] for r in per_v if not r.get("skipped")]
    ps = [r["p_value"] for r in per_v if not r.get("skipped")]
    print()
    print(f"=== {args.lang} summary ({time.time()-t_start:.0f}s total) ===")
    if zs:
        print(f"  z-scores: min={min(zs):.2f}, median={np.median(zs):.2f}, "
              f"max={max(zs):.2f}")
        print(f"  p-values: min={min(ps):.4f}, median={np.median(ps):.4f}, "
              f"max={max(ps):.4f}")
        print(f"  all p<0.05?  {all(p < 0.05 for p in ps)}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
