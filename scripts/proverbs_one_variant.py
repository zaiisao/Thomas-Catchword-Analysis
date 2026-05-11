#!/usr/bin/env python3
"""
Run a single (lang, variant) permutation test and write to its own file.
Designed to be launched 40 times in parallel (10 variants × 4 target langs).

Usage:
  python scripts/proverbs_one_variant.py --lang greek --variant 5
Writes:
  data/proverbs/permutation/variant_{lang}_{variant}.json
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

from scripts.proverbs_permutation_test import (  # noqa: E402
    PHON_THRESHOLD, FILTER_PCT, HEB_FILE,
    load_translations, compute_blocked, precompute_matrix,
    stats_for_order, run_permutation,
)

OUT_DIR = REPO_ROOT / "data" / "proverbs" / "permutation"
N_PERMUTATIONS = 1000
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True,
                     choices=["hebrew", "greek", "syriac", "aramaic", "arabic"])
    ap.add_argument("--variant", type=int, required=True)
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args()

    if args.lang == "hebrew" and args.variant > 0:
        print("Hebrew is the source language; only variant 0 is meaningful.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"variant_{args.lang}_{args.variant}.json"
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    hdata = json.loads(HEB_FILE.read_text(encoding="utf-8"))
    ids_master = [r["unit_id"] for r in hdata if r.get("hebrew_text")]
    t0 = time.time()
    print(f"[{args.lang} v{args.variant}] loading translations…", flush=True)
    translations = load_translations(args.lang, variant_idx=args.variant)
    n_loaded = sum(1 for t in translations.values() if t)
    usable_ids = [i for i in ids_master if translations.get(i)]
    if len(usable_ids) < len(ids_master) * 0.5:
        out_path.write_text(json.dumps({
            "language": args.lang, "variant": args.variant, "skipped": True,
            "n_loaded": n_loaded,
        }, indent=2))
        print(f"SKIP: only {n_loaded}/{len(ids_master)}")
        return

    blocked = compute_blocked(translations, FILTER_PCT)
    print(f"[{args.lang} v{args.variant}] building matrix… "
          f"({len(usable_ids)} units, {len(blocked)} blocked)", flush=True)
    matrix = precompute_matrix(translations, blocked, args.lang, usable_ids)
    st = stats_for_order(usable_ids, matrix, [2])
    nl = run_permutation(matrix, usable_ids, args.n_perms, SEED, min_freqs=(2,))
    true_rec = st["recurring_2plus"]
    null_mean = float(nl["recurring_2plus"].mean())
    null_std = float(nl["recurring_2plus"].std())
    z = (true_rec - null_mean) / null_std if null_std > 0 else 0.0
    p = float((nl["recurring_2plus"] >= true_rec).mean())
    elapsed = time.time() - t0
    rec = {
        "language": args.lang,
        "variant": args.variant,
        "skipped": False,
        "n_loaded": n_loaded,
        "n_units_used": len(usable_ids),
        "n_blocked": len(blocked),
        "true_recurring_2plus": int(true_rec),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": float(z),
        "p_value": p,
        "elapsed_s": elapsed,
    }
    out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[{args.lang} v{args.variant}] true={true_rec}, "
          f"null={null_mean:.1f}±{null_std:.1f}, z={z:.2f}, "
          f"p={p:.4f}  ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
