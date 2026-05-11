#!/usr/bin/env python3
"""
Run main 10k-perm permutation test for a single language. Output to its
own file, designed to be launched in parallel (5 langs in parallel).
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
N_PERMUTATIONS = 10000
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"main_{args.lang}.json"

    hdata = json.loads(HEB_FILE.read_text(encoding="utf-8"))
    ids_master = [r["unit_id"] for r in hdata if r.get("hebrew_text")]
    t0 = time.time()
    print(f"[main {args.lang}] loading…", flush=True)
    translations = load_translations(args.lang, args.variant)
    usable_ids = [i for i in ids_master if translations.get(i)]
    blocked = compute_blocked(translations, FILTER_PCT)
    print(f"[main {args.lang}] matrix build ({len(usable_ids)} units)…", flush=True)
    matrix = precompute_matrix(translations, blocked, args.lang, usable_ids)
    st = stats_for_order(usable_ids, matrix, [2, 3])
    nl = run_permutation(matrix, usable_ids, args.n_perms, SEED,
                          min_freqs=(2, 3))
    p_2 = float((nl["recurring_2plus"] >= st["recurring_2plus"]).mean())
    p_3 = float((nl["recurring_3plus"] >= st["recurring_3plus"]).mean())
    eff_2 = ((st["recurring_2plus"] - nl["recurring_2plus"].mean())
             / max(nl["recurring_2plus"].std(), 1e-9))
    eff_3 = ((st["recurring_3plus"] - nl["recurring_3plus"].mean())
             / max(nl["recurring_3plus"].std(), 1e-9))
    sorted_pairs = sorted(st["pair_locations"].items(),
                            key=lambda x: -len(x[1]))
    rec = {
        "language": args.lang,
        "n_loaded": sum(1 for v in translations.values() if v),
        "n_units_used": len(usable_ids),
        "n_blocked": len(blocked),
        "true_recurring_2plus": int(st["recurring_2plus"]),
        "true_recurring_3plus": int(st["recurring_3plus"]),
        "null_mean_2plus": float(nl["recurring_2plus"].mean()),
        "null_std_2plus":  float(nl["recurring_2plus"].std()),
        "null_mean_3plus": float(nl["recurring_3plus"].mean()),
        "null_std_3plus":  float(nl["recurring_3plus"].std()),
        "p_2plus": p_2,
        "p_3plus": p_3,
        "z_2plus": float(eff_2),
        "z_3plus": float(eff_3),
        "top_pairs": [
            {"lemma_a": k[0], "lemma_b": k[1], "link_type": k[2],
             "frequency": len(v), "boundaries": v}
            for k, v in sorted_pairs[:20]
        ],
        "_raw_recurring_2plus": nl["recurring_2plus"].tolist(),
        "_raw_recurring_3plus": nl["recurring_3plus"].tolist(),
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"[main {args.lang}] rec≥2={st['recurring_2plus']} "
          f"null={nl['recurring_2plus'].mean():.1f}±{nl['recurring_2plus'].std():.1f} "
          f"z={eff_2:.2f} p={p_2:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
