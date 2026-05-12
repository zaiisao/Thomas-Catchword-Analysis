#!/usr/bin/env python3
"""
Permutation test on TOTAL catchword count T per logia ordering.

Companion to permutation_test_recurring.py. That script tests *distinct
recurring pairs* (≥2-boundary repeats). This script tests the simpler
statistic: T = total number of catchword pairs appearing across the 114
adjacent-logion boundaries.

For each shuffled order of the 115 logia, walk the 114 adjacencies and sum
the cell sizes (distinct (lemma_a, lemma_b, link_type) tuples per boundary).
Compare T_observed (true Thomas order) to the null distribution.

Reuses the 115×115 catchword matrix from permutation_test_recurring.py.

Output:
  data/processed/permutation_total_count_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.permutation_test_recurring import (  # noqa: E402
    N_LOGIA,
    SEED,
    FILTER_PCT,
    PHON_THRESHOLD,
    compute_blocked,
    load_gemini_variants,
    precompute_catchword_matrix,
)

OUT = REPO_ROOT / "data" / "processed" / "permutation_total_count_results.json"


def total_count_for_order(order: list[int],
                           matrix: dict[tuple[int, int], frozenset]) -> int:
    """T = sum over the 114 adjacencies of |cell|."""
    return sum(len(matrix.get((order[k], order[k + 1]), frozenset()))
                for k in range(len(order) - 1))


def run_total_permutation(matrix, n_perms: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    base = list(range(N_LOGIA))
    out = np.empty(n_perms, dtype=np.int32)
    t0 = time.time()
    for p in range(n_perms):
        shuf = base.copy()
        rng.shuffle(shuf)
        out[p] = total_count_for_order(shuf, matrix)
        if (p + 1) % 2000 == 0:
            print(f"    perm {p+1}/{n_perms} ({time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("=== Total-catchword-count permutation test ===")
    print(f"  variant: {args.variant}, n_perms: {args.n_perms}")
    print(f"  calibration: filter_pct={FILTER_PCT}, phon={PHON_THRESHOLD}")
    print()

    print(f"[1/3] Loading Gemini variant {args.variant}…")
    translations = load_gemini_variants(args.variant)
    n_loaded = sum(1 for t in translations.values() if t)
    print(f"  loaded {n_loaded}/{N_LOGIA} logia")
    blocked = compute_blocked(translations, FILTER_PCT)
    print(f"  blocked {len(blocked)} top-frequent lemmas")

    print()
    print("[2/3] Building 115×115 catchword matrix…")
    matrix = precompute_catchword_matrix(translations, blocked)

    true_order = list(range(N_LOGIA))
    T_obs = total_count_for_order(true_order, matrix)
    print(f"  T_observed (true Thomas order) = {T_obs}")

    print()
    print(f"[3/3] Running {args.n_perms} shuffles…")
    null = run_total_permutation(matrix, args.n_perms, args.seed)

    mean = float(null.mean()); std = float(null.std())
    z = (T_obs - mean) / std if std > 0 else float("nan")
    p_ge = float((null >= T_obs).mean())
    p_le = float((null <= T_obs).mean())

    print()
    print(f"  Null: mean={mean:.1f}, std={std:.1f}, "
          f"min={int(null.min())}, max={int(null.max())}")
    print(f"  T_obs={T_obs}, z={z:.2f}")
    print(f"  P(null >= T_obs) = {p_ge:.4f}")
    print(f"  P(null <= T_obs) = {p_le:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "variant_idx": args.variant,
            "n_permutations": args.n_perms,
            "seed": args.seed,
            "filter_pct": FILTER_PCT,
            "phon_threshold": PHON_THRESHOLD,
            "n_logia": N_LOGIA,
        },
        "T_observed": int(T_obs),
        "null_distribution": {
            "mean": mean,
            "std":  std,
            "min":  int(null.min()),
            "max":  int(null.max()),
            "_raw": null.tolist(),
        },
        "stats": {"z": z, "p_greater_equal": p_ge, "p_less_equal": p_le},
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
