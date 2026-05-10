#!/usr/bin/env python3
"""
Run Phase 1.4 Monte Carlo at the calibration point identified by the
sensitivity sweep:

  filter_pct = 80  (drop lemmas appearing in > 80% of logia — i.e., the
                    "Jesus said" speech-formula effect)
  phonological_threshold = 0.65  (closest match to Perrin's Coptic count)

Outputs:
  data/processed/monte_carlo_results.json
  data/processed/monte_carlo_pair_totals.npy
  analysis/figures/monte_carlo_*.png  (via plot_monte_carlo.py)
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

from phase1_montecarlo.monte_carlo import MonteCarloRunner  # noqa: E402

THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
LEX = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
RESULTS_JSON = REPO_ROOT / "data" / "processed" / "monte_carlo_results.json"
RAW_NPY = REPO_ROOT / "data" / "processed" / "monte_carlo_pair_totals.npy"

# Perrin's reported numbers for comparison
PERRIN_TOTAL = 502
PERRIN_BOTH_PCT = 89.0
PERRIN_ISO_PCT = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--filter-pct", type=float, default=80.0)
    args = ap.parse_args()

    print(f"Phase 1.4 Monte Carlo")
    print(f"  threshold       = {args.threshold}")
    print(f"  filter_pct      = {args.filter_pct}")
    print(f"  n_iterations    = {args.n_iterations}")
    print(f"  seed            = {args.seed}")
    print()

    print("Building runner (this is the slow part)…")
    t0 = time.time()
    runner = MonteCarloRunner(
        thomas_logia_jsonl=THOMAS,
        lex_map_jsonl=LEX,
        phonological_threshold=args.threshold,
        filter_pct=args.filter_pct,
    ).build()
    print(f"  build took {time.time()-t0:.1f}s")
    print()

    print(f"Running {args.n_iterations} simulations…")
    t0 = time.time()
    results = runner.run(n_iterations=args.n_iterations, seed=args.seed)
    print(f"  run took {time.time()-t0:.1f}s "
          f"({args.n_iterations/(time.time()-t0):.0f} iter/s)")
    print()

    js = results.to_json()

    # Summary
    overall_mean = js["overall"]["total"]["mean"]
    overall_p05 = js["overall"]["total"]["p05"]
    overall_p95 = js["overall"]["total"]["p95"]
    both_mean = js["overall"]["both_sides_pct"]["mean"]
    iso_mean = js["overall"]["isolated_pct"]["mean"]

    # Approximate p-value: probability the simulated total ≥ Perrin's claim
    totals = results.overall_total_per_iter
    p_geq_perrin = float(np.mean(totals >= PERRIN_TOTAL))

    print("=" * 70)
    print(f"RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total catchwords (Syriac, MC):  "
          f"mean={overall_mean:.1f}  90%CI=[{overall_p05:.0f}, {overall_p95:.0f}]")
    print(f"  Perrin reports:                 502")
    print(f"  P(MC total ≥ 502):              {p_geq_perrin:.4f}")
    print()
    print(f"  Connectivity, both sides:       "
          f"mean={both_mean:.1f}%   "
          f"90%CI=[{js['overall']['both_sides_pct']['p05']:.1f}, "
          f"{js['overall']['both_sides_pct']['p95']:.1f}]")
    print(f"  Perrin reports:                 89.0%")
    print()
    print(f"  Isolated logia:                 "
          f"mean={iso_mean:.1f}%   "
          f"90%CI=[{js['overall']['isolated_pct']['p05']:.1f}, "
          f"{js['overall']['isolated_pct']['p95']:.1f}]")
    print(f"  Perrin reports:                 0.0%")
    print()

    # Per-pair report for Perrin's specific cited examples
    perrin_pairs = [(10, 11), (16, 17), (82, 83), (29, 30), (85, 86),
                    (14, 15), (46, 47), (113, 114), (13, 14), (17, 18)]
    print(f"Perrin's specifically cited adjacent pairs:")
    print(f"  {'Pair':<10} {'mean':>6} {'p05':>5} {'p95':>5} {'P(≥1)':>7} {'P(≥3)':>7}")
    pp = js["per_pair"]
    for a, b in perrin_pairs:
        key = f"{a}-{b}"
        if key in pp:
            s = pp[key]
            print(f"  {key:<10} "
                  f"{s['mean']:>6.2f} {s['p05']:>5.0f} {s['p95']:>5.0f} "
                  f"{s['prob_at_least_one']:>7.3f} "
                  f"{s['prob_at_least_three']:>7.3f}")

    # Persist
    js["perrin_p_geq_total"] = p_geq_perrin
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    np.save(RAW_NPY, results.pair_totals_per_iter)
    print(f"\nSaved: {RESULTS_JSON}")
    print(f"Saved: {RAW_NPY}")


if __name__ == "__main__":
    main()
