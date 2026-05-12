#!/usr/bin/env python3
"""Plot histogram of total catchword count T across random permutations.

Reads data/processed/permutation_total_count_results.json.
Output: analysis/figures/permutation_total_count.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "data" / "processed" / "permutation_total_count_results.json"
OUT = REPO_ROOT / "analysis" / "figures" / "permutation_total_count.png"


def main():
    r = json.loads(RES.read_text(encoding="utf-8"))
    null = np.asarray(r["null_distribution"]["_raw"], dtype=np.int32)
    T = int(r["T_observed"])
    mean = float(r["null_distribution"]["mean"])
    std = float(r["null_distribution"]["std"])
    z = float(r["stats"]["z"])
    p_ge = float(r["stats"]["p_greater_equal"])

    lo = min(null.min(), T) - 5
    hi = max(null.max(), T) + 5
    bin_w = max(1, (hi - lo) // 60)
    bins = np.arange(lo, hi + bin_w, bin_w)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.hist(null, bins=bins, color="C0", alpha=0.65, edgecolor="white",
            label=f"Null (mean={mean:.1f}, std={std:.1f}, "
                  f"min={int(null.min())}, max={int(null.max())})")
    ax.axvline(T, color="C3", linestyle="--", linewidth=2.5,
                label=f"True Thomas order: T = {T}  (z={z:.2f})")

    extreme = "off the chart" if p_ge == 0 else f"P(null ≥ T) = {p_ge:.4f}"
    ax.set_xlabel("Total catchword pairs across the 114 logion boundaries")
    ax.set_ylabel("Number of shuffled permutations")
    ax.set_title("Permutation test — total catchword count per logia ordering\n"
                  f"(N={len(null):,} shuffles, Gemini variant "
                  f"{r['config']['variant_idx']})   {extreme}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
