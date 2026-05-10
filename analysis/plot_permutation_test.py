#!/usr/bin/env python3
"""
Generate figures for the permutation test on recurring catchword patterns.

Inputs:
  data/processed/permutation_test_results.json
Output:
  analysis/figures/permutation_recurring.png
  analysis/figures/permutation_recurring_3plus.png
  analysis/figures/permutation_top_pairs.png
  analysis/figures/permutation_variant_robustness.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "data" / "processed" / "permutation_test_results.json"
OUT_DIR = REPO_ROOT / "analysis" / "figures"


def hist_with_true(ax, null_array, true_value, p_value, title, xlabel):
    """Histogram of null distribution with vertical true-value line."""
    null_array = np.asarray(null_array)
    bins = max(20, min(60, int((null_array.max() - null_array.min()) + 1)))
    ax.hist(null_array, bins=bins, color="C0", alpha=0.65, edgecolor="white",
            label=f"Null (mean={null_array.mean():.1f}, std={null_array.std():.1f})")
    ax.axvline(true_value, color="C3", linestyle="--", linewidth=2.5,
                label=f"True order = {true_value}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# of shuffled permutations")
    ax.set_title(f"{title}\np = {p_value:.4f}  "
                  f"({'extreme' if p_value < 0.05 else 'within null'})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")


def main():
    if not RES.exists():
        raise SystemExit(f"Missing {RES}")
    r = json.loads(RES.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    null_2 = r["null_distribution"]["_raw_recurring_2plus"]
    null_3 = r["null_distribution"]["_raw_recurring_3plus"]
    null_max = r["null_distribution"]["_raw_max_freq"]

    # ---- Figure 1: Histogram (recurring ≥2) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hist_with_true(
        ax, null_2,
        r["true_order"]["recurring_2plus"],
        r["p_values"]["p_recurring_2plus"],
        "Permutation test — distinct catchword pairs recurring at ≥ 2 logion boundaries",
        "Number of recurring pairs",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "permutation_recurring.png",
                 dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/permutation_recurring.png")

    # ---- Figure 2: Histogram (recurring ≥3) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hist_with_true(
        ax, null_3,
        r["true_order"]["recurring_3plus"],
        r["p_values"]["p_recurring_3plus"],
        "Permutation test — pairs recurring at ≥ 3 boundaries",
        "Number of recurring pairs",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "permutation_recurring_3plus.png",
                 dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/permutation_recurring_3plus.png")

    # ---- Figure 3: Top recurring pairs ----
    top = r["true_order"]["top_pairs"][:10]
    if top:
        fig, ax = plt.subplots(figsize=(11, 6))
        labels = [f"({p['lemma_a']}, {p['lemma_b']})\n[{p['link_type']}]" for p in top]
        freqs = [p["frequency"] for p in top]
        # Estimate null mean per-pair frequency from the null max distribution
        # (rough — top pair in null has expected mean ≈ null_max_mean for shuffled)
        null_max_mean = r["null_distribution"]["max_freq_mean"]
        null_max_std = r["null_distribution"]["max_freq_std"]
        ax.bar(labels, freqs, color="C3", alpha=0.85)
        # Draw the null max-frequency band as a horizontal reference
        ax.axhline(null_max_mean, color="gray", linestyle="--",
                    label=f"Null max-freq mean = {null_max_mean:.1f} "
                          f"(±{null_max_std:.1f})")
        ax.fill_between(np.arange(-0.5, len(top), 1),
                          null_max_mean - null_max_std,
                          null_max_mean + null_max_std,
                          color="gray", alpha=0.2)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Number of boundaries where pair recurs")
        ax.set_title("Top-10 most recurring catchword pairs in TRUE Thomas order\n"
                      "(red bars) vs null max-frequency in 10,000 shuffles (gray)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "permutation_top_pairs.png",
                     dpi=140, bbox_inches="tight")
        print(f"Wrote {OUT_DIR}/permutation_top_pairs.png")

    # ---- Figure 4: variant robustness ----
    if r.get("variant_robustness"):
        var = r["variant_robustness"]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(var))
        p2 = [v["p_2plus"] for v in var]
        p3 = [v["p_3plus"] for v in var]
        ax.bar(x - 0.2, p2, 0.4, color="C0", label="p (≥2 boundaries)")
        ax.bar(x + 0.2, p3, 0.4, color="C2", label="p (≥3 boundaries)")
        ax.axhline(0.05, color="C3", linestyle="--", alpha=0.7,
                    label="α = 0.05")
        ax.set_xticks(x)
        ax.set_xticklabels([f"v{v['variant']}" for v in var])
        ax.set_xlabel("LLM variant")
        ax.set_ylabel("Permutation p-value (1-tailed: true ≥ null)")
        ax.set_title("Permutation-test p-values across 10 Gemini variants "
                      "(1,000 perms each)")
        ax.set_ylim(0, max(0.20, max(max(p2), max(p3)) * 1.1))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "permutation_variant_robustness.png",
                     dpi=140, bbox_inches="tight")
        print(f"Wrote {OUT_DIR}/permutation_variant_robustness.png")

    # ---- Print verdict ----
    print()
    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    p2 = r["p_values"]["p_recurring_2plus"]
    p3 = r["p_values"]["p_recurring_3plus"]
    pmax = r["p_values"]["p_max_freq"]
    if p2 < 0.01:
        v = "STRONG evidence of non-random arrangement (Perrin's argument survives)"
    elif p2 < 0.05:
        v = "Significant evidence of non-random arrangement"
    elif p2 < 0.10:
        v = "Marginal evidence"
    else:
        v = "No evidence — arrangement consistent with chance"
    print(f"  Recurring (≥2): p={p2:.4f}    →  {v}")
    print(f"  Recurring (≥3): p={p3:.4f}")
    print(f"  Max freq:        p={pmax:.4f}")
    if r.get("beam_cross_validation"):
        bp = r["beam_cross_validation"]
        print(f"  Beam λ=0.3 cross-check (≥2): p={bp['p_2plus']:.4f}")


if __name__ == "__main__":
    main()
