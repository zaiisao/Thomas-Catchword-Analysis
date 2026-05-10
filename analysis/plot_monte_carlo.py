#!/usr/bin/env python3
"""
Visualize the Phase 1.4 Monte Carlo results.

Inputs:
  data/processed/monte_carlo_results.json
  data/processed/monte_carlo_pair_totals.npy

Outputs:
  analysis/figures/mc_total_distribution.png
  analysis/figures/mc_per_pair_distribution.png
  analysis/figures/mc_perrin_pairs.png
  analysis/figures/mc_summary.png   (4-panel composite)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "data" / "processed" / "monte_carlo_results.json"
RAW_NPY = REPO_ROOT / "data" / "processed" / "monte_carlo_pair_totals.npy"
OUT = REPO_ROOT / "analysis" / "figures"

PERRIN_TOTAL = 502
PERRIN_BOTH = 89.0
PERRIN_ISO = 0.0

# Pairs Perrin specifically discusses in JETS 2006
PERRIN_PAIRS = [
    ("10-11",  "nūrā/nuhrā",        "fire/light"),
    ("16-17",  "nūrā/nuhrā (eyes)", "fire/light"),
    ("82-83",  "nūrā/nuhrā",        "fire/light"),
    ("29-30",  "ʿetar/ʾatar",       "wealth/place"),
    ("85-86",  "ʿetar/ʾatar",       "wealth/place"),
    ("14-15",  "naš/nesse",         "someone/women"),
    ("46-47",  "nesse/naš",         "women/someone"),
    ("113-114","naš/nesse",         "someone/women"),
    ("13-14",  "panni/penayim",     "returned/districts"),
    ("17-18",  "idaʿ noun/verb",    "hand/know"),
]


def main():
    with RESULTS_JSON.open() as f:
        js = json.load(f)
    pair_totals = np.load(RAW_NPY)  # shape (n_iter, n_pairs)

    overall_totals = pair_totals.sum(axis=1)
    pair_keys = list(js["per_pair"].keys())
    n_logia = js["n_logia"]

    # Connectivity per iteration
    has_left = (np.zeros_like(pair_totals, dtype=bool))
    has_right = (np.zeros_like(pair_totals, dtype=bool))
    # has_right[i, j] iff pair j (logion j ↔ j+1) has ≥1 catchword in iter i
    # so left-neighbor catchword for logion j+1, right-neighbor for j
    has_pair = pair_totals > 0
    has_right_per_logion = np.zeros((pair_totals.shape[0], n_logia), dtype=bool)
    has_left_per_logion = np.zeros((pair_totals.shape[0], n_logia), dtype=bool)
    for j in range(pair_totals.shape[1]):
        has_right_per_logion[:, j]   |= has_pair[:, j]
        has_left_per_logion[:, j+1]  |= has_pair[:, j]
    both = has_left_per_logion & has_right_per_logion
    iso = ~(has_left_per_logion | has_right_per_logion)
    both_pct = (both.sum(axis=1) / n_logia) * 100
    iso_pct = (iso.sum(axis=1) / n_logia) * 100

    # ─── Composite 4-panel figure ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Phase 1.4 Monte Carlo — N={js['n_iterations']:,} simulations\n"
        f"Coptic Gospel of Thomas → sampled Syriac translation → catchword detection\n"
        f"Threshold={js['phonological_threshold']}, Filter={js['filter_pct']}%",
        fontsize=12, y=0.99,
    )

    # (a) Total catchword distribution with Perrin's claim
    ax = axes[0, 0]
    ax.hist(overall_totals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(overall_totals), color="black", linestyle="--",
               linewidth=2, label=f"MC mean = {np.mean(overall_totals):.0f}")
    ax.axvspan(np.percentile(overall_totals, 5),
               np.percentile(overall_totals, 95),
               alpha=0.15, color="steelblue", label="MC 90% CI")
    ax.axvline(PERRIN_TOTAL, color="crimson", linewidth=2.5,
               label=f"Perrin's claim = {PERRIN_TOTAL}")
    ax.set_xlabel("Total catchwords across 114 adjacent logion pairs")
    ax.set_ylabel("Number of simulations")
    ax.set_title("(a) Total catchwords per simulation\n"
                 f"P(MC ≥ {PERRIN_TOTAL}) = {js['perrin_p_geq_total']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # annotate gap
    gap = PERRIN_TOTAL - np.mean(overall_totals)
    ax.annotate("", xy=(PERRIN_TOTAL, ax.get_ylim()[1] * 0.7),
                xytext=(np.mean(overall_totals), ax.get_ylim()[1] * 0.7),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
    ax.text((np.mean(overall_totals) + PERRIN_TOTAL) / 2,
            ax.get_ylim()[1] * 0.75,
            f"gap: +{gap:.0f}", ha="center", color="crimson", fontsize=10)

    # (b) Connectivity comparison
    ax = axes[0, 1]
    ax.hist(both_pct, bins=30, color="forestgreen", alpha=0.7,
            edgecolor="white", label="Both-sides %")
    ax.hist(iso_pct, bins=30, color="orange", alpha=0.7,
            edgecolor="white", label="Isolated %")
    ax.axvline(PERRIN_BOTH, color="darkgreen", linestyle="--", linewidth=2.5,
               label=f"Perrin both = {PERRIN_BOTH}%")
    ax.axvline(PERRIN_ISO, color="darkorange", linestyle="--", linewidth=2.5,
               label=f"Perrin iso = {PERRIN_ISO}%")
    ax.set_xlabel("% of logia")
    ax.set_ylabel("Number of simulations")
    ax.set_title("(b) Connectivity per simulation\n"
                 f"MC both ~ {np.mean(both_pct):.1f}%   "
                 f"MC iso ~ {np.mean(iso_pct):.1f}%")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # (c) Per-pair box plot for ALL adjacent pairs
    ax = axes[1, 0]
    # Sample every 4th pair for readability (114 pairs is too many to label)
    step = 4
    sub_pair_idxs = list(range(0, pair_totals.shape[1], step))
    bp_data = [pair_totals[:, j] for j in sub_pair_idxs]
    bp_labels = [pair_keys[j] for j in sub_pair_idxs]
    bp = ax.boxplot(bp_data, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xticklabels(bp_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Catchwords per pair")
    ax.set_title(f"(c) Per-pair distribution (every {step}th pair shown)\n"
                 f"All {pair_totals.shape[1]} adjacent pairs span the x-axis")
    ax.grid(True, axis="y", alpha=0.3)

    # (d) Perrin's specifically cited pairs — bar chart with P(≥1) and P(≥3)
    ax = axes[1, 1]
    available = [(k, n, g) for k, n, g in PERRIN_PAIRS if k in js["per_pair"]]
    keys = [k for k, _, _ in available]
    means = [js["per_pair"][k]["mean"] for k in keys]
    p_ge1 = [js["per_pair"][k]["prob_at_least_one"] for k in keys]
    p_ge3 = [js["per_pair"][k]["prob_at_least_three"] for k in keys]

    x = np.arange(len(keys))
    width = 0.4
    bars1 = ax.bar(x - width/2, p_ge1, width, color="navy",
                   alpha=0.85, label="P(≥1 catchword)")
    bars2 = ax.bar(x + width/2, p_ge3, width, color="crimson",
                   alpha=0.85, label="P(≥3 catchwords)")
    # annotate mean value on each pair
    for i, m in enumerate(means):
        ax.text(i, max(p_ge1[i], p_ge3[i]) + 0.03,
                f"μ={m:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    labels = [f"{k}\n{n}" for k, n, _ in available]
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Probability under random translation")
    ax.set_ylim(0, 1.18)
    ax.set_title("(d) Perrin's specifically cited pairs — robustness under MC")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "mc_summary.png"
    pdf = OUT / "mc_summary.pdf"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
