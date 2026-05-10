#!/usr/bin/env python3
"""
Phase 2B Step 9 — figures and FINDINGS.md row update.

Inputs:
  data/processed/phase2b_results.json
Output:
  analysis/figures/phase2b_distribution.png
  analysis/figures/phase2b_vs_control.png
  analysis/figures/phase2b_per_pair.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "data" / "processed" / "phase2b_results.json"
OUT_DIR = REPO_ROOT / "analysis" / "figures"

# Reference values for context
PHASE1_MC = 195
PHASE1_MAP = 305
PHASE2A_BEAM = 320
PHASE2C_CONS = 324
COPTIC_BASELINE = 235
ROUNDTRIP_MAX = 1.23
PERRIN = 502

# Perrin's 10 specific cited pairs (to highlight on the per-pair plot)
PERRIN_PAIRS = ["logion_010-logion_011", "logion_016-logion_017",
                "logion_082-logion_083", "logion_029-logion_030",
                "logion_085-logion_086", "logion_014-logion_015",
                "logion_046-logion_047", "logion_113-logion_114",
                "logion_013-logion_014", "logion_017-logion_018"]


def main():
    if not RES.exists():
        raise SystemExit(f"Missing {RES}. "
                          f"Run scripts/phase2b_detect_catchwords.py first.")
    r = json.loads(RES.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: distribution of total catchwords across variant combinations ----
    pp = r["per_pair"]
    # Approximate: each "complete translation" = sum of one combination per pair
    # We have means and p05/p95 per pair, so represent as: total_mean,
    # plus error band from sum of per-pair stds (independence assumption)
    pp_means = [p["mean"] for p in pp.values()]
    pp_std = [p.get("std", 0) for p in pp.values()]

    fig, ax = plt.subplots(figsize=(11, 5))
    # Bar of one method per reference + Phase 2B
    methods = ["Phase 1\nMC", "Phase 1\nMAP", "Phase 2A\nbeam λ=0.3",
               "Phase 2C\nconstrained", "Phase 2B\nLLM canonical",
               "Phase 2B\nLLM mean", "Perrin\n(2002)"]
    totals = [PHASE1_MC, PHASE1_MAP, PHASE2A_BEAM, PHASE2C_CONS,
              r["canonical_total_cws"], r["mean_total_cws"], PERRIN]
    colors = ["#1F77B4", "#5DADE2", "#2CA02C", "#9467BD",
              "#E377C2", "#F7B6D2", "#D62728"]
    bars = ax.bar(methods, totals, color=colors)
    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 8,
                f"{v:.0f}", ha="center", fontsize=9)
    ax.axhline(PERRIN, color="#D62728", linestyle="--", alpha=0.5)
    # Round-trip ceiling reference
    ax.axhline(ROUNDTRIP_MAX * COPTIC_BASELINE, color="black",
                linestyle=":", alpha=0.5,
                label=f"Round-trip ceiling ({ROUNDTRIP_MAX}× = "
                       f"{ROUNDTRIP_MAX*COPTIC_BASELINE:.0f})")
    ax.set_ylabel("Total catchwords (114 adjacent pairs)")
    ax.set_title("Phase 2B — LLM translation vs other methods")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2b_distribution.png", dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/phase2b_distribution.png")

    # ---- Figure 2: Thomas vs control per-pair ----
    if r.get("control_mean_per_pair") is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        thomas_pp = [p["mean"] for p in pp.values()]
        # Control pair means stored aggregate-only; we don't have per-pair list,
        # so plot as a single point
        ax.boxplot([thomas_pp], labels=["Thomas\npairs"],
                    showmeans=True, widths=0.5)
        ax.scatter([1.5], [r["control_mean_per_pair"]],
                    color="C3", marker="D", s=100, zorder=5,
                    label=f"Control mean ({r['control_mean_per_pair']:.2f})")
        ax.set_ylabel("Catchwords per pair (mean across variant combinations)")
        p_str = (f"{r['thomas_vs_control_p']:.4f}"
                 if r.get("thomas_vs_control_p") is not None else "n/a")
        ax.set_title(f"Phase 2B — Thomas vs control passages\n"
                      f"Mann-Whitney p (Thomas > control): {p_str}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "phase2b_vs_control.png", dpi=140, bbox_inches="tight")
        print(f"Wrote {OUT_DIR}/phase2b_vs_control.png")

    # ---- Figure 3: per-pair box plot, Perrin pairs highlighted ----
    fig, ax = plt.subplots(figsize=(14, 5))
    pair_keys = list(pp.keys())
    means = [pp[k]["mean"] for k in pair_keys]
    p05s = [pp[k].get("p05", pp[k]["mean"]) for k in pair_keys]
    p95s = [pp[k].get("p95", pp[k]["mean"]) for k in pair_keys]
    err_lo = [max(m - lo, 0) for m, lo in zip(means, p05s)]
    err_hi = [max(hi - m, 0) for m, hi in zip(means, p95s)]
    is_perrin = [k in PERRIN_PAIRS for k in pair_keys]
    colors = ["red" if p else "C0" for p in is_perrin]
    x = np.arange(len(pair_keys))
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="o", markersize=3,
                  ecolor="lightgray", capsize=2, alpha=0.5)
    ax.scatter(x, means, c=colors, s=10, zorder=5)
    ax.set_xlabel("Adjacent logion pair index")
    ax.set_ylabel("Catchwords per pair (mean ± 90% CI across variant combinations)")
    ax.set_title("Phase 2B — Per-pair catchword density (red = Perrin's 10 cited pairs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2b_per_pair.png", dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/phase2b_per_pair.png")

    # ---- Print summary table ----
    print()
    print("=" * 68)
    print(f"PHASE 2B FINAL SUMMARY")
    print("=" * 68)
    print(f"  LLM canonical total:    {r['canonical_total_cws']} "
          f"(ratio {r['canonical_ratio']:.2f}×)")
    print(f"  LLM mean total:         {r['mean_total_cws']:.0f} "
          f"(ratio {r['mean_ratio']:.2f}×)")
    print(f"  Both-sides %:            {r['canonical_both_pct']:.1f}%")
    print(f"  Isolated %:              {r['canonical_iso_pct']:.1f}%")
    print()
    print("Reference ladder:")
    print(f"  0.83×  Phase 1 MC")
    print(f"  1.23×  Round-trip ceiling")
    print(f"  1.30×  Phase 1 MAP")
    print(f"  1.36×  Phase 2A beam")
    print(f"  1.38×  Phase 2C constrained")
    print(f"  {r['canonical_ratio']:.2f}×  ← Phase 2B LLM canonical")
    print(f"  {r['mean_ratio']:.2f}×  ← Phase 2B LLM mean")
    print(f"  1.87×  Perrin (target)")
    if r.get("control_mean_per_pair") is not None:
        print()
        print(f"Control comparison:")
        print(f"  Thomas mean per-pair:   {r['thomas_mean_per_pair']:.2f}")
        print(f"  Control mean per-pair:  {r['control_mean_per_pair']:.2f}")
        print(f"  Mann-Whitney p:         {r['thomas_vs_control_p']:.4f}")


if __name__ == "__main__":
    main()
