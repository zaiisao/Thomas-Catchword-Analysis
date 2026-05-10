#!/usr/bin/env python3
"""
Cross-phase comparison — produces a single figure summarizing the three
methodologically independent tests of Perrin's claim:

  Phase 1  — Monte Carlo  (random sampling from EM lexical map)
  Phase 2  — Neural NMT translation
  Phase 3  — Contrastive attention (catchword discovery without supervision)

Plus the calibrated Coptic baseline and Perrin's reported numbers for
direct comparison.

Inputs:
  data/processed/monte_carlo_results.json
  data/processed/phase2_results.json
  data/processed/phase3_attention_catchwords.jsonl  (optional)

Output:
  analysis/figures/cross_phase_summary.png / .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
P1 = REPO_ROOT / "data" / "processed" / "monte_carlo_results.json"
P2 = REPO_ROOT / "data" / "processed" / "phase2_results.json"
OUT = REPO_ROOT / "analysis" / "figures"

PERRIN_TOTAL = 502
PERRIN_BOTH = 89.0
PERRIN_ISO = 0.0
PERRIN_COPTIC = 269

CALIB_COPTIC_BOTH = 53.9
CALIB_COPTIC_ISO = 11.3
CALIB_COPTIC = 235


def main():
    if not P1.exists():
        raise SystemExit(f"Missing {P1}")
    p1 = json.loads(P1.read_text())

    p2 = None
    if P2.exists():
        p2 = json.loads(P2.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Cross-phase summary — testing Perrin's 502-catchword claim",
        fontsize=12,
    )

    # Panel A — total catchword counts
    ax = axes[0]
    labels = ["Coptic\n(automated\nbaseline)",
              "MC mean\n(random\ntranslation)",
              "Phase 2\n(neural\ntranslation)" if p2 else "Phase 2\n(pending)",
              "Perrin\n(2002)"]
    values = [
        CALIB_COPTIC,
        round(p1["overall"]["total"]["mean"], 1),
        p2["neural_total_catchwords"] if p2 else 0,
        PERRIN_TOTAL,
    ]
    p1_errs_low = p1["overall"]["total"]["mean"] - p1["overall"]["total"]["p05"]
    p1_errs_hi = p1["overall"]["total"]["p95"] - p1["overall"]["total"]["mean"]
    err = [
        [0, p1_errs_low, 0, 0],
        [0, p1_errs_hi, 0, 0],
    ]
    colors = ["#779ECC", "#1F77B4", "#2CA02C", "#D62728"]
    bars = ax.bar(labels, values, color=colors, yerr=err, capsize=5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 8,
                f"{v}", ha="center", fontsize=10)
    ax.set_ylabel("Total catchwords across 114 adjacent logion pairs")
    ax.set_title("(A) Total catchword counts")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(PERRIN_TOTAL, color="#D62728", linestyle="--", alpha=0.5)

    # Panel B — connectivity
    ax = axes[1]
    width = 0.18
    x = np.arange(2)  # both-sides %, isolated %
    automated = [CALIB_COPTIC_BOTH, CALIB_COPTIC_ISO]
    mc = [p1["overall"]["both_sides_pct"]["mean"], p1["overall"]["isolated_pct"]["mean"]]
    perrin = [PERRIN_BOTH, PERRIN_ISO]
    if p2:
        neural = [p2["neural_both_pct"], p2["neural_iso_pct"]]
    else:
        neural = [0, 0]
    ax.bar(x - 1.5*width, automated, width, label="Coptic (automated)", color=colors[0])
    ax.bar(x - 0.5*width, mc,        width, label="MC (random tr.)",   color=colors[1])
    ax.bar(x + 0.5*width, neural,    width, label="Phase 2 (neural)",   color=colors[2])
    ax.bar(x + 1.5*width, perrin,    width, label="Perrin",             color=colors[3])
    ax.set_xticks(x, ["Both-sides %", "Isolated %"])
    ax.set_ylabel("Percent of logia")
    ax.set_title("(B) Connectivity")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.94))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cross_phase_summary.png", dpi=140, bbox_inches="tight")
    fig.savefig(OUT / "cross_phase_summary.pdf", bbox_inches="tight")
    print(f"Wrote {OUT}/cross_phase_summary.png")
    print(f"Wrote {OUT}/cross_phase_summary.pdf")


if __name__ == "__main__":
    main()
