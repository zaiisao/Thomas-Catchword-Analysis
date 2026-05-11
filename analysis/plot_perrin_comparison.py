#!/usr/bin/env python3
"""
Step 5: visualizations for Perrin pair comparison.

Outputs:
  analysis/figures/perrin_per_boundary.png      — per-boundary count: Perrin vs ours
  analysis/figures/perrin_canonical_split.png   — bar of canonical vs Perrin-specific
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PERRIN_BOUND = ROOT / "data" / "processed" / "perrin_catchwords" / "perrin_per_boundary.json"
PAIR = ROOT / "data" / "processed" / "perrin_catchwords" / "pair_comparison.json"
FIG_DIR = ROOT / "analysis" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    perrin = json.loads(PERRIN_BOUND.read_text(encoding="utf-8"))
    pair = json.loads(PAIR.read_text(encoding="utf-8"))
    rows = pair["per_boundary"]
    overall = pair["overall"]

    # --- Figure 1: per-boundary counts (Perrin vs ours) ---
    boundaries = [r["boundary"] for r in rows]
    perrin_counts = np.array([r["perrin_total"] for r in rows])
    our_counts = np.array([r["our_perrin_count"] for r in rows])

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(boundaries))
    width = 0.42
    ax.bar(x - width / 2, perrin_counts, width, label="Perrin (Syriac)",
           color="#c0392b", alpha=0.85)
    ax.bar(x + width / 2, our_counts, width, label="Phase 2B Gemini (ours)",
           color="#2980b9", alpha=0.85)
    ax.set_xlabel("Logion boundary")
    ax.set_ylabel("Catchword word count (Perrin's word counting)")
    ax.set_title(f"Per-boundary Syriac catchword counts: Perrin vs Phase 2B Gemini "
                 f"(total Perrin={perrin_counts.sum()}, ours={our_counts.sum()})")
    # Show only every 5th boundary label
    step = max(1, len(boundaries) // 25)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(boundaries[::step], rotation=60, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perrin_per_boundary.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: canonical vs Perrin-specific split ---
    canon = np.array([r["perrin_canonical"] for r in rows])
    spec = np.array([r["perrin_specific"] for r in rows])
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x, canon, label="Canonical (also in our translation)",
           color="#27ae60", alpha=0.85)
    ax.bar(x, spec, bottom=canon, label="Perrin-specific (not in ours)",
           color="#e67e22", alpha=0.85)
    ax.set_xlabel("Logion boundary")
    ax.set_ylabel("Perrin Syriac catchwords")
    pct = 100.0 * overall["canonical_match"] / max(1, overall["perrin_total"])
    ax.set_title(f"Per-boundary: Perrin Syriac catchwords split as canonical vs "
                 f"Perrin-specific (overall canonical = {pct:.1f}%)")
    ax.set_xticks(x[::step])
    ax.set_xticklabels(boundaries[::step], rotation=60, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perrin_canonical_split.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: cumulative running totals ---
    fig, ax = plt.subplots(figsize=(11, 5))
    p_cum = np.cumsum(perrin_counts)
    o_cum = np.cumsum(our_counts)
    ax.plot(x, p_cum, color="#c0392b", lw=2, label=f"Perrin (final={p_cum[-1]})")
    ax.plot(x, o_cum, color="#2980b9", lw=2, label=f"Phase 2B Gemini (final={o_cum[-1]})")
    ax.set_xlabel("Logion boundary index (Prologue→1 ... 113→114)")
    ax.set_ylabel("Cumulative Syriac catchword count")
    ax.set_title("Cumulative catchword count along the Thomas sequence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perrin_cumulative.png", dpi=140)
    plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("perrin_*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
