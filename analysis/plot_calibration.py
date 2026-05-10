#!/usr/bin/env python3
"""
Visualize the detector-calibration sweep as a 4-panel figure.

Inputs:
  data/processed/detector_calibration.csv

Output:
  analysis/figures/detector_calibration.png
  analysis/figures/detector_calibration.pdf  (vector for paper-ready use)

Panels:
  (a) Coptic catchword count across the (filter, threshold) grid — heatmap.
  (b) Syriac catchword count across the same grid — heatmap, same color scale.
  (c) Syriac / Coptic ratio — heatmap with Perrin's claimed 1.87 contour.
  (d) Calibration scatter: Coptic count (x) vs Syriac count (y) for every cell;
      Perrin's reported (269, 502) marked as a star, and a y=x line shown for
      reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV = REPO_ROOT / "data" / "processed" / "detector_calibration.csv"
OUT_DIR = REPO_ROOT / "analysis" / "figures"
PERRIN_COPTIC = 269
PERRIN_SYRIAC = 502
PERRIN_RATIO = PERRIN_SYRIAC / PERRIN_COPTIC


def to_pivot(df, value_col):
    return df.pivot(index="filter_pct", columns="threshold", values=value_col).sort_index()


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} cells from {CSV}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Detector calibration — Phase 1.3 sensitivity sweep\n"
        "Filter%: drop lemmas appearing in > X% of 115 logia.   "
        "Threshold: phonological similarity score cutoff.",
        fontsize=12, y=0.99,
    )

    # ── Panel (a): Coptic count ──────────────────────────────────────────────
    pivot_c = to_pivot(df, "coptic_total")
    ax = axes[0, 0]
    vmax = max(pivot_c.values.max(), to_pivot(df, "syriac_total").values.max())
    im = ax.imshow(pivot_c.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(pivot_c.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot_c.columns])
    ax.set_yticks(range(len(pivot_c.index)))
    ax.set_yticklabels([f"{f}" for f in pivot_c.index])
    ax.set_xlabel("Phonological threshold")
    ax.set_ylabel("Filter % (drop lemmas in > X% logia)")
    ax.set_title(f"(a) Coptic catchword count   [Perrin: {PERRIN_COPTIC}]")
    for i in range(pivot_c.shape[0]):
        for j in range(pivot_c.shape[1]):
            v = pivot_c.values[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v > vmax * 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel (b): Syriac count ──────────────────────────────────────────────
    pivot_s = to_pivot(df, "syriac_total")
    ax = axes[0, 1]
    im = ax.imshow(pivot_s.values, aspect="auto", cmap="Reds",
                   vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(pivot_s.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot_s.columns])
    ax.set_yticks(range(len(pivot_s.index)))
    ax.set_yticklabels([f"{f}" for f in pivot_s.index])
    ax.set_xlabel("Phonological threshold")
    ax.set_ylabel("Filter %")
    ax.set_title(f"(b) Syriac catchword count (MAP-translated)   [Perrin: {PERRIN_SYRIAC}]")
    for i in range(pivot_s.shape[0]):
        for j in range(pivot_s.shape[1]):
            v = pivot_s.values[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v > vmax * 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel (c): S/C ratio with Perrin contour ─────────────────────────────
    pivot_r = to_pivot(df, "ratio_s_over_c")
    ax = axes[1, 0]
    rmax = max(pivot_r.values.max(), PERRIN_RATIO + 0.2)
    im = ax.imshow(pivot_r.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.5, vmax=rmax, origin="lower")
    ax.set_xticks(range(len(pivot_r.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot_r.columns])
    ax.set_yticks(range(len(pivot_r.index)))
    ax.set_yticklabels([f"{f}" for f in pivot_r.index])
    ax.set_xlabel("Phonological threshold")
    ax.set_ylabel("Filter %")
    ax.set_title(f"(c) Syriac / Coptic ratio   [Perrin claims: {PERRIN_RATIO:.2f}]")
    # Cell labels
    for i in range(pivot_r.shape[0]):
        for j in range(pivot_r.shape[1]):
            v = pivot_r.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    # Contour at Perrin's claimed ratio
    cs = ax.contour(np.arange(pivot_r.shape[1]),
                    np.arange(pivot_r.shape[0]),
                    pivot_r.values,
                    levels=[PERRIN_RATIO], colors="black", linewidths=2.5,
                    linestyles="--")
    ax.clabel(cs, inline=True, fmt=f"Perrin {PERRIN_RATIO:.2f}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="S/C ratio")

    # ── Panel (d): Calibration scatter ───────────────────────────────────────
    ax = axes[1, 1]
    # Color by filter %, marker by threshold
    filters = sorted(df["filter_pct"].unique())
    thresholds = sorted(df["threshold"].unique())
    cmap = plt.get_cmap("viridis")

    for k, fp in enumerate(filters):
        sub = df[df["filter_pct"] == fp].sort_values("threshold")
        color = cmap(k / max(1, len(filters) - 1))
        ax.plot(sub["coptic_total"], sub["syriac_total"], "-",
                color=color, alpha=0.6, linewidth=1)
        ax.scatter(sub["coptic_total"], sub["syriac_total"],
                   s=60, c=[color], edgecolors="black", linewidths=0.5,
                   label=f"filter={fp}%", zorder=3)

    # Perrin's point
    ax.scatter([PERRIN_COPTIC], [PERRIN_SYRIAC],
               marker="*", s=400, c="gold", edgecolors="black", linewidths=1.5,
               zorder=5, label=f"Perrin (269, 502)")
    # y=x reference
    lim = max(df["coptic_total"].max(), df["syriac_total"].max(), PERRIN_SYRIAC) + 50
    ax.plot([0, lim], [0, lim], "k:", alpha=0.4, linewidth=1, label="y = x")
    # y = 1.87 x reference (Perrin's claimed ratio)
    ax.plot([0, lim], [0, lim * PERRIN_RATIO], "r--", alpha=0.5, linewidth=1,
            label=f"y = {PERRIN_RATIO:.2f}x  (Perrin's claimed ratio)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Coptic catchword count")
    ax.set_ylabel("Syriac catchword count (MAP-translated)")
    ax.set_title("(d) Calibration scatter — every cell of the sweep")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "detector_calibration.png"
    pdf = OUT_DIR / "detector_calibration.pdf"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}\nWrote {pdf}")


if __name__ == "__main__":
    main()
