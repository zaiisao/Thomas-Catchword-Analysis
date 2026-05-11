#!/usr/bin/env python3
"""
Cross-linguistic permutation test — Step 6: figures.

Outputs:
  analysis/figures/crossling_permutation.png   — 4 panels, one per language,
       null distribution + true-order marker.
  analysis/figures/crossling_effect_sizes.png — bar chart of z-scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "processed" / "crossling_permutation_results.json"
FIG_DIR = ROOT / "analysis" / "figures"

LANG_ORDER = ["syriac", "hebrew", "arabic", "greek"]
LANG_LABELS = {"syriac": "Syriac", "hebrew": "Hebrew",
                "arabic": "Arabic", "greek": "Greek"}
LANG_COLORS = {"syriac": "#c0392b", "hebrew": "#2980b9",
                "arabic": "#27ae60", "greek": "#8e44ad"}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    per = data["per_language"]

    # --- Figure 1: 4-panel null distribution ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)
    for ax, lang in zip(axes, LANG_ORDER):
        s = per.get(lang)
        if not s or s.get("skipped"):
            ax.set_title(f"{LANG_LABELS[lang]} (skipped)", fontsize=12)
            ax.axis("off")
            continue
        null = np.array(s["_raw_recurring_2plus"])
        true = s["true_recurring_2plus"]
        ax.hist(null, bins=40, color=LANG_COLORS[lang], alpha=0.7,
                edgecolor="black", linewidth=0.4)
        ax.axvline(true, color="black", lw=2.0, ls="--",
                    label=f"True order = {true}")
        # 95th percentile shading
        p95 = np.percentile(null, 95)
        ax.axvline(p95, color="grey", lw=1.0, ls=":",
                    label=f"95th pctile = {p95:.0f}")
        p = s["p_2plus"]
        z = s["z_2plus"]
        ax.set_title(f"{LANG_LABELS[lang]} — p = {p:.4f}, z = {z:+.2f}",
                      fontsize=12)
        ax.set_xlabel("recurring pairs (≥2 boundaries)")
        if lang == LANG_ORDER[0]:
            ax.set_ylabel("# permutations")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle("Cross-linguistic permutation test — does Thomas's true "
                  "ordering produce more recurring catchwords than random?",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "crossling_permutation.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: effect-size bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, zs, ps, colors = [], [], [], []
    for lang in LANG_ORDER:
        s = per.get(lang)
        if not s or s.get("skipped"):
            xs.append(LANG_LABELS[lang]); zs.append(0); ps.append(1.0)
            colors.append("#bdc3c7")
            continue
        xs.append(LANG_LABELS[lang])
        zs.append(s["z_2plus"])
        ps.append(s["p_2plus"])
        colors.append(LANG_COLORS[lang])
    bars = ax.bar(xs, zs, color=colors, alpha=0.85, edgecolor="black", lw=0.6)
    for b, p, z in zip(bars, ps, zs):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.05 if z >= 0 else b.get_height() - 0.18,
                f"p={p:.3f}", ha="center", fontsize=10, weight="bold")
    ax.axhline(1.645, color="grey", ls="--", lw=1.0, label="z = 1.645 (p ≈ 0.05)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Effect size (z-score vs null)")
    ax.set_title("Cross-linguistic permutation: effect sizes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crossling_effect_sizes.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: variant-robustness bar chart (if available) ---
    var = data.get("variant_robustness") or {}
    if var:
        fig, ax = plt.subplots(figsize=(10, 5))
        offsets = np.linspace(-0.3, 0.3, len(LANG_ORDER))
        for off, lang in zip(offsets, LANG_ORDER):
            rows = var.get(lang)
            if not rows:
                continue
            xs = [r["variant"] for r in rows]
            ys = [r["p_2plus"] for r in rows]
            ax.scatter([x + off for x in xs], ys, color=LANG_COLORS[lang],
                       label=LANG_LABELS[lang], s=60, edgecolor="black", lw=0.4)
        ax.axhline(0.05, color="grey", ls="--", lw=1.0, label="p = 0.05")
        ax.set_yscale("log")
        ax.set_xlabel("Gemini variant index")
        ax.set_ylabel("p-value (log scale)")
        ax.set_title("Variant robustness: p-value across all 10 LLM variants")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "crossling_variant_robustness.png", dpi=140)
        plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("crossling_*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
