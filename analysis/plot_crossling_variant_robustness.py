#!/usr/bin/env python3
"""
Cross-linguistic variant robustness — figures + summary.

Loads:
  data/processed/crossling_variant_robustness/{syriac,hebrew,arabic,greek}.json

Writes:
  analysis/figures/crossling_variant_z_scores.png   — box+strip of z per lang
  analysis/figures/crossling_variant_p_values.png   — same for p-values
  data/processed/crossling_variant_robustness/summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "crossling_variant_robustness"
FIG_DIR = ROOT / "analysis" / "figures"
SUMMARY = DATA_DIR / "summary.txt"

LANGS = ["syriac", "hebrew", "arabic", "greek"]
LABELS = {"syriac": "Syriac", "hebrew": "Hebrew",
           "arabic": "Arabic", "greek": "Greek"}
COLORS = {"syriac": "#c0392b", "hebrew": "#2980b9",
           "arabic": "#27ae60", "greek": "#8e44ad"}


def load_lang(lang: str) -> list[dict]:
    p = DATA_DIR / f"{lang}.json"
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text(encoding="utf-8"))["results"]
            if not r.get("skipped")]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = {l: load_lang(l) for l in LANGS}
    missing = [l for l, rows in data.items() if not rows]
    if missing:
        print(f"WARNING: no data for {missing}. Skipping those.")

    # ---- Figure 1: z-score box+strip ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = list(range(len(LANGS)))
    z_per_lang: dict[str, list[float]] = {}
    p_per_lang: dict[str, list[float]] = {}
    for i, lang in enumerate(LANGS):
        rows = data[lang]
        if not rows:
            continue
        zs = [r["z_score"] for r in rows]
        ps = [r["p_value"] for r in rows]
        z_per_lang[lang] = zs
        p_per_lang[lang] = ps
        bp = ax.boxplot([zs], positions=[i], widths=0.55,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "lw": 1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang])
            box.set_alpha(0.5)
            box.set_edgecolor("black")
        # Jittered scatter
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.12, 0.12, size=len(zs))
        ax.scatter([i + j for j in jitter], zs, s=42,
                   color=COLORS[lang], edgecolor="black", linewidth=0.5,
                   zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[l] for l in LANGS])
    ax.axhline(1.645, color="grey", ls="--", lw=1.0, label="z = 1.645 (p ≈ 0.05)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Permutation z-score (true vs null, 1k shuffles)")
    ax.set_title("Cross-linguistic variant robustness — z-score across 10 LLM variants")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crossling_variant_z_scores.png", dpi=140)
    plt.close(fig)

    # ---- Figure 2: p-values (log scale) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, lang in enumerate(LANGS):
        if lang not in p_per_lang:
            continue
        ps = p_per_lang[lang]
        # Replace 0 with 1/(n_perms+1) for log display
        ps_disp = [max(p, 1.0 / 1001) for p in ps]
        bp = ax.boxplot([ps_disp], positions=[i], widths=0.55,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "lw": 1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(100 + i)
        jitter = rng.uniform(-0.12, 0.12, size=len(ps_disp))
        ax.scatter([i + j for j in jitter], ps_disp, s=42,
                   color=COLORS[lang], edgecolor="black", linewidth=0.5,
                   zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[l] for l in LANGS])
    ax.axhline(0.05, color="grey", ls="--", lw=1.0, label="p = 0.05")
    ax.set_ylabel("p-value (log scale, 1k shuffles per variant)")
    ax.set_title("Cross-linguistic variant robustness — p-value across 10 LLM variants")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crossling_variant_p_values.png", dpi=140)
    plt.close(fig)

    # ---- Summary text ----
    lines = []
    lines.append("Cross-Linguistic Variant Robustness Summary")
    lines.append("=" * 72)
    lines.append(f"{'Language':<10} {'Med z':>8} {'Mean z':>8} {'Min z':>8} "
                 f"{'Max z':>8} {'Med p':>10} {'all p<.05?':>11}")
    lines.append("-" * 72)
    for lang in LANGS:
        zs = z_per_lang.get(lang) or []
        ps = p_per_lang.get(lang) or []
        if not zs:
            lines.append(f"{LABELS[lang]:<10} (no data)")
            continue
        all_sig = all(p < 0.05 for p in ps)
        lines.append(
            f"{LABELS[lang]:<10} "
            f"{np.median(zs):>8.2f} "
            f"{np.mean(zs):>8.2f} "
            f"{min(zs):>8.2f} "
            f"{max(zs):>8.2f} "
            f"{np.median(ps):>10.4f} "
            f"{('YES' if all_sig else 'NO'):>11}"
        )
    lines.append("=" * 72)
    lines.append("")
    lines.append("z-score ranges per language:")
    for lang in LANGS:
        zs = z_per_lang.get(lang) or []
        if zs:
            lines.append(f"  {LABELS[lang]:<8}: [{min(zs):.2f}, {max(zs):.2f}]")
    if all(lang in z_per_lang for lang in LANGS):
        all_mins = [min(z_per_lang[l]) for l in LANGS]
        all_maxs = [max(z_per_lang[l]) for l in LANGS]
        overall_overlap = max(all_mins) < min(all_maxs)
        lines.append("")
        lines.append(f"All four ranges overlap: "
                     f"{'YES' if overall_overlap else 'NO'}")
        # Mann-Whitney Syriac vs Greek (z-scores)
        u, p_syr_vs_grk = mannwhitneyu(z_per_lang["syriac"], z_per_lang["greek"],
                                         alternative="greater")
        lines.append(f"Mann-Whitney (Syriac z > Greek z, n=10 each): "
                     f"U={u:.1f}, p={p_syr_vs_grk:.4f}")
        # Pairwise Mann-Whitney
        lines.append("")
        lines.append("Pairwise Mann-Whitney (one-sided, A > B z):")
        for a in LANGS:
            for b in LANGS:
                if a == b: continue
                u, p = mannwhitneyu(z_per_lang[a], z_per_lang[b],
                                      alternative="greater")
                lines.append(f"  {LABELS[a]} > {LABELS[b]}:  p = {p:.4f}")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Wrote summary: {SUMMARY}")
    print(f"Wrote figures: {FIG_DIR}/crossling_variant_*.png")


if __name__ == "__main__":
    main()
