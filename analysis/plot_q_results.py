#!/usr/bin/env python3
"""
Figures for the Q source analysis:

  q_crossling_permutation.png    — 5-panel null-distribution histograms
  q_crossling_z_scores.png       — bar chart of single-variant z-scores
  q_variant_z_scores.png         — box+strip of z-scores across 10 variants
  q_variant_p_values.png         — same for p-values
  q_vs_thomas.png                — side-by-side Q vs Thomas z-score distributions
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
Q_PERM   = ROOT / "data" / "processed".__class__()  # placeholder; replaced below
Q_PERM   = ROOT / "data" / "q_source" / "permutation" / "main_results.json"
Q_VARDIR = ROOT / "data" / "q_source" / "permutation"
T_VARDIR = ROOT / "data" / "processed" / "crossling_variant_robustness"
FIG_DIR  = ROOT / "analysis" / "figures" / "q_source"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["greek", "aramaic", "syriac", "hebrew", "arabic"]
LABELS = {"greek": "Greek (source)", "aramaic": "Aramaic",
           "syriac": "Syriac", "hebrew": "Hebrew", "arabic": "Arabic"}
COLORS = {"greek": "#8e44ad", "aramaic": "#16a085",
           "syriac": "#c0392b", "hebrew": "#2980b9",
           "arabic": "#27ae60"}


def load_q_variant(lang: str) -> list[dict]:
    p = Q_VARDIR / f"variant_{lang}.json"
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text(encoding="utf-8")).get("results", [])
            if not r.get("skipped")]


def load_thomas_variant(lang: str) -> list[dict]:
    p = T_VARDIR / f"{lang}.json"
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text(encoding="utf-8")).get("results", [])
            if not r.get("skipped")]


def main() -> None:
    main_data = json.loads(Q_PERM.read_text(encoding="utf-8")) if Q_PERM.exists() else None

    # ---- Fig 1: 5-panel null distributions (single variant) ----
    if main_data:
        per = main_data["per_language"]
        n_ok = [l for l in LANGS if per.get(l) and not per[l].get("skipped")]
        fig, axes = plt.subplots(1, len(n_ok), figsize=(4 * len(n_ok), 4.3),
                                   sharey=False)
        if len(n_ok) == 1:
            axes = [axes]
        for ax, lang in zip(axes, n_ok):
            s = per[lang]
            null = np.array(s["_raw_recurring_2plus"])
            true = s["true_recurring_2plus"]
            ax.hist(null, bins=30, color=COLORS[lang], alpha=0.7,
                    edgecolor="black", linewidth=0.4)
            ax.axvline(true, color="black", lw=2.0, ls="--",
                        label=f"True = {true}")
            p95 = np.percentile(null, 95)
            ax.axvline(p95, color="grey", lw=1.0, ls=":",
                        label=f"95th = {p95:.0f}")
            ax.set_title(f"{LABELS[lang]} — p={s['p_2plus']:.4f}, z={s['z_2plus']:+.2f}",
                          fontsize=11)
            ax.set_xlabel("recurring pairs (≥2 boundaries)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        n_per = main_data["n_pericopes"]
        fig.suptitle(f"Q source — permutation test (single variant), N={n_per} pericopes",
                      fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIG_DIR / "q_crossling_permutation.png", dpi=140)
        plt.close(fig)

    # ---- Fig 2: variant-robustness box+strip ----
    q_z_per_lang = {l: [r["z_score"] for r in load_q_variant(l)] for l in LANGS}
    q_p_per_lang = {l: [r["p_value"] for r in load_q_variant(l)] for l in LANGS}
    have = [l for l in LANGS if q_z_per_lang[l]]
    if have:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        positions = list(range(len(have)))
        for i, lang in enumerate(have):
            zs = q_z_per_lang[lang]
            bp = ax.boxplot([zs], positions=[i], widths=0.55,
                             patch_artist=True, showfliers=False,
                             medianprops={"color":"black","lw":1.6})
            for box in bp["boxes"]:
                box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
            rng = np.random.default_rng(i)
            jitter = rng.uniform(-0.12, 0.12, size=len(zs))
            ax.scatter([i+j for j in jitter], zs, s=42,
                       color=COLORS[lang], edgecolor="black", linewidth=0.5,
                       zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[l] for l in have])
        ax.axhline(1.645, color="grey", ls="--", lw=1.0,
                    label="z = 1.645 (p ≈ 0.05)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("Permutation z-score (true vs null, 1k shuffles)")
        ax.set_title("Q source — variant robustness across 10 LLM variants")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q_variant_z_scores.png", dpi=140)
        plt.close(fig)

        # p-values
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, lang in enumerate(have):
            ps_disp = [max(p, 1/1001) for p in q_p_per_lang[lang]]
            bp = ax.boxplot([ps_disp], positions=[i], widths=0.55,
                             patch_artist=True, showfliers=False,
                             medianprops={"color":"black","lw":1.6})
            for box in bp["boxes"]:
                box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
            rng = np.random.default_rng(100+i)
            jitter = rng.uniform(-0.12, 0.12, size=len(ps_disp))
            ax.scatter([i+j for j in jitter], ps_disp, s=42,
                       color=COLORS[lang], edgecolor="black", linewidth=0.5,
                       zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[l] for l in have])
        ax.axhline(0.05, color="grey", ls="--", lw=1.0, label="p = 0.05")
        ax.set_ylabel("p-value (log scale, 1k shuffles per variant)")
        ax.set_title("Q source — variant-robustness p-values")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q_variant_p_values.png", dpi=140)
        plt.close(fig)

    # ---- Fig 3: Q vs Thomas comparison ----
    t_z_per_lang = {l: [r["z_score"] for r in load_thomas_variant(l)]
                     for l in ["syriac", "hebrew", "arabic", "greek"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    # Thomas: 4 languages
    t_langs = ["syriac", "hebrew", "arabic", "greek"]
    t_labels = {"syriac": "Syriac", "hebrew": "Hebrew",
                 "arabic": "Arabic", "greek": "Greek"}
    for i, lang in enumerate(t_langs):
        zs = t_z_per_lang.get(lang) or []
        if not zs: continue
        bp = ax1.boxplot([zs], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops={"color":"black","lw":1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(i)
        ax1.scatter([i + j for j in rng.uniform(-0.12, 0.12, size=len(zs))],
                    zs, s=40, color=COLORS[lang], edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(t_langs)))
    ax1.set_xticklabels([t_labels[l] for l in t_langs])
    ax1.set_title("Thomas (N=115 logia)")
    ax1.set_ylabel("Permutation z-score")
    ax1.axhline(1.645, color="grey", ls="--", lw=1.0)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.grid(axis="y", alpha=0.3)
    # Q: 5 languages (Greek source + 4 targets)
    for i, lang in enumerate(LANGS):
        zs = q_z_per_lang.get(lang) or []
        if not zs: continue
        bp = ax2.boxplot([zs], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops={"color":"black","lw":1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(100 + i)
        ax2.scatter([i + j for j in rng.uniform(-0.12, 0.12, size=len(zs))],
                    zs, s=40, color=COLORS[lang], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(LANGS)))
    ax2.set_xticklabels([LABELS[l] for l in LANGS], rotation=15, ha="right")
    n_q = main_data["n_pericopes"] if main_data else "?"
    ax2.set_title(f"Q (N={n_q} pericopes)")
    ax2.axhline(1.645, color="grey", ls="--", lw=1.0, label="z=1.645 (p≈0.05)")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right")
    fig.suptitle("Permutation z-score across 10 LLM variants — Thomas vs Q",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "q_vs_thomas.png", dpi=140)
    plt.close(fig)

    # ---- Summary text ----
    lines = []
    lines.append("Q source — variant robustness summary")
    lines.append("=" * 75)
    lines.append(f"{'Lang':18s} {'Med z':>8s} {'Mean z':>8s} {'Min z':>8s} "
                 f"{'Max z':>8s} {'Med p':>10s} {'all p<.05?':>11s}")
    lines.append("-" * 75)
    for lang in LANGS:
        zs = q_z_per_lang.get(lang) or []
        ps = q_p_per_lang.get(lang) or []
        if not zs:
            lines.append(f"{LABELS[lang]:18s} (no data)"); continue
        all_sig = all(p < 0.05 for p in ps)
        lines.append(
            f"{LABELS[lang]:18s} {np.median(zs):>8.2f} {np.mean(zs):>8.2f} "
            f"{min(zs):>8.2f} {max(zs):>8.2f} {np.median(ps):>10.4f} "
            f"{('YES' if all_sig else 'NO'):>11s}"
        )
    lines.append("=" * 75)
    lines.append("")
    lines.append("Pairwise Mann-Whitney (one-sided, A > B z, n=10 each):")
    have_data = {l: zs for l, zs in q_z_per_lang.items() if zs}
    for a, za in have_data.items():
        for b, zb in have_data.items():
            if a == b: continue
            u, p = mannwhitneyu(za, zb, alternative="greater")
            lines.append(f"  {LABELS[a]} > {LABELS[b]}:  p = {p:.4f}")
    SUMMARY = Q_VARDIR / "summary.txt"
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Wrote {SUMMARY}")
    print(f"Wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
