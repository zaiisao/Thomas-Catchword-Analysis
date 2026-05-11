#!/usr/bin/env python3
"""
Figures for Proverbs analysis:

  proverbs_crossling_permutation.png   — 5-panel null-distribution histograms
  proverbs_variant_z_scores.png        — box+strip of z across 10 variants/lang
  proverbs_variant_p_values.png        — same for p-values
  proverbs_aggregate_density.png       — Prov vs Ctrl density per language
  three_corpus_comparison.png          — Proverbs vs Thomas vs Q side-by-side
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
P_PERM = ROOT / "data" / "proverbs" / "permutation" / "main_results.json"
P_VARDIR = ROOT / "data" / "proverbs" / "permutation"
P_DENSITY = ROOT / "data" / "proverbs" / "aggregate_density.json"

# Comparison sources
Q_VARDIR = ROOT / "data" / "q_source" / "permutation"
T_VARDIR = ROOT / "data" / "processed" / "crossling_variant_robustness"

FIG_DIR = ROOT / "analysis" / "figures" / "proverbs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["hebrew", "greek", "syriac", "aramaic", "arabic"]
LABELS = {"hebrew": "Hebrew (source)", "greek": "Greek", "syriac": "Syriac",
           "aramaic": "Aramaic", "arabic": "Arabic"}
COLORS = {"hebrew": "#d35400", "greek": "#8e44ad", "syriac": "#c0392b",
           "aramaic": "#16a085", "arabic": "#27ae60"}


def load_variant(lang: str, vardir: Path, fname_pattern: str = "variant_{}.json"
                 ) -> list[dict]:
    p = vardir / fname_pattern.format(lang)
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text(encoding="utf-8")).get("results", [])
            if not r.get("skipped")]


def main() -> None:
    main_data = json.loads(P_PERM.read_text(encoding="utf-8")) if P_PERM.exists() else None

    # ---- Fig 1: 5-panel null distributions ----
    if main_data:
        per = main_data["per_language"]
        n_ok = [l for l in LANGS if per.get(l) and not per[l].get("skipped")]
        fig, axes = plt.subplots(1, len(n_ok), figsize=(4.2 * len(n_ok), 4.3),
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
        n = main_data["n_units"]
        fig.suptitle(f"Proverbs 10–29 (N={n} verses) — permutation test, single variant",
                      fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIG_DIR / "proverbs_crossling_permutation.png", dpi=140)
        plt.close(fig)

    # ---- Fig 2: variant robustness ----
    p_z = {l: [r["z_score"] for r in load_variant(l, P_VARDIR)] for l in LANGS}
    p_p = {l: [r["p_value"] for r in load_variant(l, P_VARDIR)] for l in LANGS}
    have = [l for l in LANGS if p_z[l]]
    if have:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        positions = list(range(len(have)))
        for i, lang in enumerate(have):
            zs = p_z[lang]
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
        ax.set_title("Proverbs 10–29 — variant robustness across 10 LLM variants")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "proverbs_variant_z_scores.png", dpi=140)
        plt.close(fig)
        # p-values
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, lang in enumerate(have):
            ps_disp = [max(p, 1/1001) for p in p_p[lang]]
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
        ax.set_ylabel("p-value (log scale)")
        ax.set_title("Proverbs 10–29 — variant-robustness p-values")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "proverbs_variant_p_values.png", dpi=140)
        plt.close(fig)

    # ---- Fig 3: aggregate density ----
    if P_DENSITY.exists():
        dens = json.loads(P_DENSITY.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(10, 5))
        xs, p_vals, c_vals, p_p_raw, p_p_den = [], [], [], [], []
        cols = []
        for lang in LANGS:
            r = dens.get(lang) or {}
            if r.get("skipped"): continue
            xs.append(LABELS[lang])
            p_vals.append(r["proverbs_mean_density"])
            c_vals.append(r["control_mean_density"])
            p_p_raw.append(r["mw_prov_gt_ctrl_raw_p"])
            p_p_den.append(r["mw_prov_gt_ctrl_density_p"])
            cols.append(COLORS[lang])
        idx = np.arange(len(xs))
        w = 0.35
        ax.bar(idx - w/2, p_vals, w, label="Proverbs", color=cols, alpha=0.85,
               edgecolor="black", lw=0.5)
        ax.bar(idx + w/2, c_vals, w, label="Controls (narrative)",
               color="#7f8c8d", alpha=0.7, edgecolor="black", lw=0.5)
        for i, (pd, pr) in enumerate(zip(p_p_den, p_p_raw)):
            txt = f"p_d={pd:.3f}"
            ax.text(i, max(p_vals[i], c_vals[i]) * 1.05, txt,
                    ha="center", fontsize=9)
        ax.set_xticks(idx)
        ax.set_xticklabels(xs)
        ax.set_ylabel("Catchwords per 100×100 word pair")
        ax.set_title("Proverbs vs narrative controls — length-normalised density")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "proverbs_aggregate_density.png", dpi=140)
        plt.close(fig)

    # ---- Fig 4: THREE-CORPUS COMPARISON (the key figure) ----
    # Load Thomas variant data
    t_z = {l: [r["z_score"] for r in load_variant(l, T_VARDIR, "{}.json")]
            for l in ["syriac", "greek", "hebrew", "arabic"]}
    # Q variant data
    q_z = {l: [r["z_score"] for r in load_variant(l, Q_VARDIR)]
            for l in ["greek", "aramaic", "syriac", "hebrew", "arabic"]}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    Q_LANGS = ["greek", "aramaic", "syriac", "hebrew", "arabic"]
    Q_LABEL = {"greek":"Greek (source)", "aramaic":"Aramaic",
                "syriac":"Syriac", "hebrew":"Hebrew", "arabic":"Arabic"}
    T_LANGS = ["syriac", "greek", "hebrew", "arabic"]
    T_LABEL = {"syriac":"Syriac", "greek":"Greek", "hebrew":"Hebrew",
                "arabic":"Arabic"}

    # Proverbs (leftmost)
    ax = axes[0]
    for i, lang in enumerate(LANGS):
        zs = p_z.get(lang) or []
        if not zs: continue
        bp = ax.boxplot([zs], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops={"color":"black","lw":1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(i)
        ax.scatter([i + j for j in rng.uniform(-0.12, 0.12, size=len(zs))],
                   zs, s=40, color=COLORS[lang], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([LABELS[l] for l in LANGS], rotation=15, ha="right")
    n_p = main_data["n_units"] if main_data else "?"
    ax.set_title(f"Proverbs 10–29 (N={n_p}, source=Hebrew)")
    ax.set_ylabel("Permutation z-score")
    ax.axhline(1.645, color="grey", ls="--", lw=1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(axis="y", alpha=0.3)

    # Thomas (middle)
    ax = axes[1]
    for i, lang in enumerate(T_LANGS):
        zs = t_z.get(lang) or []
        if not zs: continue
        bp = ax.boxplot([zs], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops={"color":"black","lw":1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(100+i)
        ax.scatter([i + j for j in rng.uniform(-0.12, 0.12, size=len(zs))],
                   zs, s=40, color=COLORS[lang], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(T_LANGS)))
    ax.set_xticklabels([T_LABEL[l] for l in T_LANGS])
    ax.set_title("Thomas (N=115, source=Coptic — translated)")
    ax.axhline(1.645, color="grey", ls="--", lw=1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(axis="y", alpha=0.3)

    # Q (rightmost)
    ax = axes[2]
    for i, lang in enumerate(Q_LANGS):
        zs = q_z.get(lang) or []
        if not zs: continue
        bp = ax.boxplot([zs], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops={"color":"black","lw":1.6})
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
        rng = np.random.default_rng(200+i)
        ax.scatter([i + j for j in rng.uniform(-0.12, 0.12, size=len(zs))],
                   zs, s=40, color=COLORS[lang], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(Q_LANGS)))
    ax.set_xticklabels([Q_LABEL[l] for l in Q_LANGS], rotation=15, ha="right")
    ax.set_title("Q (N=56, source=Greek)")
    ax.axhline(1.645, color="grey", ls="--", lw=1.0, label="z=1.645 (p≈0.05)")
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle("Three-corpus pipeline validation — permutation z-scores across LLM variants",
                  fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "three_corpus_comparison.png", dpi=140)
    plt.close(fig)

    # ---- Summary text ----
    lines = []
    lines.append("Proverbs 10–29 — variant robustness summary")
    lines.append("=" * 75)
    lines.append(f"{'Lang':18s} {'Med z':>8s} {'Mean z':>8s} {'Min z':>8s} "
                 f"{'Max z':>8s} {'Med p':>10s} {'all p<.05?':>11s}")
    lines.append("-" * 75)
    for lang in LANGS:
        zs = p_z.get(lang) or []
        ps = p_p.get(lang) or []
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
    lines.append("Pairwise Mann-Whitney (one-sided, A > B z):")
    have_data = {l: zs for l, zs in p_z.items() if zs}
    for a, za in have_data.items():
        for b, zb in have_data.items():
            if a == b: continue
            u, p = mannwhitneyu(za, zb, alternative="greater")
            star = "  ✓" if p < 0.05 else ""
            lines.append(f"  {LABELS[a]:18s} > {LABELS[b]:18s}:  p = {p:.4f}{star}")
    out = P_VARDIR / "summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Wrote {out}")
    print(f"Wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
