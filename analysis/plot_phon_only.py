#!/usr/bin/env python3
"""
Phonological-only cross-linguistic permutation test — aggregator + plots.

Reads data/phon_only/{corpus}_{lang}_v{variant}.json and produces:
  data/phon_only/summary.json                    — aggregated table
  data/phon_only/summary.txt                     — human-readable table
  analysis/figures/phon_only_comparison.png      — all-vs-phon-only bars
  analysis/figures/phon_only_proverbs_validation.png — does Hebrew lead?
  analysis/figures/phon_only_thomas.png          — Syriac for Thomas?
  analysis/figures/phon_only_variant_robustness.png — box plots across variants
"""
from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "phon_only"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_SOURCE = {"proverbs": "hebrew", "q": "greek", "thomas": "syriac"}
CORPUS_LANGS = {
    "proverbs": ["hebrew", "greek", "syriac", "aramaic", "arabic"],
    "q":        ["greek", "hebrew", "syriac", "aramaic", "arabic"],
    "thomas":   ["syriac", "hebrew", "greek", "arabic"],
}
COLORS = {
    "hebrew":  "#d35400",
    "greek":   "#8e44ad",
    "syriac":  "#c0392b",
    "aramaic": "#16a085",
    "arabic":  "#27ae60",
}
LABELS = {
    "hebrew":  "Hebrew",
    "greek":   "Greek",
    "syriac":  "Syriac",
    "aramaic": "Aramaic",
    "arabic":  "Arabic",
}


def load_all() -> dict:
    """Return d[corpus][lang][variant] = record."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for p in sorted(IN_DIR.glob("*.json")):
        if p.name in ("summary.json",):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        corpus = d.get("corpus"); lang = d.get("lang"); var = d.get("variant")
        if corpus and lang and var is not None and not d.get("skipped"):
            out[corpus][lang][var] = d
    return out


def main() -> None:
    data = load_all()

    # -------- variant 0 main table --------
    rows = []
    for corpus in ["proverbs", "thomas", "q"]:
        for lang in CORPUS_LANGS[corpus]:
            rec = data.get(corpus, {}).get(lang, {}).get(0)
            if not rec:
                continue
            r = rec["results"]
            diag = rec["diagnostic"]
            rows.append({
                "corpus": corpus, "lang": lang,
                "is_source": rec["is_source"],
                "n": rec["n_units_used"],
                "phon_per_boundary": diag["per_boundary_phon_plus_etym"],
                "sem_per_boundary":  diag["per_boundary_semantic"],
                "z_all":  r["all"]["z_score"],
                "p_all":  r["all"]["p_value"],
                "z_phon": r["phon"]["z_score"],
                "p_phon": r["phon"]["p_value"],
                "z_sem":  r["sem"]["z_score"],
                "p_sem":  r["sem"]["p_value"],
            })

    # -------- variant sweep z-scores (per filter) --------
    sweep = {}  # sweep[corpus][lang][filter] = list of z-scores across variants
    for corpus, langs in data.items():
        sweep[corpus] = {}
        for lang, vs in langs.items():
            sweep[corpus][lang] = {"all": [], "phon": [], "sem": []}
            for v, rec in sorted(vs.items()):
                r = rec["results"]
                for f in ("all", "phon", "sem"):
                    sweep[corpus][lang][f].append(r[f]["z_score"])

    # -------- print + save summary text --------
    lines = []
    lines.append("=" * 100)
    lines.append("PHONOLOGICAL-ONLY CROSS-LINGUISTIC PERMUTATION TEST")
    lines.append("=" * 100)
    lines.append(f"{'corpus':<10} {'lang':<8} {'src?':<5} {'N':>4} "
                  f"{'phon/B':>7} {'sem/B':>7} "
                  f"{'z_all':>6} {'p_all':>8} "
                  f"{'z_phon':>7} {'p_phon':>8} "
                  f"{'z_sem':>6} {'p_sem':>8}")
    lines.append("-" * 100)
    for r in rows:
        src = "SRC" if r["is_source"] else ""
        lines.append(
            f"{r['corpus']:<10} {LABELS[r['lang']]:<8} {src:<5} "
            f"{r['n']:>4d} "
            f"{r['phon_per_boundary']:>7.2f} "
            f"{r['sem_per_boundary']:>7.2f} "
            f"{r['z_all']:>6.2f} {r['p_all']:>8.4f} "
            f"{r['z_phon']:>7.2f} {r['p_phon']:>8.4f} "
            f"{r['z_sem']:>6.2f} {r['p_sem']:>8.4f}"
        )
    lines.append("=" * 100)
    lines.append("")
    lines.append("Variant-sweep medians (phon-only) — across 1 (source) or 10 variants:")
    lines.append("-" * 80)
    for corpus in ["proverbs", "thomas", "q"]:
        for lang in CORPUS_LANGS[corpus]:
            zs = sweep.get(corpus, {}).get(lang, {}).get("phon") or []
            if not zs: continue
            src = "SRC" if lang == CORPUS_SOURCE[corpus] else "   "
            sig = sum(1 for z in zs if z >= 1.645)
            lines.append(
                f"  {corpus:<10} {LABELS[lang]:<8} {src} "
                f"n_var={len(zs):>2d}  med_z={np.median(zs):>6.2f}  "
                f"range=[{min(zs):>5.2f}, {max(zs):>5.2f}]  "
                f"sig={sig}/{len(zs)}"
            )
    lines.append("=" * 80)

    # -------- source-leads test (one-sample: source has 1 variant, so we test
    # whether source z is greater than the EMPIRICAL DISTRIBUTION of target z's) --------
    lines.append("")
    lines.append("SOURCE-LANGUAGE-LEADS TEST (one-sample: rank of source z vs target variant z's)")
    lines.append("-" * 80)
    for corpus in ["proverbs", "thomas", "q"]:
        src_lang = CORPUS_SOURCE[corpus]
        src_zs = [rec["results"]["phon"]["z_score"]
                  for _, rec in sorted(data.get(corpus, {}).get(src_lang, {}).items())]
        if not src_zs:
            continue
        src_z = src_zs[0]  # variant 0 only for source
        lines.append(f"  Corpus = {corpus}, source = {LABELS[src_lang]} "
                      f"(z_phon = {src_z:.2f}):")
        # Pool all target variants into one comparison
        all_target_zs = []
        for other in CORPUS_LANGS[corpus]:
            if other == src_lang: continue
            zs = [rec["results"]["phon"]["z_score"]
                  for _, rec in sorted(data.get(corpus, {}).get(other, {}).items())]
            all_target_zs.extend(zs)
            if zs:
                n_higher = sum(1 for z in zs if z > src_z)
                tag = "BEATS all" if n_higher == 0 else f"beaten by {n_higher}/{len(zs)}"
                cmp_med = "≥" if src_z >= np.median(zs) else "<"
                lines.append(f"    vs {LABELS[other]:8s}: target med_z={np.median(zs):>5.2f}, "
                              f"range=[{min(zs):>5.2f}, {max(zs):>5.2f}], "
                              f"source {cmp_med} target_median, source {tag}")
        # Pooled rank test
        if all_target_zs:
            rank = sum(1 for z in all_target_zs if z >= src_z) + 1
            total = len(all_target_zs) + 1
            empirical_p = rank / total
            lines.append(f"    POOLED: source rank = {rank}/{total} target variants "
                          f"(empirical p = {empirical_p:.3f})  "
                          f"{'✓ SOURCE LEADS' if rank == 1 else '✗ source does NOT lead'}")
    lines.append("=" * 80)
    print("\n".join(lines))
    (IN_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Save structured summary
    summary = {
        "variant0_rows": rows,
        "variant_sweep": {c: {l: {f: list(z) for f, z in d.items()}
                               for l, d in cl.items()}
                            for c, cl in sweep.items()},
    }
    (IN_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    # -------- Figure 1: all vs phon-only bars (variant 0, all corpora) --------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, corpus in zip(axes, ["proverbs", "thomas", "q"]):
        my_rows = [r for r in rows if r["corpus"] == corpus]
        if not my_rows: continue
        langs = [r["lang"] for r in my_rows]
        z_all = [r["z_all"]  for r in my_rows]
        z_phon = [r["z_phon"] for r in my_rows]
        x = np.arange(len(langs))
        w = 0.36
        for i, (l, za, zp) in enumerate(zip(langs, z_all, z_phon)):
            ax.bar(i - w/2, za, w, color=COLORS[l], alpha=0.95,
                    edgecolor="black", lw=0.7,
                    label="all catchwords" if i == 0 else None)
            ax.bar(i + w/2, zp, w, color=COLORS[l], alpha=0.4,
                    edgecolor="black", lw=0.7, hatch="//",
                    label="phon+etym only" if i == 0 else None)
            if my_rows[i]["is_source"]:
                ax.text(i, max(za, zp) + 0.15, "SRC", ha="center",
                        fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[l] for l in langs])
        ax.axhline(1.645, color="grey", ls="--", lw=1, label="z=1.645 (p≈0.05)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{corpus.title()} (variant 0, 10k perms)")
        ax.grid(axis="y", alpha=0.3)
        if corpus == "proverbs":
            ax.set_ylabel("Permutation z-score")
            ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("All catchwords vs. phon-only — does removing the thematic component "
                  "let the source language lead?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "phon_only_comparison.png", dpi=140)
    plt.close(fig)

    # -------- Figure 2: Proverbs phon-only validation --------
    prov_rows = [r for r in rows if r["corpus"] == "proverbs"]
    if prov_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        langs = [r["lang"] for r in prov_rows]
        z_phon = [r["z_phon"] for r in prov_rows]
        x = np.arange(len(langs))
        for i, (l, zp) in enumerate(zip(langs, z_phon)):
            ax.bar(i, zp, color=COLORS[l], alpha=0.85,
                    edgecolor="black", lw=0.8)
            if prov_rows[i]["is_source"]:
                ax.text(i, zp + 0.1, "SOURCE", ha="center",
                         fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[l] for l in langs])
        ax.axhline(1.645, color="grey", ls="--", lw=1, label="z=1.645")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("Phon-only z (10k perms)")
        ax.set_title("Proverbs phon-only validation — does Hebrew (source) now lead?")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "phon_only_proverbs_validation.png", dpi=140)
        plt.close(fig)

    # -------- Figure 3: Thomas phon-only --------
    thom_rows = [r for r in rows if r["corpus"] == "thomas"]
    if thom_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        langs = [r["lang"] for r in thom_rows]
        z_phon = [r["z_phon"] for r in thom_rows]
        x = np.arange(len(langs))
        for i, (l, zp) in enumerate(zip(langs, z_phon)):
            ax.bar(i, zp, color=COLORS[l], alpha=0.85,
                    edgecolor="black", lw=0.8)
            if thom_rows[i]["is_source"]:
                ax.text(i, zp + 0.1, "[Syr ⇐ Coptic]", ha="center",
                         fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[l] for l in langs])
        ax.axhline(1.645, color="grey", ls="--", lw=1, label="z=1.645")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("Phon-only z (10k perms)")
        ax.set_title("Thomas phon-only — Syriac-specific arrangement?")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "phon_only_thomas.png", dpi=140)
        plt.close(fig)

    # -------- Figure 4: variant-robustness box plots (phon-only) --------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, corpus in zip(axes, ["proverbs", "thomas", "q"]):
        langs = CORPUS_LANGS[corpus]
        positions = list(range(len(langs)))
        for i, lang in enumerate(langs):
            zs = sweep.get(corpus, {}).get(lang, {}).get("phon") or []
            if not zs:
                continue
            bp = ax.boxplot([zs], positions=[i], widths=0.55,
                             patch_artist=True, showfliers=False,
                             medianprops={"color":"black","lw":1.6})
            for box in bp["boxes"]:
                box.set_facecolor(COLORS[lang]); box.set_alpha(0.5)
            rng = np.random.default_rng(hash((corpus, lang)) & 0xfffff)
            jitter = rng.uniform(-0.12, 0.12, size=len(zs))
            ax.scatter([i+j for j in jitter], zs, s=40,
                        color=COLORS[lang], edgecolor="black", linewidth=0.5,
                        zorder=3)
            if lang == CORPUS_SOURCE[corpus]:
                ax.text(i, max(zs) + 0.2, "SRC", ha="center", fontsize=9,
                         fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[l] for l in langs])
        ax.axhline(1.645, color="grey", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{corpus.title()}")
        ax.grid(axis="y", alpha=0.3)
        if corpus == "proverbs":
            ax.set_ylabel("Phon-only z (per variant)")
    fig.suptitle("Phon-only variant robustness — does the source language lead consistently?",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "phon_only_variant_robustness.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("phon_only_*.png")):
        print(f"  {p.name}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
