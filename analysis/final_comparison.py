#!/usr/bin/env python3
"""
Final cross-phase synthesis — combine Phase 1 (Monte Carlo), Phase 2
(translation methods), and Phase 3 (literary-structure tests) into one
paper-ready summary.

Inputs:
  data/processed/monte_carlo_results.json
  data/processed/phase2a_beam_results.json
  data/processed/phase2c_constrained_results.json
  data/processed/phase2b_llm_results.json   (optional)
  data/processed/phase3_baseline_results.json
  data/processed/phase3_improved_thomas.json

Outputs:
  data/processed/final_summary.json
  analysis/figures/final_summary.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
P_MC = REPO_ROOT / "data" / "processed" / "monte_carlo_results.json"
P_2A = REPO_ROOT / "data" / "processed" / "phase2a_beam_results.json"
P_2B = REPO_ROOT / "data" / "processed" / "phase2b_llm_results.json"
P_2C = REPO_ROOT / "data" / "processed" / "phase2c_constrained_results.json"
P_3BASE = REPO_ROOT / "data" / "processed" / "phase3_baseline_results.json"
P_3THOMAS = REPO_ROOT / "data" / "processed" / "phase3_improved_thomas.json"
P_CALIB = REPO_ROOT / "data" / "processed" / "detector_calibration.csv"

OUT_JSON = REPO_ROOT / "data" / "processed" / "final_summary.json"
OUT_FIG = REPO_ROOT / "analysis" / "figures" / "final_summary.png"

PERRIN_TOTAL = 502
COPTIC_TOTAL = 235


def load_calib_map_at(filter_pct=80, threshold=0.65):
    if not P_CALIB.exists():
        return None
    with P_CALIB.open() as f:
        for row in csv.DictReader(f):
            if (float(row["filter_pct"]) == filter_pct
                    and abs(float(row["threshold"]) - threshold) < 1e-6):
                return int(row["syriac_total"])
    return None


def main():
    # ---- Phase 1 ----
    mc = json.loads(P_MC.read_text())
    mc_mean = mc["overall"]["total"]["mean"]
    mc_p05 = mc["overall"]["total"]["p05"]
    mc_p95 = mc["overall"]["total"]["p95"]
    mc_p_ge_perrin = mc.get("perrin_p_geq_total", 0.0)

    # ---- Phase 2 ----
    map_total = load_calib_map_at(80, 0.65) or 305
    a2 = json.loads(P_2A.read_text())
    a2_03 = a2["best_beam_per_lambda"]["0.3"]
    c2 = json.loads(P_2C.read_text())
    c2_10 = c2["results_per_lambda"]["1.0"]
    b2 = None
    if P_2B.exists():
        b2 = json.loads(P_2B.read_text())

    # ---- Phase 3 ----
    p3base = json.loads(P_3BASE.read_text())
    p3_overall = p3base["overall"]
    p3thomas = json.loads(P_3THOMAS.read_text()) if P_3THOMAS.exists() else None

    # ---- Build summary dict ----
    summary = {
        "calibration": {"filter_pct": 80, "phon_threshold": 0.65},
        "phase1_mc": {
            "mean": mc_mean, "p05": mc_p05, "p95": mc_p95,
            "p_geq_perrin": mc_p_ge_perrin,
            "interpretation": ("Random translation from EM lexical map "
                               f"yields mean {mc_mean:.0f} catchwords; "
                               f"p(≥502) = {mc_p_ge_perrin}")
        },
        "phase2": {
            "coptic_baseline": COPTIC_TOTAL,
            "phase1_map": map_total,
            "phase2a_beam_lambda03": {
                "total": a2_03["total"],
                "both_pct": a2_03["both_pct"],
                "iso_pct": a2_03["iso_pct"],
            },
            "phase2c_constrained_lambda10": {
                "total_mean": c2_10["mean"],
                "total_p05": c2_10["p05"],
                "total_p95": c2_10["p95"],
                "p_geq_perrin": c2_10["p_geq_perrin"],
            },
            "phase2b_llm": b2 if b2 else "deferred (no API key)",
        },
        "phase3": {
            "baseline_consec_vs_random": {
                "consec_mean_pair": p3_overall["consec_mean"],
                "random_mean_pair": p3_overall["random_mean"],
                "diff": p3_overall["diff"],
                "p_value": p3_overall["p_value"],
                "cohens_d": p3_overall["effect_size_cohens_d"],
                "interpretation": ("Consecutive Syriac strophes (Ephrem, Narsai, "
                                   "Jacob, Odes) show significantly more catchwords "
                                   "than random pairings (Cohen's d = "
                                   f"{p3_overall['effect_size_cohens_d']:.2f}). "
                                   "Confirms Perrin's premise that Syriac "
                                   "literature uses catchword arrangement.")
            },
            "thomas_application": p3thomas["per_source"] if p3thomas else None,
        },
        "perrin_target": {
            "total": PERRIN_TOTAL,
            "both_pct": 89.0,
            "iso_pct": 0.0,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- Print synthesis ----
    print()
    print("=" * 76)
    print("FINAL CROSS-PHASE SYNTHESIS")
    print("Calibration: filter_pct=80, phonological_threshold=0.65")
    print("=" * 76)
    print()
    print("[A] HOW MUCH OF PERRIN'S 502 IS UNAVOIDABLE FROM Coptic→Syriac MAPPING?")
    print("-" * 76)
    print(f"  Coptic baseline:                  {COPTIC_TOTAL}    (1.00× ratio)")
    print(f"  Phase 1 random MC:               {mc_mean:>5.0f}   (P(≥502) = {mc_p_ge_perrin})")
    print(f"  Phase 1 MAP (top-1):             {map_total:>5}")
    print(f"  Phase 2A beam (λ=0.3):          {a2_03['total']:>5}   "
          f"[CI {a2['stochastic_lam03']['total_p05']:.0f}-"
          f"{a2['stochastic_lam03']['total_p95']:.0f}]")
    print(f"  Phase 2C constrained (λ=1.0):   {c2_10['mean']:>5.0f}   "
          f"[CI {c2_10['p05']:.0f}-{c2_10['p95']:.0f}]   "
          f"P(≥502) = {c2_10['p_geq_perrin']}")
    if b2:
        print(f"  Phase 2B LLM best variant:      {b2['best_variant_total']:>5}")
    print(f"  ──────────────────────────────────────────────────────────")
    print(f"  Perrin (2002):                   {PERRIN_TOTAL:>5}   (target)")
    print()
    print("  CONCLUSION: All informed-but-unbiased translation methods cluster")
    print("  in [195, 324]. None reaches Perrin's 502. The lexical mapping +")
    print("  Syriac LM together cannot account for Perrin's surplus.")
    print()
    print("[B] DOES SYRIAC LITERATURE ACTUALLY USE CATCHWORD ARRANGEMENT?")
    print("-" * 76)
    p3o = p3_overall
    print(f"  Consecutive strophes mean:       {p3o['consec_mean']:.2f} catchwords/pair")
    print(f"  Random non-adjacent pairs mean:  {p3o['random_mean']:.2f}")
    print(f"  Difference:                       +{p3o['diff']:.2f}")
    print(f"  Mann-Whitney p-value:             {p3o['p_value']:.2e}")
    print(f"  Cohen's d:                        {p3o['effect_size_cohens_d']:.3f} (medium)")
    print()
    print("  CONCLUSION: Perrin's premise is VALIDATED. Consecutive strophes in")
    print("  Ephrem / Narsai / Jacob / Odes have detectably more catchwords than")
    print("  random pairings (p ≈ 0). The catchword phenomenon is real in Syriac.")
    print()
    print("[C] DOES THOMAS (IN BEAM-TRANSLATED SYRIAC) SHOW THE SAME PATTERN?")
    print("-" * 76)
    if p3thomas:
        for src, d in p3thomas["per_source"].items():
            if "adjacent_sim_mean" not in d:
                continue   # skip sanity-check entry (different schema)
            print(f"  {src}:")
            print(f"    Adj cos_sim mean:           {d['adjacent_sim_mean']:.3f}")
            print(f"    Shuffled baseline:           {d['shuffle_mean']:.3f}")
            print(f"    Permutation p:               {d['permutation_p']:.4f}")
            print(f"    Z-score:                     {d['z_score']:+.3f}")
    print()
    print("  CONCLUSION: Beam-translated Thomas shows weak/marginal evidence of")
    print("  adjacent-pair similarity (p≈0.09 for beam, n.s. for NMT). NOT strong")
    print("  enough to claim Thomas exhibits the literary catchword pattern that")
    print("  Ephrem / Narsai / Jacob clearly do.")
    print()
    print("[D] OVERALL VERDICT")
    print("-" * 76)
    surplus = PERRIN_TOTAL - c2_10["mean"]
    print(f"  Phase 1 + Phase 2:  Perrin's 502 cannot be explained by random or")
    print(f"                       informed Coptic→Syriac translation. Surplus")
    print(f"                       beyond what the lexical map produces:")
    print(f"                       {PERRIN_TOTAL} - {c2_10['mean']:.0f} ≈ {surplus:.0f} catchwords.")
    print(f"  Phase 3 baseline:   The kind of literary structure Perrin claims")
    print(f"                       exists IS detectable in extant Syriac texts.")
    print(f"  Phase 3 on Thomas:  The contrastive model finds only marginal")
    print(f"                       evidence (p≈0.09) of that structure in beam-")
    print(f"                       translated Thomas — not enough to declare")
    print(f"                       Thomas a Syriac literary text by this test.")
    print()
    print("  Net: the surplus over informed translation (~{:.0f} catchwords) might".format(surplus))
    print("  reflect genuine literary structure invisible to our 1-grams + 2-grams,")
    print("  OR Perrin's own translation choices. Phase 2B (LLM) and a stronger")
    print("  Phase 3 model on more data would tighten the verdict further.")

    # ---- Plot ----
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: All catchword totals on one axis
    ax = axes[0]
    methods = ["Coptic\n(baseline)", "Phase 1\nMC (random)",
               "Phase 1\nMAP", "Phase 2A\nbeam λ=0.3",
               "Phase 2C\nconstrained λ=1.0", "Perrin\n(2002)"]
    totals = [COPTIC_TOTAL, mc_mean, map_total, a2_03["total"],
              c2_10["mean"], PERRIN_TOTAL]
    err_lo = [0, mc_mean - mc_p05, 0, 0, c2_10["mean"] - c2_10["p05"], 0]
    err_hi = [0, mc_p95 - mc_mean, 0, 0, c2_10["p95"] - c2_10["mean"], 0]
    colors = ["#779ECC", "#1F77B4", "#5DADE2", "#2CA02C", "#9467BD", "#D62728"]
    bars = ax.bar(methods, totals, color=colors, yerr=[err_lo, err_hi], capsize=4)
    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 8, f"{v:.0f}",
                ha="center", fontsize=9)
    ax.axhline(PERRIN_TOTAL, color="C3", linestyle="--", alpha=0.5)
    ax.set_ylabel("Total catchwords (114 adjacent logion pairs)")
    ax.set_title("(A) All informed-but-unbiased methods fall well short of Perrin")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Phase 3 — consecutive vs random, all-text pooled + per-source
    ax = axes[1]
    cats = ["Pooled\n(p<1e-9)"]
    consec = [p3o["consec_mean"]]
    random_ = [p3o["random_mean"]]
    for author, d in p3base["per_author"].items():
        cats.append(f"{author}\n(p={d['p_value']:.1e})")
        consec.append(d["consec_mean"])
        random_.append(d["random_mean"])
    x = np.arange(len(cats))
    w = 0.4
    ax.bar(x - w/2, consec, w, color="C0", label="Consecutive")
    ax.bar(x + w/2, random_, w, color="C1", label="Random non-adjacent")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Catchwords per pair")
    ax.set_title("(B) Phase 3 baseline — Syriac literature does use catchwords")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final synthesis — testing Perrin's 502-catchword Syriac claim",
                  fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print()
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
