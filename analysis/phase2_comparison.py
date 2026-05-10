#!/usr/bin/env python3
"""
Phase 2 integration — compare all four translation methods at the SAME
calibration (filter_pct=80, threshold=0.65) and produce a single table +
figure.

Methods:
  Phase 1 MC      — random translation from EM lexical map distribution
  Phase 1 MAP     — top-1 deterministic from EM lexical map
  Phase 2A beam   — beam search over lexical map + Syriac bigram LM
  Phase 2B LLM    — Claude (claude-sonnet-4-20250514), 20 variants × 115 logia
  Phase 2C constr.— per-logion stochastic sampling, 1000 sims (LM-informed)
  Perrin (2002)   — manual retroversion (the claim under test)

Inputs:
  data/processed/monte_carlo_results.json
  data/processed/detector_calibration.csv  (Phase 1 MAP at calibration point)
  data/processed/phase2a_beam_results.json
  data/processed/phase2b_llm_results.json     (optional)
  data/processed/phase2c_constrained_results.json

Outputs:
  data/processed/phase2_all_methods_summary.json
  analysis/figures/phase2_all_methods.png
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
P_MC   = REPO_ROOT / "data" / "processed" / "monte_carlo_results.json"
P_CALIB = REPO_ROOT / "data" / "processed" / "detector_calibration.csv"
P_2A   = REPO_ROOT / "data" / "processed" / "phase2a_beam_results.json"
P_2B   = REPO_ROOT / "data" / "processed" / "phase2b_llm_results.json"
P_2C   = REPO_ROOT / "data" / "processed" / "phase2c_constrained_results.json"
OUT_JSON = REPO_ROOT / "data" / "processed" / "phase2_all_methods_summary.json"
OUT_FIG  = REPO_ROOT / "analysis" / "figures" / "phase2_all_methods.png"

# Perrin's reported numbers + Coptic baseline (calibrated to filter_pct=80)
PERRIN_TOTAL = 502
PERRIN_BOTH  = 89.0
PERRIN_ONE   = 11.0
PERRIN_ISO   = 0.0
COPTIC_TOTAL = 235        # from calibration row (filter_pct=80, threshold=0.65)
COPTIC_BOTH  = 53.9
COPTIC_ONE   = 34.8
COPTIC_ISO   = 11.3


def load_calibration_map():
    """Phase 1 MAP at filter_pct=80, threshold=0.65."""
    if not P_CALIB.exists():
        return None
    with P_CALIB.open() as f:
        for row in csv.DictReader(f):
            if (float(row["filter_pct"]) == 80
                    and abs(float(row["threshold"]) - 0.65) < 1e-6):
                return {
                    "total": int(row["syriac_total"]),
                    "both_pct": float(row["syriac_both_pct"]),
                    "one_pct":  float(row["syriac_one_pct"]),
                    "iso_pct":  float(row["syriac_iso_pct"]),
                }
    return None


def main():
    rows = []

    # Coptic baseline (the floor that any Syriac comparison must clear)
    rows.append({
        "method": "Coptic (automated, calib)",
        "total": COPTIC_TOTAL, "total_low": None, "total_high": None,
        "both": COPTIC_BOTH, "iso": COPTIC_ISO, "one": COPTIC_ONE,
        "color": "#779ECC",
    })

    # Phase 1 MC
    if P_MC.exists():
        mc = json.loads(P_MC.read_text())
        m = mc["overall"]["total"]
        rows.append({
            "method": "Phase 1: random MC",
            "total": round(m["mean"], 1),
            "total_low": round(m["p05"], 1),
            "total_high": round(m["p95"], 1),
            "both": mc["overall"]["both_sides_pct"]["mean"],
            "iso":  mc["overall"]["isolated_pct"]["mean"],
            "one":  mc["overall"]["one_side_pct"]["mean"],
            "p_geq_perrin": mc.get("perrin_p_geq_total"),
            "color": "#1F77B4",
        })

    # Phase 1 MAP at calibration point
    cal = load_calibration_map()
    if cal:
        rows.append({
            "method": "Phase 1: MAP",
            "total": cal["total"], "total_low": None, "total_high": None,
            "both": cal["both_pct"], "iso": cal["iso_pct"], "one": cal["one_pct"],
            "color": "#5DADE2",
        })

    # Phase 2A beam search at lambda=0.3 (representative middle setting)
    if P_2A.exists():
        a2 = json.loads(P_2A.read_text())
        # Best beam at lambda=0.3
        beam_03 = a2["best_beam_per_lambda"].get("0.3", {})
        # Stochastic at lambda=0.3
        stoch = a2.get("stochastic_lam03", {})
        rows.append({
            "method": "Phase 2A: beam (λ=0.3)",
            "total": beam_03.get("total"),
            "total_low": stoch.get("total_p05"),
            "total_high": stoch.get("total_p95"),
            "both": beam_03.get("both_pct"),
            "iso":  beam_03.get("iso_pct"),
            "one":  beam_03.get("one_pct"),
            "color": "#2CA02C",
        })

    # Phase 2C constrained sampling at lambda=1.0 (heavy LM)
    if P_2C.exists():
        c2 = json.loads(P_2C.read_text())
        r1 = c2["results_per_lambda"].get("1.0", {})
        rows.append({
            "method": "Phase 2C: constrained (λ=1.0)",
            "total": round(r1.get("mean", 0), 1),
            "total_low": r1.get("p05"),
            "total_high": r1.get("p95"),
            "both": r1.get("both_mean"),
            "iso":  r1.get("iso_mean"),
            "one":  r1.get("one_mean"),
            "p_geq_perrin": r1.get("p_geq_perrin"),
            "color": "#9467BD",
        })

    # Phase 2B LLM (if available)
    if P_2B.exists():
        b2 = json.loads(P_2B.read_text())
        rows.append({
            "method": "Phase 2B: LLM (best variant)",
            "total": b2.get("best_variant_total"),
            "total_low": None, "total_high": None,
            "both": b2.get("best_variant_both_pct"),
            "iso":  b2.get("best_variant_iso_pct"),
            "one":  None,
            "color": "#E377C2",
        })
        rows.append({
            "method": "Phase 2B: LLM (cross-pair mean)",
            "total": round(b2.get("cross_pair_total_mean", 0), 1),
            "total_low": None, "total_high": None,
            "both": None, "iso": None, "one": None,
            "color": "#F7B6D2",
        })

    # Perrin (the target)
    rows.append({
        "method": "Perrin (2002)",
        "total": PERRIN_TOTAL, "total_low": None, "total_high": None,
        "both": PERRIN_BOTH, "iso": PERRIN_ISO, "one": PERRIN_ONE,
        "color": "#D62728",
    })

    # ---- Print table ----
    print()
    print("=" * 92)
    print(f"  {'Method':<32s} {'Total':>10s} {'CI95':>16s} {'Both%':>7s} "
          f"{'One%':>6s} {'Iso%':>6s}")
    print("-" * 92)
    def fmt_pct(v):
        return "-" if v is None else f"{v:.1f}"

    for r in rows:
        ci = ""
        if r.get("total_low") is not None and r.get("total_high") is not None:
            ci = f"[{r['total_low']:.0f}-{r['total_high']:.0f}]"
        b = fmt_pct(r["both"]); o = fmt_pct(r["one"]); i = fmt_pct(r["iso"])
        print(f"  {r['method']:<32s} {str(r['total']):>10s} {ci:>16s} "
              f"{b:>7s} {o:>6s} {i:>6s}")
    print("=" * 92)

    # ---- Interpretation ----
    print()
    print("INTERPRETATION (calibration: filter_pct=80, phon_threshold=0.65)")
    syr_methods = [r for r in rows if r["method"].startswith("Phase ")
                   and r["total"] is not None]
    if syr_methods:
        max_phase = max(syr_methods, key=lambda r: float(r["total"]))
        ratio = float(max_phase["total"]) / PERRIN_TOTAL
        print(f"  Best Syriac method:  {max_phase['method']} = {max_phase['total']}")
        print(f"  Perrin's claim:       {PERRIN_TOTAL}")
        print(f"  Best/Perrin ratio:   {ratio:.2f}× ")
        print(f"  Coptic baseline:     {COPTIC_TOTAL}")
    if any(r.get("p_geq_perrin") is not None for r in rows):
        print()
        print("  Probability of reaching Perrin's 502 under each null:")
        for r in rows:
            if r.get("p_geq_perrin") is not None:
                print(f"    {r['method']:<32s} P(≥502) = {r['p_geq_perrin']:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump({
            "calibration": {"filter_pct": 80, "phon_threshold": 0.65},
            "rows": rows,
            "perrin_target": {"total": PERRIN_TOTAL,
                               "both_pct": PERRIN_BOTH,
                               "iso_pct": PERRIN_ISO},
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # ---- Plot ----
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plottable = [r for r in rows if r["total"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 2 — Coptic→Syriac translation methods, all at "
                  "filter_pct=80 / threshold=0.65", fontsize=11)

    # Panel A: Total catchwords
    ax = axes[0]
    labels = [r["method"] for r in plottable]
    totals = [float(r["total"]) for r in plottable]
    colors = [r["color"] for r in plottable]
    err_lo = [(float(r["total"]) - float(r["total_low"]))
                if r.get("total_low") is not None else 0 for r in plottable]
    err_hi = [(float(r["total_high"]) - float(r["total"]))
                if r.get("total_high") is not None else 0 for r in plottable]
    bars = ax.barh(labels, totals, color=colors,
                    xerr=[err_lo, err_hi], capsize=4)
    for b, v in zip(bars, totals):
        ax.text(v + 8, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                 va="center", fontsize=9)
    ax.axvline(PERRIN_TOTAL, color="C3", linestyle="--", alpha=0.5,
                label=f"Perrin {PERRIN_TOTAL}")
    ax.set_xlabel("Total catchwords (114 adjacent pairs)")
    ax.set_title("(A) Total catchwords by method")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    # Panel B: Connectivity (Both% and Iso%)
    ax = axes[1]
    plot_conn = [r for r in plottable if r.get("both") is not None]
    x = np.arange(len(plot_conn))
    w = 0.4
    boths = [r["both"] for r in plot_conn]
    isos  = [r["iso"]  for r in plot_conn]
    cs    = [r["color"] for r in plot_conn]
    ax.bar(x - w/2, boths, w, color=cs, alpha=0.85, label="Both-sides %")
    ax.bar(x + w/2, isos,  w, color=cs, alpha=0.45, label="Isolated %",
            hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"].replace("Phase ", "P") for r in plot_conn],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of logia")
    ax.set_title("(B) Connectivity")
    ax.axhline(PERRIN_BOTH, color="C3", linestyle="--", alpha=0.5,
                label="Perrin both 89%")
    ax.axhline(PERRIN_ISO, color="C3", linestyle=":", alpha=0.5,
                label="Perrin iso 0%")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
