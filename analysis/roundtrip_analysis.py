#!/usr/bin/env python3
"""
Round-trip analysis — produce summary table + figures from results.json
and pair_survival.json.

Inputs:
  data/processed/roundtrip/results.json
  data/processed/roundtrip/pair_survival.json

Outputs:
  analysis/figures/roundtrip_summary.png
  analysis/figures/roundtrip_recovery_ratio.png
  analysis/figures/roundtrip_pair_survival.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RES = REPO_ROOT / "data" / "processed" / "roundtrip" / "results.json"
SURV = REPO_ROOT / "data" / "processed" / "roundtrip" / "pair_survival.json"
OUT_DIR = REPO_ROOT / "analysis" / "figures"

# Reference numbers from the rest of the project (for Thomas comparison)
THOMAS_COPTIC = 235
THOMAS_MC = 195
THOMAS_MAP = 305
THOMAS_BEAM = 320
THOMAS_PERRIN = 502
COLORS = {
    "Ephrem":  "#1F77B4",
    "Jacob":   "#2CA02C",
    "Narsai":  "#9467BD",
    "Solomon": "#E377C2",
}


def main():
    res = json.loads(RES.read_text())
    surv = json.loads(SURV.read_text())
    per = res["per_corpus"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 96)
    print("ROUND-TRIP RECOVERY TABLE")
    print("=" * 96)
    print(f"{'Corpus':<10s} {'Original':>9s} {'Coptic':>8s} "
          f"{'MAP':>6s} {'Beam':>6s} {'MC':>6s}  "
          f"{'r_MAP':>6s} {'r_Beam':>7s} {'r_MC':>6s}")
    for c, d in sorted(per.items()):
        print(f"{c:<10s} {d['original_syriac_total']:>9d} "
              f"{d['coptic_intermediate_total']:>8d} "
              f"{d['recovered_map_total']:>6d} "
              f"{d['recovered_beam_total']:>6d} "
              f"{d['recovered_mc_mean']:>6.0f}  "
              f"{d['ratio_map']:>5.2f}× "
              f"{d['ratio_beam']:>6.2f}× "
              f"{d['ratio_mc']:>5.2f}×")
    print()
    print(f"For reference, Thomas: r_MC={THOMAS_MC/THOMAS_COPTIC:.2f}, "
          f"r_MAP={THOMAS_MAP/THOMAS_COPTIC:.2f}, "
          f"r_BEAM={THOMAS_BEAM/THOMAS_COPTIC:.2f}, "
          f"r_PERRIN={THOMAS_PERRIN/269:.2f} (Perrin's Coptic count was 269)")

    # --- Figure 1: per-stage catchword counts per corpus ---
    fig, ax = plt.subplots(figsize=(11, 6))
    corpora = sorted(per.keys())
    n = len(corpora)
    stages = ["Original\nSyriac", "Coptic\nintermediate",
              "Recovered\nMC mean", "Recovered\nMAP", "Recovered\nbeam"]
    width = 0.16
    x = np.arange(len(stages))
    for i, c in enumerate(corpora):
        d = per[c]
        vals = [d["original_syriac_total"], d["coptic_intermediate_total"],
                d["recovered_mc_mean"], d["recovered_map_total"],
                d["recovered_beam_total"]]
        ax.bar(x + (i - n/2 + 0.5) * width, vals, width,
                color=COLORS.get(c, "gray"), label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Catchword count (sum over consecutive strophe pairs)")
    ax.set_title("Round-trip — catchword counts at each stage, per corpus")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roundtrip_summary.png", dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/roundtrip_summary.png")

    # --- Figure 2: recovery ratios ---
    fig, ax = plt.subplots(figsize=(11, 6))
    methods = ["MAP", "Beam (λ=0.3)", "MC (random)"]
    width = 0.22
    x = np.arange(len(methods))
    for i, c in enumerate(corpora):
        d = per[c]
        vals = [d["ratio_map"], d["ratio_beam"], d["ratio_mc"]]
        ax.bar(x + (i - n/2 + 0.5) * width, vals, width,
                color=COLORS.get(c, "gray"), label=c)
    # Thomas reference lines
    thomas_vals = {
        "MAP":  THOMAS_MAP / THOMAS_COPTIC,
        "Beam (λ=0.3)": THOMAS_BEAM / THOMAS_COPTIC,
        "MC (random)": THOMAS_MC / THOMAS_COPTIC,
    }
    for i, m in enumerate(methods):
        ax.axhline(y=thomas_vals[m], xmin=(i)/len(methods),
                    xmax=(i+1)/len(methods), color="red", linestyle="--",
                    alpha=0.6, linewidth=2,
                    label=("Thomas ratios" if i == 0 else None))
    perrin_thomas_ratio = THOMAS_PERRIN / 269
    ax.axhline(y=perrin_thomas_ratio, color="darkred", linestyle=":",
                linewidth=2, label=f"Perrin Thomas ratio ({perrin_thomas_ratio:.2f}×)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Recovery ratio (recovered / Coptic intermediate)")
    ax.set_title("Round-trip — recovery ratios, by corpus and method, "
                  "vs Thomas's ratios")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roundtrip_recovery_ratio.png", dpi=140,
                 bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/roundtrip_recovery_ratio.png")

    # --- Figure 3: per-pair survival by link type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    surv_per = surv["per_corpus"]
    link_types = ["semantic", "etymological", "phonological"]
    width = 0.20
    x = np.arange(len(link_types))
    for i, c in enumerate(corpora):
        d = surv_per.get(c, {})
        vals = []
        for lt in link_types:
            stats = d.get("by_link_type", {}).get(lt, {"orig": 0, "survived": 0})
            o = stats.get("orig", 0)
            s = stats.get("survived", 0)
            vals.append(100 * s / o if o > 0 else 0)
        ax.bar(x + (i - n/2 + 0.5) * width, vals, width,
                color=COLORS.get(c, "gray"), label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(link_types)
    ax.set_ylabel("Survival % through MAP/MAP round-trip")
    ax.set_title("Round-trip — per-pair catchword survival by link type")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roundtrip_pair_survival.png", dpi=140,
                 bbox_inches="tight")
    print(f"Wrote {OUT_DIR}/roundtrip_pair_survival.png")

    # --- Print survival summary ---
    print()
    print("=" * 80)
    print("PER-PAIR CATCHWORD SURVIVAL")
    print("=" * 80)
    for c, d in sorted(surv_per.items()):
        print(f"\n{c}:")
        total_o = total_s = 0
        for lt in link_types:
            s = d["by_link_type"].get(lt, {})
            o = s.get("orig", 0)
            sv = s.get("survived", 0)
            total_o += o; total_s += sv
            if o > 0:
                print(f"  {lt:<14s}: {sv}/{o} survived ({100*sv/o:.1f}%)")
        if total_o:
            print(f"  TOTAL         : {total_s}/{total_o} ({100*total_s/total_o:.1f}%)")


if __name__ == "__main__":
    main()
