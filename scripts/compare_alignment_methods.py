#!/usr/bin/env python3
"""
Print a side-by-side IBM-1-vs-BinaryAlign summary for every map-dependent
experiment we re-ran during the alignment-module migration.

Reads from:
  data/processed/_ibm1_baseline/*    (snapshot taken before the swap)
  data/processed/*                   (current outputs, post-swap)

Outputs a single table to stdout, ready to drop into FINDINGS.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data" / "processed" / "_ibm1_baseline"
C = ROOT / "data" / "processed"


def load_json(p: Path):
    return json.load(p.open()) if p.exists() else None


def fmt_delta(old, new, fmt="{:.1f}"):
    if old is None or new is None:
        return "—"
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{fmt.format(new)}  (Δ {sign}{fmt.format(d).lstrip(' ')})"


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    section("Phase 1 — Monte Carlo (10,000 sims, λ=0.65, fp=80)")
    b1 = load_json(B / "monte_carlo_results.json")
    c1 = load_json(C / "monte_carlo_results.json")
    if b1 and c1:
        print(f"  {'':18s}  {'IBM-1':>10s}  {'BinaryAlign':>12s}  {'Δ':>10s}")
        for key in ("total", "both_sides_pct", "isolated_pct"):
            bv = b1["overall"][key]
            cv = c1["overall"][key]
            print(f"  {key:18s}  mean={bv['mean']:7.1f}  mean={cv['mean']:7.1f}  Δ={cv['mean']-bv['mean']:+7.1f}")
        # P(MC total >= 502)
        import numpy as np
        try:
            bt = np.load(B / "monte_carlo_pair_totals.npy")
            ct = np.load(C / "monte_carlo_pair_totals.npy")
            print(f"  P(total ≥ 502):     IBM-1={np.mean(bt >= 502):.4f}  "
                  f"BinaryAlign={np.mean(ct >= 502):.4f}")
        except FileNotFoundError:
            pass

    section("Phase 2A — Beam search (best per λ)")
    b2a = load_json(B / "phase2a_beam_results.json")
    c2a = load_json(C / "phase2a_beam_results.json")
    if b2a and c2a:
        print(f"  {'λ':>4s}  {'IBM-1 total':>12s}  {'BinaryAlign total':>18s}  {'Δ':>6s}")
        for lam in sorted(b2a["best_beam_per_lambda"], key=float):
            bv = b2a["best_beam_per_lambda"].get(lam, {}).get("total")
            cv = c2a["best_beam_per_lambda"].get(lam, {}).get("total")
            if bv is not None and cv is not None:
                print(f"  {lam:>4s}  {bv:>12d}  {cv:>18d}  {cv-bv:+6d}")

    section("Phase 2C — Constrained stochastic sampling")
    b2c = load_json(B / "phase2c_constrained_results.json")
    c2c = load_json(C / "phase2c_constrained_results.json")
    if b2c and c2c:
        print(f"  {'λ':>4s}  {'IBM-1 mean':>10s}  {'BinaryAlign mean':>18s}  {'Δ':>6s}  P(≥502) IBM/Bin")
        for lam in sorted(b2c["results_per_lambda"], key=float):
            br = b2c["results_per_lambda"].get(lam, {})
            cr = c2c["results_per_lambda"].get(lam, {})
            bv = br.get("mean")
            cv = cr.get("mean")
            if bv is not None and cv is not None:
                print(f"  {lam:>4s}  {bv:>10.1f}  {cv:>18.1f}  {cv-bv:+6.1f}  "
                      f"{br.get('p_geq_perrin',0):.4f} / {cr.get('p_geq_perrin',0):.4f}")

    section("Detector calibration sweep (Coptic count is map-invariant; only Syriac column moves)")
    b_csv = B / "detector_calibration.csv"
    c_csv = C / "detector_calibration.csv"
    if b_csv.exists() and c_csv.exists():
        b_rows = {(r["filter_pct"], r["threshold"]): r for r in csv.DictReader(b_csv.open())}
        c_rows = {(r["filter_pct"], r["threshold"]): r for r in csv.DictReader(c_csv.open())}
        print(f"  {'fp':>3s} {'thr':>5s}  {'Coptic':>7s}  {'Syriac IBM-1':>13s}  {'Syriac BinAln':>14s}  {'Δ Syr':>6s}")
        for key, br in sorted(b_rows.items()):
            cr = c_rows.get(key)
            if cr is None:
                continue
            try:
                d = int(cr["syriac_total"]) - int(br["syriac_total"])
                print(f"  {br['filter_pct']:>3s} {br['threshold']:>5s}  {br['coptic_total']:>7s}  "
                      f"{br['syriac_total']:>13s}  {cr['syriac_total']:>14s}  {d:+6d}")
            except ValueError:
                pass

    section("Round-trip (per corpus, MAP ratio)")
    br = load_json(B / "roundtrip" / "results.json")
    cr = load_json(C / "roundtrip" / "results.json")
    if br and cr:
        print(f"  {'corpus':>10s}  {'IBM-1 MAP':>10s}  {'BinaryAlign MAP':>16s}  {'Δ':>6s}")
        for corp in br.get("per_corpus", {}):
            bv = br["per_corpus"][corp]["ratio_map"]
            cv = cr.get("per_corpus", {}).get(corp, {}).get("ratio_map")
            if cv is not None:
                print(f"  {corp:>10s}  {bv:>10.3f}  {cv:>16.3f}  {cv-bv:+6.3f}")


if __name__ == "__main__":
    main()
