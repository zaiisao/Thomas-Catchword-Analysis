#!/usr/bin/env python3
"""
Calibrate the catchword detector against Perrin's reported numbers.

Sensitivity analysis along two dimensions:
  1. phonological_threshold ∈ {0.65, 0.70, 0.75, 0.80, 0.85}
  2. high-frequency-lemma filter ∈ {100, 80, 60, 40, 30, 25, 20} (% cap)

Records, per cell:
  - Coptic catchword count (no translation, Coptic profile)
  - Syriac catchword count (MAP-translated via lexical map, Syriac profile)
  - S/C ratio
  - Connectivity stats (% both-sides, % one-side, % isolated) for each language

Output:
  data/processed/detector_calibration.csv

The row where Coptic count ≈ 269 (Perrin's reported Coptic total) is the
calibration point of interest: we then ask what the Syriac count is at
the SAME threshold and filter.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LEX = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
OUT_CSV = REPO_ROOT / "data" / "processed" / "detector_calibration.csv"

THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85)
FILTER_PCTS = (100, 80, 60, 40, 30, 25, 20)
COPTIC_CONTENT_POS = {"N", "NPROP", "V", "VBD", "VSTAT", "VIMP", "ADJ", "ADV"}


def load_data():
    map_top: dict[str, str] = {}
    for line in LEX.open():
        r = json.loads(line)
        if r["candidates"]:
            map_top[r["coptic_lemma"]] = r["candidates"][0]["syriac_lemma"]

    logia: dict[int, list[dict]] = defaultdict(list)
    for line in THOMAS.open():
        r = json.loads(line)
        logia[r["logion"]].extend(r["tokens"])
    return map_top, dict(logia)


def lemma_logia_freq(logia, content_pos):
    """How many distinct logia each lemma appears in."""
    freq = Counter()
    for L, toks in logia.items():
        seen = {t["lemma"] for t in toks
                if t.get("lemma") and t.get("pos") in content_pos}
        for lem in seen:
            freq[lem] += 1
    return freq


def map_translate(logia, map_top):
    syr_logia = {}
    for L, toks in logia.items():
        syr = []
        for t in toks:
            cl = t.get("lemma")
            if cl and cl in map_top:
                syr.append({"lemma": map_top[cl], "form": map_top[cl],
                            "parse": "MS-EMP", "gloss": ""})
        syr_logia[L] = syr
    return syr_logia


def syriac_logia_freq(syr_logia):
    freq = Counter()
    for L, toks in syr_logia.items():
        seen = {t["lemma"] for t in toks if t.get("lemma")}
        for lem in seen:
            freq[lem] += 1
    return freq


def measure(detector, logia_dict, sorted_L, freq, max_logia_pct):
    n_logia = len(sorted_L)
    cutoff = max_logia_pct * n_logia / 100.0
    blocked = {lem for lem, n in freq.items() if n > cutoff}

    total = 0
    left = set()
    right = set()
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        ta = [t for t in logia_dict[L] if t.get("lemma") not in blocked]
        tb = [t for t in logia_dict[Ln] if t.get("lemma") not in blocked]
        cws = detector.detect(ta, tb)
        total += len(cws)
        if cws:
            right.add(L)
            left.add(Ln)
    both = left & right
    iso = set(sorted_L) - left - right
    one_side = (left ^ right)
    return {
        "total": total,
        "both_pct":  100 * len(both)     / n_logia,
        "one_pct":   100 * len(one_side) / n_logia,
        "iso_pct":   100 * len(iso)      / n_logia,
        "blocked": len(blocked),
    }


def main():
    print("Loading data…")
    map_top, logia = load_data()
    sorted_L = sorted(logia.keys())
    print(f"  {len(logia)} logia, {len(map_top)} lexical-map entries")

    syr_logia = map_translate(logia, map_top)
    c_freq = lemma_logia_freq(logia, COPTIC_CONTENT_POS)
    s_freq = syriac_logia_freq(syr_logia)

    rows = []
    print(f"\nSweeping {len(FILTER_PCTS)} × {len(THRESHOLDS)} = "
          f"{len(FILTER_PCTS) * len(THRESHOLDS)} cells…")

    for fpct in FILTER_PCTS:
        for thr in THRESHOLDS:
            det_c = CatchwordDetector("coptic", phonological_threshold=thr)
            det_s = CatchwordDetector("syriac", phonological_threshold=thr)
            c = measure(det_c, logia, sorted_L, c_freq, fpct)
            s = measure(det_s, syr_logia, sorted_L, s_freq, fpct)
            rows.append({
                "filter_pct": fpct,
                "threshold": thr,
                "coptic_total": c["total"],
                "syriac_total": s["total"],
                "ratio_s_over_c": (s["total"] / c["total"]) if c["total"] else 0.0,
                "coptic_both_pct": round(c["both_pct"], 1),
                "coptic_one_pct":  round(c["one_pct"],  1),
                "coptic_iso_pct":  round(c["iso_pct"],  1),
                "syriac_both_pct": round(s["both_pct"], 1),
                "syriac_one_pct":  round(s["one_pct"],  1),
                "syriac_iso_pct":  round(s["iso_pct"],  1),
                "coptic_blocked_lemmas": c["blocked"],
                "syriac_blocked_lemmas": s["blocked"],
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {OUT_CSV}")

    # Find row closest to Perrin's Coptic baseline 269
    closest = min(rows, key=lambda r: abs(r["coptic_total"] - 269))
    print()
    print("Calibration point (row where Coptic count is closest to Perrin's 269):")
    print(f"  filter_pct={closest['filter_pct']}, threshold={closest['threshold']}")
    print(f"  Coptic catchwords:  {closest['coptic_total']:>4} (Perrin: 269)")
    print(f"  Syriac catchwords:  {closest['syriac_total']:>4} (Perrin: 502)")
    print(f"  S/C ratio:          {closest['ratio_s_over_c']:>4.2f} (Perrin: 1.87)")
    print(f"  Coptic conn:        {closest['coptic_both_pct']}% both / "
          f"{closest['coptic_one_pct']}% one / {closest['coptic_iso_pct']}% iso "
          f"(Perrin: 49 / 40 / 11)")
    print(f"  Syriac conn:        {closest['syriac_both_pct']}% both / "
          f"{closest['syriac_one_pct']}% one / {closest['syriac_iso_pct']}% iso "
          f"(Perrin: 89 / 11 / 0)")


if __name__ == "__main__":
    main()
