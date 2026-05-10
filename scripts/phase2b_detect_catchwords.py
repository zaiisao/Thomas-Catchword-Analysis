#!/usr/bin/env python3
"""
Phase 2B Steps 5-7 — process LLM Syriac translations, run the Phase 1
catchword detector at the calibrated settings, and compute ratios + control
comparison.

Inputs:
  data/processed/llm_translations/*.json
  data/external/sedra/peshitta_list.txt
Output:
  data/processed/phase2b_results.json
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LLM_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
SEDRA = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
OUT = REPO_ROOT / "data" / "processed" / "phase2b_results.json"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
COPTIC_BASELINE_TOTAL = 235  # from Phase 1 calibration row

SYRIAC_RE = re.compile(r"[܀-ݏ]")
COMMENTARY_MARKERS = ("translation", "note:", "meaning:", "literally:",
                       "this translates", "the syriac", "here is", "coptic text",
                       "english:", "english translation")


def is_usable_syriac(text):
    if not text or not text.strip():
        return False
    syr = SYRIAC_RE.findall(text)
    n_syr = len(syr)
    total = len(re.sub(r"\s", "", text))
    if total == 0 or n_syr / total < 0.5:
        return False
    low = text.lower()
    for m in COMMENTARY_MARKERS:
        if m in low:
            return False
    return n_syr >= 5


def strip_voc(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def tokenize_syriac(text):
    """Strip vocalization, split on whitespace; remove punctuation chars."""
    cleaned = re.sub(r"[^܀-ݏ\s]", "", text)
    cleaned = strip_voc(cleaned)
    cleaned = re.sub(r"[܀-܏]", " ", cleaned)  # Syriac punctuation block
    tokens = []
    for w in cleaned.split():
        w = w.strip()
        if w:
            tokens.append(w)
    return tokens


def load_sedra_lookup():
    """unpointed surface → lemma."""
    out = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4: continue
            unp, lem = parts[1].strip(), parts[3].strip()
            if unp and lem and unp not in out:
                out[unp] = lem
    return out


def make_tokens(text, sedra):
    return [{"form": t, "lemma": sedra.get(t, t), "parse": "MS-EMP"}
            for t in tokenize_syriac(text)]


def load_translations():
    out = []
    for path in sorted(LLM_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        out.append(d)
    return out


def compute_blocked(strophes, filter_pct):
    """{lemma: count of strophes containing it}; block if > filter_pct%."""
    n = len(strophes)
    cutoff = filter_pct * n / 100.0
    cnt = Counter()
    for s in strophes:
        for lem in {t["lemma"] for t in s}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def filter_blocked(toks, blocked):
    return [t for t in toks if t["lemma"] not in blocked]


def main():
    if not LLM_DIR.exists() or not list(LLM_DIR.glob("*.json")):
        sys.exit(f"No translations found in {LLM_DIR}. "
                  f"Run scripts/phase2b_qwen_translate.py first.")

    print("Loading SEDRA…")
    sedra = load_sedra_lookup()
    print(f"  {len(sedra)} entries")

    print("Loading translations…")
    translations = load_translations()
    print(f"  {len(translations)} passages")

    # Split Thomas vs control
    thomas = sorted([t for t in translations if not t.get("is_control")],
                     key=lambda d: d.get("logion_number", 0))
    controls = sorted([t for t in translations if t.get("is_control")],
                       key=lambda d: d.get("passage_id", ""))
    print(f"  Thomas logia:    {len(thomas)}")
    print(f"  Control passages:{len(controls)}")

    # Process variants → token lists; filter to usable Syriac only
    for entry in thomas + controls:
        proc = []
        for v in entry["variants"]:
            if not v.get("success"): continue
            txt = v.get("syriac_text", "")
            if not is_usable_syriac(txt): continue
            proc.append(make_tokens(txt, sedra))
        entry["processed_variants"] = proc

    # Diagnostic
    n_good = [len(e["processed_variants"]) for e in thomas]
    print(f"\nThomas: usable variants per logion: "
          f"min={min(n_good)}, median={int(np.median(n_good))}, "
          f"max={max(n_good)}")
    n_logia_with_lt10 = sum(1 for n in n_good if n < 10)
    if n_logia_with_lt10:
        print(f"  WARNING: {n_logia_with_lt10} logia have < 10 usable variants")

    n_good_c = [len(e["processed_variants"]) for e in controls]
    print(f"Controls: usable variants per passage: "
          f"min={min(n_good_c)}, median={int(np.median(n_good_c))}, "
          f"max={max(n_good_c)}")

    # ---- Compute blocked set from variant 0 of each Thomas logion ----
    canonical_thomas = []
    for entry in thomas:
        if entry["processed_variants"]:
            canonical_thomas.append(entry["processed_variants"][0])
        else:
            canonical_thomas.append([])
    blocked = compute_blocked(canonical_thomas, FILTER_PCT)
    print(f"\nBlocked lemmas (filter_pct={FILTER_PCT}%): {len(blocked)}")
    if blocked:
        print(f"  Sample: {sorted(blocked)[:10]}")

    det = CatchwordDetector("syriac",
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)

    # ---- Per-pair cross-product analysis (variants × variants) ----
    per_pair_dist = {}
    print("\n=== Per-pair cross-product (variants × variants) ===")
    for i in range(len(thomas) - 1):
        a = thomas[i]; b = thomas[i + 1]
        var_a = [filter_blocked(v, blocked) for v in a["processed_variants"]]
        var_b = [filter_blocked(v, blocked) for v in b["processed_variants"]]
        if not var_a or not var_b:
            per_pair_dist[f"{a['passage_id']}-{b['passage_id']}"] = {
                "mean": 0, "std": 0, "n_combinations": 0,
            }
            continue
        counts = []
        for va in var_a:
            for vb in var_b:
                counts.append(len(det.detect(va, vb)))
        per_pair_dist[f"{a['passage_id']}-{b['passage_id']}"] = {
            "mean": float(np.mean(counts)),
            "std":  float(np.std(counts)),
            "median": float(np.median(counts)),
            "p05":  float(np.percentile(counts, 5)),
            "p95":  float(np.percentile(counts, 95)),
            "n_combinations": len(counts),
        }

    total_mean = sum(p["mean"] for p in per_pair_dist.values())

    # ---- Canonical (variant 0) total and connectivity ----
    canonical_filtered = [filter_blocked(toks, blocked) for toks in canonical_thomas]
    total_cws = 0
    cw_left = set(); cw_right = set()
    n_logia = len(canonical_filtered)
    for i in range(n_logia - 1):
        cws = det.detect(canonical_filtered[i], canonical_filtered[i + 1])
        n = len(cws)
        total_cws += n
        if n > 0:
            cw_right.add(i)
            cw_left.add(i + 1)
    both = len(cw_left & cw_right)
    iso = n_logia - len(cw_left | cw_right)
    one = n_logia - both - iso

    # ---- Control: pair adjacent control passages, cross-product ----
    print("\n=== Control passages (adjacent pairs cross-product) ===")
    control_pair_means = []
    for i in range(len(controls) - 1):
        a = controls[i]; b = controls[i + 1]
        var_a = [filter_blocked(v, blocked) for v in a["processed_variants"]]
        var_b = [filter_blocked(v, blocked) for v in b["processed_variants"]]
        if not var_a or not var_b:
            continue
        counts = [len(det.detect(va, vb)) for va in var_a for vb in var_b]
        control_pair_means.append(float(np.mean(counts)))

    thomas_pair_means = [p["mean"] for p in per_pair_dist.values()]
    if control_pair_means:
        u, p_thomas_vs_ctrl = stats.mannwhitneyu(
            thomas_pair_means, control_pair_means, alternative="greater")
    else:
        u, p_thomas_vs_ctrl = float("nan"), float("nan")

    ratio_canon = total_cws / COPTIC_BASELINE_TOTAL
    ratio_mean  = total_mean / COPTIC_BASELINE_TOTAL

    print()
    print("=" * 70)
    print("PHASE 2B (LLM) RESULTS")
    print("=" * 70)
    print(f"\nLLM canonical (variant 0):  {total_cws} catchwords  "
          f"(ratio {ratio_canon:.2f}× over Coptic baseline of {COPTIC_BASELINE_TOTAL})")
    print(f"  both-sides:   {100*both/n_logia:.1f}%")
    print(f"  one-side:     {100*one/n_logia:.1f}%")
    print(f"  isolated:     {100*iso/n_logia:.1f}%")
    print(f"\nLLM mean across variants:   {total_mean:.0f}  (ratio {ratio_mean:.2f}×)")
    print()
    print("Reference ladder:")
    print("  0.83×  Phase 1 MC (random)")
    print("  1.23×  Round-trip ceiling (max from known-catchword text)")
    print("  1.30×  Phase 1 MAP")
    print("  1.36×  Phase 2A beam")
    print(f"  {ratio_canon:.2f}×  ← Phase 2B LLM (this result, canonical)")
    print(f"  {ratio_mean:.2f}×  ← Phase 2B LLM (mean of variants)")
    print("  1.87×  Perrin")
    print()
    if control_pair_means:
        print(f"Control comparison:")
        print(f"  Thomas mean per-pair:   {np.mean(thomas_pair_means):.2f}")
        print(f"  Control mean per-pair:  {np.mean(control_pair_means):.2f}")
        print(f"  Mann-Whitney U p:       {p_thomas_vs_ctrl:.4f}")

    # ---- Save ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "config": {"phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT,
                    "coptic_baseline_total": COPTIC_BASELINE_TOTAL,
                    "n_blocked_lemmas": len(blocked)},
        "n_logia": n_logia,
        "canonical_total_cws": total_cws,
        "canonical_ratio": ratio_canon,
        "canonical_both_pct": 100 * both / n_logia,
        "canonical_one_pct":  100 * one / n_logia,
        "canonical_iso_pct":  100 * iso / n_logia,
        "mean_total_cws": total_mean,
        "mean_ratio": ratio_mean,
        "thomas_mean_per_pair": float(np.mean(thomas_pair_means)) if thomas_pair_means else None,
        "control_mean_per_pair": float(np.mean(control_pair_means)) if control_pair_means else None,
        "thomas_vs_control_p": float(p_thomas_vs_ctrl) if not np.isnan(p_thomas_vs_ctrl) else None,
        "per_pair": per_pair_dist,
        "n_thomas_passages": len(thomas),
        "n_control_passages": len(controls),
        "n_thomas_usable_variants_min": min(n_good),
        "n_thomas_usable_variants_median": int(np.median(n_good)),
        "n_thomas_usable_variants_max": max(n_good),
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
