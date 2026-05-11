#!/usr/bin/env python3
"""
Aggregate density: Proverbs adjacent-pair catchword count vs control adjacent-
pair count, per language, length-normalised.

Outputs:
  data/proverbs/aggregate_density.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402
from scripts.proverbs_permutation_test import (  # noqa: E402
    load_translations, compute_blocked,
    PHON_THRESHOLD, FILTER_PCT, HEB_FILE, TRANS_DIR, make_tokens,
)

CTRL_FILE     = REPO_ROOT / "data" / "proverbs" / "controls_hebrew.json"
CTRL_TRANS_DIR = REPO_ROOT / "data" / "proverbs" / "control_translations"
OUT = REPO_ROOT / "data" / "proverbs" / "aggregate_density.json"

LANGS = ["hebrew", "greek", "syriac", "aramaic", "arabic"]


def load_proverbs_translations(lang: str, variant_idx: int = 0
                                ) -> dict[int, list[dict]]:
    return load_translations(lang, variant_idx)


def load_control_translations(lang: str, variant_idx: int = 0
                               ) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    if lang == "hebrew":
        if not CTRL_FILE.exists():
            return out
        d = json.loads(CTRL_FILE.read_text(encoding="utf-8"))
        for r in d:
            if r.get("hebrew_text"):
                out[r["unit_id"]] = make_tokens(r["hebrew_text"], "hebrew")
        return out
    lang_dir = CTRL_TRANS_DIR / lang
    if not lang_dir.exists():
        return out
    for path in sorted(lang_dir.glob("unit_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        uid = d.get("unit_id")
        variants = d.get("variants", [])
        if uid is None or variant_idx >= len(variants):
            continue
        v = variants[variant_idx]
        if v.get("success"):
            out[uid] = make_tokens(v.get("text", ""), lang)
    return out


def per_pair_counts(translations, blocked, lang):
    det = CatchwordDetector(lang, phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    ids = sorted(translations.keys())
    rows = []
    for i in range(len(ids) - 1):
        a, b = ids[i], ids[i + 1]
        ta = [t for t in translations[a] if t["lemma"] not in blocked]
        tb = [t for t in translations[b] if t["lemma"] not in blocked]
        if not ta or not tb:
            rows.append({"a": a, "b": b, "n_catchwords": 0, "density": 0.0,
                         "len_a": len(ta), "len_b": len(tb)})
            continue
        cws = det.detect(ta, tb)
        n = len(cws)
        denom = max(len(ta) * len(tb), 1)
        density = n / denom * 10000
        rows.append({"a": a, "b": b, "n_catchwords": n, "density": density,
                     "len_a": len(ta), "len_b": len(tb)})
    return rows


def main() -> None:
    out: dict[str, dict] = {}
    for lang in LANGS:
        print(f"\n=== {lang.upper()} ===")
        p_trans = load_proverbs_translations(lang)
        c_trans = load_control_translations(lang)
        print(f"  loaded Proverbs: {len(p_trans)}, controls: {len(c_trans)}")
        if not p_trans:
            out[lang] = {"skipped": True}; continue
        blocked = compute_blocked(p_trans, FILTER_PCT)
        print(f"  blocked {len(blocked)} lemmas")
        p_pairs = per_pair_counts(p_trans, blocked, lang)
        c_pairs = per_pair_counts(c_trans, blocked, lang) if c_trans else []
        p_n = [r["n_catchwords"] for r in p_pairs]
        c_n = [r["n_catchwords"] for r in c_pairs]
        p_d = [r["density"]      for r in p_pairs]
        c_d = [r["density"]      for r in c_pairs]
        p_l = [(r["len_a"] + r["len_b"]) / 2 for r in p_pairs]
        c_l = [(r["len_a"] + r["len_b"]) / 2 for r in c_pairs]
        if p_n and c_n:
            _, p_raw = stats.mannwhitneyu(p_n, c_n, alternative="greater")
            _, p_den = stats.mannwhitneyu(p_d, c_d, alternative="greater")
        else:
            p_raw = p_den = float("nan")
        out[lang] = {
            "skipped": False,
            "n_proverbs_pairs": len(p_pairs),
            "n_control_pairs": len(c_pairs),
            "proverbs_total": int(sum(p_n)),
            "control_total":  int(sum(c_n)),
            "proverbs_mean_per_pair": float(np.mean(p_n)) if p_n else 0.0,
            "control_mean_per_pair":  float(np.mean(c_n)) if c_n else 0.0,
            "proverbs_mean_density": float(np.mean(p_d)) if p_d else 0.0,
            "control_mean_density":  float(np.mean(c_d)) if c_d else 0.0,
            "proverbs_mean_pair_len": float(np.mean(p_l)) if p_l else 0.0,
            "control_mean_pair_len":  float(np.mean(c_l)) if c_l else 0.0,
            "mw_prov_gt_ctrl_raw_p":  float(p_raw),
            "mw_prov_gt_ctrl_density_p": float(p_den),
        }
        r = out[lang]
        print(f"  Proverbs per pair: {r['proverbs_mean_per_pair']:.2f}    "
              f"Control: {r['control_mean_per_pair']:.2f}")
        print(f"  Proverbs density:  {r['proverbs_mean_density']:.2f}    "
              f"Control: {r['control_mean_density']:.2f}")
        print(f"  Pair length — Prov: {r['proverbs_mean_pair_len']:.1f}w  "
              f"Ctrl: {r['control_mean_pair_len']:.1f}w")
        print(f"  MW Prov>Ctrl raw: p={p_raw:.4f}")
        print(f"  MW Prov>Ctrl density: p={p_den:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {OUT}")

    print()
    print("=" * 92)
    print(f"{'Lang':8s}  {'Prov/pair':>9s}  {'Ctrl/pair':>9s}  {'p (raw)':>9s}  "
          f"{'Prov dens':>9s}  {'Ctrl dens':>9s}  {'p (dens)':>9s}")
    print("-" * 92)
    for lang in LANGS:
        r = out.get(lang) or {}
        if r.get("skipped"): print(f"{lang:8s}  (skipped)"); continue
        print(f"{lang:8s}  {r['proverbs_mean_per_pair']:>9.2f}  "
              f"{r['control_mean_per_pair']:>9.2f}  "
              f"{r['mw_prov_gt_ctrl_raw_p']:>9.4f}  "
              f"{r['proverbs_mean_density']:>9.2f}  "
              f"{r['control_mean_density']:>9.2f}  "
              f"{r['mw_prov_gt_ctrl_density_p']:>9.4f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
