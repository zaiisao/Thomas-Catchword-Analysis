#!/usr/bin/env python3
"""
Aggregate catchword density test: Q pericopes vs non-Q controls, per language.

This is the test we should have run before any permutation test (lesson
from Thomas Phase 2B). For each target language and the Greek source:

  - Pair Q pericopes adjacently → 55 Q pairs
  - Pair control passages adjacently → 9 control pairs
  - Compute per-pair catchword count
  - Length-normalise: catchwords / (avg_word_count_pair_A × avg_word_count_pair_B)
  - Mann-Whitney one-sided: Q > controls?

If Q's per-pair density is not greater than controls in *any* language, the
aggregate count is not Q-specific — same conclusion as Thomas Phase 2B.

Outputs:
  data/q_source/aggregate_density.json
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
from scripts.q_permutation_test import (  # noqa: E402
    load_q_translations, make_tokens, compute_blocked,
    PHON_THRESHOLD, FILTER_PCT,
)

GREEK_Q_FILE        = REPO_ROOT / "data" / "q_source" / "q_pericopes_greek.json"
GREEK_CTRL_FILE     = REPO_ROOT / "data" / "q_source" / "q_controls_greek.json"
TRANS_DIR           = REPO_ROOT / "data" / "q_source" / "translations"
CTRL_TRANS_DIR      = REPO_ROOT / "data" / "q_source" / "control_translations"

OUT = REPO_ROOT / "data" / "q_source" / "aggregate_density.json"

LANGS = ["greek", "aramaic", "syriac", "hebrew", "arabic"]


def load_translations(lang: str, kind: str, variant_idx: int = 0
                       ) -> dict[int, list[dict]]:
    """kind ∈ {'q', 'control'}. variant_idx=0 = canonical."""
    if lang == "greek":
        src = GREEK_Q_FILE if kind == "q" else GREEK_CTRL_FILE
        if not src.exists():
            return {}
        gdata = json.loads(src.read_text(encoding="utf-8"))
        return {r["pericope_id"]: make_tokens(r["greek_text"], "greek")
                for r in gdata if r.get("greek_text")}
    # Non-Greek: load from translation dir
    base = TRANS_DIR if kind == "q" else CTRL_TRANS_DIR
    lang_dir = base / lang
    if not lang_dir.exists():
        return {}
    out: dict[int, list[dict]] = {}
    for path in sorted(lang_dir.glob("pericope_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        pid = d.get("pericope_id")
        variants = d.get("variants", [])
        if pid is None or variant_idx >= len(variants):
            continue
        v = variants[variant_idx]
        if v.get("success"):
            out[pid] = make_tokens(v.get("text", ""), lang)
    return out


def per_pair_counts(translations: dict[int, list[dict]],
                     blocked: set[str], lang: str
                     ) -> list[dict]:
    """For each adjacent pair (in pericope_id ascending order), count
    catchwords and length-normalised counts."""
    det = CatchwordDetector(lang, phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    ids = sorted(translations.keys())
    rows = []
    for i in range(len(ids) - 1):
        a, b = ids[i], ids[i + 1]
        ta = [t for t in translations[a] if t["lemma"] not in blocked]
        tb = [t for t in translations[b] if t["lemma"] not in blocked]
        if not ta or not tb:
            rows.append({"a": a, "b": b, "len_a": len(ta), "len_b": len(tb),
                         "n_catchwords": 0, "density": 0.0})
            continue
        cws = det.detect(ta, tb)
        n = len(cws)
        # length-normalised: catchwords per 100×100 word pair
        denom = max(len(ta) * len(tb), 1)
        density = n / denom * 10000  # per (100 × 100 word) pair
        rows.append({"a": a, "b": b, "len_a": len(ta), "len_b": len(tb),
                     "n_catchwords": n, "density": density})
    return rows


def main() -> None:
    out: dict[str, dict] = {}

    for lang in LANGS:
        print(f"\n=== {lang.upper()} ===")
        q_trans = load_translations(lang, "q")
        c_trans = load_translations(lang, "control")
        print(f"  loaded Q: {len(q_trans)}, controls: {len(c_trans)}")
        if not q_trans:
            out[lang] = {"skipped": True}
            print("  SKIP — no Q translations")
            continue
        # Block-list is computed over Q only (controls are a comparison pool)
        blocked = compute_blocked(q_trans, FILTER_PCT)
        print(f"  blocked {len(blocked)} lemmas")

        q_pairs = per_pair_counts(q_trans, blocked, lang)
        c_pairs = per_pair_counts(c_trans, blocked, lang) if c_trans else []

        q_n     = [r["n_catchwords"] for r in q_pairs]
        c_n     = [r["n_catchwords"] for r in c_pairs]
        q_dens  = [r["density"]      for r in q_pairs]
        c_dens  = [r["density"]      for r in c_pairs]
        q_len   = [(r["len_a"] + r["len_b"]) / 2 for r in q_pairs]
        c_len   = [(r["len_a"] + r["len_b"]) / 2 for r in c_pairs]

        if q_n and c_n:
            u_n, p_n = stats.mannwhitneyu(q_n,    c_n,    alternative="greater")
            u_d, p_d = stats.mannwhitneyu(q_dens, c_dens, alternative="greater")
        else:
            u_n = u_d = p_n = p_d = float("nan")

        rec = {
            "skipped": False,
            "n_q_pairs": len(q_pairs),
            "n_control_pairs": len(c_pairs),
            "q_total_catchwords":      int(sum(q_n)),
            "control_total_catchwords": int(sum(c_n)),
            "q_mean_per_pair":     float(np.mean(q_n))    if q_n   else 0.0,
            "control_mean_per_pair": float(np.mean(c_n))  if c_n   else 0.0,
            "q_mean_density":      float(np.mean(q_dens)) if q_dens else 0.0,
            "control_mean_density": float(np.mean(c_dens)) if c_dens else 0.0,
            "q_mean_pair_len":     float(np.mean(q_len)),
            "control_mean_pair_len": float(np.mean(c_len)) if c_len else 0.0,
            "mw_q_gt_ctrl_raw_p":  float(p_n),
            "mw_q_gt_ctrl_density_p": float(p_d),
            "per_q_pair":      q_pairs,
            "per_control_pair": c_pairs,
        }
        out[lang] = rec
        print(f"  Q total:        {rec['q_total_catchwords']:>5d}    "
              f"per pair: {rec['q_mean_per_pair']:>6.2f}")
        print(f"  Control total:  {rec['control_total_catchwords']:>5d}    "
              f"per pair: {rec['control_mean_per_pair']:>6.2f}")
        print(f"  Q  density (cw / 100×100 words): "
              f"{rec['q_mean_density']:.2f}")
        print(f"  Ctrl density (cw / 100×100 words): "
              f"{rec['control_mean_density']:.2f}")
        print(f"  Avg pair length — Q: {rec['q_mean_pair_len']:.1f}w  "
              f"Control: {rec['control_mean_pair_len']:.1f}w")
        print(f"  Mann-Whitney  Q > Ctrl (raw counts):   p = {p_n:.4f}")
        print(f"  Mann-Whitney  Q > Ctrl (length-norm):  p = {p_d:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")

    # Headline
    print()
    print("=" * 88)
    print(f"{'Lang':8s}  {'Q/pair':>7s}  {'C/pair':>7s}  {'p (raw)':>9s}  "
          f"{'Q dens':>8s}  {'C dens':>8s}  {'p (dens)':>9s}")
    print("-" * 88)
    for lang in LANGS:
        r = out.get(lang) or {}
        if r.get("skipped") or not r:
            print(f"{lang:8s}  (skipped)"); continue
        print(f"{lang:8s}  {r['q_mean_per_pair']:>7.2f}  "
              f"{r['control_mean_per_pair']:>7.2f}  "
              f"{r['mw_q_gt_ctrl_raw_p']:>9.4f}  "
              f"{r['q_mean_density']:>8.2f}  "
              f"{r['control_mean_density']:>8.2f}  "
              f"{r['mw_q_gt_ctrl_density_p']:>9.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
