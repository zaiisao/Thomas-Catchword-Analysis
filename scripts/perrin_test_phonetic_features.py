#!/usr/bin/env python3
"""
Task 1 — Phonetic-feature permutation test on surface-form Syriac, v0.

Re-runs the 10,000-shuffle permutation test on surface-tokenised
Thomas (Syriac, variant 0) with the binary consonantal-skeleton
Levenshtein substitution cost replaced by the graded phonetic-feature
distance from `scripts/phonetic_features.py`.

This is the apples-to-apples comparator to
`crossling_syriac_surface/variant_0.json` (z_2plus = 3.31, p = 0.0007),
differing only in the phonological substitution-cost function.

Methodology:
  - Tokenisation: surface forms (no SEDRA collapse), variant 0.
  - Threshold:    phon_threshold = 0.65 (same as baseline).
  - filter_pct:   80%.
  - Detector hook: monkey-patch
      phase1_montecarlo.catchword_detector.weighted_levenshtein
    -> scripts.phonetic_features.feature_levenshtein
    so the existing detector + downstream code is untouched.

Outputs:
  data/processed/phonological_features/surface_v0_features.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Monkey-patch BEFORE the detector is invoked by any helper.
import phase1_montecarlo.catchword_detector as _cwd_mod  # noqa: E402
from scripts.phonetic_features import feature_levenshtein  # noqa: E402
_cwd_mod.weighted_levenshtein = feature_levenshtein

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402
from scripts.crossling_permutation_test import (  # noqa: E402
    N_LOGIA, FILTER_PCT, PHON_THRESHOLD,
    compute_blocked,
)
from scripts.crossling_syriac_surface import (  # noqa: E402
    load_syriac_surface,
)

OUT_DIR = REPO_ROOT / "data" / "processed" / "phonological_features"

N_PERMS = 10_000
SEED = 42


def precompute_matrix_thresh(translations, blocked, lang, threshold):
    """Like crossling_permutation_test.precompute_matrix but with a tunable
    phonological threshold (so we can run a calibration sweep)."""
    det = CatchwordDetector(lang, phonological_threshold=threshold,
                             require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in range(N_LOGIA)}
    matrix = {}
    t0 = time.time()
    print(f"  Precomputing matrix (threshold={threshold})...")
    for i in range(N_LOGIA):
        for j in range(N_LOGIA):
            if i == j:
                continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = frozenset()
                continue
            cws = det.detect(ta, tb)
            keys = []
            for cw in cws:
                la = cw.token_a["lemma"]; lb = cw.token_b["lemma"]
                if la <= lb:
                    keys.append((la, lb, cw.link_type))
                else:
                    keys.append((lb, la, cw.link_type))
            matrix[(i, j)] = frozenset(keys)
        if (i + 1) % 25 == 0:
            print(f"    row {i+1}/{N_LOGIA} ({time.time()-t0:.0f}s)",
                  flush=True)
    return matrix


def stats_for_order(order, matrix, min_freqs=(2, 3)):
    pair_locations = defaultdict(list)
    for pos in range(len(order) - 1):
        for k in matrix.get((order[pos], order[pos + 1]), frozenset()):
            pair_locations[k].append(pos)
    out = {f"recurring_{f}plus":
              sum(1 for locs in pair_locations.values() if len(locs) >= f)
           for f in min_freqs}
    out["max_freq"] = max((len(v) for v in pair_locations.values()), default=0)
    out["pair_locations"] = pair_locations
    return out


def run_permutation(matrix, n_perms, seed, min_freqs=(2, 3)):
    rng = random.Random(seed)
    base = list(range(N_LOGIA))
    null = {f"recurring_{f}plus": [] for f in min_freqs}
    null["max_freq"] = []
    t0 = time.time()
    for p in range(n_perms):
        sh = base.copy()
        rng.shuffle(sh)
        s = stats_for_order(sh, matrix, list(min_freqs))
        for k in null:
            null[k].append(s[k])
        if (p + 1) % 2000 == 0:
            print(f"      perm {p+1}/{n_perms} ({time.time()-t0:.0f}s)",
                  flush=True)
    return {k: np.array(v) for k, v in null.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=PHON_THRESHOLD,
                     help="Phonological threshold (default 0.65, matches "
                          "the baseline binary-Lev test).")
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--sweep", action="store_true",
                     help="Skip permutation, just report true-order stats "
                          "at the requested threshold (calibration mode).")
    ap.add_argument("--label", default="features",
                     help="Suffix for output filename, e.g. 'features' or "
                          "'features_t088'.")
    args = ap.parse_args()

    print("=" * 76)
    print(f"Task 1: phonetic-feature permutation test, Syriac surface v0")
    print("=" * 76)
    print(f"  detector: weighted_levenshtein -> feature_levenshtein "
          f"(monkey-patched)")
    print(f"  N_LOGIA={N_LOGIA}, phon_threshold={args.threshold}, "
          f"filter_pct={FILTER_PCT}, "
          f"N_PERMS={args.n_perms if not args.sweep else 'sweep'}, "
          f"seed={SEED}")
    print()

    trans = load_syriac_surface(variant_idx=0)
    n_loaded = sum(1 for v in trans.values() if v)
    print(f"  loaded {n_loaded}/{N_LOGIA} logia")
    blocked = compute_blocked(trans, FILTER_PCT)
    print(f"  blocked {len(blocked)} lemmas")

    matrix = precompute_matrix_thresh(trans, blocked, "syriac", args.threshold)
    # link-type distribution at true order
    n_phon = n_etym = n_sem = 0
    for k in range(N_LOGIA - 1):
        for la, lb, lt in matrix.get((k, k + 1), frozenset()):
            if lt == "phonological":
                n_phon += 1
            elif lt == "etymological":
                n_etym += 1
            elif lt == "semantic":
                n_sem += 1
    print(f"  TRUE adjacencies: phon={n_phon}, etym={n_etym}, sem={n_sem}")

    true_order = list(range(N_LOGIA))
    s_true = stats_for_order(true_order, matrix, [2, 3, 4])
    print(f"  true: rec>=2={s_true['recurring_2plus']}, "
          f"rec>=3={s_true['recurring_3plus']}, "
          f"rec>=4={s_true['recurring_4plus']}, "
          f"max_freq={s_true['max_freq']}")

    if args.sweep:
        # Calibration-only mode: dump density + skip the slow permutation.
        out_path = OUT_DIR / f"sweep_t{int(args.threshold*100):03d}.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "task": "phonetic_feature_calibration_sweep",
            "threshold": args.threshold,
            "true_adjacency_link_types": {
                "phonological": n_phon, "etymological": n_etym, "semantic": n_sem,
            },
            "true_recurring_2plus": int(s_true["recurring_2plus"]),
            "true_recurring_3plus": int(s_true["recurring_3plus"]),
            "true_recurring_4plus": int(s_true["recurring_4plus"]),
            "true_max_freq":         int(s_true["max_freq"]),
        }, indent=2), encoding="utf-8")
        print(f"\n[sweep] wrote {out_path}")
        return

    print(f"  running {args.n_perms} permutations...")
    null = run_permutation(matrix, args.n_perms, SEED, min_freqs=(2, 3, 4))

    def _zp(true_v, arr):
        m, s = float(arr.mean()), float(arr.std())
        z = (true_v - m) / s if s > 0 else 0.0
        p = float((arr >= true_v).mean())
        return m, s, z, p

    nm2, ns2, z2, p2 = _zp(s_true["recurring_2plus"], null["recurring_2plus"])
    nm3, ns3, z3, p3 = _zp(s_true["recurring_3plus"], null["recurring_3plus"])
    nm4, ns4, z4, p4 = _zp(s_true["recurring_4plus"], null["recurring_4plus"])

    print()
    print(f"  rec>=2: true={s_true['recurring_2plus']}, "
          f"null={nm2:.1f}±{ns2:.1f}, z={z2:.2f}, p={p2:.4f}")
    print(f"  rec>=3: true={s_true['recurring_3plus']}, "
          f"null={nm3:.1f}±{ns3:.1f}, z={z3:.2f}, p={p3:.4f}")
    print(f"  rec>=4: true={s_true['recurring_4plus']}, "
          f"null={nm4:.1f}±{ns4:.1f}, z={z4:.2f}, p={p4:.4f}")

    top = sorted(s_true["pair_locations"].items(), key=lambda x: -len(x[1]))
    out_path = OUT_DIR / f"surface_v0_{args.label}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "task": "phonetic_feature_permutation",
        "config": {
            "corpus": "thomas", "lang": "syriac", "variant": 0,
            "tokenisation": "surface_forms_no_sedra",
            "phon_threshold": args.threshold, "filter_pct": FILTER_PCT,
            "n_perms": args.n_perms, "seed": SEED,
            "detector": "feature_levenshtein (monkey-patched onto "
                         "phase1_montecarlo.catchword_detector)",
        },
        "n_loaded": n_loaded, "n_blocked": len(blocked),
        "true_adjacency_link_types": {
            "phonological": n_phon, "etymological": n_etym, "semantic": n_sem,
        },
        "true_recurring_2plus": int(s_true["recurring_2plus"]),
        "true_recurring_3plus": int(s_true["recurring_3plus"]),
        "true_recurring_4plus": int(s_true["recurring_4plus"]),
        "true_max_freq":         int(s_true["max_freq"]),
        "null_mean_2plus": nm2, "null_std_2plus": ns2,
        "null_mean_3plus": nm3, "null_std_3plus": ns3,
        "null_mean_4plus": nm4, "null_std_4plus": ns4,
        "z_2plus": z2, "p_2plus": p2,
        "z_3plus": z3, "p_3plus": p3,
        "z_4plus": z4, "p_4plus": p4,
        "top_pairs": [
            {"lemma_a": k[0], "lemma_b": k[1], "link_type": k[2],
              "frequency": len(v), "boundaries": v}
            for k, v in top[:15]
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
