#!/usr/bin/env python3
"""
Cross-lingual permutation test — Syriac with SURFACE forms (no SEDRA).

Re-runs ONLY the Syriac arm of the cross-lingual test using surface-form
tokenisation (apples-to-apples with the Hebrew / Greek / Arabic arms that
were always on surface forms). Outputs the same statistic as
`crossling_permutation_test.py` (recurring pairs at ≥2 boundaries) so the
result is directly comparable.

Hebrew / Greek / Arabic results were already correct (surface forms); we
do not re-run them here. The comparison below is:
  - new Syriac (surface)  vs  old Syriac (SEDRA-collapsed)
  - new Syriac (surface)  vs  old Hebrew, Greek, Arabic (unchanged)

Output: data/processed/crossling_syriac_surface.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402
from scripts.crossling_permutation_test import (  # noqa: E402
    tokenize, LLM_DIR_SYR, N_LOGIA, PHON_THRESHOLD, FILTER_PCT,
)

OUT_DIR = REPO_ROOT / "data" / "processed" / "crossling_syriac_surface"
SEED = 42
N_MAIN = 10000
N_VARIANT = 1000


def load_syriac_surface(variant_idx: int = 0) -> dict[int, list[dict]]:
    """Load Thomas Syriac at variant_idx, using SURFACE forms (no SEDRA)."""
    out: dict[int, list[dict]] = {}
    for i in range(N_LOGIA):
        path = LLM_DIR_SYR / f"logion_{i:03d}.json"
        if not path.exists():
            out[i] = []
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        variants = d.get("variants", [])
        if variant_idx < len(variants) and variants[variant_idx].get("success"):
            txt = variants[variant_idx]["syriac_text"]
            out[i] = [{"form": t, "lemma": t, "parse": "MS-EMP"}
                       for t in tokenize(txt, "syriac")]
        else:
            out[i] = []
    return out


def compute_blocked(translations, filter_pct):
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix(translations, blocked):
    det = CatchwordDetector("syriac",
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in range(N_LOGIA)}
    matrix = {}
    t0 = time.time()
    for i in range(N_LOGIA):
        for j in range(N_LOGIA):
            if i == j: continue
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
    print(f"  matrix built in {time.time()-t0:.0f}s", flush=True)
    return matrix


def stats_for_order(order, matrix, min_freqs):
    pair_locations: defaultdict[tuple, list[int]] = defaultdict(list)
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), frozenset())
        for k in cell:
            pair_locations[k].append(pos)
    return {
        **{f"recurring_{f}plus":
             sum(1 for locs in pair_locations.values() if len(locs) >= f)
           for f in min_freqs},
        "max_freq": max((len(v) for v in pair_locations.values()), default=0),
    }


def run_permutation(matrix, ids, n_perms, seed, min_freqs):
    import random as _r
    rng = _r.Random(seed)
    null = {f"recurring_{f}plus": [] for f in min_freqs}
    null["max_freq"] = []
    base = list(ids)
    for p in range(n_perms):
        sh = base.copy(); rng.shuffle(sh)
        s = stats_for_order(sh, matrix, list(min_freqs))
        for k in null:
            null[k].append(s[k])
    return {k: np.array(v) for k, v in null.items()}


def run_one_variant(variant_idx: int, n_perms: int = N_MAIN) -> dict:
    print(f"\n=== variant {variant_idx} ===")
    toks = load_syriac_surface(variant_idx)
    n_loaded = sum(1 for v in toks.values() if v)
    print(f"  loaded {n_loaded}/{N_LOGIA} logia")
    if n_loaded < N_LOGIA * 0.5:
        return {"variant": variant_idx, "skipped": True,
                "n_loaded": n_loaded}
    blocked = compute_blocked(toks, FILTER_PCT)
    print(f"  blocked {len(blocked)} surface forms")
    matrix = precompute_matrix(toks, blocked)
    ids = list(range(N_LOGIA))
    s_true = stats_for_order(ids, matrix, [2, 3])
    print(f"  true: rec≥2={s_true['recurring_2plus']}, "
          f"rec≥3={s_true['recurring_3plus']}, "
          f"max_freq={s_true['max_freq']}")
    null = run_permutation(matrix, ids, n_perms, SEED, min_freqs=(2, 3))
    p2 = float((null["recurring_2plus"] >= s_true["recurring_2plus"]).mean())
    p3 = float((null["recurring_3plus"] >= s_true["recurring_3plus"]).mean())
    e2 = ((s_true["recurring_2plus"] - null["recurring_2plus"].mean())
          / max(null["recurring_2plus"].std(), 1e-9))
    e3 = ((s_true["recurring_3plus"] - null["recurring_3plus"].mean())
          / max(null["recurring_3plus"].std(), 1e-9))
    print(f"  rec≥2: true={s_true['recurring_2plus']}, "
          f"null={null['recurring_2plus'].mean():.1f}±"
          f"{null['recurring_2plus'].std():.1f}, "
          f"z={e2:.2f}, p={p2:.4f}", flush=True)
    return {
        "variant": variant_idx,
        "n_loaded": n_loaded,
        "n_blocked": len(blocked),
        "true_recurring_2plus": int(s_true["recurring_2plus"]),
        "true_recurring_3plus": int(s_true["recurring_3plus"]),
        "true_max_freq": int(s_true["max_freq"]),
        "null_mean_2plus": float(null["recurring_2plus"].mean()),
        "null_std_2plus":  float(null["recurring_2plus"].std()),
        "null_mean_3plus": float(null["recurring_3plus"].mean()),
        "null_std_3plus":  float(null["recurring_3plus"].std()),
        "p_2plus": p2,
        "p_3plus": p3,
        "z_2plus": float(e2),
        "z_3plus": float(e3),
        "n_perms": n_perms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="0",
                     help="comma-separated variant indices to run")
    ap.add_argument("--n-perms", type=int, default=N_MAIN)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    var_indices = [int(v) for v in args.variants.split(",") if v.strip()]
    print(f"Re-running Thomas Syriac with SURFACE forms")
    print(f"  variants: {var_indices}")
    print(f"  n_perms:  {args.n_perms}")

    for v in var_indices:
        result = run_one_variant(v, args.n_perms)
        result["config"] = {
            "tokenization": "surface_forms_no_sedra",
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "seed": SEED,
        }
        out_p = OUT_DIR / f"variant_{v}.json"
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                          default=str), encoding="utf-8")
        print(f"  wrote {out_p}")


if __name__ == "__main__":
    main()
