#!/usr/bin/env python3
"""
Task 2 — Boundary Overlap Analysis (Jaccard Index).

Question: do Greek and surface-Syriac achieve their high z-scores by
clustering catchwords at the SAME logion boundaries (indicating a
thematic / topical signal that survives translation), or at DIFFERENT
boundaries (indicating language-specific paronomastic effects)?

Method:
  1. Load variant-0 translations for both Greek (Gemini cross-ling) and
     Syriac (Phase 2B Gemini, SURFACE-form tokenisation — apples-to-apples
     with the other languages; matches the 2026-05-11 methodology fix).
  2. Build the 115x115 adjacency-cell catchword matrix per language with
     the existing `CatchwordDetector` (phon_threshold=0.65, filter_pct=80).
  3. Walk the true Thomas order 0..114 -> 114 boundaries (indexed 0..113).
     For each boundary, take its catchword pairs (canonical key:
     sorted lemma pair + link_type).
  4. A pair is "recurring" iff it appears at >= 2 boundaries overall.
  5. For each language, build the set
        S_lang = { boundary k : count of recurring pairs at k >= 2 }.
  6. Compute Jaccard(S_g, S_s) = |S_g & S_s| / |S_g | S_s|.

Outputs:
  data/processed/boundary_overlap.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.crossling_permutation_test import (  # noqa: E402
    N_LOGIA, FILTER_PCT, PHON_THRESHOLD,
    load_translations, compute_blocked, precompute_matrix,
)
from scripts.crossling_syriac_surface import (  # noqa: E402
    load_syriac_surface,
)

OUT_PATH = REPO_ROOT / "data" / "processed" / "boundary_overlap.json"


def boundary_sets(matrix, min_pairs_per_boundary: int = 2,
                   min_recurrence: int = 2):
    """Return:
        boundary_set: set of boundary indices k (0..N_LOGIA-2) where the
            number of recurring catchword pairs at boundary k is
            >= min_pairs_per_boundary.
        per_boundary_recurring_counts: list[int] indexed by boundary k.
        recurring_pairs: dict pair_key -> list[boundary_idx].
        boundary_to_pairs: dict k -> list[pair_key] (only recurring ones).
    """
    true_order = list(range(N_LOGIA))
    pair_locations: defaultdict[tuple, list[int]] = defaultdict(list)
    boundary_pairs: list[set] = [set() for _ in range(N_LOGIA - 1)]
    for k in range(N_LOGIA - 1):
        cell = matrix.get((true_order[k], true_order[k + 1]), frozenset())
        for key in cell:
            pair_locations[key].append(k)
            boundary_pairs[k].add(key)
    recurring = {k: locs for k, locs in pair_locations.items()
                  if len(locs) >= min_recurrence}
    boundary_to_pairs: dict[int, list[tuple]] = {}
    per_boundary_counts = []
    for k in range(N_LOGIA - 1):
        rec_at_k = [p for p in boundary_pairs[k] if p in recurring]
        boundary_to_pairs[k] = rec_at_k
        per_boundary_counts.append(len(rec_at_k))
    boundary_set = {k for k, c in enumerate(per_boundary_counts)
                     if c >= min_pairs_per_boundary}
    return boundary_set, per_boundary_counts, recurring, boundary_to_pairs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    print("=" * 76)
    print("Task 2: Boundary-overlap (Jaccard) — Greek vs surface-Syriac, v0")
    print("=" * 76)
    print(f"  N_LOGIA={N_LOGIA}, phon_threshold={PHON_THRESHOLD}, "
          f"filter_pct={FILTER_PCT}")
    print()

    # ---- Greek (surface) ----
    print("--- Greek (variant 0) ---")
    g_trans = load_translations("greek", variant_idx=0)
    g_blocked = compute_blocked(g_trans, FILTER_PCT)
    print(f"  loaded {sum(1 for v in g_trans.values() if v)}/{N_LOGIA} logia, "
          f"blocked {len(g_blocked)} lemmas")
    g_matrix = precompute_matrix(g_trans, g_blocked, "greek")

    # ---- Syriac (SURFACE forms, no SEDRA collapse) ----
    print()
    print("--- Syriac (variant 0, SURFACE forms) ---")
    s_trans = load_syriac_surface(variant_idx=0)
    s_blocked = compute_blocked(s_trans, FILTER_PCT)
    print(f"  loaded {sum(1 for v in s_trans.values() if v)}/{N_LOGIA} logia, "
          f"blocked {len(s_blocked)} lemmas")
    s_matrix = precompute_matrix(s_trans, s_blocked, "syriac")

    # ---- Boundary sets ----
    print()
    print("--- Boundary sets (recurring pairs >= 2 per boundary) ---")
    g_set, g_counts, g_rec, g_b2p = boundary_sets(g_matrix)
    s_set, s_counts, s_rec, s_b2p = boundary_sets(s_matrix)

    print(f"  Greek:  {len(g_rec)} recurring pairs, "
          f"{len(g_set)}/{N_LOGIA-1} boundaries with >=2 recurring pairs")
    print(f"  Syriac: {len(s_rec)} recurring pairs, "
          f"{len(s_set)}/{N_LOGIA-1} boundaries with >=2 recurring pairs")

    # ---- Jaccard ----
    inter = sorted(g_set & s_set)
    union = sorted(g_set | s_set)
    j = jaccard(g_set, s_set)
    print()
    print(f"  | S_g | = {len(g_set)}")
    print(f"  | S_s | = {len(s_set)}")
    print(f"  intersection = {len(inter)}")
    print(f"  union        = {len(union)}")
    print(f"  Jaccard      = {j:.4f}")

    # Secondary metric: "any recurring pair at boundary" (>=1)
    g_set_any = {k for k, c in enumerate(g_counts) if c >= 1}
    s_set_any = {k for k, c in enumerate(s_counts) if c >= 1}
    j_any = jaccard(g_set_any, s_set_any)
    print()
    print(f"  Secondary (>=1 recurring pair / boundary):")
    print(f"    | S_g | = {len(g_set_any)}, | S_s | = {len(s_set_any)}, "
          f"Jaccard = {j_any:.4f}")

    # Random-overlap baseline: under a null where each language picks a
    # set of size |S_lang| uniformly from {0,...,112}, expected Jaccard is
    #   E[J] ~ p_g * p_s / (p_g + p_s - p_g*p_s)
    # with p_lang = |S_lang| / (N_LOGIA-1).
    n_b = N_LOGIA - 1
    pg, ps = len(g_set) / n_b, len(s_set) / n_b
    if pg + ps - pg * ps > 0:
        exp_j = pg * ps / (pg + ps - pg * ps)
    else:
        exp_j = 0.0
    print(f"  expected J under indep. uniform null = {exp_j:.4f}")
    print(f"  observed / expected ratio            = "
          f"{(j / exp_j) if exp_j else float('nan'):.2f}x")

    # Hypergeometric: P(|S_g & S_s| >= observed) given |S_g|, |S_s|, n_b.
    # Treat one set as the "population marked"; the other is a uniform
    # subset of size |S_s| drawn from n_b boundaries.
    K = len(g_set)   # marked
    n = len(s_set)   # drawn
    obs = len(inter)
    p_hyper = 0.0
    for x in range(obs, min(K, n) + 1):
        if x > n or (n - x) > (n_b - K):
            continue
        p_hyper += comb(K, x) * comb(n_b - K, n - x) / comb(n_b, n)
    e_inter = K * n / n_b if n_b else 0.0
    print(f"  hypergeometric: observed inter = {obs}, expected = {e_inter:.2f}, "
          f"P(>= obs) = {p_hyper:.4g}")
    is_subset = s_set.issubset(g_set)
    if is_subset:
        print(f"  CONTAINMENT: Syriac boundaries are a STRICT SUBSET of "
              f"Greek's ({len(s_set)}/{len(s_set)} contained).")

    out = {
        "task": "boundary_overlap_jaccard",
        "config": {
            "n_logia": N_LOGIA,
            "n_boundaries": n_b,
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "variant": 0,
            "syriac_tokenisation": "surface_forms_no_sedra",
            "greek_tokenisation":  "consonantal_skeleton_default",
            "min_pairs_per_boundary": 2,
            "min_recurrence": 2,
        },
        "greek": {
            "n_recurring_pairs": len(g_rec),
            "boundary_set_size": len(g_set),
            "boundary_set": sorted(g_set),
            "per_boundary_recurring_counts": g_counts,
        },
        "syriac": {
            "n_recurring_pairs": len(s_rec),
            "boundary_set_size": len(s_set),
            "boundary_set": sorted(s_set),
            "per_boundary_recurring_counts": s_counts,
        },
        "overlap": {
            "intersection": inter,
            "union": union,
            "jaccard_strict": j,
            "jaccard_any_recurring": j_any,
            "expected_jaccard_under_indep_uniform_null": exp_j,
            "hypergeometric_expected_intersection": e_inter,
            "hypergeometric_p_at_least_observed": p_hyper,
            "syriac_subset_of_greek": is_subset,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
