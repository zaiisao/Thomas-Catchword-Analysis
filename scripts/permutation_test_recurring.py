#!/usr/bin/env python3
"""
Permutation test on recurring catchword patterns in the Gospel of Thomas.

The aggregate catchword count (Phase 2B: 974) is uninformative about Thomas
specifically because total catchword inflation cancels under shuffling. But
*recurring* patterns — specific (lemma_a, lemma_b) pairs that appear at
multiple logion boundaries — depend on the actual ordering. If Thomas's true
ordering produces more recurring patterns than random shuffles, that is
direct evidence of compositional design that Phase 2B's headline number
cannot show.

Methodology:
  1. Use Phase 2B Gemini variant 0 translations (canonical) as the primary
     test, with Phase 2A beam λ=0.3 lemmatized translations as cross-check.
  2. Precompute a 115×115 catchword-pair matrix: for each (i,j), store the
     set of (lemma_a, lemma_b, link_type) keys the detector flags.
     Permutation testing is then dictionary lookup + counting.
  3. For each of N=10,000 random permutations, walk the 114 adjacencies,
     count distinct pairs appearing at >= min_freq boundaries, and the max
     frequency of any single pair.
  4. Compare true-order statistics to the null distribution.
  5. Variant robustness: repeat with each of the 10 LLM variants (1k perms).

Calibration: same detector at filter_pct=80, threshold=0.65 as the rest of
the project.

Output:
  data/processed/permutation_test_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LLM_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
BEAM_FILE = REPO_ROOT / "data" / "processed" / "phase2a_translations" / "lambda_0.3.jsonl"
SEDRA = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
OUT = REPO_ROOT / "data" / "processed" / "permutation_test_results.json"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0

N_LOGIA = 115
N_PERMUTATIONS = 10000
N_PERMUTATIONS_VARIANT = 1000
SEED = 42

SYRIAC_RE = re.compile(r"[܀-ݏ]")
PUNCT_RE = re.compile(r"[܀-܏]")


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def tokenize_syriac(text: str) -> list[str]:
    """Strip vocalization + punctuation, split on whitespace."""
    cleaned = re.sub(r"[^܀-ݏ\s]", "", text)
    cleaned = strip_voc(cleaned)
    cleaned = PUNCT_RE.sub(" ", cleaned)
    return [w for w in cleaned.split() if w]


def load_sedra_lookup() -> dict[str, str]:
    out = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            unp, lem = parts[1].strip(), parts[3].strip()
            if unp and lem and unp not in out:
                out[unp] = lem
    return out


def make_tokens(syriac_text: str, sedra: dict) -> list[dict]:
    return [{"form": t, "lemma": sedra.get(t, t), "parse": "MS-EMP"}
            for t in tokenize_syriac(syriac_text)]


def make_tokens_from_lemmas(lemmas: list) -> list[dict]:
    return [{"form": l, "lemma": l, "parse": "MS-EMP"} for l in lemmas if l]


def load_gemini_variants(variant_idx: int = 0) -> dict[int, list[dict]]:
    """Load Gemini variant_idx translations for each Thomas logion."""
    sedra = load_sedra_lookup()
    out = {}
    for i in range(N_LOGIA):
        path = LLM_DIR / f"logion_{i:03d}.json"
        if not path.exists():
            out[i] = []
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        variants = d.get("variants", [])
        if variant_idx < len(variants) and variants[variant_idx].get("success"):
            out[i] = make_tokens(variants[variant_idx]["syriac_text"], sedra)
        else:
            out[i] = []
    return out


def load_beam_translations() -> dict[int, list[dict]]:
    """Load Phase 2A beam λ=0.3 — already-lemmatized syriac sequences."""
    out = {}
    with BEAM_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["logion"]] = make_tokens_from_lemmas(r.get("syriac_lemmas", []))
    return out


def compute_blocked(translations: dict, filter_pct: float) -> set[str]:
    """Block lemmas appearing in > filter_pct% of logia."""
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_catchword_matrix(translations: dict[int, list[dict]],
                                 blocked: set[str],
                                 ) -> dict[tuple[int, int], frozenset]:
    """For each (i, j) with i != j, store the frozenset of (lemma_a, lemma_b,
    link_type) keys the detector flags. Returns dict keyed by (i, j)."""
    det = CatchwordDetector("syriac",
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    matrix = {}

    def filter_blocked(toks):
        return [t for t in toks if t["lemma"] not in blocked]

    filtered = {i: filter_blocked(translations.get(i, [])) for i in range(N_LOGIA)}

    pairs_total = N_LOGIA * (N_LOGIA - 1)
    print(f"  Precomputing {pairs_total} adjacency cells…")
    t0 = time.time()
    for i in range(N_LOGIA):
        for j in range(N_LOGIA):
            if i == j:
                continue
            ta = filtered[i]; tb = filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = frozenset()
                continue
            cws = det.detect(ta, tb)
            keys = []
            for cw in cws:
                la = cw.token_a["lemma"]
                lb = cw.token_b["lemma"]
                # Canonical key: sorted lemma pair + link type
                if la <= lb:
                    keys.append((la, lb, cw.link_type))
                else:
                    keys.append((lb, la, cw.link_type))
            matrix[(i, j)] = frozenset(keys)
        if (i + 1) % 25 == 0:
            print(f"    row {i+1}/{N_LOGIA} ({time.time()-t0:.0f}s)")
    print(f"  Matrix built in {time.time()-t0:.0f}s")
    return matrix


def stats_for_order(order: list[int],
                     matrix: dict[tuple[int, int], frozenset],
                     min_freqs: list[int]) -> dict:
    """Walk adjacencies in `order`, accumulate pair → boundary list, return
    counts at each min_freq plus the max frequency."""
    pair_locations = defaultdict(list)
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), frozenset())
        for k in cell:
            pair_locations[k].append(pos)
    out = {f"recurring_{f}plus": sum(1 for locs in pair_locations.values()
                                       if len(locs) >= f)
            for f in min_freqs}
    out["max_freq"] = max((len(locs) for locs in pair_locations.values()),
                           default=0)
    out["pair_locations"] = pair_locations
    return out


def run_permutation_test(matrix, n_perms, seed, min_freqs=(2, 3)):
    rng = random.Random(seed)
    base_order = list(range(N_LOGIA))
    null = {f"recurring_{f}plus": [] for f in min_freqs}
    null["max_freq"] = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = base_order.copy()
        rng.shuffle(shuffled)
        s = stats_for_order(shuffled, matrix, list(min_freqs))
        for k in null:
            null[k].append(s[k])
        if (p + 1) % 1000 == 0:
            print(f"    perm {p+1}/{n_perms} ({time.time()-t0:.0f}s)")
    return {k: np.array(v) for k, v in null.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0,
                     help="LLM variant index (0..9) for the main test")
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--variants-test", action="store_true",
                     help="Also run robustness check across all 10 variants")
    ap.add_argument("--cross-validate-beam", action="store_true",
                     help="Also test on Phase 2A beam λ=0.3 translations")
    args = ap.parse_args()

    print(f"=== Permutation test on recurring catchword patterns ===")
    print(f"  N_PERMUTATIONS:      {args.n_perms}")
    print(f"  Calibration:         filter_pct={FILTER_PCT}, "
          f"threshold={PHON_THRESHOLD}")
    print()

    print(f"[1/4] Loading Gemini variant {args.variant}…")
    translations = load_gemini_variants(args.variant)
    n_loaded = sum(1 for t in translations.values() if t)
    print(f"  loaded {n_loaded}/{N_LOGIA} logia")

    blocked = compute_blocked(translations, FILTER_PCT)
    print(f"  blocked {len(blocked)} lemmas (top frequent): {sorted(blocked)[:8]}")

    print()
    print(f"[2/4] Precomputing catchword matrix (115×115 cells)…")
    matrix = precompute_catchword_matrix(translations, blocked)
    nonempty = sum(1 for v in matrix.values() if v)
    avg_pairs = np.mean([len(v) for v in matrix.values() if v])
    print(f"  {nonempty}/{len(matrix)} non-empty cells, "
          f"avg {avg_pairs:.1f} pairs/cell")

    print()
    print(f"[3/4] True order statistics…")
    true_order = list(range(N_LOGIA))
    s_true = stats_for_order(true_order, matrix, [2, 3, 4])
    print(f"  Recurring pairs (≥2 boundaries): {s_true['recurring_2plus']}")
    print(f"  Recurring pairs (≥3 boundaries): {s_true['recurring_3plus']}")
    print(f"  Recurring pairs (≥4 boundaries): {s_true['recurring_4plus']}")
    print(f"  Max frequency of any pair:        {s_true['max_freq']}")

    sorted_pairs = sorted(s_true["pair_locations"].items(),
                           key=lambda x: -len(x[1]))
    print()
    print(f"  Top-15 most recurring pairs in TRUE order:")
    for (la, lb, lt), locs in sorted_pairs[:15]:
        print(f"    ({la}, {lb}) [{lt}] — {len(locs)}× at boundaries {locs}")

    print()
    print(f"[4/4] Permutation test ({args.n_perms} shuffles)…")
    null = run_permutation_test(matrix, args.n_perms, args.seed,
                                  min_freqs=(2, 3, 4))

    p_2 = float((null["recurring_2plus"] >= s_true["recurring_2plus"]).mean())
    p_3 = float((null["recurring_3plus"] >= s_true["recurring_3plus"]).mean())
    p_4 = float((null["recurring_4plus"] >= s_true["recurring_4plus"]).mean())
    p_max = float((null["max_freq"] >= s_true["max_freq"]).mean())

    print()
    print(f"  Recurring pairs (≥2 boundaries):  true={s_true['recurring_2plus']:>4d}, "
          f"null mean={null['recurring_2plus'].mean():.1f} "
          f"(±{null['recurring_2plus'].std():.1f}), p={p_2:.4f}")
    print(f"  Recurring pairs (≥3 boundaries):  true={s_true['recurring_3plus']:>4d}, "
          f"null mean={null['recurring_3plus'].mean():.1f} "
          f"(±{null['recurring_3plus'].std():.1f}), p={p_3:.4f}")
    print(f"  Recurring pairs (≥4 boundaries):  true={s_true['recurring_4plus']:>4d}, "
          f"null mean={null['recurring_4plus'].mean():.1f} "
          f"(±{null['recurring_4plus'].std():.1f}), p={p_4:.4f}")
    print(f"  Max frequency of any pair:        true={s_true['max_freq']:>4d}, "
          f"null mean={null['max_freq'].mean():.1f} "
          f"(±{null['max_freq'].std():.1f}), p={p_max:.4f}")

    # ---- Perrin's specific cited recurring patterns ----
    PERRIN_PATTERNS = {
        "nura_nuhra (fire/light)":
            [(10, 11), (16, 17), (82, 83)],
        "etar_atar (wealth/place)":
            [(29, 30), (85, 86)],
        "nas_nesse (someone/women)":
            [(14, 15), (46, 47), (113, 114)],
    }
    print()
    print("=== Perrin's specifically cited recurring patterns ===")
    perrin_recovery = {}
    for name, claimed in PERRIN_PATTERNS.items():
        found = []
        cw_at = {}
        for (i, j) in claimed:
            cell = matrix.get((i, j), frozenset())
            if cell:
                found.append((i, j))
                cw_at[(i, j)] = list(cell)[:5]
        perrin_recovery[name] = {
            "claimed": claimed,
            "found": found,
            "recovery_rate": len(found) / len(claimed),
            "sample_pairs_at_each_boundary": {f"{i}-{j}": v
                                                for (i, j), v in cw_at.items()},
        }
        print(f"  {name}:")
        print(f"    claimed at: {claimed}")
        print(f"    detected at: {found}  ({len(found)}/{len(claimed)})")

    # ---- Cross-validate with beam search ----
    beam_block = None
    if args.cross_validate_beam:
        print()
        print("=== Cross-validation: Phase 2A beam λ=0.3 ===")
        beam_trans = load_beam_translations()
        beam_blocked = compute_blocked(beam_trans, FILTER_PCT)
        print(f"  blocked: {len(beam_blocked)}")
        beam_matrix = precompute_catchword_matrix(beam_trans, beam_blocked)
        beam_true = stats_for_order(true_order, beam_matrix, [2, 3])
        beam_null = run_permutation_test(beam_matrix, args.n_perms, args.seed,
                                           min_freqs=(2, 3))
        beam_p2 = float((beam_null["recurring_2plus"]
                          >= beam_true["recurring_2plus"]).mean())
        beam_p3 = float((beam_null["recurring_3plus"]
                          >= beam_true["recurring_3plus"]).mean())
        beam_block = {
            "true_recurring_2plus": beam_true["recurring_2plus"],
            "true_recurring_3plus": beam_true["recurring_3plus"],
            "null_mean_2plus": float(beam_null["recurring_2plus"].mean()),
            "null_std_2plus":  float(beam_null["recurring_2plus"].std()),
            "p_2plus": beam_p2,
            "p_3plus": beam_p3,
        }
        print(f"  recurring (≥2): true={beam_true['recurring_2plus']}, "
              f"null mean={beam_null['recurring_2plus'].mean():.1f}, p={beam_p2:.4f}")
        print(f"  recurring (≥3): true={beam_true['recurring_3plus']}, "
              f"null mean={beam_null['recurring_3plus'].mean():.1f}, p={beam_p3:.4f}")

    # ---- Variant robustness ----
    var_results = None
    if args.variants_test:
        print()
        print(f"=== Variant robustness (10 variants × {N_PERMUTATIONS_VARIANT} perms) ===")
        var_results = []
        for v in range(10):
            print(f"\n  Variant {v}:")
            t = load_gemini_variants(v)
            blk = compute_blocked(t, FILTER_PCT)
            mat = precompute_catchword_matrix(t, blk)
            st = stats_for_order(true_order, mat, [2, 3])
            nl = run_permutation_test(mat, N_PERMUTATIONS_VARIANT, args.seed,
                                        min_freqs=(2, 3))
            pv2 = float((nl["recurring_2plus"] >= st["recurring_2plus"]).mean())
            pv3 = float((nl["recurring_3plus"] >= st["recurring_3plus"]).mean())
            var_results.append({"variant": v,
                                 "true_recurring_2plus": st["recurring_2plus"],
                                 "true_recurring_3plus": st["recurring_3plus"],
                                 "p_2plus": pv2, "p_3plus": pv3,
                                 "null_mean_2plus": float(nl["recurring_2plus"].mean()),
                                 "null_std_2plus":  float(nl["recurring_2plus"].std())})
            print(f"    rec≥2: true={st['recurring_2plus']}, "
                  f"null={nl['recurring_2plus'].mean():.1f}, p={pv2:.4f}")
            print(f"    rec≥3: true={st['recurring_3plus']}, "
                  f"null={nl['recurring_3plus'].mean():.1f}, p={pv3:.4f}")

    # ---- Save ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "config": {"phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT,
                    "n_logia": N_LOGIA, "n_permutations": args.n_perms,
                    "seed": args.seed, "variant_idx": args.variant},
        "true_order": {
            "recurring_2plus": s_true["recurring_2plus"],
            "recurring_3plus": s_true["recurring_3plus"],
            "recurring_4plus": s_true["recurring_4plus"],
            "max_freq": s_true["max_freq"],
            "top_pairs": [
                {"lemma_a": k[0], "lemma_b": k[1], "link_type": k[2],
                  "frequency": len(v), "boundaries": v}
                for k, v in sorted_pairs[:30]
            ],
        },
        "null_distribution": {
            "recurring_2plus_mean": float(null["recurring_2plus"].mean()),
            "recurring_2plus_std":  float(null["recurring_2plus"].std()),
            "recurring_3plus_mean": float(null["recurring_3plus"].mean()),
            "recurring_3plus_std":  float(null["recurring_3plus"].std()),
            "recurring_4plus_mean": float(null["recurring_4plus"].mean()),
            "recurring_4plus_std":  float(null["recurring_4plus"].std()),
            "max_freq_mean": float(null["max_freq"].mean()),
            "max_freq_std":  float(null["max_freq"].std()),
            # Save raw arrays for plotting
            "_raw_recurring_2plus": null["recurring_2plus"].tolist(),
            "_raw_recurring_3plus": null["recurring_3plus"].tolist(),
            "_raw_max_freq":         null["max_freq"].tolist(),
        },
        "p_values": {
            "p_recurring_2plus": p_2,
            "p_recurring_3plus": p_3,
            "p_recurring_4plus": p_4,
            "p_max_freq":         p_max,
        },
        "perrin_specific": perrin_recovery,
        "beam_cross_validation": beam_block,
        "variant_robustness": var_results,
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
