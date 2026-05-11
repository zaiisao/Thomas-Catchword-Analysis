#!/usr/bin/env python3
"""
Test 5: Boundary-by-boundary MAX phon score comparison across languages.

For each adjacent boundary in Thomas, compute the MAX phon-similarity score
over all (word_a in logion_left, word_b in logion_right) pairs in each
language. Compare distributions across languages.

Specifically test:
  - At what fraction of TRUE adjacent boundaries is Syriac MAX > all others?
  - Compare to NON-adjacent (random) boundaries: is the Syriac-wins fraction
    higher at TRUE than at NON-adjacent?

If Syriac has Perrin-style sound-design, Syriac should win disproportionately
at TRUE adjacencies but not at non-adjacent positions.

Output: data/perrin_direct/boundary_max_thomas.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import (  # noqa: E402
    consonantal, weighted_levenshtein,
)
from phase1_montecarlo.language_data import get_profile  # noqa: E402
from scripts.phon_only_one import load_thomas  # noqa: E402

LANGS = ["syriac", "hebrew", "greek", "arabic"]
OUT = REPO_ROOT / "data" / "perrin_direct" / "boundary_max_thomas.json"

PHON_THR = 0.6


def boundary_max_phon(toks_a, toks_b, lang_profile):
    """Return MAX phon-score over all distinct (a, b) lemma pairs."""
    best = 0.0
    seen = set()
    for ta in toks_a:
        la = ta.get("lemma", "")
        if not la: continue
        for tb in toks_b:
            lb = tb.get("lemma", "")
            if not lb: continue
            key = (la, lb)
            if key in seen: continue
            seen.add(key)
            if la == lb:
                continue  # semantic, not phon
            ca, cb = consonantal(la), consonantal(lb)
            if not ca or not cb or ca == cb:
                continue
            dist = weighted_levenshtein(ca, cb, lang_profile)
            score = 1.0 - dist / max(len(ca), len(cb))
            if score > best:
                best = score
    return best


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load all 4 langs at variant 0 (Syriac is "source"/analyzed lang for Thomas)
    print("loading translations…")
    data = {}
    for lang in LANGS:
        ids, toks = load_thomas(lang, variant_idx=0)
        usable = [i for i in ids if toks.get(i)]
        data[lang] = {"ids": ids, "toks": toks, "usable": usable,
                       "profile": get_profile(lang)}
        print(f"  {lang}: {len(usable)} logia")

    # Use intersection of available logia (some may be missing in one lang)
    common_ids = set(data[LANGS[0]]["usable"])
    for lang in LANGS[1:]:
        common_ids &= set(data[lang]["usable"])
    common_ids = sorted(common_ids)
    n = len(common_ids)
    print(f"common logia: {n}")

    # MAX-phon scores per language at every (i, j) pair
    print("computing all-pairs MAX phon-scores (this takes a few min)…")
    t0 = time.time()
    max_scores: dict[str, dict] = {lang: {} for lang in LANGS}
    for lang in LANGS:
        profile = data[lang]["profile"]
        toks_map = data[lang]["toks"]
        for k, i in enumerate(common_ids):
            for j in common_ids:
                if i == j: continue
                max_scores[lang][(i, j)] = boundary_max_phon(
                    toks_map[i], toks_map[j], profile)
            if (k + 1) % max(1, n // 5) == 0:
                print(f"  {lang} row {k+1}/{n} ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"all-pairs done {time.time()-t0:.0f}s")

    # TRUE adjacent boundaries
    true_boundaries = [(common_ids[k], common_ids[k+1]) for k in range(n-1)]
    # Statistics at true boundaries
    true_max = {lang: [max_scores[lang][b] for b in true_boundaries]
                 for lang in LANGS}

    # Mean MAX at TRUE adjacency, per language
    print()
    print("MEAN MAX phon-score at TRUE adjacent boundaries:")
    for lang in LANGS:
        vals = np.array(true_max[lang])
        print(f"  {lang:<8}: mean={vals.mean():.3f}  median={np.median(vals):.3f}"
              f"  N>={PHON_THR}: {np.sum(vals >= PHON_THR)}/{len(vals)}"
              f"  ({np.mean(vals >= PHON_THR):.1%})")

    # Per-boundary winner
    winners = []
    for k, b in enumerate(true_boundaries):
        scores = {lang: max_scores[lang][b] for lang in LANGS}
        best_lang = max(scores.keys(), key=lambda l: scores[l])
        winners.append((k, best_lang, scores[best_lang]))
    winner_count = {lang: 0 for lang in LANGS}
    for k, lang, score in winners:
        winner_count[lang] += 1
    print()
    print("Per-boundary winner (highest MAX phon-score):")
    for lang in LANGS:
        pct = winner_count[lang] / len(winners)
        print(f"  {lang:<8}: wins {winner_count[lang]}/{len(winners)} "
              f"boundaries ({pct:.1%})")
    expected_per_lang = len(winners) / len(LANGS)
    print(f"  (expected if equal: {expected_per_lang:.1f} each, "
          f"{1/len(LANGS):.1%})")

    # PERMUTATION: shuffle ids 10k times. Compute Syriac-wins fraction in each
    # shuffle. Compare to true.
    rng = random.Random(42)
    syr_win_true = winner_count["syriac"]
    null_syr_wins = []
    for _ in range(10000):
        order = common_ids.copy(); rng.shuffle(order)
        wins = 0
        for k in range(n - 1):
            b = (order[k], order[k+1])
            scores = {lang: max_scores[lang][b] for lang in LANGS}
            if max(scores.keys(), key=lambda l: scores[l]) == "syriac":
                wins += 1
        null_syr_wins.append(wins)
    null_syr_wins = np.array(null_syr_wins)
    z = (syr_win_true - null_syr_wins.mean()) / null_syr_wins.std()
    p = float((null_syr_wins >= syr_win_true).mean())
    print()
    print(f"Permutation test on 'Syriac wins MAX at boundary':")
    print(f"  TRUE order: Syriac wins {syr_win_true} boundaries")
    print(f"  NULL (10k shuffles): mean={null_syr_wins.mean():.1f}±{null_syr_wins.std():.1f}")
    print(f"  z={z:.2f}, p={p:.4f}")

    # Save
    summary = {
        "n_common_logia": n,
        "phon_threshold": PHON_THR,
        "mean_max_phon_at_true_boundaries": {
            lang: float(np.mean(true_max[lang])) for lang in LANGS},
        "median_max_phon_at_true_boundaries": {
            lang: float(np.median(true_max[lang])) for lang in LANGS},
        "winners_per_lang": winner_count,
        "winners_pct_per_lang": {l: winner_count[l]/len(winners) for l in LANGS},
        "syriac_wins_test": {
            "true_syriac_wins": int(syr_win_true),
            "null_mean": float(null_syr_wins.mean()),
            "null_std": float(null_syr_wins.std()),
            "z_score": float(z),
            "p_value": p,
            "n_perms": 10000,
        },
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
