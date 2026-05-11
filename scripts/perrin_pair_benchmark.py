#!/usr/bin/env python3
"""
Perrin specific-pair benchmark.

For each of Perrin's 502 cited Syriac catchword pairs (digitized in
data/processed/perrin_catchwords/pair_comparison.json), classify with OUR
detector (phon threshold 0.6) and tally:

  semantic / etymological / phonological / below-threshold

Then compare this rate distribution to:
  (a) RANDOM pairs drawn from the same Syriac corpus (sampled from non-adjacent
      cell entries in our Phase 2B Gemini Syriac Thomas).
  (b) Pairs at NON-PERRIN boundaries (Gemini-found catchwords that Perrin
      didn't cite).

If Perrin's pairs are systematically more sound-similar than (a) random or
(b) Gemini-canonical pairs, his selection has language-specific phonological
structure. If they're not, Williams' bias critique extends.

Output:
  data/perrin_direct/perrin_pair_benchmark.json
  analysis/figures/perrin_pair_benchmark.png
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import (  # noqa: E402
    Catchword, consonantal, weighted_levenshtein, phonological_score,
)
from phase1_montecarlo.language_data import get_profile  # noqa: E402
from scripts.crossling_permutation_test import (  # noqa: E402
    make_tokens, load_sedra_lookup, LLM_DIR_SYR, N_LOGIA,
)

PAIR_COMP = REPO_ROOT / "data" / "processed" / "perrin_catchwords" / "pair_comparison.json"
OUT_JSON = REPO_ROOT / "data" / "perrin_direct" / "perrin_pair_benchmark.json"
FIG = REPO_ROOT / "analysis" / "figures" / "perrin_pair_benchmark.png"

PHON_THRESHOLD = 0.6
SYR = get_profile("syriac")


def classify_pair(skel_a: str, skel_b: str, profile=SYR,
                   thr: float = PHON_THRESHOLD) -> tuple[str, float]:
    """Same classification logic as CatchwordDetector but on raw skeletons.
    Perrin pairs are recorded as surface forms; we treat each surface form
    as both lemma and skeleton (skel field already provided)."""
    if not skel_a or not skel_b:
        return ("invalid", 0.0)
    if skel_a == skel_b:
        return ("semantic_or_etym", 1.0)  # Perrin doesn't distinguish
    ca, cb = consonantal(skel_a), consonantal(skel_b)
    if ca == cb:
        return ("etymological", 0.8)
    dist = weighted_levenshtein(ca, cb, profile)
    score = 1.0 - dist / max(len(ca), len(cb))
    if score >= thr:
        return ("phonological", score)
    return ("below_threshold", score)


def load_perrin_pairs() -> list[tuple[str, str, str]]:
    """Return list of (boundary, syriac_word_a, syriac_word_b) for every
    cross-side pair at each Perrin boundary."""
    d = json.loads(PAIR_COMP.read_text(encoding="utf-8"))
    out = []
    for entry in d["per_boundary"]:
        b = entry["boundary"]
        a_words = [det["skel"] for det in entry["details"] if det["side"] == "a"]
        b_words = [det["skel"] for det in entry["details"] if det["side"] == "b"]
        for wa in a_words:
            for wb in b_words:
                if wa and wb:
                    out.append((b, wa, wb))
    return out


def load_syriac_token_pool() -> list[str]:
    """All Syriac surface tokens (skeleton-stripped) from variant-0 Thomas."""
    sedra = load_sedra_lookup()
    pool: list[str] = []
    for i in range(N_LOGIA):
        p = LLM_DIR_SYR / f"logion_{i:03d}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vs = d.get("variants", [])
        if vs and vs[0].get("success"):
            for tok in make_tokens(vs[0]["syriac_text"], "syriac", sedra):
                form = consonantal(tok.get("form", ""))
                if form:
                    pool.append(form)
    return pool


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    perrin_pairs = load_perrin_pairs()
    print(f"Perrin pairs: {len(perrin_pairs)}")
    perrin_classifications = []
    perrin_scores = []
    for b, wa, wb in perrin_pairs:
        kind, score = classify_pair(wa, wb)
        perrin_classifications.append(kind)
        perrin_scores.append(score)
    perrin_counts = Counter(perrin_classifications)
    print(f"\nPerrin pair classification (our detector, threshold={PHON_THRESHOLD}):")
    for k, n in sorted(perrin_counts.items(), key=lambda x: -x[1]):
        pct = n / len(perrin_pairs) * 100
        print(f"  {k:20s}: {n:>4d}  ({pct:.1f}%)")

    # Build random baseline: sample N pairs from the Syriac token pool
    pool = load_syriac_token_pool()
    print(f"\nSyriac token pool: {len(pool)} tokens")
    rng = random.Random(42)
    n_random = 10000
    random_classifications = []
    random_scores = []
    for _ in range(n_random):
        wa = rng.choice(pool); wb = rng.choice(pool)
        kind, score = classify_pair(wa, wb)
        random_classifications.append(kind)
        random_scores.append(score)
    random_counts = Counter(random_classifications)
    print(f"\nRandom pair classification (N={n_random}):")
    for k, n in sorted(random_counts.items(), key=lambda x: -x[1]):
        pct = n / n_random * 100
        print(f"  {k:20s}: {n:>4d}  ({pct:.1f}%)")

    # Statistical test
    print(f"\n{'='*70}")
    print(f"COMPARISON — fraction of pairs by class")
    print(f"{'='*70}")
    print(f"{'class':<20s} {'Perrin':>10s} {'Random':>10s} "
          f"{'Perrin/Random':>14s}")
    print("-" * 70)
    for k in ("semantic_or_etym", "etymological", "phonological",
              "below_threshold"):
        p_n = perrin_counts.get(k, 0)
        r_n = random_counts.get(k, 0)
        p_frac = p_n / len(perrin_pairs)
        r_frac = r_n / n_random if n_random else 0
        ratio = p_frac / r_frac if r_frac > 0 else float("inf")
        print(f"  {k:<18s}  {p_frac:>9.1%}  {r_frac:>9.1%}  {ratio:>13.2f}x")

    # Mann–Whitney on scores
    pn = np.array(perrin_scores)
    rn = np.array(random_scores)
    U, p_overall = mannwhitneyu(pn, rn, alternative="greater")
    print(f"\nMann–Whitney (Perrin scores > random): U={U:.0f}, p={p_overall:.2e}")
    print(f"  Median Perrin score: {np.median(pn):.3f}")
    print(f"  Median random score: {np.median(rn):.3f}")

    # Restricted comparison: only non-identical (i.e., exclude semantic)
    pn_phon = pn[pn < 1.0]
    rn_phon = rn[rn < 1.0]
    if len(pn_phon) > 0 and len(rn_phon) > 0:
        U2, p_phon = mannwhitneyu(pn_phon, rn_phon, alternative="greater")
        print(f"\nMann–Whitney (excl. identical-pairs): "
              f"U={U2:.0f}, p={p_phon:.2e}")
        print(f"  Median Perrin (non-identical) score: {np.median(pn_phon):.3f}")
        print(f"  Median random (non-identical) score: {np.median(rn_phon):.3f}")

    # Save
    summary = {
        "phon_threshold": PHON_THRESHOLD,
        "n_perrin_pairs": len(perrin_pairs),
        "n_random_pairs": n_random,
        "perrin_class_counts": dict(perrin_counts),
        "random_class_counts": dict(random_counts),
        "perrin_median_score": float(np.median(pn)),
        "random_median_score": float(np.median(rn)),
        "mannwhitney_perrin_gt_random_p": float(p_overall),
        "perrin_median_score_nonidentical":
            float(np.median(pn_phon)) if len(pn_phon) else None,
        "random_median_score_nonidentical":
            float(np.median(rn_phon)) if len(rn_phon) else None,
        "mannwhitney_nonidentical_p":
            float(p_phon) if len(pn_phon) > 0 and len(rn_phon) > 0 else None,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # Figure: histograms of scores
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(pn, bins=bins, alpha=0.7, color="#c0392b", label="Perrin pairs",
            edgecolor="black", linewidth=0.4, density=True)
    ax.hist(rn, bins=bins, alpha=0.45, color="#7f8c8d", label="Random pairs",
            edgecolor="black", linewidth=0.4, density=True)
    ax.axvline(PHON_THRESHOLD, color="grey", ls="--",
                label=f"thr = {PHON_THRESHOLD}")
    ax.set_xlabel("Phon similarity score (Syriac)")
    ax.set_ylabel("density")
    ax.set_title(f"Perrin pairs vs random Syriac pairs (all)\n"
                  f"Mann–Whitney p = {p_overall:.2e}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if len(pn_phon) and len(rn_phon):
        ax.hist(pn_phon, bins=bins, alpha=0.7, color="#c0392b",
                label="Perrin (non-identical)",
                edgecolor="black", linewidth=0.4, density=True)
        ax.hist(rn_phon, bins=bins, alpha=0.45, color="#7f8c8d",
                label="Random (non-identical)",
                edgecolor="black", linewidth=0.4, density=True)
        ax.axvline(PHON_THRESHOLD, color="grey", ls="--",
                    label=f"thr = {PHON_THRESHOLD}")
        ax.set_xlabel("Phon similarity score (Syriac)")
        ax.set_ylabel("density")
        ax.set_title(f"Restricted to non-identical pairs\n"
                      f"Mann–Whitney p = {p_phon:.2e}")
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(f"\nWrote {OUT_JSON.name}, {FIG.name}")


if __name__ == "__main__":
    main()
