#!/usr/bin/env python3
"""
Phase 3.0 — non-neural baseline: does the rule-based catchword detector find
significantly more catchwords between *consecutive* strophes than between
randomly-paired strophes from the same work?

If yes — Perrin's premise that Syriac literature uses catchword arrangement is
detectable at our calibrated threshold, and a contrastive model has a signal
to learn. If no — the catchword phenomenon is either not present in these
texts, not detectable at our threshold, or absorbed by noise. Either way it's
a finding worth reporting.

Inputs:
  data/processed/syriac_strophes.jsonl
  phase1_montecarlo/catchword_detector.py  (same calibration as Phase 1)

Output:
  data/processed/phase3_baseline_results.json
  analysis/figures/phase3_baseline.png
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
OUT_JSON = REPO_ROOT / "data" / "processed" / "phase3_baseline_results.json"
OUT_FIG = REPO_ROOT / "analysis" / "figures" / "phase3_baseline.png"

# Same calibration as Phase 1 MC: phonological_threshold=0.65.
# We have no lemmas/POS for patristic strophes, so we feed consonantal forms
# as both `form` and `lemma`, and disable the POS filter. The differential
# between consecutive and random pairs is robust to function-word inflation
# because both groups draw from the same word distribution.
PHON_THRESHOLD = 0.65
MIN_DISTANCE_FOR_RANDOM = 3  # require >=3 strophes apart for "random" pair
MIN_STROPHES_PER_WORK = 8     # need enough to form pairs and random comparisons
SEED = 42


def text_to_tokens(text_consonantal: str) -> list[dict]:
    """Each whitespace-delimited word becomes a token with consonantal form
    as both surface and lemma. parse='MS-EMP' is a syntactic placeholder so
    nothing is filtered when require_content_pos=False is the active mode."""
    tokens = []
    for w in text_consonantal.split():
        w = w.strip(":.,;!?·܀")
        if w:
            tokens.append({"form": w, "lemma": w, "parse": "MS-EMP"})
    return tokens


def load_strophes_by_work():
    works = defaultdict(list)
    with STROPHES.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r.get("author", "?"), r.get("source_file", "?"), r.get("work_title", "?"))
            works[key].append(r)
    # Sort each work by strophe_index to ensure consecutive ordering is correct
    for key, strophes in works.items():
        strophes.sort(key=lambda s: s.get("strophe_index", 0))
    return works


def consecutive_pair_counts(strophes, det):
    """Detector counts for each (i, i+1) pair within a work."""
    counts = []
    for i in range(len(strophes) - 1):
        ta = text_to_tokens(strophes[i]["text_consonantal"])
        tb = text_to_tokens(strophes[i + 1]["text_consonantal"])
        if not ta or not tb:
            continue
        cws = det.detect(ta, tb)
        counts.append(len(cws))
    return counts


def random_pair_counts(strophes, det, n_pairs, rng):
    """n_pairs random pairs of non-adjacent strophes from the same work."""
    counts = []
    n = len(strophes)
    if n < MIN_DISTANCE_FOR_RANDOM + 2:
        return counts
    attempts = 0
    while len(counts) < n_pairs and attempts < n_pairs * 20:
        i, j = rng.sample(range(n), 2)
        if abs(i - j) < MIN_DISTANCE_FOR_RANDOM:
            attempts += 1
            continue
        ta = text_to_tokens(strophes[i]["text_consonantal"])
        tb = text_to_tokens(strophes[j]["text_consonantal"])
        if not ta or not tb:
            attempts += 1
            continue
        cws = det.detect(ta, tb)
        counts.append(len(cws))
        attempts += 1
    return counts


def main():
    rng = random.Random(SEED)
    det = CatchwordDetector("syriac",
                            phonological_threshold=PHON_THRESHOLD,
                            require_content_pos=False)

    works = load_strophes_by_work()
    print(f"Loaded {sum(len(s) for s in works.values())} strophes "
          f"from {len(works)} works.")
    print(f"Filtering to works with >= {MIN_STROPHES_PER_WORK} strophes…")

    by_author = defaultdict(lambda: {"consec": [], "random": [], "n_works": 0,
                                     "n_strophes": 0})
    all_consec = []
    all_random = []

    for (author, src_file, title), strophes in works.items():
        if len(strophes) < MIN_STROPHES_PER_WORK:
            continue
        c = consecutive_pair_counts(strophes, det)
        if not c:
            continue
        r = random_pair_counts(strophes, det, len(c), rng)
        if not r:
            continue
        by_author[author]["consec"].extend(c)
        by_author[author]["random"].extend(r)
        by_author[author]["n_works"] += 1
        by_author[author]["n_strophes"] += len(strophes)
        all_consec.extend(c)
        all_random.extend(r)

    # Per-author stats + Mann-Whitney U test (one-sided: consecutive > random)
    per_author = {}
    print()
    print(f"{'Author':<20s} {'Works':>5s} {'Strophes':>8s} {'N_pairs':>7s} "
          f"{'Consec':>8s} {'Random':>8s} {'Diff':>6s} {'p_value':>10s}")
    print("-" * 88)
    for author, d in sorted(by_author.items()):
        c, r = d["consec"], d["random"]
        consec_mean = float(np.mean(c))
        random_mean = float(np.mean(r))
        diff = consec_mean - random_mean
        if len(c) >= 5 and len(r) >= 5:
            u_stat, p_val = stats.mannwhitneyu(c, r, alternative="greater")
        else:
            u_stat, p_val = float("nan"), float("nan")
        per_author[author] = {
            "n_works": d["n_works"], "n_strophes": d["n_strophes"],
            "n_pairs": len(c),
            "consec_mean": consec_mean, "consec_std": float(np.std(c)),
            "random_mean": random_mean, "random_std": float(np.std(r)),
            "diff": diff,
            "u_stat": float(u_stat) if not np.isnan(u_stat) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "consec_distribution": c,
            "random_distribution": r,
        }
        print(f"{author:<20s} {d['n_works']:>5d} {d['n_strophes']:>8d} "
              f"{len(c):>7d} {consec_mean:>8.2f} {random_mean:>8.2f} "
              f"{diff:>+6.2f} {p_val:>10.2e}")

    # Pooled overall test
    if len(all_consec) >= 5 and len(all_random) >= 5:
        overall_u, overall_p = stats.mannwhitneyu(
            all_consec, all_random, alternative="greater")
        overall = {
            "n_pairs_consec": len(all_consec),
            "n_pairs_random": len(all_random),
            "consec_mean": float(np.mean(all_consec)),
            "consec_std": float(np.std(all_consec)),
            "random_mean": float(np.mean(all_random)),
            "random_std": float(np.std(all_random)),
            "diff": float(np.mean(all_consec) - np.mean(all_random)),
            "u_stat": float(overall_u),
            "p_value": float(overall_p),
            "effect_size_cohens_d": (float(np.mean(all_consec) - np.mean(all_random))
                                     / float(np.sqrt((np.var(all_consec)
                                                      + np.var(all_random)) / 2)
                                            + 1e-9)),
        }
    else:
        overall = None

    print("-" * 88)
    if overall:
        print(f"{'POOLED':<20s} {'-':>5s} {'-':>8s} {overall['n_pairs_consec']:>7d} "
              f"{overall['consec_mean']:>8.2f} {overall['random_mean']:>8.2f} "
              f"{overall['diff']:>+6.2f} {overall['p_value']:>10.2e}")
        print(f"  Cohen's d (effect size): {overall['effect_size_cohens_d']:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "phonological_threshold": PHON_THRESHOLD,
                "min_distance_for_random": MIN_DISTANCE_FOR_RANDOM,
                "min_strophes_per_work": MIN_STROPHES_PER_WORK,
                "seed": SEED,
            },
            "per_author": per_author,
            "overall": overall,
        }, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {OUT_JSON}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    authors = list(per_author.keys())
    if not authors:
        print("No authors with sufficient data; skipping figure")
        return

    fig, axes = plt.subplots(1, len(authors), figsize=(3.2 * len(authors), 5),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for ax, author in zip(axes, authors):
        d = per_author[author]
        ax.violinplot([d["consec_distribution"], d["random_distribution"]],
                       positions=[1, 2], showmeans=True, widths=0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["consec", "random"])
        ax.set_title(f"{author}\nn={d['n_pairs']}, Δ={d['diff']:+.1f}, "
                      f"p={d['p_value']:.1e}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Catchwords per pair")
    fig.suptitle("Phase 3.0 baseline — consecutive vs random strophe pairs "
                  f"(threshold={PHON_THRESHOLD})", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
