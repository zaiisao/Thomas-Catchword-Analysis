#!/usr/bin/env python3
"""
Phase 2A — context-conditioned beam search over the Coptic→Syriac lexical map.

Translation score for a Syriac sequence s_1 ... s_n given Coptic c_1 ... c_n:

    score = sum_i [ log P(s_i | c_i) + lambda * log P_bigram(s_i | s_{i-1}) ]

P(s_i | c_i) comes from the EM-built lexical map.
P_bigram(s_i | s_{i-1}) is an add-1-smoothed bigram LM over Peshitta NT lemmas.
lambda controls how much Syriac fluency dominates lexical choice.

We sweep lambda in {0.0, 0.1, 0.3, 0.5, 1.0} and run the Phase 1 catchword
detector on the best beam at each setting. Also: stochastic variant — sample
N translations from the top-B beams weighted by score, to get a distribution
analogous to Phase 1 MC but conditioned on Syriac fluency.

Inputs:
  data/processed/lexical_mapping/coptic_to_syriac.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl  (LM training)
  data/processed/got_logia/thomas_logia.jsonl

Output:
  data/processed/phase2a_beam_results.json
  data/processed/phase2a_translations/lambda_{L}.jsonl   (per-logion translations)
  analysis/figures/phase2a_lambda_sweep.png
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LEX_MAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
PESHITTA_LEM = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
OUT_JSON = REPO_ROOT / "data" / "processed" / "phase2a_beam_results.json"
OUT_TRANS = REPO_ROOT / "data" / "processed" / "phase2a_translations"
OUT_FIG = REPO_ROOT / "analysis" / "figures" / "phase2a_lambda_sweep.png"

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0]
BEAM_WIDTH = 20
MAX_CANDIDATES_PER_WORD = 15  # cap to keep beam tractable
PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0  # match Phase 1: block Syriac lemmas in > FILTER_PCT% of logia
N_STOCHASTIC = 100  # # of stochastic translations per logion at lambda=0.3
SEED = 42

# Coptic POS tags considered "content" (matches calibrate_detector.py).
COPTIC_CONTENT_POS = frozenset({"N", "V", "VBD", "VSTAT", "PROPN", "NPROP",
                                 "ADJ", "ADV"})


def build_bigram_lm(verse_iter):
    """Return (uni_count, bi_count, vocab) for add-1 smoothing.

    P(curr | prev) = (bi_count[(prev, curr)] + 1) / (uni_count[prev] + V)
    where V = |vocab|.
    """
    uni = Counter()
    bi = Counter()
    for tokens in verse_iter:
        prev = "<BOS>"
        for tok in tokens:
            uni[prev] += 1
            bi[(prev, tok)] += 1
            prev = tok
        # Eos
        uni[prev] += 1
        bi[(prev, "<EOS>")] += 1
    vocab = set(uni.keys()) | {"<EOS>"}
    return uni, bi, vocab


def log_bigram(prev, curr, uni, bi, V):
    return math.log(bi.get((prev, curr), 0) + 1) - math.log(uni.get(prev, 0) + V)


def load_lexmap():
    """Return {coptic_lemma: [(syr_lemma, prob), ...]} sorted by prob desc, capped."""
    mp = {}
    with LEX_MAP.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cands = [(c["syriac_lemma"], c["prob"])
                     for c in r["candidates"][:MAX_CANDIDATES_PER_WORD]]
            mp[r["coptic_lemma"]] = cands
    return mp


def load_thomas_logia():
    """Return {logion_id: [coptic_lemmas]} concatenated across paragraphs.

    Filter to content POS only — same convention as Phase 1.
    """
    by_log = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for tok in r["tokens"]:
                if not tok.get("lemma"):
                    continue
                if tok.get("pos") not in COPTIC_CONTENT_POS:
                    continue
                by_log[r["logion"]].append(tok["lemma"])
    return dict(by_log)


def beam_search(coptic_lemmas, lexmap, uni, bi, V, lam):
    """Beam-search Syriac translation. Returns top-BEAM_WIDTH (score, syriac_seq).

    Each beam state: (score, [syriac_lemmas])
    """
    beams = [(0.0, [], "<BOS>")]  # (score, sequence, prev_token_for_LM)
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if not cands:
            # OOV — keep the Coptic surface so the next bigram still composes
            new = []
            for sc, seq, prev in beams:
                new.append((sc, seq + [c], c))
            beams = new
            continue
        new = []
        for sc, seq, prev in beams:
            for s_lem, prob in cands:
                if prob <= 0:
                    continue
                lex = math.log(prob)
                lm_score = log_bigram(prev, s_lem, uni, bi, V) if lam > 0 else 0.0
                new.append((sc + lex + lam * lm_score, seq + [s_lem], s_lem))
        # Prune
        new.sort(key=lambda x: -x[0])
        beams = new[:BEAM_WIDTH]
    return [(sc, seq) for sc, seq, _ in beams]


def stochastic_sample(coptic_lemmas, lexmap, uni, bi, V, lam, rng):
    """One stochastic sample: at each position, sample from candidate distribution
    proportional to P(s|c) * P_bigram(s|prev)^lambda."""
    out = []
    prev = "<BOS>"
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if not cands:
            out.append(c)
            prev = c
            continue
        weights = []
        toks = []
        for s, p in cands:
            lex = math.log(max(p, 1e-12))
            lm_s = log_bigram(prev, s, uni, bi, V) if lam > 0 else 0.0
            weights.append(lex + lam * lm_s)
            toks.append(s)
        # Softmax
        m = max(weights)
        ws = [math.exp(w - m) for w in weights]
        Z = sum(ws)
        ws = [w / Z for w in ws]
        choice = rng.choices(toks, weights=ws, k=1)[0]
        out.append(choice)
        prev = choice
    return out


def make_tokens(syriac_lemmas, blocked=None):
    """Wrap into the dict shape the detector expects.
    If `blocked` is given, drop tokens whose lemma is in the blocked set
    (matches Phase 1's filter_pct mechanism)."""
    blocked = blocked or set()
    return [{"form": s, "lemma": s, "parse": "MS-EMP"}
            for s in syriac_lemmas if s not in blocked]


def compute_blocked_set(translation_per_logion, filter_pct):
    """Block Syriac lemmas appearing in > filter_pct% of logia under the
    given translation — same logic as phase1_montecarlo/monte_carlo.py
    line 203-220."""
    n_logia = len(translation_per_logion)
    cutoff = filter_pct * n_logia / 100.0
    lemma_logia_count = Counter()
    for L, lems in translation_per_logion.items():
        for lem in set(lems):  # set: count each lemma at most once per logion
            lemma_logia_count[lem] += 1
    blocked = {lem for lem, c in lemma_logia_count.items() if c > cutoff}
    return blocked


def detect_overall(translation_per_logion, det, blocked=None):
    """Return overall stats: total catchwords, both/one/iso percentages."""
    sorted_L = sorted(translation_per_logion.keys())
    pair_counts = []
    cw_left, cw_right = set(), set()
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        ta = make_tokens(translation_per_logion[L], blocked)
        tb = make_tokens(translation_per_logion[Ln], blocked)
        cws = det.detect(ta, tb)
        pair_counts.append(len(cws))
        if len(cws) > 0:
            cw_right.add(L)
            cw_left.add(Ln)
    n = len(sorted_L)
    both = len(cw_left & cw_right)
    iso = len(set(sorted_L) - cw_left - cw_right)
    one = n - both - iso
    return {
        "total": int(sum(pair_counts)),
        "both_pct": 100 * both / n,
        "one_pct": 100 * one / n,
        "iso_pct": 100 * iso / n,
        "pair_counts": pair_counts,
    }


def main():
    rng = random.Random(SEED)

    print("=== Building bigram LM over Peshitta NT lemmas ===")
    def peshitta_iter():
        with PESHITTA_LEM.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                lems = [t["lemma"] for t in r["tokens"]
                        if t.get("lemma") and t["lemma"] != "_"]
                if lems:
                    yield lems

    uni, bi, vocab = build_bigram_lm(peshitta_iter())
    V = len(vocab)
    print(f"  vocab={V}, unigrams={sum(uni.values()):,}, bigrams={len(bi):,}")

    print()
    print("=== Loading lexical map + Thomas ===")
    lexmap = load_lexmap()
    thomas = load_thomas_logia()
    print(f"  lexmap entries: {len(lexmap)}")
    print(f"  thomas logia:   {len(thomas)}")
    print(f"  total content lemmas across logia: "
          f"{sum(len(v) for v in thomas.values())}")

    det = CatchwordDetector("syriac",
                            phonological_threshold=PHON_THRESHOLD,
                            require_content_pos=False)

    OUT_TRANS.mkdir(parents=True, exist_ok=True)

    # First produce the MAP translation (lambda=0.0 best beam) and use it
    # to derive the blocked-lemma set, exactly as Phase 1 does.
    print()
    print(f"=== Computing blocked set from MAP translation (filter_pct={FILTER_PCT}) ===")
    map_translation = {}
    for L, c_lems in thomas.items():
        if not c_lems:
            map_translation[L] = []
            continue
        beams = beam_search(c_lems, lexmap, uni, bi, V, 0.0)
        map_translation[L] = beams[0][1] if beams else []
    blocked = compute_blocked_set(map_translation, FILTER_PCT)
    print(f"  Blocked {len(blocked)} Syriac lemmas appearing in >{FILTER_PCT}% of logia.")
    if blocked:
        print(f"  Sample blocked: {list(blocked)[:8]}")

    sweep_results = {}
    print()
    print("=== Beam search lambda sweep (filter_pct=80, threshold=0.65) ===")
    print(f"{'λ':>5s} {'total':>6s} {'both%':>6s} {'one%':>6s} {'iso%':>6s} "
          f"{'mean_pair':>10s}")
    for lam in LAMBDAS:
        translation = {}
        with (OUT_TRANS / f"lambda_{lam}.jsonl").open("w", encoding="utf-8") as out:
            for L, c_lems in thomas.items():
                if not c_lems:
                    translation[L] = []
                    continue
                beams = beam_search(c_lems, lexmap, uni, bi, V, lam)
                if not beams:
                    translation[L] = []
                    continue
                _, best_seq = beams[0]
                translation[L] = best_seq
                out.write(json.dumps({"logion": L,
                                       "coptic_lemmas": c_lems,
                                       "syriac_lemmas": best_seq,
                                       "score": beams[0][0],
                                       "n_beams": len(beams)},
                                      ensure_ascii=False) + "\n")
        stats = detect_overall(translation, det, blocked)
        sweep_results[lam] = stats
        mean_pair = float(np.mean(stats["pair_counts"]))
        print(f"{lam:>5.1f} {stats['total']:>6d} {stats['both_pct']:>6.1f} "
              f"{stats['one_pct']:>6.1f} {stats['iso_pct']:>6.1f} {mean_pair:>10.2f}")

    # Stochastic distribution at lam=0.3 (a moderate weight)
    print()
    print(f"=== Stochastic sampling distribution (λ=0.3, n={N_STOCHASTIC}) ===")
    LAM_STOCH = 0.3
    totals = []
    bot, om, iso = [], [], []
    for sim in range(N_STOCHASTIC):
        translation = {}
        for L, c_lems in thomas.items():
            translation[L] = stochastic_sample(c_lems, lexmap, uni, bi, V,
                                                LAM_STOCH, rng) if c_lems else []
        stats = detect_overall(translation, det, blocked)
        totals.append(stats["total"])
        bot.append(stats["both_pct"])
        om.append(stats["one_pct"])
        iso.append(stats["iso_pct"])

    stoch = {
        "lambda": LAM_STOCH,
        "n_samples": N_STOCHASTIC,
        "total_mean": float(np.mean(totals)),
        "total_std": float(np.std(totals)),
        "total_p05": float(np.percentile(totals, 5)),
        "total_p95": float(np.percentile(totals, 95)),
        "both_mean": float(np.mean(bot)),
        "one_mean": float(np.mean(om)),
        "iso_mean": float(np.mean(iso)),
    }
    print(f"  total: mean={stoch['total_mean']:.1f} "
          f"(±{stoch['total_std']:.1f}) "
          f"[{stoch['total_p05']:.0f}–{stoch['total_p95']:.0f}]")
    print(f"  conn: both={stoch['both_mean']:.1f}% "
          f"one={stoch['one_mean']:.1f}% iso={stoch['iso_mean']:.1f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "beam_width": BEAM_WIDTH,
                "max_candidates": MAX_CANDIDATES_PER_WORD,
                "phon_threshold": PHON_THRESHOLD,
                "filter_pct": FILTER_PCT,
                "n_blocked_lemmas": len(blocked),
                "lambdas": LAMBDAS,
                "n_stochastic": N_STOCHASTIC,
                "seed": SEED,
            },
            "bigram_lm": {"vocab": V, "n_unigrams": int(sum(uni.values())),
                           "n_bigrams": len(bi)},
            "best_beam_per_lambda": {str(k): v for k, v in sweep_results.items()},
            "stochastic_lam03": stoch,
            "perrin_target_total": 502,
            "phase1_mc_mean_total": 195.4,
            "perrin_both_pct": 89.0,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUT_JSON}")

    # Plot lambda sweep
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    lams = LAMBDAS
    totals = [sweep_results[l]["total"] for l in lams]
    boths = [sweep_results[l]["both_pct"] for l in lams]
    isos = [sweep_results[l]["iso_pct"] for l in lams]
    axes[0].plot(lams, totals, "o-", color="C0", label="Phase 2A best beam")
    axes[0].axhline(195.4, color="C1", linestyle="--", label="Phase 1 MC mean (195)")
    axes[0].axhline(305, color="C2", linestyle=":", label="Phase 1 MAP (305)")
    axes[0].axhline(502, color="C3", linestyle="--", label="Perrin (502)")
    axes[0].errorbar([LAM_STOCH], [stoch["total_mean"]],
                      yerr=[[stoch["total_mean"] - stoch["total_p05"]],
                             [stoch["total_p95"] - stoch["total_mean"]]],
                      fmt="s", color="C4", label=f"Stochastic λ={LAM_STOCH}")
    axes[0].set_xlabel("λ (LM weight)")
    axes[0].set_ylabel("Total catchwords across 114 pairs")
    axes[0].set_title("Phase 2A — Beam-search catchword count vs λ")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    width = 0.18
    x = np.arange(len(lams))
    axes[1].bar(x - width, boths, width, label="Both-sides %", color="C0")
    axes[1].bar(x,        [sweep_results[l]["one_pct"] for l in lams],
                 width, label="One-side %",  color="C1")
    axes[1].bar(x + width, isos, width, label="Isolated %",  color="C2")
    axes[1].axhline(89, color="C0", linestyle=":", alpha=0.6, label="Perrin both (89%)")
    axes[1].axhline(0, color="C2", linestyle=":", alpha=0.6, label="Perrin iso (0%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(l) for l in lams])
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("% of logia")
    axes[1].set_title("Phase 2A — Connectivity vs λ")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
