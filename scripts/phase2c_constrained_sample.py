#!/usr/bin/env python3
"""
Phase 2C — constrained per-logion importance sampling.

Each logion is translated independently via stochastic sampling weighted by
P(s|c) * P_bigram(s|s_prev)^lambda. No information about adjacent logia
ever flows into the sampler — this is the catchword-blind analog of Phase 1
MC, with Syriac fluency added.

We run 1,000 simulations, each producing a complete Syriac translation for
all 115 logia, and measure the catchword distribution across simulations.
This sharpens the answer to: 'how much catchword density is unavoidable from
the Coptic→Syriac mapping + Syriac bigram structure, even when no part of
the system "knows" about adjacency?'

Calibration matches Phase 1: filter_pct=80, threshold=0.65, blocked set
derived from MAP translation.

Inputs:
  data/processed/phase2a_beam_results.json    (reuse blocked set + bigram LM build is duplicate but cheap)
  data/processed/lexical_mapping/coptic_to_syriac.jsonl
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl
  data/processed/got_logia/thomas_logia.jsonl

Output:
  data/processed/phase2c_constrained_results.json
  analysis/figures/phase2c_distribution.png
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
P2A_RESULTS = REPO_ROOT / "data" / "processed" / "phase2a_beam_results.json"
OUT_JSON = REPO_ROOT / "data" / "processed" / "phase2c_constrained_results.json"
OUT_FIG = REPO_ROOT / "analysis" / "figures" / "phase2c_distribution.png"

LAMBDAS = [0.0, 1.0]   # 0 = pure lexical map sampling = Phase 1 MC equivalent;
                       # 1 = heavy LM weight = "most fluent" sampling
N_SIMULATIONS = 200    # 200 gives ~3% CI on the mean; 1000 was 5x slower
                       # without proportionate gain in conclusion strength
PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
MAX_CANDIDATES_PER_WORD = 15
COPTIC_CONTENT_POS = frozenset({"N", "V", "VBD", "VSTAT", "PROPN", "NPROP",
                                 "ADJ", "ADV"})
SEED = 42


def load_lexmap():
    mp = {}
    with LEX_MAP.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cands = [(c["syriac_lemma"], c["prob"])
                     for c in r["candidates"][:MAX_CANDIDATES_PER_WORD]]
            mp[r["coptic_lemma"]] = cands
    return mp


def load_thomas_logia():
    by_log = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for tok in r["tokens"]:
                if not tok.get("lemma"): continue
                if tok.get("pos") not in COPTIC_CONTENT_POS: continue
                by_log[r["logion"]].append(tok["lemma"])
    return dict(by_log)


def build_bigram_lm():
    uni = Counter(); bi = Counter()
    with PESHITTA_LEM.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lems = [t["lemma"] for t in r["tokens"]
                    if t.get("lemma") and t["lemma"] != "_"]
            if not lems: continue
            prev = "<BOS>"
            for tok in lems:
                uni[prev] += 1
                bi[(prev, tok)] += 1
                prev = tok
            uni[prev] += 1
            bi[(prev, "<EOS>")] += 1
    V = len(set(uni.keys()) | {"<EOS>"})
    return uni, bi, V


def sample_translation(coptic_lemmas, lexmap, uni, bi, V, lam, rng):
    out = []
    prev = "<BOS>"
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if not cands:
            out.append(c); prev = c; continue
        weights = []
        toks = []
        for s, p in cands:
            lex = math.log(max(p, 1e-12))
            if lam > 0:
                lm = math.log(bi.get((prev, s), 0) + 1) - math.log(uni.get(prev, 0) + V)
            else:
                lm = 0.0
            weights.append(lex + lam * lm)
            toks.append(s)
        m = max(weights)
        ws = [math.exp(w - m) for w in weights]
        Z = sum(ws)
        ws = [w / Z for w in ws]
        out.append(rng.choices(toks, weights=ws, k=1)[0])
        prev = out[-1]
    return out


def compute_blocked_from_map(thomas, lexmap, uni, bi, V, filter_pct):
    """Use the deterministic top-1 MAP translation to derive blocked lemmas."""
    n_logia = len(thomas)
    cutoff = filter_pct * n_logia / 100.0
    lemma_logia_count = Counter()
    for L, c_lems in thomas.items():
        seq = []
        prev = "<BOS>"
        for c in c_lems:
            cands = lexmap.get(c)
            if not cands:
                seq.append(c); prev = c; continue
            # MAP = pick argmax of P(s|c) (no LM)
            best = max(cands, key=lambda x: x[1])[0]
            seq.append(best); prev = best
        for lem in set(seq):
            lemma_logia_count[lem] += 1
    return {lem for lem, c in lemma_logia_count.items() if c > cutoff}


def detect_overall(translation, det, blocked):
    sorted_L = sorted(translation.keys())
    n = len(sorted_L)
    pair_counts = []
    cw_left, cw_right = set(), set()
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        ta = [{"form": s, "lemma": s, "parse": "MS-EMP"}
              for s in translation[L] if s not in blocked]
        tb = [{"form": s, "lemma": s, "parse": "MS-EMP"}
              for s in translation[Ln] if s not in blocked]
        cws = det.detect(ta, tb)
        pair_counts.append(len(cws))
        if len(cws) > 0:
            cw_right.add(L); cw_left.add(Ln)
    both = len(cw_left & cw_right)
    iso = len(set(sorted_L) - cw_left - cw_right)
    one = n - both - iso
    return (sum(pair_counts), 100*both/n, 100*one/n, 100*iso/n)


def main():
    rng = random.Random(SEED)

    print("=== Loading data ===")
    lexmap = load_lexmap()
    thomas = load_thomas_logia()
    print(f"  lexmap entries: {len(lexmap)}")
    print(f"  thomas logia:   {len(thomas)}")

    print("=== Building bigram LM ===")
    uni, bi, V = build_bigram_lm()
    print(f"  vocab: {V}, unigrams: {sum(uni.values())}, bigrams: {len(bi)}")

    print(f"=== Computing blocked set (filter_pct={FILTER_PCT}) ===")
    blocked = compute_blocked_from_map(thomas, lexmap, uni, bi, V, FILTER_PCT)
    print(f"  blocked: {len(blocked)} lemmas: {list(blocked)[:8]}")

    det = CatchwordDetector("syriac",
                            phonological_threshold=PHON_THRESHOLD,
                            require_content_pos=False)

    results = {}
    print()
    print(f"=== Constrained sampling ({N_SIMULATIONS} sims/lambda) ===")
    print(f"{'λ':>5s} {'mean':>7s} {'std':>6s} {'p05':>6s} {'p95':>6s} "
          f"{'min':>5s} {'max':>5s}")
    for lam in LAMBDAS:
        totals = []
        boths = []; ones = []; isos = []
        for sim in range(N_SIMULATIONS):
            translation = {}
            for L, c_lems in thomas.items():
                if not c_lems:
                    translation[L] = []
                    continue
                translation[L] = sample_translation(c_lems, lexmap, uni, bi,
                                                     V, lam, rng)
            tot, b, o, i = detect_overall(translation, det, blocked)
            totals.append(tot); boths.append(b); ones.append(o); isos.append(i)
        results[lam] = {
            "mean": float(np.mean(totals)),
            "std":  float(np.std(totals)),
            "p05":  float(np.percentile(totals, 5)),
            "p95":  float(np.percentile(totals, 95)),
            "min":  int(min(totals)),
            "max":  int(max(totals)),
            "p_geq_perrin": float(np.mean([t >= 502 for t in totals])),
            "both_mean": float(np.mean(boths)),
            "one_mean":  float(np.mean(ones)),
            "iso_mean":  float(np.mean(isos)),
            "totals": totals,
        }
        r = results[lam]
        print(f"{lam:>5.1f} {r['mean']:>7.1f} {r['std']:>6.1f} {r['p05']:>6.0f} "
              f"{r['p95']:>6.0f} {r['min']:>5d} {r['max']:>5d}   "
              f"both={r['both_mean']:.1f}% iso={r['iso_mean']:.1f}%   "
              f"P(≥502)={r['p_geq_perrin']:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        # Strip the per-sim totals from one of them for compactness
        json.dump({
            "config": {
                "n_simulations": N_SIMULATIONS,
                "lambdas": LAMBDAS,
                "phon_threshold": PHON_THRESHOLD,
                "filter_pct": FILTER_PCT,
                "n_blocked": len(blocked),
                "seed": SEED,
            },
            "perrin_target_total": 502,
            "phase1_mc_mean_total": 195.4,
            "results_per_lambda": {str(k): v for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_JSON}")

    # Histogram figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(min(results[lam]["min"] for lam in LAMBDAS), 100),
                        max(max(results[lam]["max"] for lam in LAMBDAS), 510),
                        50)
    colors = {0.0: "C0", 1.0: "C2"}
    for lam in LAMBDAS:
        ax.hist(results[lam]["totals"], bins=bins, alpha=0.55,
                 color=colors.get(lam, "gray"),
                 label=f"λ={lam} (mean {results[lam]['mean']:.0f})",
                 density=False)
    ax.axvline(502, color="C3", linestyle="--", linewidth=2,
                label="Perrin (502)")
    ax.axvline(195.4, color="C1", linestyle=":", linewidth=2,
                label="Phase 1 MC mean (195)")
    ax.set_xlabel("Total catchwords across 114 adjacent logion pairs")
    ax.set_ylabel("# of simulations")
    ax.set_title(f"Phase 2C — Constrained sampling distribution "
                  f"(N={N_SIMULATIONS} per λ)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
