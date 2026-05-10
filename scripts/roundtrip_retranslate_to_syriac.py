#!/usr/bin/env python3
"""
Round-trip Step 3 + 4 — given the Coptic intermediate from Step 2:

  (a) Translate Coptic → Syriac three ways (MAP, beam λ=0.3, MC random sampling)
  (b) Run the Phase 1 catchword detector on consecutive strophe pairs at each
      stage: (1) original Syriac, (2) Coptic intermediate, (3) recovered Syriac
  (c) Compute the recovery ratio: catchwords(recovered) / catchwords(coptic_intermediate)

Calibration matches Phase 1: filter_pct=80 applied separately at each stage's
language. We compute blocked sets independently for original-Syriac, for
Coptic-intermediate, and for recovered-Syriac — each from its own MAP-style
ground truth, so high-frequency function-word lemmas in each language are
filtered consistently.

Inputs:
  data/processed/roundtrip/coptic_intermediate.jsonl
  data/processed/lexical_mapping/coptic_to_syriac.jsonl   (forward map)
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl  (for bigram LM)
  data/processed/syriac_strophes.jsonl   (for original Syriac stage)

Output:
  data/processed/roundtrip/results.json
  data/processed/roundtrip/recovered_syriac_{method}.jsonl
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

COPTIC_INT = REPO_ROOT / "data" / "processed" / "roundtrip" / "coptic_intermediate.jsonl"
LEX_MAP   = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
PESHITTA  = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
ORIG_SYR  = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
SEDRA     = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"

OUT_RES   = REPO_ROOT / "data" / "processed" / "roundtrip" / "results.json"
OUT_DIR   = REPO_ROOT / "data" / "processed" / "roundtrip"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
MAX_CANDIDATES = 15
BEAM_WIDTH = 20
N_MC_SIMS = 50    # 50 sims gives ~5% CI on the mean across ~thousands of
                   # strophe pairs; full N=10000 would take >hour at this scale
SEED = 42

CONTENT_PREFIXES = ("MS-", "FS-", "MP-", "FP-", "CS-", "CP-",
                    "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",
                    "SHAPHEL", "ESTAPHAL", "ETHTAPHAL")


def is_syr_content(parse):
    return any(parse.startswith(p) for p in CONTENT_PREFIXES) if parse else False


def strip_voc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                    if not unicodedata.combining(c))


def load_intermediate():
    by_corpus = defaultdict(lambda: defaultdict(list))
    with COPTIC_INT.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["author"], r["source_file"])
            by_corpus[r["author"]][r["source_file"]].append(r)
    for a in by_corpus:
        for sf in by_corpus[a]:
            by_corpus[a][sf].sort(key=lambda r: r.get("strophe_index", 0))
    return by_corpus


def load_lexmap():
    mp = {}
    with LEX_MAP.open() as f:
        for line in f:
            r = json.loads(line)
            mp[r["coptic_lemma"]] = [(c["syriac_lemma"], c["prob"])
                                      for c in r["candidates"][:MAX_CANDIDATES]]
    return mp


def build_bigram_lm():
    uni = Counter(); bi = Counter()
    with PESHITTA.open() as f:
        for line in f:
            r = json.loads(line)
            lems = [t["lemma"] for t in r["tokens"]
                    if t.get("lemma") and t["lemma"] != "_"]
            if not lems:
                continue
            prev = "<BOS>"
            for tok in lems:
                uni[prev] += 1; bi[(prev, tok)] += 1; prev = tok
            uni[prev] += 1; bi[(prev, "<EOS>")] += 1
    return uni, bi, len(set(uni.keys()) | {"<EOS>"})


def map_translate(coptic_lemmas, lexmap):
    out = []
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if cands:
            best = max(cands, key=lambda x: x[1])[0]
            out.append(best)
        else:
            out.append(c)
    return out


def beam_translate(coptic_lemmas, lexmap, uni, bi, V, lam=0.3):
    beams = [(0.0, [], "<BOS>")]
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if not cands:
            beams = [(sc, seq + [c], c) for sc, seq, _ in beams]
            continue
        new = []
        for sc, seq, prev in beams:
            for s, p in cands:
                if p <= 0:
                    continue
                lex = math.log(p)
                lm = (math.log(bi.get((prev, s), 0) + 1)
                       - math.log(uni.get(prev, 0) + V)) if lam > 0 else 0.0
                new.append((sc + lex + lam * lm, seq + [s], s))
        new.sort(key=lambda x: -x[0])
        beams = new[:BEAM_WIDTH]
    return beams[0][1] if beams else []


def stoch_sample(coptic_lemmas, lexmap, rng):
    """Pure MC: P(s|c), no LM bias — matches Phase 1 MC."""
    out = []
    for c in coptic_lemmas:
        cands = lexmap.get(c)
        if not cands:
            out.append(c); continue
        toks, ps = zip(*cands)
        Z = sum(ps)
        ws = [p / Z for p in ps]
        out.append(rng.choices(toks, weights=ws, k=1)[0])
    return out


def to_tokens(lemmas):
    return [{"form": s, "lemma": s, "parse": "MS-EMP"} for s in lemmas if s]


def detect_consec(strophes_by_work, blocked, lang="syriac",
                   coptic_blocked=None):
    """Run detector on consecutive strophe pairs within each work; return
    pooled total + per-pair list. blocked is the lemma-string set to skip."""
    det_lang = "coptic" if lang == "coptic" else "syriac"
    det = CatchwordDetector(det_lang,
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    blk = blocked if blocked else set()
    total = 0
    pair_counts = []
    for work, strophes in strophes_by_work.items():
        for i in range(len(strophes) - 1):
            ta = [{"form": s, "lemma": s, "parse": "MS-EMP"}
                  for s in strophes[i] if s and s not in blk]
            tb = [{"form": s, "lemma": s, "parse": "MS-EMP"}
                  for s in strophes[i + 1] if s and s not in blk]
            cws = det.detect(ta, tb)
            n = len(cws)
            total += n
            pair_counts.append(n)
    return total, pair_counts


def compute_blocked(strophes_by_work, filter_pct):
    """Block lemmas appearing in >filter_pct% of strophes (pooled across works)."""
    all_strophes = []
    for w, sl in strophes_by_work.items():
        all_strophes.extend(sl)
    n = len(all_strophes)
    cutoff = filter_pct * n / 100.0
    lemma_strophe_count = Counter()
    for strophe in all_strophes:
        for lem in set(strophe):
            lemma_strophe_count[lem] += 1
    return {lem for lem, c in lemma_strophe_count.items() if c > cutoff}


def get_original_syr_strophes(corpus_filter):
    """Return {source_file: [[surface_word, ...], ...]} for one corpus from
    the Syriac strophes data, using consonantal surface forms (matches Phase 3.0)."""
    by_work = defaultdict(list)
    with ORIG_SYR.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("author") != corpus_filter:
                continue
            words = []
            for w in r["text_consonantal"].split():
                w = w.strip(":.,;!?·܀")
                if w:
                    words.append(w)
            by_work[r["source_file"]].append((r.get("strophe_index", 0), words))
    out = {}
    for sf, lst in by_work.items():
        lst.sort(key=lambda x: x[0])
        out[sf] = [w for _, w in lst]
    return out


def main():
    rng = random.Random(SEED)

    print("Loading Coptic intermediate…")
    by_corpus = load_intermediate()
    for c, works in by_corpus.items():
        n_str = sum(len(s) for s in works.values())
        print(f"  {c}: {n_str} strophes across {len(works)} works")

    print("Loading lexical map (Coptic→Syriac)…")
    lexmap = load_lexmap()
    print(f"  {len(lexmap)} entries")

    print("Building Peshitta bigram LM…")
    uni, bi, V = build_bigram_lm()
    print(f"  vocab={V}, bigrams={len(bi)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for corpus, works in by_corpus.items():
        print()
        print(f"=== {corpus} ===")
        t0 = time.time()

        # Original Syriac (consonantal surface forms — matches Phase 3.0)
        orig_works = get_original_syr_strophes(corpus)
        # Drop works with <2 strophes (no consec pairs)
        orig_works = {sf: sl for sf, sl in orig_works.items() if len(sl) >= 2}
        n_orig_strophes = sum(len(sl) for sl in orig_works.values())
        n_orig_pairs = sum(len(sl) - 1 for sl in orig_works.values())

        orig_blocked = compute_blocked(orig_works, FILTER_PCT)
        orig_total, orig_pairs = detect_consec(orig_works, orig_blocked, "syriac")

        # Coptic intermediate
        cop_works_dict = {sf: [s["coptic_lemmas"] for s in strophes]
                          for sf, strophes in works.items() if len(strophes) >= 2}
        cop_blocked = compute_blocked(cop_works_dict, FILTER_PCT)
        cop_total, cop_pairs = detect_consec(cop_works_dict, cop_blocked, "coptic")

        # Recovered MAP
        map_works = {sf: [map_translate(s["coptic_lemmas"], lexmap)
                          for s in strophes]
                     for sf, strophes in works.items() if len(strophes) >= 2}
        map_blocked = compute_blocked(map_works, FILTER_PCT)
        map_total, map_pairs = detect_consec(map_works, map_blocked, "syriac")

        # Recovered Beam (λ=0.3)
        beam_works = {sf: [beam_translate(s["coptic_lemmas"], lexmap, uni, bi, V, 0.3)
                            for s in strophes]
                      for sf, strophes in works.items() if len(strophes) >= 2}
        beam_blocked = compute_blocked(beam_works, FILTER_PCT)
        beam_total, beam_pairs = detect_consec(beam_works, beam_blocked, "syriac")

        # Recovered MC (200 sims, mean over sims)
        mc_totals = []
        for sim in range(N_MC_SIMS):
            sim_works = {sf: [stoch_sample(s["coptic_lemmas"], lexmap, rng)
                               for s in strophes]
                         for sf, strophes in works.items() if len(strophes) >= 2}
            sim_blocked = compute_blocked(sim_works, FILTER_PCT)
            sim_total, _ = detect_consec(sim_works, sim_blocked, "syriac")
            mc_totals.append(sim_total)

        elapsed = time.time() - t0
        print(f"  Stages (consecutive-pair catchword totals):")
        print(f"    Original Syriac:    {orig_total:>6d}  "
              f"({n_orig_pairs} pairs, blocked {len(orig_blocked)})")
        print(f"    Coptic intermediate:{cop_total:>6d}  "
              f"(blocked {len(cop_blocked)})")
        print(f"    Recovered MAP:      {map_total:>6d}  "
              f"(ratio {map_total/max(cop_total,1):.2f}× over Coptic)")
        print(f"    Recovered Beam:     {beam_total:>6d}  "
              f"(ratio {beam_total/max(cop_total,1):.2f}×)")
        mc_mean = float(np.mean(mc_totals))
        print(f"    Recovered MC mean:  {mc_mean:>6.0f}  "
              f"(ratio {mc_mean/max(cop_total,1):.2f}×, "
              f"CI [{np.percentile(mc_totals, 5):.0f}, "
              f"{np.percentile(mc_totals, 95):.0f}])")
        print(f"  Compute: {elapsed:.1f}s")

        results[corpus] = {
            "n_strophes_orig": n_orig_strophes,
            "n_strophe_pairs_orig": n_orig_pairs,
            "n_strophes_intermediate": sum(len(s) for s in works.values()),
            "original_syriac_total": orig_total,
            "coptic_intermediate_total": cop_total,
            "recovered_map_total": map_total,
            "recovered_beam_total": beam_total,
            "recovered_mc_mean": mc_mean,
            "recovered_mc_p05": float(np.percentile(mc_totals, 5)),
            "recovered_mc_p95": float(np.percentile(mc_totals, 95)),
            "ratio_map":  map_total / max(cop_total, 1),
            "ratio_beam": beam_total / max(cop_total, 1),
            "ratio_mc":   mc_mean   / max(cop_total, 1),
            "blocked_orig":   sorted(orig_blocked),
            "blocked_coptic": sorted(cop_blocked),
            "blocked_map":    sorted(map_blocked),
            "blocked_beam":   sorted(beam_blocked),
            "elapsed_s": elapsed,
        }

    print()
    print("=" * 80)
    print("ROUND-TRIP SUMMARY")
    print("=" * 80)
    print(f"{'Corpus':<10s} {'Original':>9s} {'Coptic':>8s} "
          f"{'MAP':>6s} {'Beam':>6s} {'MC':>6s}  "
          f"{'r_MAP':>6s} {'r_Beam':>7s} {'r_MC':>6s}")
    for c, d in sorted(results.items()):
        print(f"{c:<10s} {d['original_syriac_total']:>9d} "
              f"{d['coptic_intermediate_total']:>8d} "
              f"{d['recovered_map_total']:>6d} "
              f"{d['recovered_beam_total']:>6d} "
              f"{d['recovered_mc_mean']:>6.0f}  "
              f"{d['ratio_map']:>5.2f}× "
              f"{d['ratio_beam']:>6.2f}× "
              f"{d['ratio_mc']:>5.2f}×")
    print()
    print("(r_X = ratio of recovered total / Coptic-intermediate total)")
    print("(For Thomas reference: r_MC=0.83, r_MAP=1.30, r_Perrin=1.87)")

    with OUT_RES.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "phon_threshold": PHON_THRESHOLD,
                "filter_pct": FILTER_PCT,
                "n_mc_sims": N_MC_SIMS,
                "beam_lambda": 0.3,
                "beam_width": BEAM_WIDTH,
                "max_candidates": MAX_CANDIDATES,
                "seed": SEED,
            },
            "per_corpus": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_RES}")


if __name__ == "__main__":
    main()
