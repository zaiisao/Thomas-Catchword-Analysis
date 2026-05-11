#!/usr/bin/env python3
"""
Perrin test with VANILLA Levenshtein — same edit-cost function for all
languages. Eliminates the cross-language detector-liberality confound:
no confusion-group bonuses, no weak-consonant weighting. Just plain
unit-cost edit distance on consonantal skeletons.

If Syriac arrangement is genuinely language-specific (Perrin's claim),
even the FAIR detector should show Syriac leading on Thomas.

Output: data/perrin_direct/vanilla_{corpus}_{lang}_v{variant}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import consonantal  # noqa: E402
from scripts.phon_only_one import (  # noqa: E402
    load_proverbs, load_q, load_thomas, FILTER_PCT, CORPUS_LANGS, CORPUS_SOURCE,
)

OUT_DIR = REPO_ROOT / "data" / "perrin_direct"

CORPUS_LOADERS = {"proverbs": load_proverbs, "q": load_q, "thomas": load_thomas}

FILTERS = {
    "all":  None,
    "phon": frozenset({"phonological", "etymological"}),
    "sem":  frozenset({"semantic"}),
}


# ---- Vanilla detector — no language-specific profile ----

def vanilla_levenshtein(a: str, b: str) -> float:
    """Plain unit-cost edit distance. Same for all languages."""
    if a == b: return 0.0
    n, m = len(a), len(b)
    if n == 0: return float(m)
    if m == 0: return float(n)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return float(dp[m])


def classify_vanilla(la: str, lb: str, threshold: float) -> tuple[str, float] | None:
    """Same classification logic as CatchwordDetector but with vanilla Lev."""
    if not la or not lb: return None
    if la == lb:
        return ("semantic", 1.0)
    ca, cb = consonantal(la), consonantal(lb)
    if not ca or not cb: return None
    if ca == cb:
        return ("etymological", 0.8)
    dist = vanilla_levenshtein(ca, cb)
    longest = max(len(ca), len(cb))
    score = 1.0 - dist / longest
    if score >= threshold:
        return ("phonological", score)
    return None


def compute_blocked(translations, filter_pct):
    if filter_pct >= 100: return set()
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix_vanilla(translations, blocked, ids, threshold):
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in ids}
    matrix: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
    t0 = time.time()
    for n_done, i in enumerate(ids):
        for j in ids:
            if i == j: continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = []
                continue
            # Dedup by lemma pair (matches existing detector behavior)
            seen = {}
            for tok_a in ta:
                la = tok_a["lemma"]
                if not la: continue
                for tok_b in tb:
                    lb = tok_b["lemma"]
                    if not lb: continue
                    key = (la, lb)
                    if key in seen: continue
                    res = classify_vanilla(la, lb, threshold)
                    if res:
                        seen[key] = res[0]
            matrix[(i, j)] = [(k[0], k[1], lt) for k, lt in seen.items()]
        if (n_done + 1) % max(1, len(ids) // 6) == 0:
            print(f"    row {n_done+1}/{len(ids)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  matrix done {time.time()-t0:.0f}s", flush=True)
    return matrix


def total_count_for_order(order, matrix, link_filter):
    total = 0
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), [])
        if link_filter is None:
            total += len(cell)
        else:
            total += sum(1 for c in cell if c[2] in link_filter)
    return total


def run_perm_total(matrix, ids, link_filter, n_perms, seed):
    rng = random.Random(seed)
    base = list(ids)
    out = np.empty(n_perms, dtype=np.int32)
    for p in range(n_perms):
        sh = base.copy(); rng.shuffle(sh)
        out[p] = total_count_for_order(sh, matrix, link_filter)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPUS_LOADERS))
    ap.add_argument("--lang", required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phon-threshold", type=float, default=0.6)
    ap.add_argument("--filter-pct", type=float, default=FILTER_PCT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"vanilla_{args.corpus}_{args.lang}_v{args.variant}.json"
    if out_path.exists() and not args.force:
        print(f"already exists: {out_path}"); return

    is_source = args.lang == CORPUS_SOURCE[args.corpus]
    if is_source and args.variant > 0:
        out_path.write_text(json.dumps({"skipped": True}, indent=2))
        return

    t0 = time.time()
    print(f"[vanilla {args.corpus}/{args.lang} v{args.variant}] loading…", flush=True)
    ids, toks = CORPUS_LOADERS[args.corpus](args.lang, args.variant)
    usable_ids = [i for i in ids if toks.get(i)]
    if len(usable_ids) < len(ids) * 0.5:
        out_path.write_text(json.dumps({"skipped": True, "n_loaded": len(usable_ids)},
                                          indent=2))
        return
    blocked = compute_blocked(toks, args.filter_pct)
    print(f"  matrix ({len(usable_ids)} units, {len(blocked)} blocked)…",
          flush=True)
    matrix = precompute_matrix_vanilla(toks, blocked, usable_ids,
                                          args.phon_threshold)

    # Diagnostic
    n_phon = n_sem = n_etym = 0
    n_bound = len(usable_ids) - 1
    for pos in range(n_bound):
        for c in matrix.get((usable_ids[pos], usable_ids[pos+1]), []):
            if c[2] == "phonological": n_phon += 1
            elif c[2] == "etymological": n_etym += 1
            elif c[2] == "semantic": n_sem += 1

    results = {}
    for fname, link_filter in FILTERS.items():
        true_tot = total_count_for_order(usable_ids, matrix, link_filter)
        null = run_perm_total(matrix, usable_ids, link_filter,
                                args.n_perms, args.seed)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_tot - nm) / ns if ns > 0 else 0.0
        p = float((null >= true_tot).mean())
        results[fname] = {
            "true_total": int(true_tot), "null_mean": nm, "null_std": ns,
            "z_score": float(z), "p_value": p, "n_perms": args.n_perms,
        }
        print(f"  [{fname:4s}] true={true_tot:>5d}, null={nm:.1f}±{ns:.1f}, "
              f"z={z:.2f}, p={p:.4f}", flush=True)

    rec = {
        "corpus": args.corpus, "lang": args.lang, "variant": args.variant,
        "is_source": is_source, "detector": "vanilla_levenshtein",
        "phon_threshold": args.phon_threshold, "filter_pct": args.filter_pct,
        "n_units_used": len(usable_ids), "n_blocked": len(blocked),
        "diagnostic": {
            "n_boundaries": n_bound,
            "phonological_total": n_phon, "etymological_total": n_etym,
            "semantic_total": n_sem,
        },
        "results": results, "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
    print(f"  wrote {out_path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
