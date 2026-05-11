#!/usr/bin/env python3
"""
Perrin-direct test — total-count statistic (no recurrence filter).

For each (corpus, lang, variant):
  - Build the catchword matrix (with link_type per cell).
  - Statistic = TOTAL count of catchwords summed over all adjacent boundaries,
                NOT "pairs that recur at ≥2 boundaries".
  - Run permutation test with 3 filters: all / phon+etym / semantic.
  - Also try threshold/blocking sweeps.

This directly tests Perrin's claim: "specific catchwords at specific
boundaries", not "pairs recurring across multiple boundaries".

Output: data/perrin_direct/{corpus}_{lang}_v{variant}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

# Reuse the existing loaders from phon_only_one.py — they handle all 3
# corpora's filename conventions.
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


def compute_blocked(translations, filter_pct):
    if filter_pct >= 100:
        return set()
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix_full(translations, blocked, lang, ids, phon_threshold):
    """Same matrix as before but store FULL list of catchwords (with repeats
    by lemma-pair already deduped by the detector's seen-dict)."""
    det = CatchwordDetector(lang, phonological_threshold=phon_threshold,
                              require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in ids}
    matrix: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
    t0 = time.time()
    for n_done, i in enumerate(ids):
        for j in ids:
            if i == j:
                continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = []
                continue
            cws = det.detect(ta, tb)
            matrix[(i, j)] = [(c.token_a["lemma"], c.token_b["lemma"],
                                c.link_type) for c in cws]
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
    ap.add_argument("--tag", default="",
                     help="extra tag for output filename (e.g., 'thr05_noblock')")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_part = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"{args.corpus}_{args.lang}_v{args.variant}{tag_part}.json"
    if out_path.exists() and not args.force:
        print(f"already exists: {out_path}"); return

    is_source = args.lang == CORPUS_SOURCE[args.corpus]
    if is_source and args.variant > 0:
        out_path.write_text(json.dumps({
            "corpus": args.corpus, "lang": args.lang, "variant": args.variant,
            "skipped": True, "reason": "source language has only 1 variant"
        }, indent=2))
        return

    t0 = time.time()
    print(f"[{args.corpus}/{args.lang} v{args.variant} thr={args.phon_threshold} "
          f"block={args.filter_pct}] loading…", flush=True)
    ids, toks = CORPUS_LOADERS[args.corpus](args.lang, args.variant)
    usable_ids = [i for i in ids if toks.get(i)]
    if len(usable_ids) < len(ids) * 0.5:
        out_path.write_text(json.dumps({
            "corpus": args.corpus, "lang": args.lang, "variant": args.variant,
            "skipped": True, "n_loaded": len(usable_ids), "n_total": len(ids),
        }, indent=2))
        return
    blocked = compute_blocked(toks, args.filter_pct)
    print(f"  matrix ({len(usable_ids)} units, {len(blocked)} blocked, "
          f"thr={args.phon_threshold})…", flush=True)
    matrix = precompute_matrix_full(toks, blocked, args.lang, usable_ids,
                                       args.phon_threshold)

    # Boundary phon counts at TRUE order — for diagnostic
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
        null = run_perm_total(matrix, usable_ids, link_filter, args.n_perms,
                                args.seed)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_tot - nm) / ns if ns > 0 else 0.0
        p = float((null >= true_tot).mean())
        results[fname] = {
            "true_total": int(true_tot),
            "null_mean": nm, "null_std": ns,
            "z_score": float(z), "p_value": p,
            "n_perms": args.n_perms,
        }
        print(f"  [{fname:4s}] true={true_tot:>5d}, "
              f"null={nm:.1f}±{ns:.1f}, z={z:.2f}, p={p:.4f}", flush=True)

    rec = {
        "corpus": args.corpus,
        "lang": args.lang,
        "variant": args.variant,
        "is_source": is_source,
        "phon_threshold": args.phon_threshold,
        "filter_pct": args.filter_pct,
        "tag": args.tag,
        "n_units_used": len(usable_ids),
        "n_blocked": len(blocked),
        "diagnostic": {
            "n_boundaries": n_bound,
            "phonological_total": n_phon,
            "etymological_total": n_etym,
            "semantic_total": n_sem,
        },
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
    print(f"  wrote {out_path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
