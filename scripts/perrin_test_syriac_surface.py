#!/usr/bin/env python3
"""
Quick fix: run Syriac Thomas test using SURFACE FORMS (no SEDRA lemma collapse)
so it's apples-to-apples with how Hebrew/Greek/Arabic are tokenized.

The vanilla-Lev result of z_phon=0.13 was confounded by SEDRA collapsing 50.7%
of Syriac surface tokens to different lemma strings — variants of a root that
would-be-phonological in other languages get absorbed into "semantic" in
Syriac.

Output: data/perrin_direct/thomas_syriac_v0_surface.json (default detector)
        data/perrin_direct/vanilla_thomas_syriac_v0_surface.json (vanilla Lev)
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402
from scripts.crossling_permutation_test import (  # noqa: E402
    tokenize, LLM_DIR_SYR, N_LOGIA,
)
from scripts.perrin_test_vanilla import (  # noqa: E402
    classify_vanilla, vanilla_levenshtein,
)

OUT_DIR = REPO_ROOT / "data" / "perrin_direct"

PHON_THRESHOLD = 0.6
FILTER_PCT = 80.0
N_PERMS = 10000
SEED = 42

FILTERS = {
    "all":  None,
    "phon": frozenset({"phonological", "etymological"}),
    "sem":  frozenset({"semantic"}),
}


def load_syriac_surface() -> dict[int, list[dict]]:
    """Load Syriac Thomas using SURFACE forms as lemmas (no SEDRA collapse)."""
    out = {}
    for i in range(N_LOGIA):
        p = LLM_DIR_SYR / f"logion_{i:03d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vs = d.get("variants", [])
        if vs and vs[0].get("success"):
            tokens = tokenize(vs[0]["syriac_text"], "syriac")
            out[i] = [{"form": t, "lemma": t, "parse": "MS-EMP"}
                       for t in tokens]
    return out


def compute_blocked(translations, filter_pct):
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix(translations, blocked, ids, mode: str, threshold: float):
    """mode = 'default' (use lang-profile detector) or 'vanilla' (plain Lev)."""
    if mode == "default":
        det = CatchwordDetector("syriac", phonological_threshold=threshold,
                                  require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in ids}
    matrix = {}
    t0 = time.time()
    for n_done, i in enumerate(ids):
        for j in ids:
            if i == j: continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = []
                continue
            if mode == "default":
                cws = det.detect(ta, tb)
                matrix[(i, j)] = [(c.token_a["lemma"], c.token_b["lemma"],
                                    c.link_type) for c in cws]
            else:  # vanilla
                seen = {}
                for tok_a in ta:
                    la = tok_a["lemma"]
                    if not la: continue
                    for tok_b in tb:
                        lb = tok_b["lemma"]
                        if not lb: continue
                        k = (la, lb)
                        if k in seen: continue
                        res = classify_vanilla(la, lb, threshold)
                        if res:
                            seen[k] = res[0]
                matrix[(i, j)] = [(k[0], k[1], lt) for k, lt in seen.items()]
        if (n_done + 1) % 25 == 0:
            print(f"    row {n_done+1}/{len(ids)} ({time.time()-t0:.0f}s)",
                  flush=True)
    return matrix


def total_count(order, matrix, link_filter):
    total = 0
    for k in range(len(order) - 1):
        cell = matrix.get((order[k], order[k+1]), [])
        if link_filter is None:
            total += len(cell)
        else:
            total += sum(1 for c in cell if c[2] in link_filter)
    return total


def run_test(toks, blocked, ids, mode: str, label: str) -> dict:
    print(f"\n=== Syriac SURFACE Thomas, mode={mode} ===")
    print(f"  matrix build…")
    matrix = precompute_matrix(toks, blocked, ids, mode, PHON_THRESHOLD)

    n_phon = n_sem = n_etym = 0
    for k in range(len(ids) - 1):
        for c in matrix.get((ids[k], ids[k+1]), []):
            if c[2] == "phonological": n_phon += 1
            elif c[2] == "etymological": n_etym += 1
            elif c[2] == "semantic": n_sem += 1
    print(f"  TRUE adjacencies: phon={n_phon}, etym={n_etym}, sem={n_sem}")

    results = {}
    rng = random.Random(SEED)
    base = list(ids)
    for fname, lf in FILTERS.items():
        true_t = total_count(base, matrix, lf)
        null = np.empty(N_PERMS, dtype=np.int32)
        for p in range(N_PERMS):
            sh = base.copy(); rng.shuffle(sh)
            null[p] = total_count(sh, matrix, lf)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_t - nm) / ns if ns > 0 else 0.0
        pv = float((null >= true_t).mean())
        results[fname] = {"true_total": int(true_t), "null_mean": nm,
                            "null_std": ns, "z_score": float(z), "p_value": pv}
        print(f"  [{fname:4s}] true={true_t}, null={nm:.1f}±{ns:.1f}, "
              f"z={z:.2f}, p={pv:.4f}")
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Syriac Thomas with SURFACE forms (no SEDRA)...")
    toks = load_syriac_surface()
    ids = sorted(toks.keys())
    print(f"  {len(ids)} logia")
    blocked = compute_blocked(toks, FILTER_PCT)
    print(f"  blocked {len(blocked)} top-20% surface forms")

    # Two tests: default detector + vanilla detector, both on surface forms
    default_results = run_test(toks, blocked, ids, "default", "default")
    vanilla_results = run_test(toks, blocked, ids, "vanilla", "vanilla")

    out1 = OUT_DIR / "thomas_syriac_v0_surface.json"
    out1.write_text(json.dumps({
        "corpus": "thomas", "lang": "syriac", "variant": 0,
        "is_source": True, "tokenization": "surface_forms_no_sedra",
        "detector": "default_syriac_profile",
        "n_units_used": len(ids), "n_blocked": len(blocked),
        "results": default_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    out2 = OUT_DIR / "vanilla_thomas_syriac_v0_surface.json"
    out2.write_text(json.dumps({
        "corpus": "thomas", "lang": "syriac", "variant": 0,
        "is_source": True, "tokenization": "surface_forms_no_sedra",
        "detector": "vanilla_levenshtein",
        "n_units_used": len(ids), "n_blocked": len(blocked),
        "results": vanilla_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out1.name}, {out2.name}")


if __name__ == "__main__":
    main()
