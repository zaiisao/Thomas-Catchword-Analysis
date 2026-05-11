#!/usr/bin/env python3
"""
Phonological-only permutation test — one (corpus, lang, variant) at a time.

Reuses each corpus's existing tokenizer + translator output, builds the
catchword matrix once (which already stores link_type per pair), then runs
permutation tests under three filter settings:

  - all   : every link_type (the original test)
  - phon  : link_type in {'phonological', 'etymological'}  (language-specific)
  - sem   : link_type in {'semantic'}                       (thematic)

Output: data/phon_only/{corpus}_{lang}_v{variant}.json
        containing per-filter true/null/z/p, plus per-boundary diagnostic
        counts (phonological vs semantic per boundary).

Usage:
  python scripts/phon_only_one.py --corpus proverbs --lang hebrew --variant 0
  python scripts/phon_only_one.py --corpus q        --lang aramaic --variant 5
  python scripts/phon_only_one.py --corpus thomas   --lang syriac --variant 0

  --n-perms N  : permutation count (default 10000)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

# These modules carry the per-corpus tokenizer + lemma logic. Re-using them
# guarantees the matrix is identical to the original test.
from scripts.proverbs_permutation_test import (  # noqa: E402
    make_tokens as _prov_make_tokens,
    HEB_FILE as _PROV_HEB_FILE,
    TRANS_DIR as _PROV_TRANS_DIR,
    PHON_THRESHOLD, FILTER_PCT,
)
from scripts.crossling_permutation_test import (  # noqa: E402
    make_tokens as _thom_make_tokens,
    load_sedra_lookup as _thom_sedra_lookup,
    LLM_DIR_SYR as _THOM_LLM_DIR_SYR,
    CROSS_DIR as _THOM_CROSS_DIR,
    N_LOGIA as _THOM_N_LOGIA,
)
from scripts.q_permutation_test import (  # noqa: E402
    make_tokens as _q_make_tokens,
)

_Q_GREEK_FILE = REPO_ROOT / "data" / "q_source" / "q_pericopes_greek.json"
_Q_TRANS_DIR = REPO_ROOT / "data" / "q_source" / "translations"

OUT_DIR = REPO_ROOT / "data" / "phon_only"


# ---------------------------------------------------------------------------
# Corpus-specific loaders that all return {unit_id: tokens} and an ordered
# id list (true order).
# ---------------------------------------------------------------------------

def load_proverbs(lang: str, variant_idx: int):
    hdata = json.loads(_PROV_HEB_FILE.read_text(encoding="utf-8"))
    ids = [r["unit_id"] for r in hdata if r.get("hebrew_text")]
    if lang == "hebrew":
        toks = {r["unit_id"]: _prov_make_tokens(r["hebrew_text"], "hebrew")
                 for r in hdata if r.get("hebrew_text")}
        return ids, toks
    lang_dir = _PROV_TRANS_DIR / lang
    toks = {}
    for p in sorted(lang_dir.glob("unit_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        uid = d.get("unit_id")
        variants = d.get("variants", [])
        if uid is None or variant_idx >= len(variants):
            continue
        v = variants[variant_idx]
        if v.get("success"):
            toks[uid] = _prov_make_tokens(v.get("text", ""), lang)
    return ids, toks


def load_q(lang: str, variant_idx: int):
    gdata = json.loads(_Q_GREEK_FILE.read_text(encoding="utf-8"))
    ids = [r["pericope_id"] for r in gdata if r.get("greek_text")]
    if lang == "greek":
        toks = {r["pericope_id"]: _q_make_tokens(r["greek_text"], "greek")
                 for r in gdata if r.get("greek_text")}
        return ids, toks
    lang_dir = _Q_TRANS_DIR / lang
    toks = {}
    for p in sorted(lang_dir.glob("pericope_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pid = d.get("pericope_id")
        variants = d.get("variants", [])
        if pid is None or variant_idx >= len(variants):
            continue
        v = variants[variant_idx]
        if v.get("success"):
            toks[pid] = _q_make_tokens(v.get("text", ""), lang)
    return ids, toks


def load_thomas(lang: str, variant_idx: int):
    sedra = _thom_sedra_lookup() if lang == "syriac" else None
    ids = list(range(_THOM_N_LOGIA))
    toks = {}
    if lang == "syriac":
        for i in ids:
            p = _THOM_LLM_DIR_SYR / f"logion_{i:03d}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            variants = d.get("variants", [])
            if variant_idx >= len(variants):
                continue
            v = variants[variant_idx]
            if v.get("success"):
                toks[i] = _thom_make_tokens(v.get("syriac_text", ""),
                                              "syriac", sedra)
        return ids, toks
    lang_dir = _THOM_CROSS_DIR / lang
    for i in ids:
        p = lang_dir / f"logion_{i:03d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        variants = d.get("variants", [])
        if variant_idx >= len(variants):
            continue
        v = variants[variant_idx]
        if v.get("success"):
            toks[i] = _thom_make_tokens(v.get("text", ""), lang, None)
    return ids, toks


CORPUS_LOADERS = {
    "proverbs": load_proverbs,
    "q":        load_q,
    "thomas":   load_thomas,
}

# Allowed (corpus, lang) combinations. (Greek for Proverbs is a translation
# target — Hebrew is source. Greek for Q IS the source. Syriac for Thomas
# is the source-equivalent — the analysed Phase-2B Gemini language.)
CORPUS_LANGS = {
    "proverbs": ["hebrew", "greek", "syriac", "aramaic", "arabic"],
    "q":        ["greek", "hebrew", "syriac", "aramaic", "arabic"],
    "thomas":   ["syriac", "hebrew", "greek", "arabic"],
}
CORPUS_SOURCE = {"proverbs": "hebrew", "q": "greek", "thomas": "syriac"}


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_blocked(translations, filter_pct):
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix(translations, blocked, lang, ids):
    det = CatchwordDetector(lang, phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, [])
                     if t["lemma"] not in blocked]
                for i in ids}
    matrix: dict[tuple[int, int], frozenset] = {}
    t0 = time.time()
    for n_done, i in enumerate(ids):
        for j in ids:
            if i == j:
                continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = frozenset()
                continue
            cws = det.detect(ta, tb)
            keys = []
            for cw in cws:
                la = cw.token_a["lemma"]; lb = cw.token_b["lemma"]
                if la <= lb:
                    keys.append((la, lb, cw.link_type))
                else:
                    keys.append((lb, la, cw.link_type))
            matrix[(i, j)] = frozenset(keys)
        if (n_done + 1) % max(1, len(ids) // 6) == 0:
            print(f"    row {n_done+1}/{len(ids)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  matrix done {time.time()-t0:.0f}s", flush=True)
    return matrix


FILTERS = {
    "all":  None,
    "phon": frozenset({"phonological", "etymological"}),
    "sem":  frozenset({"semantic"}),
}


def filter_matrix(matrix, link_filter):
    if link_filter is None:
        return matrix
    out = {}
    for k, cell in matrix.items():
        if not cell:
            out[k] = cell
            continue
        out[k] = frozenset(key for key in cell if key[2] in link_filter)
    return out


def stats_for_order(order, matrix, min_freq=2):
    pair_locations: defaultdict = defaultdict(list)
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), frozenset())
        for k in cell:
            pair_locations[k].append(pos)
    rec = sum(1 for locs in pair_locations.values() if len(locs) >= min_freq)
    return rec


def run_perm(matrix, ids, n_perms, seed):
    import random as _r
    rng = _r.Random(seed)
    base = list(ids)
    out = np.empty(n_perms, dtype=np.int32)
    for p in range(n_perms):
        sh = base.copy(); rng.shuffle(sh)
        out[p] = stats_for_order(sh, matrix, 2)
    return out


def boundary_diagnostic(matrix, ids):
    """Per-adjacent-boundary counts of phon vs sem catchwords (true order)."""
    n_phon = n_sem = n_etym = 0
    n_bound = 0
    for pos in range(len(ids) - 1):
        cell = matrix.get((ids[pos], ids[pos + 1]), frozenset())
        n_bound += 1
        for k in cell:
            lt = k[2]
            if lt == "phonological": n_phon += 1
            elif lt == "etymological": n_etym += 1
            elif lt == "semantic": n_sem += 1
    return {"n_boundaries": n_bound,
            "phonological_total": n_phon,
            "etymological_total": n_etym,
            "semantic_total": n_sem,
            "per_boundary_phon_plus_etym":
                (n_phon + n_etym) / max(n_bound, 1),
            "per_boundary_semantic": n_sem / max(n_bound, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPUS_LOADERS))
    ap.add_argument("--lang", required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.corpus}_{args.lang}_v{args.variant}.json"
    if out_path.exists() and not args.force:
        print(f"already exists: {out_path}"); return

    if args.lang not in CORPUS_LANGS[args.corpus]:
        raise SystemExit(f"{args.lang!r} not valid for corpus {args.corpus!r}; "
                         f"choices: {CORPUS_LANGS[args.corpus]}")
    # Source-language only has 1 variant (variant 0 is the source text).
    is_source = args.lang == CORPUS_SOURCE[args.corpus]
    if is_source and args.variant > 0:
        out_path.write_text(json.dumps({
            "corpus": args.corpus, "lang": args.lang, "variant": args.variant,
            "skipped": True, "reason": "source language has only 1 variant"
        }, indent=2))
        return

    t0 = time.time()
    print(f"[{args.corpus}/{args.lang} v{args.variant}] loading…", flush=True)
    ids, toks = CORPUS_LOADERS[args.corpus](args.lang, args.variant)
    usable_ids = [i for i in ids if toks.get(i)]
    if len(usable_ids) < len(ids) * 0.5:
        out_path.write_text(json.dumps({
            "corpus": args.corpus, "lang": args.lang, "variant": args.variant,
            "skipped": True, "n_loaded": len(usable_ids), "n_total": len(ids),
        }, indent=2))
        return
    blocked = compute_blocked(toks, FILTER_PCT)
    print(f"[{args.corpus}/{args.lang} v{args.variant}] matrix "
          f"({len(usable_ids)} units, {len(blocked)} blocked)…", flush=True)
    matrix = precompute_matrix(toks, blocked, args.lang, usable_ids)

    diag = boundary_diagnostic(matrix, usable_ids)
    print(f"  per-boundary: phon+etym={diag['per_boundary_phon_plus_etym']:.2f}, "
          f"sem={diag['per_boundary_semantic']:.2f}", flush=True)

    results = {}
    for fname, link_filter in FILTERS.items():
        fm = filter_matrix(matrix, link_filter)
        true_rec = stats_for_order(usable_ids, fm, 2)
        null = run_perm(fm, usable_ids, args.n_perms, args.seed)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_rec - nm) / ns if ns > 0 else 0.0
        p = float((null >= true_rec).mean())
        results[fname] = {
            "true_recurring_2plus": int(true_rec),
            "null_mean": nm, "null_std": ns,
            "z_score": float(z), "p_value": p,
            "n_perms": args.n_perms,
        }
        print(f"  [{fname:4s}] true={true_rec:>5d}, "
              f"null={nm:.1f}±{ns:.1f}, z={z:.2f}, p={p:.4f}", flush=True)

    rec = {
        "corpus": args.corpus,
        "lang": args.lang,
        "variant": args.variant,
        "is_source": is_source,
        "n_units_used": len(usable_ids),
        "n_blocked": len(blocked),
        "diagnostic": diag,
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
    print(f"[{args.corpus}/{args.lang} v{args.variant}] wrote "
          f"{out_path.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
