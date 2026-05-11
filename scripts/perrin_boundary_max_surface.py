#!/usr/bin/env python3
"""
Re-run boundary-MAX test with SURFACE-FORM Syriac (no SEDRA collapse).
Fair apples-to-apples comparison with Hebrew/Greek/Arabic.
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
from scripts.phon_only_one import load_thomas as _load_thomas  # noqa: E402
from scripts.crossling_permutation_test import (  # noqa: E402
    tokenize as _syr_tokenize, LLM_DIR_SYR, N_LOGIA,
)

LANGS = ["syriac", "hebrew", "greek", "arabic"]
OUT = REPO_ROOT / "data" / "perrin_direct" / "boundary_max_thomas_surface.json"


def load_syriac_surface() -> dict[int, list[dict]]:
    """Syriac Thomas variant 0 using SURFACE forms (no SEDRA)."""
    out = {}
    for i in range(N_LOGIA):
        p = LLM_DIR_SYR / f"logion_{i:03d}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vs = d.get("variants", [])
        if vs and vs[0].get("success"):
            tokens = _syr_tokenize(vs[0]["syriac_text"], "syriac")
            out[i] = [{"form": t, "lemma": t, "parse": "MS-EMP"} for t in tokens]
    return out


def boundary_max_phon(toks_a, toks_b, lang_profile):
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
            if la == lb: continue
            ca, cb = consonantal(la), consonantal(lb)
            if not ca or not cb or ca == cb: continue
            dist = weighted_levenshtein(ca, cb, lang_profile)
            score = 1.0 - dist / max(len(ca), len(cb))
            if score > best: best = score
    return best


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("loading translations…")
    data = {}
    for lang in LANGS:
        if lang == "syriac":
            toks = load_syriac_surface()
            ids = sorted(toks.keys())
            usable = ids
        else:
            ids_all, toks = _load_thomas(lang, variant_idx=0)
            usable = [i for i in ids_all if toks.get(i)]
        data[lang] = {"toks": toks, "usable": usable,
                       "profile": get_profile(lang)}
        print(f"  {lang}: {len(usable)} logia")

    common_ids = set(data[LANGS[0]]["usable"])
    for lang in LANGS[1:]:
        common_ids &= set(data[lang]["usable"])
    common_ids = sorted(common_ids)
    n = len(common_ids)
    print(f"common logia: {n}")

    print("computing all-pairs MAX phon-scores…")
    t0 = time.time()
    max_scores = {lang: {} for lang in LANGS}
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

    true_boundaries = [(common_ids[k], common_ids[k+1]) for k in range(n-1)]
    true_max = {lang: [max_scores[lang][b] for b in true_boundaries]
                 for lang in LANGS}

    print()
    print("MEAN MAX phon-score at TRUE adjacent boundaries (Syriac = SURFACE FORMS):")
    for lang in LANGS:
        vals = np.array(true_max[lang])
        print(f"  {lang:<8}: mean={vals.mean():.3f}  "
              f"N>=0.6: {np.sum(vals >= 0.6)}/{len(vals)} "
              f"({np.mean(vals >= 0.6):.1%})")

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
    print(f"\nPermutation test on 'Syriac wins MAX at boundary':")
    print(f"  TRUE: {syr_win_true}, NULL mean={null_syr_wins.mean():.1f}±{null_syr_wins.std():.1f}")
    print(f"  z={z:.2f}, p={p:.4f}")

    summary = {
        "n_common_logia": n,
        "tokenization": "syriac=surface_forms, others=surface_forms",
        "mean_max_phon_at_true_boundaries": {
            lang: float(np.mean(true_max[lang])) for lang in LANGS},
        "winners_per_lang": winner_count,
        "winners_pct_per_lang": {l: winner_count[l]/len(winners) for l in LANGS},
        "syriac_wins_test": {
            "true_syriac_wins": int(syr_win_true),
            "null_mean": float(null_syr_wins.mean()),
            "null_std": float(null_syr_wins.std()),
            "z_score": float(z),
            "p_value": p,
        },
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
