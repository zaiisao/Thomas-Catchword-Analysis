#!/usr/bin/env python3
"""
Cross-linguistic permutation test on Q (sayings source behind Matt/Luke).

For each language in {greek, aramaic, syriac, hebrew}:
  - Load variant `--variant` translations of each Q pericope.
    (For Greek, source text directly; for the other three, Gemini retroversion.)
  - Tokenise → consonantal-skeleton lemmas.
  - Build N×N pericope-pair catchword matrix.
  - True-order vs 10,000 shuffles. Recurring pairs at ≥2 and ≥3 boundaries.

Same detector calibration as Thomas (threshold=0.65, filter_pct=80).
Output:
  data/q_source/permutation/main_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

GREEK_FILE = REPO_ROOT / "data" / "q_source" / "q_pericopes_greek.json"
TRANS_DIR  = REPO_ROOT / "data" / "q_source" / "translations"
OUT        = REPO_ROOT / "data" / "q_source" / "permutation" / "main_results.json"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
N_PERMUTATIONS = 10000
SEED = 42

SCRIPT_RE = {
    "syriac":  re.compile(r"[܀-ݏ]"),
    "hebrew":  re.compile(r"[֐-׿]"),
    "aramaic": re.compile(r"[֐-׿]"),
    "arabic":  re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
    "greek":   re.compile(r"[Ͱ-Ͽἀ-῿]"),
}
PUNCT_RE = {
    "syriac":  re.compile(r"[܀-܏]"),
    "hebrew":  re.compile("[" + "־" + "׀" + "׃" + "׆" + "׳" + "״" + "]"),
    "aramaic": re.compile("[" + "־" + "׀" + "׃" + "׆" + "׳" + "״" + "]"),
    "arabic":  re.compile("[" + "،" + "؛" + "؟" + "۔" + "٪" + "٫" + "٬" + "٭" + "]"),
    "greek":   re.compile(r"[ʹ͵;·;.,·]"),
}


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def tokenize(text: str, lang: str) -> list[str]:
    rx = SCRIPT_RE[lang]
    cleaned = "".join(ch for ch in text if rx.match(ch) or ch.isspace())
    cleaned = strip_voc(cleaned)
    cleaned = PUNCT_RE[lang].sub(" ", cleaned)
    if lang == "greek":
        cleaned = cleaned.lower()
    return [w for w in cleaned.split() if w]


def make_tokens(text: str, lang: str) -> list[dict]:
    return [{"form": t, "lemma": t, "parse": "MS-EMP"}
            for t in tokenize(text, lang)]


def load_q_translations(lang: str, variant_idx: int = 0) -> dict[int, list[dict]]:
    """Return {pericope_id: tokens}."""
    out: dict[int, list[dict]] = {}
    if lang == "greek":
        gdata = json.loads(GREEK_FILE.read_text(encoding="utf-8"))
        for r in gdata:
            if r.get("greek_text"):
                out[r["pericope_id"]] = make_tokens(r["greek_text"], "greek")
        return out

    lang_dir = TRANS_DIR / lang
    if not lang_dir.exists():
        return out
    for path in sorted(lang_dir.glob("pericope_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        pid = d.get("pericope_id")
        if pid is None:
            continue
        variants = d.get("variants", [])
        if variant_idx < len(variants) and variants[variant_idx].get("success"):
            text = variants[variant_idx].get("text", "")
            out[pid] = make_tokens(text, lang)
    return out


def compute_blocked(translations, filter_pct):
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter[str] = Counter()
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
    for i in ids:
        for j in ids:
            if i == j: continue
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
    return matrix


def stats_for_order(order, matrix, min_freqs):
    pair_locations: defaultdict[tuple, list[int]] = defaultdict(list)
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), frozenset())
        for k in cell:
            pair_locations[k].append(pos)
    return {
        **{f"recurring_{f}plus":
            sum(1 for locs in pair_locations.values() if len(locs) >= f)
           for f in min_freqs},
        "max_freq": max((len(v) for v in pair_locations.values()), default=0),
        "pair_locations": pair_locations,
    }


def run_permutation(matrix, ids, n_perms, seed, min_freqs):
    import random as _rand
    rng = _rand.Random(seed)
    null = {f"recurring_{f}plus": [] for f in min_freqs}
    null["max_freq"] = []
    base = list(ids)
    for _ in range(n_perms):
        sh = base.copy()
        rng.shuffle(sh)
        s = stats_for_order(sh, matrix, list(min_freqs))
        for k in null:
            null[k].append(s[k])
    return {k: np.array(v) for k, v in null.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="greek,aramaic,syriac,hebrew,arabic")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args()

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    summary: dict[str, dict] = {}

    # Pericope id list (shared baseline). Use the Greek file's ordering.
    gdata = json.loads(GREEK_FILE.read_text(encoding="utf-8"))
    ids_master = [r["pericope_id"] for r in gdata if r.get("greek_text")]
    print(f"=== Q permutation test ===")
    print(f"  Q pericopes (with Greek text): {len(ids_master)}")
    print(f"  languages: {langs}")
    print(f"  N_PERMUTATIONS: {args.n_perms}")
    print()

    for lang in langs:
        print(f"--- {lang.upper()} ---")
        translations = load_q_translations(lang, args.variant)
        n_loaded = sum(1 for v in translations.values() if v)
        usable_ids = [i for i in ids_master if translations.get(i)]
        print(f"  loaded {n_loaded}/{len(ids_master)} pericopes")
        if len(usable_ids) < len(ids_master) * 0.8:
            print(f"  SKIP — too few translations")
            summary[lang] = {"skipped": True, "n_loaded": n_loaded}
            continue
        blocked = compute_blocked(translations, FILTER_PCT)
        print(f"  blocked {len(blocked)} lemmas (top freq): "
              f"{sorted(blocked)[:6]}")
        matrix = precompute_matrix(translations, blocked, lang, usable_ids)
        nonempty = sum(1 for v in matrix.values() if v)
        avg = float(np.mean([len(v) for v in matrix.values() if v])) if nonempty else 0
        print(f"  {nonempty}/{len(matrix)} non-empty cells, "
              f"avg {avg:.1f} pairs/cell")
        # True order = ids in pericope_id ascending order (already in usable_ids
        # which we built from ids_master)
        true_order = usable_ids
        s_true = stats_for_order(true_order, matrix, [2, 3])
        print(f"  true: rec≥2={s_true['recurring_2plus']}, "
              f"rec≥3={s_true['recurring_3plus']}, "
              f"max_freq={s_true['max_freq']}")
        null = run_permutation(matrix, usable_ids, args.n_perms, SEED,
                                 min_freqs=(2, 3))
        p_2 = float((null["recurring_2plus"] >= s_true["recurring_2plus"]).mean())
        p_3 = float((null["recurring_3plus"] >= s_true["recurring_3plus"]).mean())
        p_max = float((null["max_freq"] >= s_true["max_freq"]).mean())
        eff_2 = ((s_true["recurring_2plus"] - null["recurring_2plus"].mean())
                 / max(null["recurring_2plus"].std(), 1e-9))
        eff_3 = ((s_true["recurring_3plus"] - null["recurring_3plus"].mean())
                 / max(null["recurring_3plus"].std(), 1e-9))

        print(f"  rec≥2: true={s_true['recurring_2plus']}, "
              f"null={null['recurring_2plus'].mean():.1f}±"
              f"{null['recurring_2plus'].std():.1f}, z={eff_2:.2f}, p={p_2:.4f}")
        print(f"  rec≥3: true={s_true['recurring_3plus']}, "
              f"null={null['recurring_3plus'].mean():.1f}±"
              f"{null['recurring_3plus'].std():.1f}, z={eff_3:.2f}, p={p_3:.4f}")

        sorted_pairs = sorted(s_true["pair_locations"].items(),
                                key=lambda x: -len(x[1]))
        summary[lang] = {
            "skipped": False,
            "n_loaded": n_loaded,
            "n_pericopes_used": len(usable_ids),
            "n_blocked": len(blocked),
            "matrix_nonempty": nonempty,
            "matrix_avg_pairs_per_cell": avg,
            "true_recurring_2plus": int(s_true["recurring_2plus"]),
            "true_recurring_3plus": int(s_true["recurring_3plus"]),
            "true_max_freq":         int(s_true["max_freq"]),
            "null_mean_2plus": float(null["recurring_2plus"].mean()),
            "null_std_2plus":  float(null["recurring_2plus"].std()),
            "null_mean_3plus": float(null["recurring_3plus"].mean()),
            "null_std_3plus":  float(null["recurring_3plus"].std()),
            "null_mean_max":   float(null["max_freq"].mean()),
            "null_std_max":    float(null["max_freq"].std()),
            "p_2plus": p_2,
            "p_3plus": p_3,
            "p_max":   p_max,
            "z_2plus": float(eff_2),
            "z_3plus": float(eff_3),
            "top_pairs": [
                {"lemma_a": k[0], "lemma_b": k[1], "link_type": k[2],
                  "frequency": len(v), "boundaries": v}
                for k, v in sorted_pairs[:15]
            ],
            "_raw_recurring_2plus": null["recurring_2plus"].tolist(),
            "_raw_recurring_3plus": null["recurring_3plus"].tolist(),
            "_raw_max_freq":         null["max_freq"].tolist(),
        }
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": {"phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT,
                    "n_permutations": args.n_perms, "seed": SEED,
                    "variant_idx": args.variant},
        "n_pericopes": len(ids_master),
        "per_language": summary,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")

    # Headline table
    print()
    print("=" * 76)
    print(f"{'Language':10s} {'True (≥2)':>10s} {'Null':>10s} {'std':>6s} "
          f"{'z':>6s} {'p':>10s}")
    print("-" * 76)
    for lang in langs:
        s = summary.get(lang)
        if not s or s.get("skipped"):
            print(f"{lang:10s} skipped"); continue
        print(f"{lang:10s} {s['true_recurring_2plus']:>10d} "
              f"{s['null_mean_2plus']:>10.1f} "
              f"{s['null_std_2plus']:>6.1f} "
              f"{s['z_2plus']:>6.2f} "
              f"{s['p_2plus']:>10.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
