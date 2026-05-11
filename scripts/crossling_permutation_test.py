#!/usr/bin/env python3
"""
Cross-linguistic permutation test on recurring catchword patterns.

For each of {syriac, hebrew, arabic, greek}:
  1. Load Gemini variant 0 translations.
  2. Tokenize, strip vocalization → consonantal-skeleton lemmas.
  3. Compute blocked lemmas (filter_pct=80).
  4. Build the 115×115 catchword-pair matrix using the same detector +
     threshold (0.65) as the project's other phases — only the
     LanguageProfile changes.
  5. Compute true-order recurring-pair counts (≥2, ≥3 boundaries) + max-freq.
  6. Run 10,000 random shuffles, derive p-values.
  7. (Optional) variant robustness across all 10 variants.

Output:
  data/processed/crossling_permutation_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LLM_DIR_SYR = REPO_ROOT / "data" / "processed" / "llm_translations"
CROSS_DIR   = REPO_ROOT / "data" / "processed" / "crossling_translations"
SEDRA = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
OUT = REPO_ROOT / "data" / "processed" / "crossling_permutation_results.json"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
N_LOGIA = 115
N_PERMUTATIONS = 10000
N_PERMUTATIONS_VARIANT = 1000
SEED = 42

# ---------- Per-language script regexes & punctuation ----------
SCRIPT_RE = {
    "syriac": re.compile(r"[܀-ݏ]"),
    "hebrew": re.compile(r"[֐-׿]"),
    "arabic": re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
    "greek":  re.compile(r"[Ͱ-Ͽἀ-῿]"),
}
# Per-language punctuation/vocalization to strip.
PUNCT_RE = {
    "syriac": re.compile(r"[܀-܏]"),
    "hebrew": re.compile("[" + "־" + "׀" + "׃" + "׆" + "׳" + "״" + "]"),
    "arabic": re.compile("[" + "،" + "؛" + "؟" + "۔" + "٪" + "٫" + "٬" + "٭" + "]"),
    "greek":  re.compile(r"[ʹ͵;·;.,·]"),
}


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def tokenize(text: str, lang: str) -> list[str]:
    """Strip vocalization + non-target-script + punctuation, split."""
    rx = SCRIPT_RE[lang]
    cleaned_chars = []
    for ch in text:
        if rx.match(ch) or ch.isspace():
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = strip_voc(cleaned)
    cleaned = PUNCT_RE[lang].sub(" ", cleaned)
    if lang == "greek":
        cleaned = cleaned.lower()
    return [w for w in cleaned.split() if w]


def load_sedra_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            unp, lem = parts[1].strip(), parts[3].strip()
            if unp and lem and unp not in out:
                out[unp] = lem
    return out


def make_tokens(text: str, lang: str, sedra: dict[str, str] | None) -> list[dict]:
    """Build {form, lemma, parse} tokens. For Syriac we look lemma up in
    SEDRA (matches the Phase 2B pipeline). For the other languages we lack
    a comparable lemma resource — use the consonantal-skeleton form as the
    lemma proxy. This is the same trick Phase 3 uses for patristic Syriac
    where SEDRA misses, and is the right granularity for cross-language
    catchword detection (catchwords work at the consonant-skeleton level)."""
    out = []
    for tok in tokenize(text, lang):
        if lang == "syriac" and sedra is not None:
            lemma = sedra.get(tok, tok)
        else:
            # Skeleton = vocalization-stripped form (already done in tokenize)
            lemma = tok
        out.append({"form": tok, "lemma": lemma, "parse": "MS-EMP"})
    return out


def load_translations(lang: str, variant_idx: int = 0) -> dict[int, list[dict]]:
    """Load variant_idx for each Thomas logion in the given language."""
    sedra = load_sedra_lookup() if lang == "syriac" else None
    out: dict[int, list[dict]] = {}

    if lang == "syriac":
        # Phase 2B Gemini Syriac
        for i in range(N_LOGIA):
            path = LLM_DIR_SYR / f"logion_{i:03d}.json"
            if not path.exists():
                out[i] = []
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            variants = d.get("variants", [])
            if variant_idx < len(variants) and variants[variant_idx].get("success"):
                out[i] = make_tokens(variants[variant_idx]["syriac_text"],
                                       "syriac", sedra)
            else:
                out[i] = []
        return out

    # Cross-lingual translations (hebrew/arabic/greek)
    lang_dir = CROSS_DIR / lang
    for i in range(N_LOGIA):
        path = lang_dir / f"logion_{i:03d}.json"
        if not path.exists():
            out[i] = []
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        variants = d.get("variants", [])
        if variant_idx < len(variants) and variants[variant_idx].get("success"):
            text = variants[variant_idx].get("text", "")
            out[i] = make_tokens(text, lang, None)
        else:
            out[i] = []
    return out


def compute_blocked(translations: dict[int, list[dict]],
                     filter_pct: float) -> set[str]:
    n = len(translations)
    cutoff = filter_pct * n / 100.0
    cnt: Counter[str] = Counter()
    for toks in translations.values():
        for lem in {t["lemma"] for t in toks}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def precompute_matrix(translations: dict[int, list[dict]],
                       blocked: set[str],
                       lang: str) -> dict[tuple[int, int], frozenset]:
    det = CatchwordDetector(lang,
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    filtered = {i: [t for t in translations.get(i, []) if t["lemma"] not in blocked]
                for i in range(N_LOGIA)}
    matrix: dict[tuple[int, int], frozenset] = {}
    print(f"  Precomputing {N_LOGIA*(N_LOGIA-1)} adjacency cells…")
    t0 = time.time()
    for i in range(N_LOGIA):
        for j in range(N_LOGIA):
            if i == j:
                continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = frozenset()
                continue
            cws = det.detect(ta, tb)
            keys = []
            for cw in cws:
                la = cw.token_a["lemma"]
                lb = cw.token_b["lemma"]
                if la <= lb:
                    keys.append((la, lb, cw.link_type))
                else:
                    keys.append((lb, la, cw.link_type))
            matrix[(i, j)] = frozenset(keys)
        if (i + 1) % 25 == 0:
            print(f"    row {i+1}/{N_LOGIA} ({time.time()-t0:.0f}s)")
    print(f"  Matrix built in {time.time()-t0:.0f}s")
    return matrix


def stats_for_order(order: list[int],
                     matrix: dict[tuple[int, int], frozenset],
                     min_freqs: list[int]) -> dict:
    pair_locations: defaultdict[tuple, list[int]] = defaultdict(list)
    for pos in range(len(order) - 1):
        cell = matrix.get((order[pos], order[pos + 1]), frozenset())
        for k in cell:
            pair_locations[k].append(pos)
    out = {f"recurring_{f}plus": sum(1 for locs in pair_locations.values()
                                       if len(locs) >= f)
            for f in min_freqs}
    out["max_freq"] = max((len(locs) for locs in pair_locations.values()),
                           default=0)
    out["pair_locations"] = pair_locations
    return out


def run_permutation(matrix, n_perms, seed, min_freqs=(2, 3)):
    rng = random.Random(seed)
    base = list(range(N_LOGIA))
    null = {f"recurring_{f}plus": [] for f in min_freqs}
    null["max_freq"] = []
    t0 = time.time()
    for p in range(n_perms):
        sh = base.copy()
        rng.shuffle(sh)
        s = stats_for_order(sh, matrix, list(min_freqs))
        for k in null:
            null[k].append(s[k])
        if (p + 1) % 2000 == 0:
            print(f"      perm {p+1}/{n_perms} ({time.time()-t0:.0f}s)")
    return {k: np.array(v) for k, v in null.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="syriac,hebrew,arabic,greek",
                     help="Comma-separated subset")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--variants-test", action="store_true")
    args = ap.parse_args()

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    print(f"=== Cross-linguistic permutation test ===")
    print(f"  languages: {langs}")
    print(f"  N_PERMUTATIONS={args.n_perms}, threshold={PHON_THRESHOLD}, "
          f"filter_pct={FILTER_PCT}")
    print()

    summary: dict[str, dict] = {}

    for lang in langs:
        print(f"--- {lang.upper()} ---")
        translations = load_translations(lang, args.variant)
        n_loaded = sum(1 for t in translations.values() if t)
        print(f"  loaded {n_loaded}/{N_LOGIA} logia")
        if n_loaded < N_LOGIA * 0.8:
            print(f"  SKIP — too few translations available for {lang}")
            summary[lang] = {"skipped": True, "n_loaded": n_loaded}
            continue
        blocked = compute_blocked(translations, FILTER_PCT)
        print(f"  blocked {len(blocked)} lemmas (top frequent): "
              f"{sorted(blocked)[:6]}")

        matrix = precompute_matrix(translations, blocked, lang)
        nonempty = sum(1 for v in matrix.values() if v)
        avg = np.mean([len(v) for v in matrix.values() if v]) if nonempty else 0
        print(f"  {nonempty}/{len(matrix)} non-empty cells, avg {avg:.1f} pairs/cell")

        true_order = list(range(N_LOGIA))
        s_true = stats_for_order(true_order, matrix, [2, 3, 4])
        print(f"  true order: rec≥2={s_true['recurring_2plus']}, "
              f"rec≥3={s_true['recurring_3plus']}, "
              f"rec≥4={s_true['recurring_4plus']}, "
              f"max_freq={s_true['max_freq']}")

        print(f"  running {args.n_perms} permutations…")
        null = run_permutation(matrix, args.n_perms, args.seed,
                                 min_freqs=(2, 3, 4))
        p_2 = float((null["recurring_2plus"] >= s_true["recurring_2plus"]).mean())
        p_3 = float((null["recurring_3plus"] >= s_true["recurring_3plus"]).mean())
        p_4 = float((null["recurring_4plus"] >= s_true["recurring_4plus"]).mean())
        p_max = float((null["max_freq"] >= s_true["max_freq"]).mean())

        eff_2 = ((s_true["recurring_2plus"] - null["recurring_2plus"].mean())
                 / max(null["recurring_2plus"].std(), 1e-9))

        print(f"  rec≥2: true={s_true['recurring_2plus']}, "
              f"null={null['recurring_2plus'].mean():.1f}±"
              f"{null['recurring_2plus'].std():.1f}, "
              f"z={eff_2:.2f}, p={p_2:.4f}")
        print(f"  rec≥3: true={s_true['recurring_3plus']}, "
              f"null={null['recurring_3plus'].mean():.1f}±"
              f"{null['recurring_3plus'].std():.1f}, p={p_3:.4f}")

        sorted_pairs = sorted(s_true["pair_locations"].items(),
                               key=lambda x: -len(x[1]))
        summary[lang] = {
            "skipped": False,
            "n_loaded": n_loaded,
            "n_blocked": len(blocked),
            "matrix_nonempty_cells": nonempty,
            "matrix_avg_pairs_per_cell": float(avg),
            "true_recurring_2plus": int(s_true["recurring_2plus"]),
            "true_recurring_3plus": int(s_true["recurring_3plus"]),
            "true_recurring_4plus": int(s_true["recurring_4plus"]),
            "true_max_freq": int(s_true["max_freq"]),
            "null_mean_2plus": float(null["recurring_2plus"].mean()),
            "null_std_2plus":  float(null["recurring_2plus"].std()),
            "null_mean_3plus": float(null["recurring_3plus"].mean()),
            "null_std_3plus":  float(null["recurring_3plus"].std()),
            "null_mean_max":   float(null["max_freq"].mean()),
            "null_std_max":    float(null["max_freq"].std()),
            "p_2plus": p_2,
            "p_3plus": p_3,
            "p_4plus": p_4,
            "p_max":   p_max,
            "z_2plus": float(eff_2),
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

    # Variant robustness — only re-runs languages that already passed the
    # main test at p < 0.10.
    var_results: dict[str, list] = {}
    if args.variants_test:
        for lang in langs:
            s = summary.get(lang)
            if not s or s.get("skipped") or s["p_2plus"] > 0.10:
                continue
            print(f"--- variant robustness: {lang} ---")
            per_v = []
            for v in range(10):
                t = load_translations(lang, v)
                n_v = sum(1 for x in t.values() if x)
                if n_v < N_LOGIA * 0.5:
                    continue
                blk = compute_blocked(t, FILTER_PCT)
                mat = precompute_matrix(t, blk, lang)
                st  = stats_for_order(list(range(N_LOGIA)), mat, [2, 3])
                nl  = run_permutation(mat, N_PERMUTATIONS_VARIANT, args.seed,
                                        min_freqs=(2, 3))
                pv2 = float((nl["recurring_2plus"]
                              >= st["recurring_2plus"]).mean())
                per_v.append({"variant": v,
                               "n_loaded": n_v,
                               "true_recurring_2plus": int(st["recurring_2plus"]),
                               "p_2plus": pv2,
                               "null_mean_2plus": float(nl["recurring_2plus"].mean()),
                               "null_std_2plus":  float(nl["recurring_2plus"].std())})
                print(f"  variant {v}: rec≥2={st['recurring_2plus']}, p={pv2:.4f}")
            var_results[lang] = per_v

    # ---- Write ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": {"phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT,
                    "n_logia": N_LOGIA, "n_permutations": args.n_perms,
                    "seed": args.seed, "variant_idx": args.variant},
        "per_language": summary,
        "variant_robustness": var_results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT}")

    # Headline table
    print()
    print("=" * 76)
    print(f"{'Language':10s} {'True (≥2)':>10s} {'Null mean':>10s} {'std':>6s} "
          f"{'z':>6s} {'p':>10s}")
    print("-" * 76)
    for lang in langs:
        s = summary.get(lang)
        if not s or s.get("skipped"):
            print(f"{lang:10s} (skipped)")
            continue
        print(f"{lang:10s} {s['true_recurring_2plus']:>10d} "
              f"{s['null_mean_2plus']:>10.1f} "
              f"{s['null_std_2plus']:>6.1f} "
              f"{s['z_2plus']:>6.2f} "
              f"{s['p_2plus']:>10.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
