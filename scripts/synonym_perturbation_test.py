#!/usr/bin/env python3
"""
Task 4 — Synonym Perturbation (Fragility) Test.

For each of {Greek, surface-Syriac} variant-0 translations:

  1. Build the variant-0 catchword matrix (same pipeline as Task 2).
  2. Identify the set R_lang of all lemmas appearing in any catchword
     pair at any of the 114 true-order adjacent boundaries.
  3. For every logion that contains >=1 token whose lemma is in R_lang,
     ask Gemini-3-Flash-Preview to rewrite the translation, replacing
     EACH target word with a one-word synonym in the same language,
     keeping the rest of the sentence as close to byte-identical as
     possible.
  4. Save perturbed translations under
       data/processed/perturbation/{lang}/logion_{NNN}.json
     (per-logion, with the targets list + perturbed text).
  5. Re-build the matrix on the perturbed translations and compute:
       - total true-order adjacency count (per link-type)
       - recurring_2plus
     vs. the baseline. Drop = (1 - perturbed/baseline) * 100.
  6. Save the aggregate result to
       data/processed/perturbation_fragility.json.

If a per-logion perturbed file already exists, it is reused (so the
script is resumable — a kill mid-run does not lose the LLM work).

This tests fragility: a catchword that is "deeply embedded" should
typically have only a few valid synonyms, and the substitution may
collapse the link OR re-create it via a different surface form (the
underlying paronomastic motif persists). A purely accidental collision
should disappear entirely under the rewrite.

Cost: ~230 Gemini calls total. ~$0.02.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.crossling_permutation_test import (  # noqa: E402
    N_LOGIA, FILTER_PCT, PHON_THRESHOLD,
    load_translations, compute_blocked, precompute_matrix, tokenize,
    LLM_DIR_SYR, CROSS_DIR,
)
from scripts.crossling_syriac_surface import (  # noqa: E402
    load_syriac_surface,
)
from scripts.crossling_translate import (  # noqa: E402
    load_env_file, ENV_FILE, translate_with_retry,
    MAX_OUTPUT_TOKENS, TEMPERATURE,
)

OUT_DIR_LOGIA = REPO_ROOT / "data" / "processed" / "perturbation"
OUT_PATH = REPO_ROOT / "data" / "processed" / "perturbation_fragility.json"

MODEL = "gemini-3-flash-preview"
SLEEP_BETWEEN_CALLS = 1.5

LANG_HUMAN = {"greek": "Koine Greek", "syriac": "Classical Syriac"}

# Prompt template asks for a minimal-edit synonym substitution.
PERTURB_SYSTEM = (
    "You are a careful linguist performing controlled lexical substitution. "
    "Given a {lang_name} sentence and a list of target words appearing in "
    "it, rewrite the sentence so that each target word is replaced by a "
    "valid one-word {lang_name} synonym (same part of speech, same "
    "register, same approximate sense). Keep EVERY other word EXACTLY "
    "as it was, in the SAME ORDER. Do not paraphrase, add, drop, or "
    "reorder anything beyond the substituted words themselves. Output "
    "ONLY the rewritten sentence in {lang_name} script — no preface, no "
    "commentary, no transliteration."
)


def load_variant0_text(lang: str) -> dict[int, str]:
    """Return the variant-0 translation TEXT (raw) for each logion."""
    out: dict[int, str] = {}
    if lang == "syriac":
        for i in range(N_LOGIA):
            p = LLM_DIR_SYR / f"logion_{i:03d}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            vs = d.get("variants", [])
            if vs and vs[0].get("success"):
                out[i] = vs[0]["syriac_text"]
        return out
    for i in range(N_LOGIA):
        p = CROSS_DIR / lang / f"logion_{i:03d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vs = d.get("variants", [])
        if vs and vs[0].get("success"):
            out[i] = vs[0].get("text", "")
    return out


def load_tokens(lang: str) -> dict[int, list[dict]]:
    """Load variant-0 tokens with the correct tokenization per language."""
    if lang == "syriac":
        return load_syriac_surface(variant_idx=0)
    return load_translations(lang, variant_idx=0)


def collect_target_lemmas(matrix) -> set[str]:
    """All distinct lemma strings appearing in adjacent-boundary catchword
    pairs under the TRUE Thomas order."""
    out: set[str] = set()
    for k in range(N_LOGIA - 1):
        for (la, lb, lt) in matrix.get((k, k + 1), frozenset()):
            out.add(la)
            out.add(lb)
    return out


def find_target_forms(tokens: list[dict], targets: set[str]) -> list[str]:
    """Return surface forms in this logion whose lemma is a target.

    De-duplicates while preserving first-appearance order so the prompt
    target list is stable across re-runs."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t["lemma"] in targets:
            f = t["form"]
            if f and f not in seen:
                seen.add(f)
                out.append(f)
    return out


def matrix_stats(matrix) -> dict:
    n_phon = n_etym = n_sem = 0
    pair_locations: dict[tuple, list[int]] = defaultdict(list)
    for k in range(N_LOGIA - 1):
        for (la, lb, lt) in matrix.get((k, k + 1), frozenset()):
            if lt == "phonological":
                n_phon += 1
            elif lt == "etymological":
                n_etym += 1
            elif lt == "semantic":
                n_sem += 1
            pair_locations[(la, lb, lt)].append(k)
    return {
        "total_adjacencies": n_phon + n_etym + n_sem,
        "n_phon": n_phon, "n_etym": n_etym, "n_sem": n_sem,
        "recurring_2plus": sum(1 for v in pair_locations.values() if len(v) >= 2),
        "recurring_3plus": sum(1 for v in pair_locations.values() if len(v) >= 3),
    }


def perturb_one(client, types, lang_name: str, text: str, targets: list[str],
                  sleep: float) -> dict:
    """Single Gemini call to perturb one logion. Returns the same dict
    shape as translate_with_retry: {text, usage, error}."""
    system_prompt = PERTURB_SYSTEM.format(lang_name=lang_name)
    target_block = "; ".join(targets)
    full_prompt = (f"Target words to replace with a one-word synonym each: "
                    f"{target_block}\n\nSentence:\n{text}\n\nRewritten "
                    f"sentence (with each target replaced by a synonym, "
                    f"everything else preserved):")
    r = translate_with_retry(client, types, system_prompt, full_prompt,
                                lang_name)
    time.sleep(sleep)
    return r


def perturb_lang(lang: str, sleep: float, dry_run: bool = False) -> dict:
    """End-to-end: build baseline matrix, identify targets, perturb every
    affected logion (one Gemini call per logion), re-tokenize, rebuild
    matrix, return baseline + perturbed stats."""
    print(f"\n=== Perturbing {lang} ===")
    tokens = load_tokens(lang)
    raw_text = load_variant0_text(lang)
    n_loaded = sum(1 for v in tokens.values() if v)
    print(f"  loaded {n_loaded}/{N_LOGIA} logia")

    blocked = compute_blocked(tokens, FILTER_PCT)
    base_matrix = precompute_matrix(tokens, blocked, lang)
    targets = collect_target_lemmas(base_matrix)
    print(f"  {len(targets)} distinct catchword-anchor lemmas")

    base_stats = matrix_stats(base_matrix)
    print(f"  baseline: total_adj={base_stats['total_adjacencies']}, "
          f"phon={base_stats['n_phon']}, "
          f"etym={base_stats['n_etym']}, sem={base_stats['n_sem']}, "
          f"rec>=2={base_stats['recurring_2plus']}")

    # Initialise Gemini client lazily so dry-run mode does not need a key
    client = types_mod = None
    if not dry_run:
        load_env_file(ENV_FILE)
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit(f"GEMINI_API_KEY not in env. Add to {ENV_FILE}.")
        from google import genai
        from google.genai import types as types_mod_local
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        types_mod = types_mod_local

    lang_out = OUT_DIR_LOGIA / lang
    lang_out.mkdir(parents=True, exist_ok=True)

    n_perturbed = n_skipped_existing = n_skipped_no_target = 0
    total_in = total_out = 0
    perturbed_tokens: dict[int, list[dict]] = {}
    t0 = time.time()

    for i in range(N_LOGIA):
        toks = tokens.get(i, [])
        if not toks or i not in raw_text:
            continue
        target_forms = find_target_forms(toks, targets)
        if not target_forms:
            # No catchword anchor in this logion: keep original tokens.
            perturbed_tokens[i] = toks
            n_skipped_no_target += 1
            continue

        out_path = lang_out / f"logion_{i:03d}.json"
        if out_path.exists():
            d = json.loads(out_path.read_text(encoding="utf-8"))
            new_text = d.get("perturbed_text", "")
            n_skipped_existing += 1
        elif dry_run:
            new_text = raw_text[i]  # no perturbation in dry-run
        else:
            print(f"  [{i:3d}] targets={target_forms[:6]}"
                   f"{'...' if len(target_forms) > 6 else ''} "
                   f"({len(target_forms)} words)", flush=True)
            r = perturb_one(client, types_mod, LANG_HUMAN[lang],
                              raw_text[i], target_forms, sleep)
            if r["error"] or not r["text"]:
                print(f"    ERROR: {r['error']}; keeping original")
                new_text = raw_text[i]
            else:
                new_text = r["text"].strip()
            if r.get("usage"):
                total_in += r["usage"].prompt_token_count or 0
                total_out += (r["usage"].candidates_token_count or 0)
            out_path.write_text(json.dumps({
                "logion_number": i,
                "language": lang,
                "original_text": raw_text[i],
                "target_forms": target_forms,
                "perturbed_text": new_text,
                "model": MODEL,
                "error": r["error"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            n_perturbed += 1

        # Re-tokenize the perturbed text and re-build per-logion tokens.
        new_toks = [{"form": tok, "lemma": tok, "parse": "MS-EMP"}
                     for tok in tokenize(new_text, lang)]
        perturbed_tokens[i] = new_toks

    elapsed = time.time() - t0
    print(f"  perturbed {n_perturbed} new, reused {n_skipped_existing} cached, "
          f"skipped {n_skipped_no_target} (no targets)  ({elapsed:.0f}s)")

    # Re-block on the perturbed corpus + rebuild matrix.
    pb_blocked = compute_blocked(perturbed_tokens, FILTER_PCT)
    pb_matrix = precompute_matrix(perturbed_tokens, pb_blocked, lang)
    pb_stats = matrix_stats(pb_matrix)
    print(f"  perturbed: total_adj={pb_stats['total_adjacencies']}, "
          f"phon={pb_stats['n_phon']}, "
          f"etym={pb_stats['n_etym']}, sem={pb_stats['n_sem']}, "
          f"rec>=2={pb_stats['recurring_2plus']}")

    def pct(b, p):
        return 0.0 if b == 0 else (1.0 - p / b) * 100.0

    drop = {
        "total_adjacencies": pct(base_stats["total_adjacencies"],
                                   pb_stats["total_adjacencies"]),
        "n_phon":            pct(base_stats["n_phon"], pb_stats["n_phon"]),
        "n_etym":            pct(base_stats["n_etym"], pb_stats["n_etym"]),
        "n_sem":             pct(base_stats["n_sem"], pb_stats["n_sem"]),
        "recurring_2plus":   pct(base_stats["recurring_2plus"],
                                   pb_stats["recurring_2plus"]),
        "recurring_3plus":   pct(base_stats["recurring_3plus"],
                                   pb_stats["recurring_3plus"]),
    }
    return {
        "lang": lang,
        "n_target_lemmas": len(targets),
        "n_perturbed_logia": n_perturbed + n_skipped_existing,
        "n_skipped_no_target": n_skipped_no_target,
        "baseline": base_stats,
        "perturbed": pb_stats,
        "drop_pct": drop,
        "tokens_in": total_in,
        "tokens_out": total_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="greek,syriac",
                     help="Comma-separated list of languages")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    ap.add_argument("--dry-run", action="store_true",
                     help="Skip Gemini calls; reuse existing per-logion "
                          "perturbed files OR substitute originals.")
    args = ap.parse_args()

    print("=" * 76)
    print("Task 4: synonym perturbation (fragility)")
    print("=" * 76)

    results = {}
    for lang in args.langs.split(","):
        lang = lang.strip()
        results[lang] = perturb_lang(lang, args.sleep, dry_run=args.dry_run)

    # Compare Greek vs Syriac drops.
    print()
    print("=" * 76)
    print("SUMMARY — percentage drop (baseline -> perturbed)")
    print("=" * 76)
    hdr = f"{'metric':>22s} " + " ".join(f"{l:>14s}" for l in results)
    print(hdr)
    print("-" * len(hdr))
    for m in ["total_adjacencies", "n_phon", "n_etym", "n_sem",
              "recurring_2plus", "recurring_3plus"]:
        row = f"{m:>22s} "
        for lang, r in results.items():
            row += f" {r['drop_pct'][m]:>13.1f}%"
        print(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "task": "synonym_perturbation_fragility",
        "config": {
            "n_logia": N_LOGIA,
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "variant": 0,
            "syriac_tokenisation": "surface_forms_no_sedra",
            "model": MODEL,
            "temperature": TEMPERATURE,
        },
        "per_language": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
