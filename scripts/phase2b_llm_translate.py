#!/usr/bin/env python3
"""
Phase 2B — Coptic→Syriac translation via the Anthropic API (Claude).

Claude has broad multilingual knowledge including some Coptic and Syriac, but
*no* knowledge of Perrin's catchword hypothesis. So it's an informed but
unbiased translator — exactly what Phase 2 wants.

For each of 115 logia (Prologue + 114 sayings), we request 20 translations
at temperature=0.7. We also translate 10 random non-Thomas Coptic NT passages
of similar length as a control, to establish the base catchword rate for
arbitrary Coptic→Syriac translation.

The 20 translations × 115 logia × adjacency = 20×20=400 cross-pair catchword
combinations per adjacent pair gives a tight distribution.

Cost estimate (claude-sonnet-4-20250514, ~600 tokens in/out per call):
  2,300 calls × ~1,200 tokens × $3/Mtok input + $15/Mtok output ≈ $30-50.

Inputs:
  data/processed/got_logia/thomas_logia.jsonl     (Coptic source)
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl (control passages)

Output:
  data/processed/llm_translations/logion_{NNN:03}_v{VV:02}.txt
  data/processed/llm_translations/control_{NNN:03}_v{VV:02}.txt
  data/processed/phase2b_llm_results.json

Requirements:
  pip install anthropic
  export ANTHROPIC_API_KEY=...

Usage:
  python scripts/phase2b_llm_translate.py
  python scripts/phase2b_llm_translate.py --resume  (skip already-saved files)
  python scripts/phase2b_llm_translate.py --analyze-only  (skip API, just score)
"""

from __future__ import annotations

import argparse
import json
import os
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

THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
COPTIC_NT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
SEDRA_LIST = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
OUT_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
OUT_JSON = REPO_ROOT / "data" / "processed" / "phase2b_llm_results.json"

MODEL = "claude-sonnet-4-20250514"
N_VARIANTS_PER_LOGION = 20
N_CONTROL_PASSAGES = 10
TEMPERATURE = 0.7
MAX_TOKENS = 600
PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
SLEEP_BETWEEN_CALLS = 0.5  # seconds, to avoid rate limiting
SEED = 42

PROMPT_TEMPLATE = """Translate the following Sahidic Coptic text into Classical Syriac \
(Estrangela script).

Output ONLY the Syriac translation as a single line of Syriac text. \
Do NOT transliterate. Do NOT include English, commentary, formatting, headings, \
or any explanation. If you are uncertain about a word, choose the most natural \
Classical Syriac equivalent.

Coptic text:
{coptic_text}

Syriac translation:"""


def strip_vocalization(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def load_thomas_logia():
    """{logion_id: full_coptic_text}, concatenating paragraphs."""
    by_log = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                by_log[r["logion"]].append(r["text"])
    return {L: " ".join(parts) for L, parts in by_log.items()}


def load_control_passages(rng, n_passages, target_len_range=(40, 120)):
    """Sample N Coptic NT verses or short passages of length within range
    (rough word count). Use Sahidica TT data."""
    candidates = []
    with COPTIC_NT.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.get("text", "")
            n = len(text.split())
            if target_len_range[0] <= n <= target_len_range[1]:
                candidates.append({
                    "ref": f"{r.get('book')}:{r.get('chapter')}:{r.get('verse')}",
                    "text": text,
                })
    rng.shuffle(candidates)
    return candidates[:n_passages]


def load_sedra_lemma_table():
    """{consonantal_form -> lemma} from SEDRA-3 peshitta_list.txt.
    File format: unpointed | pointed | lemma | gloss | parse"""
    table = {}
    if not SEDRA_LIST.exists():
        return table
    with SEDRA_LIST.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 3:
                continue
            unpointed = parts[0].strip()
            lemma = parts[2].strip()
            if unpointed and lemma:
                # Last writer wins for now
                table[unpointed] = lemma
    return table


def tokenize_syriac_output(syriac_text, sedra_table):
    """Strip vocalization, split on whitespace, look up lemma in SEDRA;
    fall back to consonantal form as lemma."""
    cleaned = re.sub(r"[^܀-ݏ\s]", "", syriac_text)  # keep only Syriac block + ws
    consonantal = strip_vocalization(cleaned)
    tokens = []
    for w in consonantal.split():
        w = w.strip()
        if not w:
            continue
        lem = sedra_table.get(w, w)
        tokens.append({"form": w, "lemma": lem, "parse": "MS-EMP"})
    return tokens


def post_process_llm_output(raw_output: str) -> str:
    """The model may include markdown, headers, transliteration, refusals.
    Try to extract just the Syriac line."""
    raw_output = raw_output.strip()
    # Remove common markdown
    raw_output = re.sub(r"```[^\n]*\n", "", raw_output)
    raw_output = raw_output.replace("```", "")
    # Look for the longest line that contains Syriac unicode
    syriac_chars = re.compile(r"[܀-ݏ]")
    best_line = ""
    for line in raw_output.split("\n"):
        line = line.strip()
        if syriac_chars.search(line) and len(line) > len(best_line):
            best_line = line
    return best_line or raw_output


def call_claude(client, coptic_text, n_variants):
    """Call the API n_variants times with the same prompt."""
    out = []
    prompt = PROMPT_TEMPLATE.format(coptic_text=coptic_text)
    for i in range(n_variants):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            out.append(post_process_llm_output(text))
        except Exception as e:
            print(f"    [variant {i}] API error: {type(e).__name__}: {e}")
            out.append("")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return out


def translate_all(client, logia, control_passages, resume=False):
    """For each logion + control, write N variants to a file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(logia) + len(control_passages)
    for idx, (L, coptic_text) in enumerate(sorted(logia.items())):
        path = OUT_DIR / f"logion_{L:03d}.json"
        if resume and path.exists():
            continue
        print(f"[{idx+1}/{total}] Logion {L} ({len(coptic_text.split())} cw)…")
        t0 = time.time()
        variants = call_claude(client, coptic_text, N_VARIANTS_PER_LOGION)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"logion": L, "coptic": coptic_text,
                        "variants": variants}, f, ensure_ascii=False, indent=2)
        print(f"    saved {len(variants)} variants in {time.time()-t0:.1f}s")
    for ci, passage in enumerate(control_passages):
        path = OUT_DIR / f"control_{ci:03d}.json"
        if resume and path.exists():
            continue
        idx = len(logia) + ci
        print(f"[{idx+1}/{total}] Control {ci} ({passage['ref']})…")
        t0 = time.time()
        variants = call_claude(client, passage["text"], N_VARIANTS_PER_LOGION)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"control_index": ci, "ref": passage["ref"],
                        "coptic": passage["text"],
                        "variants": variants}, f, ensure_ascii=False, indent=2)
        print(f"    saved {len(variants)} variants in {time.time()-t0:.1f}s")


def analyze():
    """Score all variants with the Phase 1 detector and produce summary."""
    sedra_table = load_sedra_lemma_table()
    print(f"SEDRA lemma table: {len(sedra_table)} entries")

    # Load saved variants
    logia = {}
    for path in sorted(OUT_DIR.glob("logion_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        logia[d["logion"]] = [tokenize_syriac_output(v, sedra_table)
                              for v in d["variants"]]
    controls = []
    for path in sorted(OUT_DIR.glob("control_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        controls.append([tokenize_syriac_output(v, sedra_table)
                         for v in d["variants"]])
    print(f"Loaded {len(logia)} logia × ~{N_VARIANTS_PER_LOGION} variants each")
    print(f"Loaded {len(controls)} control passages × ~{N_VARIANTS_PER_LOGION} variants")

    if not logia:
        print("No translations to analyze. Set ANTHROPIC_API_KEY and run without --analyze-only.")
        return

    # Compute blocked set from variant 0 of each logion (one MAP-equivalent
    # translation, matching Phase 1 spirit)
    n_logia = len(logia)
    cutoff = FILTER_PCT * n_logia / 100.0
    lemma_count = Counter()
    for L, var_list in logia.items():
        if var_list and var_list[0]:
            for tok in var_list[0]:
                lemma_count[tok["lemma"]] += 0
            for lem in {tok["lemma"] for tok in var_list[0]}:
                lemma_count[lem] += 1
    blocked = {lem for lem, c in lemma_count.items() if c > cutoff}
    print(f"Blocked {len(blocked)} lemmas (in >{FILTER_PCT}% of logia variant 0)")

    det = CatchwordDetector("syriac",
                            phonological_threshold=PHON_THRESHOLD,
                            require_content_pos=False)

    def filter_tokens(toks):
        return [t for t in toks if t["lemma"] not in blocked]

    sorted_L = sorted(logia.keys())

    # Best-translation total (variant 0 across all logia)
    best_translation = {L: filter_tokens(logia[L][0]) if logia[L] else []
                        for L in sorted_L}
    best_total = 0
    cw_left, cw_right = set(), set()
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        cws = det.detect(best_translation[L], best_translation[Ln])
        best_total += len(cws)
        if len(cws) > 0:
            cw_right.add(L); cw_left.add(Ln)
    n = len(sorted_L)
    best_both = 100 * len(cw_left & cw_right) / n
    best_iso = 100 * len(set(sorted_L) - cw_left - cw_right) / n

    # Cross-pair distribution (variant_a × variant_b for each adjacent pair)
    pair_means = []
    pair_stds = []
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        var_a = [filter_tokens(t) for t in logia[L]]
        var_b = [filter_tokens(t) for t in logia[Ln]]
        if not var_a or not var_b:
            pair_means.append(0); pair_stds.append(0)
            continue
        counts = []
        for a in var_a:
            for b in var_b:
                counts.append(len(det.detect(a, b)))
        pair_means.append(float(np.mean(counts)))
        pair_stds.append(float(np.std(counts)))

    cross_total_mean = float(sum(pair_means))

    # Control: pair adjacent variants of consecutive controls (cross-pair too)
    control_total = 0
    if len(controls) >= 2:
        all_controls = controls
        ctrl_pair_means = []
        for i in range(len(all_controls) - 1):
            var_a = [filter_tokens(t) for t in all_controls[i]]
            var_b = [filter_tokens(t) for t in all_controls[i + 1]]
            counts = [len(det.detect(a, b)) for a in var_a for b in var_b]
            ctrl_pair_means.append(float(np.mean(counts)))
        control_total = float(sum(ctrl_pair_means))
        control_per_pair = float(np.mean(ctrl_pair_means)) if ctrl_pair_means else 0
    else:
        ctrl_pair_means = []
        control_per_pair = 0

    summary = {
        "config": {
            "model": MODEL,
            "n_variants": N_VARIANTS_PER_LOGION,
            "temperature": TEMPERATURE,
            "phon_threshold": PHON_THRESHOLD,
            "filter_pct": FILTER_PCT,
            "n_blocked": len(blocked),
        },
        "n_logia": len(logia),
        "n_controls": len(controls),
        "best_variant_total": best_total,
        "best_variant_both_pct": best_both,
        "best_variant_iso_pct": best_iso,
        "cross_pair_total_mean": cross_total_mean,
        "cross_pair_per_pair_mean": float(np.mean(pair_means)),
        "cross_pair_per_pair_std": float(np.mean(pair_stds)),
        "control_total": control_total,
        "control_per_pair_mean": control_per_pair,
        "perrin_target_total": 502,
        "phase1_mc_mean_total": 195.4,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Best variant total catchwords: {best_total} "
          f"(both={best_both:.1f}%, iso={best_iso:.1f}%)")
    print(f"Cross-pair mean total: {cross_total_mean:.1f}")
    print(f"Cross-pair per-pair: {summary['cross_pair_per_pair_mean']:.2f} "
          f"± {summary['cross_pair_per_pair_std']:.2f}")
    print(f"Control per-pair mean: {control_per_pair:.2f}")
    print(f"Wrote {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                     help="Skip files already saved on disk")
    ap.add_argument("--analyze-only", action="store_true",
                     help="Skip API calls; just score whatever is on disk")
    args = ap.parse_args()

    if args.analyze_only:
        analyze()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. "
                  "Run `export ANTHROPIC_API_KEY=sk-...` first, "
                  "or use --analyze-only to score whatever's on disk.")
    try:
        import anthropic
    except ImportError:
        sys.exit("ERROR: pip install anthropic")
    client = anthropic.Anthropic()

    rng = random.Random(SEED)
    logia = load_thomas_logia()
    controls = load_control_passages(rng, N_CONTROL_PASSAGES)
    print(f"Will translate {len(logia)} logia × {N_VARIANTS_PER_LOGION} variants "
          f"+ {len(controls)} controls × {N_VARIANTS_PER_LOGION} variants")
    print(f"Total API calls: {(len(logia) + len(controls)) * N_VARIANTS_PER_LOGION}")
    print(f"Estimated runtime: "
          f"{(len(logia) + len(controls)) * N_VARIANTS_PER_LOGION * (SLEEP_BETWEEN_CALLS + 1) / 60:.0f} min")

    translate_all(client, logia, controls, resume=args.resume)
    analyze()


if __name__ == "__main__":
    main()
