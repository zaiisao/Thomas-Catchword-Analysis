#!/usr/bin/env python3
"""
Phase 2B Step 0 (Gemini) — small validation: 5 logia × 5 variants on
Gemini 3 Flash Preview to confirm output quality + check for Perrin's
catchword pair (nūrā / nuhrā) in Logia 10–11–16.

Notes
-----
* No grounding tools passed → no Google Search / URL Context.
* `thinking_config(thinking_budget=0)` → no chain-of-thought tokens
  (Flash 3.x defaults to using its output budget for "thinking").
* API key is read from `.env.local` in repo root, not from environment.

Usage
-----
  python scripts/phase2b_gemini_validate.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
ENV_FILE = REPO_ROOT / ".env.local"
OUT = REPO_ROOT / "data" / "processed" / "phase2b_gemini_validate.json"

MODEL = "gemini-3-flash-preview"          # version: 3-flash-preview-12-2025
N_VARIANTS = 5
MAX_OUTPUT_TOKENS = 1000
TEMPERATURE = 0.7
SLEEP_BETWEEN = 5.0                         # 12 RPM ≤ free-tier 15 RPM (pinned 3-flash-preview)
MAX_RETRIES = 4
TARGET_LOGIA = [1, 10, 11, 16, 86]

# Loose match for vocalized or unvocalized nūrā / nuhrā:
SYRIAC_RE = re.compile(r"[܀-ݏ]")
# Strip combining marks and check the consonantal skeleton matches
NURA_CONSONANTAL = "ܢܘܪܐ"
NUHRA_CONSONANTAL = "ܢܘܗܪܐ"

SYSTEM_PROMPT = (
    "You are an expert translator of ancient languages. "
    "Translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Output ONLY Syriac Unicode characters — no Hebrew, no transliteration, "
    "no English, no explanation, no commentary, no headings. "
    "Translate the full text — every sentence and clause from the source — "
    "preserving the original sequence of statements."
)


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def load_thomas_text():
    by_log = {}
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_log.setdefault(r["logion"], []).append(r.get("text", ""))
    return {L: " ".join(parts) for L, parts in by_log.items()}


def strip_combining(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def has_word(text, target_consonantal):
    """Match the target consonantal form anywhere in vocalized text by
    stripping combining diacritics first."""
    return target_consonantal in strip_combining(text)


def quality_metrics(text):
    if not text or not text.strip():
        return {"len": 0, "n_syr": 0, "syr_ratio": 0, "rep_max": 0,
                "unique_ratio": 0, "has_nura": False, "has_nuhra": False,
                "n_hebrew": 0}
    n_syr = len(SYRIAC_RE.findall(text))
    n_heb = len(re.findall(r"[֐-׿]", text))
    total = len(re.sub(r"\s", "", text))
    sr = n_syr / max(total, 1)
    toks = text.split()
    if toks:
        bigs = list(zip(toks, toks[1:]))
        rep = max(Counter(bigs).values()) / max(len(bigs), 1) if bigs else 0
        uniq = len(set(toks)) / len(toks)
    else:
        rep = uniq = 0
    return {"len": len(text), "n_syr": n_syr, "n_hebrew": n_heb,
            "syr_ratio": round(sr, 3),
            "rep_max": round(rep, 3),
            "unique_ratio": round(uniq, 3),
            "has_nura":  has_word(text, NURA_CONSONANTAL),
            "has_nuhra": has_word(text, NUHRA_CONSONANTAL)}


def main():
    load_env_file(ENV_FILE)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(f"GEMINI_API_KEY not found. Add it to {ENV_FILE}")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    thomas = load_thomas_text()
    print(f"Loaded {len(thomas)} logia.")
    print(f"Model: {MODEL}")
    print(f"Test logia: {TARGET_LOGIA}")
    print(f"Variants per logion: {N_VARIANTS}")
    print(f"thinking_budget=0 (disabled), no grounding tools (no Google Search).\n")

    results = {}
    total_in_tokens = total_out_tokens = 0
    for L in TARGET_LOGIA:
        coptic = thomas.get(L, "")
        if not coptic:
            print(f"Logion {L}: no Coptic text loaded; skipping.")
            continue
        print(f"=== Logion {L} ({len(coptic.split())} Coptic surface tokens) ===")
        variants = []
        for v in range(N_VARIANTS):
            t0 = time.time()
            text = ""; err = None; um = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=f"Coptic:\n{coptic}\n\nClassical Syriac (translate every sentence in order):",
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    text = resp.text or ""
                    um = getattr(resp, "usage_metadata", None)
                    err = None
                    break
                except Exception as e:
                    msg = str(e)
                    err = f"{type(e).__name__}: {msg[:200]}"
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        backoff = 30 * (attempt + 1)
                        print(f"  v{v}: rate-limited, sleeping {backoff}s before retry ({attempt+1}/{MAX_RETRIES})")
                        time.sleep(backoff)
                    elif "503" in msg or "UNAVAILABLE" in msg:
                        time.sleep(8)
                    else:
                        break
            if um:
                total_in_tokens  += um.prompt_token_count or 0
                total_out_tokens += um.candidates_token_count or 0
            if err is None and text:
                q = quality_metrics(text)
                q["seconds"] = round(time.time() - t0, 2)
                q["text"] = text
                variants.append({"variant": v, "success": True, **q})
                marks = []
                if q["has_nura"]:  marks.append("ܢⲩܪⲣ✓")
                if q["has_nuhra"]: marks.append("ܢⲩⲗⲣⲣ✓")
                if q["n_hebrew"] > 0: marks.append(f"!HEBREW={q['n_hebrew']}")
                marks_str = " ".join(marks) if marks else "[no perrin word]"
                print(f"  v{v}: syr%={q['syr_ratio']:.3f} uniq={q['unique_ratio']:.3f} "
                      f"({q['seconds']}s) {marks_str}")
                print(f"    {text[:200]}")
            else:
                variants.append({"variant": v, "success": False, "error": err or "unknown"})
                print(f"  v{v}: ERROR {(err or 'unknown')[:160]}")
            time.sleep(SLEEP_BETWEEN)
        results[L] = variants

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Logion':>7s}  {'#variants':>10s}  {'mean syr%':>10s}  "
          f"{'mean unique':>12s}  {'has nūrā':>10s}  {'has nuhrā':>11s}")
    for L, vs in results.items():
        good = [v for v in vs if v.get("success") and v.get("n_syr", 0) > 0]
        if not good:
            print(f"{L:>7d}  {0:>10d}  no successes")
            continue
        sr = sum(v["syr_ratio"] for v in good) / len(good)
        uq = sum(v["unique_ratio"] for v in good) / len(good)
        n_nura = sum(1 for v in good if v["has_nura"])
        n_nuhra = sum(1 for v in good if v["has_nuhra"])
        print(f"{L:>7d}  {len(good):>10d}  {sr:>10.3f}  {uq:>12.3f}  "
              f"{n_nura}/{len(good):>8d}  {n_nuhra}/{len(good):>9d}")

    # Token usage / cost estimate
    print(f"\nToken usage:  prompt={total_in_tokens}, output={total_out_tokens}")
    cost_in  = total_in_tokens  * 0.075 / 1_000_000
    cost_out = total_out_tokens * 0.30  / 1_000_000
    print(f"Approx cost (Flash pricing): ${cost_in + cost_out:.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "n_variants": N_VARIANTS,
                    "target_logia": TARGET_LOGIA,
                    "thinking_budget": 0,
                    "tokens_in": total_in_tokens,
                    "tokens_out": total_out_tokens,
                    "results": {str(L): v for L, v in results.items()}},
                   f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
