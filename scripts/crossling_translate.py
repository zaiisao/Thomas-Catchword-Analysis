#!/usr/bin/env python3
"""
Cross-linguistic permutation test — Step 1: translate Thomas into Hebrew,
Arabic, and Greek using the SAME Gemini model & calibration as Phase 2B.

Identical structure to scripts/phase2b_gemini_translate.py — only the
target language and per-language system prompt vary.

Total per language: 115 logia × 10 variants = 1,150 calls.
Across 3 languages: 3,450 calls. At 5s/call: ~290 min.
(In practice we use 1.5s spacing because we're well within free-tier RPM.)

Outputs:
  data/processed/crossling_translations/{lang}/logion_{NNN}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
ENV_FILE = REPO_ROOT / ".env.local"
OUT_DIR = REPO_ROOT / "data" / "processed" / "crossling_translations"

MODEL = "gemini-3-flash-preview"
MODEL_VERSION = "3-flash-preview-12-2025"
N_VARIANTS = 10
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1200
SLEEP_BETWEEN_CALLS = 1.5
MAX_RETRIES = 4

PROMPTS = {
    "hebrew": (
        "You are an expert translator of ancient languages. "
        "Translate Sahidic Coptic into Classical Biblical Hebrew "
        "(as found in the Masoretic Text, fully pointed with niqqud). "
        "Output ONLY Hebrew Unicode characters — no transliteration, no "
        "Aramaic / Syriac, no English, no explanation, no commentary, no "
        "headings. Translate the full text — every sentence and clause from "
        "the source — preserving the original sequence of statements."
    ),
    "arabic": (
        "You are an expert translator of ancient languages. "
        "Translate Sahidic Coptic into Classical Arabic (Quranic register, "
        "fully voweled with tashkīl). "
        "Output ONLY Arabic Unicode characters — no transliteration, no "
        "Persian, no English, no explanation, no commentary, no headings. "
        "Translate the full text — every sentence and clause from the source "
        "— preserving the original sequence of statements."
    ),
    "greek": (
        "You are an expert translator of ancient languages. "
        "Translate Sahidic Coptic into Koine Greek (with accents and "
        "breathings, in the register of the Greek New Testament). "
        "Output ONLY Greek Unicode characters — no transliteration, no Latin, "
        "no English, no explanation, no commentary, no headings. "
        "Translate the full text — every sentence and clause from the source "
        "— preserving the original sequence of statements."
    ),
}

SCRIPT_RE = {
    "hebrew": re.compile(r"[֐-׿]"),
    "arabic": re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
    "greek":  re.compile(r"[Ͱ-Ͽἀ-῿]"),
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def load_thomas_text() -> dict[int, str]:
    by_log: dict[int, list[str]] = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                by_log[r["logion"]].append(r["text"])
    return {L: " ".join(parts) for L, parts in by_log.items()}


def usable_text(text: str, lang: str) -> bool:
    if not text or not text.strip():
        return False
    rx = SCRIPT_RE[lang]
    n_target = len(rx.findall(text))
    total = len(re.sub(r"\s", "", text))
    if total == 0:
        return False
    return n_target >= 5 and (n_target / total) >= 0.5


def translate_with_retry(client, types, system_prompt: str, coptic_text: str,
                          lang_name: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=(f"Coptic:\n{coptic_text}\n\n{lang_name} "
                          f"(translate every sentence in order):"),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = resp.text or ""
            um = getattr(resp, "usage_metadata", None)
            return {"text": text, "usage": um, "error": None}
        except Exception as e:
            msg = str(e)
            last_err = f"{type(e).__name__}: {msg[:200]}"
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                backoff = 30 * (attempt + 1)
                print(f"    rate-limited, sleeping {backoff}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(backoff)
            elif "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(8)
            else:
                break
    return {"text": "", "usage": None, "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(PROMPTS),
                     help="Target language: hebrew, arabic, or greek")
    ap.add_argument("--variants", type=int, default=N_VARIANTS)
    ap.add_argument("--logia", default="all",
                     help="'all' or comma-list of logion numbers")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY not in env. Add to {ENV_FILE}.")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    lang_dir = OUT_DIR / args.lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPTS[args.lang]
    lang_human = {"hebrew": "Biblical Hebrew",
                   "arabic": "Classical Arabic",
                   "greek":  "Koine Greek"}[args.lang]

    thomas = load_thomas_text()
    if args.logia == "all":
        logia = sorted(thomas)
    else:
        wanted = {int(x) for x in args.logia.split(",")}
        logia = [L for L in sorted(thomas) if L in wanted]

    n_calls_total = len(logia) * args.variants
    print(f"=== Cross-lingual translation — {args.lang} ===")
    print(f"  Logia: {len(logia)}  Variants: {args.variants}  "
          f"Total calls: {n_calls_total}")
    est_min = n_calls_total * (args.sleep + 1.5) / 60
    print(f"  Estimated runtime: {est_min:.0f} min")
    print(f"  Output dir: {lang_dir}\n")

    t_global = time.time()
    n_done = n_skipped = n_calls = 0
    total_in = total_out = 0

    for li, L in enumerate(logia):
        out_path = lang_dir / f"logion_{L:03d}.json"
        if out_path.exists():
            n_skipped += 1
            continue

        coptic = thomas[L]
        print(f"[{li+1}/{len(logia)}] logion_{L:03d} "
              f"({len(coptic.split())} cw)")
        variants = []
        good = 0
        t0 = time.time()
        for v in range(args.variants):
            r = translate_with_retry(client, types, system_prompt, coptic,
                                       lang_human)
            n_calls += 1
            if r["usage"]:
                total_in  += r["usage"].prompt_token_count or 0
                total_out += (r["usage"].candidates_token_count or 0) + \
                             (getattr(r["usage"], "thoughts_token_count", None) or 0)
            if r["error"] is None and r["text"]:
                rec = {"variant": v, "text": r["text"], "success": True,
                        "usable": usable_text(r["text"], args.lang)}
                if rec["usable"]:
                    good += 1
            else:
                rec = {"variant": v, "text": "", "success": False,
                        "error": r["error"]}
            variants.append(rec)
            time.sleep(args.sleep)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "logion_number": L,
                "language": args.lang,
                "coptic_text": coptic,
                "model": MODEL, "model_version": MODEL_VERSION,
                "temperature": TEMPERATURE,
                "n_variants": args.variants,
                "variants": variants,
            }, f, ensure_ascii=False, indent=2)
        n_done += 1
        elapsed = time.time() - t0
        rate_min = (time.time() - t_global) / max(n_done, 1) / 60
        eta_min = (len(logia) - li - 1) * rate_min
        print(f"  done {good}/{args.variants} usable in {elapsed:.0f}s   "
              f"(ETA: {eta_min:.0f} min)")

    cost = total_in * 0.075 / 1e6 + total_out * 0.30 / 1e6
    print()
    print(f"Final: {n_done} logia, {n_skipped} skipped, {n_calls} live calls")
    print(f"Tokens: {total_in} in / {total_out} out  (~${cost:.4f})")


if __name__ == "__main__":
    main()
