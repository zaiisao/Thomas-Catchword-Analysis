#!/usr/bin/env python3
"""
Phase 2B blind verification — back-translate the LLM's Classical Syriac
output back to English using a DIFFERENT Gemini model (2.5 Flash, vs the
forward run on 3 Flash Preview), with NO context that the source is
Coptic, Thomas, or related to any catchword hypothesis. The back-
translator only knows it's translating Classical Syriac → English.

If Gemini's forward Coptic→Syriac is faithful, the back-translation
should reproduce the canonical content of each logion (foxes have holes,
fire upon the world, etc.).

Output: data/processed/phase2b_blind_verify.json with columns
  passage_id, coptic_source, syriac_v0, english_backtranslation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
ENV_FILE = REPO_ROOT / ".env.local"
OUT = REPO_ROOT / "data" / "processed" / "phase2b_blind_verify.json"

VERIFIER_MODEL = "gemini-2.5-flash"   # different family/version from forward
SLEEP_BETWEEN = 6.0
MAX_RETRIES = 4

# Strictly no mention of Coptic, Thomas, gospel, Perrin, catchword.
SYSTEM_PROMPT = (
    "You are a translator. Translate Classical Syriac (Estrangela script) "
    "into literal English. Translate exactly what the Syriac says, in the "
    "same order, sentence by sentence. Do not add commentary, headings, "
    "or footnotes. Output the English translation only."
)


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def load_thomas_translations(n_sample, seed=42):
    """Pick n_sample random already-translated Thomas logia (skip controls)."""
    rng = random.Random(seed)
    files = []
    for path in sorted(LLM_DIR.glob("logion_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("is_control"):
                continue
            # Need at least one usable variant
            if not any(v.get("success") for v in d.get("variants", [])):
                continue
            files.append((path, d))
        except Exception:
            continue
    if len(files) < n_sample:
        print(f"WARNING: only {len(files)} usable files, sampling all of them")
        return files
    return rng.sample(files, n_sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY not in env. Add to {ENV_FILE}.")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    samples = load_thomas_translations(args.n, args.seed)
    print(f"Verifying {len(samples)} random Thomas logia "
          f"(blind back-translation via {VERIFIER_MODEL}).")

    results = []
    for i, (path, d) in enumerate(samples):
        # Take variant 0
        v0 = next((v for v in d["variants"] if v.get("success")), None)
        if not v0:
            continue
        syr = v0["syriac_text"].strip()

        print(f"\n[{i+1}/{len(samples)}] {d['passage_id']}")
        # back-translate
        eng = ""; err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=VERIFIER_MODEL,
                    contents=f"Syriac:\n{syr}\n\nEnglish (literal):",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,           # deterministic-ish
                        max_output_tokens=400,
                    ),
                )
                eng = (resp.text or "").strip()
                err = None
                break
            except Exception as e:
                msg = str(e)
                err = msg[:200]
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(20 * (attempt + 1))
                else:
                    break

        results.append({
            "passage_id": d["passage_id"],
            "logion_number": d.get("logion_number"),
            "coptic": d["coptic_text"],
            "syriac_v0": syr,
            "english_backtranslation": eng,
            "back_error": err,
        })
        print(f"  Coptic: {d['coptic_text'][:120]}")
        print(f"  Syriac: {syr[:120]}")
        print(f"  EN-bt:  {eng[:200]}")
        time.sleep(SLEEP_BETWEEN)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"verifier_model": VERIFIER_MODEL, "n": len(results),
                    "results": results},
                   f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
