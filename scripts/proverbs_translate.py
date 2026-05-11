#!/usr/bin/env python3
"""
Translate Hebrew Proverbs (or controls) into Greek / Syriac / Aramaic / Arabic
via Gemini, same calibration as Phase 2B and Thomas crossling.

Each unit → 10 variants per target language.

Outputs:
  data/proverbs/translations/{lang}/unit_{NNN}.json    (Proverbs)
  data/proverbs/control_translations/{lang}/unit_{NNN}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env.local"
DEFAULT_PROV = REPO_ROOT / "data" / "proverbs" / "proverbs_hebrew.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "proverbs" / "translations"

MODEL = "gemini-3-flash-preview"
MODEL_VERSION = "3-flash-preview-12-2025"
N_VARIANTS = 10
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 800
SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 4

PROMPTS = {
    "greek": (
        "You are an expert translator. Translate the following Biblical "
        "Hebrew verse into Koine Greek (Septuagint register, NT-style accents "
        "and breathings). Output ONLY the Greek text — no transliteration, no "
        "Hebrew, no English, no commentary."
    ),
    "syriac": (
        "You are an expert translator. Translate the following Biblical "
        "Hebrew verse into Classical Syriac (Kthobonoyo / Estrangela). "
        "Output ONLY the Syriac text — no Hebrew, no transliteration, no "
        "English, no commentary."
    ),
    "aramaic": (
        "You are an expert translator. Translate the following Biblical "
        "Hebrew verse into Jewish Babylonian Aramaic (the Aramaic of the "
        "Babylonian Talmud, in Hebrew square script). Output ONLY the "
        "Aramaic text — no transliteration, no English, no commentary."
    ),
    "arabic": (
        "You are an expert translator. Translate the following Biblical "
        "Hebrew verse into Classical Arabic (Quranic register, fully voweled "
        "with tashkīl). Output ONLY the Arabic text — no transliteration, "
        "no Hebrew, no English, no commentary."
    ),
}

SCRIPT_RE = {
    "greek":   re.compile(r"[Ͱ-Ͽἀ-῿]"),
    "syriac":  re.compile(r"[܀-ݏ]"),
    "aramaic": re.compile(r"[֐-׿]"),
    "arabic":  re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
}

LANG_HUMAN = {
    "greek": "Koine Greek (LXX register)",
    "syriac": "Classical Syriac",
    "aramaic": "Jewish Babylonian Aramaic",
    "arabic": "Classical Arabic",
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


def usable_text(text: str, lang: str) -> bool:
    if not text or not text.strip():
        return False
    rx = SCRIPT_RE[lang]
    n_target = len(rx.findall(text))
    total = len(re.sub(r"\s", "", text))
    if total == 0:
        return False
    return n_target >= 3 and (n_target / total) >= 0.5


def translate_one(client, types, system_prompt: str, hebrew_text: str,
                   lang_human: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=(f"Hebrew:\n{hebrew_text}\n\n{lang_human}:"),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (resp.text or "").strip()
            um = getattr(resp, "usage_metadata", None)
            return {"text": text, "usage": um, "error": None}
        except Exception as e:
            msg = str(e)
            last_err = f"{type(e).__name__}: {msg[:200]}"
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(30 * (attempt + 1))
            elif "503" in msg:
                time.sleep(8)
            else:
                break
    return {"text": "", "usage": None, "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(PROMPTS))
    ap.add_argument("--variants", type=int, default=N_VARIANTS)
    ap.add_argument("--units", default="all",
                     help="'all' or comma-list of unit_ids")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    ap.add_argument("--input", default=str(DEFAULT_PROV),
                     help="Hebrew source JSON (Proverbs or controls)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    units = json.loads(Path(args.input).read_text(encoding="utf-8"))
    units = [u for u in units if u.get("hebrew_text")]
    if args.units != "all":
        wanted = {int(x) for x in args.units.split(",")}
        units = [u for u in units if u["unit_id"] in wanted]

    lang_dir = Path(args.out_dir) / args.lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPTS[args.lang]
    lang_human = LANG_HUMAN[args.lang]

    n_calls_total = len(units) * args.variants
    print(f"=== Proverbs translation — {args.lang} ===")
    print(f"  units: {len(units)}, variants: {args.variants}, "
          f"total calls: {n_calls_total}")
    print(f"  out dir: {lang_dir}\n")

    t_global = time.time()
    n_done = n_skipped = n_calls = 0
    for ui, u in enumerate(units):
        uid = u["unit_id"]
        out_path = lang_dir / f"unit_{uid:04d}.json"
        if out_path.exists():
            n_skipped += 1
            continue
        heb = u["hebrew_text"]
        variants = []
        good = 0
        t0 = time.time()
        for v in range(args.variants):
            r = translate_one(client, types, system_prompt, heb, lang_human)
            n_calls += 1
            if r["error"] is None and r["text"]:
                rec = {"variant": v, "text": r["text"], "success": True,
                        "usable": usable_text(r["text"], args.lang)}
                if rec["usable"]: good += 1
            else:
                rec = {"variant": v, "text": "", "success": False,
                        "error": r["error"]}
            variants.append(rec)
            time.sleep(args.sleep)
        out_path.write_text(json.dumps({
            "unit_id": uid,
            "ref": u.get("ref"),
            "language": args.lang,
            "hebrew_text": heb,
            "model": MODEL, "model_version": MODEL_VERSION,
            "temperature": TEMPERATURE,
            "n_variants": args.variants,
            "variants": variants,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        n_done += 1
        if n_done % 10 == 0 or n_done == len(units):
            elapsed = time.time() - t_global
            rate = elapsed / n_done
            eta = (len(units) - ui - 1) * rate
            print(f"[{ui+1}/{len(units)}] uid={uid}  good={good}/{args.variants}  "
                  f"({time.time()-t0:.0f}s)  ETA {eta:.0f}s", flush=True)

    print()
    print(f"Final: {n_done} units done, {n_skipped} cached, {n_calls} live calls")


if __name__ == "__main__":
    main()
