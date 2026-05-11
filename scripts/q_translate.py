#!/usr/bin/env python3
"""
Translate Q pericopes (Koine Greek source) into Aramaic / Syriac / Hebrew
via Gemini, same calibration as Phase 2B and Thomas crossling translations.

For each pericope, produce 10 variants per target language. Saves one
JSON file per pericope (matching the Thomas crossling_translations layout).

Outputs:
  data/q_source/translations/{lang}/pericope_{NNN}.json
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
DEFAULT_GREEK = REPO_ROOT / "data" / "q_source" / "q_pericopes_greek.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "q_source" / "translations"

MODEL = "gemini-3-flash-preview"
MODEL_VERSION = "3-flash-preview-12-2025"
N_VARIANTS = 10
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1500
SLEEP_BETWEEN_CALLS = 1.5
MAX_RETRIES = 4

PROMPTS = {
    "aramaic": (
        "You are an expert translator of New Testament Koine Greek into "
        "Jewish Babylonian Aramaic — the Aramaic of the Babylonian Talmud, "
        "written in Hebrew square script. "
        "Output ONLY the Aramaic translation in Hebrew Unicode characters. "
        "Do not include vocalization marks unless they are essential to "
        "disambiguate the consonantal text. "
        "Do not include transliteration, Hebrew, Greek, English, "
        "no commentary, no headings. Translate the full passage — every "
        "sentence and clause from the source — preserving the original "
        "sequence of statements."
    ),
    "syriac": (
        "You are an expert translator of New Testament Koine Greek into "
        "Classical Syriac (Kthobonoyo / Estrangela script). "
        "Output ONLY the Syriac translation in Syriac Unicode characters — "
        "no Hebrew, no transliteration, no English, no explanation, no "
        "commentary, no headings. Translate the full passage — every "
        "sentence and clause from the source — preserving the original "
        "sequence of statements."
    ),
    "hebrew": (
        "You are an expert translator of New Testament Koine Greek into "
        "Classical Biblical Hebrew (Masoretic register, fully pointed with "
        "niqqud). "
        "Output ONLY the Hebrew translation in Hebrew Unicode characters — "
        "no transliteration, no Aramaic / Syriac, no English, no "
        "explanation, no commentary, no headings. Translate the full "
        "passage — every sentence and clause from the source — preserving "
        "the original sequence of statements."
    ),
    "arabic": (
        "You are an expert translator of New Testament Koine Greek into "
        "Classical Arabic (Quranic register, fully voweled with tashkīl). "
        "Output ONLY the Arabic translation in Arabic Unicode characters — "
        "no transliteration, no Persian, no English, no explanation, no "
        "commentary, no headings. Translate the full passage — every "
        "sentence and clause from the source — preserving the original "
        "sequence of statements."
    ),
}

# Hebrew block covers Hebrew + Aramaic (JBA uses Hebrew script).
SCRIPT_RE = {
    "aramaic": re.compile(r"[֐-׿]"),
    "hebrew":  re.compile(r"[֐-׿]"),
    "syriac":  re.compile(r"[܀-ݏ]"),
    "arabic":  re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"),
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
    return n_target >= 5 and (n_target / total) >= 0.5


def translate_one(client, types, system_prompt: str, greek_text: str,
                   lang_human: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=(f"Koine Greek:\n{greek_text}\n\n{lang_human} "
                          f"(translate every sentence in order):"),
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
            elif "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(8)
            else:
                break
    return {"text": "", "usage": None, "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(PROMPTS))
    ap.add_argument("--variants", type=int, default=N_VARIANTS)
    ap.add_argument("--pericopes", default="all",
                     help="'all' or comma-list of pericope ids")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    ap.add_argument("--input", default=str(DEFAULT_GREEK),
                     help="Path to Greek source JSON (Q or controls)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                     help="Output directory root (the language subdir lives "
                          "inside this)")
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY not in env (expected in {ENV_FILE})")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    pericopes = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pericopes = [p for p in pericopes if p.get("greek_text")]
    if args.pericopes != "all":
        wanted = {int(x) for x in args.pericopes.split(",")}
        pericopes = [p for p in pericopes if p["pericope_id"] in wanted]

    out_dir_root = Path(args.out_dir)
    lang_dir = out_dir_root / args.lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPTS[args.lang]
    lang_human = {
        "aramaic": "Jewish Babylonian Aramaic",
        "syriac":  "Classical Syriac",
        "hebrew":  "Biblical Hebrew",
        "arabic":  "Classical Arabic",
    }[args.lang]

    n_calls_total = len(pericopes) * args.variants
    print(f"=== Q translation — {args.lang} ===")
    print(f"  Pericopes: {len(pericopes)}, variants: {args.variants}, "
          f"total calls: {n_calls_total}")
    print(f"  Output dir: {lang_dir}\n")

    t_global = time.time()
    n_done = n_skipped = n_calls = 0

    for pi, p in enumerate(pericopes):
        pid = p["pericope_id"]
        out_path = lang_dir / f"pericope_{pid:03d}.json"
        if out_path.exists():
            n_skipped += 1
            continue

        greek = p["greek_text"]
        print(f"[{pi+1}/{len(pericopes)}] pericope_{pid:03d} "
              f"({len(greek.split())} Greek words)", flush=True)
        variants = []
        good = 0
        t0 = time.time()
        for v in range(args.variants):
            r = translate_one(client, types, system_prompt, greek, lang_human)
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
            "pericope_id": pid,
            "label": p["label"],
            "q_ref": p["q_ref"],
            "luke_ref": p["luke_ref"],
            "language": args.lang,
            "greek_text": greek,
            "model": MODEL, "model_version": MODEL_VERSION,
            "temperature": TEMPERATURE,
            "n_variants": args.variants,
            "variants": variants,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        n_done += 1
        elapsed = time.time() - t0
        rate = (time.time() - t_global) / max(n_done, 1)
        eta = (len(pericopes) - pi - 1) * rate
        print(f"  done {good}/{args.variants} usable in {elapsed:.0f}s   "
              f"(ETA: {eta:.0f}s)")

    print()
    print(f"Final: {n_done} pericopes done, {n_skipped} cached, "
          f"{n_calls} live calls")


if __name__ == "__main__":
    main()
