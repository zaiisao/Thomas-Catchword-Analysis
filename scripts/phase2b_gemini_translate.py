#!/usr/bin/env python3
"""
Phase 2B Step 3 (Gemini) — full translation of all 115 Gospel of Thomas
logia + 10 control passages, 10 variants each, via Gemini 3 Flash Preview.

Model: gemini-3-flash-preview (version 3-flash-preview-12-2025)
Tools: NONE → no Google Search / URL Context grounding
Thinking: thinking_budget=0 (Flash supports this; no chain-of-thought tokens)
Resume: per-passage JSON files; reruns skip files already on disk

Total: 125 × 10 = 1,250 calls. At ~12 RPM (5s spacing): ~104 min minimum.

Inputs:
  data/processed/got_logia/thomas_logia.jsonl
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
Output:
  data/processed/llm_translations/{passage_id}.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
COPTIC_NT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
ENV_FILE = REPO_ROOT / ".env.local"
OUT_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"

MODEL = "gemini-3-flash-preview"
MODEL_VERSION = "3-flash-preview-12-2025"
N_VARIANTS = 10
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1200
SLEEP_BETWEEN_CALLS = 5.0   # 12 RPM
MAX_RETRIES = 4

SYSTEM_PROMPT = (
    "You are an expert translator of ancient languages. "
    "Translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Output ONLY Syriac Unicode characters — no Hebrew, no transliteration, "
    "no English, no explanation, no commentary, no headings. "
    "Translate the full text — every sentence and clause from the source — "
    "preserving the original sequence of statements."
)

CONTROL_REFS = [
    ("Romans", 8, [28, 29, 30]),
    ("1_Corinthians", 13, [1, 2, 3]),
    ("1_Corinthians", 13, [4, 5, 6, 7]),
    ("Galatians", 5, [22, 23, 24]),
    ("Ephesians", 2, [8, 9, 10]),
    ("Philippians", 2, [5, 6, 7, 8]),
    ("Philippians", 4, [6, 7]),
    ("Colossians", 3, [12, 13, 14]),
    ("1_Thessalonians", 5, [16, 17, 18]),
    ("2_Timothy", 2, [15]),
]

SYRIAC_RE = re.compile(r"[܀-ݏ]")
HEBREW_RE = re.compile(r"[֐-׿]")


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
    by_log = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                by_log[r["logion"]].append(r["text"])
    return {L: " ".join(parts) for L, parts in by_log.items()}


def load_control_passages():
    by_book = defaultdict(dict)
    with COPTIC_NT.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_book[r["book"]].setdefault(r["chapter"], {})[r["verse"]] = r.get("text", "")
    out = []
    for book, ch, verses in CONTROL_REFS:
        verse_texts = [by_book.get(book, {}).get(ch, {}).get(v, "")
                       for v in verses]
        if not all(verse_texts):
            print(f"  WARNING: missing {book} {ch}:{verses}", file=sys.stderr)
            continue
        joined = " ".join(verse_texts)
        rng = f"{verses[0]}-{verses[-1]}" if len(verses) > 1 else str(verses[0])
        out.append({
            "id": f"control_{book}_{ch}_{rng}",
            "book": book, "chapter": ch, "verses": verses,
            "coptic_text": joined,
            "is_control": True,
        })
    return out


def usable_syriac(text):
    if not text or not text.strip():
        return False
    n_syr = len(SYRIAC_RE.findall(text))
    n_heb = len(HEBREW_RE.findall(text))
    if n_heb > n_syr:
        return False
    total = len(re.sub(r"\s", "", text))
    return n_syr >= 5 and (n_syr / max(total, 1)) >= 0.5


def translate_with_retry(client, types, coptic_text):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=f"Coptic:\n{coptic_text}\n\nClassical Syriac (translate every sentence in order):",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
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
                print(f"    rate-limited, sleeping {backoff}s before retry "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
            elif "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(8)
            else:
                break
    return {"text": "", "usage": None, "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=N_VARIANTS)
    ap.add_argument("--passages", default="all",
                     help="'all', 'thomas', 'controls', or comma-list of logion numbers")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS,
                     help="seconds between calls")
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY not in env. Add to {ENV_FILE}.")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    thomas = load_thomas_text()
    controls = load_control_passages()
    print(f"Loaded {len(thomas)} Thomas logia and {len(controls)} controls.")

    if args.passages == "all":
        passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                      "coptic_text": t, "is_control": False}
                     for L, t in sorted(thomas.items())] + controls
    elif args.passages == "thomas":
        passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                      "coptic_text": t, "is_control": False}
                     for L, t in sorted(thomas.items())]
    elif args.passages == "controls":
        passages = controls
    else:
        wanted = {int(x) for x in args.passages.split(",")}
        passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                      "coptic_text": t, "is_control": False}
                     for L, t in sorted(thomas.items()) if L in wanted]

    n_calls_total = len(passages) * args.variants
    print(f"Will run {len(passages)} passages × {args.variants} variants "
          f"= {n_calls_total} calls.")
    est_min = (n_calls_total * (args.sleep + 2)) / 60
    print(f"  Estimated runtime: {est_min:.0f} min")
    print(f"  Model: {MODEL} (version {MODEL_VERSION})")
    print(f"  thinking_budget=0; no grounding tools.\n")

    t_global = time.time()
    n_done = n_skipped = n_calls_made = 0
    total_in = total_out = 0

    for pi, p in enumerate(passages):
        out_path = OUT_DIR / f"{p['id']}.json"
        if out_path.exists():
            n_skipped += 1
            continue

        coptic = p["coptic_text"]
        print(f"[{pi+1}/{len(passages)}] {p['id']} "
              f"(content={len(coptic.split())} cw, control={p['is_control']})")
        variants = []
        good = 0
        t0 = time.time()
        for v in range(args.variants):
            r = translate_with_retry(client, types, coptic)
            n_calls_made += 1
            if r["usage"]:
                total_in  += r["usage"].prompt_token_count or 0
                total_out += (r["usage"].candidates_token_count or 0) + \
                             (getattr(r["usage"], "thoughts_token_count", None) or 0)
            if r["error"] is None and r["text"]:
                rec = {"variant": v, "syriac_text": r["text"],
                        "success": True}
                if usable_syriac(r["text"]):
                    good += 1
                    rec["usable"] = True
                else:
                    rec["usable"] = False
            else:
                rec = {"variant": v, "syriac_text": "", "success": False,
                        "error": r["error"]}
            variants.append(rec)
            time.sleep(args.sleep)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "passage_id": p["id"],
                "logion_number": p.get("logion_number"),
                "coptic_text": coptic,
                "is_control": p["is_control"],
                "model": MODEL, "model_version": MODEL_VERSION,
                "temperature": TEMPERATURE,
                "n_variants": args.variants,
                "variants": variants,
            }, f, ensure_ascii=False, indent=2)
        n_done += 1
        elapsed = time.time() - t0
        rate_min = (time.time() - t_global) / max(n_done, 1) / 60
        remain_min = (len(passages) - pi - 1) * rate_min
        print(f"  done {good}/{args.variants} usable in {elapsed:.0f}s   "
              f"(ETA: {remain_min:.0f} min)")

    print()
    cost = total_in * 0.075 / 1e6 + total_out * 0.30 / 1e6
    print(f"Final: {n_done} passages, {n_skipped} skipped (cached), "
          f"{n_calls_made} live calls.")
    print(f"Tokens: {total_in} in / {total_out} out  (~${cost:.4f} estimated)")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
