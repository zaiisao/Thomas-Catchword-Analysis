#!/usr/bin/env python3
"""
Fetch Koine Greek text for each Q pericope via Gemini API.

The IQP Q reference is keyed to the Lukan version (Luke generally preserves
Q's order). For each pericope, we ask Gemini to output the canonical
Lukan Greek (NA28 / SBLGNT register) for the given verse range.

Why Gemini and not parsing SBLGNT directly:
  - SBLGNT requires extracting from a GitHub repo + parsing OSIS/USFM
  - For ~56 short passages the cost is negligible (~$0.01 total)
  - Gemini reliably reproduces the Koine NT verbatim (high-resource text)

Output:
  data/q_source/q_pericopes_greek.json
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
DEFAULT_VERSES = REPO_ROOT / "data" / "q_source" / "q_verses.json"
DEFAULT_OUT = REPO_ROOT / "data" / "q_source" / "q_pericopes_greek.json"

MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.0   # We want the exact canonical text, no creativity
MAX_OUTPUT_TOKENS = 4000
SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 4

SYSTEM_PROMPT = (
    "You are a New Testament Koine Greek text resource. "
    "Given a Lukan verse reference, you output the exact Greek text from the "
    "SBLGNT / NA28 critical edition for that verse range — nothing more. "
    "Output ONLY the Greek text in continuous lowercase or mixed-case form. "
    "Do NOT include verse numbers, sigla, English translation, transliteration, "
    "explanation, or any commentary. "
    "Use Koine Greek with accents and breathings (polytonic). "
    "Strip footnote markers."
)

GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def usable_greek(text: str) -> bool:
    if not text or not text.strip():
        return False
    n_grk = len(GREEK_RE.findall(text))
    total = len(re.sub(r"\s", "", text))
    return n_grk >= 5 and (n_grk / max(total, 1)) >= 0.5


def fetch_one(client, types, ref: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=f"Output the Koine Greek text of {ref}:",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (resp.text or "").strip()
            return {"text": text, "error": None}
        except Exception as e:
            msg = str(e)
            last_err = f"{type(e).__name__}: {msg[:200]}"
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(30 * (attempt + 1))
            elif "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(8)
            else:
                break
    return {"text": "", "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pericopes", default="all")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    ap.add_argument("--input", default=str(DEFAULT_VERSES))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    global OUT
    OUT = Path(args.output)

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY not in env (expected in {ENV_FILE})")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    verses = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.pericopes != "all":
        wanted = {int(x) for x in args.pericopes.split(",")}
        verses = [v for v in verses if v["pericope"] in wanted]

    # Load existing output if any (resume support)
    existing: dict[int, dict] = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text(encoding="utf-8")):
            existing[r["pericope_id"]] = r

    out_records: dict[int, dict] = dict(existing)
    n_fetched = 0
    n_failed = 0

    for v in verses:
        pid = v["pericope"]
        if pid in existing and existing[pid].get("greek_text"):
            continue
        ref = v["luke_ref"]
        print(f"[{pid:2d}] {ref} ({v['label']})", flush=True)
        r = fetch_one(client, types, ref)
        if r["error"] or not usable_greek(r["text"]):
            print(f"  FAILED: {r['error']}  text={r['text'][:80]!r}")
            n_failed += 1
            rec = {"pericope_id": pid, "label": v["label"], "q_ref": v["q_ref"],
                   "luke_ref": ref, "greek_text": "", "error": r["error"]}
        else:
            rec = {"pericope_id": pid, "label": v["label"], "q_ref": v["q_ref"],
                   "luke_ref": ref, "greek_text": r["text"]}
            n_fetched += 1
            print(f"  → {r['text'][:80]}")
        out_records[pid] = rec
        # Save progressively
        OUT.write_text(json.dumps(sorted(out_records.values(),
                                          key=lambda x: x["pericope_id"]),
                                   ensure_ascii=False, indent=2),
                       encoding="utf-8")
        time.sleep(args.sleep)

    print()
    print(f"Fetched: {n_fetched}, failed: {n_failed}, cached: "
          f"{len(out_records) - n_fetched - n_failed}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
