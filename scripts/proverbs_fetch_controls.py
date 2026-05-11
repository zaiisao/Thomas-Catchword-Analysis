#!/usr/bin/env python3
"""
Fetch Hebrew text for the control passages — narrative verses from non-wisdom
books that should NOT show catchword arrangement.

Output:
  data/proverbs/controls_hebrew.json — list of {unit_id, ref, hebrew_text}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env.local"
REFS = REPO_ROOT / "data" / "proverbs" / "controls_refs.json"
OUT  = REPO_ROOT / "data" / "proverbs" / "controls_hebrew.json"

MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 800
SLEEP = 0.5
MAX_RETRIES = 4

HEBREW_RE = re.compile(r"[֐-׿]")

SYSTEM_PROMPT = (
    "You are a Hebrew Bible text resource. Given a verse reference, output "
    "the Hebrew text from the Westminster Leningrad Codex (WLC) / BHS. "
    "Output ONLY the Hebrew text with niqqud, stripped of cantillation accents. "
    "No verse number, no English, no transliteration, no commentary."
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


def fetch_one(client, types, ref: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=f"Output {ref}:",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            t = (r.text or "").strip()
            if t and len(HEBREW_RE.findall(t)) >= 3:
                return t
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(30 * (attempt + 1))
            elif "503" in msg:
                time.sleep(8)
            else:
                time.sleep(2)
    return ""


def main() -> None:
    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    refs = json.loads(REFS.read_text(encoding="utf-8"))
    existing: dict[int, dict] = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text(encoding="utf-8")):
            existing[r["unit_id"]] = r

    for r in refs:
        uid = r["unit_id"]
        if uid in existing and existing[uid].get("hebrew_text"):
            continue
        print(f"[{uid}] {r['ref']}", flush=True)
        txt = fetch_one(client, types, r["passage"])
        if not txt:
            print(f"  FAILED")
        else:
            print(f"  → {txt[:60]}")
        existing[uid] = {
            "unit_id": uid,
            "ref": r["ref"],
            "hebrew_text": txt,
        }
        OUT.write_text(json.dumps(sorted(existing.values(),
                                          key=lambda x: x["unit_id"]),
                                   ensure_ascii=False, indent=2),
                       encoding="utf-8")
        time.sleep(SLEEP)

    n_ok = sum(1 for r in existing.values() if r.get("hebrew_text"))
    print(f"\nDone: {n_ok}/{len(refs)} fetched")


if __name__ == "__main__":
    main()
