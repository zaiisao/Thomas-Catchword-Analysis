#!/usr/bin/env python3
"""
Fetch the Westminster Leningrad Codex (WLC / BHS) Hebrew text for
Proverbs 10:1–29:27 via Gemini, batched 25 verses per call.

The Hebrew Bible is well-attested in Gemini's training corpus; the
WLC / BHS is the standard critical text. Per-verse correctness is
spot-checkable.

Output:
  data/proverbs/proverbs_hebrew.json  — list of {unit_id, ref, hebrew_text}
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
OUT = REPO_ROOT / "data" / "proverbs" / "proverbs_hebrew.json"

MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 6000
SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 4

HEBREW_RE = re.compile(r"[֐-׿]")

SYSTEM_PROMPT = (
    "You are a Hebrew Bible text resource. Given a verse-range request for "
    "the Westminster Leningrad Codex (WLC) / BHS critical text, you output "
    "the Hebrew text as a JSON array. Each element is an object with two "
    "fields: 'verse' (integer) and 'text' (Hebrew Unicode string with niqqud "
    "but stripped of cantillation accents). Output ONLY the JSON array. "
    "No markdown fences. No commentary. No verse number prefixes in the text "
    "field. Include all verses in the requested range."
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


def usable_verse(text: str) -> bool:
    if not text or not text.strip():
        return False
    n_heb = len(HEBREW_RE.findall(text))
    return n_heb >= 3


def fetch_range(client, types, chapter: int, v_start: int, v_end: int,
                  model: str = MODEL) -> list[dict]:
    """Fetch Hebrew text for Proverbs chapter:v_start–v_end. Returns a list
    of {verse, text} dicts."""
    ref = f"Proverbs {chapter}:{v_start}-{v_end}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=(f"Output {ref} from the Westminster Leningrad Codex / BHS, "
                          f"as a JSON array per the system instruction."),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                ),
            )
            raw = (resp.text or "").strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                last_err = f"non-list response: {type(data).__name__}"
                continue
            return data
        except Exception as e:
            msg = str(e)
            last_err = f"{type(e).__name__}: {msg[:200]}"
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(30 * (attempt + 1))
            elif "503" in msg:
                time.sleep(8)
            else:
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(2)
    print(f"  FAILED {ref}: {last_err}")
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-chapter", type=int, default=10)
    ap.add_argument("--end-chapter", type=int, default=29)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    load_env_file(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(f"GEMINI_API_KEY missing")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Resume support: load any existing output
    existing: dict[str, dict] = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text(encoding="utf-8")):
            existing[r["ref"]] = r

    verses_out: list[dict] = list(existing.values())
    next_id = max((r["unit_id"] for r in verses_out), default=0) + 1

    # Verses-per-chapter for Proverbs (Masoretic versification)
    PROV_LENGTHS = {
        10: 32, 11: 31, 12: 28, 13: 25, 14: 35, 15: 33, 16: 33, 17: 28,
        18: 24, 19: 29, 20: 30, 21: 31, 22: 29, 23: 35, 24: 34, 25: 28,
        26: 28, 27: 27, 28: 28, 29: 27,
    }

    for ch in range(args.start_chapter, args.end_chapter + 1):
        n_verses = PROV_LENGTHS[ch]
        # Are all verses in this chapter already fetched?
        n_have = sum(1 for v in range(1, n_verses + 1)
                      if f"Prov {ch}:{v}" in existing
                      and existing[f"Prov {ch}:{v}"].get("hebrew_text"))
        if n_have == n_verses:
            print(f"Ch {ch}: cached {n_have}/{n_verses}")
            continue
        print(f"Ch {ch}: fetching ({n_have}/{n_verses} cached)")
        # Fetch in one call per chapter (typical chapter is 27-35 verses)
        data = fetch_range(client, types, ch, 1, n_verses, model=args.model)
        for entry in data:
            try:
                vnum = int(entry.get("verse"))
                vtext = (entry.get("text") or "").strip()
            except (TypeError, ValueError):
                continue
            if not usable_verse(vtext):
                continue
            ref = f"Prov {ch}:{vnum}"
            if ref in existing:
                continue
            existing[ref] = {
                "unit_id": next_id,
                "ref": ref,
                "chapter": ch,
                "verse": vnum,
                "hebrew_text": vtext,
            }
            next_id += 1
        verses_out = list(existing.values())
        # Sort canonically: by (chapter, verse)
        verses_out.sort(key=lambda r: (r["chapter"], r["verse"]))
        # Reassign unit_id 1..N in canonical order
        for i, r in enumerate(verses_out, start=1):
            r["unit_id"] = i
        OUT.write_text(json.dumps(verses_out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        time.sleep(SLEEP_BETWEEN_CALLS)

    print()
    n_final = len(verses_out)
    avg_words = sum(len(r["hebrew_text"].split()) for r in verses_out) / max(n_final, 1)
    print(f"Total: {n_final} verses, avg words per verse: {avg_words:.1f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
