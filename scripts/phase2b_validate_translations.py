#!/usr/bin/env python3
"""Phase 2B Step 4 — validate generated Syriac translations.

Reads everything in data/processed/llm_translations/ and reports per-issue
counts: NO_SYRIAC / LOW_SYRIAC_RATIO / HAS_COMMENTARY / TOO_SHORT / API_ERROR.
Also writes a per-passage breakdown so we can re-run the bad ones later.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
OUT = REPO_ROOT / "data" / "processed" / "phase2b_validation_summary.json"

SYRIAC_RE = re.compile(r"[܀-ݏ]")
COMMENTARY_MARKERS = ("translation", "note:", "meaning:", "literally:",
                       "this translates", "the syriac", "here is", "coptic text",
                       "english:", "english translation")


def diagnose(text: str):
    if not text or not text.strip():
        return "TOO_SHORT", text[:120]
    syr_chars = SYRIAC_RE.findall(text)
    n_syr = len(syr_chars)
    total = len(re.sub(r"\s", "", text))
    ratio = n_syr / max(total, 1)
    if n_syr == 0:
        return "NO_SYRIAC", text[:120]
    if ratio < 0.5:
        return "LOW_SYRIAC_RATIO", f"{n_syr}/{total}"
    low = text.lower()
    for m in COMMENTARY_MARKERS:
        if m in low:
            return "HAS_COMMENTARY", m
    if len(text.strip()) < 5:
        return "TOO_SHORT", text
    return "OK", None


def main():
    if not LLM_DIR.exists():
        sys.exit(f"No translations directory: {LLM_DIR}")

    files = sorted(LLM_DIR.glob("*.json"))
    if not files:
        sys.exit("No translation files yet — run scripts/phase2b_qwen_translate.py first.")

    issues = []
    stats = {"total": 0, "ok": 0, "NO_SYRIAC": 0, "LOW_SYRIAC_RATIO": 0,
             "HAS_COMMENTARY": 0, "TOO_SHORT": 0, "API_ERROR": 0}
    per_passage = {}
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        pid = d["passage_id"]
        per_passage[pid] = {"good": 0, "bad": [], "is_control": d.get("is_control", False),
                              "n_variants": len(d["variants"])}
        for v in d["variants"]:
            stats["total"] += 1
            if not v.get("success"):
                stats["API_ERROR"] += 1
                per_passage[pid]["bad"].append((v["variant"], "API_ERROR",
                                                  v.get("error", "")))
                continue
            kind, detail = diagnose(v.get("syriac_text", ""))
            if kind == "OK":
                stats["ok"] += 1
                per_passage[pid]["good"] += 1
            elif kind in stats:
                stats[kind] += 1
                per_passage[pid]["bad"].append((v["variant"], kind, detail))
            else:
                per_passage[pid]["bad"].append((v["variant"], kind, detail))

    print(f"Total variants:       {stats['total']}")
    print(f"  Good:               {stats['ok']:>5d} "
          f"({100*stats['ok']/max(stats['total'],1):.1f}%)")
    print(f"  NO_SYRIAC:          {stats['NO_SYRIAC']:>5d}")
    print(f"  LOW_SYRIAC_RATIO:   {stats['LOW_SYRIAC_RATIO']:>5d}")
    print(f"  HAS_COMMENTARY:     {stats['HAS_COMMENTARY']:>5d}")
    print(f"  TOO_SHORT:          {stats['TOO_SHORT']:>5d}")
    print(f"  API_ERROR:          {stats['API_ERROR']:>5d}")

    bad_passages = [pid for pid, d in per_passage.items() if d["good"] < 10]
    if bad_passages:
        print(f"\nPassages with < 10 good variants ({len(bad_passages)}):")
        for pid in bad_passages[:20]:
            d = per_passage[pid]
            print(f"  {pid}: {d['good']}/{d['n_variants']} good")

    if stats["ok"] / max(stats["total"], 1) < 0.50:
        print("\n*** CRITICAL: < 50% of outputs are usable Syriac. ***")
        print("*** Try the two-step Coptic→English→Syriac fallback. ***")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"stats": stats, "per_passage": per_passage}, f,
                   indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
