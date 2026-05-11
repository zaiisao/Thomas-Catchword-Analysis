#!/usr/bin/env python3
"""
Merge per-batch Perrin table JSONs into one canonical file and validate.

Inputs:
  data/processed/perrin_catchwords/batches/batch_*.json   (from digitization agents)

Output:
  data/processed/perrin_catchwords/perrin_table_full.json (sorted, entry_id assigned)
  data/processed/perrin_catchwords/validation_report.txt
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "data" / "processed" / "perrin_catchwords" / "batches"
OUT = ROOT / "data" / "processed" / "perrin_catchwords" / "perrin_table_full.json"
REPORT = ROOT / "data" / "processed" / "perrin_catchwords" / "validation_report.txt"

EXPECTED_TOTALS = {"coptic": 271, "greek": 261, "syriac": 502}

LOGION_ORDER = ["Prologue"] + [str(i) for i in range(1, 115)]
LOGION_RANK = {l: i for i, l in enumerate(LOGION_ORDER)}


def _logion_rank(s: str) -> int:
    """Sort key tolerant of stray whitespace / case."""
    if not s:
        return 999
    s = str(s).strip()
    if s.lower().startswith("prol"):
        return 0
    # Pull leading integer
    m = re.match(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return 999


def main() -> None:
    if not BATCH_DIR.exists():
        raise SystemExit(f"missing batch directory: {BATCH_DIR}")
    batch_files = sorted(BATCH_DIR.glob("batch_*.json"))
    if not batch_files:
        raise SystemExit("no batch files found")

    all_entries: list[dict] = []
    per_batch_count: dict[str, int] = {}
    for bf in batch_files:
        try:
            data = json.loads(bf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN {bf.name}: failed to parse — {e}")
            continue
        if not isinstance(data, list):
            print(f"  WARN {bf.name}: not a list (got {type(data).__name__})")
            continue
        per_batch_count[bf.name] = len(data)
        all_entries.extend(data)

    # Sort by (page, then a within-page order). We rely on agents preserving
    # the visual top-to-bottom order. Use a stable sort keyed on page only,
    # so within-page ordering survives.
    all_entries.sort(key=lambda e: (int(e.get("page", 9999)),))

    # Assign sequential entry_id
    for i, e in enumerate(all_entries, start=1):
        e["entry_id"] = i

    # ---- Validation ----
    lines: list[str] = []
    lines.append("Perrin Table Digitization — Validation Report")
    lines.append("=" * 60)
    lines.append(f"Batches loaded: {len(per_batch_count)}")
    for n, c in per_batch_count.items():
        lines.append(f"  {n}: {c} entries")
    lines.append(f"Total entries: {len(all_entries)}")

    # Counts of non-bracket, non-null per language
    coptic_count = sum(1 for e in all_entries
                       if not e.get("is_bracket") and e.get("coptic_word"))
    greek_count = sum(1 for e in all_entries
                      if not e.get("is_bracket") and e.get("greek_word"))
    syriac_count = sum(1 for e in all_entries
                       if not e.get("is_bracket") and e.get("syriac_word"))

    def maxc(field: str) -> int:
        vals = [int(e[field]) for e in all_entries
                if e.get(field) is not None and not e.get("is_bracket")]
        return max(vals) if vals else 0

    last_coptic = maxc("coptic_cumulative")
    last_greek = maxc("greek_cumulative")
    last_syriac = maxc("syriac_cumulative")

    lines.append("")
    lines.append(f"Counts of non-bracket, non-null entries per language:")
    lines.append(f"  Coptic: {coptic_count}  (expected {EXPECTED_TOTALS['coptic']}, "
                 f"max cumulative = {last_coptic})")
    lines.append(f"  Greek:  {greek_count}   (expected {EXPECTED_TOTALS['greek']}, "
                 f"max cumulative = {last_greek})")
    lines.append(f"  Syriac: {syriac_count}  (expected {EXPECTED_TOTALS['syriac']}, "
                 f"max cumulative = {last_syriac})")

    # Logion coverage
    logia = sorted({str(e.get("logion")) for e in all_entries if e.get("logion")},
                   key=_logion_rank)
    lines.append("")
    lines.append(f"Logia covered ({len(logia)}): "
                 f"first={logia[0] if logia else None} last={logia[-1] if logia else None}")
    missing = [l for l in LOGION_ORDER if l not in logia]
    lines.append(f"Missing logia: {missing if missing else 'none'}")

    # Monotonicity check on cumulative counts
    def check_monotone(field: str) -> list[str]:
        problems = []
        prev = 0
        for e in all_entries:
            if e.get("is_bracket"):
                continue
            v = e.get(field)
            if v is None:
                continue
            if v < prev:
                problems.append(
                    f"  entry {e.get('entry_id')} (page {e.get('page')}, "
                    f"logion {e.get('logion')}): {field}={v} < prev={prev}"
                )
            prev = v
        return problems

    lines.append("")
    lines.append("Monotonicity check:")
    for fld in ("coptic_cumulative", "greek_cumulative", "syriac_cumulative"):
        probs = check_monotone(fld)
        lines.append(f"  {fld}: {'OK' if not probs else f'{len(probs)} regressions'}")
        for p in probs[:5]:
            lines.append(p)

    # Overall pass/fail
    ok = (last_coptic == EXPECTED_TOTALS["coptic"] and
          last_greek == EXPECTED_TOTALS["greek"] and
          last_syriac == EXPECTED_TOTALS["syriac"])
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if ok else 'NEEDS REVIEW'}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    OUT.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
