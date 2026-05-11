#!/usr/bin/env python3
"""
Step 2: derive per-boundary catchword counts from Perrin's full table.

For each consecutive logion pair (Prologue-1, 1-2, ..., 113-114) compute:
- forward catchwords (in left logion, direction in {forward, both})
- backward catchwords (in right logion, direction in {backward, both})
- the Perrin "word count" (forward + backward)
- per-language counts (Coptic / Greek / Syriac)
- the actual word lists (for later pair comparison)

Input:  data/processed/perrin_catchwords/perrin_table_full.json
Output: data/processed/perrin_catchwords/perrin_per_boundary.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "perrin_catchwords" / "perrin_table_full.json"
OUT = ROOT / "data" / "processed" / "perrin_catchwords" / "perrin_per_boundary.json"

LOGION_ORDER = ["Prologue"] + [str(i) for i in range(1, 115)]


def normalize_logion(label) -> str:
    """Map "108.1", "113.4", " 12 ", "Prologue" → canonical logion id."""
    s = str(label or "").strip()
    if not s:
        return ""
    if s.lower().startswith("prol"):
        return "Prologue"
    m = re.match(r"(\d+)", s)
    return m.group(1) if m else s


def parse_targets(links: str | None) -> set[str]:
    """Extract canonical logion IDs from a links_to_logion string.

    Examples:
      "1"            → {"1"}
      "2.4"          → {"2"}
      "2.1, 2"       → {"2"}            (both refer to logion 2)
      "13.1, 4"      → {"13"}
      "Cop 18.1/20.4; Gk, Syr: 20.4" → {"18", "20"}
    """
    out: set[str] = set()
    if not links:
        return out
    for tok in re.findall(r"\d+", str(links)):
        out.add(tok)
    return out


def main() -> None:
    entries = json.loads(IN.read_text(encoding="utf-8"))

    by_log: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("is_bracket"):
            continue
        by_log[normalize_logion(e.get("logion"))].append(e)

    boundaries = []
    for i in range(len(LOGION_ORDER) - 1):
        a, b = LOGION_ORDER[i], LOGION_ORDER[i + 1]
        # Side-A (forward): entries IN logion `a` whose link target is `b`
        fwd: list[dict] = []
        for e in by_log.get(a, []):
            tgts = parse_targets(e.get("links_to_logion"))
            # If links_to_logion is missing, fall back to direction logic
            if not tgts and e.get("direction") in ("forward", "both"):
                fwd.append(e)
            elif b in tgts and e.get("direction") in ("forward", "both"):
                fwd.append(e)
        # Side-B (backward): entries IN logion `b` whose link target is `a`
        bwd: list[dict] = []
        for e in by_log.get(b, []):
            tgts = parse_targets(e.get("links_to_logion"))
            if not tgts and e.get("direction") in ("backward", "both"):
                bwd.append(e)
            elif a in tgts and e.get("direction") in ("backward", "both"):
                bwd.append(e)

        def words(rows: list[dict], field: str) -> list:
            out = []
            for r in rows:
                w = r.get(field)
                if w is None:
                    continue
                out.append(w)
            return out

        # Perrin's word counting: count entries with a Syriac word filled
        syr_fwd = [e for e in fwd if e.get("syriac_word")]
        syr_bwd = [e for e in bwd if e.get("syriac_word")]
        cop_fwd = [e for e in fwd if e.get("coptic_word")]
        cop_bwd = [e for e in bwd if e.get("coptic_word")]
        gr_fwd = [e for e in fwd if e.get("greek_word")]
        gr_bwd = [e for e in bwd if e.get("greek_word")]

        boundaries.append({
            "boundary": f"{a}-{b}",
            "logion_a": a,
            "logion_b": b,
            "perrin_syriac_fwd": len(syr_fwd),
            "perrin_syriac_bwd": len(syr_bwd),
            "perrin_syriac_total": len(syr_fwd) + len(syr_bwd),
            "perrin_coptic_total": len(cop_fwd) + len(cop_bwd),
            "perrin_greek_total":  len(gr_fwd) + len(gr_bwd),
            "english_a": [e.get("english_word") for e in fwd],
            "english_b": [e.get("english_word") for e in bwd],
            "syriac_a":  words(fwd, "syriac_word"),
            "syriac_b":  words(bwd, "syriac_word"),
            "syriac_translit_a": words(fwd, "syriac_translit"),
            "syriac_translit_b": words(bwd, "syriac_translit"),
            "coptic_a":  words(fwd, "coptic_word"),
            "coptic_b":  words(bwd, "coptic_word"),
            "greek_a":   words(fwd, "greek_word"),
            "greek_b":   words(bwd, "greek_word"),
        })

    OUT.write_text(json.dumps(boundaries, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    perrin_total_syr = sum(b["perrin_syriac_total"] for b in boundaries)
    perrin_total_cop = sum(b["perrin_coptic_total"] for b in boundaries)
    perrin_total_grk = sum(b["perrin_greek_total"]  for b in boundaries)
    print(f"Wrote {OUT}  ({len(boundaries)} boundaries)")
    print(f"  Sum across boundaries: Syriac={perrin_total_syr}, "
          f"Coptic={perrin_total_cop}, Greek={perrin_total_grk}")
    print("  (Should match expected 502/271/261)")


if __name__ == "__main__":
    main()
