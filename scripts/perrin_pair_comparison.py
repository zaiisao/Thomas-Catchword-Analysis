#!/usr/bin/env python3
"""
Steps 3 & 4: align counting methodology and run Perrin pair-by-pair comparison.

Approach:
  1. Re-run the Phase-1 catchword detector on the canonical (variant 0)
     Phase 2B Gemini Syriac translations and store the detected lemma pairs
     for every (logion i, logion i+1) boundary.
  2. Convert our pair-counting output to Perrin's word-counting (count unique
     content lemmas per logion that participate in any cross-boundary link).
  3. For each Perrin Syriac catchword (502 entries), normalise to consonantal
     skeleton and check whether the same skeleton appears as a participating
     lemma at the same boundary in our detections.

Inputs:
  data/processed/perrin_catchwords/perrin_per_boundary.json
  data/processed/llm_translations/logion_*.json     (Gemini canonical text)
  data/external/sedra/peshitta_list.txt             (lemma lookup)

Outputs:
  data/processed/perrin_catchwords/our_gemini_per_boundary.json
  data/processed/perrin_catchwords/pair_comparison.json
  data/processed/perrin_catchwords/comparison_summary.txt
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

PERRIN_BOUND = ROOT / "data" / "processed" / "perrin_catchwords" / "perrin_per_boundary.json"
LLM_DIR = ROOT / "data" / "processed" / "llm_translations"
SEDRA = ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"

OUR_OUT = ROOT / "data" / "processed" / "perrin_catchwords" / "our_gemini_per_boundary.json"
PAIR_OUT = ROOT / "data" / "processed" / "perrin_catchwords" / "pair_comparison.json"
SUMMARY_OUT = ROOT / "data" / "processed" / "perrin_catchwords" / "comparison_summary.txt"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0

LOGION_ORDER = ["Prologue"] + [str(i) for i in range(1, 115)]

SYRIAC_RE = re.compile(r"[܀-ݏ]")
COMMENTARY_MARKERS = ("translation", "note:", "meaning:", "literally:",
                       "this translates", "the syriac", "here is", "coptic text",
                       "english:", "english translation")


def is_usable_syriac(text: str) -> bool:
    if not text or not text.strip():
        return False
    syr = SYRIAC_RE.findall(text)
    n_syr = len(syr)
    total = len(re.sub(r"\s", "", text))
    if total == 0 or n_syr / total < 0.5:
        return False
    low = text.lower()
    for m in COMMENTARY_MARKERS:
        if m in low:
            return False
    return n_syr >= 5


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def consonantal(text: str) -> str:
    """Strip vocalization + non-Syriac chars; collapse to consonants only."""
    if not text:
        return ""
    t = strip_voc(text)
    t = re.sub(r"[^܀-ݏ]", "", t)
    t = re.sub(r"[܀-܏]", "", t)  # punctuation block
    return t


def tokenize_syriac(text: str) -> list[str]:
    cleaned = re.sub(r"[^܀-ݏ\s]", "", text)
    cleaned = strip_voc(cleaned)
    cleaned = re.sub(r"[܀-܏]", " ", cleaned)
    return [w for w in cleaned.split() if w]


def load_sedra_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            unp, lem = parts[1].strip(), parts[3].strip()
            if unp and lem and unp not in out:
                out[unp] = lem
    return out


def make_tokens(text: str, sedra: dict[str, str]) -> list[dict]:
    return [{"form": t, "lemma": sedra.get(t, t), "parse": "MS-EMP"}
            for t in tokenize_syriac(text)]


def load_gemini_canonical() -> dict[int, list[dict]]:
    """Return {logion_int: tokens} for variant-0 of each Thomas logion."""
    out: dict[int, list[dict]] = {}
    sedra = load_sedra_lookup()
    for path in sorted(LLM_DIR.glob("logion_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("is_control"):
            continue
        ln = d.get("logion_number")
        if ln is None:
            continue
        # Pick first usable variant as "canonical"
        toks: list[dict] = []
        for v in d.get("variants", []):
            if not v.get("success"):
                continue
            txt = v.get("syriac_text", "")
            if is_usable_syriac(txt):
                toks = make_tokens(txt, sedra)
                break
        out[int(ln)] = toks
    return out


def compute_blocked(strophes: list[list[dict]], filter_pct: float) -> set[str]:
    n = len(strophes)
    cutoff = filter_pct * n / 100.0
    cnt: Counter[str] = Counter()
    for s in strophes:
        for lem in {t["lemma"] for t in s}:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def boundary_to_logion_indices(boundary: str) -> tuple[int | None, int | None]:
    """Map "Prologue-1" → (0, 1), "1-2" → (1, 2), etc.

    Phase 2B uses logion_000 = Prologue, logion_001 = GT 1, ..., logion_114 = GT 114.
    """
    a, b = boundary.split("-", 1)
    def to_int(x: str) -> int | None:
        x = x.strip()
        if x.lower().startswith("prol"):
            return 0
        try:
            return int(x)
        except ValueError:
            return None
    return to_int(a), to_int(b)


def compute_our_catchwords(canonical: dict[int, list[dict]],
                            blocked: set[str]) -> dict[str, dict]:
    """For each boundary, return {boundary: {lemmas_a: [...], lemmas_b: [...], pairs: [...]}}.

    lemmas_a / lemmas_b are the unique participating lemmas per side.
    """
    det = CatchwordDetector("syriac",
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    out: dict[str, dict] = {}
    for i in range(len(LOGION_ORDER) - 1):
        ln_a = i        # 0=Prologue, 1=GT1, ...
        ln_b = i + 1
        toks_a = [t for t in canonical.get(ln_a, []) if t["lemma"] not in blocked]
        toks_b = [t for t in canonical.get(ln_b, []) if t["lemma"] not in blocked]
        boundary = f"{LOGION_ORDER[i]}-{LOGION_ORDER[i+1]}"
        if not toks_a or not toks_b:
            out[boundary] = {"lemmas_a": [], "lemmas_b": [], "pairs": []}
            continue
        cws = det.detect(toks_a, toks_b)
        lemmas_a = sorted({c.token_a["lemma"] for c in cws})
        lemmas_b = sorted({c.token_b["lemma"] for c in cws})
        pairs = [{
            "lemma_a": c.token_a["lemma"],
            "lemma_b": c.token_b["lemma"],
            "link_type": c.link_type,
            "score": c.score,
        } for c in cws]
        out[boundary] = {"lemmas_a": lemmas_a, "lemmas_b": lemmas_b,
                         "pairs": pairs}
    return out


def lemma_skel_set(lemmas: list[str]) -> set[str]:
    """Map each lemma to its consonantal skeleton; return the set of skeletons."""
    return {consonantal(l) for l in lemmas if l}


def perrin_word_skel(word: str | None, translit: str | None = None) -> str | None:
    """Get a consonantal skeleton from Perrin's Syriac word.

    If `word` is Syriac Unicode, strip vocalization and return.
    If it's a Latin transliteration, drop vowels and return uppercase consonants.
    """
    if word and any('܀' <= c <= 'ݏ' for c in word):
        return consonantal(word)
    src = word or translit
    if not src:
        return None
    # Treat as transliteration: keep consonants only
    src = src.lower()
    # Drop typical Latin vowels
    skel = re.sub(r"[aeiouyʼʽ‘’' \-_.]+", "", src)
    return skel or None


def main() -> None:
    print("Loading Gemini canonical translations…")
    canonical = load_gemini_canonical()
    print(f"  {len(canonical)} logia loaded")

    print("Computing blocked lemma set (filter_pct={:.0f}%)…".format(FILTER_PCT))
    strophes = [canonical.get(i, []) for i in range(len(LOGION_ORDER))]
    blocked = compute_blocked(strophes, FILTER_PCT)
    print(f"  {len(blocked)} blocked lemmas")

    print("Detecting our catchwords on Gemini canonical text…")
    our = compute_our_catchwords(canonical, blocked)
    OUR_OUT.write_text(json.dumps(our, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"  → {OUR_OUT}")

    # Perrin per-boundary
    perrin_by_b = {b["boundary"]: b
                   for b in json.loads(PERRIN_BOUND.read_text(encoding="utf-8"))}

    # ---- Compare ----
    rows = []
    overall = {"perrin_total": 0, "canonical_match": 0, "perrin_specific": 0}
    perrin_word_count_per_boundary = []
    our_word_count_per_boundary = []

    for boundary, p in perrin_by_b.items():
        skels_a = lemma_skel_set(our.get(boundary, {}).get("lemmas_a", []))
        skels_b = lemma_skel_set(our.get(boundary, {}).get("lemmas_b", []))

        # Perrin words on side A and B
        p_words_a = list(zip(
            p.get("syriac_a", []),
            p.get("syriac_translit_a", [None] * len(p.get("syriac_a", []))),
            p.get("english_a", [None] * len(p.get("syriac_a", []))),
        ))
        p_words_b = list(zip(
            p.get("syriac_b", []),
            p.get("syriac_translit_b", [None] * len(p.get("syriac_b", []))),
            p.get("english_b", [None] * len(p.get("syriac_b", []))),
        ))

        per_word_match = []
        canon = 0
        spec = 0
        for w, tr, en in p_words_a:
            sk = perrin_word_skel(w, tr)
            ok = sk is not None and sk in skels_a
            per_word_match.append({"side": "a", "english": en, "syriac": w,
                                   "translit": tr, "skel": sk, "matched": ok})
            if ok: canon += 1
            else:  spec += 1
        for w, tr, en in p_words_b:
            sk = perrin_word_skel(w, tr)
            ok = sk is not None and sk in skels_b
            per_word_match.append({"side": "b", "english": en, "syriac": w,
                                   "translit": tr, "skel": sk, "matched": ok})
            if ok: canon += 1
            else:  spec += 1

        total = canon + spec
        perrin_word_count_per_boundary.append(total)
        # Our Perrin-style count: |unique participating skel A| + |...B|
        our_word_count_per_boundary.append(len(skels_a) + len(skels_b))

        rows.append({
            "boundary": boundary,
            "perrin_total":     total,
            "perrin_canonical": canon,
            "perrin_specific":  spec,
            "our_perrin_count": len(skels_a) + len(skels_b),
            "our_pair_count":   len(our.get(boundary, {}).get("pairs", [])),
            "details": per_word_match,
        })
        overall["perrin_total"] += total
        overall["canonical_match"] += canon
        overall["perrin_specific"] += spec

    PAIR_OUT.write_text(json.dumps({"per_boundary": rows, "overall": overall},
                                    ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  → {PAIR_OUT}")

    # ---- Summary ----
    p_total = overall["perrin_total"]
    canon = overall["canonical_match"]
    spec = overall["perrin_specific"]
    pct_canon = 100.0 * canon / p_total if p_total else 0.0

    lines = [
        "Perrin Pair-by-Pair Comparison vs Phase 2B Gemini",
        "=" * 60,
        f"Perrin entries (Syriac, non-bracket):     {p_total}",
        f"Detected as canonical (skel match):       {canon}  ({pct_canon:.1f}%)",
        f"Perrin-specific (no skel match):          {spec}  ({100-pct_canon:.1f}%)",
        "",
        f"Sum of Perrin per-boundary counts:        {sum(perrin_word_count_per_boundary)}",
        f"Sum of our (Perrin-style) word counts:    {sum(our_word_count_per_boundary)}",
        f"Ratio Perrin / ours (whole table):        "
        f"{sum(perrin_word_count_per_boundary) / max(1, sum(our_word_count_per_boundary)):.2f}x",
        "",
        "Boundaries where Perrin scores ≥ 5 catchwords but ours scores 0:",
    ]
    rows_sorted = sorted(rows, key=lambda r: r["perrin_total"], reverse=True)
    for r in rows_sorted:
        if r["perrin_total"] >= 5 and r["our_perrin_count"] == 0:
            lines.append(f"  {r['boundary']}: Perrin={r['perrin_total']} ours=0")
    lines.append("")
    lines.append("Top 10 boundaries by Perrin total:")
    for r in rows_sorted[:10]:
        lines.append(f"  {r['boundary']}: Perrin={r['perrin_total']} "
                     f"ours_perrin_style={r['our_perrin_count']} "
                     f"ours_pairs={r['our_pair_count']} "
                     f"canonical={r['perrin_canonical']}")
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{Path(SUMMARY_OUT).name}:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
