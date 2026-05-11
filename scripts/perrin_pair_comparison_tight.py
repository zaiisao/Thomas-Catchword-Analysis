#!/usr/bin/env python3
"""
Tighter Perrin pair-by-pair comparison.

Improvements over `perrin_pair_comparison.py` (FINDINGS outstanding item #2):
  1. **Pool all 10 Gemini variants** instead of just variant 0. A lemma
     counts as "canonical" if it appears as a participating lemma at the
     same boundary in ANY of the 10 stochastic Gemini retroversions.
  2. **Match at SEDRA root level**, not just consonantal skeleton. Two
     surface forms that map to the same SEDRA root_id count as canonical
     even if their skeletons differ (e.g., ܡܠܟܐ "king" and ܡܠܟܘܬܐ
     "kingdom" both root ܡܠܟ).

The original 22% canonical / 78% Perrin-specific result used variant-0
only + skeleton matching. This script lifts both restrictions to give the
upper-bound canonical fraction.

Output: data/processed/perrin_catchwords/pair_comparison_tight.json
        data/processed/perrin_catchwords/comparison_summary_tight.txt
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
SEDRA_LIST = ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
LEMMA_TO_ROOT = ROOT / "data" / "processed" / "syriac_lemma_to_root.json"

OUT_PAIR = ROOT / "data" / "processed" / "perrin_catchwords" / "pair_comparison_tight.json"
OUT_SUM = ROOT / "data" / "processed" / "perrin_catchwords" / "comparison_summary_tight.txt"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0
N_VARIANTS = 10
LOGION_ORDER = ["Prologue"] + [str(i) for i in range(1, 115)]

SYRIAC_RE = re.compile(r"[܀-ݏ]")
COMMENTARY_MARKERS = ("translation", "note:", "meaning:", "literally:",
                        "this translates", "the syriac", "here is",
                        "coptic text", "english:", "english translation")


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def consonantal(text: str) -> str:
    if not text: return ""
    t = strip_voc(text)
    t = re.sub(r"[^܀-ݏ]", "", t)
    t = re.sub(r"[܀-܏]", "", t)
    return t


def tokenize_syriac(text: str) -> list[str]:
    cleaned = re.sub(r"[^܀-ݏ\s]", "", text)
    cleaned = strip_voc(cleaned)
    cleaned = re.sub(r"[܀-܏]", " ", cleaned)
    return [w for w in cleaned.split() if w]


def is_usable_syriac(text: str) -> bool:
    if not text or not text.strip(): return False
    syr = SYRIAC_RE.findall(text)
    n_syr = len(syr)
    total = len(re.sub(r"\s", "", text))
    if total == 0 or n_syr / total < 0.5: return False
    low = text.lower()
    return all(m not in low for m in COMMENTARY_MARKERS) and n_syr >= 5


def load_sedra_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    with SEDRA_LIST.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4: continue
            unp, lem = parts[1].strip(), parts[3].strip()
            if unp and lem and unp not in out:
                out[unp] = lem
    return out


def load_lemma_to_root() -> dict[str, str]:
    """lemma → root_id"""
    d = json.loads(LEMMA_TO_ROOT.read_text(encoding="utf-8"))
    return {lemma: info.get("root_id") for lemma, info in d.items()}


def make_tokens(text: str, sedra: dict[str, str]) -> list[dict]:
    return [{"form": t, "lemma": sedra.get(t, t), "parse": "MS-EMP"}
            for t in tokenize_syriac(text)]


def load_all_variants() -> dict[int, list[list[dict]]]:
    """{logion_int: [tokens_v0, tokens_v1, ..., tokens_v9]} — all usable variants."""
    sedra = load_sedra_lookup()
    out: dict[int, list[list[dict]]] = {}
    for path in sorted(LLM_DIR.glob("logion_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("is_control"): continue
        ln = d.get("logion_number")
        if ln is None: continue
        variants_toks = []
        for v in d.get("variants", []):
            if not v.get("success"): continue
            txt = v.get("syriac_text", "")
            if is_usable_syriac(txt):
                variants_toks.append(make_tokens(txt, sedra))
        out[int(ln)] = variants_toks
    return out


def compute_blocked_pool(all_variants, filter_pct: float) -> set[str]:
    """Block lemmas that appear in >filter_pct% of logia (across pooled variants)."""
    n_logia = len(all_variants)
    cutoff = filter_pct * n_logia / 100.0
    cnt: Counter = Counter()
    for variants_toks in all_variants.values():
        # Union across this logion's variants
        lemmas_here = set()
        for ts in variants_toks:
            for t in ts:
                lemmas_here.add(t["lemma"])
        for lem in lemmas_here:
            cnt[lem] += 1
    return {lem for lem, c in cnt.items() if c > cutoff}


def pooled_catchwords_at_boundary(all_variants, ln_a: int, ln_b: int,
                                       blocked: set[str], det):
    """Across all 10 variants of (ln_a, ln_b), collect the UNION of detected
    catchword lemmas on each side. Returns (lemmas_a, lemmas_b)."""
    union_a, union_b = set(), set()
    for ts_a in all_variants.get(ln_a, []):
        for ts_b in all_variants.get(ln_b, []):
            toks_a = [t for t in ts_a if t["lemma"] not in blocked]
            toks_b = [t for t in ts_b if t["lemma"] not in blocked]
            if not toks_a or not toks_b: continue
            for cw in det.detect(toks_a, toks_b):
                union_a.add(cw.token_a["lemma"])
                union_b.add(cw.token_b["lemma"])
    return union_a, union_b


def perrin_word_skel(word: str | None, translit: str | None = None):
    if word and any('܀' <= c <= 'ݏ' for c in word):
        return consonantal(word)
    src = word or translit
    if not src: return None
    src = src.lower()
    skel = re.sub(r"[aeiouyʼʽ‘’' \-_.]+", "", src)
    return skel or None


def perrin_word_root(word: str | None, lemma_to_root: dict[str, str]):
    """If we can match Perrin's Syriac word to a lemma in the SEDRA root map,
    return the root_id. Tries exact match first, then consonantal-skeleton."""
    if not word: return None
    # Exact (with vocalization stripped)
    stripped = strip_voc(word)
    if stripped in lemma_to_root:
        return lemma_to_root[stripped]
    # Consonantal skeleton match
    skel = consonantal(word)
    if skel in lemma_to_root:
        return lemma_to_root[skel]
    return None


def lemma_root_set(lemmas: set[str], lemma_to_root: dict[str, str]) -> set[str]:
    """Map each lemma to its root_id (if known); return the set of root_ids."""
    return {lemma_to_root[l] for l in lemmas if l in lemma_to_root}


def lemma_skel_set(lemmas: set[str]) -> set[str]:
    return {consonantal(l) for l in lemmas if l}


def boundary_to_indices(boundary: str):
    a, b = boundary.split("-", 1)
    def to_int(x: str):
        x = x.strip()
        if x.lower().startswith("prol"): return 0
        try: return int(x)
        except ValueError: return None
    return to_int(a), to_int(b)


def main():
    print("Loading all 10 Gemini variants per logion…")
    all_variants = load_all_variants()
    n_with_data = sum(1 for v in all_variants.values() if v)
    avg_var = sum(len(v) for v in all_variants.values()) / max(n_with_data, 1)
    print(f"  {n_with_data} logia, avg {avg_var:.1f} usable variants each")

    print("Loading SEDRA lemma→root map…")
    lemma_to_root = load_lemma_to_root()
    print(f"  {len(lemma_to_root)} entries")

    print(f"Computing blocked lemmas (filter_pct={FILTER_PCT}%)…")
    blocked = compute_blocked_pool(all_variants, FILTER_PCT)
    print(f"  {len(blocked)} blocked")

    det = CatchwordDetector("syriac", phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)

    print("Computing pooled catchwords per boundary (10 variants each)…")
    perrin_by_b = {b["boundary"]: b
                   for b in json.loads(PERRIN_BOUND.read_text(encoding="utf-8"))}

    rows = []
    overall = {"perrin_total": 0,
               "canonical_skel": 0, "canonical_root": 0, "canonical_either": 0,
               "perrin_specific_strict": 0}
    for boundary in perrin_by_b:
        ln_a, ln_b = boundary_to_indices(boundary)
        if ln_a is None or ln_b is None: continue
        union_a, union_b = pooled_catchwords_at_boundary(
            all_variants, ln_a, ln_b, blocked, det)
        skel_a = lemma_skel_set(union_a)
        skel_b = lemma_skel_set(union_b)
        root_a = lemma_root_set(union_a, lemma_to_root)
        root_b = lemma_root_set(union_b, lemma_to_root)
        p = perrin_by_b[boundary]

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

        per_word = []
        canon_skel = canon_root = canon_either = strict_spec = 0
        for side, words in (("a", p_words_a), ("b", p_words_b)):
            tgt_skel = skel_a if side == "a" else skel_b
            tgt_root = root_a if side == "a" else root_b
            for w, tr, en in words:
                sk = perrin_word_skel(w, tr)
                root = perrin_word_root(w, lemma_to_root)
                ok_skel = sk is not None and sk in tgt_skel
                ok_root = root is not None and root in tgt_root
                ok_either = ok_skel or ok_root
                per_word.append({
                    "side": side, "english": en, "syriac": w,
                    "translit": tr, "skel": sk, "root_id": root,
                    "matched_skel": ok_skel, "matched_root": ok_root,
                    "matched_either": ok_either,
                })
                if ok_skel: canon_skel += 1
                if ok_root: canon_root += 1
                if ok_either: canon_either += 1
                if not ok_either: strict_spec += 1
        total = len(p_words_a) + len(p_words_b)
        rows.append({
            "boundary": boundary,
            "perrin_total": total,
            "canonical_skel": canon_skel,
            "canonical_root": canon_root,
            "canonical_either": canon_either,
            "perrin_specific_strict": strict_spec,
            "details": per_word,
        })
        overall["perrin_total"] += total
        overall["canonical_skel"] += canon_skel
        overall["canonical_root"] += canon_root
        overall["canonical_either"] += canon_either
        overall["perrin_specific_strict"] += strict_spec

    OUT_PAIR.write_text(json.dumps({"per_boundary": rows, "overall": overall},
                                       ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"  → {OUT_PAIR}")

    p_total = overall["perrin_total"]
    cs = overall["canonical_skel"]
    cr = overall["canonical_root"]
    ce = overall["canonical_either"]
    ss = overall["perrin_specific_strict"]
    lines = [
        "Perrin Pair-by-Pair Comparison (TIGHT: 10 variants + SEDRA root)",
        "=" * 70,
        f"Perrin entries (Syriac, non-bracket):      {p_total}",
        "",
        f"Canonical match (skeleton, variant 0 only)       — historical 22.2%",
        f"Canonical match (skeleton, ALL 10 variants):     {cs}  ({100*cs/p_total:.1f}%)",
        f"Canonical match (SEDRA root, ALL 10 variants):   {cr}  ({100*cr/p_total:.1f}%)",
        f"Canonical match (either, ALL 10 variants):       {ce}  ({100*ce/p_total:.1f}%)",
        f"Perrin-specific (no match by either method):     {ss}  ({100*ss/p_total:.1f}%)",
        "",
        f"The ALL-VARIANTS + SEDRA-ROOT-aware canonical fraction is the",
        f"upper bound on what an unbiased frontier-LLM translator reproduces.",
        f"The Perrin-specific fraction ({100*ss/p_total:.1f}%) is the lower bound on",
        f"Williams' bias-critique purchase across the table.",
    ]
    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
