#!/usr/bin/env python3
"""
Round-trip Step 2 — translate the four Syriac literary corpora (Ephrem,
Narsai, Jacob, Odes of Solomon) to Coptic via the reverse lexical map
(MAP / argmax — single deterministic Coptic intermediate per strophe).

Pipeline per strophe:
  consonantal-stripped surface words
  → SEDRA surface→lemma+parse lookup
  → drop function-word lemmas (parse-based)
  → reverse_map[lemma] → top-1 Coptic lemma
  → drop reverse-map OOV (very rare: 0.6%)

Inputs:
  data/processed/syriac_strophes.jsonl
  data/external/sedra/peshitta_list.txt
  data/processed/lexical_mapping/syriac_to_coptic.jsonl

Output:
  data/processed/roundtrip/coptic_intermediate.jsonl
    {"author", "source_file", "strophe_index", "syriac_content_lemmas",
     "coptic_lemmas", "n_oov_sedra", "n_oov_reversemap"}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
SEDRA = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
RMAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "syriac_to_coptic.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "roundtrip" / "coptic_intermediate.jsonl"
OOV_REPORT = REPO_ROOT / "data" / "processed" / "roundtrip" / "translation_oov.json"

CONTENT_PREFIXES = ("MS-", "FS-", "MP-", "FP-", "CS-", "CP-",
                    "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",
                    "SHAPHEL", "ESTAPHAL", "ETHTAPHAL")


def is_content(parse: str) -> bool:
    return any(parse.startswith(p) for p in CONTENT_PREFIXES) if parse else False


def load_sedra_table():
    """Surface → (lemma, parse). The file is TAB-separated:
       id, unpointed, pointed, lemma, gloss, parse"""
    table = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            unp = parts[1].strip()
            lem = parts[3].strip()
            parse = parts[5].strip()
            if unp and lem:
                table[unp] = (lem, parse)
    return table


def load_reverse_map():
    """syriac_lemma → top-1 coptic_lemma (MAP)."""
    map_top = {}
    with RMAP.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cands = sorted(r["candidates"], key=lambda x: -x["prob"])
            if cands:
                map_top[r["syriac_lemma"]] = cands[0]["coptic_lemma"]
    return map_top


def load_strophes():
    by_corpus = defaultdict(list)
    with STROPHES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_corpus[r.get("author", "?")].append(r)
    for k in by_corpus:
        by_corpus[k].sort(key=lambda r: (r.get("source_file", ""),
                                           r.get("strophe_index", 0)))
    return by_corpus


def main():
    print("Loading SEDRA surface table…")
    sedra = load_sedra_table()
    print(f"  {len(sedra)} entries")

    print("Loading reverse map (Syriac→Coptic, MAP top-1)…")
    rmap = load_reverse_map()
    print(f"  {len(rmap)} entries")

    print("Loading strophes…")
    by_corpus = load_strophes()
    for c, sl in by_corpus.items():
        print(f"  {c}: {len(sl)} strophes")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"strophes": 0, "n_words": 0, "n_content": 0,
                                   "n_oov_sedra": 0, "n_oov_rmap": 0,
                                   "n_translated": 0})
    out_count = 0
    with OUT.open("w", encoding="utf-8") as out:
        for corpus, strophes in by_corpus.items():
            for s in strophes:
                syr_lemmas_content = []
                cop_lemmas = []
                n_words = n_oov_sedra = n_oov_rmap = 0
                for w in s["text_consonantal"].split():
                    w = w.strip(":.,;!?·܀")
                    if not w:
                        continue
                    n_words += 1
                    entry = sedra.get(w)
                    if entry is None:
                        n_oov_sedra += 1
                        continue
                    lem, parse = entry
                    if not is_content(parse):
                        continue
                    cop = rmap.get(lem)
                    if cop is None:
                        n_oov_rmap += 1
                        continue
                    syr_lemmas_content.append(lem)
                    cop_lemmas.append(cop)
                rec = {
                    "author": s.get("author"),
                    "source_file": s.get("source_file"),
                    "strophe_index": s.get("strophe_index"),
                    "syriac_text_consonantal": s.get("text_consonantal"),
                    "syriac_content_lemmas": syr_lemmas_content,
                    "coptic_lemmas": cop_lemmas,
                    "n_words": n_words,
                    "n_oov_sedra": n_oov_sedra,
                    "n_oov_rmap": n_oov_rmap,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_count += 1
                d = stats[corpus]
                d["strophes"] += 1
                d["n_words"] += n_words
                d["n_content"] += len(syr_lemmas_content)
                d["n_oov_sedra"] += n_oov_sedra
                d["n_oov_rmap"] += n_oov_rmap
                d["n_translated"] += len(cop_lemmas)

    print()
    print(f"Wrote {out_count} strophes → {OUT}")
    print()
    print("=== Per-corpus translation statistics ===")
    print(f"{'Author':<10s} {'Strophes':>8s} {'Words':>7s} {'Content':>8s} "
          f"{'OOV(SED)%':>10s} {'OOV(RMap)%':>11s} {'Translated':>11s}")
    for corpus, d in sorted(stats.items()):
        sed_pct = 100 * d["n_oov_sedra"] / max(d["n_words"], 1)
        rmap_pct = 100 * d["n_oov_rmap"] / max(d["n_words"], 1)
        print(f"{corpus:<10s} {d['strophes']:>8d} {d['n_words']:>7d} "
              f"{d['n_content']:>8d} {sed_pct:>9.1f}% "
              f"{rmap_pct:>10.1f}% {d['n_translated']:>11d}")

    with OOV_REPORT.open("w") as f:
        json.dump({"per_corpus": dict(stats)}, f, indent=2)
    print(f"\nWrote OOV report → {OOV_REPORT}")


if __name__ == "__main__":
    main()
