#!/usr/bin/env python3
"""
Round-trip Step 6 — per-pair catchword survival diagnostic.

For each corpus, identify the catchwords detected in the *original* Syriac
strophe pairs. Then trace each catchword through the round trip and report:
  - did the SAME pair (or a corresponding pair) emerge in the recovered Syriac?
  - broken down by link_type (semantic, etymological, phonological)

This shows whether catchword loss is from (a) lexical divergence — same word
fails to round-trip back to itself — or (b) noise — different words emerge as
catchwords, washing out the signal.

Inputs:
  data/processed/syriac_strophes.jsonl
  data/processed/roundtrip/coptic_intermediate.jsonl
  data/processed/lexical_mapping/syriac_to_coptic.jsonl     (reverse, for surf→cop)
  data/processed/lexical_mapping/coptic_to_syriac.jsonl     (forward MAP for round-trip target)
  data/external/sedra/peshitta_list.txt

Output:
  data/processed/roundtrip/pair_survival.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

ORIG_SYR  = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
COPTIC_INT = REPO_ROOT / "data" / "processed" / "roundtrip" / "coptic_intermediate.jsonl"
RMAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "syriac_to_coptic.jsonl"
LMAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
SEDRA = REPO_ROOT / "data" / "external" / "sedra" / "peshitta_list.txt"
OUT_JSON = REPO_ROOT / "data" / "processed" / "roundtrip" / "pair_survival.json"

PHON_THRESHOLD = 0.65
FILTER_PCT = 80.0

CONTENT_PREFIXES = ("MS-", "FS-", "MP-", "FP-", "CS-", "CP-",
                    "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",
                    "SHAPHEL", "ESTAPHAL", "ETHTAPHAL")


def is_content(parse):
    return any(parse.startswith(p) for p in CONTENT_PREFIXES) if parse else False


def load_sedra():
    table = {}
    with SEDRA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6: continue
            unp, lem, parse = parts[1].strip(), parts[3].strip(), parts[5].strip()
            if unp and lem: table[unp] = (lem, parse)
    return table


def load_rmap_top():
    """syriac_lemma → top-1 coptic_lemma (MAP)."""
    out = {}
    with RMAP.open() as f:
        for line in f:
            r = json.loads(line)
            cands = sorted(r["candidates"], key=lambda x: -x["prob"])
            if cands:
                out[r["syriac_lemma"]] = cands[0]["coptic_lemma"]
    return out


def load_lmap_top():
    """coptic_lemma → top-1 syriac_lemma (MAP forward)."""
    out = {}
    with LMAP.open() as f:
        for line in f:
            r = json.loads(line)
            cands = sorted(r["candidates"], key=lambda x: -x["prob"])
            if cands:
                out[r["coptic_lemma"]] = cands[0]["syriac_lemma"]
    return out


def get_orig_syr_with_lemmas(sedra):
    """Yield strophes as {(corpus, source_file, idx): [(surface, lemma, parse), ...]}"""
    by_key = defaultdict(list)
    with ORIG_SYR.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tokens = []
            for w in r["text_consonantal"].split():
                w = w.strip(":.,;!?·܀")
                if not w: continue
                entry = sedra.get(w)
                if entry is None:
                    tokens.append((w, w, ""))   # OOV: surface as lemma
                    continue
                lem, parse = entry
                tokens.append((w, lem, parse))
            by_key[(r["author"], r["source_file"])].append(
                (r.get("strophe_index", 0), tokens))
    out = {}
    for k, lst in by_key.items():
        lst.sort(key=lambda x: x[0])
        out[k] = [(idx, toks) for idx, toks in lst]
    return out


def main():
    print("Loading SEDRA…")
    sedra = load_sedra()
    print(f"  {len(sedra)} entries")
    print("Loading reverse map (Syr→Cop MAP)…")
    rmap = load_rmap_top()
    print(f"  {len(rmap)} entries")
    print("Loading forward map (Cop→Syr MAP)…")
    lmap = load_lmap_top()
    print(f"  {len(lmap)} entries")
    print("Lemmatizing original Syriac strophes…")
    orig = get_orig_syr_with_lemmas(sedra)
    print(f"  {len(orig)} works")

    # First pass: build the FULL map of {orig_lemma → recovered_lemma} via the
    # deterministic round-trip used in this diagnostic (Syriac MAP → Coptic
    # MAP → Syriac MAP). Apply it to every content lemma in the corpus.
    rt_lemma = {}
    for syr_lem in rmap:
        cop = rmap[syr_lem]
        rt = lmap.get(cop, cop)
        rt_lemma[syr_lem] = rt

    det = CatchwordDetector("syriac",
                              phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)

    per_corpus = defaultdict(lambda: {
        "orig_pairs_with_cw": 0,
        "by_link_type": defaultdict(lambda: {"orig": 0, "survived": 0}),
        "n_works": 0,
        "samples_survived": [],
        "samples_lost": [],
    })

    n_corpora = len({k[0] for k in orig.keys()})
    print(f"\nProcessing {n_corpora} corpora…\n")

    for (corpus, source_file), strophes in orig.items():
        d = per_corpus[corpus]
        d["n_works"] += 1
        for i in range(len(strophes) - 1):
            idx_a, toks_a = strophes[i]
            idx_b, toks_b = strophes[i + 1]
            # Restrict to content-word lemmas
            content_a = [(s, l) for s, l, p in toks_a if is_content(p) and l in rmap]
            content_b = [(s, l) for s, l, p in toks_b if is_content(p) and l in rmap]
            if not content_a or not content_b: continue

            # Original catchwords (lemma-pair-level)
            ta = [{"form": s, "lemma": l, "parse": "MS-EMP"}
                  for s, l in content_a]
            tb = [{"form": s, "lemma": l, "parse": "MS-EMP"}
                  for s, l in content_b]
            orig_cws = det.detect(ta, tb)
            if not orig_cws: continue
            d["orig_pairs_with_cw"] += 1

            # Round-trip both sides
            rt_a = [(s, rt_lemma.get(l, l)) for s, l in content_a]
            rt_b = [(s, rt_lemma.get(l, l)) for s, l in content_b]
            ta_rt = [{"form": s, "lemma": l, "parse": "MS-EMP"} for s, l in rt_a]
            tb_rt = [{"form": s, "lemma": l, "parse": "MS-EMP"} for s, l in rt_b]
            rt_cws = det.detect(ta_rt, tb_rt)

            # For each original catchword, check whether its specific
            # round-tripped lemma pair appears in the recovered catchwords
            rt_pair_set = {(c.token_a["lemma"], c.token_b["lemma"], c.link_type)
                           for c in rt_cws}
            rt_lemma_a_to_orig = {rt_lemma.get(l, l): l for s, l in content_a}
            rt_lemma_b_to_orig = {rt_lemma.get(l, l): l for s, l in content_b}

            for cw in orig_cws:
                la = cw.token_a["lemma"]; lb = cw.token_b["lemma"]
                lt = cw.link_type
                d["by_link_type"][lt]["orig"] += 1
                rta = rt_lemma.get(la, la)
                rtb = rt_lemma.get(lb, lb)
                # Check whether the round-tripped lemma pair appears as
                # ANY catchword in the recovered detection (not just the
                # same link type — phonological-→-semantic flips count as
                # survival, since the connection re-emerged)
                survived = any((rt_pair_set_member[0] == rta
                                 and rt_pair_set_member[1] == rtb)
                                for rt_pair_set_member in rt_pair_set)
                if survived:
                    d["by_link_type"][lt]["survived"] += 1
                    if len(d["samples_survived"]) < 8:
                        d["samples_survived"].append({
                            "orig": (la, lb), "rt": (rta, rtb),
                            "link": lt,
                            "ref": f"{source_file}:{idx_a}-{idx_b}",
                        })
                else:
                    if len(d["samples_lost"]) < 8:
                        d["samples_lost"].append({
                            "orig": (la, lb), "rt": (rta, rtb),
                            "link": lt,
                            "ref": f"{source_file}:{idx_a}-{idx_b}",
                        })

    # Print + save
    print("=" * 80)
    print("PER-PAIR SURVIVAL THROUGH ROUND TRIP")
    print(f"  (deterministic MAP/MAP round-trip on content-word lemmas only)")
    print("=" * 80)
    for corpus, d in sorted(per_corpus.items()):
        print(f"\n{corpus}:")
        for lt in ("semantic", "etymological", "phonological"):
            o = d["by_link_type"][lt]["orig"]
            s = d["by_link_type"][lt]["survived"]
            if o == 0:
                print(f"  {lt:<14s}: (no original instances)")
            else:
                print(f"  {lt:<14s}: {s}/{o} survived ({100*s/o:.1f}%)")
        total_o = sum(d["by_link_type"][lt]["orig"] for lt in d["by_link_type"])
        total_s = sum(d["by_link_type"][lt]["survived"] for lt in d["by_link_type"])
        if total_o:
            print(f"  TOTAL         : {total_s}/{total_o} ({100*total_s/total_o:.1f}%)")

    out_data = {
        "config": {"phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT},
        "per_corpus": {
            c: {
                "orig_pairs_with_cw": d["orig_pairs_with_cw"],
                "n_works": d["n_works"],
                "by_link_type": {
                    lt: dict(stats) for lt, stats in d["by_link_type"].items()
                },
                "samples_survived": d["samples_survived"],
                "samples_lost": d["samples_lost"],
            } for c, d in per_corpus.items()
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
