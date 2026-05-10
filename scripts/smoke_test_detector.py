#!/usr/bin/env python3
"""
End-to-end smoke test of the catchword detector on real data.

Three scenarios:

  1. PESHITTA SANITY CHECK
     Run the detector on Peshitta Matt 3:10-11 (which contains nūrā "fire")
     vs Peshitta Matt 5:14-16 (which contains nuhrā "light"). Verifies the
     detector finds Perrin's most-cited catchword pair in actual Syriac text.

  2. THOMAS MAP-TRANSLATION (a deterministic precursor of the Monte Carlo)
     For each Coptic Thomas logion, take its content lemmas, look each one up
     in coptic_to_syriac.jsonl, and pick the highest-probability Syriac lemma.
     Run the detector on adjacent (Logion N, Logion N+1) pairs. Report total
     catchwords + how many appear in Perrin's JETS examples.

  3. CROSS-LANGUAGE BASELINE (Williams' constraint)
     Run the detector on the same Coptic Thomas logia using the Coptic profile
     (no translation). The detector logic is identical; only the language
     profile changes. Reports catchword count for direct comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

PESH_LEM = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
LEX_MAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


# ---------------------------------------------------------------------------
# Scenario 1: Peshitta NT sanity check
# ---------------------------------------------------------------------------

def scenario_peshitta_sanity():
    print("=" * 70)
    print("SCENARIO 1: Peshitta Matt 3:10-11 vs Matt 5:14-16")
    print("  Expect: detector finds the nūrā / nuhrā phonological link")
    print("=" * 70)

    pesh = load_jsonl(PESH_LEM)
    by_ref = {(r["book"], r["chapter"], r["verse"]): r for r in pesh}

    # "Fire" passage (John the Baptist)
    fire_tokens = []
    for v in (10, 11, 12):
        r = by_ref.get(("Matt", 3, v))
        if r:
            for t in r["tokens"]:
                fire_tokens.append({**t, "ref": f"Matt 3:{v}"})

    # "Light" passage (Sermon on the Mount)
    light_tokens = []
    for v in (14, 15, 16):
        r = by_ref.get(("Matt", 5, v))
        if r:
            for t in r["tokens"]:
                light_tokens.append({**t, "ref": f"Matt 5:{v}"})

    det = CatchwordDetector("syriac")
    catchwords = det.detect(fire_tokens, light_tokens)

    semantic = [c for c in catchwords if c.link_type == "semantic"]
    etymological = [c for c in catchwords if c.link_type == "etymological"]
    phonological = [c for c in catchwords if c.link_type == "phonological"]

    print(f"  Total: {len(catchwords)}  "
          f"(semantic={len(semantic)}, etym={len(etymological)}, phon={len(phonological)})")
    print()
    found_nura_nuhra = False
    for cw in catchwords:
        la = cw.token_a.get("lemma", "")
        lb = cw.token_b.get("lemma", "")
        if {la, lb} == {"ܢܘܪܐ", "ܢܘܗܪܐ"}:
            found_nura_nuhra = True
            print(f"  ★ PERRIN'S CATCHWORD FOUND")
        print(f"    [{cw.link_type:13s}] score={cw.score:.2f}  "
              f"{la} ({cw.token_a.get('gloss', '?')}) ↔ "
              f"{lb} ({cw.token_b.get('gloss', '?')})")
    if not found_nura_nuhra:
        print("  WARNING: nūrā / nuhrā pair NOT detected — check thresholds.")


# ---------------------------------------------------------------------------
# Scenario 2: Thomas MAP-translation (deterministic Monte Carlo precursor)
# ---------------------------------------------------------------------------

def scenario_thomas_map():
    print("\n" + "=" * 70)
    print("SCENARIO 2: Coptic Thomas → MAP Syriac translation, run detector")
    print("  Each Coptic content lemma → its single most-likely Syriac lemma")
    print("  This is a deterministic preview of what the Monte Carlo will sample.")
    print("=" * 70)

    # Load lexical map: c_lemma → top syriac_lemma
    map_top: dict[str, dict] = {}
    for line in LEX_MAP.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["candidates"]:
            map_top[rec["coptic_lemma"]] = {
                "syriac_lemma": rec["candidates"][0]["syriac_lemma"],
                "prob": rec["candidates"][0]["prob"],
                "coptic_pos": rec["coptic_pos"],
            }
    print(f"  Loaded {len(map_top)} Coptic→Syriac map entries.")

    # Load Thomas logia, group by logion (concatenate paragraphs)
    logia: dict[int, list[dict]] = {}
    for line in THOMAS.open(encoding="utf-8"):
        r = json.loads(line)
        logia.setdefault(r["logion"], []).extend(r["tokens"])

    # Translate each logion's content tokens to Syriac (MAP translation)
    syriac_tokens_per_logion: dict[int, list[dict]] = {}
    untranslated_per_logion: dict[int, int] = {}
    for L, toks in logia.items():
        syr = []
        n_untrans = 0
        for t in toks:
            c_lem = t.get("lemma")
            if not c_lem:
                continue
            entry = map_top.get(c_lem)
            if not entry:
                n_untrans += 1
                continue
            syr.append({
                "lemma": entry["syriac_lemma"],
                "form": entry["syriac_lemma"],
                "parse": "MS-EMP",  # default content-word parse to satisfy POS filter
                "gloss": "",
                "_coptic_lemma": c_lem,
                "_translation_prob": entry["prob"],
            })
        syriac_tokens_per_logion[L] = syr
        untranslated_per_logion[L] = n_untrans

    # Run detector on each adjacent logion pair
    det = CatchwordDetector("syriac")
    pair_counts = []
    perrin_pairs_found = []
    perrin_target_pairs = {(10, 11), (16, 17), (82, 83), (29, 30), (85, 86),
                          (14, 15), (46, 47), (113, 114), (13, 14), (17, 18)}

    sorted_logia = sorted(logia.keys())
    for i, L in enumerate(sorted_logia[:-1]):
        L_next = sorted_logia[i + 1]
        cws = det.detect(syriac_tokens_per_logion[L],
                         syriac_tokens_per_logion[L_next])
        pair_counts.append((L, L_next, len(cws), cws))
        if (L, L_next) in perrin_target_pairs:
            phon = [c for c in cws if c.link_type == "phonological"]
            perrin_pairs_found.append((L, L_next, len(cws), len(phon)))

    total = sum(n for _, _, n, _ in pair_counts)
    connected_one = sum(1 for _, _, n, _ in pair_counts if n > 0)

    # Connected on both sides: a logion is connected on both sides if neighbors
    # on both left and right have >= 1 catchword to it.
    cw_with_left = {L_next for L, L_next, n, _ in pair_counts if n > 0}
    cw_with_right = {L for L, L_next, n, _ in pair_counts if n > 0}
    all_logia = set(sorted_logia)
    connected_both = cw_with_left & cw_with_right
    isolated = all_logia - cw_with_left - cw_with_right

    print(f"\n  Logia: {len(sorted_logia)}  pairs: {len(pair_counts)}")
    print(f"  Total catchwords across all adjacent pairs: {total}")
    print(f"    Connected on ≥1 side: {len(cw_with_left | cw_with_right)} "
          f"({100*len(cw_with_left | cw_with_right)/len(all_logia):.1f}%)")
    print(f"    Connected on BOTH sides: {len(connected_both)} "
          f"({100*len(connected_both)/len(all_logia):.1f}%)")
    print(f"    Isolated: {len(isolated)} "
          f"({100*len(isolated)/len(all_logia):.1f}%)")
    print(f"\n  Perrin reports for Syriac: 502 catchwords, 89% both-sides, 0% isolated")

    print(f"\n  Perrin's specifically-cited adjacent pairs:")
    for L, L_next, total, phon in perrin_pairs_found:
        print(f"    {L:3d}–{L_next:<3d}: {total:>3d} catchwords (phonological={phon})")


# ---------------------------------------------------------------------------
# Scenario 3: Coptic Thomas — same detector logic, Coptic profile
# ---------------------------------------------------------------------------

def scenario_coptic_baseline():
    print("\n" + "=" * 70)
    print("SCENARIO 3: Coptic Thomas with Coptic profile (no translation)")
    print("  Williams' apples-to-apples baseline. Same algorithm, Coptic data.")
    print("=" * 70)

    logia: dict[int, list[dict]] = {}
    for line in THOMAS.open(encoding="utf-8"):
        r = json.loads(line)
        logia.setdefault(r["logion"], []).extend(r["tokens"])

    det = CatchwordDetector("coptic")
    pair_counts = []
    sorted_logia = sorted(logia.keys())
    for i, L in enumerate(sorted_logia[:-1]):
        L_next = sorted_logia[i + 1]
        cws = det.detect(logia[L], logia[L_next])
        pair_counts.append((L, L_next, len(cws), cws))

    total = sum(n for _, _, n, _ in pair_counts)
    cw_with_left = {L_next for L, L_next, n, _ in pair_counts if n > 0}
    cw_with_right = {L for L, L_next, n, _ in pair_counts if n > 0}
    all_logia = set(sorted_logia)
    connected_both = cw_with_left & cw_with_right
    isolated = all_logia - cw_with_left - cw_with_right

    print(f"\n  Logia: {len(sorted_logia)}  pairs: {len(pair_counts)}")
    print(f"  Total catchwords: {total}")
    print(f"    Connected on ≥1 side: {len(cw_with_left | cw_with_right)} "
          f"({100*len(cw_with_left | cw_with_right)/len(all_logia):.1f}%)")
    print(f"    Connected on BOTH sides: {len(connected_both)} "
          f"({100*len(connected_both)/len(all_logia):.1f}%)")
    print(f"    Isolated: {len(isolated)} "
          f"({100*len(isolated)/len(all_logia):.1f}%)")
    print(f"\n  Perrin reports for Coptic: 269 catchwords, 49% both-sides, 11% isolated")


def main():
    scenario_peshitta_sanity()
    scenario_thomas_map()
    scenario_coptic_baseline()


if __name__ == "__main__":
    main()
