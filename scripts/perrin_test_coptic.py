#!/usr/bin/env python3
"""
Test 6: Phon-arrangement on the actual Coptic Thomas (the extant manuscript).

If Perrin's Syriac claim is right (Thomas was originally composed in Syriac
with deliberate sound-play), then:
  - Coptic Thomas should NOT show as much phon-arrangement (Syriac sound-play
    didn't carry over to Coptic).
  - Or alternatively, the Coptic was translated FROM a Syriac source which
    introduces the arrangement signal.

If Coptic shows similar or stronger phon-arrangement than Syriac retroversion,
then arrangement is generic (not Syriac-specific).

Output: data/perrin_direct/coptic_thomas_v0.json
"""
from __future__ import annotations

import json
import random
import sys
import time
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

LLM_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"
OUT = REPO_ROOT / "data" / "perrin_direct" / "coptic_thomas_v0.json"

COPTIC_RX = re.compile(r"[Ⲁ-ⳣ]")  # Coptic + Old Coptic + Coptic Supplement
COPTIC_PUNCT = re.compile(r"[⁕⳾⳼·̱̇̄]")

PHON_THRESHOLD = 0.6
FILTER_PCT = 80.0
N_PERMS = 10000
SEED = 42

FILTERS = {
    "all":  None,
    "phon": frozenset({"phonological", "etymological"}),
    "sem":  frozenset({"semantic"}),
}


def strip_voc(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


def tokenize_coptic(text: str) -> list[str]:
    """Coptic tokens are already space-separated in the source."""
    cleaned = "".join(c for c in text if COPTIC_RX.match(c) or c.isspace())
    cleaned = strip_voc(cleaned)
    cleaned = COPTIC_PUNCT.sub(" ", cleaned)
    cleaned = cleaned.lower()
    return [w for w in cleaned.split() if w]


def make_tokens(text: str) -> list[dict]:
    return [{"form": t, "lemma": t, "parse": "MS-EMP"}
            for t in tokenize_coptic(text)]


def load_coptic_thomas() -> dict[int, list[dict]]:
    out = {}
    for i in range(115):
        p = LLM_DIR / f"logion_{i:03d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ct = d.get("coptic_text")
        if ct:
            out[i] = make_tokens(ct)
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("loading Coptic Thomas…")
    toks = load_coptic_thomas()
    ids = sorted(toks.keys())
    print(f"  {len(ids)} logia, "
          f"avg {np.mean([len(toks[i]) for i in ids]):.1f} tokens/logion")

    n = len(ids)
    cutoff = FILTER_PCT * n / 100.0
    cnt: Counter = Counter()
    for tt in toks.values():
        for lem in {t["lemma"] for t in tt}:
            cnt[lem] += 1
    blocked = {lem for lem, c in cnt.items() if c > cutoff}
    print(f"  blocked {len(blocked)} top-{100-FILTER_PCT:.0f}% lemmas")

    det = CatchwordDetector("coptic", phonological_threshold=PHON_THRESHOLD,
                              require_content_pos=False)
    filt = {i: [t for t in toks[i] if t["lemma"] not in blocked] for i in ids}

    print(f"building {n}×{n-1} matrix…")
    t0 = time.time()
    matrix = {}
    for n_done, i in enumerate(ids):
        for j in ids:
            if i == j: continue
            ta, tb = filt[i], filt[j]
            if not ta or not tb:
                matrix[(i, j)] = []
                continue
            cws = det.detect(ta, tb)
            matrix[(i, j)] = [(c.token_a["lemma"], c.token_b["lemma"],
                                c.link_type) for c in cws]
        if (n_done + 1) % 25 == 0:
            print(f"  row {n_done+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  matrix done {time.time()-t0:.0f}s")

    # Diagnostic
    n_phon = n_sem = n_etym = 0
    n_bound = n - 1
    for k in range(n_bound):
        for c in matrix.get((ids[k], ids[k+1]), []):
            if c[2] == "phonological": n_phon += 1
            elif c[2] == "etymological": n_etym += 1
            elif c[2] == "semantic": n_sem += 1
    print(f"\nDIAGNOSTIC (true order):")
    print(f"  phon+etym/boundary = {(n_phon+n_etym)/n_bound:.2f}")
    print(f"  semantic/boundary = {n_sem/n_bound:.2f}")

    # Permutation
    def total(order, link_filter):
        t = 0
        for k in range(len(order)-1):
            cell = matrix.get((order[k], order[k+1]), [])
            if link_filter is None:
                t += len(cell)
            else:
                t += sum(1 for c in cell if c[2] in link_filter)
        return t

    rng = random.Random(SEED)
    base = list(ids)
    results = {}
    for fname, lf in FILTERS.items():
        true_tot = total(base, lf)
        null = np.empty(N_PERMS, dtype=np.int32)
        for p in range(N_PERMS):
            sh = base.copy(); rng.shuffle(sh)
            null[p] = total(sh, lf)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_tot - nm) / ns if ns > 0 else 0.0
        pv = float((null >= true_tot).mean())
        results[fname] = {"true_total": int(true_tot),
                            "null_mean": nm, "null_std": ns,
                            "z_score": float(z), "p_value": pv}
        print(f"  [{fname:4s}] true={true_tot}, null={nm:.1f}±{ns:.1f}, "
              f"z={z:.2f}, p={pv:.4f}")

    rec = {
        "corpus": "thomas", "lang": "coptic",
        "n_logia": n, "n_blocked": len(blocked),
        "phon_threshold": PHON_THRESHOLD, "filter_pct": FILTER_PCT,
        "diagnostic": {
            "n_boundaries": n_bound,
            "phonological_total": n_phon, "etymological_total": n_etym,
            "semantic_total": n_sem,
        },
        "results": results,
    }
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
