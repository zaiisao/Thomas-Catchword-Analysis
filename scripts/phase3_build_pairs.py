#!/usr/bin/env python3
"""
Phase 3.1 — build positive/negative strophe-pair dataset for contrastive
training.

  POSITIVE pair: two strophes that occur consecutively within the same
                 source file (i.e., (S_t, S_{t+1}) in the same work). If
                 catchwords are a real organizing principle, these pairs
                 should share linking tokens that an attention-based
                 contrastive model can learn to attend to.

  NEGATIVE pair: two strophes that are NOT consecutive — either far apart
                 in the same work (>= MIN_DISTANCE strophes apart) or from
                 a different source file. These should not have catchword
                 links beyond chance.

Output:
  data/processed/phase3_pairs.jsonl
    {"anchor", "candidate", "label", "source"}
      label: 1 = positive (consecutive), 0 = negative
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
OUT = REPO_ROOT / "data" / "processed" / "phase3_pairs.jsonl"

MIN_NEG_DISTANCE = 5      # negatives must be ≥ this many strophes apart
NEGATIVES_PER_POSITIVE = 1  # 1:1 balance for InfoNCE-style training
SEED = 42


def main():
    rng = random.Random(SEED)

    # Group strophes by source file, preserving in-file order
    by_file: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    with STROPHES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_file[r["source_file"]].append(r)
            n_total += 1
    print(f"Loaded {n_total} strophes across {len(by_file)} files")

    # Filter to files with ≥ MIN_NEG_DISTANCE+2 strophes (so we can pick negatives)
    eligible_files = {fn: sts for fn, sts in by_file.items()
                      if len(sts) >= MIN_NEG_DISTANCE + 2}
    print(f"  {len(eligible_files)} files with ≥ {MIN_NEG_DISTANCE+2} strophes (eligible)")

    # All strophes pooled, for cross-file negatives
    all_strophes = [s for sts in by_file.values() for s in sts]

    n_pos = n_neg = 0
    with OUT.open("w", encoding="utf-8") as out:
        for fn, sts in eligible_files.items():
            for t in range(len(sts) - 1):
                anchor = sts[t]
                pos = sts[t + 1]
                out.write(json.dumps({
                    "anchor_text": anchor["text_consonantal"],
                    "candidate_text": pos["text_consonantal"],
                    "label": 1,
                    "source": "consecutive",
                    "anchor_ref": f"{fn}:{anchor['strophe_index']}",
                    "candidate_ref": f"{fn}:{pos['strophe_index']}",
                }, ensure_ascii=False) + "\n")
                n_pos += 1

                # Sample NEGATIVES_PER_POSITIVE negatives.
                # Half come from same file (distant), half cross-file.
                for _ in range(NEGATIVES_PER_POSITIVE):
                    if rng.random() < 0.5 and len(sts) > 2 * MIN_NEG_DISTANCE:
                        # In-file distant
                        far_indices = [
                            i for i in range(len(sts))
                            if abs(i - t) >= MIN_NEG_DISTANCE
                        ]
                        neg = sts[rng.choice(far_indices)]
                        src = "in-file-distant"
                    else:
                        # Cross-file random
                        neg = rng.choice(all_strophes)
                        src = "cross-file"
                    out.write(json.dumps({
                        "anchor_text": anchor["text_consonantal"],
                        "candidate_text": neg["text_consonantal"],
                        "label": 0,
                        "source": src,
                        "anchor_ref": f"{fn}:{anchor['strophe_index']}",
                        "candidate_ref": f"{neg['source_file']}:{neg['strophe_index']}",
                    }, ensure_ascii=False) + "\n")
                    n_neg += 1

    print(f"\nWrote {n_pos + n_neg} pairs ({n_pos} positive, {n_neg} negative) -> {OUT}")
    print(f"  Class balance: {n_pos / (n_pos + n_neg):.2%} positive")


if __name__ == "__main__":
    main()
