#!/usr/bin/env python3
"""
Phase 3 sanity check — after the contrastive model is trained, run it on
randomly-sampled Ephrem / Narsai / Jacob strophe pairs (from training data)
and dump top-K attention tokens per strophe, plus the predicted similarity
between each anchor and:
  - its TRUE next strophe (positive)
  - a random FAR strophe (negative)

This verifies the model has actually learned to:
  (1) discriminate consecutive vs random strophes
  (2) attend to specific tokens (not produce uniform attention)

If the model is uniform-attention or always-equal-similarity, the
attention-as-catchword interpretation is meaningless. This script is a
qualitative check to catch that failure mode.

Usage:
  python scripts/phase3_sanity_check.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase3_contrastive.model import CatchwordContrastiveModel  # noqa: E402

CKPT = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_best.pt"
TOK_PATH = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = CatchwordContrastiveModel(
        vocab_size=ckpt["vocab_size"],
        d_model=a["d_model"], nhead=a["nhead"],
        num_layers=a["n_layers"], projection_dim=a["proj_dim"],
        pad_id=ckpt["pad_id"], temperature_init=a["temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id("[PAD]")
    model = load_model(device)

    # Group strophes by source file
    from collections import defaultdict
    by_file: dict[str, list[dict]] = defaultdict(list)
    with STROPHES.open() as f:
        for line in f:
            r = json.loads(line)
            by_file[r["source_file"]].append(r)

    eligible_files = [fn for fn, sts in by_file.items() if len(sts) >= 8]
    print(f"Sampling {args.n_samples} strophe pairs from {len(eligible_files)} files…\n")

    pos_sims = []
    neg_sims = []

    for sample_i in range(args.n_samples):
        fn = rng.choice(eligible_files)
        sts = by_file[fn]
        t = rng.randrange(len(sts) - 1)
        anchor = sts[t]
        pos = sts[t + 1]
        # Sample a far negative within same file
        far_idx = rng.choice([i for i in range(len(sts)) if abs(i - t) >= 5])
        neg = sts[far_idx]

        with torch.no_grad():
            for label, st in [("anchor", anchor), ("positive", pos), ("negative", neg)]:
                enc = tok.encode(st["text_consonantal"])
                ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
                z, alpha, _ = model.encode(ids)
                st["_z"] = z.squeeze(0)
                st["_alpha"] = alpha.squeeze(0).cpu().numpy()
                st["_tokens"] = enc.tokens

        pos_sim = F.cosine_similarity(anchor["_z"].unsqueeze(0),
                                       pos["_z"].unsqueeze(0)).item()
        neg_sim = F.cosine_similarity(anchor["_z"].unsqueeze(0),
                                       neg["_z"].unsqueeze(0)).item()
        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)

        a_top = sorted(
            ((anchor["_tokens"][i], float(anchor["_alpha"][i]))
             for i in range(len(anchor["_tokens"]))
             if anchor["_tokens"][i] not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")),
            key=lambda x: -x[1],
        )[:args.top_k]
        p_top = sorted(
            ((pos["_tokens"][i], float(pos["_alpha"][i]))
             for i in range(len(pos["_tokens"]))
             if pos["_tokens"][i] not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")),
            key=lambda x: -x[1],
        )[:args.top_k]

        print(f"=== Sample {sample_i+1}: {fn}  strophes {t} → {t+1} (pos), → {far_idx} (neg)")
        print(f"  anchor:    {anchor['text_consonantal'][:70]!r}")
        print(f"  positive:  {pos['text_consonantal'][:70]!r}")
        print(f"  negative:  {neg['text_consonantal'][:70]!r}")
        print(f"  cos(anchor, pos) = {pos_sim:+.3f}")
        print(f"  cos(anchor, neg) = {neg_sim:+.3f}")
        print(f"  anchor top-α: {[f'{tk}({a:.2f})' for tk, a in a_top]}")
        print(f"  pos top-α:    {[f'{tk}({a:.2f})' for tk, a in p_top]}")
        # Overlap of top-K tokens between anchor and positive: candidate catchwords
        a_set = {tk for tk, _ in a_top}
        p_set = {tk for tk, _ in p_top}
        overlap = a_set & p_set
        print(f"  top-K overlap (anchor∩pos): {sorted(overlap)}")
        print()

    import statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  pos similarity: mean={statistics.mean(pos_sims):+.3f}  "
          f"(min={min(pos_sims):+.3f}, max={max(pos_sims):+.3f})")
    print(f"  neg similarity: mean={statistics.mean(neg_sims):+.3f}  "
          f"(min={min(neg_sims):+.3f}, max={max(neg_sims):+.3f})")
    pos_higher = sum(1 for p, n in zip(pos_sims, neg_sims) if p > n)
    print(f"  pos > neg:      {pos_higher}/{len(pos_sims)} samples "
          f"({100*pos_higher/len(pos_sims):.0f}%)")


if __name__ == "__main__":
    main()
