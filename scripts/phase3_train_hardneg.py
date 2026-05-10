#!/usr/bin/env python3
"""
Phase 3.1 (lite) — hard-negative contrastive training.

GPU-memory-constrained variant of Phase 3.1. Keeps the from-scratch
transformer architecture from Phase 3 (since the cluster's GPUs are
currently 95%+ allocated to another job, mBERT 178M params won't fit).
The substantive change vs the previous retrain is:

  1. Hard negatives — same-work strophes 3–5 positions away, not cross-file.
     If the previous result (val_acc=0.52 with cross-file negatives) was due
     to the model picking up on author/topic rather than catchword links,
     hard negatives should remove that shortcut and either reveal real
     catchword learning or expose its absence.
  2. All-pairs InfoNCE within batch (B-1 negatives per anchor).
  3. Triplet auxiliary loss: max(0, margin + sim(anchor, neg) - sim(anchor, pos))
  4. Larger batch (32 → 64) to give InfoNCE more contrast.
  5. More epochs (30 → 50) since hard negatives are harder to learn.

Inputs:
  data/processed/syriac_strophes.jsonl
  data/processed/tokenizer/bpe.json    (the existing fixed BPE tokenizer)

Outputs:
  data/processed/checkpoints/contrastive_hardneg_best.pt
  data/processed/checkpoints/contrastive_hardneg_log.json

Usage:
  python scripts/phase3_train_hardneg.py --device cuda:N --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase3_contrastive.model import CatchwordContrastiveModel  # noqa: E402

STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
TOK = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
CKPT = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_hardneg_best.pt"
LOG = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_hardneg_log.json"


def load_strophes_grouped():
    works = defaultdict(list)
    with STROPHES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            works[(r.get("author", "?"), r.get("source_file", "?"))].append(r)
    for k in works:
        works[k].sort(key=lambda s: s.get("strophe_index", 0))
    return works


def build_hard_pairs(works, hard_neg_offsets=(3, 4, 5, -3, -4, -5),
                       min_strophes=8, rng=None):
    """Triples (anchor, positive, hard_neg) where hard_neg is from the same
    work but ≥3 strophes away from anchor."""
    rng = rng or random.Random(0)
    triples = []
    keys = list(works.keys())
    for key, strophes in works.items():
        if len(strophes) < min_strophes:
            continue
        for t in range(len(strophes) - 1):
            anchor = strophes[t]["text_consonantal"]
            positive = strophes[t + 1]["text_consonantal"]
            valid = [t + d for d in hard_neg_offsets
                     if 0 <= t + d < len(strophes)]
            if valid:
                neg_idx = rng.choice(valid)
                hard_neg = strophes[neg_idx]["text_consonantal"]
            else:
                other = rng.choice([k for k in keys if k != key])
                hard_neg = rng.choice(works[other])["text_consonantal"]
            triples.append((anchor, positive, hard_neg))
    return triples


class TripletDataset(Dataset):
    def __init__(self, triples, tokenizer, max_len=128):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len
        self.pad = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.triples)

    def _enc(self, text):
        ids = self.tok.encode(text).ids[:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, i):
        a, p, n = self.triples[i]
        return self._enc(a), self._enc(p), self._enc(n)


def info_nce(z_anchor, z_positive, temperature):
    sim = z_anchor @ z_positive.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels), sim


def evaluate(model, loader, device):
    model.eval()
    correct_pos = total = 0
    losses = []
    with torch.no_grad():
        for ai, pi, ni in loader:
            ai, pi, ni = ai.to(device), pi.to(device), ni.to(device)
            za, _, _ = model.encode(ai)
            zp, _, _ = model.encode(pi)
            zn, _, _ = model.encode(ni)
            sim_pos = (za * zp).sum(-1)
            sim_neg = (za * zn).sum(-1)
            correct_pos += (sim_pos > sim_neg).sum().item()
            total += za.size(0)
            T = model.temperature.clamp(min=1e-3)
            loss, _ = info_nce(za, zp, T)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0, correct_pos / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triplet-weight", type=float, default=0.5)
    ap.add_argument("--triplet-margin", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = Tokenizer.from_file(str(TOK))
    pad_id = tok.token_to_id("[PAD]")
    vocab_size = tok.get_vocab_size()
    print(f"Tokenizer: vocab={vocab_size}, pad={pad_id}")

    works = load_strophes_grouped()
    rng = random.Random(args.seed)
    triples = build_hard_pairs(works, rng=rng)
    rng.shuffle(triples)
    n_val = max(int(len(triples) * 0.05), 256)
    val_triples = triples[:n_val]
    train_triples = triples[n_val:]
    print(f"Hard-negative triples: train={len(train_triples)}, val={len(val_triples)}")

    train_ds = TripletDataset(train_triples, tok, args.max_len)
    val_ds = TripletDataset(val_triples, tok, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = CatchwordContrastiveModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.n_layers,
        projection_dim=args.proj_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=1e-4)
    n_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=n_steps,
        pct_start=0.05, anneal_strategy="cos")

    log = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = train_acc_sum = 0.0
        n_batches = 0
        for ai, pi, ni in train_loader:
            ai, pi, ni = ai.to(device), pi.to(device), ni.to(device)
            za, _, _ = model.encode(ai)
            zp, _, _ = model.encode(pi)
            zn, _, _ = model.encode(ni)

            T = model.temperature.clamp(min=1e-3)
            nce_loss, sim = info_nce(za, zp, T)

            sim_pos = (za * zp).sum(-1) / T
            sim_hard = (za * zn).sum(-1) / T
            triplet = F.relu(args.triplet_margin + sim_hard - sim_pos).mean()

            loss = nce_loss + args.triplet_weight * triplet

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            train_loss_sum += loss.item()
            preds = sim.argmax(dim=-1)
            labels = torch.arange(sim.size(0), device=sim.device)
            train_acc_sum += (preds == labels).float().mean().item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        train_acc = train_acc_sum / max(n_batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "vocab_size": vocab_size, "pad_id": pad_id,
                "epoch": epoch,
                "val_loss": val_loss, "val_acc": val_acc,
            }, CKPT)
        marker = "  ← best" if is_best else ""
        print(f"  ep {epoch:>3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({elapsed:.0f}s){marker}")
        log.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "elapsed_s": elapsed, "is_best": is_best,
        })
        with LOG.open("w") as f:
            json.dump({"epochs": log, "args": vars(args),
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss}, f, indent=2)

    print()
    print(f"Best val_acc: {best_val_acc:.3f}  val_loss: {best_val_loss:.4f}")
    print(f"Saved: {CKPT}")


if __name__ == "__main__":
    main()
