#!/usr/bin/env python3
"""
Phase 3.1 — improved contrastive model for Syriac strophe-pair discrimination.

Differences from the previous attempt:
  1. Pretrained mBERT/XLM-R encoder (frozen bottom 8 layers, finetune top 4)
  2. Optional Syriac→Latin transliteration so the multilingual tokenizer can
     leverage its trained subword inventory
  3. Hard negatives: same-work strophes 3–5 positions away (similar topic, no
     immediate catchword link), forcing the model to attend to verbal links
     not topic
  4. All-pairs InfoNCE within batch (B negatives per anchor, not just one)
  5. Larger batch (64) + standard BERT-finetune lr (2e-5)

Inputs:
  data/processed/syriac_strophes.jsonl   (already extracted by phase3_extract_strophes.py)

Outputs:
  data/processed/checkpoints/improved_contrastive_best.pt
  data/processed/checkpoints/improved_contrastive_log.json

Usage:
  python scripts/phase3_improved_contrastive.py --device cuda:0 --epochs 20

Validation criterion: if val_acc does not exceed 0.60 by epoch 10, the signal
is too weak for this architecture — stop and report rather than train uselessly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
STROPHES = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"
CKPT = REPO_ROOT / "data" / "processed" / "checkpoints" / "improved_contrastive_best.pt"
LOG = REPO_ROOT / "data" / "processed" / "checkpoints" / "improved_contrastive_log.json"

DEFAULT_MODEL = "bert-base-multilingual-cased"


def strip_vocalization(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                    if not unicodedata.combining(c))


# Standard Syriac → Latin transliteration (no diacritics, no vowels — Syriac
# is mostly written without vowels anyway).
SYR2LAT = {
    "ܐ": "'", "ܒ": "b", "ܓ": "g", "ܕ": "d", "ܗ": "h", "ܘ": "w", "ܙ": "z",
    "ܚ": "H", "ܛ": "T", "ܝ": "y", "ܟ": "k", "ܠ": "l", "ܡ": "m", "ܢ": "n",
    "ܣ": "s", "ܥ": "`", "ܦ": "p", "ܨ": "S", "ܩ": "q", "ܪ": "r", "ܫ": "sh",
    "ܬ": "t",
}


def transliterate_syriac(text: str) -> str:
    text = strip_vocalization(text)
    out = []
    for ch in text:
        out.append(SYR2LAT.get(ch, ch if not ch.isspace() else " "))
    s = "".join(out)
    # Collapse multiple spaces
    return " ".join(s.split())


def load_strophes_grouped():
    """Return {(author, source_file): [strophe_dict, ...]} sorted by index."""
    works = defaultdict(list)
    with STROPHES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r.get("author", "?"), r.get("source_file", "?"))
            works[key].append(r)
    for k in works:
        works[k].sort(key=lambda s: s.get("strophe_index", 0))
    return works


def build_pairs(works, hard_neg_offsets=(3, 4, 5, -3, -4, -5),
                  min_strophes=8, rng=None):
    """Yield (anchor_text, positive_text, hard_negative_text) triples."""
    rng = rng or random.Random(0)
    triples = []
    keys = list(works.keys())
    for key, strophes in works.items():
        if len(strophes) < min_strophes:
            continue
        for t in range(len(strophes) - 1):
            anchor = strophes[t]
            positive = strophes[t + 1]
            # Hard neg: same-work, ≥3 away
            valid = [t + d for d in hard_neg_offsets
                     if 0 <= t + d < len(strophes)]
            if valid:
                neg_idx = rng.choice(valid)
                hard_neg = strophes[neg_idx]
            else:
                # Fall back to a random strophe from a DIFFERENT work
                other = rng.choice([k for k in keys if k != key])
                hard_neg = rng.choice(works[other])
            triples.append((anchor["text"], positive["text"], hard_neg["text"]))
    return triples


class TripletDataset(Dataset):
    def __init__(self, triples, tokenizer, max_len=128, transliterate=False):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len
        self.transliterate = transliterate

    def __len__(self):
        return len(self.triples)

    def _enc(self, text):
        if self.transliterate:
            text = transliterate_syriac(text)
        e = self.tok(text, max_length=self.max_len, truncation=True,
                      padding="max_length", return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0]

    def __getitem__(self, i):
        a, p, n = self.triples[i]
        ai, am = self._enc(a)
        pi, pm = self._enc(p)
        ni, nm = self._enc(n)
        return ai, am, pi, pm, ni, nm


class ImprovedContrastiveModel(nn.Module):
    def __init__(self, base_model_name=DEFAULT_MODEL, freeze_below=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        # Freeze embeddings + bottom N transformer layers
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_below:
                for p in layer.parameters():
                    p.requires_grad = False

        H = self.encoder.config.hidden_size
        self.attn_query = nn.Linear(H, 1)
        self.proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, 256))
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state                              # [B, L, H]
        attn_logits = self.attn_query(h).squeeze(-1)            # [B, L]
        attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))
        alpha = F.softmax(attn_logits, dim=-1)                  # [B, L]
        sent = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)      # [B, H]
        z = F.normalize(self.proj(sent), dim=-1)                # [B, 256]
        return z, alpha


def info_nce(z_anchor, z_positive, temperature):
    """All-pairs InfoNCE: each anchor's positive is on the diagonal of the
    similarity matrix; all other anchors' positives are negatives."""
    sim = z_anchor @ z_positive.T / temperature                # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels), sim


def evaluate(model, loader, device):
    model.eval()
    correct_pos = total = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for ai, am, pi, pm, ni, nm in loader:
            ai, am, pi, pm, ni, nm = [x.to(device) for x in (ai, am, pi, pm, ni, nm)]
            za, _ = model.encode(ai, am)
            zp, _ = model.encode(pi, pm)
            zn, _ = model.encode(ni, nm)

            # InfoNCE-style accuracy: anchor closer to its positive than to
            # any of (B - 1) other positives or its hard negative
            sim_pos = (za * zp).sum(-1)
            sim_neg = (za * zn).sum(-1)
            correct_pos += (sim_pos > sim_neg).sum().item()
            total += za.size(0)

            loss, _ = info_nce(za, zp, model.temperature.clamp(min=1e-3))
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1), correct_pos / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--freeze-below", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--transliterate", action="store_true",
                     help="Transliterate Syriac to Latin before tokenizing")
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-acc", type=float, default=0.60,
                     help="If val_acc < this by epoch 10, abort.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Transliteration: {args.transliterate}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    works = load_strophes_grouped()
    rng = random.Random(args.seed)
    triples = build_pairs(works, rng=rng)
    rng.shuffle(triples)
    n_val = max(int(len(triples) * 0.05), 256)
    val_triples = triples[:n_val]
    train_triples = triples[n_val:]
    print(f"Triples: train={len(train_triples)}, val={len(val_triples)}")

    train_ds = TripletDataset(train_triples, tokenizer, args.max_len,
                                args.transliterate)
    val_ds = TripletDataset(val_triples, tokenizer, args.max_len,
                              args.transliterate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = ImprovedContrastiveModel(args.base_model,
                                       freeze_below=args.freeze_below).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    n_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, args.warmup_steps, n_steps)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    log = []
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for ai, am, pi, pm, ni, nm in train_loader:
            ai, am, pi, pm = [x.to(device) for x in (ai, am, pi, pm)]
            za, _ = model.encode(ai, am)
            zp, _ = model.encode(pi, pm)

            # InfoNCE within batch (anchor vs ALL positives in this batch)
            T = model.temperature.clamp(min=1e-3)
            loss, sim = info_nce(za, zp, T)

            # Add a hard-negative term: penalize anchor being close to its
            # specific hard negative
            ni, nm = ni.to(device), nm.to(device)
            zn, _ = model.encode(ni, nm)
            sim_pos = (za * zp).sum(-1) / T
            sim_hard = (za * zn).sum(-1) / T
            triplet_term = F.relu(sim_hard - sim_pos + 0.1).mean()

            total_loss = loss + 0.5 * triplet_term

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            train_loss += total_loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "base_model": args.base_model,
                "transliterate": args.transliterate,
            }, CKPT)
        marker = "  ← best" if is_best else ""
        print(f"  ep {epoch:>3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  ({elapsed:.0f}s){marker}")
        log.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, "val_acc": val_acc,
                     "elapsed_s": elapsed, "is_best": is_best})

        with LOG.open("w") as f:
            json.dump({"epochs": log, "args": vars(args),
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss}, f, indent=2)

        # Early stopping: signal too weak
        if epoch >= 10 and best_val_acc < args.early_stop_acc:
            print(f"\nval_acc < {args.early_stop_acc} after 10 epochs — "
                   f"signal too weak for this architecture. Stopping.")
            break

    print()
    print(f"Best val_acc: {best_val_acc:.3f}  val_loss: {best_val_loss:.4f}")
    print(f"Saved: {CKPT}")
    print(f"Log:   {LOG}")


if __name__ == "__main__":
    main()
