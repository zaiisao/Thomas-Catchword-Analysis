#!/usr/bin/env python3
"""
Phase 3.2 — train the contrastive Syriac-strophe model.

In each batch we use only POSITIVE (consecutive) pairs and rely on
in-batch negatives (SimCLR style). We tokenize with our existing
joint Coptic+Syriac BPE — Syriac vocabulary is well-covered.

Outputs:
  data/processed/checkpoints/contrastive_best.pt
  data/processed/checkpoints/contrastive_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase3_contrastive.model import CatchwordContrastiveModel, count_params  # noqa: E402

PAIRS = REPO_ROOT / "data" / "processed" / "phase3_pairs.jsonl"
TOK = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
CKPT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"


class StrophePairDataset(Dataset):
    """Positive pairs only. Negatives come from in-batch other examples."""

    def __init__(self, jsonl_path: Path, tokenizer: Tokenizer, max_len: int = 128,
                 only_positives: bool = True):
        self.tok = tokenizer
        self.pad = tokenizer.token_to_id("[PAD]")
        self.records = []
        for line in Path(jsonl_path).open():
            r = json.loads(line)
            if only_positives and r["label"] != 1:
                continue
            self.records.append((r["anchor_text"], r["candidate_text"]))
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def _enc(self, text):
        ids = self.tok.encode(text).ids
        return ids[:self.max_len]

    def __getitem__(self, idx):
        a, c = self.records[idx]
        return torch.tensor(self._enc(a), dtype=torch.long), torch.tensor(self._enc(c), dtype=torch.long)


def make_collate(pad_id):
    def collate(batch):
        a_list = [b[0] for b in batch]
        c_list = [b[1] for b in batch]
        a_max = max(len(t) for t in a_list)
        c_max = max(len(t) for t in c_list)
        def pad_to(lst, n):
            return torch.stack([
                torch.cat([t, torch.full((n - len(t),), pad_id, dtype=torch.long)])
                if len(t) < n else t[:n]
                for t in lst
            ])
        return pad_to(a_list, a_max), pad_to(c_list, c_max)
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--nhead", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = Tokenizer.from_file(str(TOK))
    pad_id = tok.token_to_id("[PAD]")
    vocab_size = tok.get_vocab_size()

    ds = StrophePairDataset(PAIRS, tok, only_positives=True)
    print(f"Positive pairs: {len(ds)}")
    n_train = int(0.95 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  train={len(train_ds)} val={len(val_ds)}")

    collate = make_collate(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    model = CatchwordContrastiveModel(
        vocab_size=vocab_size,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.n_layers, projection_dim=args.proj_dim,
        pad_id=pad_id, temperature_init=args.temperature,
    ).to(device)
    print(f"Model: {count_params(model)/1e6:.1f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.1)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log = {"epochs": [], "args": vars(args)}
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        accs = []
        for anchor_ids, cand_ids in train_loader:
            anchor_ids = anchor_ids.to(device)
            cand_ids = cand_ids.to(device)
            loss, diag = model.info_nce_loss(anchor_ids, cand_ids)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
            accs.append((diag["acc_a2c"] + diag["acc_c2a"]) / 2)

        train_loss = sum(losses) / len(losses)
        train_acc = sum(accs) / len(accs)

        # Validation
        model.eval()
        val_losses, val_accs = [], []
        alpha_max_sum = 0.0
        n_alpha = 0
        with torch.no_grad():
            for anchor_ids, cand_ids in val_loader:
                anchor_ids = anchor_ids.to(device)
                cand_ids = cand_ids.to(device)
                loss, diag = model.info_nce_loss(anchor_ids, cand_ids)
                val_losses.append(loss.item())
                val_accs.append((diag["acc_a2c"] + diag["acc_c2a"]) / 2)
                alpha_max_sum += diag["alpha_max_a"] + diag["alpha_max_c"]
                n_alpha += 2

        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_acc = sum(val_accs) / max(1, len(val_accs))
        alpha_max = alpha_max_sum / max(1, n_alpha)

        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": ep,
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "pad_id": pad_id,
            }, CKPT_DIR / "contrastive_best.pt")

        marker = "  ← best" if is_best else ""
        print(f"  ep {ep:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}   "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}   "
              f"α_max={alpha_max:.3f}   ({elapsed:.0f}s){marker}")
        log["epochs"].append({
            "epoch": ep,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "alpha_max": alpha_max,
            "elapsed_s": elapsed, "is_best": is_best,
        })
        with (CKPT_DIR / "contrastive_log.json").open("w") as f:
            json.dump(log, f, indent=2)

    print(f"\nBest val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
