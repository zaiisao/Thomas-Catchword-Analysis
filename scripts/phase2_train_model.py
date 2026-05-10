#!/usr/bin/env python3
"""
Phase 2.3 — train the small Coptic→Syriac transformer.

Inputs:
  data/processed/parallel_corpus/coptic_syriac_pairs.jsonl
  data/processed/tokenizer/bpe.json

Outputs:
  data/processed/checkpoints/best.pt           — best checkpoint by val loss
  data/processed/checkpoints/training_log.json — per-epoch metrics

Usage:
  python scripts/phase2_train_model.py [--epochs 80] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase2_neural_translation.data import ParallelDataset, make_collate  # noqa: E402
from phase2_neural_translation.model import Seq2SeqTransformer, count_params  # noqa: E402

PAIRS = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "coptic_syriac_pairs.jsonl"
TOK = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
CKPT_DIR = REPO_ROOT / "data" / "processed" / "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--dim-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = Tokenizer.from_file(str(TOK))
    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    vocab_size = tok.get_vocab_size()
    print(f"Tokenizer: vocab={vocab_size}  pad={pad_id} bos={bos_id} eos={eos_id}")

    train_ds = ParallelDataset(PAIRS, tok, splits=("train",))
    val_ds = ParallelDataset(PAIRS, tok, splits=("val",))
    test_ds = ParallelDataset(PAIRS, tok, splits=("test",))
    print(f"Splits — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    collate = make_collate(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.n_layers,
        num_decoder_layers=args.n_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
    ).to(device)
    print(f"Model: {count_params(model)/1e6:.1f}M params")

    # Optimizer + LR schedule with linear warmup, then cosine decay.
    # The previous run (no warmup, peak 5e-4) plateaued at val=7.66; the missing
    # ingredient was warmup. We use the standard "Noam"-style warmup-then-decay
    # but with cosine instead of inverse-square-root for the decay phase.
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    n_steps_per_epoch = max(1, len(train_loader))
    total_steps = n_steps_per_epoch * args.epochs
    warmup_steps = max(500, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay from 1.0 → 0.1
        import math
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    print(f"Schedule: warmup={warmup_steps} steps, total={total_steps} steps, peak_lr={args.lr}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log = {"epochs": [], "args": vars(args)}
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_tokens = 0
        for batch in train_loader:
            src = batch["src"].to(device, non_blocking=True)
            dec_in = batch["dec_in"].to(device, non_blocking=True)
            dec_out = batch["dec_out"].to(device, non_blocking=True)
            src_mask = batch["src_pad_mask"].to(device, non_blocking=True)
            tgt_mask = batch["tgt_pad_mask"].to(device, non_blocking=True)

            logits = model(src, dec_in, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, vocab_size), dec_out.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()  # step-level schedule (warmup + cosine)

            n_tokens = (dec_out != pad_id).sum().item()
            train_loss_sum += loss.item() * n_tokens
            train_tokens += n_tokens

        train_loss = train_loss_sum / max(1, train_tokens)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                dec_in = batch["dec_in"].to(device)
                dec_out = batch["dec_out"].to(device)
                src_mask = batch["src_pad_mask"].to(device)
                tgt_mask = batch["tgt_pad_mask"].to(device)
                logits = model(src, dec_in, src_mask, tgt_mask)
                loss = loss_fn(logits.reshape(-1, vocab_size), dec_out.reshape(-1))
                n_tokens = (dec_out != pad_id).sum().item()
                val_loss_sum += loss.item() * n_tokens
                val_tokens += n_tokens
        val_loss = val_loss_sum / max(1, val_tokens)

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
                "pad_id": pad_id, "bos_id": bos_id, "eos_id": eos_id,
            }, CKPT_DIR / "best.pt")

        marker = "  ← best" if is_best else ""
        print(f"  ep {ep:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  ({elapsed:.0f}s){marker}")
        log["epochs"].append({
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": optim.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
            "is_best": is_best,
        })
        with (CKPT_DIR / "training_log.json").open("w") as f:
            json.dump(log, f, indent=2)

    print(f"\nBest val_loss = {best_val:.4f}")
    print(f"Saved best ckpt: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
