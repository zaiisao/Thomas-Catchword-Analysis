#!/usr/bin/env python3
"""
BinaryAlign — word alignment as per-pair binary classification on top of a
multilingual encoder (default: microsoft/mdeberta-v3-base; XLM-R-large also
supported).

Replaces the IBM-1 EM aligner used by build_lexical_map.py /
build_reverse_lexical_map.py.

Pipeline per verse pair (src_words, tgt_words):
  1. For each source word w_X:
       - surround it with unique separator tokens [ALIGN_BEG] / [ALIGN_END]
         in the source sentence (the markers are added as special tokens to
         the tokenizer, so they are not split by subword BPE)
       - encode  [CLS] marked_src [SEP] tgt [SEP]  with the foundation model
       - take the final encoded representation of each target subword y_k
       - score it:
             logit z_k  =  W · h_k  + b              (binary classification)
             p(a_k = 1 | w_X) = sigmoid(z_k)
         The linear head W,b is loaded from a checkpoint if --head-ckpt is
         given; otherwise we fall back to a cosine-similarity proxy between
         the marker-conditioned source representation and h_k:
             z_k = T · cos( h_src , h_k )
         (T = learned/fixed temperature, default 10).  The proxy makes the
         module runnable out of the box; for trained-head behaviour, supply
         a fine-tuned BinaryAlign checkpoint via --head-ckpt.
  2. Max-aggregate subword probs into word-level probs:
             p(w_X -> w_Y) = max_{k in subwords(w_Y)} p(a_k = 1 | w_X)
  3. (Optional, default ON) Symmetrize:  run the same procedure with
     src/tgt swapped, then average
             p_sym(i,j) = 0.5 * ( p_fwd(i,j) + p_rev(j,i) )
  4. Threshold at 0.5 for hard alignment.  The (soft) p_sym matrix is also
     returned so downstream callers can build a probabilistic lexical map.

The tokenizer is invertible (HF fast-tokenizer with `is_split_into_words=True`
+ `word_ids()` mapping), so we round-trip cleanly between word indices and
subword positions — handles BPE continuation symbols transparently.

CLI usage (standalone, for inspection / debugging):
  python scripts/align_binary.py \
      --src "the cat sat on the mat" \
      --tgt "le chat est assis sur le tapis"

  python scripts/align_binary.py --pairs-jsonl pairs.jsonl --out aligns.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Separator tokens that bracket the queried source word.  Added as special
# tokens so they survive subword tokenisation as a single piece each.
ALIGN_BEG = "[ALIGN_BEG]"
ALIGN_END = "[ALIGN_END]"


@dataclass
class BinaryAlignerConfig:
    model_name: str = "microsoft/mdeberta-v3-base"
    head_ckpt: str | None = None              # path to trained linear head
    max_length: int = 256
    threshold: float = 0.5
    symmetrize: bool = True
    batch_size: int = 32                      # source-word queries per GPU fwd pass
    device: str | None = None                 # auto-detect if None
    # Cosine-proxy temperature (only used when head_ckpt is None).  Applied
    # to z-normalised cosine scores, so a temperature of ~2 keeps the most
    # likely targets around p ≈ 0.9 and the least likely around p ≈ 0.1.
    cos_temperature: float = 2.0
    # Layer to read hidden states from.  -1 means final encoder layer.
    hidden_layer: int = -1


class BinaryAligner:
    """BinaryAlign inference.  Stateless beyond model weights — one instance
    can be reused across many sentence pairs."""

    def __init__(self, cfg: BinaryAlignerConfig | None = None):
        self.cfg = cfg or BinaryAlignerConfig()
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=True
        )
        # Add marker tokens.  resize_token_embeddings is needed on the model.
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [ALIGN_BEG, ALIGN_END]}
        )
        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval().to(self.device)

        hidden_size = self.model.config.hidden_size
        # Linear classification head:  z = W h + b  →  σ(z) = P(a=1).
        # Random init unless a checkpoint is loaded.
        self.head: nn.Linear | None = nn.Linear(hidden_size, 1).to(self.device)
        self._head_trained = False
        if self.cfg.head_ckpt:
            state = torch.load(self.cfg.head_ckpt, map_location=self.device)
            self.head.load_state_dict(state)
            self.head.eval()
            self._head_trained = True

        self.beg_id = self.tokenizer.convert_tokens_to_ids(ALIGN_BEG)
        self.end_id = self.tokenizer.convert_tokens_to_ids(ALIGN_END)

    # ---------------------------------------------------------- public API

    @torch.no_grad()
    def soft_align_one_direction(
        self, src_words: Sequence[str], tgt_words: Sequence[str]
    ) -> torch.Tensor:
        """[n_src, n_tgt] matrix of  p(a=1 | src_i)  per word pair, after
        max-aggregation over target subwords.  One source-word query per
        row; batched up to `cfg.batch_size` queries per GPU forward pass.
        No symmetrization."""
        n_src, n_tgt = len(src_words), len(tgt_words)
        if n_src == 0 or n_tgt == 0:
            return torch.zeros(n_src, n_tgt)
        M = torch.zeros(n_src, n_tgt, device=self.device)
        B = max(1, self.cfg.batch_size)

        for start in range(0, n_src, B):
            idxs = list(range(start, min(start + B, n_src)))
            srcs = [
                list(src_words[:i])
                + [ALIGN_BEG, src_words[i], ALIGN_END]
                + list(src_words[i + 1 :])
                for i in idxs
            ]
            tgts = [list(tgt_words)] * len(idxs)

            encoding = self.tokenizer(
                srcs, tgts,
                is_split_into_words=True,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = encoding["input_ids"].to(self.device)
            attn = encoding["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=(self.cfg.hidden_layer != -1))
            if self.cfg.hidden_layer == -1:
                hidden = out.last_hidden_state               # [B, S, D]
            else:
                hidden = out.hidden_states[self.cfg.hidden_layer]

            for b_idx, src_idx in enumerate(idxs):
                seq_ids = encoding.sequence_ids(b_idx)
                word_ids = encoding.word_ids(b_idx)

                tgt_positions = [
                    k for k, sid in enumerate(seq_ids)
                    if sid == 1 and word_ids[k] is not None
                ]
                if not tgt_positions:
                    continue

                tgt_word_id_per_subword = [word_ids[k] for k in tgt_positions]
                tgt_hidden = hidden[b_idx, tgt_positions]    # [T_tgt, D]

                if self._head_trained:
                    logits = self.head(tgt_hidden).squeeze(-1)
                    probs = torch.sigmoid(logits)
                else:
                    # Cosine-similarity proxy with row z-normalisation
                    # (see design notes in the docstring).
                    ids_list = input_ids[b_idx].tolist()
                    try:
                        beg_pos = ids_list.index(self.beg_id)
                        end_pos = ids_list.index(self.end_id)
                        inner = list(range(beg_pos + 1, end_pos))
                        if not inner:
                            inner = [beg_pos, end_pos]
                        h_src = hidden[b_idx, inner].mean(dim=0)
                    except ValueError:
                        src_positions = [
                            k for k, sid in enumerate(seq_ids) if sid == 0
                        ]
                        h_src = hidden[b_idx, src_positions].mean(dim=0)

                    cos = F.cosine_similarity(
                        tgt_hidden,
                        h_src.unsqueeze(0).expand_as(tgt_hidden),
                        dim=-1,
                    )
                    mu = cos.mean()
                    sigma = cos.std(unbiased=False).clamp_min(1e-6)
                    z = (cos - mu) / sigma
                    probs = torch.sigmoid(self.cfg.cos_temperature * z)

                # Max-aggregate subwords -> words (vectorised via scatter_max
                # would be ideal; for small T_tgt the python loop is fine).
                row = torch.zeros(n_tgt, device=self.device)
                for k, wid in enumerate(tgt_word_id_per_subword):
                    if wid is None or wid >= n_tgt:
                        continue
                    if probs[k] > row[wid]:
                        row[wid] = probs[k]
                M[src_idx] = row

        return M.cpu()

    @torch.no_grad()
    def align(
        self,
        src_words: Sequence[str],
        tgt_words: Sequence[str],
    ) -> dict:
        """Full alignment: forward + (optional) reverse + symmetrize +
        threshold.  Returns:
            {
              "p_fwd":  [n_src, n_tgt] float,
              "p_rev":  [n_src, n_tgt] float or None,
              "p_sym":  [n_src, n_tgt] float,
              "pairs":  [(src_idx, tgt_idx, prob), ...]   # above threshold
            }
        """
        p_fwd = self.soft_align_one_direction(src_words, tgt_words)
        if self.cfg.symmetrize:
            p_rev_T = self.soft_align_one_direction(tgt_words, src_words)  # [n_tgt, n_src]
            p_rev = p_rev_T.T
            p_sym = 0.5 * (p_fwd + p_rev)
        else:
            p_rev = None
            p_sym = p_fwd

        pairs: list[tuple[int, int, float]] = []
        thresh = self.cfg.threshold
        ns, nt = p_sym.shape
        for i in range(ns):
            for j in range(nt):
                pij = float(p_sym[i, j])
                if pij >= thresh:
                    pairs.append((i, j, pij))
        return {
            "p_fwd": p_fwd,
            "p_rev": p_rev,
            "p_sym": p_sym,
            "pairs": pairs,
        }


# ============================================================ CLI / dry-run

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="space-separated source sentence")
    ap.add_argument("--tgt", help="space-separated target sentence")
    ap.add_argument(
        "--pairs-jsonl",
        help="JSONL with one object per pair {src: [tok...], tgt: [tok...]}",
    )
    ap.add_argument("--out", help="JSONL output for --pairs-jsonl mode")
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--head-ckpt", default=None,
                    help="Trained BinaryAlign linear head; "
                         "if omitted, cosine-similarity proxy is used.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-symmetrize", action="store_true")
    args = ap.parse_args()

    cfg = BinaryAlignerConfig(
        model_name=args.model,
        head_ckpt=args.head_ckpt,
        threshold=args.threshold,
        symmetrize=not args.no_symmetrize,
    )
    aligner = BinaryAligner(cfg)
    if not aligner._head_trained:
        print("[binary-align] no trained head — using cosine-similarity proxy "
              f"(T={cfg.cos_temperature}).", file=sys.stderr)

    if args.src and args.tgt:
        src = args.src.split()
        tgt = args.tgt.split()
        result = aligner.align(src, tgt)
        for i, j, p in result["pairs"]:
            print(f"{i}\t{src[i]}\t{j}\t{tgt[j]}\t{p:.3f}")
        return

    if args.pairs_jsonl and args.out:
        in_path = Path(args.pairs_jsonl)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
            for line in f:
                r = json.loads(line)
                src = r["src"]
                tgt = r["tgt"]
                result = aligner.align(src, tgt)
                g.write(json.dumps({
                    "src": src,
                    "tgt": tgt,
                    "pairs": [(i, j, round(p, 4)) for i, j, p in result["pairs"]],
                }, ensure_ascii=False) + "\n")
        return

    ap.error("must supply either --src/--tgt or --pairs-jsonl/--out")


if __name__ == "__main__":
    _cli()
