"""
Small Coptic→Syriac translation transformer (Phase 2.3).

Per project guide:
  6-layer encoder + 6-layer decoder, d_model=256, nhead=8, dim_ff=1024,
  dropout=0.1, joint BPE tokenizer (~14K vocab).

Net params ~30M depending on tokenizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1), :]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tied weights
        self.lm_head.weight = self.embedding.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.dropout(x)

    @staticmethod
    def causal_mask(size: int, device) -> torch.Tensor:
        # nn.Transformer wants additive mask: 0 for visible, -inf for blocked.
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )

    def encode(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor):
        return self.transformer.encoder(
            self._embed(src_ids),
            src_key_padding_mask=src_pad_mask,
        )

    def decode(self, tgt_ids, memory, tgt_pad_mask, memory_pad_mask):
        causal = self.causal_mask(tgt_ids.size(1), tgt_ids.device)
        return self.transformer.decoder(
            self._embed(tgt_ids),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ):
        memory = self.encode(src_ids, src_pad_mask)
        h = self.decode(tgt_ids, memory, tgt_pad_mask, src_pad_mask)
        return self.lm_head(h)

    @torch.no_grad()
    def beam_search(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 256,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """Return top-k decoded id sequences for a batch of size 1."""
        assert src_ids.size(0) == 1, "beam_search batches one sequence at a time"
        device = src_ids.device
        memory = self.encode(src_ids, src_pad_mask)  # (1, S, D)
        # Replicate memory to beam dimension
        memory = memory.expand(beam_size, -1, -1).contiguous()
        mem_mask = src_pad_mask.expand(beam_size, -1)

        # Start each beam with [BOS]
        seqs = torch.full((beam_size, 1), self.bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), -float("inf"), device=device)
        scores[0] = 0.0  # only beam 0 is "alive" at step 0
        finished: list[tuple[float, list[int]]] = []

        for step in range(max_len):
            tgt_pad = (seqs == self.pad_id)
            logits = self.lm_head(self.decode(seqs, memory, tgt_pad, mem_mask))
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B, V)
            V = log_probs.size(-1)

            # Combine with running scores
            cand = scores.unsqueeze(1) + log_probs   # (B, V)
            flat = cand.view(-1)
            top_scores, top_idx = flat.topk(beam_size)
            beam_idx = top_idx // V
            tok_idx = top_idx % V

            seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = top_scores

            # Move finished beams aside
            keep = []
            for b in range(beam_size):
                if tok_idx[b].item() == self.eos_id:
                    L = seqs.size(1)
                    norm = ((5 + L) / 6) ** length_penalty
                    finished.append((scores[b].item() / norm, seqs[b].tolist()))
                    scores[b] = -float("inf")
                else:
                    keep.append(b)
            if not keep:
                break

        # Add any unfinished beams
        for b in range(beam_size):
            if scores[b].item() > -float("inf"):
                L = seqs.size(1)
                norm = ((5 + L) / 6) ** length_penalty
                finished.append((scores[b].item() / norm, seqs[b].tolist()))

        finished.sort(key=lambda x: x[0], reverse=True)
        return [seq for _, seq in finished[:beam_size]]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
