"""
Contrastive Syriac strophe model.

Architecture (per project guide):
  - Token embedding + positional encoding
  - Transformer encoder (default: 6 layers, d_model=256)
  - **Attention-weighted pooling**: a learnable query produces softmax
    weights α over input tokens; pooled = α @ h. The α weights are
    interpretable as catchword-probability scores after training.
  - Projection head (MLP → 128) + L2 normalization
  - InfoNCE contrastive loss with in-batch negatives (SimCLR style)

If the model learns to distinguish consecutive strophes from random
distant ones, its attention weights must concentrate on the tokens that
do the linking — the catchwords.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CatchwordContrastiveModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        projection_dim: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Attention-weighted pooling — the layer whose weights we'll inspect
        # after training to read off catchword candidates.
        self.attention_query = nn.Linear(d_model, 1)

        # Projection head for contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )

        # Learnable temperature, log-parameterized to stay positive
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

    @property
    def temperature(self):
        return 1.0 / self.log_temperature.exp()

    def encode(self, token_ids: torch.Tensor):
        """Returns (z_proj, alpha, hidden_states).

        z_proj:    (B, projection_dim)   L2-normalized projection
        alpha:     (B, L)                token-level catchword probabilities
        hidden:    (B, L, D)             raw encoder output
        """
        pad_mask = (token_ids == self.pad_id)
        x = self.embedding(token_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.dropout(self.pos_enc(x))
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        attn_logits = self.attention_query(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))
        # When ALL positions are padded (extremely short batches), softmax
        # produces NaN; clamp.
        all_pad = pad_mask.all(dim=-1, keepdim=True)
        attn_logits = torch.where(all_pad.expand_as(attn_logits),
                                   torch.zeros_like(attn_logits), attn_logits)
        alpha = F.softmax(attn_logits, dim=-1)

        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, D)
        z = F.normalize(self.projection(pooled), dim=-1)
        return z, alpha, h

    def info_nce_loss(self, anchor_ids, candidate_ids):
        """SimCLR-style: pair (a_i, c_i) is positive; all other (a_i, c_j) are
        in-batch negatives."""
        z_a, alpha_a, _ = self.encode(anchor_ids)
        z_c, alpha_c, _ = self.encode(candidate_ids)

        # Logits: B × B similarity matrix scaled by temperature
        sim = z_a @ z_c.T * (1.0 / self.temperature.detach())
        # Diagonal targets — i-th anchor matches i-th candidate
        targets = torch.arange(sim.size(0), device=sim.device)
        loss_a2c = F.cross_entropy(sim, targets)
        loss_c2a = F.cross_entropy(sim.T, targets)
        loss = 0.5 * (loss_a2c + loss_c2a)

        # Diagnostics
        with torch.no_grad():
            acc_a2c = (sim.argmax(dim=-1) == targets).float().mean()
            acc_c2a = (sim.T.argmax(dim=-1) == targets).float().mean()
            mean_alpha_max_a = alpha_a.max(dim=-1).values.mean()
            mean_alpha_max_c = alpha_c.max(dim=-1).values.mean()

        return loss, {
            "acc_a2c": acc_a2c.item(),
            "acc_c2a": acc_c2a.item(),
            "alpha_max_a": mean_alpha_max_a.item(),
            "alpha_max_c": mean_alpha_max_c.item(),
            "temperature": self.temperature.detach().item(),
        }

    def attention_scores(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convenience: just return α weights for downstream interpretation."""
        with torch.no_grad():
            _, alpha, _ = self.encode(token_ids)
        return alpha


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
