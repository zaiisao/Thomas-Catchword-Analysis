#!/usr/bin/env python3
"""
Phase 3.3 — apply the trained contrastive model to the Coptic Gospel of
Thomas (translated into Syriac in Phase 2.4) and extract attention-weighted
catchword candidates from each logion.

This is the methodologically novel test: if the contrastive model has
learned that consecutive Syriac strophes in Ephrem / Narsai / Jacob /
Odes are linked by catchwords, its attention weights α should highlight
exactly those tokens. Applying the same model to (Phase 2 neural
translation of) the Gospel of Thomas tells us which tokens it considers
catchword candidates — which we can then compare to Perrin's claims.

Inputs:
  data/processed/checkpoints/contrastive_best.pt
  data/processed/tokenizer/bpe.json
  data/processed/thomas_neural_syriac.jsonl   (Phase 2 output)

Output:
  data/processed/phase3_attention_catchwords.jsonl
    one record per logion with top-K tokens by α and their scores.
  data/processed/phase3_pair_similarities.csv
    contrastive embedding cosine-similarity for each adjacent (L, L+1)
    in the neural Syriac Thomas.
"""

from __future__ import annotations

import argparse
import csv
import json
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
THOMAS_NEURAL = REPO_ROOT / "data" / "processed" / "thomas_neural_syriac.jsonl"
THOMAS_LOGIA = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
LEX_MAP = REPO_ROOT / "data" / "processed" / "lexical_mapping" / "coptic_to_syriac.jsonl"

OUT_ATTN = REPO_ROOT / "data" / "processed" / "phase3_attention_catchwords.jsonl"
OUT_SIM = REPO_ROOT / "data" / "processed" / "phase3_pair_similarities.csv"


def load_thomas_neural() -> dict[int, str]:
    out = {}
    for line in THOMAS_NEURAL.open():
        r = json.loads(line)
        out[r["logion"]] = r["neural_syriac_text"]
    return out


def load_thomas_map() -> dict[int, str]:
    """MAP translation: each Coptic content lemma → its top Syriac lemma."""
    map_top = {}
    for line in LEX_MAP.open():
        r = json.loads(line)
        if r["candidates"]:
            map_top[r["coptic_lemma"]] = r["candidates"][0]["syriac_lemma"]
    by_logion: dict[int, list[str]] = {}
    for line in THOMAS_LOGIA.open():
        r = json.loads(line)
        L = r["logion"]
        for t in r["tokens"]:
            cl = t.get("lemma")
            if cl and cl in map_top:
                by_logion.setdefault(L, []).append(map_top[cl])
    return {L: " ".join(words) for L, words in by_logion.items()}


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
    print(f"Loaded contrastive ckpt epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--top-k", type=int, default=8,
                    help="Top-K attention-weighted catchword candidates per logion")
    ap.add_argument("--source", choices=["neural", "map", "both"], default="both",
                    help="Which Syriac Thomas to score: Phase 2 neural, MAP, or both.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id("[PAD]")

    model = load_model(device)

    sources_to_run = []
    if args.source in ("neural", "both") and THOMAS_NEURAL.exists():
        sources_to_run.append(("neural", load_thomas_neural()))
    if args.source in ("map", "both"):
        sources_to_run.append(("map", load_thomas_map()))
    if not sources_to_run:
        sys.exit("No Thomas Syriac source available. Run phase2_translate_thomas.py first.")

    # Use the first source for the canonical OUT_ATTN/OUT_SIM files; emit
    # additional files for other sources.
    primary_name, primary_logia = sources_to_run[0]
    sorted_L = sorted(primary_logia.keys())

    OUT_ATTN.parent.mkdir(parents=True, exist_ok=True)
    import statistics

    for src_name, logia_text in sources_to_run:
        attn_out = OUT_ATTN if src_name == primary_name else OUT_ATTN.with_suffix(f".{src_name}.jsonl")
        sim_out  = OUT_SIM  if src_name == primary_name else OUT_SIM.with_suffix(f".{src_name}.csv")

        print(f"\n--- Source: {src_name}  ({len(logia_text)} logia) ---")
        local_sorted = sorted(logia_text.keys())

        z_per_logion = {}
        with attn_out.open("w", encoding="utf-8") as out, torch.no_grad():
            for L in local_sorted:
                text = logia_text[L]
                enc = tok.encode(text)
                ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
                z, alpha, _ = model.encode(ids)
                z_per_logion[L] = z.squeeze(0)
                alpha = alpha.squeeze(0).cpu().numpy()
                tokens = enc.tokens

                ranked = sorted(
                    ((tokens[i], float(alpha[i]), i) for i in range(len(tokens))
                     if tokens[i] not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")),
                    key=lambda x: -x[1],
                )[:args.top_k]
                out.write(json.dumps({
                    "logion": L,
                    "source": src_name,
                    "syriac_text": text,
                    "top_attention_tokens": [
                        {"token": t, "alpha": a, "position": i}
                        for t, a, i in ranked
                    ],
                    "all_alphas": alpha.tolist(),
                    "all_tokens": tokens,
                }, ensure_ascii=False) + "\n")

        with sim_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["logion_a", "logion_b", "cos_sim_proj"])
            sims = []
            for i, L in enumerate(local_sorted[:-1]):
                Ln = local_sorted[i + 1]
                cos = F.cosine_similarity(z_per_logion[L].unsqueeze(0),
                                           z_per_logion[Ln].unsqueeze(0)).item()
                w.writerow([L, Ln, round(cos, 4)])
                sims.append(cos)

        print(f"  Adjacent-pair cosine similarities (proj space):")
        print(f"    mean = {statistics.mean(sims):.3f}, std = {statistics.stdev(sims):.3f}")
        print(f"    min  = {min(sims):.3f}, max = {max(sims):.3f}")
        print(f"    > 0.5 (high-similarity pairs): "
              f"{sum(1 for s in sims if s > 0.5)} / {len(sims)}")
        print(f"  Wrote: {attn_out}, {sim_out}")


if __name__ == "__main__":
    main()
