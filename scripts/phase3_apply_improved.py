#!/usr/bin/env python3
"""
Phase 3.2 — apply the improved (hard-negative) contrastive model to:

  (a) Peshitta Gospel verse pairs (sanity check — does the model find
      attention concentrated on tokens that the rule-based detector also
      flags as catchwords?)
  (b) Thomas in available Syriac renderings:
        - Phase 2A beam (λ=0.3) translation, our best lexical-map output
        - Phase 2B LLM (if available)
        - Existing neural translation from Phase 2 retrain (caveat: NMT
          model is data-limited)
  (c) For each Thomas adjacent pair, compare cos_sim of consecutive vs
      shuffled adjacent pairs (i.e., a within-Thomas permutation test).

Inputs:
  data/processed/checkpoints/contrastive_hardneg_best.pt   (or original)
  data/processed/tokenizer/bpe.json
  data/processed/parallel_corpus/peshitta_nt_lemmatized.jsonl
  data/processed/phase2a_translations/lambda_0.3.jsonl
  data/processed/llm_translations/  (optional)
  data/processed/thomas_neural_syriac.jsonl

Outputs:
  data/processed/phase3_improved_thomas.json
  data/processed/phase3_improved_attention.jsonl
  analysis/figures/phase3_improved_thomas.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase3_contrastive.model import CatchwordContrastiveModel  # noqa: E402

CKPT_HARD = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_hardneg_best.pt"
CKPT_OLD  = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_best.pt"
TOK_PATH  = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
PESHITTA  = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "peshitta_nt_lemmatized.jsonl"
P2A_DIR   = REPO_ROOT / "data" / "processed" / "phase2a_translations"
LLM_DIR   = REPO_ROOT / "data" / "processed" / "llm_translations"
NEURAL    = REPO_ROOT / "data" / "processed" / "thomas_neural_syriac.jsonl"

OUT_JSON = REPO_ROOT / "data" / "processed" / "phase3_improved_thomas.json"
OUT_ATTN = REPO_ROOT / "data" / "processed" / "phase3_improved_attention.jsonl"
OUT_FIG  = REPO_ROOT / "analysis" / "figures" / "phase3_improved_thomas.png"

SEED = 42
N_SHUFFLE = 1000


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CatchwordContrastiveModel(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
        d_model=args.get("d_model", 256),
        nhead=args.get("nhead", 8),
        num_layers=args.get("n_layers", 6),
        projection_dim=args.get("proj_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path.name}: epoch={ckpt.get('epoch')} "
          f"val_acc={ckpt.get('val_acc')}")
    return model


def load_phase2a(lam=0.3):
    """Load Thomas Syriac from beam-search lambda."""
    path = P2A_DIR / f"lambda_{lam}.jsonl"
    if not path.exists():
        return None
    by_log = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            by_log[r["logion"]] = " ".join(r["syriac_lemmas"])
    return by_log


def load_neural():
    if not NEURAL.exists():
        return None
    by_log = {}
    with NEURAL.open() as f:
        for line in f:
            r = json.loads(line)
            by_log[r["logion"]] = r.get("neural_syriac_text", "")
    return by_log


def load_llm():
    if not LLM_DIR.exists():
        return None
    by_log = {}
    for path in sorted(LLM_DIR.glob("logion_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("variants"):
            # Use the first non-empty variant (representative single translation)
            for v in d["variants"]:
                if v.strip():
                    by_log[d["logion"]] = v
                    break
    return by_log


def encode_text(text, tok, model, max_len, pad_id, device):
    enc = tok.encode(text)
    ids = enc.ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        z, alpha, _ = model.encode(ids_t)
    return z.squeeze(0), alpha.squeeze(0).cpu().numpy(), enc.tokens


def cos_sim(z1, z2):
    return F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="auto",
                     help="auto picks hardneg if available, else original")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.ckpt == "auto":
        ckpt_path = CKPT_HARD if CKPT_HARD.exists() else CKPT_OLD
    else:
        ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint at {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}")

    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id("[PAD]")
    model = load_model(ckpt_path, device)

    sources = {}
    p2a = load_phase2a(0.3)
    if p2a:
        sources["phase2a_beam"] = p2a
        print(f"  source phase2a_beam: {len(p2a)} logia")
    llm = load_llm()
    if llm:
        sources["phase2b_llm"] = llm
        print(f"  source phase2b_llm: {len(llm)} logia")
    neural = load_neural()
    if neural:
        sources["phase2_nmt"] = neural
        print(f"  source phase2_nmt: {len(neural)} logia (caveat: model data-limited)")

    if not sources:
        sys.exit("No Thomas Syriac sources found.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_ATTN.parent.mkdir(parents=True, exist_ok=True)

    # Per-source analysis
    results = {}
    attn_records = []
    for src_name, by_log in sources.items():
        sorted_L = sorted(by_log.keys())
        z_per_logion = {}
        for L in sorted_L:
            text = by_log[L] if by_log[L].strip() else "[empty]"
            z, alpha, tokens = encode_text(text, tok, model, args.max_len,
                                            pad_id, device)
            z_per_logion[L] = z
            # Top-k attention tokens
            n_compare = min(len(tokens), len(alpha))
            ranked = sorted(
                ((tokens[i], float(alpha[i]), i) for i in range(n_compare)
                 if tokens[i] not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")),
                key=lambda x: -x[1],
            )[:args.top_k]
            attn_records.append({
                "source": src_name, "logion": L, "syriac_text": text,
                "top_attention_tokens": [
                    {"token": t, "alpha": a, "position": i} for t, a, i in ranked],
            })

        # Adjacent-pair similarities
        adj_sims = []
        for i, L in enumerate(sorted_L[:-1]):
            Ln = sorted_L[i + 1]
            adj_sims.append(cos_sim(z_per_logion[L], z_per_logion[Ln]))

        # Shuffle test: shuffle the logion order, recompute adjacent sims,
        # repeat N_SHUFFLE times. Compare mean adjacent sim under shuffles
        # vs the true mean.
        true_mean = float(np.mean(adj_sims))
        shuffled_means = []
        L_list = list(sorted_L)
        for _ in range(N_SHUFFLE):
            shuffled = L_list[:]
            rng.shuffle(shuffled)
            ss = []
            for i in range(len(shuffled) - 1):
                ss.append(cos_sim(z_per_logion[shuffled[i]],
                                    z_per_logion[shuffled[i + 1]]))
            shuffled_means.append(float(np.mean(ss)))
        # One-tailed permutation p-value: P(shuffled >= true)
        p_value = float(np.mean([m >= true_mean for m in shuffled_means]))
        z_score = (true_mean - float(np.mean(shuffled_means))) / (
            float(np.std(shuffled_means)) + 1e-9)
        results[src_name] = {
            "n_logia": len(sorted_L),
            "adjacent_sim_mean": true_mean,
            "adjacent_sim_std": float(np.std(adj_sims)),
            "adjacent_sim_min": float(min(adj_sims)),
            "adjacent_sim_max": float(max(adj_sims)),
            "shuffle_mean": float(np.mean(shuffled_means)),
            "shuffle_std": float(np.std(shuffled_means)),
            "permutation_p": p_value,
            "z_score": z_score,
            "adj_sims": adj_sims,
        }
        print()
        print(f"=== {src_name} ===")
        print(f"  Adjacent-pair cos_sim: mean={true_mean:.3f} "
              f"(±{float(np.std(adj_sims)):.3f}, "
              f"range [{min(adj_sims):.3f}, {max(adj_sims):.3f}])")
        print(f"  Shuffle baseline: mean={float(np.mean(shuffled_means)):.3f} "
              f"(±{float(np.std(shuffled_means)):.3f})")
        print(f"  Permutation p:    {p_value:.4f}")
        print(f"  Z-score:          {z_score:+.3f}")

    # Sanity: Peshitta Matt 1:1-1:5 pairs
    print()
    print("=== Sanity check: Peshitta Matt 1 verses ===")
    matt_verses = []
    with PESHITTA.open() as f:
        for line in f:
            r = json.loads(line)
            if r["book"] == "Matt" and r["chapter"] == 1 and r["verse"] <= 10:
                matt_verses.append(r["syriac_consonantal"])
    if matt_verses:
        zs = []
        for text in matt_verses:
            z, _, _ = encode_text(text, tok, model, args.max_len, pad_id, device)
            zs.append(z)
        adj = [cos_sim(zs[i], zs[i + 1]) for i in range(len(zs) - 1)]
        print(f"  Matt 1:1–10 adjacent verse sim mean: {np.mean(adj):.3f}")
        results["sanity_peshitta_matt1"] = {
            "adj_sims": adj, "mean": float(np.mean(adj))}

    with OUT_JSON.open("w") as f:
        json.dump({
            "checkpoint": str(ckpt_path.name),
            "n_shuffle": N_SHUFFLE,
            "per_source": results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    with OUT_ATTN.open("w", encoding="utf-8") as f:
        for rec in attn_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_ATTN}")

    # Plot: adjacent vs shuffled distribution per source
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    real_sources = [s for s in results if s != "sanity_peshitta_matt1"]
    fig, axes = plt.subplots(1, len(real_sources), figsize=(5 * len(real_sources), 5),
                              squeeze=False)
    axes = axes[0]
    for ax, src in zip(axes, real_sources):
        d = results[src]
        ax.hist(d["adj_sims"], bins=20, alpha=0.6, color="C0",
                 label=f"adjacent\nmean={d['adjacent_sim_mean']:.3f}")
        ax.axvline(d["shuffle_mean"], color="C1", linestyle="--",
                    label=f"shuffle mean\n{d['shuffle_mean']:.3f}")
        ax.set_xlabel("Adjacent-pair cos_sim")
        ax.set_ylabel("# pairs")
        ax.set_title(f"{src}\np={d['permutation_p']:.4f}, z={d['z_score']:+.2f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 3 — Adjacent-pair similarity vs shuffled baseline", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
