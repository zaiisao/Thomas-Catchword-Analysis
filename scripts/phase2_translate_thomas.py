#!/usr/bin/env python3
"""
Phase 2.4 — translate the Coptic Gospel of Thomas with the trained
transformer, run the catchword detector on the neural Syriac output,
and compare to Perrin's reported numbers.

Inputs:
  data/processed/checkpoints/best.pt
  data/processed/tokenizer/bpe.json
  data/processed/got_logia/thomas_logia.jsonl   (the Coptic input)

Outputs:
  data/processed/thomas_neural_syriac.jsonl     (neural translations per logion)
  data/processed/phase2_results.json            (catchword counts vs Perrin)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase2_neural_translation.model import Seq2SeqTransformer  # noqa: E402
from phase1_montecarlo.catchword_detector import CatchwordDetector  # noqa: E402

CKPT = REPO_ROOT / "data" / "processed" / "checkpoints" / "best.pt"
TOK_PATH = REPO_ROOT / "data" / "processed" / "tokenizer" / "bpe.json"
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
OUT_TR = REPO_ROOT / "data" / "processed" / "thomas_neural_syriac.jsonl"
OUT_RES = REPO_ROOT / "data" / "processed" / "phase2_results.json"


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = Seq2SeqTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=args["d_model"], nhead=args["nhead"],
        num_encoder_layers=args["n_layers"],
        num_decoder_layers=args["n_layers"],
        dim_feedforward=args["dim_ff"], dropout=args["dropout"],
        pad_id=ckpt["pad_id"], bos_id=ckpt["bos_id"], eos_id=ckpt["eos_id"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    model, ckpt = load_model(device)

    # Group Thomas tokens by logion (concatenate paragraphs)
    logia: dict[int, list[dict]] = defaultdict(list)
    coptic_text_per_logion: dict[int, list[str]] = defaultdict(list)
    for line in THOMAS.open():
        r = json.loads(line)
        logia[r["logion"]].extend(r["tokens"])
        coptic_text_per_logion[r["logion"]].append(r["text"])
    sorted_L = sorted(logia.keys())

    print(f"Translating {len(sorted_L)} logia (beam={args.beam_size})…")
    OUT_TR.parent.mkdir(parents=True, exist_ok=True)

    # Translate each logion. Each logion is a sequence of paragraphs;
    # we translate the concatenated text per logion.
    neural_per_logion: dict[int, dict] = {}
    with OUT_TR.open("w", encoding="utf-8") as out, torch.no_grad():
        for L in sorted_L:
            coptic_text = " ".join(coptic_text_per_logion[L])
            enc = tok.encode(coptic_text)
            src_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
            src_pad = torch.zeros_like(src_ids, dtype=torch.bool)
            beams = model.beam_search(
                src_ids, src_pad,
                beam_size=args.beam_size, max_len=args.max_len,
            )
            best = beams[0] if beams else []
            # Strip BOS / EOS / PAD
            best = [t for t in best if t not in (pad_id, bos_id, eos_id)]
            syriac_text = tok.decode(best, skip_special_tokens=True)
            rec = {
                "logion": L,
                "coptic_text": coptic_text[:200],
                "neural_syriac_text": syriac_text,
                "neural_syriac_tokens": tok.decode(best).split(),
                "n_beams": len(beams),
            }
            neural_per_logion[L] = rec
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_TR}")

    # Run catchword detector on the neural Syriac.
    # We don't have lemmas, so we tokenize on whitespace and treat each word's
    # consonantal form AS the lemma (poor proxy, but uniform).
    print("\nRunning catchword detector on neural translations…")
    det = CatchwordDetector("syriac",
                            phonological_threshold=0.65,
                            require_content_pos=False)  # no POS in NMT output

    pair_counts = []
    perrin_pairs = [(10,11),(16,17),(82,83),(29,30),(85,86),
                    (14,15),(46,47),(113,114),(13,14),(17,18)]

    def make_tokens(text: str) -> list[dict]:
        # Treat each surface word as both form and lemma (no morph available)
        return [{"form": w, "lemma": w, "parse": "MS-EMP"}
                for w in text.split() if w]

    cw_left = set()
    cw_right = set()
    for i, L in enumerate(sorted_L[:-1]):
        Ln = sorted_L[i + 1]
        ta = make_tokens(neural_per_logion[L]["neural_syriac_text"])
        tb = make_tokens(neural_per_logion[Ln]["neural_syriac_text"])
        cws = det.detect(ta, tb)
        pair_counts.append({"a": L, "b": Ln, "n": len(cws),
                            "cws": [{"a": cw.token_a["lemma"],
                                     "b": cw.token_b["lemma"],
                                     "type": cw.link_type,
                                     "score": cw.score}
                                    for cw in cws[:5]]})
        if len(cws) > 0:
            cw_right.add(L); cw_left.add(Ln)

    total = sum(p["n"] for p in pair_counts)
    n_logia = len(sorted_L)
    both = cw_left & cw_right
    iso = set(sorted_L) - cw_left - cw_right
    one_side = (cw_left ^ cw_right)

    # Sanity check: are translations degenerate (e.g., same output for everything)?
    n_unique_outputs = len({neural_per_logion[L]["neural_syriac_text"] for L in sorted_L})
    avg_output_len = sum(len(neural_per_logion[L]["neural_syriac_text"].split())
                         for L in sorted_L) / n_logia
    print(f"\n  Unique neural outputs: {n_unique_outputs}/{n_logia}")
    print(f"  Mean output length:    {avg_output_len:.1f} tokens")
    if n_unique_outputs < n_logia * 0.5:
        print("  WARNING: model is producing many duplicate outputs — "
              "translation quality is likely poor. Catchword counts may be inflated.")

    summary = {
        "checkpoint_epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "model_size_M_params": sum(p.numel() for p in model.parameters()) / 1e6,
        "beam_size": args.beam_size,
        "n_logia": n_logia,
        "n_adjacent_pairs": len(pair_counts),
        "neural_total_catchwords": total,
        "neural_both_pct": 100 * len(both) / n_logia,
        "neural_one_pct": 100 * len(one_side) / n_logia,
        "neural_iso_pct": 100 * len(iso) / n_logia,
        "perrin_total": 502,
        "perrin_both_pct": 89.0,
        "perrin_one_pct": 11.0,
        "perrin_iso_pct": 0.0,
        "phase1_mc_mean_total": 195.4,
        "phase1_mc_p_geq_perrin": 0.0,
        "perrin_specific_pairs": {
            f"{a}-{b}": next((p["n"] for p in pair_counts if p["a"] == a and p["b"] == b), None)
            for a, b in perrin_pairs
        },
        "all_pair_counts": pair_counts,
    }
    with OUT_RES.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("PHASE 2 RESULTS")
    print("=" * 70)
    print(f"  Neural translation total catchwords:  {total}")
    print(f"  Phase 1 MC mean (random translation): 195.4")
    print(f"  Perrin's claim:                       502")
    print()
    print(f"  Neural connectivity: both={summary['neural_both_pct']:.1f}%   "
          f"one={summary['neural_one_pct']:.1f}%   iso={summary['neural_iso_pct']:.1f}%")
    print(f"  Perrin reports:      both=89.0%   one=11.0%   iso=0.0%")
    print()
    print(f"  Saved {OUT_RES}")


if __name__ == "__main__":
    main()
