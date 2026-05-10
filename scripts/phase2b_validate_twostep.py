#!/usr/bin/env python3
"""
Phase 2B Step 0b — compare direct vs two-step Coptic→Syriac on Qwen3-14B.

The earlier validation showed direct Coptic→Syriac produces Syriac script,
but the *content* is heavily repetitive (Logion 11 cycled "ܘܥܒܪܬܐ ܕܐܢܫܐ
ܘܥܒܪܬܐ ܕܡܠܟܐ" indefinitely). The two-step path goes Coptic → English →
Syriac; English→Syriac has more training signal in any general-purpose
multilingual LLM.

Decision criteria:
  - repetition score: longest repeated bigram block / total bigrams
  - unique_token_ratio: unique tokens / total tokens
  - syr_ratio: Syriac chars / non-whitespace chars

Lower repetition + higher unique-token ratio = better quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"

SYRIAC_RE = re.compile(r"[܀-ݏ]")

PROMPT_DIRECT_SYS = (
    "You are an expert translator of ancient languages. "
    "You translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Produce only the Syriac translation — no transliteration, no explanation, "
    "no commentary. Output Syriac Unicode characters only."
)
PROMPT_TWOSTEP_A_SYS = "You translate Sahidic Coptic into literal English."
PROMPT_TWOSTEP_B_SYS = (
    "You translate English into Classical Syriac (Estrangela script). "
    "Output only Syriac Unicode characters. No transliteration."
)


def load_thomas_text():
    by_log = {}
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_log.setdefault(r["logion"], []).append(r.get("text", ""))
    return {L: " ".join(parts) for L, parts in by_log.items()}


def quality_metrics(text):
    """Compute repetition and uniqueness diagnostics."""
    if not text or not text.strip():
        return {"len": 0, "n_tokens": 0, "unique_ratio": 0,
                "rep_bigram_max": 0, "n_syr": 0, "syr_ratio": 0}
    n_syr = len(SYRIAC_RE.findall(text))
    total = len(re.sub(r"\s", "", text))
    syr_ratio = n_syr / max(total, 1)
    tokens = text.split()
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {"len": len(text), "n_tokens": 0, "unique_ratio": 0,
                "rep_bigram_max": 0, "n_syr": n_syr, "syr_ratio": syr_ratio}
    unique_ratio = len(set(tokens)) / n_tokens
    bigrams = list(zip(tokens, tokens[1:]))
    if bigrams:
        bg_counts = Counter(bigrams)
        rep_bigram_max = max(bg_counts.values()) / len(bigrams)
    else:
        rep_bigram_max = 0
    return {
        "len": len(text), "n_tokens": n_tokens,
        "unique_ratio": round(unique_ratio, 3),
        "rep_bigram_max": round(rep_bigram_max, 3),
        "n_syr": n_syr, "syr_ratio": round(syr_ratio, 3),
    }


def chat(model, tok, system, user, max_new_tokens=400, temperature=0.7):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, top_p=0.9, repetition_penalty=1.15,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True).strip(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"Loading {args.model}…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                   dtype=torch.float16).to(args.device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    thomas = load_thomas_text()

    test_logia = [10, 11, 86, 1, 14, 16]
    results = {"direct": {}, "twostep": {}}

    for L in test_logia:
        coptic = thomas.get(L, "")
        if not coptic:
            continue
        print(f"\n=== Logion {L} ({len(coptic.split())} Coptic tokens) ===")

        # DIRECT
        direct_out, dt_d = chat(model, tok, PROMPT_DIRECT_SYS,
                                 f"Translate this Sahidic Coptic text into "
                                 f"Classical Syriac:\n\n{coptic}")
        qd = quality_metrics(direct_out)
        qd["seconds"] = round(dt_d, 1)
        qd["preview"] = direct_out[:200]
        results["direct"][L] = qd
        print(f"  DIRECT: rep={qd['rep_bigram_max']:.3f} "
              f"unique={qd['unique_ratio']:.3f} syr%={qd['syr_ratio']:.3f}")
        print(f"    {direct_out[:160]}")

        # TWO-STEP
        en_out, dt_a = chat(model, tok, PROMPT_TWOSTEP_A_SYS,
                             f"Translate this Sahidic Coptic text into "
                             f"literal English:\n\n{coptic}",
                             max_new_tokens=400, temperature=0.3)
        syr_out, dt_b = chat(model, tok, PROMPT_TWOSTEP_B_SYS,
                              f"Translate this English text into Classical "
                              f"Syriac:\n\n{en_out}",
                              max_new_tokens=400, temperature=0.7)
        qt = quality_metrics(syr_out)
        qt["seconds"] = round(dt_a + dt_b, 1)
        qt["preview"] = syr_out[:200]
        qt["english_intermediate"] = en_out[:200]
        results["twostep"][L] = qt
        print(f"  EN: {en_out[:160]}")
        print(f"  TWO-STEP: rep={qt['rep_bigram_max']:.3f} "
              f"unique={qt['unique_ratio']:.3f} syr%={qt['syr_ratio']:.3f}")
        print(f"    {syr_out[:160]}")

    # Summary
    print()
    print("=" * 70)
    print("AGGREGATE QUALITY METRICS (lower rep + higher unique = better)")
    print("=" * 70)
    print(f"{'Method':<10s} {'rep_max':>8s} {'unique':>8s} {'syr%':>6s} {'n_tok':>6s}")
    for method in ("direct", "twostep"):
        rs = results[method].values()
        rep = sum(r["rep_bigram_max"] for r in rs) / max(len(rs), 1)
        unq = sum(r["unique_ratio"] for r in rs) / max(len(rs), 1)
        sr  = sum(r["syr_ratio"] for r in rs) / max(len(rs), 1)
        nt  = sum(r["n_tokens"] for r in rs) / max(len(rs), 1)
        print(f"{method:<10s} {rep:>8.3f} {unq:>8.3f} {sr:>6.3f} {nt:>6.0f}")

    out_path = REPO_ROOT / "data" / "processed" / "phase2b_validation_twostep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
