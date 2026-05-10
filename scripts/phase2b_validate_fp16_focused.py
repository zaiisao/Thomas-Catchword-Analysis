#!/usr/bin/env python3
"""
Targeted fp16 retest of Qwen3-32B on the 3 logia where 4-bit failed worst:
  - Logion 10 (fire): 4-bit produced "ܢⲩⲣⲩⲣⲩ" (close-miss); two-step EN
                       hallucinated Sinai/Solomon
  - Logion 11 (light): 4-bit fell into "ܕⲩⲩⲣⲣ" repetition; EN was
                       "shepherd shepherd shepherd"
  - Logion 86 (foxes/holes): 4-bit decayed into numeric garbage; 24% syr%

5 variants per logion, fp16 across 2 GPUs.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"

SYRIAC_RE = re.compile(r"[܀-ݏ]")
NURA_RE = re.compile(r"\bܢ[ܘܼ]+ܪ[ܐ]?\b")    # nūrā = fire
NUHRA_RE = re.compile(r"\bܢ[ܘܼ]+ܗ[ܐ]?ܪ[ܐ]?\b") # nuhrā = light

SYSTEM_PROMPT = (
    "You are an expert translator of ancient languages. "
    "You translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Produce only the Syriac translation — no transliteration, no explanation, "
    "no commentary. Output Syriac Unicode characters only."
)


def load_thomas():
    by_log = {}
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_log.setdefault(r["logion"], []).append(r.get("text", ""))
    return {L: " ".join(p) for L, p in by_log.items()}


def quality(text):
    if not text or not text.strip():
        return {"len": 0, "n_syr": 0, "syr_ratio": 0, "rep_max": 0,
                "unique_ratio": 0, "has_nura": False, "has_nuhra": False}
    n_syr = len(SYRIAC_RE.findall(text))
    total = len(re.sub(r"\s", "", text))
    sr = n_syr / max(total, 1)
    toks = text.split()
    if toks:
        bigs = list(zip(toks, toks[1:]))
        rep = max(Counter(bigs).values()) / max(len(bigs), 1) if bigs else 0
        uniq = len(set(toks)) / len(toks)
    else:
        rep = uniq = 0
    return {
        "len": len(text), "n_syr": n_syr,
        "syr_ratio": round(sr, 3),
        "rep_max": round(rep, 3),
        "unique_ratio": round(uniq, 3),
        "has_nura":  bool(NURA_RE.search(text)),
        "has_nuhra": bool(NUHRA_RE.search(text)),
    }


def main():
    model_name = "Qwen/Qwen3-32B"
    print(f"Loading {model_name} fp16 across all GPUs (device_map=auto)…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    devices = set(p.device for p in model.parameters())
    print(f"  weights distributed across {len(devices)} device(s): {devices}")

    thomas = load_thomas()
    targets = [
        (10, "fire / world / I keep watch until it blazes"),
        (11, "this heaven passes / dead vs living / become two"),
        (86, "foxes have holes / birds have nests / son of man"),
    ]

    n_var = 5
    results = {}
    for L, gloss in targets:
        coptic = thomas.get(L, "")
        print(f"\n=== Logion {L} ({gloss}) — {len(coptic.split())} Coptic tokens ===")
        results[L] = []
        for v in range(n_var):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                     f"Translate this Sahidic Coptic text into Classical Syriac:\n\n{coptic}"}]
            try:
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=400, temperature=0.7,
                    do_sample=True, top_p=0.9, repetition_penalty=1.15,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()
            q = quality(text)
            q["seconds"] = round(time.time() - t1, 1)
            q["preview"] = text[:200]
            results[L].append(q)
            mark = []
            if q["has_nura"]:  mark.append("ܢⲩⲣⲣ✓")
            if q["has_nuhra"]: mark.append("ܢⲩⲗⲩⲣⲣ✓")
            print(f"  v{v}: rep={q['rep_max']:.3f} uniq={q['unique_ratio']:.3f} "
                  f"syr%={q['syr_ratio']:.3f} {' '.join(mark) if mark else '[no perrin]'}")
            print(f"    {text[:160]}")

    print()
    print("=" * 70)
    print("SUMMARY (Qwen3-32B fp16, 5 variants per logion)")
    print("=" * 70)
    print(f"{'Logion':<8s} {'has nūrā':>10s} {'has nuhrā':>11s} "
          f"{'mean syr%':>10s} {'mean unique':>12s}")
    for L, _ in targets:
        rs = results[L]
        n_nura = sum(r["has_nura"] for r in rs)
        n_nuhra = sum(r["has_nuhra"] for r in rs)
        mean_sr = sum(r["syr_ratio"] for r in rs) / len(rs)
        mean_uq = sum(r["unique_ratio"] for r in rs) / len(rs)
        print(f"{L:<8d} {n_nura}/{len(rs):>8s} {n_nuhra}/{len(rs):>9s} "
              f"{mean_sr:>10.3f} {mean_uq:>12.3f}")

    out_path = REPO_ROOT / "data" / "processed" / "phase2b_qwen32b_fp16_focused.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"model": model_name, "dtype": "fp16",
                    "device_map": "auto", "n_variants": n_var,
                    "results": {str(L): rs for L, rs in results.items()}},
                   f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
