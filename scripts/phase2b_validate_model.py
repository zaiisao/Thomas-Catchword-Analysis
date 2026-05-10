#!/usr/bin/env python3
"""
Phase 2B Step 0 — validate that the chosen LLM can produce Classical Syriac.

Strategy: load the model, run 3-4 test prompts (Coptic GoT logia covering
Perrin's catchword examples + a simple English→Syriac fallback), inspect
outputs for Syriac script presence.

If direct Coptic→Syriac fails, try the two-step Coptic→English→Syriac
fallback documented in the task spec. If both fail, switch model.

Usage:
  python scripts/phase2b_validate_model.py [--model Qwen/Qwen3-14B] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"

SYRIAC_RE = re.compile(r"[܀-ݏ]")

SYSTEM_PROMPT = (
    "You are an expert translator of ancient languages. "
    "You translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Produce only the Syriac translation — no transliteration, no explanation, "
    "no commentary. Output Syriac Unicode characters only."
)


def load_thomas_text():
    """Return {logion_int: full Coptic surface text}."""
    by_log = {}
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            L = r["logion"]
            by_log.setdefault(L, []).append(r.get("text", ""))
    return {L: " ".join(parts) for L, parts in by_log.items()}


def chat_translate(model, tok, system, user, max_new_tokens=400,
                    temperature=0.7, top_p=0.9):
    """Use the chat template for an instruction-tuned model."""
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
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    text = tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()
    return text, elapsed


def diagnose(text):
    """Return summary dict for one output."""
    syr = SYRIAC_RE.findall(text)
    n_syr = len(syr)
    total_non_ws = len(re.sub(r"\s", "", text))
    ratio = n_syr / max(total_non_ws, 1)
    has_latin = bool(re.search(r"[a-zA-Z]{4,}", text))
    has_arabic = bool(re.search(r"[؀-ۿ]", text))
    return {
        "len": len(text),
        "n_syr": n_syr,
        "syr_ratio": round(ratio, 3),
        "has_latin": has_latin,
        "has_arabic": has_arabic,
        "preview": text[:160],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    args = ap.parse_args()

    print(f"Loading {args.model} (quant={args.quant}) …")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = {"torch_dtype": torch.float16}
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                            bnb_4bit_compute_dtype=torch.float16)
    elif args.quant == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if args.quant == "none":
        # Single-device load
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    else:
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    print(f"  device(s): {set(p.device for p in model.parameters())}")

    thomas = load_thomas_text()
    print()
    print(f"Thomas logia loaded: {len(thomas)}")

    # ---- TEST 1: Direct Coptic → Syriac (Perrin example logia) ----
    test_logia = [10, 11, 86, 1]
    print()
    print("=== TEST 1: Direct Coptic → Syriac ===")
    test1_results = {}
    direct_works = True
    for L in test_logia:
        coptic = thomas.get(L, "")
        if not coptic:
            print(f"\nLogion {L}: no Coptic text loaded; skipping")
            continue
        print(f"\nLogion {L}  (Coptic, {len(coptic.split())} surface tokens):")
        print(f"  {coptic[:200]}")
        out, dt = chat_translate(
            model, tok, SYSTEM_PROMPT,
            f"Translate this Sahidic Coptic text into Classical Syriac:\n\n{coptic}",
        )
        d = diagnose(out)
        d["seconds"] = round(dt, 2)
        test1_results[L] = {"output": out, **d}
        print(f"  →  {d}")
        # Threshold: at least 10 Syriac chars AND ratio > 0.5
        if d["n_syr"] < 10 or d["syr_ratio"] < 0.5:
            direct_works = False

    # ---- TEST 2: Two-step fallback (only if direct mostly fails) ----
    if not direct_works:
        print()
        print("=== TEST 2: Two-step (Coptic → English → Syriac) — direct mostly failed ===")
        test2_results = {}
        for L in test_logia[:2]:
            coptic = thomas.get(L, "")
            # Step A: Coptic → English
            en, dt_a = chat_translate(
                model, tok,
                "You translate Sahidic Coptic into literal English.",
                f"Translate this Sahidic Coptic text into literal English:\n\n{coptic}",
            )
            print(f"\nLogion {L} EN: {en[:200]}")
            # Step B: English → Syriac
            syr, dt_b = chat_translate(
                model, tok,
                "You translate English into Classical Syriac (Estrangela script). "
                "Output only Syriac Unicode characters.",
                f"Translate this English text into Classical Syriac:\n\n{en}",
            )
            d = diagnose(syr)
            d["seconds"] = round(dt_a + dt_b, 2)
            d["english_intermediate"] = en[:200]
            test2_results[L] = {"output": syr, **d}
            print(f"  →  {d}")
    else:
        test2_results = None

    # ---- Verdict ----
    print()
    print("=" * 70)
    syr_ok = sum(1 for d in test1_results.values()
                 if d["n_syr"] >= 10 and d["syr_ratio"] >= 0.5)
    print(f"VERDICT: direct Coptic→Syriac usable on {syr_ok}/{len(test1_results)} logia")
    if syr_ok == len(test1_results):
        print(f"  → Proceed with direct translation in scripts/phase2b_llm_translate.py")
    elif syr_ok >= 1:
        print(f"  → Mixed results; consider direct + retry-with-strict-prompt")
    else:
        print(f"  → Direct translation does not work. Use two-step fallback.")

    out_path = REPO_ROOT / "data" / "processed" / "phase2b_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "quant": args.quant,
            "direct_test": test1_results,
            "two_step_test": test2_results,
            "verdict_syr_usable": syr_ok,
            "n_test_logia": len(test1_results),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved validation report → {out_path}")


if __name__ == "__main__":
    main()
