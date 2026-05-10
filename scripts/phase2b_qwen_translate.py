#!/usr/bin/env python3
"""
Phase 2B Step 3 — translate the 115 Gospel of Thomas logia + 10 control
passages from the Pauline epistles via Qwen3 (or any HF causal LM).

For each passage we generate VARIANTS_PER_PASSAGE=20 translations at
temperature=0.7 to give the downstream catchword analysis a distribution
rather than a point estimate. Resume support: per-passage JSON files; reruns
skip files that already exist.

Inputs:
  data/processed/got_logia/thomas_logia.jsonl       (115 logia)
  data/processed/parallel_corpus/sahidica_nt_coptic_tt.jsonl
                                                    (control passages)

Output:
  data/processed/llm_translations/{passage_id}.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
THOMAS = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"
COPTIC_NT = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "sahidica_nt_coptic_tt.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed" / "llm_translations"

SYRIAC_RE = re.compile(r"[܀-ݏ]")

VARIANTS_PER_PASSAGE = 20

SYSTEM_PROMPT = (
    "You are an expert translator of ancient languages. "
    "You translate Sahidic Coptic into Classical Syriac (Estrangela script). "
    "Produce only the Syriac translation — no transliteration, no explanation, "
    "no commentary. Output Syriac Unicode characters only."
)

STRICT_RETRY_PROMPT_PREFIX = (
    "You are translating ancient Sahidic Coptic into Classical Syriac.\n\n"
    "IMPORTANT RULES:\n"
    "- Output ONLY Syriac Unicode characters "
    "(ܐ ܒ ܓ ܕ ܗ ܘ ܙ ܚ ܛ ܝ ܟ ܠ ܡ ܢ ܣ ܥ ܦ ܨ ܩ ܪ ܫ ܬ).\n"
    "- Do NOT use Latin/English transliteration.\n"
    "- Do NOT include any explanation.\n"
    "- Do NOT use Arabic script.\n\n"
    "Coptic text:\n"
)

# Pauline epistle control passages — refs that don't parallel any Thomas logion
CONTROL_REFS = [
    ("Romans", 8, [28, 29, 30]),
    ("1_Corinthians", 13, [1, 2, 3]),
    ("1_Corinthians", 13, [4, 5, 6, 7]),
    ("Galatians", 5, [22, 23, 24]),
    ("Ephesians", 2, [8, 9, 10]),
    ("Philippians", 2, [5, 6, 7, 8]),
    ("Philippians", 4, [6, 7]),
    ("Colossians", 3, [12, 13, 14]),
    ("1_Thessalonians", 5, [16, 17, 18]),
    ("2_Timothy", 2, [15]),
]


def load_thomas_text():
    """{logion_int: full Coptic surface text}"""
    by_log = defaultdict(list)
    with THOMAS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                by_log[r["logion"]].append(r["text"])
    return {L: " ".join(parts) for L, parts in by_log.items()}


def load_control_text():
    """[{id: 'Romans_8_28-30', book, chapter, verses, coptic_text, is_control:True}, ...]"""
    by_book = defaultdict(dict)
    with COPTIC_NT.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_book[r["book"]].setdefault(r["chapter"], {})[r["verse"]] = r.get("text", "")
    out = []
    for book, ch, verses in CONTROL_REFS:
        verse_texts = []
        ok = True
        for v in verses:
            txt = by_book.get(book, {}).get(ch, {}).get(v, "")
            if not txt:
                ok = False
                break
            verse_texts.append(txt)
        if not ok:
            print(f"  WARNING: missing {book} {ch}:{verses}", file=sys.stderr)
            continue
        joined = " ".join(verse_texts)
        rng = f"{verses[0]}-{verses[-1]}" if len(verses) > 1 else str(verses[0])
        out.append({
            "id": f"control_{book}_{ch}_{rng}",
            "book": book, "chapter": ch, "verses": verses,
            "coptic_text": joined,
            "is_control": True,
        })
    return out


def load_model(model_name, device, quant):
    print(f"Loading {model_name} (quant={quant}) on {device}…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": torch.float16}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = "auto"
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif quant == "auto":
        kwargs["device_map"] = "auto"
    if "device_map" not in kwargs:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    return tok, model


def translate(tok, model, system, user, max_new_tokens=500,
               temperature=0.7, top_p=0.9):
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
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, top_p=top_p, repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()


def is_usable_syriac(text):
    syr = SYRIAC_RE.findall(text)
    n_syr = len(syr)
    total = len(re.sub(r"\s", "", text))
    return n_syr >= 5 and (n_syr / max(total, 1)) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quant", choices=["none", "auto", "4bit", "8bit"], default="none")
    ap.add_argument("--variants", type=int, default=VARIANTS_PER_PASSAGE)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--passages", default="all",
                     help="'all', 'thomas', 'controls', or a comma-list of logion numbers")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load passages ----
    thomas = load_thomas_text()
    print(f"Thomas logia: {len(thomas)}")
    controls = load_control_text()
    print(f"Control passages: {len(controls)}")

    # Build the passage list
    if args.passages == "all":
        thomas_passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                             "coptic_text": t, "is_control": False}
                            for L, t in sorted(thomas.items())]
        all_passages = thomas_passages + controls
    elif args.passages == "thomas":
        all_passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                          "coptic_text": t, "is_control": False}
                         for L, t in sorted(thomas.items())]
    elif args.passages == "controls":
        all_passages = controls
    else:
        wanted = {int(x) for x in args.passages.split(",")}
        all_passages = [{"id": f"logion_{L:03d}", "logion_number": L,
                          "coptic_text": t, "is_control": False}
                         for L, t in sorted(thomas.items()) if L in wanted]

    print(f"Will translate {len(all_passages)} passages × {args.variants} variants "
          f"= {len(all_passages) * args.variants} generations")

    tok, model = load_model(args.model, args.device, args.quant)

    t_total_start = time.time()
    n_done = 0
    for pi, p in enumerate(all_passages):
        out_path = OUT_DIR / f"{p['id']}.json"
        if out_path.exists():
            n_done += 1
            continue

        coptic = p["coptic_text"]
        print(f"\n[{pi+1}/{len(all_passages)}] {p['id']} "
              f"(content={len(coptic.split())} tokens, "
              f"is_control={p['is_control']})")

        variants = []
        good = 0
        t0 = time.time()
        for v in range(args.variants):
            success = False; err = None; text = ""
            try:
                text = translate(tok, model, SYSTEM_PROMPT,
                                  f"Translate this Sahidic Coptic text into "
                                  f"Classical Syriac:\n\n{coptic}",
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
                # If output isn't usable Syriac, try a strict retry once
                if not is_usable_syriac(text):
                    text2 = translate(tok, model, SYSTEM_PROMPT,
                                       STRICT_RETRY_PROMPT_PREFIX + coptic +
                                       "\n\nSyriac translation:",
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature)
                    if is_usable_syriac(text2):
                        text = text2
                success = True
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            variants.append({"variant": v, "syriac_text": text,
                              "success": success, **({"error": err} if err else {})})
            if success and is_usable_syriac(text):
                good += 1
            if v % 5 == 4:
                torch.cuda.empty_cache()

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "passage_id": p["id"],
                "logion_number": p.get("logion_number"),
                "coptic_text": coptic,
                "is_control": p["is_control"],
                "model": args.model,
                "temperature": args.temperature,
                "variants": variants,
            }, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t0
        n_done += 1
        rate = (time.time() - t_total_start) / max(n_done, 1)
        remain = (len(all_passages) - n_done) * rate
        print(f"  done {good}/{args.variants} good Syriac in {elapsed:.0f}s   "
              f"(ETA: {remain/60:.1f} min for remaining)")

    print()
    print(f"Total runtime: {(time.time() - t_total_start)/60:.1f} min")
    print(f"Output dir:    {OUT_DIR}")


if __name__ == "__main__":
    main()
