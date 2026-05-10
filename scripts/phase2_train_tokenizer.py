#!/usr/bin/env python3
"""
Phase 2.2 — train a joint BPE tokenizer over the train + val splits of the
Coptic↔Syriac parallel corpus.

We use a single tokenizer for both source (Coptic) and target (Syriac)
because:
  - the corpus is small (~7000 verses); shared subword inventory increases
    coverage of low-frequency tokens
  - the project guide specifies "BPE tokenization trained on combined
    Coptic+Syriac"
  - special tokens (BOS/EOS/PAD) are language-agnostic

Reads:  data/processed/parallel_corpus/coptic_syriac_pairs.jsonl
Writes: data/processed/tokenizer/bpe.json
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS = REPO_ROOT / "data" / "processed" / "parallel_corpus" / "coptic_syriac_pairs.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed" / "tokenizer"
OUT_FILE = OUT_DIR / "bpe.json"

VOCAB_SIZE = 16000
MIN_FREQUENCY = 2

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def iter_corpus():
    """Stream both Coptic and Syriac texts from train+val splits.

    We use CONSONANTAL Syriac as the target form (vowel diacritics stripped).
    Two reasons:
      - Catchword detection runs on consonantal forms anyway.
      - Pointed Syriac fragments under BPE because each diacritic mark is
        a distinct character; consonantal Syriac compresses 2-3x better
        with the same vocab budget.
    """
    with PAIRS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] not in ("train", "val"):
                continue
            yield r["coptic_text"]
            yield r["syriac_text_consonantal"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE}, min_freq={MIN_FREQUENCY})…")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL,
        # Mark word-final subwords with </w> so the BPEDecoder (suffix='</w>')
        # can re-insert spaces on decode. Without this the decoder reassembles
        # subwords into a single continuous string per sentence — which silently
        # breaks any downstream tool that splits on whitespace.
        end_of_word_suffix="</w>",
        show_progress=False,
    )

    # Materialize corpus to count lines + size
    corpus = list(iter_corpus())
    print(f"  corpus: {len(corpus)} sentences "
          f"({sum(len(s) for s in corpus):,} chars)")

    tok.train_from_iterator(corpus, trainer=trainer)

    # Add BOS/EOS template for the encoder (source side).
    # The decoder side gets BOS prepended manually during training; we keep
    # the tokenizer language-agnostic.
    tok.post_processor = processors.TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[("[EOS]", tok.token_to_id("[EOS]"))],
    )

    tok.save(str(OUT_FILE))
    print(f"  saved → {OUT_FILE}")
    print(f"  vocab size: {tok.get_vocab_size()}")

    # Sanity check: encode and decode a Coptic and a Syriac sentence
    test_c = "ⲡⲉϫⲉ ⲓⲏⲥⲟⲩⲥ"
    test_s = "ܐܡܪ ܠܗܘܢ ܝܫܘܥ"
    print()
    for text in (test_c, test_s):
        enc = tok.encode(text)
        print(f"  '{text}' -> {len(enc.tokens)} tokens: {enc.tokens}")
        print(f"          ids: {enc.ids}")


if __name__ == "__main__":
    main()
