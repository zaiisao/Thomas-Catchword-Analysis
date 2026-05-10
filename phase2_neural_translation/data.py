"""Dataset + collate for the parallel-NT training set."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class ParallelDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer: Tokenizer, splits: tuple[str, ...]):
        self.tok = tokenizer
        self.bos = tokenizer.token_to_id("[BOS]")
        self.eos = tokenizer.token_to_id("[EOS]")
        self.pad = tokenizer.token_to_id("[PAD]")

        self.records = []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["split"] not in splits:
                    continue
                if not r["coptic_text"] or not r["syriac_text_consonantal"]:
                    continue
                self.records.append((r["coptic_text"], r["syriac_text_consonantal"],
                                     r["book"], r["chapter"], r["verse"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        c, s, book, ch, v = self.records[idx]
        # Encoder input: source + EOS (tokenizer post-processor adds EOS).
        # Decoder input: BOS + target;  Decoder target: target + EOS.
        src = self.tok.encode(c).ids                  # ends with EOS
        tgt = self.tok.encode(s).ids                  # ends with EOS
        dec_in = [self.bos] + tgt[:-1]                # shift right
        dec_out = tgt
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "dec_in": torch.tensor(dec_in, dtype=torch.long),
            "dec_out": torch.tensor(dec_out, dtype=torch.long),
            "ref": (book, ch, v, s),
        }


def make_collate(pad_id: int):
    def collate(batch):
        src_list = [b["src"] for b in batch]
        dec_in_list = [b["dec_in"] for b in batch]
        dec_out_list = [b["dec_out"] for b in batch]

        def pad_to(lst, max_len):
            return torch.stack([
                torch.cat([t, torch.full((max_len - len(t),), pad_id, dtype=torch.long)])
                if len(t) < max_len else t[:max_len]
                for t in lst
            ])

        src_max = max(len(t) for t in src_list)
        tgt_max = max(len(t) for t in dec_in_list)
        src = pad_to(src_list, src_max)
        dec_in = pad_to(dec_in_list, tgt_max)
        dec_out = pad_to(dec_out_list, tgt_max)
        src_pad_mask = (src == pad_id)
        tgt_pad_mask = (dec_in == pad_id)
        refs = [b["ref"] for b in batch]
        return {
            "src": src, "dec_in": dec_in, "dec_out": dec_out,
            "src_pad_mask": src_pad_mask, "tgt_pad_mask": tgt_pad_mask,
            "refs": refs,
        }
    return collate
