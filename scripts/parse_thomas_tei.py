#!/usr/bin/env python3
"""
Parse the Coptic Gospel of Thomas TEI/XML into logion-segmented JSONL.

The Coptic SCRIPTORIUM thomas-gospel TEI file uses:
  <div1 n="N">       — N=0 is the Prologue, N=1..114 are the 114 sayings
    <p n="K">         — K=1..n_K are sub-paragraphs (Perrin cites these as "14.4", "16.2", etc.)
      <s style="...">
        <phr>
          <w type="POS" lemma="LEMMA">FORM</w>
          ...

This is the same source the CoNLL-U release was derived from — but using TEI directly
gets us logion + paragraph boundaries that the CoNLL-U release does not preserve.

Output JSONL (one record per logion-paragraph pair):
  {"logion": 14, "paragraph": 4, "is_prologue": false,
   "ref": "14.4",
   "text": "...", "tokens": [{"form", "lemma", "pos"}, ...]}

Run from repo root:
  python scripts/parse_thomas_tei.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parent.parent
TEI_PATH = REPO_ROOT / "data" / "raw" / "coptic_repo" / "thomas-gospel" / "thomas.gospel_TEI" / "thomas_gospel.xml"
OUTPUT = REPO_ROOT / "data" / "processed" / "got_logia" / "thomas_logia.jsonl"


def extract_words(elem):
    """Recursively yield (form, lemma, pos) for every <w> descendant."""
    for w in elem.iter("w"):
        form_parts = []
        for child in w.iter():
            if child.tag in ("w", "m"):
                if child.text:
                    form_parts.append(child.text.strip())
                if child.tail and child is not w:
                    form_parts.append(child.tail.strip())
            else:
                if child.text:
                    form_parts.append(child.text.strip())
        if not form_parts and w.text:
            form_parts.append(w.text.strip())
        form = "".join(p for p in form_parts if p)
        if not form:
            continue
        yield {
            "form": form,
            "lemma": w.get("lemma"),
            "pos": w.get("type"),
            "lang": w.get("{http://www.w3.org/XML/1998/namespace}lang"),
        }


def main():
    if not TEI_PATH.exists():
        sys.exit(f"Missing: {TEI_PATH}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(str(TEI_PATH), parser=parser)
    root = tree.getroot()

    # Strip default TEI namespace if present so xpath stays simple
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    body = root.find(".//body") or root
    records = []
    n_logia_seen = set()

    for div1 in body.iter("div1"):
        n_attr = div1.get("n")
        if n_attr is None:
            continue
        try:
            logion = int(n_attr)
        except ValueError:
            continue
        n_logia_seen.add(logion)

        paragraphs = list(div1.findall("p"))
        if not paragraphs:
            paragraphs = [div1]

        for para in paragraphs:
            p_attr = para.get("n")
            try:
                paragraph_n = int(p_attr) if p_attr is not None else 1
            except ValueError:
                paragraph_n = 1
            tokens = list(extract_words(para))
            if not tokens:
                continue
            text = " ".join(t["form"] for t in tokens)
            records.append({
                "logion": logion,
                "paragraph": paragraph_n,
                "is_prologue": logion == 0,
                "ref": f"{logion}.{paragraph_n}",
                "text": text,
                "tokens": tokens,
            })

    with OUTPUT.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    expected = set(range(0, 115))  # 0=Prologue, 1..114=sayings
    missing = sorted(expected - n_logia_seen)
    extra = sorted(n_logia_seen - expected)

    print(f"Logia seen:        {len(n_logia_seen)} (expected 115: Prologue + 114)")
    if missing:
        print(f"  MISSING logia:   {missing}")
    if extra:
        print(f"  UNEXPECTED logia: {extra}")
    print(f"Total records:     {len(records)} (logion-paragraph pairs)")
    n_tokens = sum(len(r["tokens"]) for r in records)
    print(f"Total tokens:      {n_tokens}")
    print(f"Output:            {OUTPUT}")

    # Per-logion paragraph counts (for spot-checking)
    counts: dict[int, int] = {}
    for r in records:
        counts[r["logion"]] = counts.get(r["logion"], 0) + 1
    multi_p = [(L, c) for L, c in counts.items() if c > 1]
    if multi_p:
        print(f"\nLogia with multiple paragraphs: {len(multi_p)}")
        for L, c in sorted(multi_p)[:20]:
            print(f"  Logion {L}: {c} paragraphs")
        if len(multi_p) > 20:
            print(f"  ... and {len(multi_p)-20} more")


if __name__ == "__main__":
    main()
