#!/usr/bin/env python3
"""
Phase 3.1 — extract strophes from Syriac poetic corpora for contrastive
training.

Sources, in priority order (markup conventions vary):
  - Odes of Solomon (~40 files):   <lg type="stanza" n="N">
  - Ephrem the Syrian (~111 files): <div type="section" n="N">
  - Jacob of Serug (~39 files):     <div type="section" n="N">  (similar to Ephrem)

Output JSONL (one record per strophe):
  {"author", "source_file", "work_title",
   "strophe_index", "strophe_label",  // numeric or label like "section 5"
   "lines": ["...", ...],
   "text": "joined text",
   "text_consonantal": "vowels stripped",
   "n_tokens": int}

Each (author, source_file) gets its strophes in document order, allowing
downstream code to construct (consecutive-strophe) positive pairs.

Usage:
  python scripts/phase3_extract_strophes.py
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parent.parent
TEI_DIR = REPO_ROOT / "data" / "raw" / "peshitta_repo" / "data" / "tei"
OUT = REPO_ROOT / "data" / "processed" / "syriac_strophes.jsonl"

TARGET_AUTHORS = {
    "Ephrem the Syrian":  "Ephrem",
    "Jacob of Serugh":    "Jacob",
    "Narsai":             "Narsai",
}

NS = "{http://www.tei-c.org/ns/1.0}"


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def get_author(root) -> str | None:
    for a in root.iter(f"{NS}author"):
        text = "".join(a.itertext()).strip()
        if text:
            return text
    return None


def get_title(root) -> str:
    title = root.find(f".//{NS}title[@level='a']")
    if title is not None:
        return " ".join("".join(title.itertext()).split())[:120]
    title = root.find(f".//{NS}title")
    if title is not None:
        return " ".join("".join(title.itertext()).split())[:120]
    return ""


def is_poetic_div(elem) -> bool:
    """Heuristic: a <div> with type 'section' or 'stanza' that contains <l> lines."""
    t = elem.get("type", "")
    if t in ("section", "stanza"):
        # must contain at least one <l>
        for _ in elem.iter(f"{NS}l"):
            return True
    return False


def extract_strophes_lg(body) -> list[dict]:
    """Odes of Solomon style: <lg type='stanza' n='N'>."""
    out = []
    for lg in body.iter(f"{NS}lg"):
        if lg.get("type") not in ("stanza", "verse", None):
            continue
        n = lg.get("n", "")
        lines = []
        for l in lg.iter(f"{NS}l"):
            txt = " ".join("".join(l.itertext()).split())
            if txt:
                lines.append(txt)
        if lines:
            out.append({"label": f"stanza {n}", "lines": lines})
    return out


def extract_strophes_section_div(body) -> list[dict]:
    """Ephrem / Jacob style: <div type='section' n='N'>."""
    out = []
    for div in body.iter(f"{NS}div"):
        if div.get("type") not in ("section", "stanza"):
            continue
        n = div.get("n", "")
        lines = []
        for l in div.iter(f"{NS}l"):
            txt = " ".join("".join(l.itertext()).split())
            if txt:
                lines.append(txt)
        if lines:
            out.append({"label": f"section {n}", "lines": lines})
    return out


def extract_strophes_l_grouped(body, lines_per_strophe: int = 4) -> list[dict]:
    """Narsai style: every line is <l n='N'> at the body level.
    We group consecutive lines into pseudo-strophes."""
    all_lines = []
    for l in body.iter(f"{NS}l"):
        txt = " ".join("".join(l.itertext()).split())
        if txt:
            all_lines.append(txt)
    out = []
    for i in range(0, len(all_lines), lines_per_strophe):
        chunk = all_lines[i:i + lines_per_strophe]
        if chunk:
            out.append({"label": f"lines {i+1}-{i+len(chunk)}",
                        "lines": chunk})
    return out


def parse_file(path: Path) -> list[dict]:
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    try:
        tree = etree.parse(str(path), parser=parser)
    except etree.XMLSyntaxError:
        return []
    root = tree.getroot()
    if root is None:
        return []

    author = get_author(root)
    if author not in TARGET_AUTHORS and "Odes of Solomon" not in get_title(root):
        return []

    title = get_title(root)
    body = root.find(f".//{NS}body")
    if body is None:
        return []

    # Try strategies in order: stanza-lg → section-div → grouped-l
    strophes = extract_strophes_lg(body)
    strategy = "lg"
    if not strophes:
        strophes = extract_strophes_section_div(body)
        strategy = "div_section"
    if not strophes:
        strophes = extract_strophes_l_grouped(body)
        strategy = "l_grouped"

    if not strophes:
        return []

    out = []
    label_author = TARGET_AUTHORS.get(author, "Solomon" if "Odes of Solomon" in title else "Other")
    for i, st in enumerate(strophes):
        text = " ".join(st["lines"])
        out.append({
            "author": label_author,
            "author_full": author or "",
            "source_file": path.name,
            "work_title": title,
            "strophe_index": i,
            "strophe_label": st["label"],
            "extraction_strategy": strategy,
            "lines": st["lines"],
            "text": text,
            "text_consonantal": strip_marks(text),
            "n_tokens": len(text.split()),
        })
    return out


def main():
    if not TEI_DIR.exists():
        raise SystemExit(f"Missing {TEI_DIR}. Run scripts/fetch_data.sh first.")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    counts_per_author: dict[str, int] = {}
    counts_per_file: dict[str, int] = {}
    n_files_used = 0
    n_strophes = 0

    with OUT.open("w", encoding="utf-8") as out:
        for path in sorted(TEI_DIR.glob("*.xml")):
            recs = parse_file(path)
            if not recs:
                continue
            n_files_used += 1
            counts_per_file[path.name] = len(recs)
            for r in recs:
                counts_per_author[r["author"]] = counts_per_author.get(r["author"], 0) + 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_strophes += 1

    print(f"Used {n_files_used} files; emitted {n_strophes} strophes -> {OUT}")
    print(f"\nStrophes per author:")
    for a, c in sorted(counts_per_author.items(), key=lambda kv: -kv[1]):
        print(f"  {a:<10s}  {c:>5d}")
    print(f"\nLargest single files:")
    for fn, c in sorted(counts_per_file.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {fn:<10s}  {c:>4d} strophes")


if __name__ == "__main__":
    main()
