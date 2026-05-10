#!/usr/bin/env bash
# One-shot reproducible data acquisition for the project.
# Idempotent: existing repos/files are not re-fetched.
#
# Pulls:
#   data/raw/peshitta_repo/      — Digital Syriac Corpus (~13 MB)
#   data/raw/coptic_repo/        — CopticScriptorium/corpora, sparse-checkout
#                                   limited to {thomas-gospel, sahidica.nt,
#                                   sahidica.mark, sahidica.1corinthians,
#                                   bohairic.nt}
#
# Run from repo root:
#   bash scripts/fetch_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
mkdir -p "$RAW_DIR"

# ---------- Digital Syriac Corpus (Peshitta + Syriac patristics) ----------
SYRIAC_DIR="$RAW_DIR/peshitta_repo"
if [ ! -d "$SYRIAC_DIR/.git" ]; then
  echo "[1/2] Cloning Digital Syriac Corpus -> $SYRIAC_DIR"
  git clone --depth 1 https://github.com/srophe/syriac-corpus.git "$SYRIAC_DIR"
else
  echo "[1/2] Syriac corpus already present at $SYRIAC_DIR — skipping"
fi

# ---------- Coptic SCRIPTORIUM (sparse: NT + Thomas only) ----------
COPTIC_DIR="$RAW_DIR/coptic_repo"
COPTIC_PATHS=(
  thomas-gospel
  sahidica.nt
  sahidica.mark
  sahidica.1corinthians
  bohairic.nt
)
if [ ! -d "$COPTIC_DIR/.git" ]; then
  echo "[2/2] Sparse-cloning CopticScriptorium/corpora -> $COPTIC_DIR"
  git clone --filter=blob:none --no-checkout --depth=1 \
    https://github.com/CopticScriptorium/corpora.git "$COPTIC_DIR"
  cd "$COPTIC_DIR"
  git sparse-checkout init --cone
  git sparse-checkout set "${COPTIC_PATHS[@]}"
  git checkout
  cd - >/dev/null
else
  echo "[2/2] Coptic corpus already present at $COPTIC_DIR — skipping clone"
fi

# Extract Sahidica NT TreeTagger archive — this is the only release format
# that contains all 27 NT books fully annotated (the CoNLL-U release covers
# only Mark + 1 Cor).
TT_ZIP="$COPTIC_DIR/sahidica.nt/sahidica.nt_TT.zip"
TT_DIR="$COPTIC_DIR/sahidica.nt/sahidica.nt_TT"
if [ -f "$TT_ZIP" ] && [ ! -d "$TT_DIR" ]; then
  echo "[2/2] Extracting $TT_ZIP -> $TT_DIR"
  mkdir -p "$TT_DIR"
  unzip -q -o "$TT_ZIP" -d "$TT_DIR"
elif [ -d "$TT_DIR" ]; then
  echo "[2/2] Sahidica TT already extracted — skipping"
fi

# ---------- SEDRA-3 Syriac morphology data (via fhardison/peshitta-tools) ----------
SEDRA_FILE="$REPO_ROOT/data/external/sedra/peshitta_list.txt"
if [ ! -f "$SEDRA_FILE" ]; then
  mkdir -p "$(dirname "$SEDRA_FILE")"
  echo "[3/3] Fetching SEDRA-3 Peshitta NT word data -> $SEDRA_FILE"
  curl -L -s -o "$SEDRA_FILE" \
    https://raw.githubusercontent.com/fhardison/peshitta-tools/master/peshitta_list.txt
  echo "  $(wc -l < "$SEDRA_FILE") word records loaded."
  echo "  License: SEDRA-3 academic use only — see data/external/sedra/SOURCES.md"
else
  echo "[3/3] SEDRA data already present at $SEDRA_FILE — skipping"
fi

echo
echo "Done."
echo "  Syriac TEI:           $SYRIAC_DIR/data/tei/"
echo "  Coptic Thomas:        $COPTIC_DIR/thomas-gospel/thomas.gospel_CONLLU/thomas_gospel.conllu"
echo "  Coptic NT (TT, full): $TT_DIR/  (all 27 books, ~7900 verses)"
echo "  Coptic NT (CoNLL-U):  $COPTIC_DIR/sahidica.nt/sahidica.nt_CONLLU/  (Mark + 1 Cor only)"
echo "  SEDRA Peshitta NT:    $SEDRA_FILE  (109654 words, lemma + gloss + parse)"
echo
echo "Next:"
echo "  python scripts/parse_peshitta_tei.py            # full Peshitta NT verses"
echo "  python scripts/annotate_peshitta_lemmas.py      # attach SEDRA lemmas"
echo "  python scripts/parse_thomas_tei.py              # Coptic Thomas (115 logia)"
echo "  python scripts/parse_coptic_tt.py               # full Sahidic NT (use this for parallel corpus)"
