# Data Acquisition Status

Last updated: 2026-05-09

## Summary

| Resource | Source | Coverage | Annotation | Status |
|---|---|---|---|---|
| Peshitta NT | Digital Syriac Corpus (srophe/syriac-corpus) | All 27 books, 7958 verses | Vocalized + consonantal text | ✅ Loaded |
| Peshitta NT lemmas + parse | SEDRA-3 via fhardison/peshitta-tools | All 27 books, 109654 word records | Lemma + gloss + morphological parse code | ✅ Loaded — 100% verse-level join |
| Syriac patristics (Aphrahat, Ephrem, Odes, etc.) | Digital Syriac Corpus | 600+ texts | Vocalized text only | ✅ Loaded (not yet selected) |
| Coptic Gospel of Thomas | Coptic SCRIPTORIUM (`thomas-gospel/`) | Full text | **Lemmatized + POS + morph features (CoNLL-U)** | ✅ Loaded (389 sents, 6521 tokens) |
| Coptic NT (Sahidic) — full 27 books | Coptic SCRIPTORIUM (`sahidica.nt_TT.zip`, TreeTagger format) | 27/27 books, ~7906 verses | Lemmatized + POS + funcs (auto-tagged) | ✅ Loaded |
| Coptic NT (Sahidic) — Mark, 1 Cor (gold annotations) | Coptic SCRIPTORIUM (CoNLL-U) | Mark + 1 Cor | Manually-checked CoNLL-U | ✅ Loaded (overlap with TT) |
| Coptic NT (Bohairic) | Coptic SCRIPTORIUM (`bohairic.nt/`) | Likely full | TEI/CoNLL-U (not yet inspected) | ✅ Cloned, not parsed |
| Greek NT | TBD (SBLGNT, NA28) | n/a | n/a | ⏳ Not started |
| SEDRA Syriac morphological DB | sedra.bethmardutho.org | All Syriac NT (lemmas, roots) | Required for Syriac lemma/root extraction | ⏳ **Not started — critical** |
| Perrin's full 502-pair table | *Thomas and Tatian* (2002), pp. 57–155 | Complete catchword inventory | Manual entry from print | ❌ Not yet entered |
| Perrin's JETS 2006 examples | `data/raw/perrin_catchwords/` | 11 illustrative pairs + headline stats | Loaded from PDF | ✅ Loaded |

## Critical gaps

1. ~~**Sahidic Coptic Matthew, Luke, John are unavailable in CopticScriptorium.**~~ **RESOLVED.** The CoNLL-U release ships only Mark + 1 Cor, but the TreeTagger (`sahidica.nt_TT.zip`) release contains all 27 NT books fully annotated. `scripts/parse_coptic_tt.py` extracts them. **Caveat:** John 8 (the *Pericope Adulterae*) is omitted from this Sahidic edition — consistent with the manuscript tradition. The TreeTagger annotations are described as `tagging="automatic"` (vs the gold-standard CoNLL-U), so for evaluation we may want to spot-check accuracy on Mark by comparing TT vs CoNLL-U output.

2. ~~**Syriac is unlemmatized in the Digital Syriac Corpus.**~~ **RESOLVED.** SEDRA-3 word data is available via [fhardison/peshitta-tools](https://github.com/fhardison/peshitta-tools) (`peshitta_list.txt`). 109,654 word records covering the entire Peshitta NT — `unpointed | pointed | lemma | gloss | parse_code`. `scripts/annotate_peshitta_lemmas.py` joins to our parsed Peshitta with 100% verse-level coverage. **License**: SEDRA academic-use-only, no redistribution of altered files (see `data/external/sedra/SOURCES.md`).

   **Outstanding:** SEDRA's separate ROOT table (triliteral roots, distinct from lemmas) is not in this export. For Perrin's *etymological* catchwords (different lemmas sharing a root), we would need to add the ROOT layer from `peshitta/sedrajs`. Lemma-level + consonantal-skeleton coverage is sufficient for the bulk of Perrin's examples (which are *phonological* / *homophone* rather than etymological).

3. **Perrin's full 502-pair ground-truth table requires the 2002 book.** The JETS 2006 paper gives only ~10 illustrative pairs. For pair-by-pair validation of our detector, we need the full table from *Thomas and Tatian* pp. 57–155 (manual entry).

## Output artifacts

After running `bash scripts/fetch_data.sh && python scripts/parse_*.py ...`:

```
data/processed/
├── parallel_corpus/
│   ├── peshitta_nt.jsonl              # 7958 verses, full Peshitta NT (raw + consonantal)
│   ├── peshitta_nt_lemmatized.jsonl   # 7958 verses + SEDRA-3 lemmas/glosses/parse codes
│   ├── sahidica_nt_coptic_tt.jsonl    # 7906 verses, full Sahidic NT (27 books, lemmatized)
│   └── sahidica_nt_coptic.jsonl       # 1111 sents, gold CoNLL-U (Mark + 1 Cor only)
├── lexical_mapping/
│   └── coptic_to_syriac.jsonl         # 3831 Coptic content lemmas → P(Syriac lemma)
└── got_logia/
    ├── thomas_logia.jsonl             # 388 logion-paragraph records, all 115 logia
    └── thomas_coptic.jsonl            # 389 sents (legacy CoNLL-U-based extraction)
```

### Coptic → Syriac lexical map (Phase 1 input)

Built by IBM Model 1 EM (`scripts/build_lexical_map.py`) on 7,834 aligned NT verse pairs (Coptic Sahidica TT × Peshitta SEDRA), 12 EM iterations. Output: `data/processed/lexical_mapping/coptic_to_syriac.jsonl` — 3,831 content-word entries with P(Syriac_lemma | Coptic_lemma) distributions.

Validation against Perrin's catchword examples (all top-rank):

| Coptic | English | Top Syriac (P) | Perrin's Syriac |
|---|---|---|---|
| ⲕⲱϩⲧ | fire (in 16.2) | ܢܘܪܐ (0.98) | *nūrā* ✓ |
| ⲥⲁⲧⲉ | fire (in 82.1) | ܢܘܪܐ (0.87) | *nūrā* ✓ |
| ⲟⲩⲟⲉⲓⲛ | light | ܢܘܗܪܐ (0.79) | *nuhrā* ✓ |
| ⲃⲁⲗ | eye | ܥܝܢܐ (0.99) | *ʿaynā* ✓ |
| ⲙⲁ | place | ܐܬܪܐ (0.23, top rank) | *ʾatar* ✓ |
| ⲥϩⲓⲙⲉ | woman | ܐܢܬܬܐ (0.99) | *nesse* (= plural of same lemma) ✓ |

### Four-Gospel parallel corpus availability (NT alignment input for Phase 2)

| Gospel | Peshitta verses | Sahidic verses | Notes |
|---|---|---|---|
| Matthew | 1071 | 1071 | full alignment expected |
| Mark    |  678 |  678 | full alignment expected |
| Luke    | 1151 | 1151 | full alignment expected |
| John    |  879 |  868 | Sahidic missing John 8 (pericope adulterae) |
| **Total** | **3779** | **3768** | ~99.7% verse coverage on both sides |

Note: Coptic Thomas is segmented by sentence, not by logion. Mapping sentences → logia (114 + Prologue) is a Phase 1 prerequisite.
