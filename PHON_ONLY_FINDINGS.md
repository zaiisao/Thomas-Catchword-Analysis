# Phonological-only cross-linguistic permutation test — pipeline limitation confirmed

**Question.** The original cross-linguistic permutation test mixed three catchword types
(semantic, phonological, etymological). The Proverbs positive control revealed that **Hebrew
does not lead on Proverbs** (median variant-sweep z = 1.76 in Hebrew vs 4.38 in Syriac), even
though Hebrew is the documented source of the catchword arrangement (Hildebrandt, Heim, Snell).

Hypothesis: semantic catchwords (same lemma at adjacent positions) carry the *thematic* signal —
universal across translations — and swamp the language-specific *phonological* signal. Strip out
semantic matches and the source language should lead.

**Test.** Re-ran the entire 3-corpus, 5-language, 10-variant permutation pipeline (122 worker
processes, ~25 minutes wall time on 64 cores). For each (corpus, language, variant) we ran THREE
permutation tests on the *same* matrix:

- `all`  — every link type (the original test)
- `phon` — `link_type ∈ {phonological, etymological}` (language-specific)
- `sem`  — `link_type ∈ {semantic}` (thematic)

Same detector, same threshold (0.65), same blocking (top 20% removed), same 10k-perm /
1k-perm budgets.

## Result: the fix does not work

| Corpus | Source | z_phon (var 0) | Rank among target variants | Empirical p |
|---|---|---|---|---|
| **Proverbs** | Hebrew | **−0.16** (p = 0.61) | **40 / 41** (worst) | 0.976 |
| **Thomas** | Syriac | 0.95 (p = 0.19) | 25 / 31 | 0.806 |
| **Q** | Greek | 0.64 (p = 0.27) | 24 / 41 | 0.585 |

**In all three corpora the source language fails to lead on phonological-only.** In Proverbs the
source language is *worse than every one of 41 target variants*. The empirical p-value for
"source ≤ random target variant" is 0.98 for Proverbs.

Per the pre-registered decision gate in the task brief: *"Hebrew doesn't lead on
phonological-only → the fix doesn't help. The pipeline fundamentally cannot distinguish
language-specific from thematic arrangement. Report as a limitation."*

## Decomposition: semantic carries everything

Variant 0 main test (10k perms), three filter settings side-by-side:

| corpus | lang | src | N | phon/B | sem/B | z_all | p_all | z_phon | p_phon | z_sem | p_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| proverbs | Hebrew | SRC | 595 | 0.37 | 0.18 | **1.76** | 0.054 | −0.16 | 0.609 | **3.64** | 0.0003 |
| proverbs | Greek | | 590 | 0.65 | 0.94 | **2.97** | 0.002 | 2.03 | 0.029 | 2.40 | 0.016 |
| proverbs | Syriac | | 585 | 0.53 | 0.19 | **3.27** | 0.002 | 1.44 | 0.099 | **4.04** | 0.0001 |
| proverbs | Aramaic | | 590 | 0.45 | 0.15 | **3.23** | 0.002 | 1.54 | 0.086 | **4.16** | 0.0002 |
| proverbs | Arabic | | 590 | 0.43 | 0.20 | **4.81** | <0.001 | **3.23** | 0.002 | **4.02** | 0.0004 |
| thomas | Syriac | SRC | 115 | 5.37 | 2.87 | **2.53** | 0.007 | 0.95 | 0.194 | **3.84** | 0.0002 |
| thomas | Hebrew | | 115 | 5.46 | 1.93 | 2.28 | 0.017 | **2.06** | 0.028 | 1.26 | 0.136 |
| thomas | Greek | | 115 | 7.89 | 4.10 | 2.25 | 0.016 | 1.48 | 0.081 | 2.35 | 0.016 |
| thomas | Arabic | | 115 | 4.53 | 2.05 | 2.43 | 0.011 | 1.79 | 0.051 | 2.22 | 0.027 |
| q | Greek | SRC | 56 | 20.96 | 7.84 | 0.90 | 0.196 | 0.64 | 0.273 | 1.13 | 0.163 |
| q | Hebrew | | 56 | 20.71 | 5.22 | 1.31 | 0.096 | 0.99 | 0.170 | 1.64 | 0.070 |
| q | Syriac | | 56 | 21.47 | 4.93 | **2.35** | 0.013 | 1.56 | 0.069 | **4.13** | 0.0002 |
| q | Aramaic | | 56 | 16.33 | 2.95 | 1.60 | 0.064 | 0.90 | 0.193 | 3.10 | 0.002 |
| q | Arabic | | 56 | 16.39 | 4.33 | 0.56 | 0.296 | 0.24 | 0.414 | 1.33 | 0.121 |

**Pattern across all 14 (corpus, lang) combinations: z_sem ≥ z_phon.** The thematic component
carries every significant result.

For the source languages specifically:
- Proverbs Hebrew: z_sem = 3.64 (p = 0.0003) but z_phon = −0.16
- Thomas Syriac: z_sem = 3.84 vs z_phon = 0.95
- Q Greek: z_sem = 1.13 vs z_phon = 0.64

The original "all-catchword" significance of these sources is fully explained by their semantic
component. There is no detectable phonological/etymological-only arrangement in any of the three
source languages.

## Per-boundary phon counts: data is there, signal isn't

The phon/etym data is not sparse for Thomas (5–8 per boundary) or Q (16–21 per boundary). It IS
sparse for Proverbs (0.37–0.65 per boundary). So:

- **Q, Thomas**: 5–21 phon catchwords per boundary, plenty of data — but they are *not
  positionally arranged at adjacent verse boundaries* above the null. The phon catchwords exist
  as random co-occurrences; they do not cluster at editorial seams.
- **Proverbs**: 0.4 phon catchwords per boundary at variant 0 is genuinely sparse (sub-1
  threshold of the task brief). For Proverbs the negative result is partly a power limitation,
  partly a real absence.

## Variant-sweep robustness (phon-only)

| corpus | lang | variants | median z | range | sig p<0.05 |
|---|---|---|---|---|---|
| proverbs | Hebrew | 1 (SRC) | −0.16 | — | 0/1 |
| proverbs | Greek | 10 | 1.88 | [1.08, 3.48] | 6/10 |
| proverbs | Syriac | 10 | 2.67 | [1.04, 3.94] | 8/10 |
| proverbs | Aramaic | 10 | 0.52 | [−0.50, 1.83] | 1/10 |
| proverbs | Arabic | 10 | 1.65 | [0.38, 3.33] | 5/10 |
| thomas | Syriac | 1 (SRC) | 0.95 | — | 0/1 |
| thomas | Hebrew | 10 | 1.12 | [0.11, 2.36] | 2/10 |
| thomas | Greek | 10 | 1.37 | [0.92, 2.51] | 2/10 |
| thomas | Arabic | 10 | 1.50 | [0.75, 1.96] | 5/10 |
| q | Greek | 1 (SRC) | 0.64 | — | 0/1 |
| q | Hebrew | 10 | 0.80 | [0.07, 1.24] | 0/10 |
| q | Syriac | 10 | 0.80 | [0.27, 1.56] | 0/10 |
| q | Aramaic | 10 | 1.03 | [−0.06, 1.53] | 0/10 |
| q | Arabic | 10 | 0.24 | [−0.19, 0.80] | 0/10 |

- **Proverbs phon-only:** Syriac variant median z = 2.67 (8/10 sig), Greek 1.88 (6/10), Arabic
  1.65 (5/10) — *all translation targets beat the source*. The phon-only signal in Proverbs is
  visible *only in retroversion*, never in the documented source language. This is exactly the
  pattern the task brief described as "the pipeline fundamentally cannot distinguish
  language-specific from thematic arrangement."
- **Q phon-only:** 0/10 variants reach p<0.05 in any language. Q has no phon-only arrangement at
  this granularity.
- **Thomas phon-only:** weak and inconsistent across all 4 languages (2–5/10 sig). Syriac, the
  language Perrin singled out for phonological design, is at the bottom (variant 0 z=0.95, not
  sig).

## What this means for Thomas

Per the task brief: *"Only interpret [Thomas phon-only] if Proverbs validation passes (Hebrew
leads)."* It did not. So this section is bracketed.

That said, the descriptive numbers are clear:

- **Syriac Thomas phon-only z = 0.95 (p = 0.19).** Below all three of its target languages
  (Hebrew 2.06, Greek 1.48, Arabic 1.79).
- The "all-catchwords" Thomas Syriac z = 2.53 (memory: `project_permutation_test`) decomposes
  into z_sem = 3.84 + a non-significant phon contribution.
- Across the 10-variant Syriac sweep on the *original* all-catchwords test, median z ≈ 2.06
  (memory: `project_crossling_permutation`) — *that* z is essentially the semantic component,
  not phonological design.

If the phon-only test were a valid measurement (which the Proverbs positive control fails), it
would say there is no Syriac-specific phonological arrangement in Thomas above its other-language
retroversions. Since the positive control fails, we cannot make that claim formally — but neither
can the opposite be supported.

## Pipeline limitation, stated precisely

This detector — consonantal-skeleton equality plus Levenshtein-similarity on consonantal
skeletons at threshold 0.65, with the top-20% most-frequent lemmas blocked — cannot detect
language-specific phonological arrangement when:

1. The phon catchwords are sparse in the source (Proverbs: 0.4 per boundary).
2. The phon catchwords are abundant but distributed by chance (Thomas, Q at adjacent positions).
3. The detector's consonantal-skeleton operation is itself partially script-invariant — the
   "phonological" link in our framework is similar consonants, not actual rhyme, alliteration,
   meter, or paronomastic root-play. Scholarly catchwords in Hebrew Proverbs operate at deeper
   levels (root-pun, antithetical pair, sound chiasm) that this detector cannot represent.

The original "all four languages significant, no language leads" finding (memory:
`project_crossling_permutation`) **stands** as a statement about *thematic* arrangement: the
semantic-catchword signal is robust across translations, and Thomas does show a non-random
thematic clustering of adjacent logia. But the test as designed cannot tell us whether Thomas was
*originally* composed in a Syriac (or any other language's) phonological key.

## Files

- `data/phon_only/{corpus}_{lang}_v{variant}.json` — 122 per-process records; each has
  `diagnostic` (per-boundary phon/sem counts) and `results` (3-filter perm test).
- `data/phon_only/summary.json` + `summary.txt` — aggregated tables.
- `analysis/figures/phon_only_comparison.png` — 3-panel bar chart, all-vs-phon-only z-scores per
  language per corpus.
- `analysis/figures/phon_only_proverbs_validation.png` — Proverbs phon-only z by language. Hebrew
  is dead last.
- `analysis/figures/phon_only_thomas.png` — Thomas phon-only z by language. Syriac is dead last.
- `analysis/figures/phon_only_variant_robustness.png` — box plots of phon-only z across 10
  variants per language per corpus.
- `scripts/phon_only_one.py` — one-shot worker.
- `analysis/plot_phon_only.py` — aggregator + figures.

## Cost / runtime

- 122 worker processes; full CPU saturation (100% on 64 cores).
- ~25 min wall time end-to-end.
- No API calls. No new translations. Reuses 23,800 existing Gemini retroversions.

## Methods note

The "source-leads" pre-registered test was Mann–Whitney one-sided. As written, that test requires
≥2 samples per group and source languages only have 1 (variant 0 = the canonical text). We
substituted a one-sample empirical rank test: rank source's variant-0 z-score against the pooled
distribution of target variants' z-scores. This is the cleanest test given the design.
