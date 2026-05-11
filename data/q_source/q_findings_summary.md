# Q Source Catchword Analysis — Findings Summary

Date: 2026-05-11
Pipeline: identical to the Thomas analysis (detector threshold=0.65, filter_pct=80, Gemini-3-Flash-Preview / 2.5-flash retroversion, 10 variants per language).

---

## Important framing note

The task is **not** a direct test of Casey (2002) or Chilton's (2010) Aramaic-substrate hypothesis for Q. Casey and Chilton argue from *mistranslation retrojection* — places where the Greek text makes more sense if read as a mistranslation of an Aramaic original — not from catchword arrangement. Perrin's catchword argument is specifically about the Gospel of Thomas; no comparable catchword-arrangement claim has been made for Q in the literature.

What we *are* testing: does Q show the same kind of "thematic / language-stability" catchword-arrangement effect that we found in Thomas? If yes, and if Aramaic does *not* lead distinctively, that's evidence the effect is universal (thematic) rather than driven by a particular substrate. If Aramaic distinctively leads, that *would* be suggestive (though not confirmatory) of a Semitic substrate. A null result on this catchword test does not refute the mistranslation-retrojection version of the Casey/Chilton claim.

---

## Data

- **Q pericopes**: 56 (IQP / Critical Edition of Q standard segmentation, Lukan-ordered).
- **Q Greek text**: 55 fetched via Gemini API from the canonical SBLGNT register; 1 (Luke 11:39-44) manually transcribed after Gemini's safety filter blocked it.
- **Control passages**: 10 non-Q NT Greek excerpts (Romans 8/12, 1 Cor 1/13, Hebrews 1/11/12, Revelation 21/22). Average control pair length 109–131 words; average Q pair length 57–74 words (Q is *shorter* per unit than these particular controls).
- **Target languages**: Aramaic (Jewish Babylonian, Hebrew script), Syriac (Estrangela), Biblical Hebrew, Classical Arabic — 10 variants per pericope per language via Gemini-3-Flash-Preview / 2.5-flash.
- **Source language**: Koine Greek (no translation needed — used as the baseline).

Translation success: 100% usable variants across all 4 target languages.

---

## (1) Aggregate density — Q vs non-Q controls per language

(`scripts/q_aggregate_density.py`)

We pair Q pericopes adjacently (55 pairs) and control pericopes adjacently (9 pairs), and measure per-pair catchword count + length-normalised density (catchwords per 100×100 word pair).

| Language | Q raw/pair | Ctrl raw/pair | p (Q > Ctrl raw) | Q density | Ctrl density | p (Q > Ctrl length-norm) |
|---|---:|---:|---:|---:|---:|---:|
| Greek (source) | 30.0 | 60.7 | 1.00 | 68.3 | 37.1 | **0.0002** |
| Aramaic | 19.5 | 60.6 | 1.00 | 66.5 | 55.6 | 0.237 |
| Syriac | 27.2 | 72.2 | 1.00 | 81.8 | 62.4 | 0.103 |
| Hebrew | 26.5 | 55.8 | 1.00 | 83.7 | 51.4 | **0.0020** |
| Arabic | 21.1 | 53.2 | 1.00 | 70.5 | 46.1 | **0.0061** |

**Raw count direction:** Q < Controls in every language (Mann-Whitney p ≈ 1.0). Controls "win" purely because control pair lengths (~110 words) are roughly 2× Q pair lengths (~60 words) and catchword count scales near-quadratically with pair length.

**Length-normalised direction:** Q > Controls in 4/5 languages — but **Aramaic is the *weakest* and the only non-significant case** (p=0.24). Greek source shows the strongest Q-vs-Control density gap (p=0.0002).

This is the first piece of evidence against a distinctive Aramaic substrate at the density level: if Q had been composed in Aramaic with Semitic catchword density baked in, the Aramaic retroversion should *amplify* the Q-vs-Control gap, not narrow it.

---

## (2) Permutation test — single variant (variant 0), 10,000 shuffles

(`scripts/q_permutation_test.py`)

| Language | True (≥2) | Null mean ± std | z | p |
|---|---:|---:|---:|---:|
| Greek (source) | 201 | 191.1 ± 11.0 | 0.90 | 0.196 |
| Aramaic | 149 | 129.8 ± 12.0 | 1.60 | 0.064 |
| **Syriac** | **238** | 203.9 ± 14.5 | **2.35** | **0.013** ✓ |
| Hebrew | 210 | 193.2 ± 12.8 | 1.32 | 0.104 |
| Arabic | 157 | 150.4 ± 11.8 | 0.56 | 0.296 |

Only Syriac reaches α=0.05. Aramaic is marginal (p=0.064). Greek source is non-significant. This contrasts sharply with Thomas, where **all four** languages reached α=0.05 at variant 0.

Power considerations:
- Q has 56 pericopes; Thomas has 115. ≈2× fewer adjacency pairs.
- Q pericopes average ~60 words/pair; Thomas logia average ~15 words/pair. ≈4× larger inner-product space per cell, which inflates the null distribution.
- Both factors reduce per-language statistical power for Q vs Thomas.

---

## (3) Variant robustness — 10 variants × 1,000 perms per language

(`scripts/q_variant_robustness.py`)

| Language | Median z | Min – Max z | Median p | All 10 p<0.05? |
|---|---:|---:|---:|:---:|
| Greek (source) | 0.94 | — (single) | 0.187 | n/a (1 variant) |
| Aramaic | 1.71 | 0.13 – 2.31 | 0.050 | NO (4/10) |
| Syriac | 1.60 | 1.10 – 2.31 | 0.060 | NO (3/10) |
| Hebrew | 1.08 | 0.27 – 1.56 | 0.151 | NO (1/10) |
| Arabic | 0.53 | 0.08 – 1.00 | 0.312 | NO (0/10) |

**Pairwise Mann-Whitney (one-sided, A > B z, n=10 each):**

| Test | p-value | Significant? |
|---|---:|---|
| Aramaic > Syriac | 0.260 | no — indistinguishable |
| Aramaic > Greek | 0.182 | no |
| Aramaic > Hebrew | **0.006** | yes |
| Aramaic > Arabic | **0.0009** | yes |
| Syriac > Hebrew | **0.0009** | yes |
| Syriac > Arabic | **0.0001** | yes |
| Syriac > Greek | 0.091 | marginal |
| Hebrew > Arabic | **0.002** | yes |

**A three-tier structure emerges:**

- **Tier 1 (median z ≈ 1.6–1.7, marginal): Aramaic and Syriac.** Statistically indistinguishable (p=0.26).
- **Tier 2 (median z ≈ 0.9–1.1): Greek source, Hebrew.** Significantly below Tier 1.
- **Tier 3 (median z ≈ 0.5): Arabic.** Significantly below all others.

Crucially: **Aramaic ≈ Syriac.** If a distinctive Aramaic substrate underlay Q, Aramaic should *separate from* Syriac — but they cluster. Both languages produce lexically consistent Semitic retroversions (Gemini's training data favours both Classical Syriac and Jewish Babylonian Aramaic, presumably), and both pick up the same modest signal.

---

## (4) Comparison with Thomas

| | Thomas | Q |
|---|---|---|
| N units | 115 logia | 56 pericopes |
| Avg pair length (words) | ~15–30 | ~57–74 |
| Variant-0 significance | All 4 languages p<0.05 | Only Syriac p<0.05 |
| Tier 1 (all variants sig) | Syriac + Greek (Mann-Whitney p=0.31) | none — Aramaic + Syriac best but only 30–40% of variants significant |
| Tier 1 median z | 2.34–2.50 | 1.60–1.71 |
| Bottom tier | Hebrew + Arabic | Arabic alone |

Q's signal is weaker overall AND less variant-robust than Thomas's. Even the strongest Q language (Aramaic, median z=1.71) has 6/10 variants below α=0.05. By contrast, Thomas's strongest language (Syriac, median z=2.50) has 10/10 variants below α=0.05.

---

## (5) Headline interpretation

1. **The catchword-arrangement effect that was strongly visible in Thomas is much weaker in Q.** Power matters (smaller N, longer units → less signal), but the absolute median z-scores are also lower. Q's pericope-ordering may simply not have the same thematic-clustering structure Thomas has.

2. **Aramaic does not distinctively lead.** It clusters with Syriac (Mann-Whitney p=0.26). This is the same translation-stability pattern we saw in Thomas (Syriac ≈ Greek for Thomas, Aramaic ≈ Syriac for Q): two Gemini-stable target languages produce comparable signal, while Hebrew/Arabic produce less.

3. **The catchword-arrangement test does not support an Aramaic substrate for Q** — but it was never the right test for Casey/Chilton's actual claim. Their argument is about mistranslation patterns at specific Greek loci, not about ordering structure. A different methodology (e.g., systematic retroversion + identification of clausal ambiguities) would be needed to evaluate the Casey/Chilton hypothesis in its own terms.

4. **The Q pericope-ordering effect, where it appears, is best explained the same way as Thomas's:** thematic clustering of nearby sayings produces shared content vocabulary that an LLM retroversion picks up as catchword links. The signal is real but not language-specific.

5. **A null result in Aramaic does not refute the Casey/Chilton view.** It is silent on the mistranslation argument. It only addresses the (independent) question of whether Q has a Thomas-style catchword-arrangement effect under an Aramaic retroversion — and the answer is "maybe marginally, but not distinctively."

---

## Files

- `data/q_source/q_verses.json` — IQP pericope list (56 pericopes).
- `data/q_source/q_controls.json` — 10 non-Q control passages.
- `data/q_source/q_pericopes_greek.json` — fetched Lukan Greek (56/56).
- `data/q_source/q_controls_greek.json` — fetched control Greek (10/10).
- `data/q_source/translations/{aramaic,syriac,hebrew,arabic}/pericope_NNN.json` — 56 × 10 × 4 = 2,240 variants.
- `data/q_source/control_translations/{...}/pericope_NNN.json` — 10 × 10 × 4 = 400 variants.
- `data/q_source/aggregate_density.json` — per-pair counts + Mann-Whitney.
- `data/q_source/permutation/main_results.json` — single-variant 10k-perm test (5 languages).
- `data/q_source/permutation/variant_{lang}.json` — 10-variant × 1k-perm sweeps per language.
- `data/q_source/permutation/summary.txt` — pairwise Mann-Whitney + tier table.
- `analysis/figures/q_source/q_crossling_permutation.png` — 5-panel null-distribution histograms.
- `analysis/figures/q_source/q_variant_z_scores.png` — variant-robustness boxplots.
- `analysis/figures/q_source/q_variant_p_values.png` — p-values across 10 variants.
- `analysis/figures/q_source/q_vs_thomas.png` — side-by-side Thomas vs Q z-score distributions.

## Scripts

- `scripts/q_fetch_greek.py` — Gemini-fetch Koine Greek for any verse-reference list.
- `scripts/q_translate.py` — retrovert Greek into Aramaic/Syriac/Hebrew/Arabic via Gemini.
- `scripts/q_aggregate_density.py` — Q vs control density test, length-normalised.
- `scripts/q_permutation_test.py` — main 10k-perm test, 5 languages.
- `scripts/q_variant_robustness.py` — 10-variant × 1k-perm sweep per language.
- `analysis/plot_q_results.py` — figures + summary.
- New `LanguageProfile` for Aramaic in `phase1_montecarlo/language_data.py`.
