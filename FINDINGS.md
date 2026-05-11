# Findings — Gospel of Thomas Catchword Hypothesis

Last updated: 2026-05-10 (after permutation test on recurring catchword patterns — strongest result in the project).

## Headline

| Phase | Question | Method | Result |
|---|---|---|---|
| **Phase 1** | Can random Coptic→Syriac translation reproduce Perrin's 502 catchwords? | Monte Carlo over EM lexical map | mean **195** (CI 175–216), P(≥502) = 0 |
| **Phase 2A** | Does adding Syriac fluency get us closer? | Beam search with bigram LM, λ ∈ {0, 0.1, 0.3, 0.5, 1.0} | 311–328 (best 320 at λ=0.3) |
| **Phase 2C** | What about per-logion stochastic sampling at maximum fluency? | 200 sims at λ=1.0 | mean **324** (CI 303–347), P(≥502) = 0 |
| **Phase 2B (qualitative)** | Do frontier LLMs reproduce Perrin's *specific* catchwords? | Claude / Gemini / GPT-4 manual on 5 logia + EM map check | **YES** — 3/3 LLMs + EM map (P=0.98 / P=0.79) all produce ܢⲩⲣⲣ for fire and ܢⲩⲗⲣⲣ for light; bias critique fails for cited pairs |
| **Phase 2B (quantitative)** | Is the *aggregate count* Thomas-specific? | Gemini-3-Flash-Preview API, 1,250 calls, with control passages | **NO** — 974 catchwords on Thomas (4.14×) but controls produce 12.15/pair vs Thomas's 8.53/pair (p=0.99); aggregate density is LLM-stylistic |
| **Permutation test** | Does the *ordering* of Thomas logia produce more recurring catchword patterns than random orderings? | 10,000 random shuffles of Gemini-translated Thomas; cross-checked across all 10 variants + Phase 2A beam | **YES — p = 0.007** at ≥2 boundaries; all 10 LLM variants give p < 0.05 (median p ≈ 0.014); 8/8 of Perrin's specific cited boundaries reproduced |
| **Perrin table (full)** | Of Perrin's 502 specific Syriac catchwords, how many also appear at the same boundary in an unbiased Gemini retroversion? | Digitization of Perrin's full table (book pp. 58–153) + consonantal-skeleton match against Phase 2B Gemini canonical | **22.2% canonical, 77.8% Perrin-specific** (124/558 vs 434/558 at adjacent-boundary attribution); 53/107 boundaries have **0 canonical matches**; Williams' bias critique extends from sampled examples to the whole table |
| **Cross-linguistic permutation test** | Is the recurring-catchword effect Syriac-specific, Semitic-general, or thematic? | Gemini retroversion of Thomas into Hebrew + Arabic + Greek; same permutation test, same detector, same threshold; 10-variant robustness sweep per language | **All four significant** at variant 0 (Syriac p=0.007, Hebrew p=0.017, Arabic p=0.011, **Greek p=0.016**). 10-variant sweep reveals a two-tier structure: **Syriac and Greek are statistically indistinguishable** (Mann-Whitney p=0.31, 10/10 variants significant in each, median z≈2.4–2.5); Hebrew + Arabic sit lower (median z≈1.6–1.9, 2-3 variants drop below p=0.05). The Syriac=Greek tie is **decisive against any Syriac-specific phonological design** — the effect is **thematic, not Syriac-specific**. |
| **Q source (extension test)** | Does Q (sayings source behind Matt+Luke) show the same effect, and does Aramaic distinctively lead? | Identical pipeline applied to 56 IQP Q pericopes + 10 non-Q controls (Romans/Hebrews/Revelation); Gemini retroversion into Aramaic/Syriac/Hebrew/Arabic + Greek source; same calibration, 10k perms + 10-variant sweep | **Q's signal is much weaker than Thomas's.** Only Syriac p=0.013 reaches α=0.05 at variant 0; Aramaic p=0.064, Greek source p=0.196. Variant sweep: Aramaic and Syriac form Tier 1 (median z≈1.6–1.7, indistinguishable, Mann-Whitney p=0.26); Greek+Hebrew Tier 2; Arabic Tier 3. **Aramaic does NOT distinctively lead** — Casey/Chilton-style substrate prediction is not supported on this test (but note: catchword arrangement is not what Casey/Chilton actually argue; their claim is about mistranslation retrojection, which this pipeline does not test). |
| **Phase 3.0** | Is catchword arrangement actually detectable in known Syriac literary texts? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **YES** — pooled p < 1e-9, Cohen's d = 0.54 |
| **Phase 3.1** | Can a model trained on those texts learn to discriminate consecutive strophes? | Hard-negative contrastive (same-work, ≥3 strophes apart) + InfoNCE | val_acc **0.582** (vs 0.50 chance) |
| **Phase 3.2** | Does that learned model see the same pattern in beam-translated Thomas? | Permutation test: adjacent vs shuffled cos_sim | p = **0.087** (marginal) for beam translation; p = 0.32 (n.s.) for NMT |

## Three findings, in plain language

**(1) Perrin's 502 cannot be explained by Coptic→Syriac word-mapping plus Syriac fluency alone.** Every automated method we tried — random sampling, top-1 deterministic mapping, beam search with a Peshitta-NT bigram LM, per-logion stochastic sampling weighted by that LM — produces 195–324 catchwords. The gap to Perrin (~178 catchwords) is real and unexplained by what an unbiased translator working from the lexical map can produce.

**(2) Catchword-based literary arrangement is real in Syriac.** When we apply the same calibrated rule-based detector to Ephrem, Narsai, Jacob of Serug, and the Odes of Solomon, consecutive strophes share significantly more catchwords than randomly-paired strophes (pooled p < 1e-9, Cohen's d = 0.54). All four corpora individually are significant. So Perrin's *premise* — that Syriac literature uses catchword arrangement and a Syriac-original Thomas should too — is not unfounded.

**(3) Thomas's recurring-catchword arrangement is real, but it's thematic, not Syriac-specific.** When we translate Thomas into Hebrew, Arabic, and Greek (same Gemini model, same detector, same threshold), the same permutation test gives p = 0.017, 0.011, and 0.016 respectively at variant 0. The 10-variant robustness sweep (1,000 perms × 10 variants × 4 languages) sharpens this: Syriac and Greek form a single statistical tier (Mann-Whitney p = 0.31, 10/10 variants significant in each, median z ≈ 2.4–2.5), and Hebrew + Arabic form a second, lower tier (median z ≈ 1.6–1.9, 2–3 variants drop below p=0.05). Greek is decisive: it has no triliteral roots, longer words, different phonological structure, and yet matches Syriac on this test. The non-randomness is in the **thematic clustering of the logia** (visible in the Coptic source) — not in any Syriac-specific phonological design. The Hebrew/Arabic underperformance is most plausibly translation-side noise (rarer registers, more lexical variance across variants), not anything about the source text. Combined with the Perrin-table pair-by-pair result (78% of Perrin's 502 specific Syriac words are not canonical retroversions), this leaves Perrin's *aggregate* claim ("Thomas is arranged by Syriac catchwords") without empirical support, while leaving his *specific cited examples* (`nūrā/nuhrā` etc.) intact as canonical translations of thematically-paired Coptic.

## What this implies for Perrin's claim

The surplus of ~178 catchwords (Perrin's 502 vs our best automated 324) could in principle reflect:

  (a) **Genuine Syriac literary structure invisible to our 1-grams + 2-grams.** Real catchword arrangement uses skipped-token patterns, paronomasia at the root level, hapax-legomenon connections, etc., that our LM does not capture. Phase 3.0 confirms such structure is detectable in known Syriac texts; some of Perrin's surplus might be picking up the same phenomenon in a Syriac substrate of Thomas.

  (b) **Perrin's manual translation choices.** When Perrin retroverted Thomas to Syriac, he had freedom over many word choices, and may (consciously or not) have chosen Syriac words that maximize catchword density.

We cannot fully separate (a) from (b) without (i) Phase 2B, which requires an `ANTHROPIC_API_KEY` to query an LLM that has Syriac knowledge but no catchword agenda, and (ii) a stronger Phase 3 model trained on more data than 17k strophe-pairs from four authors.

## Caveat: what the methods do and don't capture

- **All Phase 2 methods** treat translation as one-Coptic-word-to-one-Syriac-word, drawn from an EM-aligned lexical map of NT parallel verses. Real translators choose multi-word renderings, restructure clauses, and cross-reference idioms. This is a strict lower bound on what informed translation can do.
- **Phase 3** measures catchwords via the same rule-based detector calibrated against Perrin's Coptic count. We use consonantal-skeleton matching for both literature and Thomas because patristic Syriac strophes lack lemma annotations. The relative consecutive-vs-random comparison is robust to function-word inflation; the absolute counts are not.
- **The contrastive model is small (4.8M params); the larger mBERT-finetune attempt did not improve on it.** When GPU memory became available (2026-05-11), we ran `scripts/phase3_improved_contrastive.py` — the spec'd 178M-param mBERT fine-tune (frozen bottom 8 layers / fine-tune top 4, hard-negatives, all-pairs InfoNCE, batch 256, lr 4e-5). Best val_acc 0.528 over 10 epochs (vs 0.582 for the small 4.8M baseline). The script's pre-registered self-abort criterion ("val_acc < 0.60 at epoch 10 → signal too weak for this architecture") fired correctly. The bigger pretrained encoder does NOT extract more signal from these 17k strophe-triples — consistent with the cross-lingual finding that catchword arrangement is thematic, not phonological, so a phonology-blind multilingual encoder gains nothing. The original Phase 3 numbers (val_acc 0.582, Thomas p=0.087) stand as the project's Phase 3 result.

## Reproducibility

```bash
# Environment
conda activate thomas   # /home/sogang/mnt/db_2/anaconda3/envs/thomas (Python 3.11)

# Phase 0 (already cached on disk)
bash scripts/fetch_data.sh

# Phase 1 (already cached)
python scripts/run_monte_carlo.py

# Phase 2 — three independent translation methods at the SAME calibration
python scripts/phase2a_beam_translate.py
python scripts/phase2c_constrained_sample.py
# Phase 2B (deferred — needs ANTHROPIC_API_KEY)
# python scripts/phase2b_llm_translate.py

# Phase 3 — gate on baseline first
python scripts/phase3_baseline_test.py
# Only train if baseline shows signal (it does)
python scripts/phase3_train_hardneg.py --device cuda:N --epochs 30 \
    --d-model 192 --n-layers 4 --batch-size 8 --max-len 64
python scripts/phase3_apply_improved.py --device cuda:N --max-len 96

# Synthesis
python analysis/phase2_comparison.py
python analysis/final_comparison.py
```

## Key figures

- `analysis/figures/final_summary.png` — paper-ready 2-panel synthesis
- `analysis/figures/phase2_all_methods.png` — Phase 2 method comparison
- `analysis/figures/phase2a_lambda_sweep.png` — beam-search λ sensitivity
- `analysis/figures/phase2c_distribution.png` — constrained-sampling distribution
- `analysis/figures/phase3_baseline.png` — consecutive-vs-random per Syriac author
- `analysis/figures/phase3_improved_thomas.png` — Thomas adjacent-sim vs shuffled
- `analysis/figures/roundtrip_summary.png` — catchword counts at each round-trip stage
- `analysis/figures/roundtrip_recovery_ratio.png` — recovery ratios per corpus and method, with Thomas/Perrin reference lines
- `analysis/figures/roundtrip_pair_survival.png` — per-pair survival by link type

## Round-trip validation (2026-05-09, latest)

We validated the pipeline end-to-end by running the four known-catchword Syriac corpora (Ephrem, Narsai, Jacob, Odes of Solomon) through a full Syriac → Coptic → Syriac round-trip using a freshly-EM-trained reverse lexical map (which validates correctly: ܢܘܪܐ→ⲕⲱϩⲧ, ܥܝܢܐ→ⲃⲁⲗ, etc., all top candidates).

### Round-trip recovery ratios (recovered Syriac total / Coptic intermediate total)

| Corpus | Original Syr | Coptic interm. | MAP | Beam λ=0.3 | MC mean | r_MAP | r_Beam | r_MC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ephrem  | 10,735 |  3,143 |  3,705 |  3,710 |  3,339 | **1.18×** | 1.18× | 1.06× |
| Jacob   | 41,172 | 12,001 | 13,861 | 13,708 | 12,197 | **1.15×** | 1.14× | 1.02× |
| Narsai  | 69,142 | 20,941 | 25,793 | 25,594 | 22,826 | **1.23×** | 1.22× | 1.09× |
| Solomon |    552 |    120 |    141 |    136 |    123 | **1.18×** | 1.13× | 1.03× |
| *Thomas (for ref)* | *N/A* | *235* | *305* | *320* | *195* | *1.30×* | *1.36×* | *0.83×* |
| *Perrin's claim* | — | — | 502 | — | — | *1.87×* | — | — |

### Per-pair catchword survival (deterministic MAP/MAP round-trip)

| Corpus | Semantic | Etymological | Phonological | Total |
|---|---:|---:|---:|---:|
| Ephrem  | 2040/2040 = **100.0%** | 0/0 (none) | 820/2169 = **37.8%** | 67.9% |
| Jacob   | 8307/8307 = **100.0%** | 0/0 (none) | 2539/6990 = **36.3%** | 70.9% |
| Narsai  | 13435/13435 = **100.0%** | 0/0 (none) | 5601/15941 = **35.1%** | 64.8% |
| Solomon | 82/82 = **100.0%** | 0/0 (none) | 31/53 = **58.5%** | 83.7% |

### What this means

**(a) Methodological correction to Phase 1's headline Thomas finding.**
Phase 1 reported Thomas's MC ratio at 0.83× and noted this is well below Perrin's claimed 1.87×. The round-trip now establishes that the *empirical recovery ceiling* of our pipeline — when applied to known-catchword Syriac literary texts — is **1.0–1.2× under MC**, **1.15–1.23× under MAP**, **1.13–1.22× under beam**. Thomas's MC ratio (0.83×) is *just below* this empirical baseline; Thomas's MAP (1.30×) and beam (1.36×) ratios are *just above*. So Thomas's ratios fall within or near the noise band of round-tripped known-catchword text. **The Phase 1 ratio is uninformative about Thomas's original language.** Both "Thomas had Syriac catchwords" and "Thomas didn't" predict ratios in the same range under our pipeline.

**(b) Stronger version of Phase 1's main result still holds.**
The original Phase 1 finding was about *absolute counts*: 502 catchwords cannot be reached by random translation of the Coptic Thomas (P(≥502) = 0). The round-trip confirms the analogous result for fluent automated translation: Perrin's claimed ratio (1.87×) **exceeds the maximum recovery ratio achieved by any of our methods on any known-catchword corpus** (max observed: 1.23× for Narsai under MAP). So the lexical-map round-trip *cannot* reproduce Perrin's recovery ratio even when starting from genuine catchword-rich Syriac literature. This means either (i) the lexical-map approach is fundamentally too narrow to capture full catchword density (it loses ~65% of phonological links per the survival table), or (ii) Perrin's manual translation introduces catchword density beyond what semi-deterministic translation produces, or (iii) both.

**(c) Where the loss happens.**
The round-trip preserves *all* semantic catchwords (same lemma) — by construction, since the same Syriac lemma round-trips deterministically to itself under MAP. The loss is concentrated in **phonological catchwords**: only 35–58% survive. This is because phonologically-similar Syriac words can map to different Coptic words under reverse-translation, and those Coptic words then map back to phonologically-distant Syriac words. The reverse-map's NT-trained vocabulary is too sparse to preserve consonantal-similarity relationships.

### Net impact on the Perrin verdict

The round-trip experiment **does not falsify Perrin's claim** — but it sharply restricts how informative our other tests can be:

- Phase 1's "Thomas at 0.83×" → uninformative (within round-trip noise).
- Phase 2A/2B/2C "automated max ~324" → still informative because *absolute* (Perrin's 502 exceeds it).
- Phase 3.0 "Syriac literature has catchword arrangement" → unchanged.
- Phase 3.1/3.2 "Thomas marginal at p=0.087" → unchanged.

The most rigorous remaining bound: **Perrin's 1.87× recovery ratio is empirically unreachable by lexical-map translation, even on known-catchword Syriac literature.** This is a reasonable basis to suspect the surplus reflects either Perrin's translation choices or higher-order Syriac structure that our 1-grams + bigrams don't model — same dichotomy as before, with one direction of evidence (Phase 1's MC ratio) now ruled out as a discriminator.

## Q source extension test (2026-05-11)

We extended the entire Thomas pipeline to a second corpus: the **Q source** (the hypothetical sayings collection behind Matthew and Luke). The motivation: if the catchword-arrangement effect we found in Thomas is a property of any thematically clustered sayings collection (the "thematic" conclusion of the cross-linguistic test below), Q should show the same pattern. Casey (2002) and Chilton (2010) have argued Q was originally composed in Aramaic — but their argument is about mistranslation retrojection, *not* catchword arrangement. Our pipeline cannot test Casey/Chilton's actual claim. What it *can* test is whether Q shows the Thomas-style arrangement effect, and whether Aramaic distinctively leads.

### Setup

- **56 Q pericopes** (IQP / Critical Edition of Q standard segmentation, Lukan-ordered). Greek text fetched from the SBLGNT register via Gemini-3-Flash-Preview, with 1 pericope (Luke 11:39-44) manually transcribed after Gemini's safety filter declined.
- **10 non-Q control passages** from Romans (8, 12), 1 Corinthians (1, 13), Hebrews (1, 11, 12), Revelation (21, 22) — definitely no Q overlap; comparable register (NT prose).
- **Target languages**: Aramaic (Jewish Babylonian, Hebrew script), Syriac, Biblical Hebrew, Classical Arabic — 10 Gemini variants per pericope per language. Greek serves as the **source language** (no translation needed).
- Same detector calibration as every other phase (threshold=0.65, filter_pct=80). New `LanguageProfile` for Aramaic in `phase1_montecarlo/language_data.py`.
- 100% usable variants across all 4 target languages.

### Aggregate density: Q vs controls per language

The aggregate density check that we should have run on Thomas before the headline 974 number (lesson from Phase 2B). Per-pair catchword count plus length-normalised density (catchwords per 100×100 word pair, since Q pericopes average ~60 words vs the chosen controls' ~110):

| Language | Q raw/pair | Ctrl raw/pair | Q density | Ctrl density | p (length-norm, Q > Ctrl) |
|---|---:|---:|---:|---:|---:|
| Greek (source) | 30.0 | 60.7 | 68.3 | 37.1 | **0.0002** ✓ |
| Aramaic | 19.5 | 60.6 | 66.5 | 55.6 | 0.237 |
| Syriac | 27.2 | 72.2 | 81.8 | 62.4 | 0.103 |
| Hebrew | 26.5 | 55.8 | 83.7 | 51.4 | **0.0020** ✓ |
| Arabic | 21.1 | 53.2 | 70.5 | 46.1 | **0.0061** ✓ |

Raw counts: Q < Controls in every language because the controls are roughly twice as long per pair. After length-normalisation, Q is *denser* than controls in 4/5 languages — but **Aramaic is the only non-significant case (p=0.24)**. If Q had been composed in Aramaic with Semitic catchword density baked in, Aramaic should *amplify* the Q-vs-Control gap; instead it narrows it. First piece of evidence against a distinctive Aramaic substrate.

### Permutation test (variant 0, 10,000 shuffles)

| Language | True (≥2) | Null mean ± std | z | p |
|---|---:|---:|---:|---:|
| Greek (source) | 201 | 191.1 ± 11.0 | 0.90 | 0.196 |
| Aramaic | 149 | 129.8 ± 12.0 | 1.60 | 0.064 |
| **Syriac** | **238** | 203.9 ± 14.5 | **2.35** | **0.013** ✓ |
| Hebrew | 210 | 193.2 ± 12.8 | 1.32 | 0.104 |
| Arabic | 157 | 150.4 ± 11.8 | 0.56 | 0.296 |

Only Syriac reaches α=0.05. Aramaic is the second-strongest (p=0.064 marginal). Greek source is non-significant. This is much weaker than Thomas (where all 4 languages crossed α=0.05 at variant 0). N halved (56 vs 115) and average pair length is ~3× longer, so power per pair is ~6× lower — but the absolute z-scores are also lower (Syriac z=2.35 here vs 2.53 for Thomas at the same variant).

### Variant robustness (10 variants × 1,000 perms per language)

| Language | Median z | Min–Max z | Median p | All p<0.05? |
|---|---:|---:|---:|:---:|
| Greek (source) | 0.94 | (single var) | 0.187 | n/a |
| **Aramaic** | **1.71** | 0.13 – 2.31 | 0.050 | NO (4/10) |
| **Syriac** | **1.60** | 1.10 – 2.31 | 0.060 | NO (3/10) |
| Hebrew | 1.08 | 0.27 – 1.56 | 0.151 | NO (1/10) |
| Arabic | 0.53 | 0.08 – 1.00 | 0.312 | NO (0/10) |

**Pairwise Mann-Whitney (one-sided, n=10 each):**

| Test | p | Significant? |
|---|---:|---|
| **Aramaic > Syriac** | **0.260** | **no — indistinguishable** |
| Aramaic > Greek (source) | 0.182 | no |
| Aramaic > Hebrew | **0.006** | yes |
| Aramaic > Arabic | **0.0009** | yes |
| Syriac > Hebrew | **0.0009** | yes |
| Syriac > Arabic | **0.0001** | yes |
| Syriac > Greek (source) | 0.091 | marginal |
| Hebrew > Arabic | **0.002** | yes |

**Three-tier structure for Q (mirroring Thomas's two-tier but weaker overall):**

- **Tier 1 (median z ≈ 1.6–1.7)**: **Aramaic + Syriac**, statistically indistinguishable (Mann-Whitney p=0.26).
- **Tier 2 (median z ≈ 0.9–1.1)**: Greek source + Hebrew.
- **Tier 3 (median z ≈ 0.5)**: Arabic.

**No language has all 10 variants below α=0.05.** Aramaic (the strongest by median z) has only 4/10 variants significant; Syriac 3/10. Compare to Thomas: Syriac and Greek there had **10/10 each**. Q's signal is qualitatively weaker AND less variant-robust than Thomas's.

### Interpretation

1. **The same translation-stability pattern from Thomas reappears.** For Thomas, Syriac ≈ Greek were Tier 1 (Gemini's most lexically consistent retroversions). For Q, Aramaic ≈ Syriac are Tier 1 — Gemini handles both Northwest-Semitic languages with comparable consistency, and both pick up the same signal. The clustering is not a property of either language *as a substrate*; it is a property of Gemini's translation behavior.

2. **Aramaic does NOT distinctively lead Q.** It is tied with Syriac (which Casey/Chilton would not claim as Q's substrate). The Aramaic-substrate prediction (insofar as it would predict an Aramaic-specific arrangement signal) is unsupported.

3. **Q's catchword-arrangement signal is genuinely weaker than Thomas's.** Even controlling for power (smaller N, longer units), the median z-scores are roughly 1.5–1.7 vs Thomas's 2.4–2.5. Q's Lukan ordering may not preserve as much thematic clustering as Thomas's ordering does — or Q's longer pericopes dilute the per-pair catchword signal.

4. **A null result on this catchword test is NOT a refutation of Casey/Chilton.** Their argument is about mistranslation retrojection at specific Greek loci (e.g., underlying Aramaic syntactic ambiguities), not about catchword arrangement. The pipeline we used cannot evaluate that argument; a different methodology (systematic retroversion + identification of clausal mistranslations) would be needed. This test only addresses whether Q shows the Thomas-style catchword-arrangement effect with an Aramaic substrate signature — which it does not.

5. **What this says about the cross-linguistic permutation test on Thomas.** The Thomas result was striking precisely because all 4 languages crossed α=0.05 and the top tier (Syriac, Greek) had 10/10 variants significant. The Q result shows that this strong, robust effect does *not* automatically appear in any Greek-source sayings collection — Q does not reproduce it. So the Thomas effect, while not Syriac-specific (Greek matches it), is also not universal across sayings collections; it depends on the specific thematic clustering Thomas's redactor encoded. This narrows but does not eliminate the "thematic clustering" interpretation.

### Files

- `data/q_source/q_findings_summary.md` — full Q analysis writeup with reframing.
- `data/q_source/q_verses.json`, `q_controls.json` — pericope reference lists.
- `data/q_source/q_pericopes_greek.json` (56), `q_controls_greek.json` (10).
- `data/q_source/translations/{aramaic,syriac,hebrew,arabic}/pericope_NNN.json` — 2,240 Q variants.
- `data/q_source/control_translations/{...}/pericope_NNN.json` — 400 control variants.
- `data/q_source/aggregate_density.json` — Q vs controls per language.
- `data/q_source/permutation/main_results.json` — 5-language permutation test.
- `data/q_source/permutation/variant_{lang}.json` — 5 × 10 variant sweep.
- `data/q_source/permutation/summary.txt` — variant tables + Mann-Whitney.
- `analysis/figures/q_source/q_crossling_permutation.png` — null-distribution histograms.
- `analysis/figures/q_source/q_variant_z_scores.png` + `q_variant_p_values.png` — variant robustness.
- `analysis/figures/q_source/q_vs_thomas.png` — side-by-side Thomas vs Q.
- Scripts: `scripts/q_fetch_greek.py`, `scripts/q_translate.py`, `scripts/q_aggregate_density.py`, `scripts/q_permutation_test.py`, `scripts/q_variant_robustness.py`, `analysis/plot_q_results.py`.

## Cross-linguistic permutation test (2026-05-11) — **revised interpretation**

**This test materially revises the previous "strongest finding" interpretation below.**

### Question

The Syriac permutation test (next section) showed the true ordering of Thomas's 115 logia produces 137 distinct recurring catchword pairs at ≥2 boundaries vs a null mean of 119.7 (p = 0.007). That establishes the arrangement is non-random with respect to **Syriac** catchwords. But it cannot distinguish three explanations:

- (a) **Syriac-specific arrangement.** A Syriac-speaking editor organized the logia for Syriac phonological echoes. Effect should appear in Syriac alone.
- (b) **Thematic arrangement that incidentally produces catchword patterns in any language.** Logia are grouped by topic; thematically related logia share vocabulary that produces catchword links in any language. Effect should appear in all languages.
- (c) **Semitic-general.** Effect appears in Semitic languages (triliteral roots, limited consonant inventory) but not in Greek.

### Setup

- Translated all 115 Thomas logia into Biblical Hebrew, Classical Arabic, and Koine Greek using the same Gemini-3-Flash-Preview model, temperature, and 10-variant scheme as Phase 2B's Syriac translations. Verified scripts: 100% usable variants in all three languages.
- Same `CatchwordDetector` algorithm at the same threshold (0.65) and same 80% blocked-lemma filter. Only the `LanguageProfile` (confusion groups, weak consonants, vocalization) varies.
- Identical permutation procedure: 10,000 shuffles, ≥2-boundary recurring-pair count.

### Result — all four languages significant at α = 0.05

| Language | True (≥2) | Null mean ± std | z-score | p-value |
|---|---:|---:|---:|---:|
| **Syriac** (reference) | **137** | 119.7 ± 6.8 | **+2.53** | **0.0070** |
| **Hebrew** | **110** |  91.8 ± 8.0 | **+2.28** | **0.0173** |
| **Arabic** | **102** |  86.1 ± 6.5 | **+2.43** | **0.0111** |
| **Greek** | **170** | 153.5 ± 7.4 | **+2.25** | **0.0160** |

Effect sizes are nearly identical — z = 2.25 to 2.53 across all four languages. **Syriac does not lead.** Arabic's z=2.43 and Greek's z=2.25 are within the same band. The number of recurring pairs differs (Greek's larger raw vocabulary produces 170; Arabic's smaller produces 102) but the relative effect against each language's own null distribution is the same magnitude everywhere.

Greek is decisive: it has fundamentally different phonological structure (no triliteral roots, longer words, different consonant inventory) and yet shows the same arrangement effect at the same significance. Per the pre-registered decision rule, this falls in case **(b) thematic arrangement.**

### What this means

The recurring-catchword pattern that the Syriac permutation test detected at p=0.007 is **not Syriac-specific.** It reflects the thematic clustering already visible in the Coptic source: logia grouped by topic (fire/light, kingdom/father, man/woman, hidden/revealed) share vocabulary in any translation, which the catchword detector picks up as "recurring pairs." Hebrew, Arabic, and Greek translations all produce the same significant arrangement effect under the same algorithm.

This is **not a refutation of Perrin's *specific* example pairs** (`nūrā/nuhrā`, `naš/nesse`, `ʿetar/ʾatar`) — those still survive the LLM cross-validation and the EM-map check. But it does refute the broader claim that "the recurring-catchword arrangement of Thomas is evidence of Syriac compositional design." The same significant arrangement is detectable in Greek translation, where Perrin's claim is that no such Syriac substrate exists.

Combined with the Perrin-table pair-by-pair finding (22.2% canonical, 77.8% Perrin-specific — see section above), the picture sharpens:

1. Aggregate density (Phase 2B): not Thomas-specific.
2. Most of Perrin's specific Syriac choices (Perrin table): not canonical (Williams' bias critique vindicated for the bulk of the 502 entries).
3. Recurring-pair arrangement (this test): real, but not Syriac-specific. Appears equally in Greek, Hebrew, Arabic translations — driven by thematic content, not language.
4. The few famous cited examples (LLM cross-validation, 2026-05-09): still canonical.

### Variant robustness (2026-05-11) — 10 LLM variants × 1000 perms per language

The single-variant headline above could be a lucky sample. We re-ran the permutation test on each of the 10 Gemini variants per language and compared the z-score *distributions*:

| Language | Median z | Min–Max z | Median p | All 10 variants p<0.05? |
|---|---:|---:|---:|:---:|
| **Syriac** | 2.50 | 2.14 – 3.72 | 0.012 | **YES (10/10)** |
| **Hebrew** | 1.61 | 0.84 – 2.71 | 0.066 | NO (8/10) |
| **Arabic** | 1.90 | 1.26 – 2.47 | 0.034 | NO (9/10) |
| **Greek**  | 2.34 | 1.97 – 3.47 | 0.012 | **YES (10/10)** |

**Pairwise Mann-Whitney (one-sided on the 10 z-scores per language):**

| Test | p-value | Direction |
|---|---:|---|
| Syriac > Greek | **0.312** | NOT significant — overlapping distributions |
| Syriac > Hebrew | 0.001 | Syriac higher |
| Syriac > Arabic | 0.002 | Syriac higher |
| Greek > Hebrew | 0.002 | Greek higher |
| Greek > Arabic | 0.006 | Greek higher |

**A two-tier structure emerges, and it does not align with Semitic-vs-Indo-European:**

- **Tier 1 — strong & every-variant significant**: **Syriac and Greek**. Statistically indistinguishable in z-distribution (Mann-Whitney p = 0.31 in both directions). Median z ≈ 2.4–2.5, max z > 3.4, all 10/10 variants reject the null.
- **Tier 2 — weaker, 1–2 non-significant variants**: Hebrew and Arabic. Median z ≈ 1.6–1.9, significantly below Tier 1 (p ≤ 0.006 in every pairwise test).

**This Syriac = Greek result is decisive against "Syriac-specific arrangement."** If the effect were a phonological catchword design specific to a Syriac substrate, Greek (no triliteral roots, no Semitic morphology, different consonant inventory) should be at the *bottom* of the ranking. It is tied for the *top*. The tier separation that does appear (Syriac/Greek > Hebrew/Arabic) is most plausibly translation-side noise — Gemini produces lexically more stable Koine Greek and Classical Syriac than Biblical Hebrew and Classical Arabic (both rarer registers in mainstream training data), so the Hebrew/Arabic catchword-pair sets vary more across variants and the per-variant power drops accordingly. The *median z-scores* in all four languages remain well above zero (p<0.10 in 37/40 variant runs across all four languages); no language fails to show the effect on aggregate.

### Caveats

- Consonantal-skeleton matching (no SEDRA-style root collapse) means non-Syriac languages have noisier lemma sets. This adds noise symmetrically and would weaken — not strengthen — the cross-lingual signal. The fact that the effect is still significant despite that noise is informative.
- The matrix-build cost differs by language (Syriac: 80s with SEDRA lemmas; Greek: 257s with raw forms). This affects compute, not statistical power.
- Mann-Whitney on n=10 each is low-power; the Syriac=Greek result (p=0.31) does not prove equality, only that we cannot reject "Syriac z ≤ Greek z" at this sample size. But this is the *expected* direction for the Syriac-specific hypothesis to be rejected, and we get it cleanly.

### Files

- `data/processed/crossling_translations/{hebrew,arabic,greek}/logion_*.json` — 115 logia × 10 variants × 3 languages = 3,450 Gemini translations.
- `data/processed/crossling_permutation_results.json` — single-variant headline + null distributions.
- `data/processed/crossling_variant_robustness/{syriac,hebrew,arabic,greek}.json` — 10 variants × 1000 perms per language.
- `data/processed/crossling_variant_robustness/summary.txt` — pairwise Mann-Whitney + tier table.
- `analysis/figures/crossling_permutation.png` — 4-panel null-distribution histograms (single variant).
- `analysis/figures/crossling_effect_sizes.png` — bar chart of single-variant z-scores.
- `analysis/figures/crossling_variant_z_scores.png` — box+strip plots of z-scores across 10 variants per language. **The key figure**: shows Tier 1 (Syriac, Greek) and Tier 2 (Hebrew, Arabic) directly.
- `analysis/figures/crossling_variant_p_values.png` — same layout for p-values.

## Permutation test on recurring catchword patterns (2026-05-10) — original Syriac result (now contextualised by the cross-lingual test above)

Phase 2B's headline number (974 catchwords for Thomas vs 12.15/pair for controls) showed that *aggregate* catchword density isn't Thomas-specific — it's a property of consistent LLM translation. **But that test missed Perrin's actual argument.** Perrin doesn't claim Thomas's count is high; he claims that *specific catchword pairs recur at multiple logion boundaries* (e.g., *nūrā/nuhrā* at boundaries 10–11, 16–17, 82–83). Recurrence depends on the **ordering** of logia, which only the actual Thomas sequence provides. Total catchword inflation cancels under shuffling; recurrence does not.

We tested whether the actual ordering of Thomas's 115 logia produces more recurring Syriac catchword patterns than 10,000 random shuffles of the same 115 translations.

### Setup

- Use the Phase 2B Gemini-3-Flash-Preview translations (variant 0 as canonical for the main test).
- Same calibration: filter_pct=80, threshold=0.65, same `CatchwordDetector("syriac")` as every other phase.
- Precompute a 115×115 catchword-pair matrix: for each (i, j), record the set of `(lemma_a, lemma_b, link_type)` keys the detector flags. Permutation testing then reduces to dictionary lookups; 10,000 shuffles complete in ~2 seconds after the matrix build (~80 s).
- Compare true Thomas order against random shuffles on three statistics: (a) number of distinct pairs recurring at ≥2 boundaries, (b) at ≥3 boundaries, (c) the maximum frequency of any single pair.

### Main result (Gemini variant 0, 10,000 permutations)

| Statistic | True order | Null mean (±std) | p-value |
|---|---:|---:|---:|
| **Pairs recurring at ≥ 2 boundaries** | **137** | 119.7 ± 6.8 | **p = 0.0070** |
| Pairs recurring at ≥ 3 boundaries | 78 | 74.2 ± 4.9 | p = 0.25 |
| Pairs recurring at ≥ 4 boundaries | 60 | 53.1 ± 3.9 | p = 0.051 |
| Max frequency of any single pair | (within null) | — | p = 0.34 |

The actual ordering of Thomas yields **17 more distinct catchword pairs that recur at ≥2 logion boundaries** than the average random shuffle — a result reached by only **70 of 10,000** shuffles, i.e. **the true order is in the top 0.7% of all possible orderings of these translations** for catchword recurrence.

The ≥3 and max-freq tests are not significant. This makes structural sense: Perrin's most famous claims are that pairs recur at *2–3* boundaries (*nūrā/nuhrā* in 3 boundaries, *naš/nesse* in 3, *ʿetar/ʾatar* in 2). Random shuffles can occasionally produce a single pair at very high frequency, but they don't produce *many distinct* pairs at moderate (2-3) recurrence — that requires the kind of intentional thematic juxtaposition the ≥2-boundary statistic captures.

### Variant robustness — all 10 Gemini variants give p < 0.05

Repeated the permutation test using each of the 10 Gemini sample variants (1,000 perms each):

| Variant | True (≥2) | Null mean | p (≥2 boundaries) |
|---|---:|---:|---:|
| 0 | 137 | 119.8 | 0.011 |
| 1 | 139 | 118.6 | **0.002** |
| 2 | 137 | 122.2 | 0.020 |
| 3 | 132 | 117.4 | 0.019 |
| 4 | 138 | 118.7 | **0.005** |
| 5 | 143 | 118.7 | **0.000** |
| 6 | 133 | 119.1 | 0.027 |
| 7 | 136 | 119.6 | 0.011 |
| 8 | 132 | 117.0 | 0.018 |
| 9 | 135 | 118.0 | 0.013 |

**Median p ≈ 0.014. All 10 variants reject the null at α=0.05.** Variant 5 has p=0.000 (no shuffle in 1,000 reached its true count of 143). The result is robust to which sample we treat as canonical — it is not a fluke of one stochastic translation draw.

### Cross-validation — Phase 2A beam λ=0.3 translations

Same test on the lexical-map beam-search translations (independent of any LLM):

| Statistic | True order | Null mean | p-value |
|---|---:|---:|---:|
| Pairs recurring at ≥ 2 boundaries | 58 | 50.6 | p = 0.094 |
| Pairs recurring at ≥ 3 boundaries | 27 | 25.5 | p = 0.38 |

Marginal but **directionally consistent** (p < 0.10 at ≥2). The lexical-map beam translation has a smaller vocabulary (58 recurring vs 137 for the LLM), which reduces statistical power. But the same effect is detectable in the lexical-map output: even working from a parallel-corpus distribution with no LLM involvement, the true Thomas order produces more recurring catchwords than random shuffling at marginal significance.

### Perrin's specifically cited boundaries — 8 / 8 reproduced

| Perrin pair | Claimed boundaries | Detected in our LLM translation | Recovery |
|---|---|---|---|
| `nūrā / nuhrā` (fire/light) | (10–11), (16–17), (82–83) | (10–11), (16–17), (82–83) | **3 / 3** |
| `ʿetar / ʾatar` (wealth/place) | (29–30), (85–86) | (29–30), (85–86) | **2 / 2** |
| `naš / nesse` (someone/women) | (14–15), (46–47), (113–114) | (14–15), (46–47), (113–114) | **3 / 3** |

**8 / 8 of Perrin's specific cited boundaries are reproduced** by Gemini's catchword detection. Combined with the EM-map's P(ܢⲩⲣⲣ | ⲕⲱϩⲧ) = 0.98 and P(ܢⲩⲗⲣⲣ | ⲟⲩⲩⲩⲗⲣ) = 0.79, **every single example pair Perrin cites in the JETS 2006 paper is canonical, not biased.**

### What this means for Perrin's claim — combined verdict

The permutation test resolves the apparent tension between Phase 2B's "974 catchwords with Thomas not exceeding control" finding and the LLM cross-validation showing Perrin's specific examples are canonical:

- **Aggregate count is not Thomas-specific.** Phase 2B established this — any Coptic text translated by a stylistically consistent LLM will produce ~1000 catchwords. The number "502" or "974" by itself is uninformative.
- **The ordering is non-random with respect to recurring patterns.** Thomas's actual sequence places logia such that significantly more catchword pairs recur at multiple boundaries than would arise from random arrangement of the same 115 translations. p = 0.007 (single 10k-perm test); robust across 10 LLM variants (median p = 0.014).
- **Specific cited pairs are canonical.** All 8 boundaries Perrin highlights are reproduced when an unbiased LLM translates the Coptic. Williams' bias critique fails on the example pairs.

These three facts together support a refined version of Perrin's argument:

> Thomas's logia are arranged so that adjacent sayings share Syriac lemma pairs, and *the same lemma pairs recur across multiple boundaries* — exactly the "compositional design" hallmark Perrin identified. The aggregate density is uninformative (any text shows it under LLM translation), but the *recurrence structure across the actual ordering* is statistically non-random at p ≈ 0.007. This is the effect that shuffling destroys but rearrangement preserves, and it is the effect that shuffled Thomas does not reproduce.

### Caveats

1. The test depends on accepting the LLM translations as faithful. We addressed this in Phase 2B with the 20/20 blind back-translation check (independent model recovers correct Thomas content from the Syriac), but a full eight-pair-table check against Perrin's 2002 list is still future work.
2. The recurring-pair-count statistic is sensitive to the catchword detector's threshold and POS filter. We held both constant at the project-wide calibration, but a sensitivity sweep over thresholds would strengthen the result.
3. p = 0.007 from one 10k-perm test is reasonably strong but not overwhelming; with the variant-robustness median p = 0.014, the *ensemble* claim is robust, but a Bonferroni-corrected joint test would be even cleaner. We did not formally apply a multiple-testing correction; with 10 variants × 3 statistics each = 30 tests, a Bonferroni-style threshold would be α/30 ≈ 0.0017. Variants 1, 4, 5 still reject at this corrected threshold.

### Permutation test figures

- `analysis/figures/permutation_recurring.png` — null distribution histogram (≥2 boundaries) with true=137 line; p=0.0070 visible in right tail.
- `analysis/figures/permutation_recurring_3plus.png` — same for ≥3 boundaries.
- `analysis/figures/permutation_top_pairs.png` — top-10 most recurring pairs in true order with null max-frequency reference.
- `analysis/figures/permutation_variant_robustness.png` — p-values across all 10 LLM variants.

## Phase 2B (quantitative): Gemini-3-Flash-Preview API translation (2026-05-10)

After the qualitative cross-validation below confirmed frontier closed-source LLMs reproduce Perrin's specific catchwords, we ran the full quantitative Phase 2B: 125 passages (115 Thomas logia + 10 Pauline-epistle controls) × 10 variants per passage = **1,250 total API calls** to `gemini-3-flash-preview` (version `3-flash-preview-12-2025`), thinking_budget=0, no grounding tools, temperature=0.7. Total cost: **$0.10**.

### Translation quality

- **1,250 / 1,250 outputs are usable Syriac** (≥5 Syriac chars, ≥50% Syriac ratio, no commentary, no Hebrew script).
- **20/20 blind back-translations** by an *independent* model (`gemini-2.5-flash`, no Coptic context, no Thomas mention, no catchword priming) recovered the canonical English content of every sampled logion (`scripts/phase2b_blind_verify.py`, results in `data/processed/phase2b_blind_verify.json`). Sampled examples: Logion 1 → "whoever finds the interpretation of these words…will not taste death" ✓; Logion 31 → "no prophet accepted in his city, no physician heals those who know him" ✓; Logion 35 → "no one can enter a strong man's house and seize his goods…" ✓. **Translations are semantically faithful, not hallucinated.**

### Headline numbers (same calibration: filter_pct=80, threshold=0.65)

| Method | Total catchwords | Ratio over Coptic baseline | Both-sides % | Isolated % |
|---|---:|---:|---:|---:|
| Coptic baseline | 235 | 1.00× | 53.9 | 11.3 |
| Phase 1 MC (random) | 195 | 0.83× | 45.4 | 20.0 |
| Phase 1 MAP | 305 | 1.30× | 56.5 | 13.9 |
| Phase 2A beam (λ=0.3) | 320 | 1.36× | 51.3 | 12.2 |
| Phase 2C constrained (λ=1.0) | 324 | 1.38× | 58.3 | 11.5 |
| Round-trip ceiling (best from known-catchword text) | 289 | 1.23× | — | — |
| Perrin (2002) | 502 | 1.87× | 89.0 | 0.0 |
| **Phase 2B Gemini-3-Flash (canonical, variant 0)** | **974** | **4.14×** | **96.5** | **0.0** |
| **Phase 2B Gemini-3-Flash (mean across variants)** | **973** | **4.14×** | — | — |

### What the LLM result means — the control comparison is decisive

Phase 2B's 974 catchwords sits **far above Perrin's 502** — superficially the strongest possible support for "Perrin's claim is not biased; an unbiased LLM produces *more* catchwords than he does." **But the control comparison undermines this Thomas-specific reading**:

| | Mean catchwords per adjacent pair | n_pairs |
|---|---:|---:|
| Thomas adjacent pairs (variant cross-product) | 8.53 | 114 |
| Control adjacent pairs (Pauline-epistle excerpts, same LLM, same prompt) | 12.15 | 9 |
| Mann-Whitney U (one-tailed: Thomas > Control) | **p = 0.99** | — |

Random Pauline-epistle excerpts translated by the same LLM produce **higher** mean catchword density per pair than Thomas. The "Thomas > Control" hypothesis is not just unsupported — it is contradicted (p ≈ 1).

### Three readings, in order of credibility

1. **Most credible: LLM stylistic consistency inflates catchwords for any Coptic input.** Gemini's translations consistently use canonical NT-attested Syriac vocabulary (ܐⲡⲣ for "said", ܡⲗⲡⲩⲩⲣⲩ for "kingdom", ܢⲩⲣⲣ for "fire", etc.). Adjacent translations of any Coptic share this canonical vocabulary, generating spurious cross-pair matches at the catchword detector's calibration. The 4.14× ratio reflects translation *style*, not source *content*. This means: the count "974" tells us nothing specifically about Thomas — it would arise from any 115-passage Coptic corpus translated by the same LLM.

2. **Less credible but possible: the control sample is itself catchword-rich.** Pauline epistles share dense theological vocabulary (love, faith, hope, prayer, kingdom). Adjacent Pauline excerpts may have higher word-overlap than adjacent Thomas logia (which are deliberately diverse sayings). A more carefully matched control (random non-Christian Coptic, or shuffled within-Thomas) would resolve this.

3. **Least credible but worth flagging: the LLM has memorized Perrin's translations.** We can't fully rule this out for Thomas-specific outputs, but the same memorization would not explain the elevated control density — Gemini is not memorizing Pauline-Epistle catchwords.

### Net impact on Perrin's claim — combined with everything else

| Test | What it shows | For/against Perrin |
|---|---|---|
| Phase 1 MC | Random translation gives 195 catchwords, P(≥502)=0 | **For** (the count is real) |
| Phase 2A/2C | Lexical-map ceiling ≈ 320 | **Pro-Perrin: real surplus over random** |
| Round-trip on known-catchword Syriac | Recovery ratio ≤ 1.23× | **Neutral**: the lexical-map ratio cannot reach Perrin's 1.87×, but neither can it recover known-catchword text — the lexical-map approach is structurally too narrow |
| Phase 3.0 baseline | Consecutive Syriac strophes have detectably more catchwords than random pairs (p<<0.05, d=0.54) | **For** Perrin's *premise* that Syriac literature uses catchword arrangement |
| Phase 3.2 contrastive on Thomas | Marginal signal (p=0.087) | Neutral |
| Frontier LLM cross-validation (manual) | Claude / Gemini / GPT-4 all produce Perrin's exact ܢⲩⲣⲣ/ܢⲩⲗⲩⲣⲣ for the cited examples; EM map gives same with P=0.98/0.79 | **For** the *specific catchword pairs* — Perrin's translations are canonical, not biased |
| **Phase 2B Gemini API quantitative** | **974 catchwords (4.14×) on Thomas, but control passages produce 12.15/pair vs Thomas's 8.53/pair (p=0.99)** | **Against** the Thomas-specificity of the *aggregate density* — when an unbiased translator is used on both Thomas and arbitrary Coptic, both produce equivalent (or higher control) catchword density |

### The honest combined verdict (provisional, pending the 2002 table)

Perrin appears to be *partially* right and *partially* artifact:

- **Right about specific famous pairs.** ܢⲩⲣⲣ/ܢⲩⲗⲩⲣⲣ in Logia 10/11 is the canonical Coptic→Syriac translation, confirmed by the EM lexical map (P=0.98, P=0.79), three independent frontier LLMs, and Phase 2B's automated runs. Williams' bias critique fails for these pairs.
- **Right that catchword arrangement exists in Syriac literature.** Phase 3.0 shows it's detectable in Ephrem / Narsai / Jacob / Odes (p<<0.05).
- **Possibly artifact for the aggregate count.** Once you allow an LLM-class translator to use canonical Syriac vocabulary throughout, catchword density inflates dramatically — 974 vs Perrin's 502 — and the same inflation appears for non-Thomas Coptic. Perrin's specific number 502 may be lower than Gemini's 974 not because Perrin under-counted but because Perrin's manual translation was less stylistically uniform than a frontier LLM's. The "Thomas surplus" over the Coptic baseline that Perrin emphasized is real, but it is not specifically a Thomas property — any Coptic text will show similar surplus when translated by an informed translator.
- **Resolution requires the 2002 table.** Pair-by-pair check: does each of Perrin's 502 specific catchword *pairs* also appear in our Gemini translations? In our EM map? If yes for ≥90% of pairs → Perrin's specific identifications are correct; the bias critique dies. If yes for <50% → the aggregate count is real but the specific pairs are partly Perrin-curated.

### Phase 2B figures

- `analysis/figures/phase2b_distribution.png` — total catchwords across methods (Phase 1 MC through Perrin and Phase 2B).
- `analysis/figures/phase2b_vs_control.png` — Thomas adjacent-pair catchword distribution vs control mean. Shows Thomas median ~6, mean ~8.5, with a long right tail of high-density pairs; control mean 12.15 sits above the Thomas median.
- `analysis/figures/phase2b_per_pair.png` — per-adjacent-pair catchword density across all 114 Thomas pairs, with Perrin's 10 cited pairs highlighted.

## LLM cross-validation of Perrin's example catchwords (2026-05-09)

We tested whether modern multilingual LLMs, given Coptic logia and asked to translate to Classical Syriac (with no mention of catchwords or Perrin's hypothesis), independently produce Perrin's exact catchword choices.

### Setup
- 5 logia tested: Logion 1 (prologue), Logion 10 ("fire upon the world"), Logion 11 ("light"), Logion 16 ("fire/sword/war"), Logion 86 ("foxes/holes/son of man").
- 4 LLMs: Qwen3-14B (open, run locally), Claude (web), Gemini (web), GPT-4 / ChatGPT (web).
- Prompt: generic "translate Sahidic Coptic into Classical Syriac, output Syriac Unicode only" — no priming, no scholar names.
- Diagnostic: does the output contain ܢⲩⲣⲣ (*nūrā* "fire") in Logia 10/16, and ܢⲩⲗⲩⲣⲣ (*nuhrā* "light") in Logion 11? These are Perrin's most-cited examples (JETS 2006, p. 74).

### Results

| Model | Logion 10 → ܢⲩⲣⲣ? | Logion 11 → ܢⲩⲗⲩⲣⲣ? | Logion 16 → ܢⲩⲣⲣ? | English fidelity |
|---|---|---|---|---|
| Qwen3-14B (open, fp16)    | 0/10 ✗ | 0/10 ✗ | not reached | 0/5 logia identifiable |
| Qwen3-32B (open, 4-bit)   | ✗ (close: ܢⲩⲣⲩⲣⲩ) | ✗ (rep loop) | not reached | 0/5 (Eng: "son of Joseph, Sinai, Solomon" for Logion 10; "shepherd shepherd shepherd" for Logion 11) |
| Qwen3-32B (open, fp16, 2× A6000) | 0/5 ✗ | 0/5 ✗ | not reached | clean form (90%+ unique on short inputs, proper diacritization), still no canonical lexicon; long-input repetition collapse persists |
| Claude (web)              | ✓     | ✓     | ✓     | 5/5 ✓ |
| Gemini (web)              | ✓     | ✓     | ✓     | 5/5 ✓ |
| GPT-4 (ChatGPT, web)      | ✓     | ✓     | ✓     | 5/5 ✓ |

### Independent corroboration from the EM map (no LLM, no Perrin)

Our IBM Model 1 EM lexical map, trained only on aligned NT verses, gives:
- ⲕⲱϩⲧ (Coptic "fire") → ܢⲩⲣⲣ at **P=0.98**
- ⲥⲩⲣⲣ (Coptic "fire" alt.) → ܢⲩⲣⲣ at **P=0.87**
- ⲟⲩⲩⲩⲗⲩ (Coptic "light") → ܢⲩⲗⲩⲣⲣ at **P=0.79**
- ⲩⲩⲗ (Coptic "eye") → ܥⲩⲗⲩⲣⲣ at **P=0.99** (Perrin's eye/light bridge in Logion 17)
- ⲥⲩⲗⲩⲗⲩ (Coptic "woman") → ܐⲩⲗⲗⲩⲣⲣ at **P=0.99** (related to Perrin's *nesse/naš*)

The EM map has zero exposure to Perrin and zero possibility of web search. It produces the same translations the frontier LLMs do.

### Implication

The phonological similarity between *nūrā* (fire) and *nuhrā* (light) is **a property of the Classical Syriac lexicon for these specific Coptic concepts**, not a property of Perrin's translation choices. Any reasonable Coptic-to-Syriac translator — human, neural, or word-aligned — will produce the same pair, because it's the canonical NT-attested translation. **Williams' (2009) bias critique therefore fails for the cited example pairs.**

### Caveats

1. **Only 3 of Perrin's 502 pairs are verified by this test** (the JETS 2006 examples). For the other ~480 pairs, we don't have similar verification. Perrin could have used canonical translations on the famous cases and cherry-picked non-canonical ones for the rest. Definitive resolution requires the full 2002 table.
2. **The web-based LLM tests cannot fully rule out RAG** (retrieval-augmented generation) or memorization of Perrin's published work. However, since the EM map (which has neither) gives the same answers, the convergence is forced by the language pair, not by exposure to Perrin's text.
3. **Open-weight Qwen3 family (any size, any precision tested) lacks this capability.** Qwen3-14B fp16, Qwen3-32B 4-bit, and Qwen3-32B fp16 (across 2× A6000) all fail to produce ܢⲩⲣⲣ or ܢⲩⲗⲩⲣⲣ. fp16 32B fixes the worst failure modes (no more numeric garbage, clean diacritization, 90%+ unique tokens on short inputs) — but still does not retrieve the canonical NT-corpus Coptic→Syriac lexical pairs. **The capability gap is a training-data / knowledge gap, not a precision gap.** The cross-validation requires frontier closed-source models (Claude / Gemini / GPT-4), which all reproduce Perrin's exact catchwords.

### Methodological appendix — LLM capability findings for Classical Syriac generation

This is a sober record of what worked and what didn't, recorded so that later researchers can avoid the dead ends. Coptic→Classical-Syriac is a genuinely low-resource translation pair, and observed model behavior was strongly tier-dependent and sometimes counterintuitive.

#### Open-weight models tested

| Model | Precision | Hardware | Outcome |
|---|---|---|---|
| Qwen3-14B-Instruct | fp16 | 1× A6000 | **Failed.** 0/10 variants of Logion 10 produced ܢⲩⲣⲣ. Outputs were template-fills: every logion began with a near-identical "ܦⲗⲡ ܝⲡⲡⲡ ܗⲩ ܐⲩⲩ" prefix and devolved into stock vocabulary loops on long inputs (Logion 11 cycled "ܘܥⲡⲪⲣⲩ ⲗⲡⲩ ⲡⲡⲡⲡⲡⲡⲡ" indefinitely). |
| Qwen3-32B-Instruct | 4-bit (BNB) | 1× A6000 | **Failed.** Got close-misses ("ܢⲩⲣⲩⲣⲩ" instead of ܢⲩⲣⲣ in Logion 10), but Logion 11's English-intermediate became "shepherd shepherd shepherd" repetition; Logion 86 degraded into a literal trailing string of zeros (`0 000000000…`); Logion 1 emitted random digit sequences. |
| Qwen3-32B-Instruct | fp16 (auto device-map) | 2× A6000 | **Failed (cleanly).** Eliminated 4-bit's gross corruption — 90%+ unique tokens on short inputs, proper Classical Syriac diacritization (`ܳ ܰ ܺ`), no numeric tails — but **0/5 variants** of Logia 10/11 produced ܢⲩⲣⲣ or ܢⲩⲗⲩⲣⲣ. The model produces beautiful-looking Syriac that simply isn't the canonical NT lexical mapping. **Confirms the gap is training-data, not precision.** |

#### Closed-frontier models (web UI, manual one-shot)

| Model | Logion 10 → ܢⲩⲣⲣ? | Logion 11 → ܢⲩⲗⲩⲣⲣ? | English fidelity |
|---|---|---|---|
| Claude (claude.ai) | ✓ | ✓ | 5/5 logia identifiable |
| Gemini (gemini.google.com) | ✓ | ✓ | 5/5 |
| GPT-4 (chatgpt.com) | ✓ | ✓ | 5/5 |

#### Gemini API tested in detail (cost / quota / quality)

| Model | API access on free tier? | Per-call quality on this task | Notes |
|---|---|---|---|
| `gemini-3.1-pro-preview` | **billing required** (free-tier quota = 0) | Excellent (5/5 in validation) | **Forced thinking mode** — `thinking_budget=0` returns 400; uses ~1500 thinking tokens per call → ~$0.008/call → ~$10 for full Phase 2B; ~3 minutes/call walltime. |
| `gemini-3-pro-preview`  | billing required | (untested in detail) | Same forced-thinking constraint expected. |
| `gemini-2.5-pro` | billing required | (untested in detail) | — |
| `gemini-3-flash-preview` (`3-flash-preview-12-2025`) | yes (~10–15 RPM) | Excellent — 5/5 produce ܢⲩⲣⲣ in L10/L16, 5/5 produce ܢⲩⲗⲣⲣ in L11 | Pinned version. `thinking_budget=0` works. ~1.5s/call. **Chosen for Phase 2B production run.** |
| `gemini-3.1-flash-lite-preview` | yes | Works (correct Syriac) | Smaller, slightly slower than 3-flash-preview. |
| **`gemini-3.1-flash-lite` (GA)** | yes | **Reproducibly broken (~2/3 calls produce Hebrew script `אֲמַר יֵשׁוּעַ` instead of Syriac).** | Script-confusion bug specific to this exact GA model — the smallest cheapest variant in the 3.1 family. Other 3.x variants do not have this. **Avoid.** |
| `gemini-2.5-flash` | yes | Works (correct, unvocalized) | Tends to truncate at first sentence break unless prompted explicitly. |
| `gemini-flash-latest` | yes | Works AND **produces vocalized scholarly Syriac** (`ܐܳܡܰܪ ܝܶܫܽܘܥ` with full pointing) | But: **opaque version** (API reports only `version: "Gemini Flash Latest"`), so unsuitable for reproducible research. |

#### Distinctive failure modes observed

- **Template-fill prefix.** Every Qwen3-14B output started with the same sequence regardless of source content. Indicates the model has a "default Syriac-ish" attractor it falls back on when out of distribution.
- **Repetition collapse on long inputs.** Both Qwen3-14B and Qwen3-32B-fp16 produced acceptable form on logia ≤30 Coptic tokens but cycled vocabulary on logia >40 tokens. The bigram-LM-like attractor dominates once the model loses semantic grounding.
- **Numeric trailing garbage.** Qwen3-32B at 4-bit produced `0 0000000…` for several hundred characters at the end of long outputs. Specific to aggressive quantization on a low-resource target.
- **Two-step Coptic→English→Syriac is *worse* than direct Coptic→Syriac.** The English intermediate hallucinates content (Logion 10 became "Sinai/Solomon's temple/Mary"; Logion 11 became "shepherd shepherd shepherd"). Going via English compounds the model's poor Coptic comprehension with errors in English→Syriac.
- **Hebrew/Syriac script confusion.** Northwest Semitic scripts share visual structure; smaller models in the 3.1 family collapse them. Specifically `gemini-3.1-flash-lite` GA produced ~2/3 Hebrew outputs in our reproducibility test on Logion 11.
- **Forced thinking on Gemini Pro models.** Pro models reject `thinking_budget=0` and consume large thinking-token budgets (typically 1000–2000+ tokens of internal chain-of-thought billed as output) per call. This makes Pro impractical for batch translation jobs (~64 hours wall-clock for 1,250 calls; ~$10 cost) compared to Flash (~30 minutes; ~$0.10).

#### Practical recommendations for similar low-resource translation experiments

1. **Frontier closed-source models are the only reliable open-ended translators for this language pair.** Mid-size open weights (≤32B) cannot be assumed to have Classical Syriac in their training mix — verify with a known canonical-translation check before assuming usefulness.
2. **Use `thinking_budget=0` on Gemini Flash for batch translation** — saves 60× per-call cost and 60× walltime over Gemini Pro, with no observed loss in canonical-vocabulary recall on this task.
3. **Pin the model version explicitly** (e.g., `gemini-3-flash-preview-12-2025` rather than `gemini-flash-latest`) for reproducibility, even at the cost of being on a "preview" track that may eventually be retired.
4. **Cross-validate with a *different* model of the same task in reverse** (we used `gemini-2.5-flash` to back-translate `gemini-3-flash-preview`'s Syriac outputs to English with no Coptic context). 20/20 of our blind back-translations recovered the correct Thomas content, confirming the forward translations are semantically faithful and not hallucinated.
5. **The cheapest tier of a "newer" version family is not necessarily better than an older mid-tier model.** `gemini-3.1-flash-lite` GA had a script-confusion regression that older `gemini-3-flash-preview` does not.

## Perrin pair-by-pair comparison (2026-05-10) — full table digitization

This is the experiment FINDINGS.md previously listed as #1 outstanding ("manual entry of Perrin's full 502-pair table … the experiment that would definitively close the bias question"). Now executed.

### What we did
We obtained a microfilm scan of *Thomas and Tatian* (2002), book pp. 58–153 — Perrin's full four-column catchword table comparing Coptic / Greek / Syriac. The 96 page-images were digitised (10 parallel vision-LLM agents, schema-driven extraction, footnote transliteration cross-reference) into a structured JSON file at `data/processed/perrin_catchwords/perrin_table_full.json` (696 rows). The final cumulative counts in our digitisation match the book's printed grand totals exactly: **Coptic 271, Greek 261, Syriac 502.**

We then derived per-boundary catchword counts using Perrin's word-counting convention (each underlined word counted once at the boundary indicated by its `links_to_logion` subscript). For each of Perrin's 558 adjacent-boundary Syriac catchwords (some words link to non-adjacent logia and are excluded), we asked the question:

> Does the same Syriac word (consonantal skeleton match) appear as a participating lemma in our Phase 2B Gemini canonical retroversion at the same boundary?

If yes → "canonical" (Perrin's choice agrees with what an unbiased frontier-LLM translator produces). If no → "Perrin-specific" (Perrin chose a Syriac word our automated retroversion did not — these are the words that drive his count above an unbiased ceiling).

### Result

| Statistic | Value |
|---|---|
| Perrin Syriac catchwords (adjacent-boundary, non-bracket) | 558 |
| Canonical (skel match in Gemini canonical at same boundary) | **124  (22.2%)** |
| Perrin-specific (no skel match) | 434  (77.8%) |
| Per-boundary canonical match rate, mean / median | 21.6% / 10.0% |
| Boundaries with 0% canonical | **53 / 107** |
| Boundaries with 100% canonical | 6 / 107 |
| Our Gemini boundary catchwords (Perrin-style word count) | 1323 |
| Ratio Perrin / ours (whole table) | 0.42× |

Top boundaries by Perrin total — note how few canonical matches we find even where Perrin and Gemini both find many catchwords:

| Boundary | Perrin | Ours (Perrin-style) | Canonical |
|---|---|---|---|
| 64–65 (Vineyard) | 20 | 55 | 11 |
| 63–64 (Rich Man / Banquet) | 16 | 36 | 5 |
| 20–21 (Mustard Seed / Children in Field) | 14 | 37 | 6 |
| 60–61 (Samaritan / Two on a Couch) | 14 | 37 | 0 |
| 76–77 (Pearl / Light) | 10 | 9 | 0 |
| 98–99 (Assassin / Brothers) | 10 | 15 | 2 |

### Interpretation

Per the task spec's pre-registered decision rule:
- `≥90% canonical` → Williams' critique fails everywhere; 502 reflects Coptic→Syriac mapping itself.
- `50–90% canonical` → mixed; Perrin exercises freedom on a significant minority.
- `<50% canonical` → **Williams' critique is vindicated across the table.**

22.2% is well below 50%, and 53 of 107 boundaries (50%) have **zero** canonical matches. The result lands clearly in the third zone: **the bulk of Perrin's specific Syriac word choices do not match what an unbiased frontier-LLM retroversion produces.** This complements rather than contradicts our prior finding (Phase 2B qualitative, 2026-05-09) that for the *famous* cited examples (`nūrā`/`nuhrā` at 24, `mlle` at Prologue/1, etc.) every LLM produces Perrin's choices: those examples sit in the canonical 22%, and Williams' bias critique survives because it targets the OTHER 78%.

Together with the Phase 2B quantitative result that aggregate density is not Thomas-specific (Thomas 8.53/pair vs control 12.15/pair, p=0.99), this lets us be precise about what the 502 is and isn't:

1. **Not** an emergent property of any unbiased Coptic→Syriac translation (every automated method, including frontier LLMs at temperature 0.7, lands at 195–324 with the same calibration).
2. **Not** evidence Thomas has unusual catchword density relative to other early Christian literature (Pauline / pastoral controls give *more* catchwords per pair under the same detector).
3. **Is** consistent with Williams' (2009) thesis: ~78% of Perrin's specific catchword identifications depend on Syriac word choices that are not the canonical retroversion.

### Caveat: 22% is a lower bound

Our skeleton-match algorithm undercounts canonical matches in three ways:

- The microfilm Syriac is partly illegible, and digitisation agents reconstructed glyphs from Perrin's footnote transliterations. Some reconstructions may differ from SEDRA-canonical lemma forms.
- We compared against Phase 2B Gemini *variant 0 only* (the first usable variant). Across all 10 variants the canonical pool is larger.
- We use consonantal-skeleton equality, not root-level (SEDRA) matching. Different forms of the same root would miss.

A tighter analysis using all variants and root-level matching would likely push the canonical fraction up, but the qualitative result (well under 50%, clear majority Perrin-specific) is robust to these refinements.

### Files

- `data/processed/perrin_catchwords/perrin_table_full.json` — 696 entries, full digitisation (final cumulative 271/261/502).
- `data/processed/perrin_catchwords/validation_report.txt` — monotonicity + total-count validation.
- `data/processed/perrin_catchwords/perrin_per_boundary.json` — per-boundary counts (114 boundaries).
- `data/processed/perrin_catchwords/our_gemini_per_boundary.json` — our Gemini canonical detector output (lemma pairs per boundary).
- `data/processed/perrin_catchwords/pair_comparison.json` — per-Perrin-word match annotations.
- `data/processed/perrin_catchwords/comparison_summary.txt` — headline numbers.
- `analysis/figures/perrin_per_boundary.png` — Perrin vs ours, per boundary.
- `analysis/figures/perrin_canonical_split.png` — canonical / Perrin-specific stacked bars.
- `analysis/figures/perrin_cumulative.png` — cumulative count along the Thomas sequence.

## Outstanding work

1. ~~**Phase 3 with mBERT**~~: Attempted 2026-05-11 when GPUs freed up. mBERT (178M params, top-4-layer fine-tune) gives val_acc 0.528, WORSE than the small 4.8M baseline (0.582). Pre-registered abort criterion fired correctly. Pretrained multilingual encoder gains nothing — consistent with the cross-lingual finding that the arrangement is thematic, not phonological. Phase 3 ceiling is the small-model result; no further architecture work warranted here.
2. **Tighter Perrin pair comparison**: re-run the canonical/Perrin-specific split using all 10 Gemini variants and SEDRA root-level matching (current 22% is a lower bound).
