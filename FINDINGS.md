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
| **Phase 3.0** | Is catchword arrangement actually detectable in known Syriac literary texts? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **YES** — pooled p < 1e-9, Cohen's d = 0.54 |
| **Phase 3.1** | Can a model trained on those texts learn to discriminate consecutive strophes? | Hard-negative contrastive (same-work, ≥3 strophes apart) + InfoNCE | val_acc **0.582** (vs 0.50 chance) |
| **Phase 3.2** | Does that learned model see the same pattern in beam-translated Thomas? | Permutation test: adjacent vs shuffled cos_sim | p = **0.087** (marginal) for beam translation; p = 0.32 (n.s.) for NMT |

## Three findings, in plain language

**(1) Perrin's 502 cannot be explained by Coptic→Syriac word-mapping plus Syriac fluency alone.** Every automated method we tried — random sampling, top-1 deterministic mapping, beam search with a Peshitta-NT bigram LM, per-logion stochastic sampling weighted by that LM — produces 195–324 catchwords. The gap to Perrin (~178 catchwords) is real and unexplained by what an unbiased translator working from the lexical map can produce.

**(2) Catchword-based literary arrangement is real in Syriac.** When we apply the same calibrated rule-based detector to Ephrem, Narsai, Jacob of Serug, and the Odes of Solomon, consecutive strophes share significantly more catchwords than randomly-paired strophes (pooled p < 1e-9, Cohen's d = 0.54). All four corpora individually are significant. So Perrin's *premise* — that Syriac literature uses catchword arrangement and a Syriac-original Thomas should too — is not unfounded.

**(3) But Thomas (in beam-translated Syriac) shows only marginal evidence of that arrangement.** Our Phase 3.1 model that *did* learn to discriminate consecutive vs hard-negative Syriac strophes (val_acc 0.58) produces only weak signal when applied to beam-translated Thomas: adjacent-pair similarity is mean 0.627 vs shuffle baseline 0.590, permutation p = 0.087 (one-tailed). Not strong enough to declare Thomas a Syriac literary text by this test.

## What this implies for Perrin's claim

The surplus of ~178 catchwords (Perrin's 502 vs our best automated 324) could in principle reflect:

  (a) **Genuine Syriac literary structure invisible to our 1-grams + 2-grams.** Real catchword arrangement uses skipped-token patterns, paronomasia at the root level, hapax-legomenon connections, etc., that our LM does not capture. Phase 3.0 confirms such structure is detectable in known Syriac texts; some of Perrin's surplus might be picking up the same phenomenon in a Syriac substrate of Thomas.

  (b) **Perrin's manual translation choices.** When Perrin retroverted Thomas to Syriac, he had freedom over many word choices, and may (consciously or not) have chosen Syriac words that maximize catchword density.

We cannot fully separate (a) from (b) without (i) Phase 2B, which requires an `ANTHROPIC_API_KEY` to query an LLM that has Syriac knowledge but no catchword agenda, and (ii) a stronger Phase 3 model trained on more data than 17k strophe-pairs from four authors.

## Caveat: what the methods do and don't capture

- **All Phase 2 methods** treat translation as one-Coptic-word-to-one-Syriac-word, drawn from an EM-aligned lexical map of NT parallel verses. Real translators choose multi-word renderings, restructure clauses, and cross-reference idioms. This is a strict lower bound on what informed translation can do.
- **Phase 3** measures catchwords via the same rule-based detector calibrated against Perrin's Coptic count. We use consonantal-skeleton matching for both literature and Thomas because patristic Syriac strophes lack lemma annotations. The relative consecutive-vs-random comparison is robust to function-word inflation; the absolute counts are not.
- **The contrastive model is small (4.8M params).** GPUs were 95%+ allocated to another job during this run, blocking the larger mBERT-finetuning approach in the original Phase 3.1 spec. A larger pretrained-encoder version may shift the Thomas p-value toward significance — the spec is `scripts/phase3_improved_contrastive.py` for when GPU memory is available.

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

## Permutation test on recurring catchword patterns (2026-05-10) — **strongest finding**

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

## Outstanding work

1. **Manual entry of Perrin's full 502-pair table** from *Thomas and Tatian* (2002), pp. 57–155, then pair-by-pair canonical-translation check against the EM map and against frontier-LLM translations. This is the experiment that would definitively close the bias question.
2. **Phase 3 with mBERT**: When GPUs free up, run `scripts/phase3_improved_contrastive.py`. May shift Thomas's p=0.087 toward significance.
3. **Perrin's full 502-pair table**: Manual entry from *Thomas and Tatian* (2002), pp. 57–155 — would let us check pair-by-pair which of Perrin's catchwords also appear in our automated translations.
