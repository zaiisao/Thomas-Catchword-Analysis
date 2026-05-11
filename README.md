# Gospel of Thomas Catchword Analysis

A computational test of Nicholas Perrin's claim ([*Thomas and Tatian*, 2002](https://www.sbl-site.org/publications/Books_AcademiaBiblica.aspx); [*JETS* 49/1, 2006](data/raw/perrin_catchwords/perrin_2006_jets.pdf)) that the Gospel of Thomas was originally composed in Syriac, indexed by the density of catchwords linking adjacent logia. We replace Perrin's manual retroversion with automated, unbiased Coptic→Syriac translation (Monte Carlo over an EM lexical map, beam-search NMT, frontier-LLM API into Syriac/Hebrew/Arabic/Greek) and apply a uniform catchword detector across every language — directly addressing P. J. Williams' ([2009](https://brill.com/view/journals/vc/63/1/article-p71_3.xml)) bias critique.

## Headline finding

> **Thomas's logia are arranged non-randomly with respect to recurring catchword pairs — but the effect is thematic, not Syriac-specific.** Permutation tests on 10,000 random shuffles give p = 0.007 (Syriac), 0.017 (Hebrew), 0.011 (Arabic), and **0.016 (Greek)** at variant 0 of the Gemini retroversion. A 10-variant robustness sweep separates the languages into two tiers: Syriac and Greek are statistically indistinguishable (Mann-Whitney p = 0.31, 10/10 variants significant in each, median z ≈ 2.4–2.5); Hebrew and Arabic sit lower (median z ≈ 1.6–1.9). Greek matching Syriac is decisive against any Syriac-specific phonological design — the arrangement is thematic clustering that any faithful translation preserves.

Combined with the **full Perrin-table digitization** (book pp. 58–153, 696 entries, totals 271/261/502 matching the book): of Perrin's 558 adjacent-boundary Syriac catchwords, only **22.2% are canonical** (skeleton-match the unbiased Gemini retroversion at the same boundary); **77.8% are Perrin-specific**, with 53 of 107 boundaries showing 0 canonical matches. Williams' bias critique is vindicated across the whole table, while the few famous examples Perrin cites (e.g., *nūrā/nuhrā*) remain canonical (Phase 2B qualitative + EM map check).

See [FINDINGS.md](FINDINGS.md) for the full results, [data/q_source/q_findings_summary.md](data/q_source/q_findings_summary.md) for the Q-source extension test, [WRITEUP.md](WRITEUP.md) for the long-form Phase 1–3 writeup, and [PHASE1_FINDINGS.md](PHASE1_FINDINGS.md) for the Phase 1 standalone Monte Carlo report.

## Phases

| Phase | Question | Method | Result |
|---|---|---|---|
| **1** | Can random Coptic→Syriac translation reproduce Perrin's 502 catchwords? | Monte Carlo over EM lexical map (10k samples) | mean **195** (CI 175–216), P(≥502) = 0 |
| **2A** | Does adding Syriac fluency get us closer? | Beam search with bigram LM | best **320** at λ=0.3 |
| **2C** | Per-logion stochastic sampling at maximum fluency | 200 sims at λ=1.0 | mean **324** (CI 303–347), P(≥502) = 0 |
| **2B** | Does a frontier LLM (no catchword agenda) reproduce Perrin's specific pairs? | Gemini-3-Flash-Preview, 1,250 calls, with controls | **Yes** for specific pairs (8/8); aggregate count NOT Thomas-specific (controls 12.15/pair > Thomas 8.53/pair, p ≈ 0.99) |
| **Permutation (Syriac)** | Is the *ordering* of logia non-random? | 10,000 shuffles of Syriac LLM retroversion | **p = 0.007** at ≥2-recurrence; robust across 10 LLM variants (median p ≈ 0.014) |
| **Cross-lingual permutation** | Is that ordering effect Syriac-specific, Semitic-general, or thematic? | Same test on Gemini Hebrew/Arabic/Greek retroversions; 10-variant Mann-Whitney across z-scores | **Thematic.** All 4 languages significant. Greek ≈ Syriac (M-W p = 0.31, 10/10 variants in each tier); Hebrew/Arabic below. |
| **Perrin table (full)** | Of Perrin's 502 specific Syriac catchwords, how many also appear at the same boundary in unbiased Gemini retroversion? | Digitization of book pp. 58–153 + consonantal-skeleton match | **22.2% canonical / 77.8% Perrin-specific**; 53/107 boundaries with 0 canonical |
| **Q-source extension** | Does the same effect appear in Q? Does Aramaic distinctively lead? | Same pipeline on 56 IQP Q pericopes + 10 controls, in Aramaic/Syriac/Hebrew/Arabic + Greek source | Weaker signal; only Syriac p<0.05 at variant 0; **Aramaic does NOT distinctively lead** (tied with Syriac, M-W p = 0.26) |
| **3.0** | Is catchword arrangement detectable in known Syriac literature? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **Yes** — pooled p < 1e-9, Cohen's d = 0.54 |
| **3.1** | Can a model learn to discriminate consecutive Syriac strophes? | Hard-negative contrastive (InfoNCE) | val_acc **0.582** (vs 0.50 chance); 178M-param mBERT fine-tune does worse (0.528) |
| **3.2** | Does that model see the same pattern in beam-translated Thomas? | Permutation test on cos_sim | p = 0.087 (marginal) |
| **Round-trip** | Empirical ceiling of lexical-map recovery on known-catchword text? | Syr → Cop → Syr round-trip on Ephrem/Narsai/Jacob/Odes | MAP ≤ 1.23×; MC ≤ 1.09×. Perrin's 1.87× is **unreachable by any lexical-map method**. |

## Layout

```
phase1_montecarlo/           # uniform catchword detector, Monte Carlo over EM lexical map
phase2_neural_translation/   # NMT model + tokenizer (parallel-NT-trained Coptic→Syriac)
phase3_contrastive/          # contrastive encoder for adjacent-strophe discrimination
analysis/                    # cross-phase comparison, plots, statistical tests
analysis/figures/            # paper-ready PNG/PDF figures
analysis/figures/q_source/   # Q-source figures
configs/config.yaml          # single source of truth — thresholds, paths, calibration targets
scripts/                     # ingestion, translation, permutation tests (Thomas + Q + cross-lingual + Perrin table)
data/raw/perrin_catchwords/  # JETS 2006 PDF + extracted catchword examples (small, tracked)
data/raw/...                 # large external corpora — gitignored, regenerable via scripts/fetch_data.sh
data/processed/              # built artifacts (lexical map, translations, checkpoints) — gitignored
data/q_source/               # Q inputs + summary findings (small, tracked); Gemini translations gitignored
```

The intermediate computational artifacts (lexical-map JSONLs, Phase 2/3 model checkpoints, processed corpora, bulk Gemini retroversion outputs) are not committed — see [DATA_STATUS.md](DATA_STATUS.md) and `.gitignore` for what regenerates from where.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Phase 0 — fetch external corpora (Peshitta, Coptic SCRIPTORIUM, Ephrem, etc.)
bash scripts/fetch_data.sh
python scripts/parse_peshitta_tei.py
python scripts/parse_coptic_tt.py
python scripts/parse_thomas_tei.py
python scripts/annotate_peshitta_lemmas.py
python scripts/build_lexical_map.py
```

For any Gemini API step (Phase 2B, cross-lingual retroversion, Q-source fetch/translate), put `GEMINI_API_KEY=...` in `.env.local` (gitignored).

## Reproduction (post-Phase-0)

```bash
# Phase 1 — Monte Carlo
python scripts/run_monte_carlo.py

# Phase 2 — three independent translation methods at the same calibration
python scripts/phase2a_beam_translate.py
python scripts/phase2c_constrained_sample.py
python scripts/phase2b_gemini_translate.py     # 1250 API calls, ~30 min, ~$0.10

# Permutation test on recurring catchword patterns (the original Syriac result)
python scripts/permutation_test_recurring.py

# Cross-lingual permutation — translate Thomas into Hebrew/Arabic/Greek and re-run the test
python scripts/crossling_translate.py
python scripts/crossling_permutation_test.py
python scripts/crossling_variant_robustness.py

# Perrin table — full digitization + pair-by-pair comparison
#   (requires data/processed/perrin_catchwords/batches/ from the vision-LLM digitization run;
#   the source scans are copyrighted Brill material and are gitignored)
python scripts/perrin_merge_batches.py
python scripts/perrin_derive_boundaries.py
python scripts/perrin_pair_comparison.py

# Q-source extension
python scripts/q_fetch_greek.py
python scripts/q_translate.py
python scripts/q_aggregate_density.py
python scripts/q_permutation_test.py
python scripts/q_variant_robustness.py

# Phase 3 — gate on baseline first
python scripts/phase3_baseline_test.py
python scripts/phase3_train_hardneg.py --device cuda:N --epochs 30 \
    --d-model 192 --n-layers 4 --batch-size 8 --max-len 64
python scripts/phase3_apply_improved.py --device cuda:N --max-len 96

# Round-trip pipeline-validation experiment
python scripts/build_reverse_lexical_map.py
python scripts/roundtrip_translate_to_coptic.py
python scripts/roundtrip_retranslate_to_syriac.py
python scripts/roundtrip_pair_survival.py

# Synthesis figures
python analysis/phase2_comparison.py
python analysis/final_comparison.py
python analysis/roundtrip_analysis.py
python analysis/phase2b_analysis.py
python analysis/plot_permutation_test.py
python analysis/plot_crossling_permutation.py
python analysis/plot_crossling_variant_robustness.py
python analysis/plot_perrin_comparison.py
python analysis/plot_q_results.py
```

## Key figures

- [`analysis/figures/final_summary.png`](analysis/figures/final_summary.png) — paper-ready 2-panel synthesis
- [`analysis/figures/crossling_variant_z_scores.png`](analysis/figures/crossling_variant_z_scores.png) — **the key figure**: Tier 1 (Syriac ≈ Greek) vs Tier 2 (Hebrew, Arabic) across 10 LLM variants per language
- [`analysis/figures/crossling_permutation.png`](analysis/figures/crossling_permutation.png) — 4-panel null-distribution histograms at variant 0
- [`analysis/figures/perrin_canonical_split.png`](analysis/figures/perrin_canonical_split.png) — per-boundary canonical (22.2%) vs Perrin-specific (77.8%)
- [`analysis/figures/perrin_per_boundary.png`](analysis/figures/perrin_per_boundary.png) — Perrin vs Gemini per-boundary catchword counts
- [`analysis/figures/permutation_recurring.png`](analysis/figures/permutation_recurring.png) — Syriac null distribution with true=137 in the right tail (p = 0.007)
- [`analysis/figures/q_source/q_variant_z_scores.png`](analysis/figures/q_source/q_variant_z_scores.png) — Q-source variant robustness (Aramaic ≈ Syriac, not distinctively leading)
- [`analysis/figures/phase2_all_methods.png`](analysis/figures/phase2_all_methods.png) — Phase 2 method comparison
- [`analysis/figures/phase3_baseline.png`](analysis/figures/phase3_baseline.png) — consecutive-vs-random per Syriac author
- [`analysis/figures/roundtrip_recovery_ratio.png`](analysis/figures/roundtrip_recovery_ratio.png) — round-trip ceiling per corpus and method

## Outstanding work

1. **Tighter Perrin pair comparison**: re-run the canonical / Perrin-specific split using all 10 Gemini variants and SEDRA root-level matching. Current 22% canonical is a lower bound (skeleton-only equality, variant 0 only).
2. **Bonferroni-corrected joint test** across the variant × statistic grid for the cross-lingual permutation result.
3. **Direct test of the Casey/Chilton mistranslation-retrojection version of Q's Aramaic-substrate claim** — different methodology from this catchword test; the null result here does not address that argument.

## References

- Perrin, N. (2002). *Thomas and Tatian: The Relationship between the Gospel of Thomas and the Diatessaron*. Academia Biblica 5. SBL/Brill.
- Perrin, N. (2006). "Thomas: The Fifth Gospel?" *Journal of the Evangelical Theological Society* 49/1: 67–80.
- Williams, P. J. (2009). "Alleged Syriac Catchwords in the Gospel of Thomas." *Vigiliae Christianae* 63: 71–82.
- Shedinger, R. F. (2003). Review of Perrin (2002). *Journal of Biblical Literature* 122/2.
- Casey, M. (2002). *An Aramaic Approach to Q*. Cambridge University Press.
- Chilton, B. (2010). "Aramaic and Targumic Antecedents of Pauline 'Justification'." *NovT* 52/3.

External corpora: [Digital Syriac Corpus](https://syriaccorpus.org/) · [Coptic SCRIPTORIUM](https://copticscriptorium.org/) · [SEDRA](https://sedra.bethmardutho.org/) · [fhardison/peshitta-tools](https://github.com/fhardison/peshitta-tools) · [SBLGNT](https://sblgnt.com/) · [IQP / Critical Edition of Q](https://en.wikipedia.org/wiki/Q_source).
