# Gospel of Thomas Catchword Analysis

A computational test of Nicholas Perrin's claim ([*Thomas and Tatian*, 2002](https://www.sbl-site.org/publications/Books_AcademiaBiblica.aspx); [*JETS* 49/1, 2006](data/raw/perrin_catchwords/perrin_2006_jets.pdf)) that the Gospel of Thomas was originally composed in Syriac, indexed by the density of catchwords linking adjacent logia. We replace Perrin's manual retroversion with automated, unbiased Coptic→Syriac translation (Monte Carlo over an EM lexical map, beam-search NMT, frontier-LLM API into Syriac/Hebrew/Arabic/Greek) and apply a uniform catchword detector across every language — directly addressing P. J. Williams' ([2009](https://brill.com/view/journals/vc/63/1/article-p71_3.xml)) bias critique.

## Headline finding (updated 2026-05-12)

> **Perrin found something real, but Williams' bias critique still bites.**
>
> Under apples-to-apples surface-form tokenization (correcting a SEDRA-lemmatization asymmetry that was biasing the Syriac arm), **Thomas Syriac LEADS the cross-language phonological-arrangement test** at z = 3.39, p = 0.0005 — ahead of Greek (3.27), Hebrew (2.91), Arabic (2.62). This supports Perrin's directional prediction that Syriac sound-play is concentrated at adjacent logia.
>
> Perrin's 990 specifically cited Syriac word pairs are **3.48× more phonologically similar than random Syriac word pairs** (p < 1e-9), validating his selection process as non-random.
>
> However, **Williams' pair-by-pair critique holds firm**: of Perrin's 558 adjacent-boundary Syriac catchwords, only **25.1% are canonical** (match the unbiased frontier-LLM Gemini retroversion at the same boundary, pooled over 10 variants with SEDRA root-level matching). **74.9% are Perrin-specific** — Syriac word choices that exceed what unbiased retroversion produces. Many of Perrin's 502 are inflated by his translation-side choices.
>
> **Reconciled picture:** Perrin observed real Syriac sound-design at adjacent boundaries (this work confirms it directionally), AND he over-claimed the count via selection bias.

Combined with the **10-variant cross-lingual permutation test**: all 4 target languages plus Coptic source show significant phon-arrangement at adjacent boundaries. Surface-form Syriac (median z = 2.79) and Greek (median z = 2.45) form Tier 1 (Mann-Whitney p = 0.31, indistinguishable, 10/10 variants p<0.05 each); Arabic (z = 1.97, 7/10) and Hebrew (z = 1.63, 3/10) form Tier 2.

See [FINDINGS.md](FINDINGS.md) — the single authoritative project record (1,300 lines, contains every experiment, every bug fix, every "how to cite" note). Sub-doc files (`PERRIN_DIRECT_FINDINGS.md`, `PROVERBS_FINDINGS.md`, `PHON_ONLY_FINDINGS.md`, `CODEBASE_REVIEW.md`) are retained for traceability but superseded by `FINDINGS.md`.

## Phases

| Phase | Question | Method | Result |
|---|---|---|---|
| **1** | Can random Coptic→Syriac translation reproduce Perrin's 502 catchwords? | Monte Carlo over EM lexical map (10k samples) | mean **195** (CI 175–216), P(≥502) = 0 |
| **2A** | Does adding Syriac fluency get us closer? | Beam search with bigram LM | best **320** at λ=0.3 |
| **2C** | Per-logion stochastic sampling at maximum fluency | 200 sims at λ=1.0 | mean **324** (CI 303–347), P(≥502) = 0 |
| **2B** | Does a frontier LLM (no catchword agenda) reproduce Perrin's specific pairs? | Gemini-3-Flash-Preview, 1,250 calls, with controls | **Yes** for specific pairs (8/8); aggregate count NOT Thomas-specific (controls 12.15/pair > Thomas 8.53/pair, p ≈ 0.99) |
| **Permutation (Syriac)** | Is the *ordering* of logia non-random? | 10,000 shuffles of Syriac LLM retroversion | **p = 0.007** at ≥2-recurrence; robust across 10 LLM variants (median p ≈ 0.014) |
| **Cross-lingual permutation** | Is the recurring-catchword effect Syriac-specific or universal? | Same test on Gemini Hebrew/Arabic/Greek retroversions; 10-variant Mann-Whitney | All 4 languages significant on the **all-catchwords** test. Surface-Syriac and Greek are statistically indistinguishable Tier 1 (M-W p=0.31), Hebrew/Arabic Tier 2. |
| **Perrin table (full)** | Of Perrin's 502 specific Syriac catchwords, how many also appear at the same boundary in unbiased Gemini retroversion? | Digitization of book pp. 58–153 + skeleton/SEDRA-root match (10 variants pooled) | **25.1% canonical / 74.9% Perrin-specific** (tightened 2026-05-12 from initial 22.2% lower bound); 53/107 boundaries with 0 canonical |
| **Q-source extension** | Does the same effect appear in Q? Does Aramaic distinctively lead? | Same pipeline on 56 IQP Q pericopes + 10 controls | Weaker signal than Thomas; only Syriac p<0.05 at variant 0; **Aramaic does NOT distinctively lead** (tied with Syriac, M-W p = 0.26) |
| **Proverbs 10-29 positive control** | Does the pipeline detect arrangement in a *documented* catchword-arranged text (Hildebrandt, Heim, Snell)? | Same pipeline, 595 Hebrew verses + Gemini retroversion to 4 langs | **Yes** — all 5 langs significant; Hebrew (source) z=1.76 main / z=3.60 with vanilla Lev (2nd of 5, top-tier); Syriac strongest target (z=3.27 main, 4.38 variant median). Validates the methodology. |
| **Direct Perrin verification** | Six targeted tests of Perrin's Syriac-paronomasia claim, post-SEDRA correction | Total-count perm test, vanilla Lev, per-boundary MAX, Coptic source, Perrin's 990 pairs vs random, threshold sweep | Under fair surface-form tokenization: **Thomas Syriac z_phon = 3.39 (p = 0.0005) — LEADS Thomas** (Greek 3.27, Hebrew 2.91, Arabic 2.62). Perrin's specific pairs are 3.48× more phonologically similar than random (p<1e-9). |
| **3.0** | Is catchword arrangement detectable in known Syriac literature? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **Yes** — pooled p < 1e-9, Cohen's d = 0.54 |
| **3.1** | Can a model learn to discriminate consecutive Syriac strophes? | Hard-negative contrastive (InfoNCE) | val_acc **0.582** (vs 0.50 chance); 178M-param mBERT fine-tune does worse (0.528) |
| **3.2** | Does that model see the same pattern in beam-translated Thomas? | Permutation test on cos_sim | p = 0.087 (marginal) |
| **Round-trip** | Empirical ceiling of lexical-map recovery on known-catchword text? | Syr → Cop → Syr round-trip on Ephrem/Narsai/Jacob/Odes | MAP ≤ 1.23×; MC ≤ 1.09×. Perrin's 1.87× is **unreachable by any lexical-map method**. |
| **Codebase audit + tests** | Is the pipeline trustworthy? | 201-test pytest suite + audit of every script | **YES.** Caught the SEDRA tokenization asymmetry, a Greek PUNCT_RE bug, and a `filter_pct=0` footgun. All fixed/documented. Test suite runs in ~28s. |

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
# Use the dedicated conda env (Python 3.11 + torch 2.5.1+cu121).
# See FINDINGS.md "Project environment" for full details.
conda activate thomas   # /home/sogang/mnt/db_2/anaconda3/envs/thomas

# Or for a fresh setup:
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

For any Gemini API step (Phase 2B, cross-lingual retroversion, Q-source fetch/translate, Proverbs translation), put `GEMINI_API_KEY=...` in `.env.local` (gitignored).

## Tests

```bash
python -m pytest tests/ phase1_montecarlo/tests/
```

201 tests passing, 3 skipped (stochastic Perrin-pair variants), runtime ~28 seconds. No GPU, no external network required. The suite catches: tokenization parity across the Proverbs / Q / Thomas scripts, the documented SEDRA asymmetry, all `filter_pct` edge cases, permutation reproducibility, and end-to-end synthetic planted-truth tests for each pipeline. See `CODEBASE_REVIEW.md` (merged into FINDINGS.md).

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
# Tightened version (2026-05-12): all 10 variants pooled + SEDRA root-level match
python scripts/perrin_pair_comparison_tight.py
# Build the SEDRA lemma→root table that the tightened version needs
python scripts/build_sedra_root_map.py

# Q-source extension
python scripts/q_fetch_greek.py
python scripts/q_translate.py
python scripts/q_aggregate_density.py
python scripts/q_permutation_test.py
python scripts/q_variant_robustness.py

# Proverbs 10-29 positive control
python scripts/proverbs_fetch_hebrew.py
python scripts/proverbs_fetch_controls.py
python scripts/proverbs_translate.py
python scripts/proverbs_aggregate_density.py
python scripts/proverbs_permutation_test.py
python scripts/proverbs_variant_robustness.py
# Or parallelised:
python scripts/proverbs_one_main.py --lang hebrew
python scripts/proverbs_one_variant.py --lang syriac --variant 0

# Phon-only re-test (historical first-pass; superseded by direct-Perrin below)
python scripts/phon_only_one.py --corpus thomas --lang syriac --variant 0
python analysis/plot_phon_only.py

# Direct Perrin verification (six tests + SEDRA correction, 2026-05-11)
python scripts/perrin_test_one.py --corpus thomas --lang syriac --variant 0
python scripts/perrin_test_vanilla.py --corpus thomas --lang syriac --variant 0
python scripts/perrin_test_syriac_surface.py     # apples-to-apples Syriac (no SEDRA lemmatize)
python scripts/perrin_test_syriac_with_roots.py  # SEDRA root-equivalence enabled
python scripts/perrin_boundary_max.py
python scripts/perrin_boundary_max_surface.py    # corrected boundary-MAX
python scripts/perrin_test_coptic.py             # the actual extant Coptic source
python scripts/perrin_pair_benchmark.py          # Perrin 990 pairs vs random Syriac

# Cross-lingual permutation re-run with surface-form Syriac (the SEDRA-bug correction)
python scripts/crossling_syriac_surface.py --variants 0,1,2,3,4,5,6,7,8,9 --n-perms 10000

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
python analysis/plot_proverbs_results.py
python analysis/plot_phon_only.py
```

## Key figures

- [`analysis/figures/final_summary.png`](analysis/figures/final_summary.png) — paper-ready 2-panel synthesis
- [`analysis/figures/crossling_variant_z_scores.png`](analysis/figures/crossling_variant_z_scores.png) — Tier 1 (Syriac ≈ Greek) vs Tier 2 (Hebrew, Arabic) across 10 LLM variants per language
- [`analysis/figures/crossling_permutation.png`](analysis/figures/crossling_permutation.png) — 4-panel null-distribution histograms at variant 0
- [`analysis/figures/perrin_canonical_split.png`](analysis/figures/perrin_canonical_split.png) — per-boundary canonical / Perrin-specific split
- [`analysis/figures/perrin_per_boundary.png`](analysis/figures/perrin_per_boundary.png) — Perrin vs Gemini per-boundary catchword counts
- [`analysis/figures/perrin_pair_benchmark.png`](analysis/figures/perrin_pair_benchmark.png) — Perrin's 990 pairs vs random Syriac (3.48× phon enrichment)
- [`analysis/figures/permutation_recurring.png`](analysis/figures/permutation_recurring.png) — Syriac null distribution with true=137 in the right tail (p = 0.007)
- [`analysis/figures/q_source/q_variant_z_scores.png`](analysis/figures/q_source/q_variant_z_scores.png) — Q-source variant robustness (Aramaic ≈ Syriac, not distinctively leading)
- [`analysis/figures/proverbs/three_corpus_comparison.png`](analysis/figures/proverbs/three_corpus_comparison.png) — Proverbs vs Thomas vs Q side-by-side
- [`analysis/figures/proverbs/proverbs_variant_z_scores.png`](analysis/figures/proverbs/proverbs_variant_z_scores.png) — Proverbs positive-control variant robustness
- [`analysis/figures/phon_only_comparison.png`](analysis/figures/phon_only_comparison.png) — all-catchwords vs phon-only z-scores per language per corpus
- [`analysis/figures/phase2_all_methods.png`](analysis/figures/phase2_all_methods.png) — Phase 2 method comparison
- [`analysis/figures/phase3_baseline.png`](analysis/figures/phase3_baseline.png) — consecutive-vs-random per Syriac author
- [`analysis/figures/roundtrip_recovery_ratio.png`](analysis/figures/roundtrip_recovery_ratio.png) — round-trip ceiling per corpus and method

## Outstanding work

All three previously-listed outstanding items have been addressed (see [FINDINGS.md § Outstanding work](FINDINGS.md#outstanding-work-addressed-2026-05-12)):

1. ~~**Tighter Perrin pair comparison**~~ — done 2026-05-12. With all 10 Gemini variants pooled + SEDRA root-level matching: canonical fraction rises from 22.2% to 25.1%. Williams' 74.9% Perrin-specific holds firm.
2. ~~**SEDRA root-enabled detector for Thomas Syriac**~~ — done 2026-05-12. Almost no change to z-scores (catches 5 extra etymological pairs out of ~1500 catchwords). Skeleton-Lev was already catching most root-equivalent pairs.
3. ~~**10k-perm cross-lingual surface-Syriac sweep**~~ — done 2026-05-12. Confirms 1k-perm result. Median z = 2.79, all 10/10 variants p<0.05.

**Still open:**
- **Full phonetic-feature phon detector** — consonant features (place, manner, voicing, emphatic) rather than skeleton-Lev. Would catch sound-pairs like b↔v or p↔f voicing pairs that the current detector misses. Out of scope for this round.
- **Direct test of the Casey/Chilton mistranslation-retrojection version of Q's Aramaic-substrate claim** — different methodology from this catchword test; the null result here does not address that argument.

## References

- Perrin, N. (2002). *Thomas and Tatian: The Relationship between the Gospel of Thomas and the Diatessaron*. Academia Biblica 5. SBL/Brill.
- Perrin, N. (2006). "Thomas: The Fifth Gospel?" *Journal of the Evangelical Theological Society* 49/1: 67–80.
- Williams, P. J. (2009). "Alleged Syriac Catchwords in the Gospel of Thomas." *Vigiliae Christianae* 63: 71–82.
- Shedinger, R. F. (2003). Review of Perrin (2002). *Journal of Biblical Literature* 122/2.
- Casey, M. (2002). *An Aramaic Approach to Q*. Cambridge University Press.
- Chilton, B. (2010). "Aramaic and Targumic Antecedents of Pauline 'Justification'." *NovT* 52/3.

External corpora: [Digital Syriac Corpus](https://syriaccorpus.org/) · [Coptic SCRIPTORIUM](https://copticscriptorium.org/) · [SEDRA](https://sedra.bethmardutho.org/) · [fhardison/peshitta-tools](https://github.com/fhardison/peshitta-tools) · [SBLGNT](https://sblgnt.com/) · [IQP / Critical Edition of Q](https://en.wikipedia.org/wiki/Q_source).
