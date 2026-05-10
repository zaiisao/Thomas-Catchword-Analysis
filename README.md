# Gospel of Thomas Catchword Analysis

A computational test of Nicholas Perrin's claim ([*Thomas and Tatian*, 2002](https://www.sbl-site.org/publications/Books_AcademiaBiblica.aspx); [*JETS* 49/1, 2006](data/raw/perrin_catchwords/perrin_2006_jets.pdf)) that the Gospel of Thomas was originally composed in Syriac, indexed by the density of catchwords linking adjacent logia. We replace Perrin's manual retroversion with automated, unbiased Coptic→Syriac translation (Monte Carlo over an EM lexical map, beam-search NMT, frontier-LLM API) and apply a uniform catchword detector across all three languages — directly addressing P. J. Williams' ([2009](https://brill.com/view/journals/vc/63/1/article-p71_3.xml)) bias critique.

## Headline finding

> **The actual ordering of Thomas's 115 logia produces significantly more recurring catchword pairs than 10,000 random shuffles of the same translations.** True order: 137 distinct pairs recurring at ≥2 boundaries vs null mean 119.7. **p = 0.0070** (single 10k-perm test); robust across 10 independent LLM-translation variants (median p ≈ 0.014). Perrin's specifically cited boundaries are reproduced 8/8 by an unbiased frontier LLM.

Aggregate catchword *count* turns out to be uninformative — any Coptic text translated by a stylistically consistent LLM produces high catchword density (Phase 2B controls produce 12.15/pair vs Thomas's 8.53/pair, p≈0.99). What survives the bias critique is the **non-randomness of the ordering**: the recurring-pair structure that disappears under shuffling but is preserved under the actual Thomas sequence.

See [FINDINGS.md](FINDINGS.md) for the full results, [WRITEUP.md](WRITEUP.md) for the long-form writeup as of the Phase 1–3 main analysis, and [PHASE1_FINDINGS.md](PHASE1_FINDINGS.md) for the Phase 1 standalone Monte Carlo report.

## Phases

| Phase | Question | Method | Result |
|---|---|---|---|
| **1** | Can random Coptic→Syriac translation reproduce Perrin's 502 catchwords? | Monte Carlo over EM lexical map (10k samples) | mean **195** (CI 175–216), P(≥502) = 0 |
| **2A** | Does adding Syriac fluency get us closer? | Beam search with bigram LM | best **320** at λ=0.3 |
| **2C** | Per-logion stochastic sampling at maximum fluency | 200 sims at λ=1.0 | mean **324** (CI 303–347), P(≥502) = 0 |
| **2B** | Does a frontier LLM (no catchword agenda) reproduce Perrin's specific pairs? | Gemini-3-Flash-Preview, 1,250 calls, with controls | **Yes** for specific pairs (8/8); aggregate count NOT Thomas-specific (controls 12.15/pair > Thomas 8.53/pair) |
| **Permutation** | Is the *ordering* of logia non-random? | 10,000 shuffles of LLM translations | **p = 0.007** at ≥2-recurrence; robust across 10 LLM variants (median p ≈ 0.014) |
| **3.0** | Is catchword arrangement detectable in known Syriac literature? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **Yes** — pooled p < 1e-9, Cohen's d = 0.54 |
| **3.1** | Can a model learn to discriminate consecutive Syriac strophes? | Hard-negative contrastive (InfoNCE) | val_acc **0.582** (vs 0.50 chance) |
| **3.2** | Does that model see the same pattern in beam-translated Thomas? | Permutation test on cos_sim | p = 0.087 (marginal) |
| **Round-trip** | What's the empirical ceiling of lexical-map recovery on known-catchword text? | Syr → Cop → Syr round-trip on Ephrem/Narsai/Jacob/Odes | MAP ≤ 1.23×; MC ≤ 1.09×. Perrin's 1.87× is **unreachable by any lexical-map method**. |

## Layout

```
phase1_montecarlo/        # uniform catchword detector, Monte Carlo over EM lexical map
phase2_neural_translation/  # NMT model + tokenizer (parallel-NT-trained Coptic→Syriac)
phase3_contrastive/       # contrastive encoder for adjacent-strophe discrimination
analysis/                 # cross-phase comparison, plots, statistical tests
analysis/figures/         # paper-ready PNG/PDF figures (~2 MB total)
configs/config.yaml       # single source of truth — thresholds, paths, calibration targets
scripts/                  # ingestion, training, translation, permutation tests
data/raw/perrin_catchwords/  # JETS 2006 PDF + extracted catchword examples (small, tracked)
data/raw/...              # large external corpora — gitignored, regenerable via scripts/fetch_data.sh
data/processed/           # built artifacts (lexical map, translations, checkpoints) — gitignored
```

The intermediate computational artifacts (lexical-map JSONLs, Phase 2/3 model checkpoints, processed corpora) are not committed — see [DATA_STATUS.md](DATA_STATUS.md) and `.gitignore` for what regenerates from where.

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

For Phase 2B (Gemini API) and Phase 2B-blind-verify, set `GEMINI_API_KEY` in `.env.local` (gitignored).

## Reproduction (post-Phase-0)

```bash
# Phase 1 — Monte Carlo
python scripts/run_monte_carlo.py

# Phase 2 — three independent translation methods at the same calibration
python scripts/phase2a_beam_translate.py
python scripts/phase2c_constrained_sample.py
python scripts/phase2b_gemini_translate.py     # 1250 API calls, ~30 min, ~$0.10

# Permutation test on recurring catchword patterns (the headline finding)
python scripts/permutation_test_recurring.py

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
```

## Key figures

- [`analysis/figures/final_summary.png`](analysis/figures/final_summary.png) — paper-ready 2-panel synthesis
- [`analysis/figures/permutation_recurring.png`](analysis/figures/permutation_recurring.png) — null distribution with true=137 in the right tail (p = 0.007)
- [`analysis/figures/permutation_variant_robustness.png`](analysis/figures/permutation_variant_robustness.png) — p-values across all 10 LLM variants
- [`analysis/figures/phase2_all_methods.png`](analysis/figures/phase2_all_methods.png) — Phase 2 method comparison
- [`analysis/figures/phase3_baseline.png`](analysis/figures/phase3_baseline.png) — consecutive-vs-random per Syriac author
- [`analysis/figures/roundtrip_recovery_ratio.png`](analysis/figures/roundtrip_recovery_ratio.png) — round-trip ceiling per corpus and method

## Outstanding work

1. **Manual entry of Perrin's full 502-pair table** from *Thomas and Tatian* (2002), pp. 57–155. The pair-by-pair canonical-translation check (against the EM map and Gemini translations) is the experiment that would definitively close the bias question for *all* of Perrin's pairs, not just the JETS 2006 examples.
2. **mBERT-finetune Phase 3.1** when GPU memory frees up — `scripts/phase3_improved_contrastive.py`. May shift Thomas's p=0.087 to significance.
3. **Bonferroni-corrected joint test** across the 10 LLM variants × 3 statistics for the permutation result.

## References

- Perrin, N. (2002). *Thomas and Tatian: The Relationship between the Gospel of Thomas and the Diatessaron*. Academia Biblica 5. SBL/Brill.
- Perrin, N. (2006). "Thomas: The Fifth Gospel?" *Journal of the Evangelical Theological Society* 49/1: 67–80.
- Williams, P. J. (2009). "Alleged Syriac Catchwords in the Gospel of Thomas." *Vigiliae Christianae* 63: 71–82.
- Shedinger, R. F. (2003). Review of Perrin (2002). *Journal of Biblical Literature* 122/2.

External corpora: [Digital Syriac Corpus](https://syriaccorpus.org/) · [Coptic SCRIPTORIUM](https://copticscriptorium.org/) · [SEDRA](https://sedra.bethmardutho.org/) · [fhardison/peshitta-tools](https://github.com/fhardison/peshitta-tools).
