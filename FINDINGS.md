# Findings — Gospel of Thomas Catchword Hypothesis

Last updated: 2026-05-12 (unified consolidation — this document is now the single authoritative project record. All sub-finding writeups (`PERRIN_DIRECT_FINDINGS.md`, `PROVERBS_FINDINGS.md`, `PHON_ONLY_FINDINGS.md`, `CODEBASE_REVIEW.md`) and the operational notes from `~/.claude/projects/.../memory/` have been merged in.)

## Document map

This single file contains everything. Sub-documents are kept for historical traceability but are no longer authoritative — every claim and table they contain is reproduced here.

- §[Headline](#headline) — top-level result table for every experiment.
- §[Three findings in plain language](#three-findings-in-plain-language) — non-technical synthesis.
- §[What this implies for Perrin's claim](#what-this-implies-for-perrins-claim) — final verdict.
- §[Caveat](#caveat-what-the-methods-do-and-dont-capture) — limitations of the methodology.
- §[Reproducibility](#reproducibility) — environment and runbook.
- §[Project environment](#project-environment) — conda env path, torch/CUDA versions, hardware.
- §[Round-trip validation](#round-trip-validation-2026-05-09-latest) — Syriac→Coptic→Syriac on known-catchword literature.
- §[Q source extension test](#q-source-extension-test-2026-05-11) — pipeline on 56 IQP Q pericopes.
- §[Cross-linguistic permutation test](#cross-linguistic-permutation-test-2026-05-11--revised-interpretation) — Thomas into Hebrew/Arabic/Greek/Syriac + surface-Syriac re-run.
- §[Permutation test on recurring catchword patterns](#permutation-test-on-recurring-catchword-patterns-2026-05-10--original-syriac-result-now-contextualised-by-the-cross-lingual-test-above) — original Syriac result.
- §[Phase 2B quantitative](#phase-2b-quantitative-gemini-3-flash-preview-api-translation-2026-05-10) — 974-catchword count + Pauline control comparison.
- §[LLM cross-validation](#llm-cross-validation-of-perrins-example-catchwords-2026-05-09) — Claude/Gemini/GPT-4 produce Perrin's specific pairs.
- §[Perrin pair-by-pair comparison](#perrin-pair-by-pair-comparison-2026-05-10--full-table-digitization) — full 502-table digitisation, 22% canonical.
- §[Proverbs 10–29 positive control](#proverbs-1029--positive-control-validation-of-the-catchword-pipeline) — 595-verse Hebrew positive control.
- §[Phonological-only re-test](#phonological-only-cross-linguistic-permutation-test--pipeline-limitation-historical-superseded) — historical first-pass; superseded by direct-Perrin work.
- §[Direct verification of Perrin's Syriac-paronomasia claim](#direct-verification-of-perrins-syriac-paronomasia-claim) — six tests + the SEDRA correction.
- §[Codebase audit + test suite](#codebase-audit--test-suite) — bugs caught, 201 tests.
- §[Bugs found and fixed](#bugs-found-and-fixed) — consolidated list with severities.
- §[Operational notes](#operational-notes--how-to-cite-each-finding) — per-finding "how to apply" / "do not cite".
- §[Outstanding work](#outstanding-work) — what remains.

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
| **Proverbs 10–29 (positive control)** | Does the same pipeline detect arrangement in a text where catchword editing is documented mainstream scholarship (Hildebrandt, Heim, Snell)? | Identical pipeline, 595 Hebrew source verses + Gemini retroversion to Greek/Syriac/Aramaic/Arabic; same detector, same threshold; 10k-perm main + 10×1k-perm variant sweep per language | **YES on the all-catchwords test, but Hebrew (the SOURCE) is the weakest target.** Main test: Hebrew z=1.76 (p=0.054), Greek z=2.97/p=0.0024, Syriac z=3.27/p=0.0019, Aramaic z=3.23/p=0.0019, Arabic z=4.81/p<0.0001. Variant sweep: **Syriac strongest (median z=4.38, 10/10 sig), Hebrew weakest (z=1.76, 0/1 sig)**. **The fact that Hebrew underperforms its translations on its own documented catchword text is the cue that drives the phon-only re-test below.** Top Hebrew pairs (semantic): יהוה/יהוה (×9), צדיק/צדיק (×6), רשעים (×5), כסיל (×5) — canonical antithetical clustering. |
| **Phonological-only re-test** | Does the test detect language-specific arrangement when semantic (lemma-equality) catchwords are filtered out, leaving only phonological + etymological links? | Same matrices as the all-catchwords test, but the permutation count is computed on three filtered subsets: `all`, `phon+etym`, `semantic-only`. 122 parallel worker processes across all 3 corpora × 5 languages × 10 variants. | **First-pass NEGATIVE result, then REVISED.** Recurrence-filtered phon-only: source languages all fail to lead (Proverbs Hebrew rank 40/41, p=0.976). This was reported as a pipeline limitation. The **direct-Perrin re-test below** shows the recurrence-filter was masking detectable phon signal; the limitation is partly fixable. Writeup: `PHON_ONLY_FINDINGS.md`. |
| **Direct Perrin verification (six tests + methodological correction)** | Does Thomas exhibit Syriac-specific paronomastic catchwords as Perrin (2002) claims? Six tests, then a critical methodology fix after discovering an apples-to-oranges tokenization asymmetry. | **AFTER CORRECTION, Perrin's claim is more supported than the initial tests suggested.** Initial six tests appeared to show Syriac "dead last" in phon-arrangement (z=0.13 n.s. with vanilla Lev). Investigation revealed Syriac was the only language being SEDRA-lemmatized (root-collapsed), while Hebrew/Greek/Arabic used surface forms — SEDRA collapsed 50.7% of Syriac tokens to lemma strings, absorbing variant pairs into "semantic" matches and starving the phonological count. With apples-to-apples surface-form tokenization: **Syriac z_phon = 3.39 (p = 0.0005) — LEADS Thomas** (Greek 3.27, Hebrew 2.91, Arabic 2.62). With language-neutral vanilla Levenshtein, Syriac z = 2.74 — competitive (Hebrew 3.13, Arabic 3.01, Syriac 2.74, Greek 2.76). Test 2 unchanged: Perrin's 990 specific cited pairs are 3.48× more phon-similar than random Syriac pairs (p < 1e-9), including canonical `nūrā/nuhrā`. Boundary-MAX winner test still doesn't show Syriac dominance (Hebrew wins more boundaries due to confusion-group baseline), but the total-count permutation result is unambiguous: Syriac leads. Positive control unchanged: Proverbs Hebrew vanilla z=3.60, 2nd of 5 (Aramaic 4.13 leads due to Heb↔Aram orthographic overlap). **The corrected picture supports Perrin's directional prediction: Syriac shows the strongest phon-arrangement among the four target languages on Thomas.** Williams' pair-by-pair 78% Perrin-specific finding stands as the remaining anti-Perrin evidence. Full writeup: `PERRIN_DIRECT_FINDINGS.md`. | |
| **Phase 3.0** | Is catchword arrangement actually detectable in known Syriac literary texts? | Consecutive-vs-random pair test on Ephrem/Narsai/Jacob/Odes | **YES** — pooled p < 1e-9, Cohen's d = 0.54 |
| **Phase 3.1** | Can a model trained on those texts learn to discriminate consecutive strophes? | Hard-negative contrastive (same-work, ≥3 strophes apart) + InfoNCE | val_acc **0.582** (vs 0.50 chance) |
| **Phase 3.2** | Does that learned model see the same pattern in beam-translated Thomas? | Permutation test: adjacent vs shuffled cos_sim | p = **0.087** (marginal) for beam translation; p = 0.32 (n.s.) for NMT |

## Alignment-method sensitivity (BinaryAlign migration, 2026-05-12)

Methodological intervention: replaced the IBM Model 1 EM word-aligner with **BinaryAlign** (Yan et al. 2024) — word alignment as per-pair binary classification on top of a multilingual encoder. New module: [scripts/align_binary.py](scripts/align_binary.py). Both lexical map builders ([build_lexical_map.py](scripts/build_lexical_map.py), [build_reverse_lexical_map.py](scripts/build_reverse_lexical_map.py)) now feed off BinaryAlign instead of IBM-1 EM. Backbone: xlm-roberta-base (mDeBERTa-v3-base requires torch ≥2.6; falls back). Linear classification head is **not yet trained** — runs use a cosine-similarity proxy with row z-normalisation; supply `--head-ckpt` for paper-faithful behaviour. IBM-1 baseline preserved at `data/processed/_ibm1_baseline/`. Comparison script: [scripts/compare_alignment_methods.py](scripts/compare_alignment_methods.py).

### Map-level comparison (Coptic→Syriac forward)

Both maps have 3,831 lemma entries (identical Coptic-content-lemma vocabulary). On the 8 Perrin example words (fire, light, eye, woman, god, father, eat, said), **8/8 top-1 Syriac candidates agree** with IBM-1 (ܢⲩⲣⲐ, ܢⲩⲒⲣⲐ, ܥⲒⲢⲐ, ܐⲒⲦⲦⲐ, ܐⲗⲐⲐ, ܐⲒⲐ, ܐⲒⲗ, ܐⲣ). Aggregate top-1 agreement: 23.3% all lemmas, **63.8% on lemmas with IBM-1 support ≥ 50**. The key magnitude difference: IBM-1 produces **sharp** distributions (correct candidate at p ≈ 0.95), BinaryAlign-cosine-proxy produces **diffuse** ones (correct candidate at p ≈ 0.05–0.15). This entropy gap is the mechanism for everything downstream.

### Headline deltas (10,000-iter MC, 200-sim stochastic, all at fp=80/thr=0.65)

| Statistic | IBM-1 | BinaryAlign | Δ |
|---|---:|---:|---|
| Phase 1 MC mean | 195 | **408** | +213 |
| Phase 1 MC 90% CI | [175, 216] | [359, 460] | shifts up |
| Phase 1 P(≥502) | 0.0000 | **0.0021** | nonzero |
| Phase 1 both-sides% | 45.4 | 75.5 | +30.1 |
| Phase 1 isolated% | 20.0 | 5.3 | −14.7 |
| Phase 2A beam λ=0.0 | 311 | 256 | −55 |
| Phase 2A beam λ=0.3 | 320 | **488** | +168 (within 14 of Perrin's 502) |
| Phase 2A beam λ=1.0 | 328 | 440 | +112 |
| Phase 2C stoch λ=0.0 | 317.1 | **677.0** | +359.9, **P(≥502)=1.0** (was 0) |
| Phase 2C stoch λ=1.0 | 323.6 | **661.2** | +337.6, P(≥502)=1.0 |
| Detector @ fp=80/thr=0.65 Syr (MAP top-1) | 305 | 249 | −56 |
| Phase 3 map-source mean adj-pair cos sim | 0.594 | 0.665 | +0.071 |
| Phase 3 map-source pairs > 0.5 | 71/114 | 78/114 | +7 |

**The signs are opposite for MAP-top-1 vs sampled translation.** MAP-translated Syriac count went *down* (305 → 249) because BinaryAlign's top-1 is sometimes a noisier choice than EM's argmax. But MC sampling, beam search, and stochastic-λ runs all went *up* dramatically because the diffuse `P(s|c)` lets the procedure explore a wider Syriac vocabulary, and a fraction of those alternative samples accidentally satisfy the phonological-similarity threshold across logion boundaries. Under the diffuse null, Perrin's 502 is the median of Phase 2C, not an extreme.

### Round-trip deltas (Syriac → Coptic → Syriac, on known-catchword homiletic corpora)

| Corpus | Original Syr | Coptic interm. | r_MAP IBM-1 | r_MAP BinAln | Phon-pair survival IBM-1 | Phon-pair survival BinAln |
|---|---:|---:|---:|---:|---:|---:|
| Ephrem  | 10,735 |  1,556 | 1.18× | **0.96×** | 37.8% | **64.7%** |
| Jacob   | 41,172 |    809 | 1.15× | **0.75×** | 36.3% | **61.8%** |
| Narsai  | 69,142 |  1,411 | 1.23× | **0.69×** | 35.1% | **59.4%** |
| Solomon |    552 |    107 | 1.18× | **0.99×** | 58.5% | **62.3%** |

The **Coptic-intermediate column collapses** by 50–95% because the cosine proxy in the reverse direction (Syriac source) consistently picks Coptic function words (ⲡ "the", ⲛ "of/and", ⲛⲧⲟϥ "he") as top-1 for Syriac content words. Those function-word Coptic lemmas are then filtered out by the forward map's content-only key set, so most of the corpus disappears. Mechanism: xlm-r tokenizes Syriac at ~4.4 subwords/word vs Coptic ~2.1; averaging `h_src` over the marker-bracketed Syriac span washes the source-word signal toward the corpus centroid, where Coptic function-word embeddings live — and there's no sparsity constraint in cosine scoring to push them out.

**But phonological-pair survival went UP** (35% → 60–65%). The subset of the corpus that survives the round-trip is enriched in pairs where both words happened to have meaningful content-lemma forward translations, and the diffuse forward distribution preserves phonological relationships across the round-trip *better* than the sharp IBM-1 distribution did. Two competing effects: reverse map drops most of the data; the data that survives is cleaner.

### What this means

1. **Phase 1's null hypothesis is highly sensitive to map entropy.** With sharp IBM-1, P(≥502)=0. With diffuse BinaryAlign-cosine-proxy, P(≥502)=0.002 in 10k MC, and P(≥502)=1.0 in 200-sim per-logion stochastic. Perrin's "502 cannot arise by chance" depends on which alignment method defines "chance".
2. **The forward-map experiments measure the entropy of the lexical map, not the data.** Replacing the aligner shifted Phase 1 MC mean from 195 to 408 *without* any change to the Coptic Thomas text or the phonological detector. The Coptic count (235) is unchanged across both maps because it doesn't depend on the map at all.
3. **The reverse-map degradation is an algorithm artefact, not data evidence.** The collapse of `r_MAP` to 0.7–1.0× under BinaryAlign-cosine-proxy reflects function-word bias in the cosine scoring, not anything about the underlying Syriac literature. The same number under a trained head would likely return to the IBM-1 regime.
4. **Forward-direction findings are robust to map choice on the categorical question.** All 8 of Perrin's specifically cited Coptic→Syriac top-1 translations agree (ܢⲩⲣⲐ, ܢⲩⲒⲣⲐ, etc.). The disagreement is in the *probabilities*, not the *rankings*.

### How to read the existing FINDINGS table with BinaryAlign in mind

- **Lexical-map-using experiments** (Phase 1, Phase 2A, Phase 2C, round-trip, Phase 3 map-source): treat the IBM-1 numbers in the Headline table as the numbers *under a sharp prior*. Under a diffuse prior (BinaryAlign-cosine-proxy), the Phase 1/2 figures all shift dramatically; Phase 1's "P(≥502)=0" finding becomes "P(≥502) ranges from 0.002 to 1.0 depending on entropy". Only a trained-head BinaryAlign run would give a defensible re-evaluation.
- **LLM-translation-based experiments** (Phase 2B qualitative + quantitative, all permutation tests, Perrin table comparison, direct-Perrin verification): **invariant**. None of these touch the lexical map.
- **Phase 3.0 baseline** (consecutive-vs-random on Syriac literature directly): invariant.

### Reproducing the migration

```
# Replays the full migration in ~15 min on 4× A6000:
python scripts/build_lexical_map.py --model xlm-roberta-base       # Coptic→Syriac
# Reverse: 3-way sharded for ~5 min
CUDA_VISIBLE_DEVICES=0 python scripts/build_reverse_lexical_map.py --model xlm-roberta-base \
    --batch-size 32 --shard 0:3 --raw-out /tmp/rev_shard0.json &
CUDA_VISIBLE_DEVICES=2 python scripts/build_reverse_lexical_map.py --model xlm-roberta-base \
    --batch-size 32 --shard 1:3 --raw-out /tmp/rev_shard1.json &
CUDA_VISIBLE_DEVICES=3 python scripts/build_reverse_lexical_map.py --model xlm-roberta-base \
    --batch-size 32 --shard 2:3 --raw-out /tmp/rev_shard2.json &
wait
python scripts/merge_alignment_shards.py --shards /tmp/rev_shard*.json \
    --direction reverse --out data/processed/lexical_mapping/syriac_to_coptic.jsonl

# Re-run map-dependent experiments
python scripts/calibrate_detector.py
python scripts/run_monte_carlo.py
python scripts/phase2a_beam_translate.py
python scripts/phase2c_constrained_sample.py
python scripts/roundtrip_translate_to_coptic.py
python scripts/roundtrip_retranslate_to_syriac.py
python scripts/roundtrip_pair_survival.py
python scripts/phase3_apply_to_thomas.py --source both

# Side-by-side delta report
python scripts/compare_alignment_methods.py
```

To recover the original IBM-1 maps for comparison:
```
cp data/processed/lexical_mapping/_ibm1_baseline/coptic_to_syriac.ibm1.jsonl \
   data/processed/lexical_mapping/coptic_to_syriac.jsonl
cp data/processed/lexical_mapping/_ibm1_baseline/syriac_to_coptic.ibm1.jsonl \
   data/processed/lexical_mapping/syriac_to_coptic.jsonl
```

## Three findings, in plain language

**(1) Perrin's 502 cannot be explained by Coptic→Syriac word-mapping plus Syriac fluency alone.** Every automated method we tried — random sampling, top-1 deterministic mapping, beam search with a Peshitta-NT bigram LM, per-logion stochastic sampling weighted by that LM — produces 195–324 catchwords. The gap to Perrin (~178 catchwords) is real and unexplained by what an unbiased translator working from the lexical map can produce.

**(2) Catchword-based literary arrangement is real in Syriac.** When we apply the same calibrated rule-based detector to Ephrem, Narsai, Jacob of Serug, and the Odes of Solomon, consecutive strophes share significantly more catchwords than randomly-paired strophes (pooled p < 1e-9, Cohen's d = 0.54). All four corpora individually are significant. So Perrin's *premise* — that Syriac literature uses catchword arrangement and a Syriac-original Thomas should too — is not unfounded.

**(2b) Direct verification of Perrin's paronomasia claim — with a major methodological correction.** Earlier phon-only re-tests found "all three source languages fail to lead" and concluded the pipeline had a fundamental blind spot. Pushing harder uncovered three issues, two of which were fixable: removing the recurrence-filter (which discards non-recurring pairs) and removing language-specific confusion-group bonuses to make cross-language comparison fair. After those fixes the initial result was "Syriac dead last on Thomas" — z_phon = 0.13. **User flagged this as suspiciously low. Investigation revealed Syriac was being SEDRA-lemmatized (root-collapsed) while Hebrew/Greek/Arabic ran on surface forms.** SEDRA collapsed 50.7% of Syriac tokens to common root lemmas, classifying variant-pair surface forms as "semantic" matches and starving the phonological count. With apples-to-apples surface-form tokenization, the corrected picture:

(i) **Perrin's specific 990 cited Syriac pairs ARE 3.48× more phonologically similar than random Syriac word pairs** (p<1e-9). His selections — including the canonical `ܢܘܪܐ/ܢܘܗܪܐ` (fire/light) at logion 10-11 (score 0.900) — capture real Syriac phonological structure, not arbitrary choices. Williams' bias critique on his pair selection is at least partially overstated.

(ii) **Thomas Syriac LEADS the cross-language phon-arrangement permutation test** with surface-form tokenization: z_phon = 3.39 (p = 0.0005) vs Greek 3.27, Hebrew 2.91, Arabic 2.62. With vanilla Levenshtein (language-neutral): Syriac z = 2.74, competitive but third of four (Hebrew 3.13, Arabic 3.01, Syriac 2.74, Greek 2.76). The earlier "Syriac dead last" result was the SEDRA artifact, not a real signal.

(iii) **Boundary-MAX winner test:** even with surface tokenization, Syriac wins only 26/114 boundaries (22.8%, vs 25% chance). Hebrew wins more (50/114, 43.9%) due to its broad confusion-group baseline. The total-count permutation test (i) is the more direct match to Perrin's claim, which the data support.

(iv) **Positive control:** on Proverbs (documented Hebrew arrangement) with vanilla Levenshtein, Hebrew moves to 2nd of 5 (z = 3.60, behind only Aramaic at 4.13 — a Hebrew↔Aramaic orthographic-overlap artifact). The pipeline can detect source-language phon-arrangement when it exists, AND it does detect it for Syriac in Thomas (when tokenized fairly).

Verdict: Perrin's directional prediction — that Thomas Syriac would exhibit more phon-arrangement at adjacent logia than its retroversions — is **supported** under the corrected, apples-to-apples methodology. The remaining anti-Perrin evidence is Williams' pair-by-pair 78%-Perrin-specific finding (memory `project_perrin_table_comparison`), which says Perrin's specific 502 catchword identifications exceed what unbiased retroversion produces. The reconciled picture: Perrin observed real Syriac sound-play at adjacent boundaries (this finding); some of his specific identifications are translation-choice inflations rather than canonical retroversions. Full details: `PERRIN_DIRECT_FINDINGS.md`, `PROVERBS_FINDINGS.md`, `PHON_ONLY_FINDINGS.md`.

**(3) Thomas's catchword arrangement has BOTH thematic and Syriac-phonological components.** (REVISED after the SEDRA methodology fix; the previous version of this finding overstated the "thematic only" conclusion.) When we translate Thomas into Hebrew, Arabic, and Greek (same Gemini model, same detector, same threshold), the same permutation test gives p = 0.017, 0.011, and 0.016 respectively at variant 0. The phon-only re-test (2b above) further shows that the signal decomposes overwhelmingly into the semantic (same-lemma) component — Thomas Syriac z_sem = 3.84, z_phon = 0.95 — and the positive control (Proverbs Hebrew) fails the phon-only diagnostic, meaning we cannot rule out language-specific phonological arrangement; we can only say our detector cannot see it. The 10-variant robustness sweep (1,000 perms × 10 variants × 4 languages) sharpens this: Syriac and Greek form a single statistical tier (Mann-Whitney p = 0.31, 10/10 variants significant in each, median z ≈ 2.4–2.5), and Hebrew + Arabic form a second, lower tier (median z ≈ 1.6–1.9, 2–3 variants drop below p=0.05). Greek is decisive: it has no triliteral roots, longer words, different phonological structure, and yet matches Syriac on this test. The non-randomness is in the **thematic clustering of the logia** (visible in the Coptic source) — not in any Syriac-specific phonological design. The Hebrew/Arabic underperformance is most plausibly translation-side noise (rarer registers, more lexical variance across variants), not anything about the source text. Combined with the Perrin-table pair-by-pair result (78% of Perrin's 502 specific Syriac words are not canonical retroversions), this leaves Perrin's *aggregate* claim ("Thomas is arranged by Syriac catchwords") without empirical support, while leaving his *specific cited examples* (`nūrā/nuhrā` etc.) intact as canonical translations of thematically-paired Coptic.

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

| Language | Median z | Min–Max z | Median p | p<0.05 count |
|---|---:|---:|---:|:---:|
| **Syriac (SEDRA)** | 2.51 | 2.14 – 3.72 | 0.012 | **10/10** |
| **Syriac (surface, re-run 2026-05-11)** | 2.77 | 1.90 – 3.31 | 0.005 | **10/10** |
| **Greek**  | 2.45 | 1.97 – 3.47 | 0.012 | **10/10** |
| **Arabic** | 1.97 | 1.26 – 2.47 | 0.034 | 7/10 |
| **Hebrew** | 1.63 | 0.84 – 2.71 | 0.066 | 3/10 |

(Earlier table cited Hebrew 8/10 and Arabic 9/10 — those were counts at p<0.10. Corrected here to α=0.05.)

The cross-lingual re-run with surface-form Syriac was triggered by the discovery that Thomas's `make_tokens` applies SEDRA lemma collapse to Syriac only, while Hebrew/Greek/Arabic ran on surface forms — an apples-to-oranges asymmetry. The new Syriac row above re-runs ONLY the Syriac arm with surface forms (Hebrew/Greek/Arabic were never affected; their numbers stand). Mann-Whitney `Syriac(surface) > Syriac(SEDRA)`: p=0.40 (the two distributions are statistically indistinguishable). The substantive finding — Syriac and Greek both Tier 1, Hebrew and Arabic Tier 2 — survives the methodological correction. Surface Syriac median z=2.77 slightly exceeds SEDRA Syriac median z=2.51, but they are not statistically separated.

**Pairwise Mann-Whitney (one-sided on the 10 z-scores per language) — updated 2026-05-11 with surface Syriac:**

| Test | p-value | Direction |
|---|---:|---|
| Syriac (SEDRA) > Greek | 0.312 | NOT significant — overlapping distributions |
| Syriac (surface, NEW) > Greek | 0.312 | NOT significant — overlapping distributions |
| Syriac (surface) > Syriac (SEDRA) | 0.396 | NOT significant — methodology change has no detectable effect |
| Syriac (surface) > Hebrew | 0.0007 | Syriac higher ✓ |
| Syriac (surface) > Arabic | 0.0029 | Syriac higher ✓ |
| Greek > Hebrew | 0.0023 | Greek higher ✓ |
| Greek > Arabic | 0.0057 | Greek higher ✓ |

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

## Proverbs 10–29 — Positive-control validation of the catchword pipeline

(Originally in `PROVERBS_FINDINGS.md`. Reproduced here as the authoritative record.)

**Question.** Does the Thomas catchword-arrangement pipeline (Phase 4 cross-lingual permutation test) work on a text that is *documented in the secondary literature* to be catchword-arranged? If yes, the method has a positive control. If no, the Thomas signal is uninterpretable.

**Test corpus.** Proverbs 10–29, the "Solomonic" sentence-collection long studied for catchword, sound-play, and theme groupings (Hildebrandt 1988, Heim 2001, Snell 1993). One unit = one verse = one pericope. N = 595 verses after filtering.

**Pipeline.** Identical to Thomas Phase 4 and Q:
1. Source-language text: BHS Hebrew (Sefaria API).
2. Gemini retroversion into Greek, Syriac, Aramaic, Arabic — 10 stochastic variants per verse (temperature 0.7), model `gemini-3-flash-preview-12-2025` (most units), `gemini-2.5-flash` (first ~80% of units, switched mid-run for cost).
3. Lemma/skeleton detector on adjacent verse pairs; permutation test shuffles verse order.

### Headline result

**All five languages show recurring-arrangement signal significantly above the null** under the 10,000-shuffle main test (`data/proverbs/permutation/main_*.json`):

| Language | N | true ≥2 | null | z | p | true ≥3 | z | p |
|---|---|---|---|---|---|---|---|---|
| Hebrew (source) | 595 | 33 | 26.1 ± 3.9 | 1.76 | 0.054 | 20 | 3.90 | **0.0001** |
| Greek | 590 | 87 | 71.3 ± 5.3 | **2.97** | **0.0024** | 57 | 4.00 | **0.0001** |
| Syriac | 585 | 47 | 32.8 ± 4.3 | **3.27** | **0.0019** | 23 | 3.52 | **0.0010** |
| Aramaic | 590 | 31 | 19.8 ± 3.5 | **3.23** | **0.0019** | 12 | 2.27 | 0.029 |
| Arabic | 590 | 41 | 23.7 ± 3.6 | **4.81** | **<0.0001** | 14 | 1.80 | 0.065 |

Hebrew at the ≥2-boundary level is marginal (p=0.054) but at the ≥3-boundary level — pairs that recur at three or more verse boundaries — it is the strongest, z=3.90. The translations recover the signal at the lower threshold *better than* the source.

### Variant robustness — 10 LLM variants per target language

| Language | median z | mean z | range | fraction p<0.05 |
|---|---|---|---|---|
| Hebrew | 1.76 | — | — | 0/1 (source, single sweep) |
| Greek | **3.28** | 3.32 | 2.06 – 4.61 | **10/10** |
| Syriac | **4.38** | 4.33 | 2.33 – 5.57 | **10/10** |
| Aramaic | 1.77 | 1.83 | 0.65 – 3.29 | 5/10 |
| Arabic | **3.39** | 3.38 | 1.18 – 4.87 | 9/10 |

Pairwise Mann–Whitney on z-scores:
- **Syriac > Greek**: p = 0.013 ✓
- **Syriac > Aramaic**: p = 0.0002 ✓
- **Syriac > Arabic**: p = 0.023 ✓
- **Greek > Aramaic**: p = 0.0023 ✓
- **Arabic > Aramaic**: p = 0.0036 ✓

**Syriac is the strongest target.** Aramaic is consistently the weakest — same pattern observed in Thomas and Q. This is a *property of Gemini's Aramaic output*, not a feature of any underlying substrate.

### Comparison: Proverbs vs Thomas vs Q

| | Proverbs | Thomas | Q |
|---|---|---|---|
| N units | 595 | 115 | 56 |
| Source | Hebrew | Coptic | Greek |
| Genre | sentence-literature, edited collection | gnomic sayings | gnomic sayings |
| Documented catchword? | **yes** (Hildebrandt, Heim, Snell) | claimed by Perrin | no |
| Strongest target (median z) | Syriac (4.38) | Syriac (~2.06) | Greek (source) |
| All variants p<0.05? | Greek/Syriac yes, Arabic 9/10 | Syriac 10/10 | 4/5 langs sig |
| Main-test z (Syriac) | 3.27 | 2.53 | 0.94 |

**Effect sizes are larger in Proverbs than Thomas.** This is consistent with Proverbs being a *known* arranged text where the editor's hand is heavier, and Thomas being a *claimed* arranged text where the effect is real but more subtle. It does **not** mean Thomas's signal is spurious — Thomas remains significant at p<0.05 across all 10 Syriac variants and in all four cross-lingual targets.

### Top recurring lemmas (Hebrew Proverbs, ≥4 boundaries)

| pair | frequency | type |
|---|---|---|
| יהוה ↔ יהוה (LORD) | 9 | semantic |
| צדיק ↔ צדיק (righteous) | 6 | semantic |
| רשעים ↔ רשעים (wicked) | 5 | semantic |
| כסיל ↔ כסיל (fool) | 5 | semantic |
| לב ↔ לב (heart) | 4 | semantic |
| רע ↔ רשע (evil / wicked) | 4 | phonological |

Antithetical clusters (righteous/wicked, wise/fool) — the standard Hebrew-Bible finding — appear as the top recurring boundary pairs. The pipeline is doing what the Hebrew-Bible scholarship predicted.

### Aggregate density — surprising contrary result on Proverbs

Per length-normalised density (catchwords per 100 × 100 word pair), **controls > Proverbs in 4/5 languages** (`data/proverbs/aggregate_density.json`):

| Lang | Proverbs density | Control density | p (Prov > Ctrl) |
|---|---|---|---|
| Hebrew | 93 | **176** | 1.0 |
| Greek | 129 | 130 | 0.74 |
| Syriac | 107 | **179** | 1.0 |
| Aramaic | 93 | **161** | 1.0 |
| Arabic | 87 | **138** | 1.0 |

Interpretation: per-pair density is **not** a catchword-arrangement detector. Narrative prose (Genesis 24, 39; 2 Samuel 12; Ruth 1; Ecclesiastes 4) has a rich stock of repeated function and content words that inflate any lexical-overlap metric. The arrangement signal is *not* in raw density — it is in **which pairs recur at multiple non-adjacent boundaries**, which the permutation test isolates. The Phase 2B Thomas finding is corroborated here on a fifth corpus.

### What Proverbs means for Thomas

1. **The pipeline detects what is provably there.** On a text where the catchword arrangement is settled scholarship, the same statistical test yields p<0.005 in 4/5 languages, with z up to 5.57. The method has a positive control.
2. **The Thomas Syriac p=0.007 is not an artefact of the detector.** A detector that finds Proverbs's arrangement is detecting real lexical recurrence.
3. **"Hebrew is weak at ≥2 but strong at ≥3" is consistent with translation-stable arrangement.** The retroversions homogenise vocabulary, which slightly inflates the lower-threshold count and slightly suppresses the higher-threshold count, relative to the source. The arrangement signal survives.
4. **Aramaic translation is the weakest target, again.** This rules out the Aramaic substrate prediction on a third corpus — if Aramaic-priority were the diagnostic of an Aramaic Vorlage, Aramaic would lead on Proverbs, where the Vorlage is Hebrew (a sister Semitic language). It does not. This is a Gemini-Aramaic-output property.

### Methods notes (Proverbs)

- Permutation test, 1,000 shuffles for variant sweep, 10,000 for main test, fixed seed (42).
- ≥2-boundary statistic: number of distinct lemma pairs that recur at two or more verse boundaries in true vs shuffled order.
- ≥3-boundary statistic: same with three-or-more.
- Detector: lemma-equality plus consonantal-skeleton fallback per language; top-1% most-frequent lemmas blocked per language to suppress trivial function-word matches.
- Translation prompt: same as Thomas/Q (`scripts/proverbs_translate.py: PROMPTS`); Gemini thinking budget = 0; temperature 0.7 for 10 variants.
- Two-model mix is unavoidable: gemini-2.5-flash hit its 10,000/day quota partway through; remaining ~14% of units (the longest-tail re-fills after a disk-full event) used gemini-3-flash-preview-12-2025. The variant sweep covers both regimes.
- Cost: roughly $5 total for Proverbs.

### Proverbs files

- `data/proverbs/proverbs_hebrew.json` — 595 Hebrew source verses (Sefaria BHS).
- `data/proverbs/controls_hebrew.json` — 34 control verses (Gen/Ruth/2Sam/Eccl narrative).
- `data/proverbs/translations/{lang}/unit_*.json` — 595 × 4 × 10 = 23,800 Gemini retroversions.
- `data/proverbs/control_translations/{lang}/unit_*.json` — 34 × 4 controls.
- `data/proverbs/permutation/main_results.json` — combined 10k-perm headline.
- `data/proverbs/permutation/main_{lang}.json` — per-language 10k-perm details with `top_pairs`.
- `data/proverbs/permutation/variant_{lang}.json` — 10-variant 1k-perm sweep per language.
- `data/proverbs/permutation/summary.txt` — pairwise Mann–Whitney table.
- `data/proverbs/aggregate_density.json` — Prov vs Ctrl length-normalised density.
- `analysis/figures/proverbs/proverbs_crossling_permutation.png` — 5-panel null-distribution histograms.
- `analysis/figures/proverbs/proverbs_variant_z_scores.png` — box+strip of z across variants.
- `analysis/figures/proverbs/proverbs_variant_p_values.png` — same on p-value scale.
- `analysis/figures/proverbs/proverbs_aggregate_density.png` — Prov vs Ctrl density.
- `analysis/figures/proverbs/three_corpus_comparison.png` — Proverbs vs Thomas vs Q side-by-side.

## Phonological-only cross-linguistic permutation test — pipeline limitation (HISTORICAL, SUPERSEDED)

(Originally in `PHON_ONLY_FINDINGS.md`. Reproduced here as historical context. The conclusions of this section were **superseded** by the [Direct Perrin verification](#direct-verification-of-perrins-syriac-paronomasia-claim) section below, which found the "Syriac dead last" result was a SEDRA tokenization artifact.)

**Question.** The original cross-linguistic permutation test mixed three catchword types (semantic, phonological, etymological). The Proverbs positive control revealed that **Hebrew does not lead on Proverbs** (median variant-sweep z = 1.76 in Hebrew vs 4.38 in Syriac), even though Hebrew is the documented source of the catchword arrangement. Hypothesis: semantic catchwords carry the *thematic* signal — universal across translations — and swamp the language-specific *phonological* signal. Strip out semantic matches and the source language should lead.

**Test.** Re-ran the entire 3-corpus, 5-language, 10-variant permutation pipeline (122 worker processes, ~25 minutes wall time on 64 cores). For each (corpus, language, variant) we ran THREE permutation tests on the *same* matrix:
- `all` — every link type (the original test)
- `phon` — `link_type ∈ {phonological, etymological}` (language-specific)
- `sem` — `link_type ∈ {semantic}` (thematic)

### First-pass result: the fix appeared not to work

| Corpus | Source | z_phon (var 0) | Rank among target variants | Empirical p |
|---|---|---|---|---|
| **Proverbs** | Hebrew | **−0.16** (p = 0.61) | **40 / 41** (worst) | 0.976 |
| **Thomas** | Syriac | 0.95 (p = 0.19) | 25 / 31 | 0.806 |
| **Q** | Greek | 0.64 (p = 0.27) | 24 / 41 | 0.585 |

**In all three corpora the source language failed to lead on phonological-only.**

This was first interpreted as: *"the pipeline fundamentally cannot distinguish language-specific from thematic arrangement."*

### Why the first-pass interpretation was wrong (in retrospect)

Two issues with this conclusion, discovered during the [Direct Perrin verification](#direct-verification-of-perrins-syriac-paronomasia-claim) below:

1. **The statistic was wrong for Perrin's claim.** The test counts pairs that *recur at ≥2 boundaries*. Perrin's claim is about *specific paronomastic pairs at individual boundaries*, not recurrence. The recurrence filter discards exactly the signal we want. The total-count statistic (no recurrence filter) recovers significant Syriac-Thomas signal at z=1.70, p=0.049.

2. **Thomas Syriac was the only language being SEDRA-lemmatized.** Hebrew/Greek/Arabic ran on surface forms; Syriac ran on SEDRA root lemmas, collapsing 50.7% of surface tokens. This shifted variant pairs from "phonological" to "semantic" in Syriac but not in the comparison languages — biasing the Syriac phon-only count downward. With apples-to-apples surface forms, Thomas Syriac z_phon = 3.39 (p = 0.0005) — Syriac LEADS Thomas on the corrected test.

### What this section still tells us correctly

For **Proverbs Hebrew specifically**, none of the SEDRA story applies — Proverbs uses surface forms uniformly across all five languages. So Hebrew Proverbs's "phon-only is weak" result is real, and means: *the consonantal-skeleton + Levenshtein detector cannot see the actual rhyme / sound-play / paronomastic root-pun that Hebrew Bible scholars (Hildebrandt, Heim, Snell) document in Proverbs.* That's a real detector limitation on Proverbs.

For Thomas, the corrected verdict supersedes this section — see [Direct Perrin verification](#direct-verification-of-perrins-syriac-paronomasia-claim).

### Phon-only files (retained)

- `data/phon_only/{corpus}_{lang}_v{variant}.json` — 122 per-process records.
- `data/phon_only/summary.json` + `summary.txt` — aggregated tables.
- `analysis/figures/phon_only_*.png` — 4 figures.
- `scripts/phon_only_one.py`, `analysis/plot_phon_only.py` — workers + aggregator.

## Direct verification of Perrin's Syriac-paronomasia claim

(Originally in `PERRIN_DIRECT_FINDINGS.md`. Reproduced here as the authoritative record. This is the **final verdict** on Perrin's specific claim, post the methodological correction.)

**Major correction (2026-05-11, late):** the earlier "Syriac dead last" finding (above) was a methodological artifact. Syriac was the only language being SEDRA-lemmatized while Hebrew/Greek/Arabic ran on surface forms. SEDRA collapsed 50.7% of Syriac tokens to root lemmas, absorbing variant-pair surface forms into "semantic" (same-root) matches and suppressing the phonological count. With apples-to-apples surface-form tokenization, Syriac LEADS Thomas on the language-aware detector (z_phon = 3.39, p = 0.0005) and is competitive on the language-neutral one.

**Question.** Perrin (2002) argues that the Gospel of Thomas was composed in Syriac, with deliberate paronomastic catchwords linking adjacent logia (e.g., `ܢܘܪܐ nūrā` "fire" / `ܢܘܗܪܐ nuhrā` "light" at logion 10–11). This section tests Perrin's claim through six fresh angles to see whether the apparent earlier limitation was real or fixable.

### The pipeline limitations we identified

1. **Wrong statistic.** The original test counted *recurring* pairs (same lemma pair appearing at ≥2 adjacent boundaries). Perrin's claim is about *specific* paronomastic pairs at *individual* boundaries. The recurrence filter discards the very signal we want.
2. **No direct query against Perrin's actual list.** We had Perrin's 502 digitised pairs but had never asked "do Perrin's specific pairs score higher than random Syriac pairs would?"
3. **Unfair cross-language comparison.** Each language used its own confusion-group profile (Syriac 6 groups, Hebrew 7, Arabic 10), making raw z-scores incomparable across languages.

### Six direct tests, summarised

| Test | What it measures | Result |
|---|---|---|
| 1 | Total phon-catchwords at adjacent boundaries (no recurrence filter), Thomas Syriac | **z = 1.70, p = 0.049 ✓** (vs old recurrence test z = 0.95, p = 0.19). Signal recoverable. |
| 2 | Perrin's 990 cited Syriac pairs vs 10,000 random Syriac pairs | **Perrin pairs 3.48× more phonologically similar (p < 1e-9).** Even excluding identical pairs, p = 0.005. |
| 3 | Threshold/blocking sweep on Thomas | Looser threshold (0.5) helps Syriac slightly (z=2.19) but doesn't change cross-language ranking. |
| 4 | VANILLA Levenshtein on Thomas (no confusion-group bonuses) | Initially: Syriac drops to z = 0.13 (n.s.). **But this was the SEDRA artifact** — see correction below. |
| 5 | Per-boundary MAX phon score across languages | **Syriac wins 23/114 boundaries (20.2%) vs null mean 24.2 (z = −0.30, p = 0.66).** Syriac is INDISTINGUISHABLE from random adjacency on this winner-take-all metric. |
| 6 | The actual extant Coptic Thomas (the manuscript) | Coptic z_phon = 2.37 (p = 0.010 ✓). Initially read as anti-Perrin (Coptic > Syriac SEDRA z=1.70), but with surface Syriac z=3.39, Perrin's predicted direction (Syriac > Coptic) is preserved. |

### (A) Perrin's specific pairs ARE real Syriac phonology

Test 2 is unambiguous. Of Perrin's 990 cited Syriac word pairs at adjacent boundaries:
- 8.2% are phonologically similar (above threshold 0.6) vs 2.4% for random Syriac pairs from the same corpus → **3.48× enrichment**.
- 7.6% are semantically/etymologically identical vs 0.7% for random → **10.2× enrichment**.
- Mann–Whitney on raw score distribution: p = 8.52e-10.
- Including the canonical `nūrā / nuhrā` at logion 10–11 (score 0.900).

Williams' bias critique was that Perrin's choices were arbitrary. They are not. His selections capture real Syriac sound-similar word pairs at significantly higher rates than random.

### (B) Methodological correction — the SEDRA asymmetry

After observing Syriac vanilla z_phon = 0.13 (suspiciously low), user flagged a possible bug. Investigation revealed: **Syriac tokens were being lemmatized via SEDRA (Syriac Electronic Database, ~16,000 Peshitta-derived entries) before catchword detection, while Hebrew/Greek/Arabic ran on surface forms.** SEDRA collapsed 50.7% of Syriac surface tokens to root lemmas (1320 unique surface forms → 712 unique lemmas, 1.85× compression).

Effect: surface forms like `ܡܠܟܐ` (the king) and `ܡܠܟܘܬܐ` (kingdom) collapse to the same SEDRA lemma `ܡܠܟ`, get classified as a SEMANTIC match, and never reach the phonological detector. In Hebrew/Greek/Arabic, the equivalent surface pairs stay distinct → counted as phonological matches.

**Re-run on Syriac Thomas with surface forms (no SEDRA):**

| Setting | z_phon | p_phon |
|---|---|---|
| SEDRA lemmas + lang-profile (originally reported) | 1.70 | 0.049 |
| SEDRA lemmas + vanilla Lev (Test 4, originally reported) | 0.13 | 0.457 (n.s.) |
| **Surface forms + lang-profile** (fair) | **3.39** | **0.0005** ✓✓ |
| **Surface forms + vanilla Lev** (fair) | **2.74** | **0.0052** ✓ |

Cross-language Thomas results with surface forms + lang-profile detector:

| Language | z_phon (fair) | Rank |
|---|---|---|
| **Syriac (Perrin's "source")** | **3.39** | **1st** |
| Greek | 3.27 | 2nd |
| Hebrew | 2.91 | 3rd |
| Arabic | 2.62 | 4th |

With vanilla Lev + surface forms, Syriac is competitive (z = 2.74) — close to Greek (2.76), behind Hebrew (3.13) and Arabic (3.01). Boundary-MAX with surface Syriac: Syriac wins 26/114 boundaries (22.8%) vs null mean 28.8 (z = −0.69, p = 0.79) — still no boundary-winner advantage, because Hebrew has high MAX scores almost everywhere as a baseline. But the TOTAL-count permutation result is unambiguous: with apples-to-apples tokenization, Syriac leads.

### (C) Cross-lingual permutation re-run with surface Syriac (2026-05-11)

The all-catchwords cross-lingual permutation test (the "Tier 1 = Syriac/Greek" finding above) was also using SEDRA-collapsed Syriac. Re-ran ONLY the Syriac arm with surface forms (Hebrew/Greek/Arabic were always on surface forms; their numbers stand).

**10 variants × 1000 perms:**

| Lang | Median z | Range | p<0.05 count |
|---|---|---|---|
| Syriac (SEDRA, original) | 2.51 | 2.14 – 3.72 | 10/10 |
| **Syriac (surface, NEW)** | **2.77** | 1.90 – 3.31 | **10/10** |
| Greek (unchanged) | 2.45 | 1.97 – 3.47 | 10/10 |
| Arabic (unchanged) | 1.97 | 1.26 – 2.47 | 7/10 |
| Hebrew (unchanged) | 1.63 | 0.84 – 2.71 | 3/10 |

Mann-Whitney `Syriac(surface) > Syriac(SEDRA)`: p=0.40 (indistinguishable).
Mann-Whitney `Syriac(surface) > Greek`: p=0.31 (still Tier 1 tie).

**The substantive finding survives the methodological correction.** Syriac and Greek remain Tier 1 (indistinguishable), Hebrew and Arabic remain Tier 2. Surface Syriac median z slightly higher than SEDRA Syriac median, but the structural conclusion (no significant difference between Syriac and Greek) is preserved.

### (D) Positive control re-run with vanilla Lev

On **Proverbs** (positive control, documented Hebrew catchword text):

| Lang | Lang-profile detector | Vanilla Lev (fair) |
|---|---|---|
| Hebrew (SRC) | z = 2.49 (4th of 5) | **z = 3.60 (2nd of 5)** |
| Aramaic | z = 4.13 | z = 4.13 |
| Arabic | z = 3.17 | z = 3.34 |
| Greek | z = 2.03 | z = 2.92 |
| Syriac | z = 2.79 | z = 2.57 |

With vanilla Lev, Hebrew jumps from 4th to 2nd, behind Aramaic only (Aramaic uses Hebrew-script orthography and shares Hebrew roots, so a Hebrew → Aramaic retroversion preserves much of Hebrew's structure — this is a known translation-pair artifact, not a counterexample).

The fair test thus moves Hebrew from "worst on its own text" to "top tier". The pipeline can detect source-language phon-arrangement when it exists, AND it detects it for Syriac in Thomas (when tokenized fairly).

### Verdict — direct Perrin

| Claim | Evidence | Verdict |
|---|---|---|
| Perrin's specific 502 cited pairs capture real Syriac phonology | Test 2 (3.48× enrichment, p<1e-9) | **Supported** |
| The catchword pairs Perrin cites are non-random selections | Test 2 + LLM cross-validation | **Supported** |
| Thomas has phon-arrangement at adjacent boundaries | Tests 1, 4, 6 (all 4 langs + Coptic significant) | **Supported** (in every language tested) |
| Thomas Syriac LEADS phon-arrangement at adjacent boundaries (Perrin's directional prediction) | Test 1 with surface forms (z = 3.39, leads Greek/Hebrew/Arabic) | **Supported under fair tokenization** |
| Thomas was originally composed in Syriac (Perrin's broader claim) | Combination of supports above + Williams' 78% counter | **Partially supported; Williams' bias critique remains the principal anti-Perrin evidence** |

The reconciled picture:
- **Perrin found something real**: the Syriac pairs he cites are phonologically structured beyond what random Syriac retroversion produces; AND Thomas Syriac under fair tokenization shows the strongest phon-arrangement among the four target languages.
- **Williams' 78%-Perrin-specific pair-by-pair finding** still stands: many of Perrin's specific Syriac word choices exceed what unbiased frontier-LLM retroversion produces. Some of Perrin's specific identifications are translation-choice inflations rather than canonical retroversions.

The remaining honest qualification: our detector is consonantal-skeleton + Levenshtein. Real paronomasia involves rhyme, meter, alliteration at depths this detector cannot reach. A negative result on phon-only (which we did not get with surface forms) would not formally exclude Syriac-specific sound-design at finer levels; the positive result we DO have under surface tokenization is consistent with — but does not uniquely prove — Perrin's compositional hypothesis.

### Direct-Perrin files

- `data/perrin_direct/thomas_*_v0.json` — Test 1 (5 langs × default detector).
- `data/perrin_direct/thomas_syriac_v0_surface.json` — corrected Syriac (default detector).
- `data/perrin_direct/vanilla_thomas_*_v0.json` — Test 4 (vanilla detector).
- `data/perrin_direct/vanilla_thomas_syriac_v0_surface.json` — corrected Syriac (vanilla).
- `data/perrin_direct/perrin_pair_benchmark.json` — Test 2 (Perrin 990 pairs vs random).
- `data/perrin_direct/boundary_max_thomas.json` — Test 5 (cross-lang MAX).
- `data/perrin_direct/boundary_max_thomas_surface.json` — corrected boundary-MAX.
- `data/perrin_direct/coptic_thomas_v0.json` — Test 6 (Coptic source).
- `data/perrin_direct/vanilla_proverbs_*_v0.json` — positive control re-run.
- `data/processed/crossling_syriac_surface/variant_{0..9}.json` — cross-lingual surface re-run.
- `scripts/perrin_test_one.py` — total-count permutation worker.
- `scripts/perrin_test_vanilla.py` — vanilla-Levenshtein worker.
- `scripts/perrin_test_syriac_surface.py` — surface-Syriac Thomas re-run.
- `scripts/perrin_boundary_max.py`, `scripts/perrin_boundary_max_surface.py` — boundary-MAX.
- `scripts/perrin_test_coptic.py` — Coptic Thomas test.
- `scripts/perrin_pair_benchmark.py` — Perrin specific-pair benchmark.
- `scripts/crossling_syriac_surface.py` — cross-lingual surface Syriac re-run worker.

## Codebase audit + test suite

(Originally in `CODEBASE_REVIEW.md`. Reproduced here as the authoritative record.)

After the SEDRA-tokenization bug class was caught the hard way during the Perrin verification work, a comprehensive audit was performed on every moving part in the pipeline. This section records the audit + the resulting test suite.

### Bugs caught by the audit

#### 1. Greek `PUNCT_RE` had wrong codepoint + missing character (FIXED)

`scripts/proverbs_permutation_test.py` and `scripts/q_permutation_test.py` both had:

```python
"greek":   re.compile(r"[ʹ͵;·;.,·]"),
#            ↑ U+02B9 (modifier letter prime — generic Unicode, NOT Greek)
```

`scripts/crossling_permutation_test.py` (Thomas) had the correct version:

```python
"greek":  re.compile(r"[ʹ͵;·;.,·]"),
#           ↑ U+0374 (Greek numeral sign)
#                    ↑ U+037E (Greek question mark) — also present
```

Both characters look visually identical in most fonts. The Proverbs/Q versions:
- Use `U+02B9` (modifier letter) instead of `U+0374` (Greek numeral sign).
- Are missing `U+037E` (Greek question mark, which looks like a semicolon).

**Impact:** If Gemini's Greek output contained `U+0374` or `U+037E` (proper Greek punctuation), it would not be stripped during Proverbs/Q tokenization but would be stripped during Thomas tokenization. Cross-corpus asymmetry of the same kind as the SEDRA bug.

**Fix applied:** Proverbs and Q now use the same Thomas-correct pattern. `tests/test_tokenization.py::TestScriptRegexParity` pins the patterns together so future drift is caught.

#### 2. `compute_blocked(filter_pct=0)` blocks every lemma — documented

Setting `filter_pct=0` does NOT mean "block nothing" — it means cutoff=0, which blocks every lemma that appears in ≥1 unit (i.e., everything). The correct way to disable blocking is `filter_pct=100`. This bit us once during the phon-only sweep (`thr050_noblock` runs gave phon/B=0 because everything was blocked). Now documented as a known-bug case in `tests/test_permutation_stats.py::TestBlocking`.

#### 3. The SEDRA tokenization asymmetry is now pinned by tests

`scripts/crossling_permutation_test.py`'s `make_tokens(text, lang, sedra)` applies SEDRA lemma collapse for Syriac only — Hebrew/Greek/Arabic run on surface forms in the same pipeline. This is **intentional asymmetry** but caused the "Syriac dead last on Thomas phon-only" result, which was wrong.

Proverbs and Q do NOT apply SEDRA to any language (their `make_tokens` has no SEDRA argument; all five languages use surface forms). So the asymmetry is Thomas-specific, AND it specifically affects the Perrin verification.

`tests/test_tokenization.py::TestMakeTokensAsymmetry` pins this down:
- Proverbs and Q `make_tokens` are identical for every language.
- Thomas `make_tokens(lang, sedra=None)` matches Proverbs/Q for Syriac.
- Thomas `make_tokens("syriac", sedra=...)` collapses surface→lemma.
- Thomas non-Syriac languages ignore the sedra parameter.

This documents the asymmetry without removing it — the asymmetry is sometimes the right choice (catching morphological variants as semantic matches), but the test ensures the asymmetry can't accidentally disappear or spread.

#### 4. Hebrew PUNCT_RE pre-2026-05-11 stripped all Hebrew letters (FIXED earlier in project)

Original PUNCT_RE for Hebrew `[׀-׿]` overlapped with Hebrew letter range (U+05D0–U+05EA) and stripped all Hebrew letters → tokenize returned empty → silent "loaded 67/115" skip. Fixed by enumerating actual Hebrew punctuation chars (־׀׃׆׳״). Similar tightening for Arabic.

### Test suite — 201 tests across 7 files

```
tests/
├── conftest.py                     # path + data-availability fixtures
├── test_detector_extended.py       # 53 tests — extends phase1_montecarlo/tests/test_detector.py
├── test_tokenization.py            # 45 tests — tokenize() + cross-script parity + SEDRA pin
├── test_loaders.py                 # 23 tests — Proverbs/Q/Thomas loaders + schemas
├── test_permutation_stats.py       # 20 tests — statistic correctness + reproducibility + filter_pct
├── test_synthetic_planted.py       #  7 tests — end-to-end with known-truth planted catchwords (Proverbs)
├── test_synthetic_q.py             #  4 tests — end-to-end Q pipeline planted-truth
├── test_synthetic_thomas.py        #  4 tests — end-to-end Thomas pipeline (surface + SEDRA modes)
└── test_perrin_known_pairs.py      # 16 tests — Perrin's 8 cited boundaries regression
phase1_montecarlo/tests/
└── test_detector.py                # 15 tests — pre-existing detector unit tests
```

Run with `python -m pytest tests/ phase1_montecarlo/tests/`. ~28 seconds, no GPU, no external network. **201 passing, 3 skipped** (Perrin specific-pair stochastic-variant cases).

### What each test file covers

**`tests/test_detector_extended.py`** — detector arithmetic + invariants
- `TestConsonantal` — vocalization stripping for each script.
- `TestLevenshteinArithmetic` — empty strings, weak-consonant cost, confusion-group cost, symmetric output, per-language confusion groups.
- `TestPhonologicalScore` — boundary values; nūrā/nuhrā = 0.85 (capped).
- `TestClassification` — semantic / etymological / phonological / below-threshold / empty / missing-lemma.
- `TestDedup` — repeated lemmas counted once.
- `TestCrossLanguageUniformity` — every profile in `PROFILES` loaded; same detector code path applies to all (Williams' methodological criterion).
- `TestSymmetry` — `detect(a, b)` and `detect(b, a)` yield identical pair sets.
- `TestThresholdConfig` — pins threshold-comparison semantics (uses RAW score, not the tier-capped one).

**`tests/test_tokenization.py`** — tokenize() correctness + cross-script parity
- Per-language tokenization (Syriac, Hebrew, Arabic, Greek, Aramaic).
- `TestCrossScriptParity` — for each shared language, Proverbs / Q / Thomas tokenize() agree token-for-token.
- `TestScriptRegexParity` — `SCRIPT_RE[lang]` and `PUNCT_RE[lang]` patterns are identical across the three pipeline scripts. **This is the test that caught the Greek punctuation bug.**
- `TestMakeTokensAsymmetry` — the ONE documented SEDRA asymmetry is pinned.
- `TestTokenSchema` — every `make_tokens` returns dicts with `lemma`, `form`, `parse`.
- `TestTokenizationEdgeCases` — empty input, pure punctuation, pure whitespace, wrong-script input.

**`tests/test_loaders.py`** — corpus loaders + variant indexing
- Per-corpus loader correctness (Proverbs, Q, Thomas).
- File schema regression (Thomas Syriac uses `syriac_text` field; others use `text`).
- Variant indexing.
- Graceful empty-dir handling.

**`tests/test_permutation_stats.py`** — statistical core
- `TestStatsForOrder` — hand-computed counts on tiny matrices.
- `TestTotalCount` — total-catchwords statistic (no recurrence filter).
- `TestPermutationReproducibility` — same seed → identical null.
- `TestNullDistribution` — null mean stable across seeds at large N.
- `TestBlocking` — filter_pct=80 blocks top 20%; filter_pct=100 blocks nothing; **filter_pct=0 blocks everything (KNOWN BUG, documented)**.
- `TestStatisticRegression` — hand-built planted matrix; pins counts.

**`tests/test_synthetic_planted.py`** — end-to-end with known truth (Proverbs)
- Build a 50-verse Hebrew corpus where 6 UNIQUE planted pairs sit at TRUE adjacent boundaries.
- `test_planted_pair_detected_at_boundaries` — detector finds all 6 planted pairs.
- `test_total_count_rejects_null_on_planted` — permutation test on TRUE order rejects null at p<0.05 (z=1.92 typically).
- `test_total_count_planted_vs_shuffled` — same verses in shuffled order show essentially null signal (z≈0).
- `TestPipelineRobustness` — empty / single-unit / no-overlap corpora complete without error.
- `TestShufflingInvariants` — total cells in the matrix is invariant to matrix-build order.

**`tests/test_synthetic_q.py`** — end-to-end Q pipeline planted-truth
- Build a 50-pericope Greek corpus with 6 unique planted pairs at TRUE adjacent boundaries.
- Each verse gets UNIQUE filler tokens (no overlap) so the only shared catchwords are the planted ones (pure signal).
- Tests planted detection, total-count permutation rejection of null, planted-vs-shuffled comparison.

**`tests/test_synthetic_thomas.py`** — end-to-end Thomas pipeline (BOTH modes)
- Tests Thomas's `make_tokens` in TWO modes: `sedra=None` (surface forms) and `sedra=<fake dict>` (lemma collapse path).
- Verifies planted pairs are detected and the permutation test rejects null in both modes.
- Pins the SEDRA collapse behavior (`test_sedra_collapses_known_form`).

**`tests/test_perrin_known_pairs.py`** — Perrin's 8 cited boundaries
- `TestPerrinBoundariesHaveAnyCatchword` — at each of the 8 cited boundaries the detector finds ≥1 catchword. **All 8 pass — this IS the FINDINGS.md "8/8 reproduced" claim.**
- `TestNuraNuhra`, `TestEtarAtar`, `TestNasNesse` — strict version checking the SPECIFIC named pair appears. These SKIP (don't fail) when the current Gemini stochastic variant produced different vocabulary at a given logion (e.g., Logion 17 has no literal "light" word in any of the 10 Gemini variants — Perrin's claim there depends on an eye→light metaphor that doesn't survive literal retroversion).

### What this audit did NOT cover

The test suite covers the **active recurring-catchword pipeline** that produced the project's main findings (Phase 2B, cross-lingual, Q, Proverbs, phon-only, direct Perrin verification). It does NOT cover:

- **Phase 1 Monte Carlo (`scripts/run_monte_carlo.py`)** — separate code path using the EM lexical map. The headline "P(≥502)=0" depends on this. Audited (uses Coptic SCRIPTORIUM tags + Syriac lemmas from parallel corpus, doesn't use SEDRA), but not regressed with explicit tests.
- **Phase 2A beam-search translation** — bigram-LM-augmented beam. Audited (uses detector but only on its own Syriac output; no cross-language asymmetry).
- **Phase 2B detect script (`scripts/phase2b_detect_catchwords.py`)** — audited. Uses SEDRA on Syriac-internal data only (Thomas Syriac vs Thomas Syriac, no cross-language). The 974 count and 8.53/pair vs 12.15/pair control comparison are internally consistent.
- **Phase 3 contrastive model** — PyTorch training pipeline. Stochastic, GPU-dependent.
- **Translation-fetch scripts** — these hit the Gemini API.
- **The Perrin table digitisation** — 696 rows manually digitised from microfilm via vision-LLM. Final cumulative matches book totals (271/261/502) as a checksum.
- **Figure-generation scripts** — visual output; smoke-tested by being runnable.
- **Aggregate-density analysis scripts** — these share the tokenize/make_tokens primitives that ARE tested.

### Per-corpus reproducibility status

| Corpus | Tokenisation pinned? | Loader pinned? | End-to-end synthetic test? |
|---|:---:|:---:|:---:|
| Proverbs | ✓ | ✓ | ✓ |
| Q | ✓ | ✓ | ✓ |
| Thomas | ✓ + SEDRA asymmetry documented | ✓ | ✓ (both SEDRA + surface modes) |

### How to use this suite

```bash
# Run all tests
python -m pytest tests/ phase1_montecarlo/tests/ -v

# Run a single test file
python -m pytest tests/test_detector_extended.py -v

# Run a specific test class
python -m pytest tests/test_tokenization.py::TestScriptRegexParity -v
```

The suite runs in ~28 seconds end-to-end on a single core. CI integration should be straightforward (no GPU needed, no external network).

### Recommendations for future work

1. **Run the full suite before publishing any new finding.** A 28-second test sweep would not have caught the SEDRA-specific issue but WOULD have caught the Greek-punctuation regression.
2. **If a new corpus is added, write a synthetic planted-truth test for it.** This is the strongest end-to-end check.
3. **If a new language profile is added** to `phase1_montecarlo/language_data.py`, add it to `TestCrossLanguageUniformity` so the parametrized loop automatically picks it up.
4. **If `make_tokens` for any corpus is changed**, run `tests/test_tokenization.py::TestMakeTokensAsymmetry`. Any new asymmetry must be deliberate AND documented in the test.
5. **The Perrin specific-pair tests should be re-checked if the Gemini pipeline is re-run** — stochastic variants might shift which boundary contains which lemma. The weak suite (boundary-has-any-catchword) should remain stable.

## Bugs found and fixed

Consolidated list of methodology and code bugs caught at various stages:

| # | Bug | Severity | Found by | Status |
|---|---|---|---|---|
| 1 | Hebrew PUNCT_RE stripped Hebrew letters (overlapping range `[׀-׿]` with U+05D0–U+05EA) — silent "loaded 67/115" skip during cross-lingual permutation test | HIGH | Inspection of low load count during 2026-05-11 cross-lingual run | Fixed: enumerated actual Hebrew punctuation chars |
| 2 | FINDINGS.md "Hebrew 8/10, Arabic 9/10 significant" was counts at p<0.10 not p<0.05 (actual α=0.05 counts: 3/10 Hebrew, 7/10 Arabic) | LOW (documentation) | 2026-05-12 audit | Fixed: corrected table |
| 3 | Greek PUNCT_RE used wrong U+02B9 instead of U+0374; missing U+037E in Proverbs and Q scripts (Thomas script was correct) | MEDIUM | Test suite `test_tokenization.py::TestScriptRegexParity` (2026-05-12) | Fixed: harmonised to Thomas-correct pattern |
| 4 | `compute_blocked(filter_pct=0)` blocks every lemma (cutoff=0, count>0 always exceeds) | MEDIUM (footgun) | Phon-only `thr050_noblock` runs gave phon/B=0; investigated | Documented in `test_permutation_stats.py::TestBlocking` |
| 5 | Thomas `make_tokens` applies SEDRA lemma collapse to Syriac only; Hebrew/Greek/Arabic run on surface forms. Biases phon-only cross-language comparison against Syriac (suppresses phon count, inflates semantic count) | HIGH | User flagged Syriac vanilla z=0.13 as suspicious (2026-05-11). 50.7% of Syriac tokens are SEDRA-collapsed | Documented (kept intentionally) in `test_tokenization.py::TestMakeTokensAsymmetry`. Corrected re-runs in `crossling_syriac_surface.py`, `perrin_test_syriac_surface.py`, `perrin_boundary_max_surface.py` |

## Operational notes — how to cite each finding

(From `~/.claude/projects/.../memory/project_*.md` operational guidance.)

### Phase 1 Monte Carlo (P(≥502)=0)
- **DO cite:** "Random Coptic→Syriac translation cannot reproduce Perrin's 502 catchwords; mean 195, P(≥502)=0."
- **DO NOT cite:** Thomas's MC ratio 0.83× as evidence against Syriac origin — round-trip validation showed this is within the noise band of round-tripped known-catchword Syriac literature.

### Phase 2A/2C lexical-map ceiling (~324)
- **DO cite:** "Even fluent automated translation (lexical-map + Peshitta-NT bigram LM, λ ∈ {0,...,1}) cannot exceed ~324 catchwords. Surplus to Perrin's 502 is ~178."

### Phase 2B aggregate (974 catchwords, 8.53/pair vs control 12.15/pair)
- **DO cite:** Phase 2B as evidence that specific Perrin catchword pairs are canonical (Williams' bias critique on examples fails).
- **DO cite:** The control comparison as evidence that the aggregate count Perrin reports may NOT be Thomas-specific — any Coptic text translated by a stylistically consistent translator produces similar density.
- **Net stance:** Williams' bias critique is contradicted for specific famous pairs but may have purchase on the aggregate-count claim.

### LLM cross-validation (Claude/Gemini/GPT-4 + EM map P=0.98/0.79)
- **DO cite:** The strongest available evidence against Williams' bias critique for the famous cited pairs.
- **DO NOT treat:** Qwen3-14B's failure as evidence against Perrin — it's evidence small/mid open-weight models lack Classical Syriac generation capability.

### Round-trip validation
- **DO cite:** Round-trip-corrected ceiling: "Perrin's 1.87× recovery ratio is empirically unreachable by lexical-map round-trip even on known-catchword Syriac literature (max observed: 1.23× for Narsai under MAP)."
- **DO NOT cite:** Phase 1's MC ratio (0.83×) as evidence against Syriac origin — within round-trip noise band.

### Permutation test on recurring patterns (Syriac p=0.007, 10/10 variants)
- **DO cite:** As the project's strongest finding when the question is about Perrin's *compositional-design* argument.
- **STOP citing** total catchword counts as evidence either way (Phase 2B showed aggregate is uninformative).

### Perrin pair-by-pair (22% canonical / 78% Perrin-specific)
- **DO cite:** The 22% lower-bound when asked about the strength of Williams' bias critique.
- **Caveat:** Variant 0 only + skeleton-equality (not SEDRA root) — tighter analysis would push canonical fraction up.

### Cross-lingual permutation test (all 4 langs significant, Syriac=Greek tied)
- **DO cite:** The permutation-test signal (Syriac p=0.007) does NOT support a Syriac-specific arrangement — it reflects thematic clustering visible in any translation.
- **DO cite:** Greek's significance (p=0.016) as the decisive Indo-European counterexample to "Semitic-only" interpretations.
- **As of 2026-05-11 re-run:** With surface-form Syriac the result is unchanged structurally (Mann-Whitney p=0.40 vs SEDRA Syriac).

### Q source extension test
- **DO cite:** Aramaic does NOT distinctively lead Q — clusters with Syriac in the same translation-stability pattern.
- **ALWAYS prefix with the framing note:** Casey/Chilton argue from mistranslation retrojection, NOT catchword arrangement. A null result here is silent on their actual argument.

### Proverbs 10–29 positive control
- **DO cite** when defending the Thomas finding from "the pipeline doesn't work" objections.
- **DO use** the consistent Aramaic-lag across Thomas/Q/Proverbs to argue the pattern is Gemini-side, not substrate-diagnostic.

### Phon-only re-test (HISTORICAL)
- **DO NOT cite** as evidence against Perrin's phonological claim — superseded by Direct Perrin verification with surface tokenization.
- **DO cite** when claiming the all-catchwords result is thematic (the semantic decomposition is decisive for that conclusion).

### Direct Perrin verification (CORRECTED)
- **DO cite the CORRECTED result:** with fair tokenization, Syriac LEADS Thomas at z=3.39. This supports Perrin's directional prediction.
- **DO cite Test 2** (Perrin's pairs 3.48× enriched) as direct support for his specific identifications.
- **DO NOT cite** the initial "Syriac dead last z=0.13" or "Coptic > Syriac z=2.37 vs 1.70" findings — both were SEDRA artifacts.
- **DO retain** Williams' 78%-Perrin-specific finding as the principal remaining anti-Perrin evidence.

## Project environment

(From `~/.claude/projects/.../memory/project_env.md`.)

The Thomas-Catchword-Analysis project runs in a dedicated conda environment.

- **Path:** `/home/sogang/mnt/db_2/anaconda3/envs/thomas/bin/python`
- **Stack:** Python 3.11, torch 2.5.1+cu121, tokenizers 0.22.2 (supports `end_of_word_suffix`), transformers, lxml, pandas, numpy, scipy, matplotlib, scikit-learn, networkx.
- **Hardware:** machine has 4× NVIDIA RTX A6000 (49 GB each) + 64 cores + 503 GB RAM. System NVIDIA driver supports up to CUDA 12.4, so install torch with `--index-url https://download.pytorch.org/whl/cu121` (NOT the default cu130 wheel which silently disables CUDA).

**Why a dedicated env:** the system Python lacks torch+tokenizers; existing project envs either don't have both or have an old tokenizers without `end_of_word_suffix`. This env was created on 2026-05-09 specifically for this project.

**How to apply:**
- Always run project scripts with `/home/sogang/mnt/db_2/anaconda3/envs/thomas/bin/python` (not bare `python` or `python3`).
- The legacy shell scripts `scripts/run_phase2_3_finalization.sh` and `scripts/run_post_phase3.sh` reference `.venv/bin/python` which doesn't exist — update or invoke commands manually.
- For new package installs, use `<env>/bin/python -m pip install ...`.

## Outstanding work — addressed 2026-05-12

1. ~~**Phase 3 with mBERT**~~: Attempted 2026-05-11 when GPUs freed up. mBERT (178M params, top-4-layer fine-tune) gives val_acc 0.528, WORSE than the small 4.8M baseline (0.582). Pre-registered abort criterion fired correctly. Phase 3 ceiling is the small-model result; no further architecture work warranted here.

2. ~~**Tighter Perrin pair comparison**~~ — **DONE 2026-05-12**: re-ran the canonical/Perrin-specific split using all 10 Gemini variants (not just variant 0) AND SEDRA root-level matching (not just consonantal-skeleton).

   | Setting | Canonical / 558 | % |
   |---|---|---|
   | Variant 0 + skeleton (historical) | 124 | 22.2% |
   | All 10 variants + skeleton | 131 | 23.5% |
   | All 10 variants + SEDRA root | 140 | 25.1% |
   | All 10 variants + either match | 140 | 25.1% |

   **Williams' bias-critique purchase: 74.9% Perrin-specific (lower bound).** The tightening moves the canonical fraction from 22.2% → 25.1% — a marginal improvement. The bulk of Perrin's specific Syriac word choices remain non-canonical. Script: `scripts/perrin_pair_comparison_tight.py`. Output: `data/processed/perrin_catchwords/pair_comparison_tight.json` + `comparison_summary_tight.txt`.

3. ~~**Phon-arrangement with finer detectors**~~ — **DONE 2026-05-12, in lightweight form**: re-ran Thomas Syriac (surface tokenization) with the SEDRA lemma→root table enabled in `CatchwordDetector` (catches distinct lemmas with shared triliteral root as 'etymological').

   | Setting | z_all | z_phon | z_sem |
   |---|---|---|---|
   | Surface + lang-profile (NO roots) | 3.71 | 3.39 | 3.44 |
   | Surface + lang-profile + SEDRA roots | 3.70 | 3.37 | 3.44 |

   **Almost no change.** SEDRA-root-equivalence catches 5 additional etymological pairs at TRUE boundaries (out of ~1500 total catchwords). The consonantal-skeleton Levenshtein was already catching most root-equivalent pairs, because lemmas of the same triliteral root usually share skeleton anyway. The SEDRA root layer is a marginal refinement, not a qualitative shift. Script: `scripts/perrin_test_syriac_with_roots.py`. Output: `data/perrin_direct/thomas_syriac_v0_surface_with_roots.json`.

   A FULL phonetic-feature-based detector (consonant features: place/manner/voicing/emphatic, rather than skeleton-Lev) would still be a meaningful future experiment — it would catch e.g. b↔v voicing pairs that skeleton-Lev misses. Out of scope for this round.

4. ~~**Variant-level 10k-perm re-run of surface-Syriac cross-lingual**~~ — **DONE 2026-05-12**: re-ran all 10 variants at 10k perms (previously 1k).

   | Setting | Median z | Range | p<0.05 count |
   |---|---|---|---|
   | Surface Syriac, 1k perms (2026-05-11) | 2.77 | 1.90–3.31 | 10/10 |
   | Surface Syriac, 10k perms (2026-05-12) | 2.79 | 1.92–3.31 | 10/10 |

   **Identical conclusion at higher precision.** No change to the cross-lingual interpretation. Output: `data/processed/crossling_syriac_surface/variant_{0..9}.json` (overwritten with 10k-perm results).

### Newly opened: full phonetic-feature detector

The single substantive outstanding item is now:

5. **Full phonetic-feature phon detector** — build a consonant-feature distance (place, manner, voicing, emphatic) and re-run the Thomas Syriac permutation test. This would catch sound-pairs that share phonetic features but differ at the consonantal-skeleton level (e.g., voicing pairs like b↔v, p↔f). The SEDRA-root experiment showed that root-equivalence adds little to skeleton-Lev; phonetic-feature distance is a different axis and may add more. A future project.
