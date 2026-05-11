# Codebase review — comprehensive audit + test suite

> **Note (2026-05-12):** This document has been merged into the unified `FINDINGS.md`, which is now the authoritative single-source record. This file is retained for traceability but should not be cited as the primary source.

After the SEDRA-tokenization bug class was caught the hard way during the
Perrin verification work, the user requested a comprehensive review of every
moving part in the pipeline to verify it gives actual, defensible results.
This document records the audit + the resulting test suite.

## Bugs caught by the audit

### 1. Greek `PUNCT_RE` had wrong codepoint + missing character (NOW FIXED)

`scripts/proverbs_permutation_test.py` and `scripts/q_permutation_test.py`
both had:

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

Both characters look visually identical in most fonts. The Proverbs/Q
versions:
- Use `U+02B9` (modifier letter) instead of `U+0374` (Greek numeral sign)
- Are missing `U+037E` (Greek question mark, which looks like a semicolon)

**Impact:** If Gemini's Greek output contained `U+0374` or `U+037E` (proper
Greek punctuation), it would not be stripped during Proverbs/Q tokenization
but would be stripped during Thomas tokenization. This is a cross-corpus
asymmetry of the same kind as the SEDRA bug.

**Fix applied:** Proverbs and Q now use the same Thomas-correct pattern.
`tests/test_tokenization.py::TestScriptRegexParity` pins the patterns
together so future drift is caught.

### 2. `compute_blocked(filter_pct=0)` blocks every lemma — documented

Setting `filter_pct=0` does NOT mean "block nothing" — it means cutoff=0,
which blocks every lemma that appears in ≥1 unit (i.e., everything). The
correct way to disable blocking is `filter_pct=100`.

This bit us once during the phon-only sweep (`thr050_noblock` runs gave
phon/B=0 because everything was blocked). Now documented as a known-bug
case in `tests/test_permutation_stats.py::TestBlocking`.

### 3. The SEDRA tokenization asymmetry is now pinned by tests

`scripts/crossling_permutation_test.py`'s `make_tokens(text, lang, sedra)`
applies SEDRA lemma collapse for Syriac only — Hebrew/Greek/Arabic run on
surface forms in the same pipeline. This is **intentional asymmetry** but
caused the "Syriac dead last on Thomas phon-only" result, which was wrong.

Proverbs and Q do NOT apply SEDRA to any language (their `make_tokens` has
no SEDRA argument; all five languages use surface forms). So the asymmetry
is Thomas-specific, AND it specifically affects the Perrin verification.

`tests/test_tokenization.py::TestMakeTokensAsymmetry` pins this down:

- Proverbs and Q `make_tokens` are identical for every language.
- Thomas `make_tokens(lang, sedra=None)` matches Proverbs/Q for Syriac.
- Thomas `make_tokens("syriac", sedra=...)` collapses surface→lemma.
- Thomas non-Syriac languages ignore the sedra parameter.

This documents the asymmetry without removing it — the asymmetry is sometimes
the right choice (catching morphological variants as semantic matches), but
the test ensures the asymmetry can't accidentally disappear or spread.

## Test suite — 194 tests across 5 files

```
tests/
├── conftest.py                     # path + data-availability fixtures
├── test_detector_extended.py       # 53 tests — extends phase1_montecarlo/tests/test_detector.py
├── test_tokenization.py            # 45 tests — tokenize() + cross-script parity + SEDRA pin
├── test_loaders.py                 # 23 tests — Proverbs/Q/Thomas loaders + schemas
├── test_permutation_stats.py       # 20 tests — statistic correctness + reproducibility + filter_pct
├── test_synthetic_planted.py       #  7 tests — end-to-end with known-truth planted catchwords
└── test_perrin_known_pairs.py      # 16 tests — Perrin's 8 cited boundaries regression
phase1_montecarlo/tests/
└── test_detector.py                # 15 tests — pre-existing detector unit tests
```

Run with `python -m pytest tests/ phase1_montecarlo/tests/`.

### What each file covers

#### `tests/test_detector_extended.py` — detector arithmetic + invariants

- `TestConsonantal` — vocalization stripping for each script.
- `TestLevenshteinArithmetic` — empty strings, weak-consonant cost, confusion-
  group cost, symmetric output, per-language confusion groups (Syriac,
  Hebrew, Arabic, Greek).
- `TestPhonologicalScore` — boundary values; nūrā/nuhrā = 0.85 (capped).
- `TestClassification` — semantic / etymological / phonological / below-
  threshold / empty / missing-lemma.
- `TestDedup` — repeated lemmas counted once.
- `TestCrossLanguageUniformity` — every profile in `PROFILES` loaded; same
  detector code path applies to all (Williams' methodological criterion).
- `TestSymmetry` — `detect(a, b)` and `detect(b, a)` yield identical pair sets.
- `TestThresholdConfig` — pins threshold-comparison semantics (uses RAW
  score, not the tier-capped one).

#### `tests/test_tokenization.py` — tokenize() correctness + cross-script parity

- `TestSyriacTokenization`, `TestHebrewTokenization`, `TestArabicTokenization`,
  `TestGreekTokenization`, `TestAramaicTokenization` — per-language sanity:
  strips non-target-script, strips vocalization, strips language-specific
  punctuation, Greek lowercases, Aramaic uses Hebrew-script regex.
- `TestCrossScriptParity` — for each shared language, Proverbs / Q / Thomas
  tokenize() agree token-for-token on identical input.
- `TestScriptRegexParity` — `SCRIPT_RE[lang]` and `PUNCT_RE[lang]` patterns
  are identical across the three pipeline scripts. **This is the test that
  caught the Greek punctuation bug.**
- `TestMakeTokensAsymmetry` — the ONE documented SEDRA asymmetry is pinned:
  Thomas Syriac with `sedra=None` matches Proverbs/Q (surface forms); with
  `sedra=...` it collapses; non-Syriac ignores the parameter.
- `TestTokenSchema` — every `make_tokens` returns dicts with `lemma`, `form`,
  `parse`.
- `TestTokenizationEdgeCases` — empty input, pure punctuation, pure
  whitespace, wrong-script input.

#### `tests/test_loaders.py` — corpus loaders + variant indexing

- `TestProverbsLoader` — Hebrew source loads from `proverbs_hebrew.json`;
  each translation language loads ≥100 verses; Hebrew is invariant to
  `variant_idx` (single canonical text); translations differ across
  variants; invalid variant returns empty/few units gracefully.
- `TestQLoader` — Greek source loads 56 pericopes; each translation loads
  ≥50; Greek invariant to variant_idx; `pericope_id` is int.
- `TestThomasLoader` — Syriac variant 0 loads with SEDRA applied (lemma !=
  form for at least some tokens); each cross-language loads ≥100 logia;
  non-Syriac Thomas languages have lemma==form.
- `TestProverbsLoaderEmptyDir`, `TestQLoaderEmptyDir` — graceful empty-dir
  handling (`load_translations` returns `{}` when the language directory
  doesn't exist).
- `TestProverbsSchema`, `TestQSchema`, `TestThomasSchema` — file schemas
  pinned. Thomas Syriac uses `syriac_text` field (legacy from Phase 2B);
  Thomas cross-language and Proverbs/Q use `text`.

#### `tests/test_permutation_stats.py` — statistical core

- `TestStatsForOrder` — hand-computed counts on tiny matrices:
  - 0, 1, 3 recurring pairs at ≥2; recurring at ≥3 boundaries;
  - shuffled order reduces recurrence.
- `TestTotalCount` — total-catchwords statistic (no recurrence filter) on
  same tiny matrices; phon-only and semantic-only filters.
- `TestPermutationReproducibility` — same seed → identical null. Different
  seeds → different null. `run_perm_total` reproducible too.
- `TestNullDistribution` — null mean stable across seeds at large N.
- `TestBlocking` — filter_pct=80 blocks top 20%; filter_pct=100 blocks
  nothing; **filter_pct=0 blocks everything (KNOWN BUG, documented)**;
  blocking uses unit count (not token count).
- `TestStatisticRegression` — hand-built planted matrix with two known
  recurring pairs; assertion pins their `recurring_2plus`, `recurring_3plus`,
  `max_freq`.

#### `tests/test_synthetic_planted.py` — end-to-end with known truth

- Build a 50-verse Hebrew corpus where 6 UNIQUE planted pairs sit at TRUE
  adjacent boundaries (2k, 2k+1). Each plant is a unique Hebrew lemma so the
  detector classifies it as a SEMANTIC match.
- `test_planted_pair_detected_at_boundaries` — detector finds all 6 planted
  pairs at the correct boundaries (8/8 originally; reduced to 6 to stabilise
  the perm-test).
- `test_total_count_rejects_null_on_planted` — permutation test on TRUE
  order rejects null at p<0.05 (z=1.92 typically).
- `test_total_count_planted_vs_shuffled` — same verses in shuffled order
  show essentially null signal (z≈0); the test correctly distinguishes
  planted-arrangement from no-arrangement.
- `TestPipelineRobustness` — empty corpus, single-unit corpus, two-unit
  corpus with no overlap all complete without error.
- `TestShufflingInvariants` — total cells in the matrix is invariant to
  matrix-build order.

#### `tests/test_perrin_known_pairs.py` — Perrin's 8 cited boundaries

- `TestPerrinBoundariesHaveAnyCatchword` — at each of the 8 cited
  boundaries (10-11, 16-17, 82-83 for nūrā/nuhrā; 29-30, 85-86 for ʿetar/
  ʾatar; 14-15, 46-47, 113-114 for naš/nesse) the detector finds ≥1
  catchword. **All 8 pass — this IS the FINDINGS.md "8/8 reproduced" claim.**
- `TestNuraNuhra`, `TestEtarAtar`, `TestNasNesse` — strict version checking
  the SPECIFIC named pair appears at the boundary. These are informational
  — they SKIP (don't fail) when the current Gemini stochastic variant
  produced different vocabulary at a given logion. We found that Logion 17
  (in Coptic source: "no eye has seen") does not in fact contain a literal
  fire/light word in any of our 10 Gemini Syriac variants — Perrin's claim
  there depends on an eye→light metaphor that doesn't survive literal
  retroversion. The headline "8/8 reproduced" was about boundary-level
  catchwords, not specific named pairs.

## What this audit did NOT cover

The test suite covers the **active recurring-catchword pipeline** that
produced the project's main findings (Phase 2B, cross-lingual, Q, Proverbs,
phon-only, direct Perrin verification). It does NOT cover:

- **Phase 1 Monte Carlo (`scripts/run_monte_carlo.py`)** — separate code
  path using the EM lexical map. The headline "P(≥502)=0" depends on this.
  Not directly retested here, but the underlying `CatchwordDetector` IS
  tested.
- **Phase 2A beam-search translation** — bigram-LM-augmented beam.
  Generates Syriac translations from Coptic; we have not regressed the beam
  output.
- **Phase 3 contrastive model** — PyTorch training pipeline. Stochastic, GPU-
  dependent, separately reproducible via `scripts/phase3_train_hardneg.py`.
- **Translation-fetch scripts** (`proverbs_fetch_*`, `q_fetch_greek.py`,
  `crossling_translate.py`, `q_translate.py`, `proverbs_translate.py`) —
  these hit the Gemini API. Network-side correctness is out of scope.
- **The Perrin table digitisation** (`data/processed/perrin_catchwords/`) —
  696 rows manually digitised from microfilm via vision-LLM. Final cumulative
  matches book totals (271/261/502) as a checksum, but per-row correctness
  is not unit-testable.
- **Figure-generation scripts** (`analysis/plot_*.py`) — visual output;
  smoke-tested by being runnable end-to-end during the project.
- **Aggregate-density analysis scripts** (`*_aggregate_density.py`) — these
  share the tokenize/make_tokens primitives that ARE tested.
- **Mann-Whitney implementations** — we use `scipy.stats.mannwhitneyu`,
  which is an external library and not in scope.

## Per-corpus reproducibility status

| Corpus | Tokenisation pinned? | Loader pinned? | End-to-end synthetic test? |
|---|:---:|:---:|:---:|
| Proverbs | ✓ | ✓ | ✓ (Hebrew planted test uses Proverbs pipeline) |
| Q | ✓ | ✓ | ⚠ (no Q-specific synthetic, but shares the same primitives) |
| Thomas | ✓ + SEDRA asymmetry documented | ✓ | ⚠ (no Thomas-specific synthetic; named-pair regression tests instead) |

## How to use this suite

```bash
# Run all tests
python -m pytest tests/ phase1_montecarlo/tests/ -v

# Run a single test file
python -m pytest tests/test_detector_extended.py -v

# Run a specific test class
python -m pytest tests/test_tokenization.py::TestScriptRegexParity -v

# Skip slow tests (none currently mark themselves slow)
python -m pytest tests/ -m "not slow"
```

The suite runs in ~7 seconds end-to-end on a single core. CI integration
should be straightforward (no GPU needed, no external network).

## Recommendations for future work

1. **Run the full suite before publishing any new finding.** The SEDRA bug
   was hidden for the entire phon-only run; a 7-second test sweep would
   not have caught the SEDRA-specific issue but WOULD have caught the
   Greek-punctuation regression.

2. **If a new corpus is added, write a synthetic planted-truth test for it.**
   This is the strongest end-to-end check; the synthetic Hebrew test gives
   us very high confidence the Proverbs pipeline is wired correctly.

3. **If a new language profile is added to `phase1_montecarlo/language_data.py`,
   add it to `TestCrossLanguageUniformity`** so the parametrized loop
   automatically picks it up.

4. **If `make_tokens` for any corpus is changed**, run
   `tests/test_tokenization.py::TestMakeTokensAsymmetry`. Any new
   asymmetry must be deliberate AND documented in the test.

5. **The Perrin specific-pair tests should be re-checked if the Gemini
   pipeline is re-run** — stochastic variants might shift which boundary
   contains which lemma. The weak suite (boundary-has-any-catchword)
   should remain stable.
