# Direct verification of Perrin's Syriac-paronomasia claim

**Major correction (2026-05-11, late):** the earlier "Syriac dead last" finding
was a methodological artifact. Syriac was the only language being
SEDRA-lemmatized while Hebrew/Greek/Arabic ran on surface forms. SEDRA
collapsed 50.7% of Syriac tokens to root lemmas, absorbing variant-pair
surface forms into "semantic" (same-root) matches and suppressing the
phonological count. With apples-to-apples surface-form tokenization, Syriac
LEADS Thomas on the language-aware detector (z_phon = 3.39, p = 0.0005)
and is competitive on the language-neutral one. See "Methodological
correction" section below.


**Question.** Perrin (2002) argues that the Gospel of Thomas was composed in
Syriac, with deliberate paronomastic catchwords linking adjacent logia (e.g.,
`ܢܘܪܐ nūrā` "fire" / `ܢܘܗܪܐ nuhrā` "light" at logion 10–11). The earlier
permutation-test framework (memory `project_phon_only`) found that source
languages fail to lead their translations on the all-catchwords test and
attributed this to a pipeline limitation. This document tests Perrin's claim
through five fresh angles to see whether the limitation is real or fixable.

## The pipeline limitations we identified

1. **Wrong statistic.** The original test counted *recurring* pairs (same
   lemma pair appearing at ≥2 adjacent boundaries). Perrin's claim is about
   *specific* paronomastic pairs at *individual* boundaries. The recurrence
   filter discards the very signal we want.
2. **No direct query against Perrin's actual list.** We had Perrin's 502
   digitised pairs but had never asked "do Perrin's specific pairs score
   higher than random Syriac pairs would?"
3. **Unfair cross-language comparison.** Each language used its own
   confusion-group profile (Syriac 6 groups, Hebrew 7, Arabic 10), making
   raw z-scores incomparable across languages.

## Five direct tests, summarised

| Test | What it measures | Result |
|---|---|---|
| 1 | Total phon-catchwords at adjacent boundaries (no recurrence filter), Thomas Syriac | **z = 1.70, p = 0.049 ✓** (vs old recurrence test z = 0.95, p = 0.19). Signal recoverable. |
| 2 | Perrin's 990 cited Syriac pairs vs 10,000 random Syriac pairs | **Perrin pairs 3.48× more phonologically similar (p < 1e-9).** Even excluding identical pairs, p = 0.005. |
| 3 | Threshold/blocking sweep on Thomas | Looser threshold (0.5) helps Syriac slightly (z=2.19) but doesn't change cross-language ranking. |
| 4 | VANILLA Levenshtein on Thomas (no confusion-group bonuses) | **Syriac drops to z = 0.13 (p = 0.46, n.s.)** while Hebrew z = 3.13, Greek z = 2.76, Arabic z = 3.01. |
| 5 | Per-boundary MAX phon score across languages | **Syriac wins 23/114 boundaries (20.2%) vs null mean 24.2 (z = −0.30, p = 0.66).** Syriac is INDISTINGUISHABLE from random adjacency. |
| 6 | The actual extant Coptic Thomas (the manuscript) | **Coptic z_phon = 2.37, p = 0.010 ✓** — *higher* than Syriac retroversion (z = 1.70). |

## The decisive findings

### (A) Perrin's specific pairs ARE real Syriac phonology

Test 2 is unambiguous. Of Perrin's 990 cited Syriac word pairs at adjacent
boundaries:
- 8.2% are phonologically similar (above threshold 0.6) vs 2.4% for random
  Syriac pairs from the same corpus → **3.48× enrichment**.
- 7.6% are semantically/etymologically identical vs 0.7% for random → **10.2×
  enrichment**.
- Mann–Whitney on raw score distribution: p = 8.52e-10.
- Including the canonical `nūrā / nuhrā` at logion 10–11 (score 0.900).

Williams' bias critique was that Perrin's choices were arbitrary. They are
not. His selections capture real Syriac sound-similar word pairs at
significantly higher rates than random.

### (B) The phon arrangement is NOT Syriac-specific in Thomas

Three converging tests show this:

**Test 4 (vanilla Levenshtein) — same detector for all languages:**

| Lang | Thomas z_phon (lang-profile) | Thomas z_phon (vanilla) |
|---|---|---|
| Syriac (the analyzed lang) | 1.70 ✓ | **0.13 n.s.** |
| Hebrew | 2.91 ✓ | 3.13 ✓ |
| Greek | 3.27 ✓ | 2.76 ✓ |
| Arabic | 2.62 ✓ | 3.01 ✓ |

The Syriac signal we saw with the lang-specific detector was *entirely* a
function of the confusion-group bonuses (ܕ/ܪ, ܒ/ܦ, ܬ/ܛ, ܣ/ܫ, ܥ/ܐ, ܚ/ܗ) and
weak-consonant-insertion discounts (ܘ, ܗ, ܝ, ܐ). Strip those and Syriac
shows no phon-arrangement above chance. Other languages remain strongly
significant.

**Test 5 (boundary-MAX) — which language has the highest MAX phon-score at
each of the 114 Thomas boundaries:**

- Hebrew wins 52/114 (45.6%)
- Greek wins 28/114 (24.6%)
- **Syriac wins 23/114 (20.2%)** — BELOW chance (25%)
- Arabic wins 11/114 (9.6%)

The "Syriac wins" count at TRUE order (23) equals the null mean under random
shuffles (24.2), z = −0.30, p = 0.66. **There is no TRUE-order arrangement
effect specific to Syriac.**

**Test 6 (Coptic Thomas, the actual extant manuscript):**

Running the same total-count test on the genuine Coptic text gives
z_phon = 2.37, p = 0.010 ✓. *Coptic shows more phon-arrangement at adjacent
boundaries than the Syriac retroversion does.* If Perrin's hypothesis were
correct (Syriac source with deliberate sound-design that didn't survive
translation to Coptic), Syriac should exceed Coptic. The opposite is
observed.

### (C) The pipeline is not fundamentally broken — the fair statistic moves Hebrew Proverbs from "worst" to "top tier"

On **Proverbs** (positive control, documented Hebrew catchword text):

| Lang | Lang-profile detector | Vanilla Lev (fair) |
|---|---|---|
| Hebrew (SRC) | z = 2.49 (4th of 5) | **z = 3.60 (2nd of 5)** |
| Aramaic | z = 4.13 | z = 4.13 |
| Arabic | z = 3.17 | z = 3.34 |
| Greek | z = 2.03 | z = 2.92 |
| Syriac | z = 2.79 | z = 2.57 |

With vanilla Lev, Hebrew jumps from 4th to 2nd, behind Aramaic only (Aramaic
in our pipeline uses Hebrew-script orthography and shares Hebrew roots, so a
Hebrew → Aramaic retroversion preserves much of Hebrew's structure — this
is a known translation-pair artifact, not a counterexample).

The fair test thus moves Hebrew from "worst on its own text" to "top tier".
Compare to Thomas Syriac: dead-last 4/4 with vanilla Lev, dead-last 4/4 on
the boundary-MAX test, *below* Coptic. On the positive control, the source
language is competitive with translations; on Thomas, Syriac is not.

## Verdict

| Claim | Evidence | Verdict |
|---|---|---|
| Perrin's specific 502 cited pairs capture real Syriac phonology | Test 2 (3.48× enrichment, p<1e-9) | **Supported** |
| The catchword pairs Perrin cites are non-random selections | Test 2 + LLM cross-validation 2026-05-09 (`project_llm_cross_validation`) | **Supported** |
| Thomas has phon-arrangement at adjacent boundaries | Tests 1, 4, 6 (all 4 langs + Coptic significant) | **Supported** (in every language tested) |
| The phon arrangement is SYRIAC-SPECIFIC (Perrin's main claim) | Tests 4, 5, 6 | **NOT supported** — Syriac is the WEAKEST among target languages, and Coptic exceeds Syriac |
| Thomas was originally composed in Syriac (Perrin's broader claim) | All six tests + memory `project_perrin_table_comparison` | **NOT supported** by these computational tests |

The two-part picture:

- **Perrin found something real**: the Syriac pairs he cites are
  phonologically structured beyond what random Syriac retroversion produces.
- **But the interpretation is wrong**: the phon arrangement is detected in
  *every* language at adjacent boundaries (Hebrew, Greek, Arabic, Coptic,
  Aramaic all significant), and Syriac is *weakest*, not strongest. The
  positional sound-similarity Perrin observed is a property of the underlying
  thematic clustering of logia that survives all retroversions, not a Syriac
  compositional design.

The remaining honest qualification: our detector is consonantal-skeleton +
Levenshtein. Real paronomasia involves rhyme, meter, alliteration at depths
this detector cannot reach. A negative result here does not formally exclude
Syriac-specific sound-design at finer levels. But there is no positive
computational evidence for it across six different test angles.

## Methodological correction (added late 2026-05-11)

After observing Syriac vanilla z_phon = 0.13 (suspiciously low), user
flagged a possible bug. Investigation revealed: **Syriac tokens were being
lemmatized via SEDRA (Syriac Electronic Database, ~16,000 Peshitta-derived
entries) before catchword detection, while Hebrew/Greek/Arabic ran on
surface forms.** SEDRA collapsed 50.7% of Syriac surface tokens to root
lemmas (1320 unique surface forms → 712 unique lemmas, 1.85× compression).

Effect: surface forms like `ܡܠܟܐ` (the king) and `ܡܠܟܘܬܐ` (kingdom) collapse
to the same SEDRA lemma `ܡܠܟ`, get classified as a SEMANTIC match, and
never reach the phonological detector. In Hebrew/Greek/Arabic, the
equivalent surface pairs stay distinct → counted as phonological matches.

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

With vanilla Lev + surface forms, Syriac is competitive (z = 2.74) — close
to Greek (2.76), behind Hebrew (3.13) and Arabic (3.01). Boundary-MAX with
surface Syriac: Syriac wins 26/114 boundaries (22.8%) vs null mean 28.8
(z = −0.69, p = 0.79) — still no boundary-winner advantage, because Hebrew
has high MAX scores almost everywhere as a baseline. But the
TOTAL-count permutation result is unambiguous: with apples-to-apples
tokenization, Syriac leads.

**This is a major revision toward Perrin.** The corrected picture:

- ✓ Perrin's specific cited pairs ARE real Syriac phonology (Test 2 stands).
- ✓ Thomas Syriac DOES lead the cross-language phon test (Test 1 corrected).
- ✓ Syriac has the highest TOTAL-count phon arrangement at adjacent
  boundaries among the 4 retroversion languages.
- ✗ The boundary-MAX winner test still doesn't show Syriac dominance
  (Hebrew wins more boundaries due to confusion-group baseline), but the
  total-count test is the more direct match to Perrin's claim.

**What stands of the prior "anti-Perrin" conclusions:**

- Test 6 (Coptic z = 2.37 > Syriac z = 1.70 with SEDRA, but Syriac with
  surface goes to 3.39): Coptic actually shows LESS phon-arrangement than
  fairly-tokenized Syriac. The original Coptic > Syriac result was also
  driven by the SEDRA asymmetry. Perrin's directional prediction (Syriac >
  Coptic) is preserved.
- The Williams pair-by-pair critique (78% Perrin-specific) is unchanged —
  Perrin's specific 502 list is partly inflated by his selections, even
  though those selections capture real Syriac phonology.

## Files

- `data/perrin_direct/thomas_*_v0.json` — Test 1 (5 langs × default detector)
- `data/perrin_direct/perrin_pair_benchmark.json` — Test 2
- `data/perrin_direct/vanilla_thomas_*_v0.json` — Test 4 (vanilla detector)
- `data/perrin_direct/boundary_max_thomas.json` — Test 5 (cross-lang MAX)
- `data/perrin_direct/coptic_thomas_v0.json` — Test 6 (Coptic source)
- `data/perrin_direct/vanilla_proverbs_*_v0.json` — positive control
- `scripts/perrin_test_one.py` — total-count permutation worker
- `scripts/perrin_test_vanilla.py` — vanilla-Levenshtein worker
- `scripts/perrin_boundary_max.py` — boundary-MAX cross-lang test
- `scripts/perrin_test_coptic.py` — Coptic Thomas test
- `scripts/perrin_pair_benchmark.py` — Perrin specific-pair benchmark
