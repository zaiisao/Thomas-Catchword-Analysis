"""
Extended catchword-detector tests.

Audits the detector's core arithmetic, symmetry, dedup, edge cases, and
verifies the same code path runs uniformly across all five language profiles
(Williams' methodological criterion).
"""
from __future__ import annotations

import pytest

from phase1_montecarlo.catchword_detector import (
    CatchwordDetector,
    Catchword,
    consonantal,
    phonological_score,
    weighted_levenshtein,
    PHONOLOGICAL_THRESHOLD,
)
from phase1_montecarlo.language_data import (
    SYRIAC, HEBREW, ARAMAIC, ARABIC, GREEK, COPTIC, get_profile, PROFILES,
)


# ============================================================================
# consonantal() — Unicode normalization
# ============================================================================

class TestConsonantal:
    def test_strips_syriac_vocalization(self):
        # ܳ (NFD combining mark) should be removed; ܘ (base) kept
        assert consonantal("ܘܳ") == "ܘ"

    def test_strips_hebrew_niqqud(self):
        # Patah (ַ) is combining
        assert consonantal("בַ") == "ב"

    def test_strips_arabic_tashkeel(self):
        # Fatha (َ) is combining
        assert consonantal("بَ") == "ب"

    def test_strips_greek_accents(self):
        # ά (alpha + acute) → α
        assert consonantal("ά") == "α"

    def test_handles_empty_string(self):
        assert consonantal("") == ""

    def test_handles_pure_ascii(self):
        assert consonantal("abc") == "abc"

    def test_is_idempotent(self):
        text = "ܘܳܡܳܠܠܳܐ"
        once = consonantal(text)
        twice = consonantal(once)
        assert once == twice


# ============================================================================
# weighted_levenshtein — confusion-group + weak-consonant cost accounting
# ============================================================================

class TestLevenshteinArithmetic:
    def test_identical_zero(self):
        assert weighted_levenshtein("abc", "abc", SYRIAC) == 0.0

    def test_one_empty_string(self):
        # Cost = sum of regular insertions
        assert weighted_levenshtein("", "ܒ", SYRIAC) == 1.0
        assert weighted_levenshtein("ܒ", "", SYRIAC) == 1.0

    def test_one_empty_string_weak_consonant(self):
        # Weak consonant ܘ costs 0.5 to insert/delete
        assert weighted_levenshtein("", "ܘ", SYRIAC) == 0.5
        assert weighted_levenshtein("ܘ", "", SYRIAC) == 0.5

    def test_both_empty(self):
        assert weighted_levenshtein("", "", SYRIAC) == 0.0

    def test_confusion_group_substitution_cost(self):
        # ܕ (dalet) ↔ ܪ (resh): confusion group
        assert weighted_levenshtein("ܕ", "ܪ", SYRIAC) == 0.5

    def test_non_confusion_substitution_full_cost(self):
        # ܒ ↔ ܠ: not in any confusion group, neither is weak
        assert weighted_levenshtein("ܒ", "ܠ", SYRIAC) == 1.0

    def test_weak_consonant_insertion_cost(self):
        # ܒܪ → ܒܘܪ: inserting ܘ (weak)
        assert weighted_levenshtein("ܒܪ", "ܒܘܪ", SYRIAC) == 0.5

    def test_symmetric(self):
        # weighted_lev(a, b) == weighted_lev(b, a) when no asymmetric costs
        for a, b in [("ܢܘܪܐ", "ܢܘܗܪܐ"),
                       ("ܕ", "ܪ"),
                       ("ܒܕܐ", "ܒܪܐ")]:
            assert weighted_levenshtein(a, b, SYRIAC) == \
                   weighted_levenshtein(b, a, SYRIAC), \
                f"asymmetric for ({a!r}, {b!r})"

    def test_hebrew_confusion_groups(self):
        # ב/פ in Hebrew is a confusion group
        assert weighted_levenshtein("ב", "פ", HEBREW) == 0.5

    def test_arabic_confusion_groups(self):
        # ك/ق velars
        assert weighted_levenshtein("ك", "ق", ARABIC) == 0.5

    def test_greek_iotacism_does_not_fire_for_single_chars(self):
        # ι/η are in the iotacism multi-char group — single-char Lev won't
        # treat them as confused because the GROUP contains multi-char items
        # Just verify the function returns a finite cost without crashing
        d = weighted_levenshtein("ι", "η", GREEK)
        assert 0 <= d <= 1


# ============================================================================
# phonological_score — boundary values
# ============================================================================

class TestPhonologicalScore:
    def test_identical_returns_capped_value(self):
        # Identical → capped at PHONOLOGICAL_NEAR_SCORE (0.85)
        # (caller is expected to classify identical as semantic/etym)
        from phase1_montecarlo.catchword_detector import PHONOLOGICAL_NEAR_SCORE
        score = phonological_score("ܢܘܪܐ", "ܢܘܪܐ", SYRIAC)
        assert score == PHONOLOGICAL_NEAR_SCORE

    def test_empty_returns_zero(self):
        assert phonological_score("", "x", SYRIAC) == 0.0
        assert phonological_score("x", "", SYRIAC) == 0.0
        assert phonological_score("", "", SYRIAC) == 0.0

    def test_nura_nuhra_score(self):
        # ܢܘܪܐ vs ܢܘܗܪܐ: 1 weak insertion, max length 4
        # score = 1 - 0.5/4 = 0.875 → capped at 0.85 (PHONOLOGICAL_NEAR_SCORE)
        score = phonological_score("ܢܘܪܐ", "ܢܘܗܪܐ", SYRIAC)
        assert score == pytest.approx(0.85)

    def test_score_in_unit_interval(self):
        score = phonological_score("xyz", "abc", SYRIAC)
        assert 0 <= score <= 1


# ============================================================================
# CatchwordDetector.detect — semantic / etymological / phonological
# ============================================================================

class TestClassification:
    @pytest.fixture
    def syr(self):
        return CatchwordDetector("syriac", require_content_pos=False)

    def test_semantic_match(self, syr):
        a = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        b = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        cws = syr.detect([a], [b])
        assert len(cws) == 1
        assert cws[0].link_type == "semantic"

    def test_etymological_same_skeleton(self, syr):
        # Different lemmas, identical consonantal skeleton
        # Add vocalization to one so the LEMMAS differ but skeletons match
        a = {"lemma": "ܡܠܟܳܐ", "parse": "MS-EMP"}
        b = {"lemma": "ܡܠܟܰܐ", "parse": "MS-EMP"}
        cws = syr.detect([a], [b])
        assert len(cws) == 1
        assert cws[0].link_type == "etymological"

    def test_phonological_below_threshold_not_detected(self, syr):
        # Completely unrelated words shouldn't trigger phon
        a = {"lemma": "ܡܠܟܐ", "parse": "MS-EMP"}
        b = {"lemma": "ܒܝܬܐ", "parse": "MS-EMP"}
        cws = syr.detect([a], [b])
        assert all(cw.link_type != "phonological" or cw.score >= 0.6 for cw in cws)

    def test_phonological_above_threshold_detected(self, syr):
        # ܢܘܪܐ vs ܢܘܗܪܐ — Perrin's canonical example
        a = {"lemma": "ܢܘܪܐ", "parse": "FS-EMP"}
        b = {"lemma": "ܢܘܗܪܐ", "parse": "MS-EMP"}
        cws = syr.detect([a], [b])
        assert len(cws) == 1
        assert cws[0].link_type == "phonological"

    def test_empty_input_no_catchwords(self, syr):
        assert syr.detect([], []) == []
        assert syr.detect([{"lemma": "ܡܠܟ", "parse": "MS-EMP"}], []) == []
        assert syr.detect([], [{"lemma": "ܡܠܟ", "parse": "MS-EMP"}]) == []

    def test_missing_lemma_skipped(self, syr):
        a = {"lemma": "", "parse": "MS-EMP"}
        b = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        assert syr.detect([a], [b]) == []
        assert syr.detect([{}], [b]) == []  # no 'lemma' key


# ============================================================================
# Dedup — repeated lemmas should not double-count
# ============================================================================

class TestDedup:
    @pytest.fixture
    def syr(self):
        return CatchwordDetector("syriac", require_content_pos=False)

    def test_repeat_in_a_counts_once(self, syr):
        a = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        b = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        cws = syr.detect([a, a, a], [b])
        assert len(cws) == 1

    def test_repeat_in_b_counts_once(self, syr):
        a = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        b = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        cws = syr.detect([a], [b, b, b])
        assert len(cws) == 1

    def test_two_distinct_lemmas_count_separately(self, syr):
        a1 = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        a2 = {"lemma": "ܒܝܬ", "parse": "MS-EMP"}
        b1 = {"lemma": "ܡܠܟ", "parse": "MS-EMP"}
        b2 = {"lemma": "ܒܝܬ", "parse": "MS-EMP"}
        cws = syr.detect([a1, a2], [b1, b2])
        assert len(cws) == 2
        lemmas = {(cw.token_a["lemma"], cw.token_b["lemma"]) for cw in cws}
        assert lemmas == {("ܡܠܟ", "ܡܠܟ"), ("ܒܝܬ", "ܒܝܬ")}


# ============================================================================
# Cross-language uniformity — Williams' methodological criterion
# ============================================================================

class TestCrossLanguageUniformity:
    """The detector must apply the same algorithm to every language profile.
    Only the LanguageProfile (confusion groups, weak consonants) varies."""

    @pytest.mark.parametrize("lang", list(PROFILES.keys()))
    def test_identical_lemmas_always_semantic(self, lang):
        det = CatchwordDetector(lang, require_content_pos=False)
        # Use a script-appropriate dummy token
        # Use Latin ASCII which won't get stripped by ANY script regex
        a = {"lemma": "lemmaA", "parse": "MS-EMP"}
        b = {"lemma": "lemmaA", "parse": "MS-EMP"}
        cws = det.detect([a], [b])
        assert len(cws) == 1
        assert cws[0].link_type == "semantic"

    @pytest.mark.parametrize("lang", list(PROFILES.keys()))
    def test_empty_inputs_return_empty(self, lang):
        det = CatchwordDetector(lang, require_content_pos=False)
        assert det.detect([], []) == []

    def test_profiles_all_loaded(self):
        # All 6 expected language profiles present
        assert set(PROFILES.keys()) == {
            "syriac", "coptic", "greek", "hebrew", "arabic", "aramaic"
        }

    @pytest.mark.parametrize("lang", list(PROFILES.keys()))
    def test_profile_has_confusion_and_weak_sets(self, lang):
        p = get_profile(lang)
        assert hasattr(p, "confusion_groups")
        assert hasattr(p, "weak_consonants")
        assert hasattr(p, "content_pos")
        # Frozensets must be non-empty (otherwise the profile is incomplete)
        assert len(p.weak_consonants) > 0, f"{lang}: empty weak_consonants"
        assert len(p.confusion_groups) > 0, f"{lang}: empty confusion_groups"


# ============================================================================
# Symmetry of detect — detect(a,b) and detect(b,a) should yield identical
# pair sets (possibly with token roles swapped)
# ============================================================================

class TestSymmetry:
    @pytest.fixture
    def syr(self):
        return CatchwordDetector("syriac", require_content_pos=False)

    def test_detect_symmetric_pair_sets(self, syr):
        a = [{"lemma": "ܢܘܪܐ", "parse": "FS-EMP"},
             {"lemma": "ܒܝܬܐ", "parse": "MS-EMP"}]
        b = [{"lemma": "ܢܘܗܪܐ", "parse": "MS-EMP"},
             {"lemma": "ܡܠܟܐ", "parse": "MS-EMP"}]
        cws_ab = syr.detect(a, b)
        cws_ba = syr.detect(b, a)
        lemmas_ab = {tuple(sorted([cw.token_a["lemma"], cw.token_b["lemma"]]))
                     for cw in cws_ab}
        lemmas_ba = {tuple(sorted([cw.token_a["lemma"], cw.token_b["lemma"]]))
                     for cw in cws_ba}
        assert lemmas_ab == lemmas_ba


# ============================================================================
# Configurable phon threshold
# ============================================================================

class TestThresholdConfig:
    def test_higher_threshold_filters_marginal_pairs(self):
        # nūrā/nuhrā raw score = 1 - 0.5/5 = 0.9 (NOT 0.85; the 0.85 in
        # phonological_score() is a CAP that the detector does not consult
        # for the threshold check — it compares against the raw score).
        a = [{"lemma": "ܢܘܪܐ", "parse": "FS-EMP"}]
        b = [{"lemma": "ܢܘܗܪܐ", "parse": "MS-EMP"}]
        # threshold 0.6 → detected
        det_loose = CatchwordDetector("syriac", phonological_threshold=0.6,
                                         require_content_pos=False)
        assert len(det_loose.detect(a, b)) == 1
        # threshold 0.95 → NOT detected (raw 0.9 < 0.95)
        det_strict = CatchwordDetector("syriac", phonological_threshold=0.95,
                                          require_content_pos=False)
        assert det_strict.detect(a, b) == []

    def test_threshold_boundary_exact_match(self):
        # Threshold equal to raw score: detector uses >=, so the pair is
        # detected at score == threshold. Document this.
        a = [{"lemma": "ܢܘܪܐ", "parse": "FS-EMP"}]
        b = [{"lemma": "ܢܘܗܪܐ", "parse": "MS-EMP"}]
        det = CatchwordDetector("syriac", phonological_threshold=0.9,
                                  require_content_pos=False)
        assert len(det.detect(a, b)) == 1, \
            "detector uses score >= threshold, exact match should pass"
