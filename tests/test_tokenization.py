"""
Tokenization correctness + cross-script tokenization-parity audit.

Direct response to the SEDRA bug class: any time a language gets unique
treatment in the tokenization or lemma stage, that asymmetry must be
intentional and documented. These tests pin down what is and isn't symmetric.
"""
from __future__ import annotations

import pytest

# Pull the same tokenize() from each pipeline script. They MUST stay in sync.
from scripts.proverbs_permutation_test import (
    tokenize as prov_tokenize,
    make_tokens as prov_make_tokens,
    SCRIPT_RE as prov_SCRIPT,
    PUNCT_RE as prov_PUNCT,
)
from scripts.q_permutation_test import (
    tokenize as q_tokenize,
    make_tokens as q_make_tokens,
    SCRIPT_RE as q_SCRIPT,
    PUNCT_RE as q_PUNCT,
)
from scripts.crossling_permutation_test import (
    tokenize as thom_tokenize,
    make_tokens as thom_make_tokens,
    SCRIPT_RE as thom_SCRIPT,
    PUNCT_RE as thom_PUNCT,
)


# ============================================================================
# Basic per-language tokenization
# ============================================================================

class TestSyriacTokenization:
    def test_simple_tokens(self):
        toks = prov_tokenize("ܡܠܟܐ ܒܝܬܐ", "syriac")
        assert toks == ["ܡܠܟܐ", "ܒܝܬܐ"]

    def test_strips_non_syriac_chars(self):
        # English / Hebrew mixed in should disappear
        toks = prov_tokenize("ܡܠܟ Hello בית", "syriac")
        assert toks == ["ܡܠܟ"]

    def test_strips_vocalization(self):
        # ܳ is combining
        toks = prov_tokenize("ܡܳܠܟܳܐ", "syriac")
        assert toks == ["ܡܠܟܐ"]

    def test_strips_syriac_punctuation(self):
        # ܀ is Syriac end-of-paragraph (in punctuation range)
        toks = prov_tokenize("ܡܠܟܐ܀ ܒܝܬܐ", "syriac")
        assert toks == ["ܡܠܟܐ", "ܒܝܬܐ"]


class TestHebrewTokenization:
    def test_simple_tokens(self):
        toks = prov_tokenize("מלך בית", "hebrew")
        assert toks == ["מלך", "בית"]

    def test_strips_niqqud(self):
        toks = prov_tokenize("מַלְכָּא", "hebrew")
        # All niqqud combining marks gone
        assert toks == ["מלכא"]

    def test_strips_maqqef(self):
        # ־ is maqqef (Hebrew punctuation)
        toks = prov_tokenize("מלך־בית", "hebrew")
        assert toks == ["מלך", "בית"]


class TestArabicTokenization:
    def test_simple_tokens(self):
        toks = prov_tokenize("ملك بيت", "arabic")
        assert toks == ["ملك", "بيت"]

    def test_strips_tashkeel(self):
        toks = prov_tokenize("مَلِكٌ", "arabic")
        assert toks == ["ملك"]

    def test_strips_arabic_punctuation(self):
        toks = prov_tokenize("ملك، بيت", "arabic")
        assert toks == ["ملك", "بيت"]


class TestGreekTokenization:
    def test_simple_tokens_lowercased(self):
        toks = prov_tokenize("Λόγος Θεός", "greek")
        # Greek lowercases
        assert all(t.islower() for t in toks)

    def test_strips_accents(self):
        toks = prov_tokenize("λόγος", "greek")
        assert toks == ["λογος"]

    def test_strips_greek_punctuation(self):
        toks = prov_tokenize("λόγος· θεός.", "greek")
        assert toks == ["λογος", "θεος"]


class TestAramaicTokenization:
    """Aramaic uses Hebrew-script: same regex as Hebrew (intentional —
    documented in scripts/proverbs_permutation_test.py)."""

    def test_simple_tokens(self):
        toks = prov_tokenize("מלכא ביתא", "aramaic")
        assert toks == ["מלכא", "ביתא"]

    def test_strips_niqqud_same_as_hebrew(self):
        toks_aram = prov_tokenize("מַלְכָּא", "aramaic")
        toks_heb = prov_tokenize("מַלְכָּא", "hebrew")
        assert toks_aram == toks_heb


# ============================================================================
# CROSS-SCRIPT PARITY — the SEDRA bug class
# ============================================================================

class TestCrossScriptParity:
    """Different pipeline scripts (Proverbs, Q, Thomas) tokenize the SAME
    input text the SAME way for each language. If this fails, the
    cross-corpus comparisons are biased."""

    SAMPLES = {
        "syriac":  "ܡܠܟܳܐ ܒܝܬܳܐ ܢܘܪܐ",
        "hebrew":  "מַלְכָּא בֵּית נוּר",
        "arabic":  "مَلِكٌ بَيْتٌ نُورٌ",
        "greek":   "λόγος θεός φῶς",
    }

    @pytest.mark.parametrize("lang", SAMPLES.keys())
    def test_prov_and_q_agree(self, lang):
        text = self.SAMPLES[lang]
        assert prov_tokenize(text, lang) == q_tokenize(text, lang), \
            f"prov vs q disagree on {lang!r}"

    @pytest.mark.parametrize("lang", SAMPLES.keys())
    def test_prov_and_thom_agree(self, lang):
        text = self.SAMPLES[lang]
        assert prov_tokenize(text, lang) == thom_tokenize(text, lang), \
            f"prov vs thomas disagree on {lang!r}"

    @pytest.mark.parametrize("lang", SAMPLES.keys())
    def test_q_and_thom_agree(self, lang):
        text = self.SAMPLES[lang]
        assert q_tokenize(text, lang) == thom_tokenize(text, lang), \
            f"q vs thomas disagree on {lang!r}"


class TestScriptRegexParity:
    """SCRIPT_RE entries that exist in multiple scripts must match."""

    @pytest.mark.parametrize("lang", ["syriac", "hebrew", "arabic", "greek"])
    def test_script_re_pattern_identical(self, lang):
        assert prov_SCRIPT[lang].pattern == q_SCRIPT[lang].pattern, \
            f"SCRIPT_RE differs between prov and q for {lang}"
        assert prov_SCRIPT[lang].pattern == thom_SCRIPT[lang].pattern, \
            f"SCRIPT_RE differs between prov and thomas for {lang}"

    @pytest.mark.parametrize("lang", ["syriac", "hebrew", "arabic", "greek"])
    def test_punct_re_pattern_identical(self, lang):
        # Bug fix 2026-05-11: Proverbs and Q both had ʹ (modifier letter
        # prime) instead of ʹ (Greek numeral sign) and were missing
        # ; (Greek question mark). Thomas had it right. All three
        # scripts are now harmonised to match Thomas.
        assert prov_PUNCT[lang].pattern == q_PUNCT[lang].pattern, \
            f"PUNCT_RE differs between prov and q for {lang}"
        assert prov_PUNCT[lang].pattern == thom_PUNCT[lang].pattern, \
            f"PUNCT_RE differs between prov and thomas for {lang}"


# ============================================================================
# make_tokens asymmetry audit — documents the SEDRA case explicitly
# ============================================================================

class TestMakeTokensAsymmetry:
    """The ONE documented asymmetry is Thomas's make_tokens applying SEDRA
    to Syriac. Proverbs and Q do not. This test pins that down so a future
    change either keeps the asymmetry deliberate or removes it deliberately."""

    SAMPLE = {
        "syriac": "ܡܠܟܳܐ",
        "hebrew": "מַלְכָּא",
        "arabic": "مَلِك",
        "greek":  "λόγος",
    }

    @pytest.mark.parametrize("lang", SAMPLE.keys())
    def test_prov_and_q_make_tokens_identical(self, lang):
        text = self.SAMPLE[lang]
        p_toks = prov_make_tokens(text, lang)
        q_toks = q_make_tokens(text, lang)
        assert [t["lemma"] for t in p_toks] == [t["lemma"] for t in q_toks], \
            f"prov vs q make_tokens differ for {lang}"
        assert [t["form"] for t in p_toks] == [t["form"] for t in q_toks]

    def test_thomas_syriac_uses_sedra_when_provided(self):
        # Thomas's make_tokens accepts a sedra parameter for Syriac. With
        # sedra=None it should match the Proverbs/Q surface-form behavior.
        text = "ܡܠܟܐ"
        thom_no_sedra = thom_make_tokens(text, "syriac", sedra=None)
        prov = prov_make_tokens(text, "syriac")
        # With sedra=None Thomas should use surface form (matches Proverbs/Q)
        assert [t["lemma"] for t in thom_no_sedra] == \
               [t["lemma"] for t in prov], (
            "Thomas make_tokens(sedra=None) should equal Proverbs for Syriac")

    def test_thomas_nonsyriac_ignores_sedra(self):
        # For non-Syriac, sedra parameter must have no effect
        text = "λόγος"
        thom_no = thom_make_tokens(text, "greek", sedra=None)
        thom_with = thom_make_tokens(text, "greek", sedra={"fake": "lookup"})
        assert thom_no == thom_with

    def test_thomas_syriac_sedra_collapses_known_form(self):
        # If SEDRA maps form X to lemma Y, then make_tokens should return Y
        text = "ܡܠܟ"
        thom_with = thom_make_tokens(text, "syriac",
                                        sedra={"ܡܠܟ": "ROOT_MLK"})
        assert thom_with[0]["lemma"] == "ROOT_MLK"
        assert thom_with[0]["form"] == "ܡܠܟ"


# ============================================================================
# Token contract — what every token dict must contain
# ============================================================================

class TestTokenSchema:
    """The downstream detector reads tok['lemma'] and tok['parse']. Every
    pipeline's make_tokens must produce both keys."""

    @pytest.mark.parametrize("lang", ["syriac", "hebrew", "arabic", "greek"])
    def test_prov_token_schema(self, lang):
        toks = prov_make_tokens("test ܡܠܟ מלך ملك λόγος", lang)
        for t in toks:
            assert "lemma" in t
            assert "form" in t
            assert "parse" in t

    @pytest.mark.parametrize("lang", ["syriac", "hebrew", "arabic", "greek"])
    def test_q_token_schema(self, lang):
        toks = q_make_tokens("test ܡܠܟ מלך ملك λόγος", lang)
        for t in toks:
            assert "lemma" in t
            assert "form" in t
            assert "parse" in t

    @pytest.mark.parametrize("lang", ["syriac", "hebrew", "arabic", "greek"])
    def test_thom_token_schema(self, lang):
        toks = thom_make_tokens("test ܡܠܟ מלך ملك λόγος", lang, None)
        for t in toks:
            assert "lemma" in t
            assert "form" in t
            assert "parse" in t


# ============================================================================
# Edge cases
# ============================================================================

class TestTokenizationEdgeCases:
    def test_empty_string(self):
        for lang in ["syriac", "hebrew", "arabic", "greek", "aramaic"]:
            assert prov_tokenize("", lang) == []

    def test_only_punctuation(self):
        # Should yield empty token list
        assert prov_tokenize("܀ ܀", "syriac") == []
        assert prov_tokenize("، ،", "arabic") == []

    def test_only_whitespace(self):
        assert prov_tokenize("   \t\n  ", "syriac") == []

    def test_only_other_script(self):
        # Hebrew text passed with lang=syriac → all stripped
        assert prov_tokenize("מלך מלכא", "syriac") == []
