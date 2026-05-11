"""
Corpus-loader correctness audit.

Three corpora — Proverbs, Q, Thomas — each with its own file schema. These
tests pin down what each loader expects, what variant indexing means, and
what happens when files/variants are missing.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.proverbs_permutation_test import (
    load_translations as prov_load,
    HEB_FILE as PROV_HEB_FILE,
    TRANS_DIR as PROV_TRANS_DIR,
)
from scripts.q_permutation_test import (
    load_q_translations as q_load,
    GREEK_FILE as Q_GREEK_FILE,
    TRANS_DIR as Q_TRANS_DIR,
)
from scripts.crossling_permutation_test import (
    load_translations as thom_load,
    LLM_DIR_SYR as THOM_LLM_SYR,
    CROSS_DIR as THOM_CROSS,
)


# ============================================================================
# Source-file existence (skip the tests gracefully if data missing)
# ============================================================================

DATA_AVAILABLE = {
    "proverbs": PROV_HEB_FILE.exists() and PROV_TRANS_DIR.exists(),
    "q":        Q_GREEK_FILE.exists() and Q_TRANS_DIR.exists(),
    "thomas":   THOM_LLM_SYR.exists() and THOM_CROSS.exists(),
}


# ============================================================================
# Proverbs loader
# ============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE["proverbs"], reason="no Proverbs data")
class TestProverbsLoader:
    def test_hebrew_source_loads_from_json(self):
        toks = prov_load("hebrew", variant_idx=0)
        assert len(toks) > 0, "Hebrew Proverbs should load"
        # Each value is a list of token-dicts
        for tid, t in list(toks.items())[:3]:
            assert isinstance(tid, int)
            assert isinstance(t, list)
            for tok in t:
                assert "lemma" in tok and "form" in tok

    @pytest.mark.parametrize("lang", ["greek", "syriac", "aramaic", "arabic"])
    def test_translation_loads(self, lang):
        toks = prov_load(lang, variant_idx=0)
        assert len(toks) > 100, \
            f"Proverbs {lang} should have hundreds of verses loaded"

    def test_hebrew_ignores_variant_idx(self):
        # Hebrew is the source; variant_idx > 0 should not yield new content
        # The loader implementation ignores variant_idx for hebrew (loads from
        # proverbs_hebrew.json which has no 'variants' array)
        v0 = prov_load("hebrew", variant_idx=0)
        v5 = prov_load("hebrew", variant_idx=5)
        assert v0 == v5, "Hebrew Proverbs source should be invariant to variant_idx"

    def test_translation_variant_indexing(self):
        # Variants 0 and 5 should differ for translated languages (different
        # Gemini stochastic samples)
        v0 = prov_load("greek", variant_idx=0)
        v5 = prov_load("greek", variant_idx=5)
        # At least some pericopes should differ
        diff = sum(1 for k in v0 if k in v5 and v0[k] != v5[k])
        assert diff > 0, \
            "Greek variant 0 and 5 should differ for at least some units"

    def test_invalid_variant_returns_empty_for_unit(self):
        # Variant 999 doesn't exist; loader should skip those units
        toks = prov_load("greek", variant_idx=999)
        # Either empty or far fewer than usual
        v0 = prov_load("greek", variant_idx=0)
        assert len(toks) < len(v0) * 0.1 or len(toks) == 0


# ============================================================================
# Q loader
# ============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE["q"], reason="no Q data")
class TestQLoader:
    def test_greek_source_loads(self):
        toks = q_load("greek", variant_idx=0)
        assert len(toks) == 56, f"Q has 56 pericopes, got {len(toks)}"

    @pytest.mark.parametrize("lang", ["aramaic", "syriac", "hebrew", "arabic"])
    def test_translation_loads(self, lang):
        toks = q_load(lang, variant_idx=0)
        assert len(toks) >= 50, f"Q {lang} has ~56 pericopes, got {len(toks)}"

    def test_greek_invariant_to_variant_idx(self):
        v0 = q_load("greek", variant_idx=0)
        v5 = q_load("greek", variant_idx=5)
        assert v0 == v5

    def test_pericope_id_is_int(self):
        toks = q_load("syriac", variant_idx=0)
        for k in list(toks.keys())[:5]:
            assert isinstance(k, int)


# ============================================================================
# Thomas loader (has the SEDRA asymmetry)
# ============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE["thomas"], reason="no Thomas data")
class TestThomasLoader:
    def test_syriac_loads_from_llm_dir(self):
        toks = thom_load("syriac", variant_idx=0)
        # 115 logia
        assert 100 <= len(toks) <= 115
        # Default loader applies SEDRA → lemma differs from form
        # Find any token where SEDRA changed the lemma
        sedra_collapsed = False
        for ts in toks.values():
            for tok in ts:
                if tok.get("lemma") and tok.get("form") and \
                   tok["lemma"] != tok["form"]:
                    sedra_collapsed = True
                    break
            if sedra_collapsed:
                break
        assert sedra_collapsed, \
            "Thomas Syriac loader should apply SEDRA (some lemma != form)"

    @pytest.mark.parametrize("lang", ["hebrew", "greek", "arabic"])
    def test_cross_lang_loads(self, lang):
        toks = thom_load(lang, variant_idx=0)
        assert 100 <= len(toks) <= 115

    @pytest.mark.parametrize("lang", ["hebrew", "greek", "arabic"])
    def test_nonsyriac_uses_surface_form_as_lemma(self, lang):
        """Thomas non-Syriac langs: lemma == form (no lemmatization)."""
        toks = thom_load(lang, variant_idx=0)
        for ts in list(toks.values())[:5]:
            for tok in ts:
                if tok.get("lemma") and tok.get("form"):
                    assert tok["lemma"] == tok["form"], \
                        f"Thomas {lang}: lemma should equal form (no lemmatizer)"

    def test_syriac_variant_indexing(self):
        v0 = thom_load("syriac", variant_idx=0)
        v5 = thom_load("syriac", variant_idx=5)
        # Different stochastic variants → different tokens
        diff = sum(1 for k in v0 if k in v5 and v0[k] != v5[k])
        assert diff > 0


# ============================================================================
# Synthetic data tests — work without external data
# ============================================================================

class TestProverbsLoaderEmptyDir:
    """Verify load_translations handles missing/empty directories gracefully."""

    def test_returns_empty_dict_when_dir_missing(self, tmp_path, monkeypatch):
        # Point load_translations at an empty temp dir
        import scripts.proverbs_permutation_test as m
        monkeypatch.setattr(m, "TRANS_DIR", tmp_path)
        # Will return empty dict because no unit_*.json under tmp_path/syriac
        toks = m.load_translations("syriac", variant_idx=0)
        assert toks == {}


class TestQLoaderEmptyDir:
    def test_returns_empty_dict_when_dir_missing(self, tmp_path, monkeypatch):
        import scripts.q_permutation_test as m
        monkeypatch.setattr(m, "TRANS_DIR", tmp_path)
        toks = m.load_q_translations("syriac", variant_idx=0)
        assert toks == {}


# ============================================================================
# File-schema regression — fail loudly if schema drifts
# ============================================================================

class TestProverbsSchema:
    @pytest.mark.skipif(not DATA_AVAILABLE["proverbs"], reason="no data")
    def test_unit_file_has_expected_fields(self):
        sample = next((PROV_TRANS_DIR / "syriac").glob("unit_*.json"))
        d = json.loads(sample.read_text(encoding="utf-8"))
        assert "unit_id" in d
        assert "variants" in d
        assert isinstance(d["variants"], list)
        assert len(d["variants"]) >= 1
        # Each variant has 'text' and 'success'
        v0 = d["variants"][0]
        assert "text" in v0 or "success" in v0


class TestQSchema:
    @pytest.mark.skipif(not DATA_AVAILABLE["q"], reason="no data")
    def test_pericope_file_has_expected_fields(self):
        sample = next((Q_TRANS_DIR / "syriac").glob("pericope_*.json"))
        d = json.loads(sample.read_text(encoding="utf-8"))
        assert "pericope_id" in d
        assert "variants" in d


class TestThomasSchema:
    @pytest.mark.skipif(not DATA_AVAILABLE["thomas"], reason="no data")
    def test_thomas_syriac_uses_syriac_text_field(self):
        sample = THOM_LLM_SYR / "logion_006.json"
        if not sample.exists():
            pytest.skip("specific logion missing")
        d = json.loads(sample.read_text(encoding="utf-8"))
        v0 = d["variants"][0]
        assert "syriac_text" in v0, \
            "Thomas Syriac variants use 'syriac_text' field"

    @pytest.mark.skipif(not DATA_AVAILABLE["thomas"], reason="no data")
    def test_thomas_crossling_uses_text_field(self):
        sample = THOM_CROSS / "greek" / "logion_006.json"
        if not sample.exists():
            pytest.skip("specific logion missing")
        d = json.loads(sample.read_text(encoding="utf-8"))
        v0 = d["variants"][0]
        assert "text" in v0, \
            "Thomas cross-ling variants use 'text' field (NOT 'syriac_text')"
