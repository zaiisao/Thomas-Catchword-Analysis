"""
End-to-end synthetic planted-truth test for the Thomas pipeline.

Thomas's `make_tokens` is the one that does the SEDRA asymmetry. We test
the pipeline in BOTH modes:

  - sedra=None (surface forms, apples-to-apples with Hebrew/Greek/Arabic).
  - sedra=fake (small mock SEDRA dict, exercises the lemma-collapse path).

In both modes, planted unique catchwords at TRUE adjacent boundaries should
be detected by the pipeline. The permutation test should reject null.
"""
from __future__ import annotations

import random

import pytest

from scripts.crossling_permutation_test import (
    make_tokens as thom_make_tokens,
    compute_blocked as thom_compute_blocked,
    precompute_matrix as thom_precompute_matrix,
    N_LOGIA as THOM_N_LOGIA,
)
from scripts.perrin_test_one import (
    total_count_for_order, run_perm_total,
)


# Override N_LOGIA for the test (we use a smaller synthetic corpus)
# The Thomas precompute_matrix uses N_LOGIA from module — we work around by
# building the matrix ourselves.

from phase1_montecarlo.catchword_detector import CatchwordDetector
from phase1_montecarlo.language_data import get_profile


def _unique_syriac_fillers(n_units: int, n_per_verse: int, seed: int):
    """Generate UNIQUE filler tokens per verse — same approach as the Q
    synthetic test. Uses pseudo-Syriac strings constructed deterministically."""
    SYRIAC_LETTERS = "ܐܒܓܕܗܘܙܚܛܝܟܠܡܢܣܥܦܨܩܪܫܬ"
    rng = random.Random(seed)
    used = set()
    out = {}
    for i in range(n_units):
        verse_fillers = []
        while len(verse_fillers) < n_per_verse:
            n = rng.randint(5, 10)
            w = "".join(rng.choice(SYRIAC_LETTERS) for _ in range(n))
            if w not in used:
                used.add(w)
                verse_fillers.append(w)
        out[i] = verse_fillers
    return out


def build_planted_syriac_corpus(n_units: int = 50,
                                   n_planted_pairs: int = 6,
                                   seed: int = 42) -> dict[int, str]:
    """Plant n_planted_pairs unique pseudo-Syriac lemmas at TRUE adjacent
    boundaries with UNIQUE filler per verse (pure signal, no noise)."""
    fillers = _unique_syriac_fillers(n_units, 6, seed)
    verses = {i: " ".join(fillers[i]) for i in range(n_units)}

    unique_plants = ["ܩܘܕܫܐ", "ܚܘܒܐ", "ܫܠܡܐ", "ܚܝܐ", "ܡܘܬܐ", "ܚܟܡܬܐ"]
    for k in range(min(n_planted_pairs, len(unique_plants))):
        a, b = 2 * k, 2 * k + 1
        if b >= n_units:
            break
        verses[a] = unique_plants[k] + " " + verses[a]
        verses[b] = unique_plants[k] + " " + verses[b]
    return verses


def _build_matrix_custom(toks_map, blocked, ids, threshold=0.6):
    """Local matrix builder that doesn't depend on Thomas's hardcoded N_LOGIA."""
    det = CatchwordDetector("syriac",
                              phonological_threshold=threshold,
                              require_content_pos=False)
    filtered = {i: [t for t in toks_map.get(i, []) if t["lemma"] not in blocked]
                for i in ids}
    matrix = {}
    for i in ids:
        for j in ids:
            if i == j: continue
            ta, tb = filtered[i], filtered[j]
            if not ta or not tb:
                matrix[(i, j)] = []
                continue
            cws = det.detect(ta, tb)
            matrix[(i, j)] = [(c.token_a["lemma"], c.token_b["lemma"],
                                 c.link_type) for c in cws]
    return matrix


class TestThomasSyntheticSurface:
    """Thomas pipeline with sedra=None (surface forms — the corrected mode)."""

    @pytest.fixture
    def planted_corpus(self):
        return build_planted_syriac_corpus(n_units=50, n_planted_pairs=6,
                                              seed=42)

    def _toks(self, corpus):
        return {i: thom_make_tokens(corpus[i], "syriac", sedra=None)
                for i in corpus}

    def test_planted_pair_detected(self, planted_corpus):
        toks = self._toks(planted_corpus)
        ids = sorted(toks.keys())
        blocked = thom_compute_blocked(toks, 80.0)
        m = _build_matrix_custom(toks, blocked, ids)
        plants = ["ܩܘܕܫܐ", "ܚܘܒܐ", "ܫܠܡܐ", "ܚܝܐ", "ܡܘܬܐ", "ܚܟܡܬܐ"]
        seen = 0
        for k in range(6):
            i, j = 2 * k, 2 * k + 1
            cell = m.get((i, j), [])
            if any(a == plants[k] and b == plants[k] for a, b, _ in cell):
                seen += 1
        assert seen == 6, f"surface-form Thomas should find all 6 plants, got {seen}/6"

    def test_total_count_rejects_null(self, planted_corpus):
        toks = self._toks(planted_corpus)
        ids = sorted(toks.keys())
        blocked = thom_compute_blocked(toks, 80.0)
        m = _build_matrix_custom(toks, blocked, ids)
        true_t = total_count_for_order(ids, m, None)
        null = run_perm_total(m, ids, None, n_perms=1000, seed=42)
        z = (true_t - null.mean()) / max(null.std(), 1e-9)
        p = float((null >= true_t).mean())
        assert p < 0.05, f"surface-form Thomas planted should reject null, got p={p}"
        assert z > 1.5


class TestThomasSyntheticSedra:
    """Thomas pipeline with a small mock SEDRA dict — exercises the
    lemma-collapse path so we can pin its behavior."""

    @pytest.fixture
    def planted_corpus(self):
        return build_planted_syriac_corpus(n_units=50, n_planted_pairs=6,
                                              seed=42)

    @pytest.fixture
    def fake_sedra(self, planted_corpus):
        # Pick a real filler token from the corpus and map it to a fake root
        sample_token = None
        for verse in planted_corpus.values():
            tokens = verse.split()
            for t in tokens:
                if t not in ["ܩܘܕܫܐ", "ܚܘܒܐ", "ܫܠܡܐ", "ܚܝܐ",
                              "ܡܘܬܐ", "ܚܟܡܬܐ"]:
                    sample_token = t
                    break
            if sample_token:
                break
        return {sample_token: "FAKE_ROOT_X"} if sample_token else {}

    def test_sedra_collapses_known_form(self, planted_corpus, fake_sedra):
        # fake_sedra contains ONE filler-token-to-fake-root mapping.
        # Verify the collapse actually applies.
        if not fake_sedra:
            pytest.skip("test setup: empty fake_sedra")
        key = next(iter(fake_sedra))
        toks = {i: thom_make_tokens(planted_corpus[i], "syriac",
                                       sedra=fake_sedra)
                for i in planted_corpus}
        found_collapsed = False
        for ts in toks.values():
            for t in ts:
                if t["form"] == key:
                    assert t["lemma"] == "FAKE_ROOT_X", \
                        f"SEDRA collapse expected for {key!r}, got {t['lemma']!r}"
                    found_collapsed = True
        assert found_collapsed, \
            f"the chosen key {key!r} should appear at least once in the corpus"

    def test_sedra_planted_still_detected(self, planted_corpus, fake_sedra):
        # Even with SEDRA active, our unique-Hebrew-style plants should
        # not be in the fake SEDRA → they keep surface form → still detected
        toks = {i: thom_make_tokens(planted_corpus[i], "syriac",
                                       sedra=fake_sedra)
                for i in planted_corpus}
        ids = sorted(toks.keys())
        blocked = thom_compute_blocked(toks, 80.0)
        m = _build_matrix_custom(toks, blocked, ids)
        true_t = total_count_for_order(ids, m, None)
        null = run_perm_total(m, ids, None, n_perms=1000, seed=42)
        z = (true_t - null.mean()) / max(null.std(), 1e-9)
        # Plants not in SEDRA dict → surface form preserved → still detected
        assert z > 1.5, f"SEDRA-mode Thomas planted should still reject null, got z={z}"
