"""
End-to-end synthetic planted-truth test for the Q pipeline.

Mirrors `test_synthetic_planted.py` but uses Q's `make_tokens` /
`load_q_translations` / `compute_blocked` / `precompute_matrix` /
`run_permutation` functions. If the Q pipeline is wired correctly,
planted catchwords at TRUE adjacent boundaries should produce z > 1.5
while random orderings give z ≈ 0.
"""
from __future__ import annotations

import random

import pytest

from scripts.q_permutation_test import (
    make_tokens as q_make_tokens,
    compute_blocked as q_compute_blocked,
    precompute_matrix as q_precompute_matrix,
    stats_for_order as q_stats_for_order,
    run_permutation as q_run_permutation,
)
from scripts.perrin_test_one import (
    total_count_for_order, run_perm_total,
)


def _unique_greek_fillers(n_units: int, n_per_verse: int, seed: int) -> dict[int, list[str]]:
    """Generate UNIQUE filler tokens per verse — no overlap between any two
    verses. Uses pseudo-Greek lemmas constructed deterministically so the
    test is reproducible."""
    GREEK_ALPHABET = "αβγδεζηθικλμνξοπρστυφχψω"
    rng = random.Random(seed)
    used = set()
    out = {}
    for i in range(n_units):
        verse_fillers = []
        while len(verse_fillers) < n_per_verse:
            # Random 5-10 letter pseudo-Greek string
            n = rng.randint(5, 10)
            w = "".join(rng.choice(GREEK_ALPHABET) for _ in range(n))
            if w not in used:
                used.add(w)
                verse_fillers.append(w)
        out[i] = verse_fillers
    return out


def build_planted_greek_corpus(n_units: int = 50,
                                  n_planted_pairs: int = 6,
                                  seed: int = 42) -> dict[int, str]:
    """Plant n_planted_pairs distinct Greek lemma pairs at TRUE adjacent
    boundaries. Each verse gets UNIQUE fillers so the only shared catchwords
    are the planted ones (pure signal, no filler noise).

    Q pipeline runs Greek as source and uses surface forms (no lemmatizer).
    """
    fillers = _unique_greek_fillers(n_units, 6, seed)
    verses = {i: " ".join(fillers[i]) for i in range(n_units)}

    unique_plants = ["καταδρομος", "ιεροτελης", "παμμεγας", "φιλοθεος",
                     "ευφημος", "αρχομαι", "βεβαιοω", "γαληνη",
                     "δουλευω", "εγρηγορα"][:n_planted_pairs]
    for k in range(n_planted_pairs):
        a, b = 2 * k, 2 * k + 1
        if b >= n_units:
            break
        verses[a] = unique_plants[k] + " " + verses[a]
        verses[b] = unique_plants[k] + " " + verses[b]
    return verses


class TestQSyntheticPlanted:
    @pytest.fixture
    def planted_corpus(self):
        return build_planted_greek_corpus(n_units=50, n_planted_pairs=6,
                                              seed=42)

    @pytest.fixture
    def shuffled_corpus(self, planted_corpus):
        rng = random.Random(99)
        verses = list(planted_corpus.values())
        rng.shuffle(verses)
        return {i: v for i, v in enumerate(verses)}

    def _build_matrix(self, corpus):
        toks = {i: q_make_tokens(corpus[i], "greek") for i in corpus}
        ids = sorted(toks.keys())
        blocked = q_compute_blocked(toks, 80.0)
        m = q_precompute_matrix(toks, blocked, "greek", ids)
        return m, ids

    def test_planted_pair_detected_at_boundaries(self, planted_corpus):
        m, ids = self._build_matrix(planted_corpus)
        plants = ["καταδρομος", "ιεροτελης", "παμμεγας", "φιλοθεος",
                  "ευφημος", "αρχομαι"]
        planted_seen = 0
        for k in range(6):
            i, j = 2 * k, 2 * k + 1
            cell = m.get((i, j), frozenset())
            # Greek tokenization lowercases — check lowercase form
            plant_lc = plants[k].lower()
            if any(a == plant_lc and b == plant_lc for a, b, _ in cell):
                planted_seen += 1
        assert planted_seen == 6, \
            f"Q pipeline should find planted pairs at all 6 boundaries, got {planted_seen}/6"

    def test_total_count_rejects_null_on_planted(self, planted_corpus):
        m, ids = self._build_matrix(planted_corpus)
        m_list = {k: list(v) for k, v in m.items()}
        true_t = total_count_for_order(ids, m_list, None)
        null = run_perm_total(m_list, ids, None, n_perms=1000, seed=42)
        z = (true_t - null.mean()) / max(null.std(), 1e-9)
        p = float((null >= true_t).mean())
        assert p < 0.05, f"Q planted should reject null, got p={p}"
        assert z > 1.5, f"Q planted should give z > 1.5, got z={z}"

    def test_planted_vs_shuffled(self, planted_corpus, shuffled_corpus):
        m_p, ids_p = self._build_matrix(planted_corpus)
        m_s, ids_s = self._build_matrix(shuffled_corpus)
        m_p_list = {k: list(v) for k, v in m_p.items()}
        m_s_list = {k: list(v) for k, v in m_s.items()}

        true_p = total_count_for_order(ids_p, m_p_list, None)
        true_s = total_count_for_order(ids_s, m_s_list, None)
        null_p = run_perm_total(m_p_list, ids_p, None, n_perms=1000, seed=42)
        null_s = run_perm_total(m_s_list, ids_s, None, n_perms=1000, seed=42)
        z_p = (true_p - null_p.mean()) / max(null_p.std(), 1e-9)
        z_s = (true_s - null_s.mean()) / max(null_s.std(), 1e-9)
        assert z_p > z_s, f"planted z {z_p:.2f} should exceed shuffled z {z_s:.2f}"
        assert z_p > 1.5
