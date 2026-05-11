"""
END-TO-END test with planted-truth catchwords.

Build a synthetic micro-corpus where we PLANT a known pattern of recurring
catchwords across specific adjacent boundaries. Then run the full pipeline
(tokenize → make_tokens → blocked-lemma filter → matrix → permutation test)
and verify that:

  (a) the detector flags the planted catchwords;
  (b) the permutation test recovers the planted arrangement (z >> 0);
  (c) a shuffled control corpus shows z ≈ 0.

If any of these fails, the pipeline is broken.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from scripts.proverbs_permutation_test import (
    make_tokens, compute_blocked, precompute_matrix,
    stats_for_order, run_permutation,
)
from scripts.perrin_test_one import (
    total_count_for_order, run_perm_total,
)


# ============================================================================
# Hebrew planted-corpus
# ============================================================================

def build_planted_hebrew_corpus(n_units: int = 30, n_planted_pairs: int = 5,
                                  seed: int = 42) -> dict[int, str]:
    """Plant n_planted_pairs DISTINCT lemma pairs at TRUE adjacent boundaries
    where each verse has unique random filler vocabulary.

    Pattern: pair k = (PLANTk_A, PLANTk_B). Place PLANTk_A in verse (2k)
    and PLANTk_B in verse (2k+1). Pairs are unique per planted boundary.

    Goal: each TRUE adjacent boundary (2k, 2k+1) has exactly one planted
    catchword. Random shuffles scatter these → fewer adjacencies retain
    the planted catchwords. The TOTAL-COUNT statistic should reject null.

    Each verse also gets ~6 random filler words drawn from a large pool to
    ensure the planted lemmas aren't blocked by the filter_pct=80 step.
    """
    rng = random.Random(seed)
    # Large diverse pool so no individual filler word recurs across many verses
    pool = [
        "מלך", "בית", "ספר", "יום", "לילה", "ים", "ארץ", "שמש",
        "ירח", "כוכב", "עץ", "פרי", "מים", "אדם", "אישה", "ילד",
        "סוס", "כלב", "צור", "גבעה", "נחל", "שדה", "כרם", "זית",
        "תאנה", "רימון", "תמר", "ורד", "שושנה", "סלע", "מדבר", "גשם",
        "אש", "אור", "חושך", "רוח", "אבן", "חול", "שלום", "מלחמה",
        "אחד", "שתי", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה",
    ]
    verses = {}
    for i in range(n_units):
        words = rng.sample(pool, 6)  # SAMPLE without replacement
        verses[i] = " ".join(words)

    # Each planted pair must be a UNIQUE Hebrew string — using digits in
    # the lemma was a mistake because the Hebrew SCRIPT_RE strips digits,
    # collapsing all unique plants into a single lemma. Use distinct
    # Hebrew root-like strings instead.
    unique_plants = [
        "אבקדה", "ידלמן", "סעפצק", "רשתחז", "טכלמנס",
        "עפצקרש", "תחזטכל", "מנסעפ", "צקרשתח", "זטכלמ",
    ][:n_planted_pairs]
    for k in range(n_planted_pairs):
        a, b = 2 * k, 2 * k + 1
        if b >= n_units:
            break
        plant = unique_plants[k]
        verses[a] = plant + " " + verses[a]
        verses[b] = plant + " " + verses[b]
    return verses


# ============================================================================
# Tests
# ============================================================================

class TestSyntheticPlantedHebrew:
    @pytest.fixture
    def planted_corpus(self):
        # Use a larger N (more shuffle entropy) so 6 unique planted pairs
        # at TRUE boundaries give clean rejection of null.
        return build_planted_hebrew_corpus(n_units=50,
                                              n_planted_pairs=6, seed=42)

    @pytest.fixture
    def shuffled_corpus(self, planted_corpus):
        """Same VERSES but in shuffled order — so the planted pairs no longer
        sit at adjacent positions."""
        rng = random.Random(99)
        verses = list(planted_corpus.values())
        rng.shuffle(verses)
        return {i: v for i, v in enumerate(verses)}

    def _build_matrix(self, corpus):
        toks = {i: make_tokens(corpus[i], "hebrew") for i in corpus}
        ids = sorted(toks.keys())
        blocked = compute_blocked(toks, 80.0)
        m = precompute_matrix(toks, blocked, "hebrew", ids)
        return m, ids

    def test_planted_pair_detected_at_boundaries(self, planted_corpus):
        """Each unique planted lemma appears at boundary (2N, 2N+1)."""
        m, ids = self._build_matrix(planted_corpus)
        plants = ["אבקדה", "ידלמן", "סעפצק", "רשתחז", "טכלמנס", "עפצקרש"]
        planted_seen = 0
        for k in range(6):
            i, j = 2 * k, 2 * k + 1
            cell = m.get((i, j), frozenset())
            plant_lemma = plants[k]
            if any(a == plant_lemma and b == plant_lemma
                   for a, b, _ in cell):
                planted_seen += 1
        assert planted_seen == 6, \
            f"planted unique pair at each TRUE boundary, got {planted_seen}/6"

    def test_total_count_rejects_null_on_planted(self, planted_corpus):
        """The TOTAL-count statistic (no recurrence filter) is the more
        sensitive test — it directly reflects "more catchwords at adjacent
        boundaries in TRUE order than at random adjacencies"."""
        m, ids = self._build_matrix(planted_corpus)
        # Convert to perrin_test_one's matrix format (list of tuples)
        m_list = {k: list(v) for k, v in m.items()}
        true_t = total_count_for_order(ids, m_list, None)
        null = run_perm_total(m_list, ids, None, n_perms=1000, seed=42)
        nm, ns = float(null.mean()), float(null.std())
        z = (true_t - nm) / max(ns, 1e-9)
        p = float((null >= true_t).mean())
        print(f"\n  Planted TOTAL: true={true_t}, null mean={nm:.1f}±{ns:.1f}, "
              f"z={z:.2f}, p={p:.4f}")
        assert p < 0.05, f"planted arrangement should reject null, got p={p}"
        assert z > 1.5, f"planted should give z > 1.5, got z={z}"

    def test_total_count_planted_vs_shuffled(self, planted_corpus,
                                                shuffled_corpus):
        """Planted true-order corpus must show stronger signal than
        the same VERSES shuffled. This is the cleanest verification:
        same verses, different orderings, different signals."""
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

        print(f"\n  Planted z={z_p:.2f}  Shuffled z={z_s:.2f}")
        assert z_p > z_s, \
            f"planted z ({z_p:.2f}) should exceed shuffled z ({z_s:.2f})"
        assert z_p > 1.5


# ============================================================================
# Empty/degenerate corpora — verify pipeline doesn't crash
# ============================================================================

class TestPipelineRobustness:
    def test_empty_corpus_does_not_crash(self):
        toks = {}
        blocked = compute_blocked(toks, 80.0)
        m = precompute_matrix(toks, blocked, "hebrew", [])
        s = stats_for_order([], m, [2])
        assert s["recurring_2plus"] == 0
        assert s["max_freq"] == 0

    def test_single_unit_corpus(self):
        toks = {0: make_tokens("מלך בית", "hebrew")}
        blocked = compute_blocked(toks, 80.0)
        m = precompute_matrix(toks, blocked, "hebrew", [0])
        s = stats_for_order([0], m, [2])
        # No boundaries → no catchwords
        assert s["recurring_2plus"] == 0

    def test_two_unit_corpus_no_overlap(self):
        toks = {
            0: make_tokens("מלך בית", "hebrew"),
            1: make_tokens("ספר יום", "hebrew"),
        }
        blocked = compute_blocked(toks, 80.0)
        m = precompute_matrix(toks, blocked, "hebrew", [0, 1])
        s = stats_for_order([0, 1], m, [2])
        # One boundary, no overlapping lemmas → 0 catchwords (and no recurrence)
        assert s["recurring_2plus"] == 0


# ============================================================================
# Sanity: shuffling preserves total catchword count
# ============================================================================

class TestShufflingInvariants:
    def test_total_catchwords_in_all_cells_invariant_to_order(self):
        """The MATRIX cells don't change; only which ones contribute to the
        statistic. So summing across all i,j is independent of ordering."""
        corpus = build_planted_hebrew_corpus(20, 6, seed=1)
        toks = {i: make_tokens(corpus[i], "hebrew") for i in corpus}
        ids = sorted(toks.keys())
        m = precompute_matrix(toks, compute_blocked(toks, 80.0), "hebrew", ids)
        total = sum(len(v) for v in m.values())
        # Now shuffle ids — total in matrix unchanged
        shuf = ids.copy()
        random.Random(0).shuffle(shuf)
        m2 = precompute_matrix(toks, compute_blocked(toks, 80.0), "hebrew", shuf)
        total2 = sum(len(v) for v in m2.values())
        # Matrix-build order shouldn't affect total cell-count
        assert total == total2
