"""
Permutation-test statistical core: reproducibility, correctness of the
recurring-≥k statistic, total-count statistic, filter-pct edge cases, and
null-distribution sanity.

Three different scripts implement variations of the permutation test. They
share the same statistical primitives. These tests pin those primitives
down with known-truth tiny matrices so any future drift is caught.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from scripts.proverbs_permutation_test import (
    compute_blocked,
    stats_for_order,
    run_permutation,
)
from scripts.perrin_test_one import (
    total_count_for_order,
    run_perm_total,
    FILTERS,
)
from scripts.phon_only_one import filter_matrix


# ============================================================================
# stats_for_order — counts pairs that recur at ≥k boundaries
# ============================================================================

class TestStatsForOrder:
    """Construct a tiny known matrix and verify recurrence counts."""

    def matrix_from(self, cells):
        """Helper: cells is dict[(i,j) -> list[tuple]]."""
        return {k: frozenset(v) for k, v in cells.items()}

    def test_zero_recurring_pairs(self):
        # Three logia, no overlapping catchword pairs at any boundary
        m = self.matrix_from({
            (0, 1): [("A", "B", "semantic")],
            (1, 2): [("C", "D", "semantic")],
        })
        s = stats_for_order([0, 1, 2], m, [2])
        assert s["recurring_2plus"] == 0
        assert s["max_freq"] == 1

    def test_one_recurring_pair(self):
        # Same pair appears at two adjacent boundaries
        m = self.matrix_from({
            (0, 1): [("A", "B", "semantic")],
            (1, 2): [("A", "B", "semantic")],
        })
        s = stats_for_order([0, 1, 2], m, [2])
        assert s["recurring_2plus"] == 1
        assert s["max_freq"] == 2

    def test_three_recurring_pairs(self):
        m = self.matrix_from({
            (0, 1): [("A", "B", "semantic"), ("C", "D", "phonological"),
                       ("E", "F", "semantic")],
            (1, 2): [("A", "B", "semantic"), ("C", "D", "phonological"),
                       ("E", "F", "semantic")],
        })
        s = stats_for_order([0, 1, 2], m, [2])
        assert s["recurring_2plus"] == 3
        assert s["max_freq"] == 2

    def test_recurring_at_three_boundaries(self):
        m = self.matrix_from({
            (0, 1): [("A", "B", "semantic")],
            (1, 2): [("A", "B", "semantic")],
            (2, 3): [("A", "B", "semantic")],
        })
        s = stats_for_order([0, 1, 2, 3], m, [2, 3])
        assert s["recurring_2plus"] == 1
        assert s["recurring_3plus"] == 1
        assert s["max_freq"] == 3

    def test_shuffled_order_reduces_recurrence(self):
        # In TRUE order, ABs recur at boundaries (0,1) and (1,2).
        # In SHUFFLED order [0, 2, 1], boundaries are (0,2) and (2,1) — only
        # one of which has the recurring pair, plus a non-recurring one
        m = self.matrix_from({
            (0, 1): [("A", "B", "semantic")],
            (1, 2): [("A", "B", "semantic")],
            (0, 2): [("X", "Y", "semantic")],
            (2, 0): [("X", "Y", "semantic")],
            (2, 1): [("A", "B", "semantic")],
            (1, 0): [("A", "B", "semantic")],
        })
        s_true = stats_for_order([0, 1, 2], m, [2])
        s_shuf = stats_for_order([0, 2, 1], m, [2])
        # True: AB recurs at (0,1) and (1,2) → 1 recurring pair
        # Shuf: looking at (0,2) → XY, (2,1) → AB → no recurrence
        assert s_true["recurring_2plus"] >= s_shuf["recurring_2plus"]


# ============================================================================
# total_count_for_order — counts ALL catchwords at adjacent boundaries
# (the Perrin direct test statistic)
# ============================================================================

class TestTotalCount:
    def matrix(self):
        return {
            (0, 1): [("A", "B", "semantic"), ("C", "D", "phonological")],
            (1, 2): [("E", "F", "semantic")],
            (0, 2): [("X", "Y", "semantic")],
        }

    def test_total_count_no_filter(self):
        # Order [0, 1, 2] → boundaries (0,1) and (1,2) → 2 + 1 = 3 cells
        total = total_count_for_order([0, 1, 2], self.matrix(), None)
        assert total == 3

    def test_phon_filter(self):
        # Only count phonological+etymological
        phon = frozenset({"phonological", "etymological"})
        total = total_count_for_order([0, 1, 2], self.matrix(), phon)
        # Boundary (0,1) has 1 phon, (1,2) has 0 → total 1
        assert total == 1

    def test_sem_filter(self):
        sem = frozenset({"semantic"})
        total = total_count_for_order([0, 1, 2], self.matrix(), sem)
        # (0,1) has 1 sem, (1,2) has 1 sem → total 2
        assert total == 2

    def test_filter_matrix_preserves_keys(self):
        m = self.matrix()
        # Add empty cell to confirm preservation
        m[(99, 100)] = []
        out = filter_matrix(m, frozenset({"semantic"}))
        assert set(out.keys()) == set(m.keys())


# ============================================================================
# Permutation reproducibility — same seed = same null
# ============================================================================

class TestPermutationReproducibility:
    def test_same_seed_same_null(self):
        matrix = {
            (0, 1): frozenset([("A", "B", "semantic")]),
            (1, 2): frozenset([("A", "B", "semantic")]),
            (2, 3): frozenset([("C", "D", "phonological")]),
            (0, 2): frozenset([("E", "F", "semantic")]),
            (1, 0): frozenset([("A", "B", "semantic")]),
        }
        ids = [0, 1, 2, 3]
        n1 = run_permutation(matrix, ids, 100, seed=42, min_freqs=(2,))
        n2 = run_permutation(matrix, ids, 100, seed=42, min_freqs=(2,))
        np.testing.assert_array_equal(n1["recurring_2plus"], n2["recurring_2plus"])

    def test_different_seeds_differ(self):
        matrix = {
            (0, 1): frozenset([("A", "B", "semantic")]),
            (1, 2): frozenset([("A", "B", "semantic")]),
            (2, 3): frozenset([("C", "D", "phonological")]),
            (0, 2): frozenset([("E", "F", "semantic")]),
            (1, 0): frozenset([("A", "B", "semantic")]),
        }
        ids = [0, 1, 2, 3]
        n1 = run_permutation(matrix, ids, 100, seed=42, min_freqs=(2,))
        n2 = run_permutation(matrix, ids, 100, seed=43, min_freqs=(2,))
        # Different seeds should produce different sequences (at least somewhat)
        assert not np.array_equal(n1["recurring_2plus"], n2["recurring_2plus"])

    def test_null_array_length(self):
        matrix = {(0, 1): frozenset([("A", "B", "semantic")]),
                  (1, 2): frozenset([("A", "B", "semantic")]),
                  (1, 0): frozenset(), (2, 1): frozenset(),
                  (0, 2): frozenset(), (2, 0): frozenset()}
        n = run_permutation(matrix, [0, 1, 2], 500, seed=42, min_freqs=(2,))
        assert len(n["recurring_2plus"]) == 500

    def test_total_count_perm_reproducibility(self):
        matrix = {
            (0, 1): [("A", "B", "semantic")],
            (1, 2): [("C", "D", "phonological")],
            (0, 2): [("E", "F", "semantic")],
            (1, 0): [("A", "B", "semantic")],
            (2, 1): [("C", "D", "phonological")],
            (2, 0): [("E", "F", "semantic")],
        }
        ids = [0, 1, 2]
        n1 = run_perm_total(matrix, ids, None, 100, seed=42)
        n2 = run_perm_total(matrix, ids, None, 100, seed=42)
        np.testing.assert_array_equal(n1, n2)


# ============================================================================
# Null-distribution sanity
# ============================================================================

class TestNullDistribution:
    def test_null_mean_stable_across_runs(self):
        """With large N perms, mean should stabilize."""
        matrix = {}
        rng = random.Random(0)
        # Build a 6×5 matrix with random catchword cells
        ids = list(range(6))
        for i in ids:
            for j in ids:
                if i == j: continue
                k = rng.randint(0, 5)
                cell = [(f"L{x}", f"L{x+1}", "semantic") for x in range(k)]
                matrix[(i, j)] = frozenset(cell)
        n1 = run_permutation(matrix, ids, 2000, seed=1, min_freqs=(2,))
        n2 = run_permutation(matrix, ids, 2000, seed=2, min_freqs=(2,))
        # Different seeds, large N → means should be within ~3 std of each other
        m1, m2 = n1["recurring_2plus"].mean(), n2["recurring_2plus"].mean()
        std = n1["recurring_2plus"].std()
        assert abs(m1 - m2) < 5 * std / np.sqrt(2000), \
            "Null mean should be stable across seeds at large N"


# ============================================================================
# compute_blocked — filter_pct edge cases (KNOWN BUG ZONE)
# ============================================================================

class TestBlocking:
    """The filter_pct argument has subtle edge cases that bit us before."""

    def _tokens_from_lemmas(self, lemma_lists):
        """[[A,B], [A,C], [D]] → {0: [{lemma:A},{lemma:B}], 1: ..., 2: ...}"""
        return {i: [{"lemma": L, "form": L, "parse": "MS-EMP"} for L in lems]
                for i, lems in enumerate(lemma_lists)}

    def test_filter_pct_80_blocks_top_20_percent(self):
        # 10 units; lemma X appears in 9 of them (90% > 80% cutoff → blocked)
        # lemma Y appears in 5 (50% < 80% → kept)
        tokens = {}
        for i in range(10):
            tokens[i] = [{"lemma": "X", "form": "X", "parse": "MS-EMP"}]
            if i >= 5:
                tokens[i].append({"lemma": "Y", "form": "Y", "parse": "MS-EMP"})
        # X appears in 10 (100%), Y in 5 (50%). cutoff = 8 → block X.
        blocked = compute_blocked(tokens, 80.0)
        assert "X" in blocked, "X (100% freq) should be blocked at filter_pct=80"
        # Drop one X so its count is 9; still 90% > 80% → blocked
        tokens[0] = [{"lemma": "Z", "form": "Z", "parse": "MS-EMP"}]
        blocked = compute_blocked(tokens, 80.0)
        assert "X" in blocked  # 9/10 = 90% > 80
        assert "Y" not in blocked  # 5/10 = 50% < 80

    def test_filter_pct_100_blocks_nothing(self):
        tokens = self._tokens_from_lemmas([["A"], ["A"], ["A"], ["B"]])
        blocked = compute_blocked(tokens, 100.0)
        # cutoff = 100 * 4 / 100 = 4. Lemma must appear in > 4 units to be
        # blocked. Max possible is 4 (all units), so nothing is blocked.
        assert blocked == set()

    def test_filter_pct_0_blocks_everything_KNOWN_BUG(self):
        """KNOWN BUG: filter_pct=0 blocks EVERY lemma that appears in any unit
        (because cutoff = 0, and any count > 0 exceeds it). This caught us
        once — see perrin_test_one.py original 'noblock' run. Document
        the behavior so future callers don't repeat the mistake."""
        tokens = self._tokens_from_lemmas([["A"], ["B"], ["C"]])
        blocked = compute_blocked(tokens, 0.0)
        assert blocked == {"A", "B", "C"}, \
            "filter_pct=0 blocks every lemma — use 100 for 'no blocking'"

    def test_blocking_uses_unit_count_not_token_count(self):
        # Lemma A appears 10 times in unit 0, 0 times elsewhere → 1 unit
        tokens = {
            0: [{"lemma": "A", "form": "A", "parse": "MS-EMP"} for _ in range(10)],
            1: [{"lemma": "B", "form": "B", "parse": "MS-EMP"}],
            2: [{"lemma": "B", "form": "B", "parse": "MS-EMP"}],
            3: [{"lemma": "B", "form": "B", "parse": "MS-EMP"}],
            4: [{"lemma": "B", "form": "B", "parse": "MS-EMP"}],
        }
        # A in 1 unit (20%), B in 4 (80%). cutoff = 4 at filter_pct=80.
        # B has count 4 → 4 > 4 is False → not blocked.
        blocked = compute_blocked(tokens, 80.0)
        assert "A" not in blocked
        assert "B" not in blocked  # Exactly 80%, > 80 fails

    def test_blocking_empty_corpus(self):
        assert compute_blocked({}, 80.0) == set()


# ============================================================================
# Statistic correctness — symbolic regression suite
# ============================================================================

class TestStatisticRegression:
    """Hand-computed counts for tiny scenarios. If any of these change in
    the future, the change is intentional (or a bug)."""

    def test_planted_two_recurring_pairs(self):
        """6 logia; same-lemma pair recurs at boundaries (0,1), (1,2), (3,4)
        in the true order. Pair ZZ recurs at (4,5) and (2,3)."""
        matrix = {}
        ids = [0, 1, 2, 3, 4, 5]
        # Fill all cells with empty by default
        for i in ids:
            for j in ids:
                if i == j: continue
                matrix[(i, j)] = frozenset()
        # Insert planted catchwords
        AB = ("A", "B", "semantic")
        ZZ = ("Y", "Z", "semantic")
        matrix[(0, 1)] = frozenset([AB])
        matrix[(1, 2)] = frozenset([AB])
        matrix[(2, 3)] = frozenset([ZZ])
        matrix[(3, 4)] = frozenset([AB])
        matrix[(4, 5)] = frozenset([ZZ])
        s = stats_for_order(ids, matrix, [2, 3])
        # Pair AB at positions (0,1), (1,2), (3,4) → 3 boundaries → in ≥2 AND ≥3
        # Pair ZZ at positions (2,3), (4,5) → 2 boundaries → in ≥2 only
        assert s["recurring_2plus"] == 2
        assert s["recurring_3plus"] == 1  # only AB
        assert s["max_freq"] == 3  # AB at 3 boundaries
