"""
Phase 1.4 — Monte Carlo simulation of unbiased Coptic→Syriac catchword density.

Procedure:
  1. Take the Coptic Gospel of Thomas, segmented into 115 logia.
  2. For each Coptic content token in each logion, sample a Syriac lemma
     from the lexical map's full P(syriac | coptic) distribution.
  3. For each adjacent logion pair (L, L+1), count catchwords between the
     sampled Syriac token sets.
  4. Repeat N times. Report per-pair distributions (mean, std, percentiles)
     and overall totals.

Speed: instead of running the detector on Python objects per iteration,
we precompute a Syriac × Syriac adjacency matrix once. Per-pair counting
then reduces to indexing + sum on a boolean numpy array.

Usage (programmatic):
    from phase1_montecarlo.monte_carlo import MonteCarloRunner
    runner = MonteCarloRunner(...).build()
    results = runner.run(n_iterations=10000, seed=42)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catchword_detector import (
    CatchwordDetector,
    consonantal,
    weighted_levenshtein,
)
from .language_data import SYRIAC


@dataclass
class MonteCarloResults:
    n_iterations: int
    seed: int
    phonological_threshold: float
    filter_pct: float
    pair_keys: list[tuple[int, int]]              # adjacent logion pairs
    pair_totals_per_iter: np.ndarray              # shape (n_iter, n_pairs)
    overall_total_per_iter: np.ndarray            # shape (n_iter,)
    n_logia_with_left: np.ndarray                  # shape (n_iter, n_logia)
    n_logia_with_right: np.ndarray                 # shape (n_iter, n_logia)
    sorted_logia: list[int]

    def per_pair_stats(self) -> dict[str, dict]:
        out = {}
        for j, (a, b) in enumerate(self.pair_keys):
            counts = self.pair_totals_per_iter[:, j]
            out[f"{a}-{b}"] = {
                "mean": float(np.mean(counts)),
                "std":  float(np.std(counts)),
                "p05":  float(np.percentile(counts, 5)),
                "p50":  float(np.percentile(counts, 50)),
                "p95":  float(np.percentile(counts, 95)),
                "prob_at_least_one":   float(np.mean(counts >= 1)),
                "prob_at_least_three": float(np.mean(counts >= 3)),
            }
        return out

    def overall_stats(self) -> dict:
        t = self.overall_total_per_iter
        # connectivity per iteration: a logion is connected on both sides iff
        # both its left and right adjacent pairs had >= 1 catchword.
        n = len(self.sorted_logia)
        both = self.n_logia_with_left & self.n_logia_with_right
        one  = self.n_logia_with_left ^ self.n_logia_with_right
        iso  = ~(self.n_logia_with_left | self.n_logia_with_right)

        both_pct = (both.sum(axis=1) / n) * 100.0
        one_pct  = (one.sum(axis=1)  / n) * 100.0
        iso_pct  = (iso.sum(axis=1)  / n) * 100.0

        return {
            "total": {
                "mean": float(np.mean(t)),
                "std":  float(np.std(t)),
                "p05":  float(np.percentile(t, 5)),
                "p50":  float(np.percentile(t, 50)),
                "p95":  float(np.percentile(t, 95)),
            },
            "both_sides_pct": {
                "mean": float(np.mean(both_pct)),
                "std":  float(np.std(both_pct)),
                "p05":  float(np.percentile(both_pct, 5)),
                "p95":  float(np.percentile(both_pct, 95)),
            },
            "one_side_pct": {
                "mean": float(np.mean(one_pct)),
                "std":  float(np.std(one_pct)),
            },
            "isolated_pct": {
                "mean": float(np.mean(iso_pct)),
                "std":  float(np.std(iso_pct)),
                "p05":  float(np.percentile(iso_pct, 5)),
                "p95":  float(np.percentile(iso_pct, 95)),
            },
        }

    def to_json(self) -> dict:
        return {
            "n_iterations": self.n_iterations,
            "seed": self.seed,
            "phonological_threshold": self.phonological_threshold,
            "filter_pct": self.filter_pct,
            "n_logia": len(self.sorted_logia),
            "n_adjacent_pairs": len(self.pair_keys),
            "overall": self.overall_stats(),
            "per_pair": self.per_pair_stats(),
        }


class MonteCarloRunner:
    """Build once (slow), run many times (fast)."""

    def __init__(
        self,
        thomas_logia_jsonl: Path,
        lex_map_jsonl: Path,
        phonological_threshold: float = 0.65,
        filter_pct: float = 80.0,
    ):
        self.thomas_path = Path(thomas_logia_jsonl)
        self.lex_map_path = Path(lex_map_jsonl)
        self.phon_threshold = phonological_threshold
        self.filter_pct = filter_pct
        self._built = False

    # ---------- build ----------

    def build(self) -> "MonteCarloRunner":
        self._load_lex_map()
        self._load_logia()
        self._compute_filter()
        self._build_adjacency_matrix()
        self._built = True
        return self

    def _load_lex_map(self):
        """For each Coptic content lemma, store (vocab_indices, cum_probs)
        for fast sampling. The Syriac vocab is the union of all candidate
        Syriac lemmas across the map."""
        syr_vocab: dict[str, int] = {}
        coptic_to_dist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        coptic_pos: dict[str, str] = {}
        for line in self.lex_map_path.open():
            r = json.loads(line)
            cands = r["candidates"]
            if not cands:
                continue
            indices = []
            probs = []
            for cand in cands:
                lem = cand["syriac_lemma"]
                if lem not in syr_vocab:
                    syr_vocab[lem] = len(syr_vocab)
                indices.append(syr_vocab[lem])
                probs.append(cand["prob"])
            arr_p = np.asarray(probs, dtype=np.float64)
            arr_p /= arr_p.sum()  # renormalize (we may have pruned tail)
            coptic_to_dist[r["coptic_lemma"]] = (
                np.asarray(indices, dtype=np.int32),
                np.cumsum(arr_p),
            )
            coptic_pos[r["coptic_lemma"]] = r["coptic_pos"]
        self.syr_vocab = syr_vocab
        self.syr_vocab_list = [None] * len(syr_vocab)
        for lem, idx in syr_vocab.items():
            self.syr_vocab_list[idx] = lem
        self.coptic_to_dist = coptic_to_dist
        self.coptic_pos = coptic_pos
        print(f"  Lex map: {len(coptic_to_dist)} Coptic content lemmas, "
              f"Syriac vocab size = {len(syr_vocab)}")

    def _load_logia(self):
        """For each logion, list the Coptic lemmas of its content tokens that
        appear in the lexical map (and so are translatable). We only count
        catchwords from translated tokens."""
        per_logion: dict[int, list[str]] = {}
        for line in self.thomas_path.open():
            r = json.loads(line)
            L = r["logion"]
            for t in r["tokens"]:
                cl = t.get("lemma")
                if not cl:
                    continue
                if cl in self.coptic_to_dist:
                    per_logion.setdefault(L, []).append(cl)
        self.sorted_logia = sorted(per_logion.keys())
        # As parallel arrays (one per logion) of distinct lemmas + their counts
        self.logion_lemmas: list[list[str]] = [per_logion[L] for L in self.sorted_logia]
        n_total_tokens = sum(len(l) for l in self.logion_lemmas)
        print(f"  Logia: {len(self.sorted_logia)}, total translatable content "
              f"tokens: {n_total_tokens}")

    def _compute_filter(self):
        """Determine which Syriac vocab indices are 'high-frequency lemmas'
        that should be excluded from catchword counting (the speech-formula
        boilerplate effect documented in the calibration sweep)."""
        n_logia = len(self.sorted_logia)
        cutoff = self.filter_pct * n_logia / 100.0
        # Best estimate of per-Syriac-lemma logia frequency: use MAP translation
        # (the most likely Syriac lemma per Coptic occurrence).
        map_logia_count = np.zeros(len(self.syr_vocab), dtype=np.int32)
        map_top: dict[str, int] = {}
        for c, (idx, cum) in self.coptic_to_dist.items():
            map_top[c] = int(idx[0])  # first candidate is highest-prob
        for lemmas in self.logion_lemmas:
            seen = {map_top[c] for c in lemmas if c in map_top}
            for s in seen:
                map_logia_count[s] += 1
        self.blocked = (map_logia_count > cutoff)
        n_blocked = int(self.blocked.sum())
        print(f"  Filter@{self.filter_pct}%: blocking {n_blocked} Syriac lemmas "
              f"(appearing in > {cutoff:.1f} of {n_logia} logia under MAP).")

    def _build_adjacency_matrix(self):
        """Compute the |V|×|V| Syriac catchword adjacency matrix.

        adj[i, j] = True iff lemma i and lemma j form a catchword link
        (semantic / etymological / phonological). Diagonal is False:
        a token paired with another instance of itself within the same
        logion is not a catchword between adjacent logia.

        We use the SAME logic as CatchwordDetector.detect, just precomputed.
        """
        V = len(self.syr_vocab)
        # Precompute consonantal forms of every lemma
        cons = [consonantal(l) for l in self.syr_vocab_list]
        adj = np.zeros((V, V), dtype=bool)

        # Group by consonantal form for fast etymological detection
        # (different lemmas with identical consonantal skeleton)
        from collections import defaultdict
        by_cons: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(cons):
            by_cons[c].append(i)
        # Etymological links (different lemmas, identical consonantal)
        for c_form, idxs in by_cons.items():
            if len(idxs) < 2:
                continue
            for a in idxs:
                for b in idxs:
                    if a != b:
                        adj[a, b] = True

        # Phonological links: edit-distance-bounded pairs.
        # We can't avoid an O(V^2) sweep, but we can prune by length difference.
        threshold = self.phon_threshold

        t0 = time.time()
        # Compute Levenshtein only for pairs whose lengths differ by at most
        # what could possibly clear the threshold: dist ≤ longest * (1 - thr)
        for i in range(V):
            ci = cons[i]
            li = len(ci)
            if li == 0:
                continue
            for j in range(i + 1, V):
                cj = cons[j]
                lj = len(cj)
                if lj == 0:
                    continue
                longest = max(li, lj)
                # Quick reject by length-difference lower bound
                if abs(li - lj) > longest * (1 - threshold) * 2:  # generous
                    continue
                if ci == cj:
                    continue  # already handled in etymological pass
                d = weighted_levenshtein(ci, cj, SYRIAC)
                score = 1.0 - d / longest
                if score >= threshold:
                    adj[i, j] = True
                    adj[j, i] = True

            if i and i % 500 == 0:
                elapsed = time.time() - t0
                print(f"    adj matrix progress: {i}/{V}  ({elapsed:.1f}s)")

        # Apply blocked-lemma mask: blocked lemmas don't form catchwords.
        adj[self.blocked, :] = False
        adj[:, self.blocked] = False

        self.adj = adj
        n_links = int(adj.sum() // 2)
        print(f"  Adjacency: {n_links} undirected catchword links across "
              f"{V} Syriac lemmas (excluding blocked).")

    # ---------- run ----------

    def run(self, n_iterations: int = 10000, seed: int = 42) -> MonteCarloResults:
        if not self._built:
            raise RuntimeError("Call .build() first.")
        rng = np.random.default_rng(seed)
        n_logia = len(self.sorted_logia)
        n_pairs = n_logia - 1
        pair_keys = [(self.sorted_logia[i], self.sorted_logia[i + 1])
                     for i in range(n_pairs)]

        # Pre-flatten: for each logion, list its (cum_prob, indices) per token
        # so we can sample N times in a vectorized way.
        per_logion_dists = []  # list of (indices_per_tok, cum_per_tok)
        for lemmas in self.logion_lemmas:
            tok_idx_arrays = []
            tok_cum_arrays = []
            for cl in lemmas:
                idx, cum = self.coptic_to_dist[cl]
                tok_idx_arrays.append(idx)
                tok_cum_arrays.append(cum)
            per_logion_dists.append((tok_idx_arrays, tok_cum_arrays))

        pair_totals = np.zeros((n_iterations, n_pairs), dtype=np.int32)
        overall = np.zeros(n_iterations, dtype=np.int32)
        has_left  = np.zeros((n_iterations, n_logia), dtype=bool)
        has_right = np.zeros((n_iterations, n_logia), dtype=bool)

        adj = self.adj
        t0 = time.time()
        report_every = max(1, n_iterations // 20)

        for it in range(n_iterations):
            # 1. Sample a Syriac lemma per token per logion
            sampled_per_logion = [None] * n_logia
            for L_i, (idx_arrays, cum_arrays) in enumerate(per_logion_dists):
                if not idx_arrays:
                    sampled_per_logion[L_i] = np.array([], dtype=np.int32)
                    continue
                samples = np.empty(len(idx_arrays), dtype=np.int32)
                # Random uniforms, one per token
                u = rng.random(len(idx_arrays))
                for k, (ind, cum) in enumerate(zip(idx_arrays, cum_arrays)):
                    pos = np.searchsorted(cum, u[k])
                    if pos >= len(ind):
                        pos = len(ind) - 1
                    samples[k] = ind[pos]
                sampled_per_logion[L_i] = np.unique(samples)

            # 2. Count catchwords per adjacent pair via adjacency lookup
            iter_total = 0
            for j in range(n_pairs):
                Sa = sampled_per_logion[j]
                Sb = sampled_per_logion[j + 1]
                if Sa.size == 0 or Sb.size == 0:
                    continue
                # Submatrix sum gives the number of catchword pairs
                count = int(adj[Sa[:, None], Sb[None, :]].sum())
                pair_totals[it, j] = count
                iter_total += count
                if count > 0:
                    has_right[it, j]     = True   # logion at index j has right catchword
                    has_left[it, j + 1]  = True   # logion at index j+1 has left catchword
            overall[it] = iter_total

            if (it + 1) % report_every == 0:
                el = time.time() - t0
                rate = (it + 1) / el
                print(f"    iter {it+1}/{n_iterations}  total={iter_total}  "
                      f"({rate:.0f} iter/s)")

        return MonteCarloResults(
            n_iterations=n_iterations,
            seed=seed,
            phonological_threshold=self.phon_threshold,
            filter_pct=self.filter_pct,
            pair_keys=pair_keys,
            pair_totals_per_iter=pair_totals,
            overall_total_per_iter=overall,
            n_logia_with_left=has_left,
            n_logia_with_right=has_right,
            sorted_logia=self.sorted_logia,
        )
