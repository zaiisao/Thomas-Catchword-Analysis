"""
Catchword detector — the core of Phase 1.

Given two lists of tokens (typically from two adjacent logia), find every
pair of tokens that could plausibly be a catchword link. A "catchword" is
defined uniformly across languages as one of:

  - SEMANTIC:     same lemma (or a known synonym)
  - ETYMOLOGICAL: different lemmas with identical consonantal skeleton
                  (e.g., panni "returned" / penayim "districts" share root p-n-y)
  - PHONOLOGICAL: different lemmas with consonantal skeletons close in edit
                  distance, possibly modulo confusion-group substitutions or
                  weak-consonant elisions
                  (e.g., nūrā ܢܘܪܐ / nuhrā ܢܘܗܪܐ differ by inserted ܗ)

The same algorithm runs for Coptic, Greek, and Syriac — only the
LanguageProfile (confusion groups, weak consonants, content POS) varies.
This is the structural answer to Williams' (2009) charge that Perrin
applied different standards across languages.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .language_data import LanguageProfile, get_profile


# Score thresholds (mirror configs/config.yaml):
SEMANTIC_SCORE = 1.0
ETYMOLOGICAL_SCORE = 0.8
PHONOLOGICAL_NEAR_SCORE = 0.85   # 1 confusion-group / weak-consonant edit
PHONOLOGICAL_PARTIAL_SCORE = 0.6  # 1 generic edit
PHONOLOGICAL_THRESHOLD = 0.6      # min score to count as a phonological catchword

# Levenshtein edit costs:
COST_REGULAR = 1.0
COST_WEAK_OR_CONFUSION = 0.5


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def consonantal(text: str) -> str:
    """Strip combining marks (vocalization, diacritics) from a string.

    Works uniformly for Syriac (vowel pointing), Greek (accents/breathings),
    and Coptic (supralinear strokes).
    """
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def _confusion_index(profile: LanguageProfile) -> dict[str, frozenset[str]]:
    """Map each confusable letter to the set of letters in its group."""
    index: dict[str, frozenset[str]] = {}
    for group in profile.confusion_groups:
        for c in group:
            index[c] = group
    return index


def _is_content_pos(pos: str | None, profile: LanguageProfile) -> bool:
    """A token is a content word iff its POS (or POS prefix for SEDRA parse
    codes like 'MS-EMP', 'APHEL-M3S-P') matches a content tag.

    SEDRA parses look like 'MS-EMP' or 'APHEL-M3S-P' — we test the prefix
    before the first '-'. Coptic/Greek POS are atomic ('N', 'VERB').
    """
    if not pos:
        return False
    head = pos.split("-", 1)[0].split("|", 1)[0]
    if head in profile.content_pos:
        return True
    # Also try the full token (Coptic/Greek single-token tags)
    return pos in profile.content_pos


def weighted_levenshtein(a: str, b: str, profile: LanguageProfile) -> float:
    """Levenshtein distance with linguistic edit costs.

    Substitution within a confusion group, or insertion/deletion of a weak
    consonant, costs 0.5 instead of 1.0.
    """
    if a == b:
        return 0.0
    confusion = _confusion_index(profile)
    weak = profile.weak_consonants
    n, m = len(a), len(b)
    if n == 0:
        return sum(COST_WEAK_OR_CONFUSION if c in weak else COST_REGULAR for c in b)
    if m == 0:
        return sum(COST_WEAK_OR_CONFUSION if c in weak else COST_REGULAR for c in a)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = sum(COST_WEAK_OR_CONFUSION if a[k] in weak else COST_REGULAR
                       for k in range(i))
    for j in range(m + 1):
        dp[0][j] = sum(COST_WEAK_OR_CONFUSION if b[k] in weak else COST_REGULAR
                       for k in range(j))

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                sub_cost = 0.0
            elif cb in confusion.get(ca, frozenset()):
                sub_cost = COST_WEAK_OR_CONFUSION
            else:
                sub_cost = COST_REGULAR

            ins_cost = COST_WEAK_OR_CONFUSION if cb in weak else COST_REGULAR
            del_cost = COST_WEAK_OR_CONFUSION if ca in weak else COST_REGULAR

            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # substitute
                dp[i - 1][j] + del_cost,       # delete from a
                dp[i][j - 1] + ins_cost,       # insert into a
            )
    return dp[n][m]


def phonological_score(lem_a: str, lem_b: str, profile: LanguageProfile) -> float:
    """Return a similarity score in [0,1] from consonantal-skeleton edit distance.

    score = 1 - distance / max(len_a, len_b)
    Capped at PHONOLOGICAL_NEAR_SCORE — perfect matches are semantic / etymological,
    not phonological.
    """
    a = consonantal(lem_a)
    b = consonantal(lem_b)
    if not a or not b:
        return 0.0
    if a == b:
        return PHONOLOGICAL_NEAR_SCORE  # but caller should classify as etymological
    dist = weighted_levenshtein(a, b, profile)
    longest = max(len(a), len(b))
    raw = max(0.0, 1.0 - dist / longest)
    return min(raw, PHONOLOGICAL_NEAR_SCORE)


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Catchword:
    token_a: dict
    token_b: dict
    link_type: str   # "semantic" | "etymological" | "phonological"
    score: float
    evidence: str

    def to_dict(self) -> dict:
        return {
            "lemma_a": self.token_a.get("lemma"),
            "lemma_b": self.token_b.get("lemma"),
            "form_a": self.token_a.get("form"),
            "form_b": self.token_b.get("form"),
            "pos_a": self.token_a.get("pos") or self.token_a.get("parse"),
            "pos_b": self.token_b.get("pos") or self.token_b.get("parse"),
            "link_type": self.link_type,
            "score": self.score,
            "evidence": self.evidence,
        }


class CatchwordDetector:
    def __init__(
        self,
        language: str | LanguageProfile = "syriac",
        phonological_threshold: float = PHONOLOGICAL_THRESHOLD,
        require_content_pos: bool = True,
        lemma_to_root: dict | None = None,
    ):
        """
        lemma_to_root: optional mapping {lemma_str: {"root_id": str, "root_syriac": str}}.
            If supplied, two distinct lemmas with the same root_id will be
            classified as 'etymological' even when their consonantal skeletons
            differ. (E.g., ܝܠܕ "to bear" and ܝܠܝܕܘܬܐ "genealogy" share root ܝܠܕ.)
            Built by scripts/build_sedra_root_map.py from SEDRA-3.
        """
        self.profile = get_profile(language) if isinstance(language, str) else language
        self.phon_threshold = phonological_threshold
        self.require_content_pos = require_content_pos
        self.lemma_to_root = lemma_to_root or {}

    def _eligible(self, token: dict) -> bool:
        """Token must have a lemma, and (optionally) be a content word."""
        if not token.get("lemma"):
            return False
        if self.require_content_pos:
            pos = token.get("pos") or token.get("parse") or ""
            if not _is_content_pos(pos, self.profile):
                return False
        return True

    def _classify(self, ta: dict, tb: dict) -> Catchword | None:
        la = ta.get("lemma") or ""
        lb = tb.get("lemma") or ""
        if not la or not lb:
            return None

        # Semantic: identical lemma string
        if la == lb:
            return Catchword(
                token_a=ta, token_b=tb,
                link_type="semantic",
                score=SEMANTIC_SCORE,
                evidence=f"shared lemma {la!r}",
            )

        ca = consonantal(la)
        cb = consonantal(lb)
        if not ca or not cb:
            return None

        # Etymological: different lemmas, identical consonantal skeleton
        if ca == cb:
            return Catchword(
                token_a=ta, token_b=tb,
                link_type="etymological",
                score=ETYMOLOGICAL_SCORE,
                evidence=f"shared consonantal skeleton {ca!r} (lemmas {la!r} vs {lb!r})",
            )

        # Etymological via SEDRA root table (different lemmas + different
        # consonantal skeletons, but they share a triliteral root, e.g.,
        # ܝܠܕ "to bear" and ܝܠܝܕܘܬܐ "genealogy"). Only fires if the caller
        # supplied a lemma_to_root map.
        if self.lemma_to_root:
            ra = self.lemma_to_root.get(la)
            rb = self.lemma_to_root.get(lb)
            if ra and rb and ra.get("root_id") and ra["root_id"] == rb["root_id"]:
                return Catchword(
                    token_a=ta, token_b=tb,
                    link_type="etymological",
                    score=ETYMOLOGICAL_SCORE,
                    evidence=f"shared SEDRA root {ra.get('root_syriac', '')!r} "
                             f"({la!r} vs {lb!r})",
                )

        # Phonological: edit distance below threshold
        dist = weighted_levenshtein(ca, cb, self.profile)
        longest = max(len(ca), len(cb))
        score = 1.0 - dist / longest
        if score >= self.phon_threshold:
            tier = (PHONOLOGICAL_NEAR_SCORE if dist <= COST_WEAK_OR_CONFUSION
                    else PHONOLOGICAL_PARTIAL_SCORE)
            return Catchword(
                token_a=ta, token_b=tb,
                link_type="phonological",
                score=tier,
                evidence=(f"weighted Levenshtein {dist:.1f} / {longest} "
                          f"between {ca!r} and {cb!r}"),
            )
        return None

    def detect(self, tokens_a: Iterable[dict], tokens_b: Iterable[dict]) -> list[Catchword]:
        """Find every catchword pair between two token lists.

        Returns at most one Catchword per (lemma_a, lemma_b) pair, taking the
        strongest link type. This avoids inflating counts when a logion repeats
        the same lemma several times.
        """
        eligible_a = [t for t in tokens_a if self._eligible(t)]
        eligible_b = [t for t in tokens_b if self._eligible(t)]
        seen: dict[tuple[str, str], Catchword] = {}
        for ta in eligible_a:
            la = ta["lemma"]
            for tb in eligible_b:
                lb = tb["lemma"]
                key = (la, lb)
                if key in seen:
                    continue
                cw = self._classify(ta, tb)
                if cw is not None:
                    seen[key] = cw
        return list(seen.values())
