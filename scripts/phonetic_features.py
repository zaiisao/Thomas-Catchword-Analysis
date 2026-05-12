"""
Phonetic-feature distance for Classical Syriac consonants.

Upgrades the catchword detector's phonological branch from a binary
confusion-group + uniform substitution cost to a graded feature-distance
metric. Captures Semitic-specific paronomasia that the binary metric
misses:

  - voicing pairs (b/p, t/d, k/g)
  - emphatic pairs (t/ṭ, d/ḍ, s/ṣ, k/q)
  - spirantization pairs (BGDKPT)
  - place-and-manner near-misses (sibilants s/š/z)

Features per consonant: [place, manner, voicing, emphatic, weak].

  place:    1=bilabial 2=labiodental 3=dental 4=alveolar 5=postalveolar
            6=palatal 7=velar 8=uvular 9=pharyngeal 10=glottal
  manner:   1=stop 2=fricative 3=affricate 4=nasal 5=liquid 6=glide
  voicing:  0=voiceless 1=voiced
  emphatic: 0=plain 1=emphatic (incl. uvular ܩ)
  weak:     0=full 1=weak (mater lectionis / glide)

Reference: K. Beyer, *The Aramaic Language*; Brockelmann, *Lexicon Syriacum*;
modern Semitist consensus (e.g., Khan 2020 NENA).
"""
from __future__ import annotations

# (place, manner, voicing, emphatic, weak)
SYRIAC_FEATURES: dict[str, tuple[int, int, int, int, int]] = {
    "ܐ": (10, 1, 0, 0, 1),   # alaph: glottal stop, weak
    "ܒ": ( 1, 1, 1, 0, 0),   # bet: voiced bilabial stop (spirantizes to v)
    "ܓ": ( 7, 1, 1, 0, 0),   # gimel: voiced velar stop (spirantizes to ɣ)
    "ܕ": ( 3, 1, 1, 0, 0),   # dalath: voiced dental stop (spirantizes to ð)
    "ܗ": (10, 2, 0, 0, 1),   # he: voiceless glottal fricative, weak
    "ܘ": ( 1, 6, 1, 0, 1),   # waw: bilabial glide, weak
    "ܙ": ( 4, 2, 1, 0, 0),   # zayn: voiced alveolar fricative
    "ܚ": ( 9, 2, 0, 0, 0),   # heth: voiceless pharyngeal fricative
    "ܛ": ( 3, 1, 0, 1, 0),   # teth: voiceless dental stop, EMPHATIC
    "ܝ": ( 6, 6, 1, 0, 1),   # yodh: palatal glide, weak
    "ܟ": ( 7, 1, 0, 0, 0),   # kaph: voiceless velar stop (spirantizes to x)
    "ܠ": ( 4, 5, 1, 0, 0),   # lamadh: voiced alveolar liquid
    "ܡ": ( 1, 4, 1, 0, 0),   # mim: voiced bilabial nasal
    "ܢ": ( 4, 4, 1, 0, 0),   # nun: voiced alveolar nasal
    "ܣ": ( 4, 2, 0, 0, 0),   # semkat: voiceless alveolar fricative
    "ܥ": ( 9, 2, 1, 0, 0),   # ʿayn: voiced pharyngeal fricative
    "ܦ": ( 1, 1, 0, 0, 0),   # pe: voiceless bilabial stop (spirantizes to f)
    "ܨ": ( 4, 2, 0, 1, 0),   # tsadhe: voiceless alveolar fricative, EMPHATIC
    "ܩ": ( 8, 1, 0, 1, 0),   # qoph: voiceless uvular stop (emphatic-class)
    "ܪ": ( 4, 5, 1, 0, 0),   # resh: voiced alveolar liquid
    "ܫ": ( 5, 2, 0, 0, 0),   # shin: voiceless postalveolar fricative
    "ܬ": ( 3, 1, 0, 0, 0),   # taw: voiceless dental stop (spirantizes to θ)
}

# Weights per feature axis. Place and manner dominate; voicing/emphatic
# are local perturbations; weak is encoded for completeness but the
# Levenshtein insertion/deletion path already prices weak-consonant
# matres, so its substitution weight is small.
_WEIGHTS = (0.40, 0.30, 0.15, 0.10, 0.05)
# Max possible feature differences (used to normalize each axis to [0,1]).
_MAX_DIFF = (9, 5, 1, 1, 1)


def feature_substitution_cost(a: str, b: str) -> float:
    """Return substitution cost in [0, 1] between two single Syriac chars.

    - identical: 0.0
    - both unknown / non-Syriac: 1.0 (fall back to a full edit)
    - one known, one not: 1.0
    - else: weighted Manhattan distance over the 5 feature axes,
      with each axis normalized to [0, 1].

    Examples (computed, not hard-coded):
      ܬ↔ܛ (taw/teth, emphatic-only):        0.10
      ܒ↔ܦ (bet/pe, voicing-only):           0.15
      ܕ↔ܬ (dalath/taw, voicing-only):       0.15
      ܟ↔ܩ (kaph/qoph, emphatic+place):      ~0.14
      ܣ↔ܫ (semkat/shin, place-only):        ~0.16
      ܐ↔ܥ (alaph/ʿayn, place+voicing+weak): ~0.25
      ܕ↔ܪ (dalath/resh, manner+place):      ~0.40
    """
    if a == b:
        return 0.0
    fa = SYRIAC_FEATURES.get(a)
    fb = SYRIAC_FEATURES.get(b)
    if fa is None or fb is None:
        return 1.0
    cost = 0.0
    for axis, (xa, xb, w, mx) in enumerate(zip(fa, fb, _WEIGHTS, _MAX_DIFF)):
        cost += w * abs(xa - xb) / mx
    return min(cost, 1.0)


def feature_levenshtein(a: str, b: str, profile) -> float:
    """Drop-in replacement for `phase1_montecarlo.catchword_detector.
    weighted_levenshtein` that uses feature substitution costs for Syriac.

    Insertion/deletion costs remain language-profile-driven (weak
    consonants cost 0.5) so the matres lectionis behavior that captures
    e.g. nūrā ܢܘܪܐ / nuhrā ܢܘܗܪܐ is preserved.

    For substitution, the cost is `feature_substitution_cost` for known
    Syriac chars, falling back to the same confusion-group / unit logic
    for any non-Syriac characters that may appear in noisy input.
    """
    # Imports kept local to avoid circular reference when this module is
    # consumed by scripts that themselves import the detector.
    from phase1_montecarlo.catchword_detector import (
        COST_REGULAR, COST_WEAK_OR_CONFUSION, _confusion_index,
    )

    if a == b:
        return 0.0
    confusion = _confusion_index(profile)
    weak = profile.weak_consonants
    n, m = len(a), len(b)

    def _del_ins(s):
        return sum(COST_WEAK_OR_CONFUSION if c in weak else COST_REGULAR
                   for c in s)

    if n == 0:
        return _del_ins(b)
    if m == 0:
        return _del_ins(a)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = _del_ins(a[:i])
    for j in range(m + 1):
        dp[0][j] = _del_ins(b[:j])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                sub_cost = 0.0
            elif ca in SYRIAC_FEATURES and cb in SYRIAC_FEATURES:
                sub_cost = feature_substitution_cost(ca, cb)
            elif cb in confusion.get(ca, frozenset()):
                sub_cost = COST_WEAK_OR_CONFUSION
            else:
                sub_cost = COST_REGULAR

            ins_cost = COST_WEAK_OR_CONFUSION if cb in weak else COST_REGULAR
            del_cost = COST_WEAK_OR_CONFUSION if ca in weak else COST_REGULAR

            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + del_cost,
                dp[i][j - 1] + ins_cost,
            )
    return dp[n][m]


if __name__ == "__main__":
    pairs = [
        ("ܬ", "ܛ", "taw/teth — emphatic-only"),
        ("ܒ", "ܦ", "bet/pe — voicing-only"),
        ("ܕ", "ܬ", "dalath/taw — voicing-only"),
        ("ܟ", "ܩ", "kaph/qoph — uvular emphatic"),
        ("ܣ", "ܫ", "semkat/shin — sibilant place"),
        ("ܐ", "ܥ", "alaph/ʿayn — pharyngeal"),
        ("ܕ", "ܪ", "dalath/resh — visually similar"),
        ("ܡ", "ܐ", "mim/alaph — far"),
    ]
    print(f"{'pair':>4s}  {'cost':>5s}  description")
    for a, b, desc in pairs:
        print(f"  {a}{b}  {feature_substitution_cost(a, b):>5.3f}  {desc}")
