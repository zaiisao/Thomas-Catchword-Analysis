"""
Per-language phonology + content-POS data for the catchword detector.

Williams (2009) argues Perrin applied different standards across Coptic, Greek,
and Syriac. To rebut that, the detection logic must be uniform — only the
language-specific tables here vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    # Sets of consonants that are easily confused (visually or phonologically).
    # Substitution within a group costs 0.5 instead of 1.0 in Levenshtein.
    confusion_groups: tuple[frozenset[str], ...]
    # Weak consonants that can elide / be inserted / deleted with low audible
    # cost. Insertion/deletion of these costs 0.5 instead of 1.0 in Levenshtein.
    weak_consonants: frozenset[str]
    # POS tags considered "content words" — only these can form catchwords.
    # Function words (articles, prepositions, conjunctions, pronouns) do not.
    content_pos: frozenset[str]


SYRIAC = LanguageProfile(
    name="syriac",
    # Standard Syriac confusable letter groups (philological + scribal).
    confusion_groups=(
        frozenset({"ܕ", "ܪ"}),       # dalath / resh — visually almost identical
        frozenset({"ܒ", "ܦ"}),       # bet / pe — labials
        frozenset({"ܬ", "ܛ"}),       # taw / teth — dentals
        frozenset({"ܣ", "ܫ"}),       # semkat / shin — sibilants
        frozenset({"ܥ", "ܐ"}),       # ʿayn / alaph — glottal/pharyngeal
        frozenset({"ܚ", "ܗ"}),       # heth / he — glottal/pharyngeal
    ),
    # Waw, He, Yodh, Alaph: matres lectionis / weak consonants. Their
    # insertion or deletion (matres) is the key mechanism behind Perrin's
    # nūrā / nuhrā homophone (which differs by an inserted ܗ).
    weak_consonants=frozenset({"ܘ", "ܗ", "ܝ", "ܐ"}),
    # SEDRA parse codes start with one of these for content words.
    # We accept any parse beginning with: noun-state codes (MS/FS/MP/FP)
    # OR verbal stems (PEAL, PAEL, APHEL, ETHPEAL, ETHPAEL, ETHTAPHAL,
    # SHAPHEL, ESTAPHAL). See data/external/sedra/SOURCES.md.
    content_pos=frozenset({
        # Noun/adj morphology codes (state markers)
        "MS", "FS", "MP", "FP",
        # Verbal stem codes (the prefix before "-Mxx-P")
        "PEAL", "PAEL", "APHEL", "ETHPEAL", "ETHPAEL",
        "SHAPHEL", "ESTAPHAL", "ETHTAPHAL",
    }),
)


COPTIC = LanguageProfile(
    name="coptic",
    # Common Coptic scribal / phonetic confusions.
    confusion_groups=(
        frozenset({"ⲃ", "ⲡ"}),   # b / p
        frozenset({"ⲇ", "ⲧ"}),   # d / t
        frozenset({"ⲅ", "ⲕ"}),   # g / k
        frozenset({"ⲏ", "ⲉ"}),   # eta / epsilon
        frozenset({"ⲱ", "ⲟ"}),   # omega / omicron
    ),
    # Coptic weak / often-elided consonants.
    weak_consonants=frozenset({"ⲩ", "ⲓ", "ⲱ"}),
    # Coptic SCRIPTORIUM POS tags for content words.
    # See sample lemma rows in thomas_logia.jsonl / sahidica_nt_coptic_tt.jsonl.
    content_pos=frozenset({
        "N", "NPROP", "V", "VBD", "VSTAT", "VIMP", "ADJ", "ADV",
    }),
)


GREEK = LanguageProfile(
    name="greek",
    # Standard Koine confusions (largely iotacism + voicing pairs).
    confusion_groups=(
        frozenset({"ι", "η", "υ", "ει", "οι", "υι"}),  # iotacism (will rarely fire — these are multi-char)
        frozenset({"ο", "ω"}),
        frozenset({"ε", "αι"}),
        frozenset({"β", "π", "φ"}),
        frozenset({"δ", "τ", "θ"}),
        frozenset({"γ", "κ", "χ"}),
    ),
    weak_consonants=frozenset({"ν", "σ"}),  # mobile-nu, sigma elision
    content_pos=frozenset({
        "NOUN", "PROPN", "VERB", "ADJ", "ADV",
    }),
)


PROFILES: dict[str, LanguageProfile] = {
    "syriac": SYRIAC,
    "coptic": COPTIC,
    "greek": GREEK,
}


def get_profile(language: str) -> LanguageProfile:
    if language not in PROFILES:
        raise ValueError(f"Unknown language: {language!r}. "
                         f"Available: {list(PROFILES)}")
    return PROFILES[language]
