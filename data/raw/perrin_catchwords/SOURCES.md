# Perrin Catchword Data — Sources & Provenance

## What's here

| File | Source | Coverage | Status |
|---|---|---|---|
| `perrin_2006_jets.pdf` | Perrin, N. (2006), *JETS* 49/1: 67–80 | ~10 illustrative pairs + headline statistics | **Loaded** |
| `perrin_jets_2006_examples.csv` | extracted from the above PDF | 11 catchword pair entries, partial | **Loaded** |
| *(missing)* `perrin_2002_full_table.csv` | Perrin, N. (2002), *Thomas and Tatian*, Brill/SBL, pp. 57–155 | All 502 Syriac catchword pairs | **Pending — manual entry from print** |

## What this means for the analysis

The 2006 JETS paper is a summary article. It gives:
- The headline counts (Coptic 269 / Greek 263 / Syriac 502)
- Connectivity statistics (49/40/11, 50/38/12, 89/11/0)
- ~10 specific illustrative catchword pairs with transliteration + p-values

The full per-logion catchword table — the actual ground-truth needed for pair-by-pair comparison against our detector — is in the **2002 book**, pp. 57–155. That table is not available digitally.

## CSV schema

```
logion_a, logion_b           — Gospel of Thomas logion numbers (numeric)
verse_a, verse_b             — sub-verse where applicable (e.g., "16.2")
syriac_a, syriac_b           — Syriac word, transliterated with diacritics
gloss_a, gloss_b             — English gloss
coptic_a, coptic_b           — Coptic source word(s) Perrin identifies (if given)
link_type                    — semantic | phonological | etymological | wordplay
perrin_p_value               — probability of incidental co-occurrence as Perrin reports it (where given)
source                       — citation
notes                         — verbatim or near-verbatim from the paper
```

Note that some `perrin_p_value` entries are duplicated across rows where Perrin reports a single p-value spanning multiple pairs (e.g., 6.8% covers all three nūrā/nuhrā pairings jointly).

## Headline statistics (also from 2006 paper, p. 73)

| Language | Total catchwords | Connected both sides | Connected one side | Isolated |
|---|---|---|---|---|
| Coptic   | 269 | 49% | 40% | 11% |
| Greek    | 263 | 50% | 38% | 12% |
| Syriac   | 502 | 89% | 11% |  0% |

Stored programmatically in `configs/config.yaml` under `perrin_targets`.

## How to extend

When the 2002 book becomes available:
1. Enter Perrin's full table to `perrin_2002_full_table.csv` using the same schema as `perrin_jets_2006_examples.csv`.
2. Keep this JETS-derived CSV as a separate file — they're different snapshots and we'll want to verify consistency.
3. Update `configs/config.yaml::perrin_targets` only if the 2002 table reports different headline numbers.
