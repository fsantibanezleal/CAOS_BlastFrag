# Diagnostics

`blastfrag.diagnostics` reports on the corpus and on fitted arms. None of its functions filters a row,
refits an arm or changes a score; each returns a report and the caller decides what to show.

---

## 1. Outlier screen

An Isolation Forest (Liu, Ting and Zhou 2008, `doi:10.1109/ICDM.2008.17`) isolates each point by
random axis-aligned cuts and scores it by how few cuts it takes; unusual points are isolated quickly.
Huan et al. 2025 (`doi:10.1038/s41598-025-96005-7`) ran this kind of screen on a 105-sample superset of
this corpus and flagged five samples, which they removed.

`outlier_screen` runs it over the seven features and the logarithm of the measured size, standardised,
with the contamination set to Huan et al.'s observed rate (5 of 105). Including the measured size is
this package's choice; the copy of the paper held for this work does not say which columns its screen
used.

On the 97 blasts it flags five: `Ru1` and `Ru4` (Reocin underground), `Db4` and `Db5` (Dongri-Buzurg)
and `Mi6` (Miami). All five come from the three most unusual campaigns: the 18 m underground bench,
the weakest rock in the corpus, and the site with the smallest fragments. On a corpus of ten
campaigns, the rows a density method calls unusual are the rows of unusual sites, and removing them
would remove the hardest part of the cross-site question. The screen is therefore reported and never
applied: `applied_as_filter` is always `False`.

## 2. Model-native importance

`native_importance` reads a fitted forest's mean decrease in impurity (Breiman 2001,
`doi:10.1023/A:1010933404324`) or a fitted XGBoost model's importance under the library's default type
(gain, Chen and Guestrin 2016, `doi:10.1145/2939672.2939785`), and for the stacked model both base
learners.

| Feature | forest, fitted on 97 | boosting (gain), fitted on 97 | Sui et al. 2025, forest | Sui et al. 2025, XGBoost |
|---|---|---|---|---|
| Young modulus `E` | 0.606 | 0.919 | 0.7129 | 0.4608 |
| in-situ block size `XB` | 0.143 | 0.040 | second | second |
| stemming ratio `T/B` | 0.132 | 0.007 | third | third |
| bench-height ratio `H/B` | 0.051 | 0.014 | low | low |
| powder factor `Pf` | 0.048 | 0.008 | not stated | not stated |
| spacing ratio `S/B` | 0.014 | 0.008 | low | low |
| burden ratio `B/D` | 0.006 | 0.004 | lowest, 0.0023 | low |

The ranking agrees with Sui et al.'s: the modulus first for both models, then block size and stemming.
The values differ, as expected: these models are fitted on all 97 rows and Sui et al.'s on a random
80 percent, and the importance type Sui et al. used for XGBoost is not stated.

## 3. Importance on a site the model has not seen

`grouped_resampling_importance` asks how much each input matters to an arm's predictions under
leave-one-site-out. Within one site several inputs are constant, the modulus always, so permuting a
column inside the test fold changes nothing. Instead, for each fold, the held-out site's values of one
feature are replaced by values drawn from the training rows, the arm predicts again, and the increase
in squared error is recorded; ten draws per feature and fold. Shares of the summed positive increases:

| Arm | `E` | `H/B` | `T/B` | `Pf` | `XB` | `B/D` | `S/B` |
|---|---|---|---|---|---|---|---|
| gradient boosting | 0.81 | 0.18 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| random forest | 0.71 | 0.00 | 0.23 | 0.05 | 0.00 | 0.00 | 0.00 |
| classical, transfer factor | 0.41 | 0.22 | 0.00 | 0.37 | 0.00 | 0.00 | 0.00 |
| published regression (in sample) | 0.42 | 0.30 | 0.14 | 0.02 | 0.04 | 0.00 | 0.07 |

The learned arms lean on the modulus far more than the equations do. In this corpus the modulus is a
**site-level constant**: every blast of a campaign carries the same value, and nine distinct values
cover ten campaigns (the two Reocin campaigns share 45 GPa). Under a random split the modulus
therefore identifies the campaign of a test row, and a tree can return that campaign's typical size.
Under leave-one-site-out, eight of the ten held-out sites have a modulus that no training blast
carries, and the tree has no campaign to return. This is a mechanism consistent with the collapse in
the [benchmark](05_protocol-sensitivity.md), stated as a reading of these measurements rather than as
a proof.
