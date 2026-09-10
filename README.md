# blastfrag

[![CI](https://github.com/fsantibanezleal/CAOS_BlastFrag/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_BlastFrag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Blast-fragmentation prediction from a bench pattern. The published ladder from the 1973 classical
mean-size equation to the 2025 stacking ensemble, scored on real measured blasts under three split
protocols, with the honest statistic named.

```bash
pip install blastfrag
```

## Why this exists

Predicting the mean fragment size of a muckpile from a drill-and-blast design is a solved-looking
problem with a large literature and one widely used model. Working through that literature against
the data it is fitted on turns up things a formula collection would hide.

**The classical model barely beats a constant.** On the published twelve-blast hold-out it explains
0.232 of the variance about the identity line, and its root-mean-square error of 0.1279 m improves on
simply predicting the training mean by 13 percent. The figure usually quoted for it, 0.57, is a
squared correlation, which is a different quantity: a model can correlate at 0.755 and still be badly
biased, and this one is.

**The corpus is dimensionless, so the classical model could not run on it at all.** It needs rock
volume and charge mass per hole and the published table has only ratios. The source's own prose gives
a hole diameter for eight of its ten sites, which closes the system, and the reconstruction is
asserted against fifteen dimensional constraints the same prose states.

**The rock factors both source papers say they estimated were never printed.** Back-solving them from
the published predictions recovers values that are near constant within each site and ordered by rock
stiffness.

## What it does

```python
import blastfrag as bf

train = bf.load_training_corpus()          # 97 real bench blasts, ten sites
holdout = bf.load_holdout(protocol="2012") # the published validation set, leakage flagged

pattern = bf.reconstruct_pattern(train[0])
pattern.rock_volume_m3, pattern.charge_mass_kg
```

Loading the training corpus runs a two-part integrity gate. The first part reproduces the source
paper's own descriptive statistics from the shipped rows; the second compares a content digest
against a pinned value. The first says the file still is the published table, the second says it has
not moved since it was corrected.

That gate found five transcription defects in the corpus as it was first assembled, two of them on
the regression target.

## Refusing rather than guessing

A model that needs a rock volume cannot run on a site whose scale is unpublished. One of the ten
sites is in exactly that position, and its six rows are the geometry negative control:

```python
miami = [b for b in train if b.site == "Miami"][0]
bf.reconstruct_pattern(miami)   # raises GeometryUnavailable, with the reason
```

Abstention is a first-class result throughout: a `Prediction` carries either a value or a reason, and
never a number that was filled in to avoid a hole.

## Naming the statistic

Two different quantities are called `R2` in this literature and they differ by a factor of two and a
half on the same twelve blasts. Every figure this package returns carries the name of what it is, and
a null model that predicts the training mean runs beside every other arm.

## Datasets

| Set | Rows | Source |
|---|---|---|
| training corpus | 97 | Hudaverdi, Kulatilake and Kuzu 2010, `doi:10.1002/nag.957`, Tables I and II |
| published hold-out | 14 | the union of the 2010 Table VIII and the 2012 Tables 4 and 5, `doi:10.1007/s10706-012-9496-3` |
| field hold-out | 5 | Sui, Zhou, Zhao, Yang and Zou 2025, `doi:10.3390/app15031254`, CC BY 4.0 |

Numeric values are experimental facts reused with citation. The source PDFs are not redistributed.

The field hold-out is an extrapolation by construction: its Young modulus of 5.6 GPa sits below the
corpus minimum of 9.57 GPa, on the feature two independent 2025 studies both rank most important.
Predictions on it are stamped.

## Documentation

The [`docs/`](docs/README.md) wiki carries the theory, every equation term by term with its source,
the data contract, and the reasoning behind each modelling choice.

## Honest scope

No mechanistic simulation: there is no discrete-element or hybrid stress blasting model here, and a
hand-rolled approximation under those names would be worse than nothing. No non-ideal detonics. No
flyrock and no ground vibration. Model constants that no primary source prints, including the timing
factor of the modified classical model and the crush-zone branch parameters, are exposed as
user-supplied values with documented ranges rather than invented.

## Licence

MIT. See [LICENSE](LICENSE).
