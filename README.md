# blastfrag

[![CI](https://github.com/fsantibanezleal/CAOS_BlastFrag/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_BlastFrag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Blast-fragmentation prediction from a bench pattern. The published ladder from the 1973 classical
mean-size equation to the 2025 stacking ensemble, scored on real measured blasts under three split
protocols, with the statistic named (variance explained, not squared correlation).

```bash
pip install blastfrag
```

## Why this exists

Predicting the mean fragment size of a muckpile from a drill-and-blast design matters downstream:
Hudaverdi, Kulatilake and Kuzu 2010 (`doi:10.1002/nag.957`) write that blasting has a significant
impact on loading, crushing and grinding, and that a suitable size distribution raises crusher and
mill throughput and lowers comminution energy. The literature offers one widely used closed-form model
and, since 2012, a series of learned ones fitted to the same 97-blast corpus. Working through it
against that corpus turns up things a formula collection would hide.

**How a model is scored decides what it appears to do.** On random 80/20 splits of the corpus, the
learned arms explain a median 0.66 to 0.75 of the variance over 100 draws. With each of the ten
campaigns held out in turn, every one of them loses between 0.74 and 4.94 of that, and the best lands
within a few hundredths of zero. The 2025 headline of 0.943 sits above every one of 100 reproductions
of its own protocol.

**Ten sites cannot separate the survivors.** The classical mean-size equation scores about 0.30 under
every protocol, whether its rock factor comes from the site itself or is predicted from Young's
modulus over the other sites. Site-resampled intervals put it, and every learned arm, across zero.
Whether the learned tier meets the declared criterion depends on whether one six-blast campaign is
scored.

**The classical equation predicts fragments larger than the blocks they come from.** It does not read
the in-situ block size, and on three Reocin blasts it predicts a mean size above it, while no measured
mean size anywhere exceeds its block. Capping it at the block, a declared choice of this package rather
than a published relation, lifts its held-out score from 0.311 to 0.352; the interval still spans zero.

**The published network's width is not where its own procedure lands.** Reproduced, the source's
selection on its hold-out picks 8 and 11 hidden units, not the published 9 and 7, and held out by site
no width from 6 to 15 explains any variance.

**The classical model, as published, barely beats a constant.** On the published twelve-blast
hold-out it explains 0.232 of the variance about the identity line, and its root-mean-square error of
0.1279 m improves on predicting the training mean by 13 percent. The figure usually quoted for it,
0.57, is a squared correlation, a different quantity.

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

## The benchmark

```python
result = bf.run_benchmark(train, bf.default_arms(), n_repeats=100, n_boot=2000)
print(result.verdict["outcome"])
```

Random protocols are repeated and reported by their spread; leave-one-site-out is pooled and scored
on every blast and on the blasts with resolvable geometry, each with a site-resampled interval; every
score is also reported on the rows every size-predicting arm answered; every arm declares what it was
fitted on, and arms fitted on the corpus itself are reported apart from the arms that transfer. See
[the protocol-sensitivity benchmark](docs/methods/05_protocol-sensitivity.md).
`bf.network_width_sweep(train, holdout)` sweeps the published network's hidden width under the
source's protocol and held out by site ([the learned rung](docs/methods/06_learned.md), section 2.5).

Pin BLAS to one thread for these runs (`OPENBLAS_NUM_THREADS=1`, likewise `OMP_` and `MKL_`). The
network's matrices are tiny, and on a loaded machine a multi-threaded BLAS spends its time spinning:
one fit at width 15 took more than six minutes against 1.9 s on one thread (2026-10-05). On one
thread the default benchmark takes about three minutes and the width sweep about two.

## Running a fitted arm elsewhere

`bf.export_arm(arm)` writes a fitted learned arm as plain JSON (network weights, support vectors, flat
tree arrays) and `bf.predict_portable(document, blast)` reads it with no dependency. The walker
reproduces the forest, the boosting arm and the stacked model exactly, and the rest to $10^{-12}$; see
[the portable export](docs/data/03_portable-export.md).

## Documentation

The [`docs/`](docs/README.md) wiki carries the theory, every equation term by term with its source,
the data contract, the benchmark's definitions and results, and the reasoning behind each modelling
choice.

## Scope

No mechanistic simulation: there is no discrete-element or hybrid stress blasting model here, and no
engine, licence or reference output was available to build one on. No non-ideal detonics. No
flyrock and no ground vibration. Model constants that no primary source prints, including the timing
factor of the modified classical model and the crush-zone branch parameters, are exposed as
user-supplied values with documented ranges rather than invented. The in-situ cap is a declared
modelling choice, labelled on every result. The design document is [`docs/design/SDD.md`](docs/design/SDD.md).

## Licence

MIT. See [LICENSE](LICENSE).
