# Methods

The prediction ladder. Each page derives its models term by term from a primary source, states where
the model fails, and records anything the source did not print.

1. [The classical rung](methods/01_classical.md), the mean-size equation and the uniformity index.
2. [Distributions](methods/02_distributions.md), Rosin-Rammler, Swebrec and the crush-zone
   composition: shapes around the classical mean size that no held dataset validates.
3. [The rock factor](methods/03_rock-factor.md), three published schemes that disagree, and a fourth
   recovered from the data.
4. [The statistical rung](methods/04_statistical.md), the group router and the two published
   regressions.
5. [The protocol-sensitivity benchmark](methods/05_protocol-sensitivity.md), **start here**: the
   result the package exists to produce.
6. [The learned rung](methods/06_learned.md), the published network reproduced, and what happened.
7. [The transfer rung](methods/07_transfer.md), the classical equation with a rock factor that uses
   nothing from the target site.
8. [Diagnostics](methods/08_diagnostics.md), the outlier screen and two views of feature importance.

## The ladder at a glance

Held out by site: variance explained about the identity line under leave-one-site-out, over every
blast, with the site-resampled 95 percent interval. Random: the median of 100 random 80/20 draws.

| Rung | Tier | Fitted on | Random | Held out by site |
|---|---|---|---|---|
| null, predict the training mean | control | training rows | -0.036 | -0.216 (-1.28 to -0.15) |
| oracle, return the measurement | control | the measurement | 1.000 | 1.000 |
| classical mean size, site factor | classical | a factor recovered from the site's own published predictions | 0.303 | 0.311 (-0.96 to 0.70) |
| classical mean size, transfer factor | classical | training sites only | 0.312 | 0.298 (-1.10 to 0.72) |
| Rosin-Rammler, Swebrec, crush zone | classical, semi-mechanistic | as the classical mean size | shape only | shape only |
| rock-factor schemes | classical | published ratings | an input | an input |
| group router | statistical | all 97 blasts, by its source | routes | exact on all 109 labelled blasts |
| published regression | statistical | **all 97 blasts, by its source (in sample)** | 0.805 | 0.802, in sample |
| refitted regression | statistical | training rows | 0.686 | -4.075 (-19.48 to 0.02) |
| published neural network | learned | training rows | 0.362 | -0.626 (-3.35 to 0.01) |
| support vector, radial and polynomial | learned | training rows | 0.665, 0.395 | -0.387, -4.546 |
| random forest | learned | training rows | 0.749 | -0.231 (-2.78 to 0.38) |
| gradient boosting | learned | training rows | 0.704 | -0.034 (-2.23 to 0.42) |
| stacking ensemble | learned | training rows | 0.703 | -0.035 (-2.25 to 0.42) |

Apart from the in-sample regression, no interval excludes zero: ten sites do not separate any of these
arms from predicting the corpus mean, or from each other.

## Not implemented, and why

A 2025 hybrid combining a convolutional network, a least-squares support-vector machine and a
Newton-Raphson-based optimiser reports strong numbers on a superset of this corpus. It is **not**
reproduced here: the optimiser's update rule is not transcribable with confidence from the copy held
for this work, and a hand-rolled approximation shipped under that name would be a fabricated method.
It appears in the documentation as prior art with its published figures, attributed.

There is no discrete-element, grain-based or hybrid stress blasting model here either. Those are
mechanistic simulations with no engine, no licence and no reference output available for this work,
and a cheap substitute under one of those names would be worse than their absence.
