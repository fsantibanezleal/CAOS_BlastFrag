# Methods

The prediction ladder. Each page derives its models term by term from a primary source, states where
the model fails, and records anything the source did not print.

1. [The classical rung](methods/01_classical.md), mean size, uniformity and the distributions.
2. [The rock factor](methods/03_rock-factor.md), three published schemes that disagree, and a fourth
   recovered from the data.
3. [The statistical rung](methods/04_statistical.md), the group router and the two published
   regressions.
4. [The learned rung](methods/06_learned.md), the published network reproduced, and what happened.
5. [The protocol-sensitivity benchmark](methods/05_protocol-sensitivity.md), **start here**: the
   result the package exists to produce.

## The ladder at a glance

Transfer is variance explained about the identity line under leave-one-site-out.

| Rung | Tier | Lane | Transfers to an unseen site? |
|---|---|---|---|
| null, predict the training mean | control | live | by construction, no |
| oracle, return the measurement | control | offline | by construction, yes |
| classical mean size | classical | live | **yes, 0.311** |
| Rosin-Rammler distribution | classical | live | shape only |
| Swebrec distribution | classical | live | shape only |
| crush-zone composition | semi-mechanistic | live | shape only |
| rock-factor schemes | classical | live | an input, not a predictor |
| group router | statistical | live | exact on all 109 labelled blasts |
| published regression | statistical | live | **yes, 0.802** |
| refitted regression | statistical | live | no, -4.075 |
| published neural network | learned | offline train, live infer | no, -0.626 |
| support vector regression | learned | offline train, live infer | no |
| random forest and gradient boosting | learned | offline train, live infer | no |
| stacking ensemble | learned | offline train, live infer | no, -0.951 |

## Not implemented, and why

A 2025 hybrid combining a convolutional network, a least-squares support-vector machine and a
Newton-Raphson-based optimiser reports strong numbers on a superset of this corpus. It is **not**
reproduced here: the optimiser's update rule is not transcribable with confidence from the copy held
for this work, and a hand-rolled approximation shipped under that name would be a fabricated method.
It appears in the documentation as prior art with its published figures, attributed.

There is no discrete-element, grain-based or hybrid stress blasting model here either. Those are
mechanistic simulations with no engine, no licence and no reference output available for this work,
and a cheap substitute under one of those names would be worse than their absence.
