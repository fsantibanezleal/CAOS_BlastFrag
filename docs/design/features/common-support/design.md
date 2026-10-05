# Design: common support

## Why

`score` drops each arm's abstentions, so two arms in one table are scored on different rows. Held out by site
the declared `geometry` support makes the classical arms comparable with the fitted ones, but the polynomial
kernel also refuses rows where it leaves the plausible range, and the random draws had no comparison on shared
rows at all.

## What

The size-predicting arms of a protocol are the arms that answer at least one row in it; the router, which
predicts a group, answers none and is left out. The common rows are the intersection of the rows those arms
answered.

- **Repeated protocols**: `_score_repeated` keeps each draw's predictions; after every arm has run, each draw's
  common rows are computed and every arm is rescored on them. Each arm's `detail["common"]` carries the spread
  (`summarise_draws` of the per-draw variance explained) and `n_rows` (the spread of the common-row counts).
- **Leave one site out**: after every arm has run, the common rows of the pooled out-of-fold predictions are
  computed once; each arm's `detail["common"]` carries the score, the site-resampled interval and `n_rows`.
- The verdict is unchanged: `SUPPORTS` stays the two declared row sets, so the criterion is not re-evaluated on a
  row set chosen after the run.

The kept predictions are removed from `detail` once the common scores are written, so the result does not grow
by the predictions of every draw.
