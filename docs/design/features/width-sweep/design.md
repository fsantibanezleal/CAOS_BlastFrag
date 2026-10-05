# Design: the network width sweep

## Why

Kulatilake, Hudaverdi and Wu 2012 swept the hidden width over 6 to 15, the range two published heuristics allow,
with eight simulations per width, and chose the width per rock-stiffness group on their hold-out by RMSE and
correlation: 9 for the high-modulus group, 7 for the low. That choice was made on the rows it is then scored on,
and its stability is part of the claim. Reproduced on 2026-10-05 (probe, one BLAS thread), the source's own
procedure lands on 8 and 11 rather than 9 and 7, and RMSE moves non-monotonically between adjacent widths (0.037
at 8 and 0.102 at 9 for the high-modulus group).

## What

`blastfrag.benchmark.network_width_sweep(corpus, holdout, *, widths=range(6, 16), n_simulations=8, seed=0)`
returns:

- `published_protocol`: per group, the table of RMSE and correlation per width on the 2012 hold-out, the width
  with the lowest RMSE, and the published optimum (it calls `PublishedNeuralNetwork.sweep_hidden_width`);
- `leave_one_site_out`: per width (the same width in both groups), the pooled out-of-fold variance explained on
  the `all` and `geometry` supports, the number of rows scored and abstained;
- `published_widths`: `{1: 9, 2: 7}`, and the configuration (`widths`, `n_simulations`, `seed`).

The held-out fits reuse `leave_one_site_out` from `blastfrag/splits.py`, so no fold sees its held-out site. The
published-width row in the held-out table is the network as the benchmark runs it (`hidden={1: 9, 2: 7}`), so the
two can be compared.

## Cost

Ten widths times ten folds is 100 fits of 16 networks, plus ten fits for the source's protocol: about two
minutes with BLAS pinned to one thread on the default device. It is offline only.
