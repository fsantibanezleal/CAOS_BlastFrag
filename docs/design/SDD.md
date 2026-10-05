# Software design document: blastfrag

ADR-0075 asks for this document before the code. **For everything up to 0.3.0 it was written afterwards, on
2026-10-05**, from the code, the tests and the research dossiers that preceded them, and it says so: each
retroactive requirement names the test that already held it, and its convergence was checked by running that
test, not assumed. The three features of 0.4.0 (the in-situ cap, the network width sweep, common support) were
designed here first, in `features/`, before their code.

## Problem and non-goals

blastfrag predicts the mean fragment size `x50` of a bench blast from its design and its rock with the published
ladder of predictors, and measures what each prediction is worth at a mine the model has not seen. It is the
engine of the Fragmenta product; a product declares no package of its own (ADR-0057), so the science is here.

Non-goals, each something a reader could reasonably assume:

- **A mechanistic simulator.** No discrete-element, grain-based or hybrid stress model is implemented; no engine
  or reference output for one was available.
- **A validated passing curve.** No held dataset carries a measured size distribution, so the curve shapes
  (Rosin-Rammler, Swebrec, the crush zone, the in-situ cap) are models, not validated predictions.
- **Published constants that no held source prints.** The timing factor, the crush-zone branch constants and the
  in-situ cap are the caller's or declared choices of this package, labelled on every result.
- **A production design tool.** With ten sites no arm fitted without the corpus has a site-resampled interval
  above zero; nothing here is validated for a new mine.
- **A training stack in the core.** The core is numpy only; scikit-learn and XGBoost are extras.

## Contracts

- **Ingestion** (`blastfrag/datasets.py`, `validate_blast`): every blast is checked against contract bounds (a
  unit or entry error, rejected outright) and against the fitted envelope, the published minima and maxima of the
  source's own descriptive statistics. Out of the envelope is rejected unless the caller passes
  `allow_extrapolation=True`, and then every result is stamped. The shipped corpus must reproduce the source's
  printed descriptive statistics on every load (the source-integrity gate) and carries a pinned digest.
- **Results** (`blastfrag/types.py`): a `Prediction` carries a value or an abstention reason, never both and never
  neither; a `SizeDistribution` is a non-decreasing passing curve on a shared sieve grid. `score` refuses a missing
  row rather than dropping it.
- **Portable models** (`blastfrag/export.py`, schema `blastfrag.portable/v1`): every fitted learned arm as plain
  JSON with its input order, walked by a dependency-free reference (`predict_portable`) exactly for the trees and
  to 1e-12 for the network and the kernels.
- **Benchmark** (`blastfrag/benchmark.py`): `BenchmarkResult` with every arm under every protocol, the verdict on
  the declared criterion, the dataset digest and each arm's provenance.

## Lanes

A library has no web lane of its own; it declares what its consumers may run where.

- **Offline**: fitting the learned arms and the benchmark (191 s for the default 100 draws with BLAS pinned to one
  thread, measured on 2026-10-05, and about two minutes more for the network width sweep; see Risks). Never in CI
  (ADR-0074): CI runs the numpy-only tests.
- **Live in a consumer's browser**: the closed forms, which a consumer ports and holds to the engine by a parity
  test, and every learned arm through the portable export.
- **Replayed**: the benchmark and the per-case predictions a consumer bakes and commits.

## Method ladder

One acceptance criterion per method, each held by a named test in `features/ladder/requirements.md`:

| Method | Source | Accepted when |
|---|---|---|
| Kuznetsov mean size with Cunningham's strength term | Hudaverdi et al. 2010 Eq. 1 | it reproduces a published prediction and abstains where no absolute geometry exists |
| Site rock factor, back-solved | the published classical columns | it is near constant within a site and inside the published valid range |
| Transfer rock factor | a calibration of this package | it never reads the held-out site |
| In-situ cap (0.4.0) | declared choice | no prediction exceeds the in-situ block, and it equals the uncapped arm wherever it does not bind |
| Rosin-Rammler, Swebrec, crush zone | Amoako et al. 2022; Ouchterlony 2005 | they pass half at the mean size and are not benchmarked as separate predictors |
| Discriminant router | Hudaverdi et al. 2010 Eq. 8 | it reproduces every published membership and abstains on size |
| Published and refitted regressions | Hudaverdi et al. 2010 Eqs. 9 and 10 | the published one beats its own papers' tables; the refit refuses too few rows |
| Published network | Kulatilake et al. 2012 | it reproduces most hold-out rows closely, clamps to its target range, and its width choice is reported (0.4.0) |
| Support vector, forest, boosting, stacking | Amoako et al. 2022; Sui et al. 2025 | they use the published parameters; the stack trains its meta-learner on in-sample predictions |
| Null and oracle | controls | they bracket every protocol |

## Cases

For an engine the cases are the datasets and the protocols:

- the training corpus, 97 blasts from ten campaigns (Hudaverdi et al. 2010, doi:10.1002/nag.957);
- the two published hold-outs, 13 blasts (2010) and 12 blasts (2012, doi:10.1007/s10706-012-9496-3), 14
  distinct blasts in all, one of which (Rc1, in the 2012 set) is also in the training table and is flagged;
- the field set, five blasts outside the envelope (Sui et al. 2025, doi:10.3390/app15031254), an extrapolation
  check by construction;
- the protocols: 100 random 80/20 draws, 100 draws after collapsing repeated input vectors, and leave one site
  out, each grouped score on two row sets (every blast; the blasts with resolvable geometry).

## Oracles

None of these is a model judging a model:

- **The source's own descriptive statistics** decide the corpus is the published one (they caught five
  transcription errors).
- **The source's narrative constraints** (bench heights, burden and spacing ranges at Murgul and Dongri-Buzurg,
  15 in all) decide the geometry reconstruction is right.
- **The published predictions** decide a transcribed equation is the published one.
- **The oracle arm** (the measured value) and **the null** (the training mean) bracket every protocol, which
  proves the harness.
- **The reference walker** decides a portable model is the fitted model.

## Deploy driver

PyPI, by trusted publishing on a published GitHub release (`.github/workflows/publish.yml`), because the product
consumes the engine as a pinned dependency (ADR-0057, ADR-0061). The measurement behind "numpy only in the core"
is the CI job that installs the core alone and runs the full integrity and reconstruction path.

## Risks and kill criteria

- **The kill criterion**, declared in `KILL_CRITERION` and pinned by a hash test: the learned tier counts as
  generalising across sites only if the best learned arm's variance explained under leave-one-site-out is both
  positive and at least 0.10 above the null's. Its positivity half was added after the first run declared success
  for an arm worse than a constant; the text has not changed since.
- **A verdict that rests on one row set**: every grouped score is reported on both supports, and a verdict that
  differs between them says so.
- **A single lucky split**: the random protocols are repeated and reported as a spread.
- **Thread oversubscription.** On a loaded machine multi-threaded BLAS spins on the network's tiny matrices: one
  fit at width 15 took more than six minutes against 1.9 s single-threaded (2026-10-05). Consumers pin BLAS to one
  thread for a bake.

## ADR fit

| ADR | Rule | Where it fits here | Carried by |
|---|---|---|---|
| ADR-0057 | science in a separate package, two contracts | the whole repo | `blastfrag/`, the contracts above |
| ADR-0061 | PyPI by trusted publishing | release | `.github/workflows/publish.yml` |
| ADR-0067 | no em-dash, no emoji | every tracked file | `scripts/check_content_standards.py` |
| ADR-0069 | method vertical with an acceptance criterion | every arm | `features/ladder/requirements.md` |
| ADR-0074 | CI runs cheap checks only, no training | CI | `.github/workflows/ci.yml` (`-m "not slow and not trains"`) |
| ADR-0075 | every requirement names its gate | this document | `scripts/check_sdd.py` in CI |
| ADR-0016, ADR-0071, ADR-0078 | the web shell and its UI floor | do not apply: a library has no UI | none |
