# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest on top.

## [0.04.000] - 2026-10-05

Three gaps the 0.3.0 audit left open, each designed before its code in the new design document, and that
document itself.

### Added

- `docs/design/SDD.md` (ADR-0075): the design of the package, written retroactively for 0.1.0 to 0.3.0 and before
  the code for this release, with seven feature documents whose 66 requirements each name the test that holds it.
  `scripts/check_sdd.py` fails CI when a named test does not exist.
- The in-situ block cap, a declared modelling choice rather than a published relation: no fragment is predicted
  larger than its in-situ block. `cap_at_in_situ_block` clamps a passing curve at the block size;
  `InSituCap(arm)` caps an arm's mean size and records where it bound; `kuznetsov-capped` joins the default
  ladder. On the corpus it binds on Rc1, Rc2 and Rc3 only (predicted 0.72 to 0.81 m against a 0.68 m block,
  measured 0.46 to 0.48 m) and lifts the classical arm from 0.311 to 0.352 held out by site and from 0.303 to
  0.399 at the median random draw; its interval still spans zero.
- Common support: every arm under every protocol is also scored on the rows every size-predicting arm answered
  (`detail["common"]`, per draw for the random protocols, with a site interval held out by site). Held out by site
  those are 79 blasts from nine sites; the tree models rise to 0.21 there because the rows other arms refuse are
  the hard ones, which is why the criterion stays on the two row sets declared before the run.
- `network_width_sweep`: the published network's hidden width under the source's protocol and held out by site.
  The source's procedure, reproduced, picks 8 and 11 hidden units rather than the published 9 and 7, and held out
  by site no width from 6 to 15 explains any variance (-0.57 to -1.49; the published pair -0.626).

### Changed

- The classical method page's closing section said the arm scores 0.311 held out by site "because it has nothing
  to overfit" and ranked it second to the in-sample regression; it now states the protocol-invariance of a
  fixed-coefficient arm, the transfer arm's 0.298 and the interval.
- The benchmark's timing note: pin BLAS to one thread. On a loaded machine one network fit at width 15 took more
  than six minutes multi-threaded against 1.9 s on one thread; the default benchmark takes about three minutes on
  one thread.

## [0.03.000] - 2026-10-04

An adversarial re-run of 0.2.2 found that several reported effects came from the evaluation design
rather than from the models. This release changes what the benchmark measures and reports; the kill
criterion's text is unchanged and a test now pins its hash.

### Added

- Repeated random protocols: `repeated_random_splits`, `repeated_deduplicated_splits`, and
  `run_benchmark(n_repeats=100)` scoring every draw on its own and reporting the spread
  (`detail["repeats"]`, `summarise_draws`, `quantile`). The protocol gap is computed from the median.
- Two supports for every leave-one-site-out score (`SUPPORTS`): every blast, and the blasts with
  resolvable geometry, where the classical arms can also answer. The criterion is evaluated on both
  and `depends_on_support` reports a disagreement.
- Site-resampled 95 percent intervals on every grouped score (`n_boot=2000`), per-site error for every
  arm, and every out-of-fold prediction.
- Arm provenance: `fitted_on`, `in_sample_corpus`, `uses_site_constant`, `router_in_sample`, and
  `shares_mean_size_with` on Kuz-Ram, Swebrec and the crush-zone composition. The verdict reports
  in-sample arms separately (`in_sample_arms`) and never lists them as transferring.
- `KuznetsovTransfer`: the classical equation with a rock factor predicted from Young's modulus by a
  log-linear fit over the training sites only. Held out by site it scores 0.298 against 0.311 for the
  arm that reads the held-out site's own factor.
- `blastfrag.diagnostics`: an Isolation Forest screen (reported, never applied as a filter),
  model-native feature importance, and resampling importance under leave-one-site-out.
- `blastfrag.export`: every fitted learned arm as plain JSON, with a dependency-free reference
  walker (`predict_portable`) that reproduces the forest, boosting and stacked arms exactly and the
  network and support-vector arms to 1e-12. Schema `blastfrag.portable/v1`.
- Docs: the protocol page rewritten with the definitions and the 0.3.0 results, new pages for the
  distributions, the transfer rung, the diagnostics and the portable export, two guides, and a
  generated figure of the three protocols.

### Changed

- **The stacking arm is built as the source describes**: base learners fitted on all training rows and
  the linear meta-learner on their in-sample predictions, the cross-validation the source cancelled.
  0.2.x used scikit-learn's out-of-fold stacking with two folds, a different method. Built this way
  the meta-learner gives the boosting learner a weight of 1.02, and held out by site the arm scores
  -0.035 instead of -0.951.
- The verdict's outcome sentence reports the row-set dependence, the absence of any interval above
  zero outside in-sample arms, and the null model's anti-correlation (-0.79) under leave-one-site-out.
- Bootstrap intervals use the interpolating quantile instead of an off-by-one index.
- CI follows ADR-0074: one Python version, no scikit-learn or xgboost installed, slow and training
  tests excluded; the full suite runs locally before every push.

### Removed

- The claims that the classical arm "improves when held out by site" and that deduplication raises
  the learned scores. Both were read off the single seed-0 draw; over 100 draws the classical arm
  scores 0.303 at the median random draw against 0.311 held out by site, and deduplication moves no
  forest, boosting or support-vector median by more than 0.015.
- `published-regression` from the list of arms that transfer: its coefficients were fitted by its
  source on these 97 blasts.
- The standalone forest's and boosting arm's comparison with the source's single-learner figures
  (0.797, 0.758): the source prints two parameter sets for those learners and does not say which
  produced them. `PUBLISHED_RANDOM_SPLIT_R2` keeps the stacking (0.943) and polynomial SVR (0.578)
  figures, whose parameters are unambiguous; 0.943 lies above all 100 reproduced draws.
- The silent fallback from XGBoost to scikit-learn's gradient boosting in the stacking arm.

## [0.02.002] - 2026-09-26

### Changed

- First release published on PyPI (trusted publishing, `publish.yml`). The tag now covers what main carries:
  the CI budget rules (ADR-0074), the archetype content guard in CI (ADR-0067) and the swept `.gitignore`.

## [0.02.001] - 2026-09-19

### Changed

- The package summary, the README, the module docstring, code comments, the docs wiki, a test
  name and a test docstring no longer use the word "honest"; each passage now says what it
  means: the statistic named (variance explained, not squared correlation), leave-one-site-out
  (held out by site), the leakage-free protocols, the other folds of a pooled score. No behaviour
  change.

## [0.02.000] - 2026-09-09

### Added

- The full ladder behind one interface: two controls, the classical rung, three distributions,
  three rock-factor schemes, the published group router, two regressions and five learned arms.
- The protocol-sensitivity benchmark, which is what the package exists to produce. Three split
  protocols, pooled folds, a kill criterion declared before the run, and a verdict.
- Levenberg-Marquardt training for the published network, written in numpy, with its Jacobian
  checked against a central finite difference.
- The metric suite. It never returns a bare variance figure, always names the statistic, and always
  runs a null model beside every arm. Bootstrap intervals resample the blast or the SITE.
- A plausibility guard: a prediction outside 0.001 to 3 m is refused with a reason rather than
  scored. One refitted fold extrapolated to 10.48 m.
- The docs wiki for every rung, written alongside the code.

### Findings

- **The learned tier does not generalise across sites.** Zero of six learned arms have positive
  variance explained under leave-one-site-out. The only two arms that transfer are the two whose
  coefficients are fixed rather than fitted: the published regression at 0.802 and the classical
  equation at 0.311. The classical equation gets BETTER under the honest protocol, from -0.027 on a
  random split, because it has nothing to overfit.
- **The published equation beats the numbers its own papers printed for it**, by 0.107 and 0.119 in
  variance explained on the two published hold-outs. Where the two papers disagree with each other,
  the recomputation lands on the earlier one four times out of four.
- **The published network's hold-out score is not reachable across thirty seeds.** The reproduction
  runs 0.167 to 0.636 against a published 0.910, and the shortfall concentrates in the two rows the
  source itself reports as its most unstable.
- **The group router is exact**, zero misassignments on all 109 labelled blasts.
- One hold-out row is irreproducible by both published models in the same direction, and no
  single-input correction reconciles them. Carried unresolved.

### Fixed

- The kill criterion, which as first written could pass on two failures. It asked only for a margin
  over the null; the first run produced a best learned arm at -0.034 against a null at -0.216 and
  the rule declared success. It now requires a positive score as well.

## [0.01.000] - 2026-09-09

### Added

- The three real corpora, as package data with full provenance: 97 published bench blasts, the
  14-row union of two published validation sets, and 5 field blasts from a CC BY source.
- A two-part source-integrity gate. It reproduces the source paper's own descriptive statistics from
  the shipped rows and compares a pinned content digest. Tested by reintroducing the original defect
  and requiring the gate to fail.
- The ingestion contract, with rejection bounds separate from the fitted envelope. Predicting outside
  the data requires an explicit opt-in and stamps every result.
- Absolute geometry reconstruction from the published per-site hole diameters, asserted against 15
  dimensional constraints the source states in prose. Nine sites reconstruct, one abstains.
- Typed records for blasts, patterns, rock, explosives, predictions and size distributions.
  A prediction carries either a value or a reason to abstain, never neither and never both.

### Fixed

- Five transcription defects in the assembled corpus against the published tables, two of them on
  the regression target: Db5 powder factor 0.33 to 0.39, Sm4 size 0.24 to 0.22, Ad15 size 0.22 to
  0.15, Ad17 powder factor 1.07 to 1.24, and Ad19 powder factor 1.47 to 1.26, where the row's own
  in situ block size had been copied into the neighbouring column.
