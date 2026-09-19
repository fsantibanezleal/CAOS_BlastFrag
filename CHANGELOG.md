# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest on top.

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
