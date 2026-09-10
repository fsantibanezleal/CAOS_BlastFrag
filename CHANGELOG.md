# Changelog

All notable changes to this project. Format follows Keep a Changelog; newest on top.

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
