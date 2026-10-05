# Requirements: the network width sweep

Designed 2026-10-05, before its code (ADR-0075), to close audit item A-12: the published network's hidden widths
are fixed at the source's optima (9 and 7), chosen on the source's own hold-out, and the sweep its docstring
describes was never shown. EARS.

| ID | Requirement | Gate |
|---|---|---|
| WS-001 | THE sweep SHALL fit the published network at every requested width, with the requested number of simulations at each. | `tests/test_width_sweep.py::test_the_sweep_fits_every_width_with_its_simulations` |
| WS-002 | THE sweep SHALL reproduce the source's protocol: train on the corpus, choose per group the width with the lowest RMSE on the 2012 hold-out, and report that choice beside the published optimum. | `tests/test_width_sweep.py::test_the_published_protocol_reports_its_choice_beside_the_published_optimum` |
| WS-003 | THE sweep SHALL score every width under leave-one-site-out, pooled, on both declared row sets. | `tests/test_width_sweep.py::test_every_width_is_scored_held_out_by_site_on_both_row_sets` |
| WS-004 | WHEN a width is scored held out by site, THE network SHALL be fitted without the held-out site. | `tests/test_width_sweep.py::test_no_fold_of_the_sweep_sees_its_held_out_site` |
| WS-005 | THE sweep SHALL return the same numbers for the same seed. | `tests/test_width_sweep.py::test_the_sweep_is_reproducible_for_a_seed` |
| WS-006 | THE sweep SHALL mark the published widths in its leave-one-site-out table. | `tests/test_width_sweep.py::test_the_published_widths_are_marked` |
