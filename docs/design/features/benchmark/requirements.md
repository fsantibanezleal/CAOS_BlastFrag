# Requirements: the protocol benchmark and the verdict

Retroactive (written 2026-10-05; the code is 0.2.0 to 0.3.0). EARS. Common support, added in 0.4.0, has its own
feature, `features/common-support/`.

| ID | Requirement | Gate |
|---|---|---|
| BM-001 | THE kill criterion text SHALL not change after it was declared, and a test SHALL pin its hash. | `tests/test_benchmark.py::test_the_kill_criterion_text_has_not_been_edited_since_it_was_declared` |
| BM-002 | THE random protocols SHALL be repeated with distinct reproducible seeds and SHALL report the median and the spread. | `tests/test_benchmark.py::test_random_protocols_are_repeated_and_report_the_median`, `tests/test_protocols.py::test_repeated_draws_are_seeded_distinct_and_reproducible` |
| BM-003 | WHEN the criterion's outcome differs between the two row sets, THE verdict SHALL say that it depends on the row set. | `tests/test_benchmark.py::test_the_verdict_depends_on_which_rows_are_scored` |
| BM-004 | THE verdict SHALL list the arms fitted on the corpus itself apart from the arms that transfer, and SHALL never count an in-sample arm as an interval above zero. | `tests/test_benchmark.py::test_in_sample_arms_are_reported_apart_from_the_arms_that_transfer`, `tests/test_benchmark.py::test_no_arm_fitted_without_the_corpus_has_an_interval_above_zero` |
| BM-005 | THE benchmark SHALL fit a fresh model on every fold. | `tests/test_benchmark.py::test_each_fold_gets_a_fresh_model` |
| BM-006 | THE benchmark SHALL stamp the digest of the dataset it ran on. | `tests/test_benchmark.py::test_the_benchmark_stamps_the_dataset_it_ran_on` |
| BM-007 | THE oracle and the null SHALL bracket every arm under every protocol. | `tests/test_benchmark.py::test_the_oracle_and_the_null_bracket_every_protocol` |
| BM-008 | THE leave-one-site-out result SHALL carry an entry for every site, and every refusal SHALL carry its reason. | `tests/test_benchmark.py::test_every_site_has_a_per_site_entry_and_refusals_carry_reasons` |
| BM-009 | THE benchmark SHALL place each published random-split figure within the distribution of reproduced draws of its protocol. | `tests/test_benchmark.py::test_the_published_stacking_figure_is_a_favourable_draw` |
| BM-010 | THE verdict SHALL report the null's anti-correlation with the measurements under leave-one-site-out. | `tests/test_benchmark.py::test_the_null_is_anti_correlated_with_the_truth_under_this_protocol` |
| BM-011 | THE benchmark SHALL show that the classical result does not depend on the held-out site's own rock factor. | `tests/test_benchmark.py::test_the_classical_result_does_not_depend_on_the_held_out_sites_rock_factor` |
| BM-012 | EVERY arm SHALL declare what its free quantities were fitted on. | `tests/test_protocols.py::test_every_arm_declares_what_it_was_fitted_on` |
| BM-013 | THE quantile used for spreads and intervals SHALL be the interpolating definition. | `tests/test_protocols.py::test_the_quantile_is_the_interpolating_definition` |
