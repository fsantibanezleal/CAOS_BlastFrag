# Requirements: the data and the ingestion contract

Retroactive (written 2026-10-05; the code is 0.1.0 to 0.3.0). EARS (Mavin et al., RE'09, doi:10.1109/RE.2009.9).
Every requirement names the test that fails when it is violated.

| ID | Requirement | Gate |
|---|---|---|
| DC-001 | WHEN the training corpus is loaded, THE loader SHALL reproduce the source's printed descriptive statistics and fail on any mismatch. | `tests/test_datasets.py::test_source_integrity_gate_passes_on_the_shipped_corpus`, `tests/test_datasets.py::test_source_integrity_gate_catches_the_defect_it_was_written_for` |
| DC-002 | IF one cell changes in a way the summary statistics would not reveal, THEN THE integrity gate SHALL still fail. | `tests/test_datasets.py::test_source_integrity_gate_catches_a_single_cell_change_the_statistics_would_miss` |
| DC-003 | THE shipped corpus SHALL carry a pinned digest. | `tests/test_datasets.py::test_dataset_digest_is_pinned` |
| DC-004 | WHEN a value lies outside the contract bounds, THE contract SHALL reject the blast as a unit or entry error. | `tests/test_datasets.py::test_the_contract_rejects_a_unit_error_outright` |
| DC-005 | IF a blast lies outside the fitted envelope and the caller has not opted in, THEN THE contract SHALL refuse it; WHERE the caller opts in, THE results SHALL be stamped as extrapolations. | `tests/test_datasets.py::test_the_contract_refuses_an_out_of_envelope_row_unless_asked`, `tests/test_datasets.py::test_field_holdout_is_an_extrapolation_and_says_so` |
| DC-006 | THE two published hold-outs SHALL have their published sizes, and THE one 2012 hold-out blast that is also in the training table SHALL be flagged. | `tests/test_datasets.py::test_holdout_protocols_have_their_published_sizes`, `tests/test_datasets.py::test_the_leaked_blast_is_present_and_flagged_in_the_2012_protocol` |
| DC-007 | THE geometry reconstruction SHALL reproduce every narrative constraint its sources state. | `tests/test_geometry.py::test_the_reconstruction_reproduces_every_published_narrative_constraint` |
| DC-008 | IF a site publishes no hole diameter, THEN THE reconstruction SHALL abstain rather than guess. | `tests/test_geometry.py::test_miami_abstains_rather_than_guessing` |
| DC-009 | IF the reconstruction is wrong, THEN THE verification SHALL fail. | `tests/test_geometry.py::test_a_broken_reconstruction_is_caught` |
| DC-010 | THE duplicate feature vectors in the corpus SHALL be exactly the recorded groups. | `tests/test_datasets.py::test_duplicate_feature_vectors_in_the_corpus_are_exactly_those_recorded` |
| DC-011 | THE prediction type SHALL carry a value or an abstention reason, never both and never neither. | `tests/test_models.py::test_a_prediction_carries_a_value_or_a_reason_but_never_both` |
| DC-012 | IF a scored row has no prediction, THEN THE metric SHALL refuse rather than drop it. | `tests/test_models.py::test_scoring_refuses_a_missing_row_rather_than_dropping_it` |
