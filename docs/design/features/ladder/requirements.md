# Requirements: the method ladder

Retroactive (written 2026-10-05; the code is 0.1.0 to 0.3.0). One acceptance criterion per method (ADR-0069), in
EARS. The in-situ cap of 0.4.0 has its own feature, `features/in-situ-cap/`.

| ID | Requirement | Gate |
|---|---|---|
| LD-001 | THE classical mean-size arm SHALL reproduce a published classical prediction. | `tests/test_models.py::test_the_classical_equation_reproduces_a_published_prediction` |
| LD-002 | IF a blast has no absolute geometry, THEN THE classical arm SHALL abstain with the reason. | `tests/test_models.py::test_the_classical_arm_abstains_where_the_scale_is_unpublished` |
| LD-003 | THE back-solved site rock factor SHALL be near constant within each site and inside the published valid range. | `tests/test_models.py::test_the_back_solved_rock_factor_is_near_constant_within_a_site`, `tests/test_models.py::test_every_recovered_rock_factor_is_inside_the_published_valid_range` |
| LD-004 | WHEN a site is held out, THE transfer arm SHALL fit its rock-factor line without that site. | `tests/test_benchmark.py::test_the_transfer_arm_never_sees_the_held_out_site` |
| LD-005 | THE Rosin-Rammler curve SHALL pass half at the mean size, and THE Swebrec curve SHALL pass half at the mean size and everything at its limit. | `tests/test_models.py::test_rosin_rammler_passes_half_at_the_mean_size`, `tests/test_models.py::test_swebrec_passes_half_at_the_mean_and_everything_at_the_limit` |
| LD-006 | THE crush-zone composition SHALL add fines to the classical curve and state that its constants are not published. | `tests/test_models.py::test_the_crush_zone_composition_adds_fines_and_says_its_constants_are_not_published` |
| LD-007 | THE benchmark SHALL leave out the curve arms that only reshape the classical mean size. | `tests/test_benchmark.py::test_the_distribution_arms_are_not_benchmarked_as_separate_predictors` |
| LD-008 | THE router SHALL reproduce every published group membership and SHALL abstain on size. | `tests/test_models.py::test_the_discriminant_reproduces_every_published_group_membership`, `tests/test_models.py::test_the_discriminant_abstains_on_size_because_that_is_not_what_it_predicts` |
| LD-009 | THE published regression SHALL beat the numbers its own papers printed, and IF a group has too few rows, THEN THE refit SHALL refuse. | `tests/test_models.py::test_the_published_equation_beats_the_numbers_its_own_papers_printed`, `tests/test_models.py::test_the_refit_refuses_when_a_group_has_too_few_rows` |
| LD-010 | THE published network SHALL reproduce most hold-out rows closely and SHALL clamp its output to the target range it was fitted on. | `tests/test_learned.py::test_most_holdout_rows_reproduce_the_published_network_closely`, `tests/test_learned.py::test_predictions_are_clamped_to_the_target_range_the_network_was_fitted_on` |
| LD-011 | THE network SHALL draw its seed before training, so a fit does not depend on how many models were fitted before it. | `tests/test_learned.py::test_the_seed_is_drawn_before_training_so_fits_are_order_independent` |
| LD-012 | THE stacking arm SHALL use the published base parameters and train its meta-learner on in-sample predictions. | `tests/test_learned.py::test_the_stacking_base_learners_use_the_final_published_parameters`, `tests/test_learned.py::test_the_stacking_meta_learner_is_trained_on_in_sample_predictions` |
| LD-013 | IF a learned arm is asked to fit too few rows, THEN THE arm SHALL refuse. | `tests/test_learned.py::test_every_learned_arm_refuses_to_fit_on_too_few_rows` |
| LD-014 | THE oracle SHALL score perfectly and THE null SHALL explain nothing, by construction. | `tests/test_models.py::test_the_oracle_scores_perfectly_which_proves_the_harness`, `tests/test_models.py::test_the_null_model_explains_nothing_by_construction` |
| LD-015 | IF a model returns a value outside the plausible fragment range, THEN THE arm SHALL abstain with the reason. | `tests/test_benchmark.py::test_a_physically_impossible_prediction_is_refused_not_scored` |
