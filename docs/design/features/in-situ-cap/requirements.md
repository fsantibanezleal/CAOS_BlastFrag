# Requirements: the in-situ block cap

Status: planned

Designed 2026-10-05, before its code (ADR-0075). The cap is a declared modelling choice of this package, not a
published relation: research dossier `wip/fragmenta/research-2026-09-09-method-survey.md` section 1.8 records that
no held source prints it. EARS.

| ID | Requirement | Gate |
|---|---|---|
| IC-001 | THE capped arm SHALL never predict a mean size larger than the blast's in-situ block size. | `tests/test_in_situ_cap.py::test_no_capped_prediction_exceeds_the_in_situ_block` |
| IC-002 | WHERE the uncapped prediction does not exceed the in-situ block, THE capped arm SHALL return it unchanged. | `tests/test_in_situ_cap.py::test_the_cap_changes_nothing_where_it_does_not_bind` |
| IC-003 | WHEN the cap binds, THE prediction SHALL record the uncapped value and that the cap bound. | `tests/test_in_situ_cap.py::test_a_binding_cap_is_recorded_on_the_prediction` |
| IC-004 | THE capped passing curve SHALL equal the uncapped curve below the in-situ block size and SHALL pass everything at and above it. | `tests/test_in_situ_cap.py::test_the_capped_curve_passes_everything_at_the_block_size` |
| IC-005 | THE capped arm SHALL declare that the cap is a choice of this package and not a published relation. | `tests/test_in_situ_cap.py::test_the_cap_declares_that_it_is_not_published` |
| IC-006 | ON the training corpus, THE cap SHALL bind exactly on the blasts where the classical prediction exceeds the block, which are Rc1, Rc2 and Rc3. | `tests/test_in_situ_cap.py::test_the_cap_binds_on_three_reocin_blasts_and_no_other` |
| IC-007 | THE default ladder SHALL include the capped arm, with the provenance of the arm it caps. | `tests/test_in_situ_cap.py::test_the_capped_arm_is_in_the_default_ladder_with_its_provenance` |
| IC-008 | IF the arm it caps abstains, THEN THE capped arm SHALL abstain with the same reason. | `tests/test_in_situ_cap.py::test_the_capped_arm_abstains_where_its_base_does` |
