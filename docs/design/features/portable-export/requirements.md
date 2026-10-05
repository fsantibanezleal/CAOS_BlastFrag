# Requirements: the portable export

Retroactive (written 2026-10-05; the code is 0.3.0). EARS.

| ID | Requirement | Gate |
|---|---|---|
| PE-001 | THE export of the refitted power law SHALL reproduce the fitted arm exactly. | `tests/test_export.py::test_the_refitted_power_law_round_trips_exactly` |
| PE-002 | THE export of the published network SHALL reproduce the fitted network through the reference walker. | `tests/test_export.py::test_the_published_network_round_trips` |
| PE-003 | THE export of every scikit-learn and boosting arm SHALL reproduce the fitted model through the reference walker. | `tests/test_export.py::test_every_scikit_and_boosting_arm_round_trips` |
| PE-004 | THE export of the transfer arm SHALL carry its rock-factor fit. | `tests/test_export.py::test_the_transfer_arm_exports_its_rock_factor_fit` |
| PE-005 | IF an arm is unfitted or of a kind the schema does not cover, THEN THE export SHALL refuse it. | `tests/test_export.py::test_an_unfitted_or_unsupported_arm_is_refused` |
| PE-006 | EVERY exported document SHALL carry the schema name and the input order. | `tests/test_export.py::test_every_document_carries_the_schema_and_the_input_order` |
| PE-007 | THE single-precision rounding the walker applies to tree inputs SHALL match IEEE single precision. | `tests/test_export.py::test_f32_matches_ieee_single_precision` |
