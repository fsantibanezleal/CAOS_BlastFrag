# Requirements: common support

Designed 2026-10-05, before its code (ADR-0075), to close audit item A-15: abstentions are dropped per arm, so
arms are compared on different rows under every protocol (the polynomial kernel abstains on 11 of 97 held out by
site, the classical arms on the six Miami blasts). EARS.

| ID | Requirement | Gate |
|---|---|---|
| CS-001 | FOR every draw of a repeated protocol, THE benchmark SHALL also score every arm on the rows that every size-predicting arm answered, and report the spread of those scores. | `tests/test_common_support.py::test_every_draw_is_also_scored_on_the_rows_every_arm_answered` |
| CS-002 | THE common rows SHALL be exactly the rows that every size-predicting arm answered. | `tests/test_common_support.py::test_the_common_rows_are_exactly_the_intersection` |
| CS-003 | THE leave-one-site-out result SHALL carry each arm's score on the common rows with its site-resampled interval. | `tests/test_common_support.py::test_the_held_out_result_carries_a_common_support_score` |
| CS-004 | THE kill criterion SHALL remain evaluated on the two declared row sets only. | `tests/test_common_support.py::test_the_criterion_is_not_evaluated_on_the_common_rows` |
| CS-005 | IF an arm answers no row in a protocol (the router predicts a group, not a size), THEN THE common rows SHALL not depend on it. | `tests/test_common_support.py::test_an_arm_that_never_answers_does_not_empty_the_common_rows` |
