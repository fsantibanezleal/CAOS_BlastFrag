# Tasks: common support

1. Keep each draw's predictions in `_score_repeated` and the pooled ones in `_score_grouped` (CS-001, CS-003).
2. `_add_common_support` after every protocol: the size-predicting arms, the intersection, the rescoring, the
   interval, the clean-up (CS-001 to CS-003, CS-005).
3. A test that the verdict reads only the two declared supports (CS-004).
4. The method page: a section in `docs/methods/05_protocol-sensitivity.md`, and the CHANGELOG entry.
