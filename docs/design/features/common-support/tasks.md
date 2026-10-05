# Tasks: common support

1. Keep each draw's predictions in `_score_repeated` and the pooled ones in `_score_grouped` (CS-001, CS-003).
2. `_add_common_support` after every protocol: the size-predicting arms, the intersection, the rescoring, the
   interval, the clean-up (CS-001 to CS-003, CS-005).
3. A test that the verdict reads only the two declared supports (CS-004).
4. The method page: a section in `docs/methods/05_protocol-sensitivity.md`, and the CHANGELOG entry.

## Convergence, 2026-10-05

| Requirement | Gate | Result |
|---|---|---|
| CS-001 to CS-005 | `tests/test_common_support.py`, five tests | passed |

On the full benchmark the common rows held out by site are 79 blasts from nine sites; the tree models rise to
0.21 there because the rows the other arms refuse are the hard ones, and no arm outside the in-sample regression
has an interval above zero. Recorded in `docs/methods/05_protocol-sensitivity.md` section 4.1.
