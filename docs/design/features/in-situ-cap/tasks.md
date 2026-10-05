# Tasks: the in-situ block cap

1. `cap_at_in_situ_block` on a `SizeDistribution` (IC-004).
2. `InSituCap` wrapping an arm, with the recorded detail and the pass-through abstention (IC-001 to IC-003, IC-005,
   IC-008).
3. `kuznetsov-capped` in `default_arms` with its provenance (IC-007).
4. The corpus check: the cap binds on Rc1, Rc2 and Rc3 and nowhere else (IC-006).
5. The method page: a section in `docs/methods/01_classical.md`, and the CHANGELOG entry.

## Convergence, 2026-10-05

| Requirement | Gate | Result |
|---|---|---|
| IC-001 to IC-008 | `tests/test_in_situ_cap.py`, eight tests | passed |

Measured with the closed-form arms (100 draws, 2000 site resamples): the cap binds on Rc1, Rc2 and Rc3 only,
and lifts the classical arm from 0.311 to 0.352 held out by site and from 0.303 to 0.399 at the median random
draw; its interval still spans zero. Recorded in `docs/methods/01_classical.md` section 5.

