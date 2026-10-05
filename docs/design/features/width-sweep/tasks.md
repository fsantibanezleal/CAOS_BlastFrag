# Tasks: the network width sweep

1. `network_width_sweep` with the source's protocol and the held-out table (WS-001 to WS-003, WS-006).
2. The leakage check per fold and the seed check (WS-004, WS-005).
3. The method page: a section in `docs/methods/06_learned.md`, and the CHANGELOG entry.

## Convergence, 2026-10-05

| Requirement | Gate | Result |
|---|---|---|
| WS-001 to WS-006 | `tests/test_width_sweep.py`, six tests | passed |

The full sweep (ten widths, eight simulations, seed 0) ran in 106 s on one BLAS thread. The source's procedure
lands on 8 and 11, not 9 and 7; held out by site every width scores below zero (-0.57 to -1.49), and the published
pair reproduces the benchmark's -0.626 for the arm. Recorded in `docs/methods/06_learned.md` section 2.5.
