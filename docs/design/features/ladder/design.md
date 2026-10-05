# Design: the method ladder

Retroactive. Every method is an `Arm` (`blastfrag/models.py`): `fit` on training rows (a no-op for closed forms),
`predict_one` returning a `Prediction` or an abstention, and three provenance flags (`in_sample_corpus`,
`uses_site_constant`, `router_in_sample`) plus `shares_mean_size_with` for a curve arm. The closed forms are in
`blastfrag/classical.py` and `blastfrag/rockfactor.py`, the router and regressions in `blastfrag/models.py`, the
learned arms in `blastfrag/learned.py`. `_guarded` refuses a value outside `PLAUSIBLE_X50_M`. `default_arms`
(`blastfrag/benchmark.py`) is the ladder the benchmark runs, one arm per distinct mean size. The method pages are
`docs/methods/01_classical.md` to `08_diagnostics.md`.
