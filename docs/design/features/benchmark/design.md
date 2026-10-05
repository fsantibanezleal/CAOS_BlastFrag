# Design: the protocol benchmark and the verdict

Retroactive. `run_benchmark` (`blastfrag/benchmark.py`) takes arm factories, builds the protocols with
`all_protocols` (`blastfrag/splits.py`), scores each arm per draw for the repeated protocols (`_score_repeated`)
and pools the out-of-fold predictions for leave one site out (`_score_grouped`), each pooled score on the two
`SUPPORTS` with a site-resampled interval (`bootstrap_interval`, `blastfrag/metrics.py`). `_verdict` applies
`KILL_CRITERION` on both supports, lists in-sample and site-constant arms from each arm's provenance, measures the
protocol gap from the median draw and places `PUBLISHED_RANDOM_SPLIT_R2` in the draws. The page is
`docs/methods/05_protocol-sensitivity.md`.
