# Running the benchmark

```python
import blastfrag as bf

result = bf.run_benchmark(
    bf.load_training_corpus(),
    bf.default_arms(),
    seed=0,          # first seed of the repeated random draws
    n_repeats=100,   # draws of each random protocol
    n_boot=2000,     # site resamples for each grouped interval (0 skips them)
)
```

About two and a half minutes on a desktop CPU, almost all of it the published network's
Levenberg-Marquardt training. With `n_repeats=5, n_boot=300` it takes about twenty seconds and every
leave-one-site-out figure is unchanged, because those do not depend on the random draws.

## What comes back

- `result.verdict["outcome"]`: the canonical sentence group, composed from the structured fields.
- `result.verdict["supports"]`: the kill criterion on every blast and on the blasts with resolvable
  geometry; `depends_on_support` says whether they agree.
- `result.verdict["in_sample_arms"]`, `["site_constant_arms"]`, `["intervals_95"]`,
  `["arms_with_interval_above_zero"]`, `["protocol_gap_random_minus_grouped"]`,
  `["dedup_minus_random_median"]`, `["published_random_split_figures"]`.
- `result.by_protocol()["random-8020"].by_arm()[name].detail["repeats"]`: the spread over draws.
- `result.by_protocol()["leave-one-site-out"].by_arm()[name].detail`: `supports` (score and interval
  on each), `per_site` (error on each held-out site) and `predictions` (every blast's out-of-fold
  prediction).
- `result.provenance[name]`: what each arm was fitted on.
- `result.table()`: one flat row per arm and protocol, with the headline figure (the median draw for a
  repeated protocol).

## Adding an arm

An arm is any subclass of `blastfrag.Arm` that implements `predict_one`, and `fit` if it learns. Pass a
factory, not an instance, so every fold gets a fresh model:

```python
arms = bf.default_arms() | {"mine": lambda: MyArm(alpha=0.1)}
```

Set `fitted_on`, and `in_sample_corpus` if the arm's coefficients came from the 97-blast corpus,
so the verdict reports it correctly.
