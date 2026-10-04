# The portable export

`blastfrag.export` writes a fitted arm as plain JSON and reads it back with a dependency-free walker.
It exists so that an application can run the learned tier in a browser, where numpy, scikit-learn and
xgboost are not available and ONNX Runtime Web has no tree-ensemble kernel.

---

## 1. The contract

Every document carries:

| Field | Meaning |
|---|---|
| `schema` | `blastfrag.portable/v1` |
| `arm` | the arm's name |
| `kind` | `network`, `svr`, `forest`, `xgboost`, `stacking`, `power-law` or `kuznetsov-transfer` |
| `engine_version` | the package version that fitted it |
| `features` | the input order, always the seven corpus features |
| `plausible_x50_m` | `[0.001, 3.0]`; a prediction outside it is an abstention |
| `provenance` | what the arm was fitted on, the same flags the benchmark reports |

and, by kind:

- **network**: the discriminant router; per stiffness group the min-max bounds of the inputs, the
  target range, the hidden width and the weight vector of each of its eight networks, laid out as
  $W_1$ row-major (inputs by hidden units), $b_1$, $W_2$, $b_2$. Each output is clamped to $[0, 1]$,
  rescaled to the target range and the eight are averaged.
- **svr**: the standardisation mean and deviation, the kernel and its resolved $\gamma$, degree and
  $c_0$, the support vectors (in standardised units), the dual coefficients and the intercept:
  $\hat y = \sum_i \alpha_i K(\mathbf{s}_i, \mathbf{z}) + b$, with
  $K = \exp(-\gamma\lVert\mathbf{s}-\mathbf{z}\rVert^2)$ or $(\gamma\,\mathbf{s}\cdot\mathbf{z} + c_0)^{d}$.
- **forest**: the standardisation, then every tree as four arrays `left`, `right`, `feature`, `t`. A
  node is a leaf when `left` is -1, and `t` is then the leaf value; otherwise `t` is the threshold and
  the walk goes left when the input is at most `t`. The prediction is the mean over trees.
- **xgboost**: the same four arrays, with `t` the 32-bit split value or leaf value and the walk going
  left when the input is strictly below `t`; the prediction is `base_score` plus the leaf values.
- **stacking**: a forest, a boosting model and the meta-learner's two weights and intercept.
- **power-law**: the router and, per group, the intercept and seven exponents.
- **kuznetsov-transfer**: the fitted line $\ln A = a + b\ln E$; the reader returns the rock factor,
  because the mean-size equation needs absolute geometry the seven ratios do not carry.

## 2. Matching the libraries to the last bit

Three details decide whether a port agrees exactly:

1. **Inputs are rounded to 32 bits before every tree comparison.** scikit-learn casts inputs to 32-bit
   floats before comparing them with 64-bit thresholds; XGBoost compares 32-bit inputs with 32-bit
   split values. In a browser that is `Math.fround`; here it is `blastfrag.export.f32`.
2. **XGBoost's split values are rounded back to 32 bits on export.** Its JSON prints each value as a
   short decimal that parses to a nearby 64-bit float; without rounding back, 3 of 100 trees took the
   other branch on one blast when an input equalled a split value.
3. **XGBoost's sum is accumulated in 32 bits**, starting from the base score.

With these, the reference walker reproduces the random forest, gradient boosting and the stacked model
**exactly** on all 116 shipped blasts, and the network and both support-vector arms to a relative
difference below $10^{-12}$, abstentions included. The test suite asserts this for every exportable
arm.

## 3. Size

Fitted on the 97-blast corpus: network 25 kB, radial SVR 10 kB, polynomial SVR 7 kB, gradient
boosting 33 kB, random forest 193 kB, stacking 225 kB, refitted power law 1 kB.

## 4. Using it

```python
import json
import blastfrag as bf

train = bf.load_training_corpus()
arm = bf.default_arms()["random-forest"]().fit(train)
document = bf.export_arm(arm)
json.dump(document, open("forest.json", "w"))

value, reason = bf.predict_portable(document, train[0])   # metres, or None and why
```
