# The learned rung

Four learned arms, three of them reproductions of fully specified published models. The chapter's
headline is a partial refutation, and that is the point: a model whose architecture, training
algorithm, normalisation and hyperparameters are all printed can be reproduced exactly rather than
approximately, and then the reproduction can disagree with the paper in a way that means something.

---

## 1. The published network

Kulatilake, Hudaverdi and Wu 2012, section 5. Fully specified:

- seven inputs, `N` hidden units with a logistic activation, one linear output, justified on
  Cybenko's universal approximation result;
- trained **separately per rock-stiffness group**, on that group's rows only;
- inputs and target **min-max normalised**, their Eq. 11;
- **Levenberg-Marquardt** training, chosen after comparing four algorithms because it "showed the
  highest stability" and "reached the global minimum with the lowest number of training cycles";
- hidden width bounded by two published heuristics to 6 through 15, swept;
- **eight simulations at each width**, scored by root mean square error and by correlation;
- published optima: 9 hidden units for the high-modulus group, 7 for the low-modulus group.

### 1.1 Levenberg-Marquardt in numpy

No mainstream Python library ships Levenberg-Marquardt for neural networks, and substituting gradient
descent would not be the published method. So it is written here: a Gauss-Newton step damped toward
gradient descent,

```
(J'J + lambda I) delta = -J' r
```

with the damping raised on a step that increases the loss and lowered on one that decreases it.

The Jacobian is hand-derived, so it is **checked against a central finite difference** in the test
suite, agreeing to better than 1e-7. A wrong Jacobian still trains, just slowly and to a worse
optimum, which is exactly the kind of defect that ships looking fine.

The seed is drawn **before** any training rather than inside it. A network whose initial weights
depend on how many models were fitted earlier in the same process is not reproducible, and every
individual run still looks correct.

### 1.2 The architecture is generous relative to the data

Seven hidden units on the 62 low-modulus blasts is **64 free parameters fitted to 62 rows**. Nine on
the 35 high-modulus blasts is **82 fitted to 35**.

The reproduction fits its training rows at a variance explained above 0.95, which is what
over-parameterisation looks like rather than what success looks like. Both counts are reported on
every prediction the arm makes.

## 2. The reproduction result

**Ten of twelve hold-out rows reproduce the published network closely**, often within 0.01 to 0.03 m
of the published mean. The reproduction works.

The two that do not are `Ru7` and `Db10`, and those are exactly the two rows the **source itself**
flags as its most unstable. Its own eight simulations at the chosen width run from 0.23 to 0.96 m on
`Ru7`, a coefficient of variation of 0.56, and from 0.16 to 0.74 on `Db10`, a coefficient of 0.76.
The paper even explains `Ru7`: it shares every input with a training blast except one.

### 2.1 Across seeds, the published score is not reachable

Thirty seeds, each training eight networks per group:

| | variance explained on the published hold-out |
|---|---|
| reproduction, minimum | 0.167 |
| reproduction, median | 0.340 |
| reproduction, maximum | 0.636 |
| **published** | **0.910** |

No seed reaches even 0.7. The published figure is above every one of them.

The mechanism separates into two parts. On `Ru7` the reproduction ranges from 0.412 to 0.935 across
seeds, spanning the published 0.63, so that row is seed luck. On `Db10` it ranges from 0.600 to 0.757
across **all thirty seeds** while the paper prints 0.33, so that row is a systematic disagreement,
and it is the same row where the published regression column disagrees with the published regression
equation by a factor of two.

### 2.2 What is claimed, and what is not

The claim is narrow: **this reproduction, built to the published specification, does not reach the
published hold-out score, and the shortfall concentrates in the rows the source itself reports as
unstable.**

It is not a claim that the published result is wrong. Unrecorded details, an initialisation scheme,
a stopping rule, a different simulation draw, could account for it. What the sweep does establish is
that the published score is not robust to the seed, on a method the paper itself shows swinging by a
factor of four between adjacent hidden widths.

### 2.3 The instability is reported per prediction

Every prediction carries its eight simulation values, their spread, their median and the coefficient
of variation. The source reports that coefficient in a table; carrying it per prediction turns "the
model is uncertain here" from a footnote into something a caller can act on.

### 2.4 Predictions are confined to the target range

The target is min-max normalised onto the unit interval, so that interval is the whole of what a
network fitted on it can assert. Outputs outside it are clamped before denormalising.

This is not cosmetic. Without it the wildest simulations on `Ru7` average to zero and drag the arm's
variance explained from positive to -1.18 on that row alone.

### 2.5 The hidden width, reproduced and held out by site

The source chose the hidden width on its own hold-out: 9 units for the high-modulus group, 7 for the
low. `network_width_sweep` asks two questions of that choice (0.4.0; eight simulations per width, seed
0, BLAS on one thread, about two minutes).

**Does the source's procedure land on the published widths?** Train on the corpus, score each width
from 6 to 15 on the 2012 hold-out, keep the width with the lowest RMSE per group:

| Group | Published | Reproduced | RMSE at the reproduced width | RMSE at the published width |
|---|---|---|---|---|
| 1, high modulus | 9 | 8 | 0.037 m | 0.102 m |
| 2, low modulus | 7 | 11 | 0.106 m | 0.133 m |

It does not, and the RMSE moves non-monotonically between adjacent widths (0.037 m at 8 and 0.102 m
at 9 for the high-modulus group), which is the instability the source's own tables show.

**Does any width transfer to a site the network has not seen?** Each width, the same in both groups,
scored on the pooled leave-one-site-out predictions:

| Hidden width | variance explained, every blast (97) | blasts with geometry (91) |
|---|---|---|
| 9 and 7, the published pair | -0.626 | -0.604 |
| 6 | -0.605 | -0.630 |
| 7 | -0.653 | -0.635 |
| 8 | -0.626 | -0.562 |
| 9 | -0.586 | -0.604 |
| 10 | -0.902 | -0.955 |
| 11 | -0.568 | -0.681 |
| 12 | -1.027 | -1.169 |
| 13 | -0.717 | -0.747 |
| 14 | -1.257 | -1.296 |
| 15 | -1.486 | -1.603 |

No width explains any variance held out by site, and the wider networks fail hardest. The published
pair reproduces the benchmark's own score for this arm exactly (-0.626), which checks the sweep. The
width choice neither explains nor rescues the network's failure to transfer.

## 3. The 2025 arms

### 3.1 Support vector regression, in two published parameterisations

Two sources tune it on this corpus and reach **opposite conclusions about the kernel**. One searched
2700 combinations across four kernels and chose a radial basis function with a regularisation of 5.25
and an epsilon of 0.04, noting radial models "have better generalization abilities". The other used a
degree-five polynomial with a regularisation of 1 and reported this arm as the worst of its three
single learners.

Both ship. Measured here on the published hold-out, the radial variant is better by a wide margin.
Averaging them would erase the disagreement, which is the interesting part.

### 3.2 Random forest, gradient boosting, and the stacking ensemble

Reproduced with the 2025 study's final hyperparameters: 76 trees at seed 27 for the forest, a
learning rate of 0.5 at seed 42 for the boosting (the number of trees is not printed, so the library
default of 100 is used), and a plain **linear** meta-learner over both, chosen by the source
deliberately "to avoid overfitting caused by excessive complexity".

The source prints a second parameter set for its single learners (forest at seed 1 with 50 trees,
boosting at a learning rate of 1.9) and does not say unambiguously which set produced its standalone
figures of 0.797 and 0.758. The standalone arms here use the final set, the same learners the stacked
model contains, so their scores are not exact reproductions of those two figures and are not
presented as such.

**Cross-validation was cancelled while the stacked model was built.** The source, in the paragraph
describing that construction: "the first attempt was to use cross-validation to increase the
generalization ability of the model, but the cross-validated model had a poor prediction effect on the
test set, and the stacking fusion model itself could effectively improve the residuals, thus canceling
the cross-validation". In a stacked model the cross-validation that can be cancelled is the
out-of-fold scheme that produces the meta-learner's training inputs (Wolpert 1992,
`doi:10.1016/S0893-6080(05)80023-1`, introduces stacking with exactly that out-of-fold step), so since
0.3.0 the meta-learner is fitted on the base learners' **in-sample** predictions. That reading is this
package's; the source does not spell out the mechanism. Before 0.3.0 the arm used scikit-learn's
out-of-fold stacking with two folds, which is a different method.

Fitted on the full corpus this way, the meta-learner's weights are **1.02 on the boosting learner and
-0.02 on the forest**: the boosting learner fits its training rows almost exactly, so a meta-learner
trained on those rows trusts it almost entirely, and the stacked model behaves like its boosting
learner under every protocol.

The boosting arm's overfitting is reproduced rather than tuned away: its training fit exceeds 0.98
while its hold-out fit is materially lower, as the source describes.

## 4. And then the protocol

On the published hold-out, whose blasts come from the same sites as the training rows, the learned
arms beat the classical one comfortably. Held out by whole site, every learned arm loses between 0.74
and 4.94 of variance explained against its median random draw, and the best of them lands within a
few hundredths of zero. That is the subject of [the protocol-sensitivity benchmark](05_protocol-sensitivity.md).

The learned arms can be run outside Python: [the portable export](../data/03_portable-export.md)
writes each fitted arm as JSON that a browser walks exactly.
