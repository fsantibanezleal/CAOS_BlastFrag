# The protocol-sensitivity benchmark

The result this package exists to produce. Same arms, same rows, three ways of splitting.

---

## 1. Why the protocol is the experiment

The 2025 state of the art on this corpus reports a variance explained of 0.943 from a **random**
80/20 split of 97 rows. Seventeen of those rows duplicate another row's feature vector, so a random
draw places duplicates on both sides of the split by construction. The same paper records that
cross-validation was tried and then removed because "the cross-validated model had a poor prediction
effect on the test set", which is the symptom this predicts.

Nothing in the literature reports what these models do under a split that does not leak. So that is
what gets measured, under three protocols:

| Protocol | Rule | What it answers |
|---|---|---|
| `random-8020` | seeded random 80/20 | reproduces the published protocol |
| `dedup-random` | duplicate feature vectors collapsed first | isolates the duplicate effect alone |
| `leave-one-site-out` | a whole campaign held out, ten folds | can this model reach a mine it has not seen? |

The third is the practitioner's question. Rows within one campaign share a rock mass, a drilling
rig, an explosive supply and a measurement operator, and one quarry supplies twenty-two of the
ninety-seven rows.

Folds are **pooled**, not averaged. A leave-one-site-out sweep has folds from six to twenty-two rows,
and averaging their scores would weight a six-row site the same as a twenty-two-row one. Pooling
scores every blast exactly once, on the fold where it was held out.

## 2. The kill criterion, declared before the run

> The learned tier counts as generalising across sites only if the best learned arm's variance
> explained under leave-one-site-out is **both positive and at least 0.10 above the null model's**.

Both halves are required, and the second half was added **after** the first run, because that run
caught the criterion failing to measure its own subject. As first written the rule asked only for a
margin over the null. The sweep produced a best learned arm at -0.034 against a null at -0.216, and
the rule declared success. An arm with negative variance explained is worse than predicting a
constant; it had simply failed less badly than a constant fitted to a different mix of sites. A gate
that can pass on two failures is not a gate.

## 3. The result

Variance explained about the identity line. Abstentions in brackets.

| Arm | Tier | random 80/20 | deduplicated | leave-one-site-out |
|---|---|---|---|---|
| null | control | -0.052 | -0.007 | -0.216 |
| oracle | control | 1.000 | 1.000 | 1.000 |
| classical mean size | classical | -0.027 | 0.116 (2) | **0.311** (6) |
| group router | statistical | abstains | abstains | abstains |
| **published regression** | statistical | 0.632 | 0.861 | **0.802** |
| refitted regression | statistical | 0.513 | 0.823 | -4.075 (4) |
| published neural net | learned | 0.270 | 0.557 | -0.626 |
| support vector, radial | learned | 0.541 | 0.792 | -0.387 (1) |
| support vector, polynomial | learned | 0.221 | 0.424 | -4.546 (11) |
| random forest | learned | 0.649 | 0.859 | -0.231 |
| gradient boosting | learned | 0.694 | 0.728 | -0.034 |
| stacking ensemble | learned | 0.667 | 0.885 | -0.951 |

**Zero of six learned arms have positive variance explained on an unseen site.** The kill criterion
fires.

## 4. What the table says

### 4.1 Only the arms that are not fitted survive

Two arms transfer, and both have **fixed** coefficients. The published regression's exponents are
constants from a paper; the classical equation's only free quantity is a per-site rock factor. Every
arm that fits itself to this corpus fails to leave it.

That is the finding, and it is more interesting than a negative result. It is a statement about what
the corpus can support: ninety-seven blasts from ten campaigns are enough to fit a model that
interpolates between campaigns it has seen, and not enough to fit one that reaches a new one.

### 4.2 The classical model gets better when held out by site

From -0.027 on a random split to **0.311** held out by site. The model did not change. The
comparison did: on a random split it competes against arms that have memorised near-duplicates of
the test rows, and on a site-held-out split it does not.

This inverts the usual reading of the classical model as the weak baseline. On a leaking protocol it
looks worse than everything; held out by site it is second only to the published regression.

### 4.3 Deduplication is not the whole story

Collapsing duplicate feature vectors and splitting randomly gives **higher** scores than the raw
random split for five of six learned arms, not lower. So the duplicates are not what props the
learned arms up. The shared **site** is.

### 4.4 A fitted power law leaves the domain entirely

With one site held out, the refitted regression extrapolates to a mean fragment size of **10.48 m**.
Scoring that number rather than refusing it produced a per-fold variance explained near -11000, which
then swamped every other fold pooled with it.

A prediction outside 0.001 to 3 m is now refused with a reason instead of scored. That is both truer
and more informative: the arm did not do badly on that site, it declined to answer. Eleven of the
polynomial support-vector arm's predictions and four of the refitted regression's are refusals under
this protocol, and the counts are reported rather than hidden in an average.

## 5. What this does not say

It does not say learned models are useless for blast fragmentation. It says **this corpus** cannot
support one that transfers between campaigns, and that a random split over it produces a number
which does not survive contact with a new mine.

It also does not say the published results are wrong. They are correct under the protocol they
report. The claim here is narrower and checkable: that protocol answers a different question from
the one a practitioner asks.

## 6. Reproducing it

```python
import blastfrag as bf

result = bf.run_benchmark(bf.load_training_corpus(), bf.default_arms(), seed=0)
print(result.verdict["outcome"])
for row in result.table():
    print(row)
```

The result carries the digest of the dataset it ran on, so a number and the rows behind it cannot
drift apart.
