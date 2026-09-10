# The statistical rung

Two published power laws, one per rock-stiffness group, and the discriminant function that decides
which one fires. Between them they are the best-performing arm in this package on an unseen site.

---

## 1. The group router

Hudaverdi, Kulatilake and Kuzu 2010 split the 97 blasts by hierarchical average-linkage clustering on
Pearson correlation distance over z-scored features. Two groups came out: 35 high-modulus blasts with
a mean Young modulus of 51.14 GPa, and 62 low-modulus blasts at 17.22 GPa.

Their own analysis of what drives the split is worth carrying, because it explains why the router
works. Wilks' lambda identifies the modulus as overwhelmingly dominant (lambda 0.161, F 540.8) and
the spacing-to-burden ratio as having **no** effect on group membership at all (lambda 0.988,
F 0.971, significance 0.280).

The discriminant function, their Eq. 8, canonical correlation 0.973:

```
L = 4.467*(S/B) - 0.551*(H/B) - 0.123*(B/D) + 1.642*(T/B)
      - 3.005*Pf + 0.309*XB + 0.208*E + 3.577
```

### 1.1 It reproduces exactly

Measured on the shipped corpus: **zero misassignments on all 97 training blasts and all 12 hold-out
blasts.** The two groups are perfectly separated, with the low-modulus maximum at 10.318 and the
high-modulus minimum at 13.067, and the boundary sits at the midpoint of the centroids, 11.821.

That matters beyond bookkeeping. It means the router is a real gate rather than a label lookup: a
blast the corpus has never seen gets routed the way the source authors would have routed it, so the
group-specific equations work on new designs.

### 1.2 It predicts a group, not a size

So it **abstains** on fragment size, with that as the reason. It is in the ladder because it is a
promised method with a published equation and a measurable accuracy, and because its accuracy bounds
everything downstream of it.

## 2. The two regressions

Hudaverdi et al. 2010 Eqs. 9 and 10, identical to Kulatilake et al. 2012 Eqs. 15 and 16:

**Group 1, high modulus.** `R = 0.841`, `R2 = 0.708`, adjusted 0.632, standard error 0.0916,
35 blasts, `F = 9.356`:

```
x50 = 208 * (S/B)^2.788 * (H/B)^0.112 * (B/D)^0.027
        * (T/B)^-0.321 * Pf^-0.360 * XB^0.233 * E^-1.802
```

**Group 2, low modulus.** `R = 0.859`, `R2 = 0.739`, adjusted 0.705, standard error 0.1119,
62 blasts, `F = 22.808`:

```
x50 = 0.60 * (S/B)^0.547 * (H/B)^0.535 * (B/D)^0.427
        * (T/B)^-0.101 * Pf^-0.115 * XB^0.434 * E^-1.202
```

### 2.1 Reading the signs

Fragment size rises with the spacing ratio, the burden-to-diameter ratio and the in-situ block size;
falls with the stemming ratio and the powder factor; and falls with modulus in both groups, so
stiffer rock fragments finer. The spacing ratio carries the largest exponent in the high-modulus
group and the second largest in the low-modulus one.

### 2.2 An apparent unit error that is not one

The two leading coefficients differ by a factor of **347** and both equations return metres. That
looks like a unit slip and it is not: the modulus exponents differ by 0.6 over a range of 9.57 to
60 GPa, so the modulus term absorbs the gap. Checked numerically before the coefficients were
accepted, and pinned by a test, because "the constants look wrong" is the kind of observation that
gets someone to helpfully rescale one of them.

## 3. The finding: the equation beats its own papers' tables

Both papers print these equations **and** a table of predictions from them. Recomputing the equations
and scoring on the same rows gives a materially better result than either table:

| Scored on | The paper's own column | Recomputed from its own equation |
|---|---|---|
| the 2010 hold-out, 13 rows | 0.747 | **0.854** |
| the 2012 hold-out, 12 rows | 0.708 | **0.827** |

The two papers also disagree with **each other** on five rows despite printing the same equations.
On four of those five, the recomputation lands on the 2010 figure and never on the 2012 one, which
settles which table is the correct application.

### 3.1 Two rows that match nothing

`Ad24` misses both by about 0.02 m.

`Db10` misses both by a factor of two: the published equation on its published inputs gives 0.324 m
where both papers print 0.16. And the independently reproduced neural network predicts 0.6 to 0.76 m
there across every seed, where the paper prints 0.33. **Two independent models disagree with the
source in the same direction on the same row.**

The natural hypothesis was that one input cell is wrong. It was tested: each of the seven inputs was
scanned for the value that reconciles the regression, and then checked against the network. **None
reconciles both.** The closest, a stiffness ratio of 1.16, is below the corpus minimum of 1.33 and
still leaves the network at 0.26 against a published 0.33.

So the row is carried with its published inputs and its inconsistency recorded. The refuted
hypothesis is pinned by a test so nobody re-derives it.

## 4. Refitting does not help

The same functional form refitted on the 97 rows is a linear least squares in logarithms, so it needs
no optimiser. On the published hold-out it generalises **slightly worse** than the coefficients the
authors published: 0.807 against 0.827. A small argument that their fit was not overtuned.

Under leave-one-site-out the difference stops being small. The published equation holds at **0.802**;
the refit collapses to **-4.075**, and on one fold extrapolates to a mean fragment size of 10.48 m,
which the plausibility guard now refuses rather than scores.

The mechanism is the whole finding of this package in one comparison. The published coefficients are
**fixed**, so they cannot overfit to whichever nine sites happen to be in the training fold. The
refit is fitted, so it does.

## 5. Refusing to fit

The refit raises rather than fitting when a group has seven or fewer rows. Seven exponents from seven
rows is an interpolation dressed as a regression, and the resulting equation would look exactly like
one with evidence behind it.
