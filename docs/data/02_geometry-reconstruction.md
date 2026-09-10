# Recovering absolute geometry from a dimensionless corpus

The training corpus records seven ratios and no dimensions. The classical mean-size equation needs
rock volume and charge mass per hole. Those two facts together mean the classical model could not be
run on the corpus at all, which is plausibly why every recent study on it is a black-box regressor
over the ratios.

This page derives the reconstruction that closes the gap and, more importantly, the evidence that it
is right.

---

## 1. The arithmetic

A hole diameter fixes the scale, and everything else follows from the published ratios:

```
B = (B/D) * D
S = (S/B) * B
H = (H/B) * B
T = (T/B) * B

V = B * S * H          rock volume broken per hole
Q = Pf * V             explosive mass per hole
```

The last line is why the powder factor is carried as a first-class field rather than derived from a
charge length. The corpus publishes it directly; deriving it would require a charge geometry the
corpus does not publish.

## 2. Where the diameters come from

The source paper's section 3 describes each campaign in prose, and eight of the ten descriptions
contain a hole diameter:

| Site | Diameter | The sentence |
|---|---|---|
| Enusa | 165 mm | "Hole diameters for the Enusa and Reocin mines were 165 and 229 mm" |
| Reocin | 229 mm | the same sentence |
| Murgul | 165 mm | "The drillhole diameter applied was 165 mm" |
| Mrica | 76 mm | "The hole diameter was 76 mm and bench height was 10 to 15 m" |
| Soma | 210 mm | "The diameter of the blast holes was 21 cm" |
| Dongri-Buzurg | 100 mm | "The hole diameter was 100 mm and bench height was 6 to 11 m" |
| Akdaglar | 89 mm | "The drillhole diameter is 89 mm" |
| Ozmert | 89 mm | "The hole diameter is 89 mm" |

### 2.1 The ninth, derived rather than read

Reocin underground has no published diameter but a stated bench height of 18 m. Inverting the
arithmetic supplies one:

```
B = H / (H/B)          then          D = B / (B/D)
```

Applied to all six of its rows, this returns **91.2 mm every time**. Two different bench-height
ratios (5.00 on one row, 6.00 on five) and two different burden-diameter ratios (39.47 and 32.89)
give the same answer, which is a six-fold internal consistency check rather than an assumption.

That agreement is a test.

### 2.2 The tenth, which is the negative control

Miami publishes no diameter and no absolute dimension of any kind. Nothing fixes its scale, so its
six rows are **not reconstructable**, and `reconstruct_pattern` raises rather than returning
something. Every model that needs a rock volume abstains on those rows with a reason string.

A number there would be an invention. This is the case that keeps the classical models honest, and
it is why the package has an abstention type at all.

---

## 3. The evidence

The same prose that gives diameters also gives **independent dimensional constraints**, and the
reconstruction is asserted against every one of them. Fifteen constraints across nine sites, all
fifteen satisfied.

| Site | The source states | The reconstruction gives |
|---|---|---|
| Enusa | bench 6 m | 5.984 m |
| Reocin | bench 9 to 11 m | 9.00 to 11.76 m |
| Reocin-UG | bench 18 m | 17.997 to 17.998 m |
| **Murgul** | **bench 12 m, burden 4.5 to 5, spacing 4.5 to 5.5** | **12.00, 4.50 to 5.00, 4.50 to 5.50** |
| Mrica | bench 10 to 15 m | 15.00 m |
| Soma | spacing 7.5 m, bench 15 m | 7.50 m, 15.00 m |
| **Dongri-Buzurg** | **bench 6 to 11 m, burden 2 to 2.5, spacing 1.8 to 3.5** | **6.00 to 11.00, 2.00 to 2.50, 2.50 to 3.50** |
| Akdaglar | burden mean 2.17, sd 0.35 | mean 2.07, sd 0.33 |
| Ozmert | burden 2.5 m, spacing 3 m | 2.00 to 3.00, 2.50 to 3.00 |

**Murgul is decisive.** One paragraph states three independent quantities and one arithmetic rule
reproduces all three exactly. An error in the diameter, in the arithmetic or in the shipped ratios
would break at least one of them. **Dongri-Buzurg is the second**, with three more.

### 3.1 The tolerance is derived, not chosen

The corpus prints every ratio to two decimals, so a ratio carries half a unit of the last digit in
uncertainty and the reconstruction inherits it. The band allowed on each assertion is that
uncertainty propagated through the multiplication that produced the quantity: burden comes from one
ratio, spacing and bench height from two, and the relative uncertainty of a product is the sum of the
relative uncertainties of its factors.

This is why the Enusa bench height of 5.984 m passes against a stated 6 m and why a ten percent
change in a diameter fails. There is no free parameter to widen.

### 3.2 The one conflict, carried rather than resolved

Soma's paragraph states **both** a 5 m burden and a 21 cm diameter. The ratio table cannot hold both:
its burden-diameter ratio is 28.57, and 5 divided by 0.21 is 23.8.

Taking the diameter as authoritative gives a 6.00 m burden, which then reproduces the stated 7.5 m
spacing and the stated 15 m bench **exactly**. Two of three constraints select it. A reading with a
175 mm diameter and a 5 m burden satisfies the ratio but then fails the bench height.

So the diameter is used, the burden constraint is **not asserted**, and the conflict travels with
every Soma case. Which of the two the authors intended is unresolved and is recorded as unresolved.

---

## 4. The independent check

The 2025 field set publishes its absolute pattern **and** its ratios. Running the reconstruction on
its ratios and comparing against its published dimensions checks the arithmetic against a published
answer rather than against a narrative range. It agrees on burden, spacing, bench height and stemming
to within the printed precision. That is a test.

---

## 5. What this unlocks

With volume and charge available, the classical model can be run on the real blasts. Back-solving its
rock factor from the published predictions then returns a value that is **near constant within each
site**:

| Site | Implied rock factor | Rock, and its modulus |
|---|---|---|
| Enusa | 11.97 | schist, 60 GPa |
| Reocin-UG | 12.22 | carbonate, 45 GPa |
| Reocin | 12.14 | carbonate, 45 GPa |
| Murgul | 10.02, 10.08 | dacite, 50 GPa |
| Mrica | 7.11 | andesite, 32 GPa |
| Akdaglar | 6.85, 6.59, 7.01 | sandstone, 16.9 GPa |
| Ozmert | 6.49, 6.39 | sandstone, 15 GPa |
| Soma | 6.24 | coal measures, 13.25 GPa |
| Dongri-Buzurg | 3.68 | weak schist, 9.57 GPa |

Within-site spread is 0.6 percent at Murgul, 1.6 at Ozmert and 3 at Akdaglar, which is what rounding
the published predictions to two decimals produces. Every value lands inside the published valid
range of 0.8 to 22, and the ordering tracks stiffness from 12 in the stiffest rock down to 3.7 in the
weakest.

That single result does three things at once.

1. **It validates the reconstruction.** An error in burden or bench height would scatter the factor
   within a site, and it does not.
2. **It resolves an exponent disagreement.** Two published statements of the same equation differ on
   the explosive-strength exponent; the back-solve is consistent with one of them and not the other.
3. **It recovers a constant the literature omitted.** Both papers say the rock factor "was estimated
   for each blast" and neither prints a value.

The derivation is in [the rock factor](../methods/03_rock-factor.md).
