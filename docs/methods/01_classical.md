# The classical rung

Closed form, microseconds to evaluate, and still the industry default. That combination is why it is
worth testing hard: a model this cheap tends to be believed.

---

## 1. Mean fragment size

Kuznetsov 1973 with Cunningham's explosive-strength correction, as printed in Hudaverdi, Kulatilake
and Kuzu 2010 (`doi:10.1002/nag.957`) Eq. 1 and Amoako, Jha and Zhong 2022
(`doi:10.3390/mining2020013`) Eqs. 2 and 3:

```
x50 = A * (V/Q)^0.8 * Q^(1/6) * (RWS/115)^(-19/30)          [centimetres]
```

| Symbol | Meaning | Units |
|---|---|---|
| `x50` | mean fragment size | cm in the equation, metres everywhere else here |
| `A` | rock factor | dimensionless, 0.8 to 22 |
| `V` | rock volume broken per hole | m3 |
| `Q` | explosive mass in that hole | kg |
| `RWS` | weight strength relative to ANFO | ANFO is 100, TNT is 115 |

`V/Q` is the reciprocal of the powder factor, which is why the two published spellings of this
equation, one in `V/Q` and one in the powder factor to the power `-0.8`, agree on that term.

The original rock factor was a three-value lookup: 7 for medium rock, 10 for hard but highly
fissured, 13 for very hard and weakly fissured. The literature calls that too coarse in its own
words, and the way out is [the rock factor](03_rock-factor.md).

### 1.1 A published disagreement, settled on this corpus

The two sources print the explosive-strength exponent differently: `-19/30` on `(E/115)` in one and
`19/20` on `(115/RWS)` in the other. At a relative weight strength of 140 they differ by about
8 percent.

Back-solving the rock factor from the published predictions settles it. With the `19/30` form the
recovered factor is near constant within each site, which is the form the source authors used. It is
the default here; the other ships as a named variant.

## 2. Uniformity index

Cunningham 1987, Amoako 2022 Eq. 7:

```
n = (2.2 - 14*B/d) * sqrt((1 + S/B)/2) * (1 - W/B)
      * (abs((BCL - CCL)/L) + 0.1)^0.1 * (L/H)
```

Multiply by 1.1 for a staggered pattern. High values mean uniform sizing; low values a wide spread
carrying both oversize and fines. The source gives the usual range as 0.7 to 2.

### 2.1 The unit trap

**The burden is in metres and the diameter in millimetres.** The published `B/d` is therefore about
0.027 for a 4.5 m burden on a 165 mm hole. It is *not* the dimensionless burden-to-diameter ratio the
corpus tabulates, which is 27.27 for the same hole and a thousand times larger.

Read as the tabulated ratio, the leading term becomes `2.2 - 382` and the index goes to about -380.
A test asserts the index lands in its published band on all 91 reconstructable blasts, which is what
catches this.

### 2.2 The charge-distribution term

It needs a bottom-charge and column-charge split the corpus does not publish. With a single
continuous column, which is what ANFO in these patterns is, the two lengths are equal and the term
reduces to `0.1^0.1`, about 0.794. That reduction is applied rather than a split being invented.

### 2.3 Where the index leaves its band, and why

Sixty-two of ninety-one blasts sit inside 0.7 to 2. The rest are genuinely unusual patterns rather
than a broken equation: **every one below the band has a charge column short relative to its bench**,
because the stemming eats the hole. The index is linear in that ratio. One Enusa blast carries
stemming of 1.17 burdens in a bench only 1.33 burdens tall, leaving almost no charge, and its index
is 0.18.

## 3. The 2005 modification, and the timing factor

Cunningham revised both equations twenty years on, "mainly as a result of the introduction of
electronic delay detonators". The revision multiplies the mean size by a timing factor and a
rock-factor correction, and rewrites the uniformity index with a timing-scatter factor and its own
correction.

**None of those four multipliers is printed in any primary source held for this work.** They default
to one and are caller-supplied, with documented ranges. A fabricated formula for them would be worse
than their absence.

Two things about the timing factor must reach any screen that shows an initiation sequence.

**It is a scalar.** It multiplies the mean size and has no spatial structure, so changing a tie-in
from row-by-row to a V-cut, or reversing the initiation direction, changes the animation and changes
nothing in the prediction. Only the aggregate delay moves the number.

**Its shape is a hypothesis.** The literature reports that fragmentation improves with delay up to a
plateau, which refutes naive stress-wave collision arguments, but no copy of that work is held here
and no number from it is used.

## 4. Distributions

### 4.1 Rosin-Rammler

```
R(x) = exp( -0.693 * (x / x50)^n )
```

`R` is the fraction retained above a mesh, so passing is one minus that. The 0.693 is `ln 2`, which
makes `x50` the fifty-percent size rather than the characteristic size. The characteristic size,
through which 63.2 percent passes, is `x50 / 0.693^(1/n)`.

The mean-size equation plus the uniformity index plus this distribution is the classical model.

### 4.2 Swebrec

Ouchterlony 2005, the distribution half of the KCO model:

```
P(x) = 1 / (1 + f(x))
f(x) = [ ln(x_max / x) / ln(x_max / x50) ]^b
```

Three parameters rather than two, and the third is an explicit upper size limit, conventionally the
larger of burden and spacing. That limit fixes the coarse tail, which the two-parameter form gets
wrong, and the extra curvature fixes the fines branch.

**A counter-example ships with it rather than after it.** Babaeian et al. 2019, on 24 blasts at one
bauxite mine measured by image analysis, found the Rosin-Rammler range closer to the measurement than
the Swebrec range, and concluded that site follows Rosin-Rammler. Swebrec is more adaptable in
general and not better everywhere.

### 4.3 The crush-zone composition

The classical model's best-documented failure is under-predicting fines. The two-component and
crush-zone models address it by splitting the distribution: tensile fracturing produces the coarse
fragments, and compressive-shear fracturing in the crushed zone around the hole produces the fines.

**The structure is sourced and the constants are not.** The papers that introduced these models are
proceedings that are not held for this work; what is held is a description of the mechanism. So the
crossover size, the fines-branch uniformity and the fines fraction are caller-supplied, with defaults
stated as plausible starting values rather than published ones, and every distribution this returns
carries that distinction in its detail.

## 5. What the classical rung actually achieves

On the published twelve-blast hold-out: variance explained about the identity line **0.232**, squared
correlation 0.570, root mean square error 0.128 m against a null model's 0.147 m. It beats predicting
a constant by 13 percent, and it is the worst of the three arms printed in its own source table.

Held out by whole site it reaches **0.311**, second only to the published regression, because it has
nothing to overfit. See [the protocol-sensitivity benchmark](05_protocol-sensitivity.md).
