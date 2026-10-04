# From a mean size to a size distribution

The mean size is one point on a curve. A crusher is specified against the 80 percent passing size, an
oversize limit and a fines fraction, which are points elsewhere on that curve. Three forms turn the
classical mean size into a full distribution. **They share the classical mean size and differ only in
shape, and no dataset held for this package records a measured size distribution, so their shapes are
model output that nothing here validates.** In code each of them carries
`shares_mean_size_with = "kuznetsov"`, and the benchmark counts the mean-size predictor once.

---

## 1. Rosin-Rammler with the Cunningham uniformity index (Kuz-Ram)

As printed in Amoako, Jha and Zhong 2022 (`doi:10.3390/mining2020013`) Eqs. 4 to 7, the fraction
retained above a mesh $x$ is

$$
R(x) = \exp\!\left[-0.693\left(\frac{x}{x_{50}}\right)^{n}\right],
\qquad P(x) = 1 - R(x),
$$

where $0.693 = \ln 2$ makes $x_{50}$ the 50 percent passing size. The characteristic size, which 63.2
percent passes, is $x_c = x_{50} / 0.693^{1/n}$. The uniformity index is Cunningham's:

$$
n = \left(2.2 - 14\frac{B}{d}\right)\sqrt{\frac{1 + S/B}{2}}\left(1 - \frac{W}{B}\right)
\left(\left|\frac{\mathrm{BCL}-\mathrm{CCL}}{L}\right| + 0.1\right)^{0.1}\frac{L}{H},
$$

with burden $B$ and spacing $S$ in metres, hole diameter $d$ in **millimetres**, drilling deviation
$W$, charge length $L$, bottom and column charge lengths $\mathrm{BCL}$ and $\mathrm{CCL}$, and bench
height $H$; multiply by 1.1 for a staggered pattern. Amoako et al. give the usual range of $n$ as 0.7
to 2. [The classical rung](01_classical.md) records the unit trap in $B/d$.

This package treats the charge as one continuous column, which is what ANFO is in these patterns, so
$|\mathrm{BCL} - \mathrm{CCL}| = 0$ and the charge-distribution term reduces to $0.1^{0.1} \approx 0.794$
instead of an invented bottom and column split. The drilling deviation $W$ is a field of the
pattern that no source publishes for this corpus, and it defaults to zero.

## 2. Swebrec, three parameters (Ouchterlony 2005)

Ouchterlony 2005 (`doi:10.1179/037178405X44539`), as printed in Amoako et al. Eqs. 10 and 11:

$$
P(x) = \frac{1}{1 + \left[\dfrac{\ln(x_{\max}/x)}{\ln(x_{\max}/x_{50})}\right]^{b}},
\qquad 0 < x < x_{\max}.
$$

The explicit upper limit $x_{\max}$ bounds the coarse tail, which the two-parameter form leaves
unbounded, and the undulation $b$ shapes the fines branch. Babaeian et al. 2019
(`doi:10.1016/j.jrmge.2018.11.006`) note that $x_{\max}$ is commonly taken as the burden or the
spacing; this package uses the larger of the two. No fit of $b$ exists for this corpus; it defaults to
2.0 and is the caller's.

**A counter-example travels with it.** Babaeian et al. measured 24 blasts at the Jajarm bauxite mine
by image analysis and found the Rosin-Rammler range closer to the measurement than the Swebrec range.
The three-parameter form is more adaptable in general and not better at every site.

## 3. The two-branch crush-zone composition

Amoako et al. section 3 describe two mechanisms acting together: tensile fracturing produces the
coarse fraction and compressive-shear fracturing in the crushed zone around the hole produces the
fines (Kanchibotla, Valery and Morrell 1999; Djordjevic 1999). The coarse branch is the Rosin-Rammler
curve above; a fines branch with its own uniformity takes over below a crossover size.

**The structure is sourced; the constants are not.** The 1999 papers are conference proceedings not
held for this work, and no held source prints the crossover size, the fines uniformity or the fines
fraction. They are `CrushZoneParameters`, supplied by the caller, and the defaults are stated as
starting points rather than published values.

## 4. What would validate these shapes

A set of blasts with both the design and a measured passing curve (sieved or from calibrated image
analysis at several sizes), so that the error at $P_{20}$, $P_{50}$ and $P_{80}$ and over a shared
log-spaced sieve grid can be scored the way the mean size is. Until such a set is held, every curve
this package returns is a shape implied by a mean size and a uniformity index, and is reported as
that.
