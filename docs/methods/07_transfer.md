# The transfer rung: a rock factor that uses nothing from the target site

`KuznetsovTransfer` is the classical mean-size equation with one change. Its rock factor is not read
from a per-site table; it is predicted from the rock's Young modulus by a line fitted over the
training sites only. Added in 0.3.0, after an audit asked whether the classical arm's cross-site
score was borrowed.

---

## 1. Why it exists

The classical arm needs a rock factor $A$, and neither source paper prints one. This package recovers
a factor per site by inverting the mean-size equation on the 2012 paper's published Kuz-Ram
predictions ([the rock factor](03_rock-factor.md)). Those predictions are for hold-out blasts **of
the same sites**. When leave-one-site-out holds out, say, Murgul, the classical arm still uses a factor
derived from the published prediction for a Murgul blast. The learned arms receive nothing comparable,
so the comparison is not symmetric, and the classical arm's 0.311 could in principle be knowledge of
the held-out site rather than transfer.

## 2. The model

For a split's training rows, take every training site $s$ that has a recovered factor $A_s$, and its
mean Young modulus $\bar E_s$. Fit one line in logarithms with one point per site:

$$
\ln A_s \;=\; a + b \,\ln \bar E_s ,
$$

by ordinary least squares, so a 22-blast quarry and a six-blast mine carry equal weight. For a blast
with modulus $E$ the factor is $\hat A = \exp(a + b \ln E)$, and the size is the classical equation
with that factor:

$$
x_{50} = \hat A \left(\frac{V}{Q}\right)^{0.8} Q^{1/6} \left(\frac{\mathrm{RWS}}{115}\right)^{-19/30}
\quad [\mathrm{cm}].
$$

Fitted on all nine sites with a recovered factor, $b > 0$: stiffer rock carries a larger factor on
this corpus, which is the ordering the recovered values show (12.1 at the Reocin open pit, 45 GPa;
3.7 at Dongri-Buzurg, 9.57 GPa). With fewer than three training sites the arm refuses to fit.

**The relation is a calibration of this package, not a published equation.** The modulus is used
because it is the one rock property the corpus records for every blast, the one Hudaverdi et al. 2010
chose to represent the rock's mechanical behaviour, and the one both 2025 studies rank first. It is not
the whole of what the factor carries: the stiffest rock in the corpus, a folded schist at 60 GPa, has a
lower recovered factor (10.97) than the 45 GPa carbonates.

## 3. Result

| Classical arm, leave-one-site-out | all 97 blasts | interval | 91 with geometry |
|---|---|---|---|
| site factor from the held-out site's published predictions | 0.311 | -0.96 to 0.70 | 0.311 |
| **transfer factor, modulus line over training sites** | **0.298** | -1.10 to 0.72 | 0.298 |
| median factor of the training sites (probe, not shipped) | 0.154 | | |

The transfer arm loses 0.013. The classical arm's cross-site score is therefore not an artefact of the
site factor. The modulus carries most of what the site factor knew; replacing it with one constant for
every site loses about half the score.

Per site the two arms trade places: the transfer factor is better at Reocin (0.166 against 0.216 m
RMSE) and Soma (0.076 against 0.132), worse at Mrica (0.213 against 0.111) and Murgul (0.110 against
0.052). Neither interval excludes zero (see [the benchmark](05_protocol-sensitivity.md) section 5.6).

## 4. Where it fails

- **Below the modulus range of the training sites** the line is extrapolated. The five field blasts of
  Sui et al. 2025 sit at 5.6 GPa against a corpus minimum of 9.57; their predictions carry the
  extrapolation stamp.
- **It needs absolute geometry**, like every classical arm, so it abstains on the Miami campaign.
- **Rock structure is absent.** In-situ block size and joint orientation enter the published rating
  schemes and not this line.

## 5. Using it

```python
import blastfrag as bf

train = bf.load_training_corpus()
arm = bf.KuznetsovTransfer().fit(train)
arm.rock_factor_for(30.0)          # the factor this fit gives a 30 GPa rock
arm.predict_one(train[0]).detail   # carries the factor, its origin and the fit sites
```
