# The corpus: what is in it, where it came from, and what is wrong with it

Three real datasets ship with `blastfrag`. This page says exactly what each one is, how it was
verified, and which defects in it are known and carried deliberately.

---

## 1. The training corpus, 97 bench blasts

**Source.** Hudaverdi, Kulatilake and Kuzu (2010), *International Journal for Numerical and
Analytical Methods in Geomechanics* 35:1318-1333, `doi:10.1002/nag.957`, Tables I and II.

Ninety-seven blasts from ten campaigns on four continents, assembled by the authors from their own
fieldwork near Istanbul and from eight earlier published studies. The paper's own claim for it, and
it holds up: "No previous study has used such a diverse blasting data base."

### 1.1 The seven features

The corpus is **dimensionless by design**. The authors chose ratios so that blasts at very different
scales could sit in one table:

| Symbol | Meaning | Range in the corpus |
|---|---|---|
| `S/B` | spacing over burden | 1.00 to 1.75 |
| `H/B` | bench height over burden, the rock-beam stiffness | 1.33 to 6.82 |
| `B/D` | burden over hole diameter | 17.98 to 39.47 |
| `T/B` | stemming over burden | 0.50 to 4.67 |
| `Pf` | powder factor, kg/m3 | 0.22 to 1.26 |
| `XB` | in situ block size, m | 0.02 to 2.35 |
| `E` | Young modulus, GPa | 9.57 to 60.00 |

The target is the measured mean fragment size `x50`, from 0.02 to 0.96 m with a mean of 0.304 m.

The paper explains the choices. Spacing over burden is set by "energy coverage of the bench", and a
square pattern has a ratio of 1. Stemming over burden is usually around 1: too low and the gases vent
early, giving flyrock and poor breakage; too high and the specific charge falls and boulders appear.
Burden over diameter is quoted as around 30 for average conditions after Ash, and 25 for a
low-density explosive such as ANFO. Bench height over burden "indicates the stiffness of the rock
beam under blast-induced stress".

Every blast in the database used ANFO, so the paper carries no explosive-type variable. That single
sentence is what fixes the relative weight strength at 100 for the whole corpus and makes the
classical model reproducible on it.

### 1.2 The ten sites

| Code | Mine | Rock | Blasts |
|---|---|---|---|
| Enusa | open-pit uranium, Spain | schist, moderately to heavily folded | 12 |
| Reocin | open-pit zinc, Spain | carbonate-hosted | 10 |
| Reocin-UG | Reocin underground | carbonate-hosted | 6 |
| Murgul | open-pit copper, Turkey | dacite and altered dacite | 7 |
| Mrica | quarry, Indonesia | andesite | 11 |
| Soma | open-pit coal, western Turkey | coal measures | 7 |
| Dongri-Buzurg | open-pit manganese, central India | micaceous and muscovite schist | 9 |
| Miami | mine, Arizona | highly fractured pinal schist | 6 |
| Akdaglar | quarry, Istanbul | sandstone, UCS 81 MPa, `E` 16.9 GPa | 22 |
| Ozmert | quarry, Istanbul | sandstone | 7 |

Site membership matters more than it looks. Twenty-two of the ninety-seven rows come from one
quarry, so a metric computed as if the rows were independent will understate its own uncertainty.
This is why the leave-one-site-out protocol exists and why bootstrap intervals resample the site
rather than the row.

### 1.3 How fragment size was measured

Sieving a muckpile is impractical, so the corpus uses **image analysis**, which the paper describes
as the main tool of the previous decade. The Istanbul quarries used WipFrag on multiple photographs
per muckpile, combined; the joint structure and in situ block size at the same benches came from
WipJoint on images of the bench face taken before each blast.

That measurement channel is itself a source of scatter, and it is one reason a model that explains
70 percent of the variance on this data is doing well rather than badly.

---

## 2. Five transcription defects, found and corrected

The corpus as first assembled disagreed with the published tables in five cells. All five were found
by diffing every cell against the paper's Tables I and II, parsed independently from **two separate
PDF copies** of the article. The two copies agree with each other and disagreed with the file in
exactly these places:

| Blast | Column | Published | As found | What happened |
|---|---|---|---|---|
| `Db5` | `Pf` | 0.39 | 0.33 | wrong digit |
| `Sm4` | `x50` | 0.22 | 0.24 | wrong digit, on the target |
| `Ad15` | `x50` | 0.15 | 0.22 | wrong digit, on the target |
| `Ad17` | `Pf` | 1.24 | 1.07 | wrong digit |
| `Ad19` | `Pf` | 1.26 | 1.47 | the row's own `XB` copied into the `Pf` column |

**How it was noticed.** The paper prints its own descriptive statistics in a summary table, and that
table gives a powder-factor maximum of 1.26 while the file contained 1.47. A paper that prints a
summary table has handed you a checksum, and nothing was reading it.

### 2.1 The gate that now reads it

`check_source_integrity` runs on every load of the training corpus and does two independent things.

**It reproduces the published descriptive statistics.** Minimum, maximum, mean and standard deviation
for all seven features, compared at the precision each was printed to. Minima and maxima are gated
exactly, because they are facts read straight off the data tables. Means and standard deviations get
one full unit of the last printed digit, which is the measured internal inconsistency of the source
itself and no more, documented below.

**It compares a content digest.** A mean over ninety-seven rows is a weak detector for a single-cell
change: correcting one powder factor by 0.06 moves it by 0.0006, well inside any rounding step. The
digest catches any change at all.

The two answer different questions. The statistics say the file still is the published table. The
digest says it has not moved since it was corrected. Both are tested, including a test that
reintroduces the original defect and requires the gate to fail.

### 2.2 The source's own summary is not perfectly consistent

Measured over the corrected corpus, against the paper's summary table:

- all fourteen minima and maxima reproduce exactly;
- five of seven standard deviations reproduce exactly; the powder factor gives 0.235455 against a
  printed 0.236 and the block size 0.532274 against 0.533, each one unit of the last digit low;
- five of seven means reproduce exactly; the stiffness ratio gives 3.3452 against a printed 3.34 and
  the block size 1.1067 against 1.10, each one unit high because the source truncated rather than
  rounded.

Four cells out of twenty-eight, all at the last digit. This is recorded rather than smoothed over,
because it is what sets how tight the gate can be: any tighter and it fires on the source's own
rounding, any looser and it stops catching real defects.

---

## 3. The published hold-out, and its three defects

Fourteen rows, the union of two published validation sets. The membership flags say which set each
row belongs to, so either published protocol can be reproduced exactly:

- `protocol="2010"`, thirteen blasts, Hudaverdi et al. 2010 Table VIII;
- `protocol="2012"`, twelve blasts, Kulatilake, Hudaverdi and Wu 2012 Tables 4 and 5;
- `protocol="union"`, all fourteen;
- `protocol="clean"`, thirteen, the union minus the one row with training membership.

### 3.1 Leakage: one validation blast is also a training blast

`Rc1` appears in the 2010 Table I as one of the thirty-five high-modulus **training** blasts, and
again in the 2012 Table 4 as one of the five high-modulus **validation** blasts, with the same
measured size of 0.46 m. The 2012 paper states that its high-modulus network was trained on those
thirty-five and validated on those five.

The 2010 set does not contain `Rc1`; it lists `Mi7` and `Ad25` instead. The substitution happened
between the two papers and carried training membership across with it.

**The effect was measured rather than assumed, and the natural assumption was wrong.** Removing
`Rc1` and rescoring the remaining eleven costs the regression arm 0.013 and the neural-net arm 0.005
in variance explained, and *gains* the classical arm 0.216, because `Rc1` is its second-worst row.
The arm the leak flatters is the classical one.

### 3.2 The hold-out shares feature vectors with the training set

Seventeen of the ninety-seven training rows duplicate another row's feature vector, in seven groups:
`En1` with `En2`, `En3` with `En5`, `En8` with `En9`, `En11` with `En12`, `Mr1` with `Mr3`, `Sm1`
through `Sm3`, and `Sm4` through `Sm7`.

The 2012 paper states the cross-set case itself without drawing the conclusion: "En13 blast has the
same values of `S/B`, `H/B`, `B/D`, `T/B`, `Pf`, `XB` and `E` as for En4 blast. Therefore, the
prediction result of `x50` for En13 blast is almost the same as the value for En4 blast." `En4` is a
training blast and `En13` is a validation blast.

This does not move the fixed hold-out numbers. It moves anything computed from a **random split** of
the ninety-seven, which is the protocol the 2025 state of the art uses.

### 3.3 The two papers disagree on their own predictions

Both papers print the same two regression equations. Their regression columns nonetheless differ on
five rows, and their classical columns on ten.

Recomputing the published low-modulus equation resolves four of the five: on `Mr12`, `Sm8`, `Oz9` and
`Ad23` the computed value rounds to the 2010 figure and not to the 2012 one. The 2010 table is the
correct application of the equation both papers print.

`Ad24` matches neither, by 0.02 m. And `Db10` matches neither by far more: the published equation on
its published inputs gives 0.324 m where both papers print 0.16, a factor of 2.03. Two readings would
close that and neither is confirmable, an in situ block size of 0.20 m rather than 1.00, or a Young
modulus of 17.3 GPa rather than 9.57. It is left **unresolved**, with its published inputs, and the
disagreement travels with the row.

---

## 4. The field hold-out, and why it is kept apart

Five production blasts in a granite mine in northeastern China, from Sui, Zhou, Zhao, Yang and Zou
(2025), `doi:10.3390/app15031254`, CC BY 4.0. Fragment size measured with Split-Desktop against a
one-metre on-site scale.

Its Young modulus is 5.6 GPa. The corpus minimum is 9.57. So **every prediction on this set is an
extrapolation**, on the feature both 2025 studies rank as the most important of the seven. That is
the honest reading of the source's own "prediction errors within 0.03 m", and it is what makes these
five rows worth carrying: they are the only out-of-envelope real test this corpus has.

They serve a second purpose. The source publishes the **absolute** pattern as well as the ratios, so
these are the one place the geometry reconstruction can be checked against a published answer rather
than against a narrative range. That check is a test.

---

## 5. The ingestion contract

Two bands, and they are not the same band.

**Contract bounds** are rejection. A Young modulus of 60000 is a unit error, not an unusual blast,
and it raises rather than loading. Same for a ratio, a powder factor or a measured size outside
physically possible ranges.

**The fitted envelope** is the corpus's own range. A row outside it loads only with
`allow_extrapolation=True`, and every prediction made on it is stamped `extrapolated`. The default
is refusal, so predicting outside the data is something a caller does on purpose.
