# The rock factor

One dimensionless number carries the whole of "what kind of rock is this" in the classical model.
Three published schemes disagree about how to compute it, and a fourth can be recovered from the data
itself.

---

## 1. Why there is more than one

Cunningham's route out of the original three-value lookup is Lilly's blastability index:

```
A  = 0.06 * BI
BI = 0.5 * (RMD + JPS + JPO + RDI + S)
```

Two primary sources print rating tables for that sum under the same attribution, and **they are not
the same table**.

| Term | Hudaverdi et al. 2010 | Babaeian et al. 2019 Table 2 |
|---|---|---|
| rock mass description | powdery 10, blocky 20, massive 50 | powdery 10, vertically jointed 20, massive 50 |
| joint plane spacing | under 0.1 m: 10, 0.1 to 1.0: 20, over 1.0: 50 | under 0.1 m: 10, to oversize: 20, to pattern size: 50 |
| joint plane orientation | horizontal 10, out of face 20, normal to face 30, into face 40 | the same four, called joint plane angle |
| density influence | `25 * density - 50` | the same |
| **strength** | **`0.05 * UCS`** | **`UCS / 3` below 50 GPa, `UCS / 5` above** |

The strength term is the one that matters. At a uniaxial compressive strength of 100 MPa the first
gives 5 and the second gives 33.3 or 20, which moves the index by up to 14 points and the rock factor
by up to **0.85**. On a typical bench that is roughly a quarter of the predicted fragment size.

A third route bypasses the sum entirely: Hustrulid's five-band lookup on the Protodyakonov strength
index, 3 for very soft rock through 13 for rigid and homogeneous.

**All three ship, side by side, named, with the source on each.** Presenting one as *the* rock factor
would hide the subjectivity that is the honest content of this parameter. The package's comparison
view exists to make the disagreement visible rather than to argue it away.

## 2. Every scheme refuses rather than defaults

A scheme that needs a uniaxial compressive strength and is not given one raises. It does not assume a
value.

The reason is that this number propagates straight into a predicted fragment size, linearly. A rock
factor produced from an assumed strength is a prediction with no evidence behind it, and it looks
exactly like one that has evidence.

## 3. The fourth scheme: recovered from the published predictions

Both source papers say the rock factor "was estimated for each blast" and **neither prints a value**.
That is recoverable. Inverting the mean-size equation on a published prediction gives:

```
A = x50 / [ (V/Q)^0.8 * Q^(1/6) * (RWS/115)^(-19/30) ]
```

with the size in centimetres, using the geometry recovered in
[geometry reconstruction](../data/02_geometry-reconstruction.md).

### 3.1 The result

| Site | Recovered factor | Blasts | Within-site spread | Rock, and its modulus |
|---|---|---|---|---|
| Reocin | 12.14 | 1 | - | carbonate, 45 GPa |
| Reocin underground | 11.19 | 1 | - | carbonate, 45 GPa |
| Enusa | 10.97 | 1 | - | folded schist, 60 GPa |
| Murgul | 9.23 | 2 | 1.5% | dacite, 50 GPa |
| Akdaglar | 6.72 | 2 | 3.7% | sandstone, 16.9 GPa |
| Ozmert | 6.44 | 2 | 1.6% | sandstone, 15 GPa |
| Mrica | 6.32 | 1 | - | andesite, 32 GPa |
| Soma | 6.24 | 1 | - | coal measures, 13.25 GPa |
| Dongri-Buzurg | 3.68 | 1 | - | weak schist, 9.57 GPa |

Miami is absent by design: it has no reconstructable geometry, so no factor can be recovered for it,
and the classical arm abstains there rather than borrowing a neighbour's.

### 3.2 Why this is evidence rather than arithmetic

**The values barely move within a site.** The widest spread is 3.7 percent, which is what rounding
the published predictions to two decimals produces. An error in the geometry reconstruction, in the
burden or the bench height, would scatter them.

So this one result does three jobs at once:

1. it validates the geometry reconstruction;
2. it selects between two published spellings of the explosive-strength exponent;
3. it recovers a constant the literature omitted.

Every value lands inside the published valid range of 0.8 to 22.

### 3.3 The inversion, named rather than smoothed

The recovered factors correlate with Young modulus at **0.87**, not near one, and there is a single
inversion worth naming: **Enusa is the stiffest rock in the corpus at 60 GPa and its factor of 10.97
sits below the Reocin carbonates at 45 GPa.**

Stiffness is not the only thing the rock factor carries. A moderately to heavily folded schist is not
a massive carbonate, and the structural term in every rating scheme exists precisely because of that.
A monotonic claim here would be false, so the test pins the inversion instead of the monotonicity.

### 3.4 Which published column, and why it matters

The two source papers print **different** classical predictions for the same blasts, so they imply
different rock factors. Murgul comes out at 9.23 from the 2012 column and 10.05 from the 2010 one,
and the 2010 column implies a higher factor at every shared site.

Mixing them is a defect. The recovery names which column it used, the 2012 one by default because
that is the table carrying all three published arms, and the difference is reported as a result
rather than averaged into noise.
