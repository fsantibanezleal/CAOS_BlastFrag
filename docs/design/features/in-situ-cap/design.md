# Design: the in-situ block cap

## Why

Hudaverdi, Kulatilake and Kuzu 2010 frame blasting as the transformation of the in-situ block size distribution
into the blasted one. A blast breaks blocks; it does not fuse them, so no fragment can be larger than the
in-situ block it came from. The classical mean-size equation does not read the block size at all, and on the
training corpus it predicts a mean size above the block on three Reocin blasts (Rc1 to Rc3: 0.72, 0.77 and
0.81 m against a 0.68 m block), while no measured mean size anywhere exceeds its block (probe of 2026-10-05).

## What

The cap reads mass a model places above the in-situ block size `XB` as unbroken blocks at that size:

- the passing curve is unchanged below `XB` and is 1 at and above it: `P_cap(x) = P(x)` for `x < XB`, else 1;
- the mean size is therefore `min(x50, XB)`: when more than half the mass would lie above `XB`, the median is
  the block size itself.

Clamping rather than renormalising is the choice: renormalising (`P(x) / P(XB)`) would move mass into the fines,
which no mechanism here supports, while unbroken blocks are what a too-weak blast leaves.

## Interfaces

- `blastfrag.classical.cap_at_in_situ_block(distribution, in_situ_block_m) -> SizeDistribution`, recording
  `in_situ_block_m`, `cap_binds` and the uncapped `x50` in `detail`.
- `blastfrag.models.InSituCap(base)`: an `Arm` wrapping another arm. Name `<base>-capped`, the base's tier, lane
  and provenance, `source` naming the declared choice, and `declared_not_published = True` in its provenance. It
  fits its base, predicts with it, and caps the value; an abstention passes through unchanged.
- `default_arms` gains `"kuznetsov-capped"`, the cap on the site-factor classical arm (the ladder's rung 7 on its
  rung 1), placed after `"kuznetsov-transfer"`.

## Data flow

Blast, base arm, prediction, cap, prediction with `detail = {"capped_from_m", "in_situ_block_m", "cap_binds"}`.

## Limits, stated with the result

No held dataset carries a measured size distribution, so the capped curve is not validated; the mean-size effect
is scored like any other arm, on the same rows and protocols.
