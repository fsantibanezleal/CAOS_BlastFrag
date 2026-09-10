"""The ladder: every rung behind one interface, so they can be compared fairly.

An arm here is anything that turns a blast into a :class:`~blastfrag.types.Prediction`. That includes
closed-form equations that need no data, published equations whose coefficients are constants,
equations refitted from the corpus, and learned models that must be trained. Putting them behind one
interface is what lets a benchmark score them on the same rows under the same protocol.

Two arms are not models at all and run on every case anyway: the null model, which predicts the
training mean, and the oracle, which returns the measurement. They bracket what any real arm can
achieve, and a rung that does not sit between them is reporting something other than skill.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .classical import (
    CrushZoneParameters,
    cunningham_uniformity_index,
    kuznetsov_x50_m,
    rosin_rammler,
    swebrec,
)
from .geometry import GeometryUnavailable, has_absolute_geometry, reconstruct_pattern
from .rockfactor import SITE_ROCK_FACTOR
from .types import Blast, Group, Prediction, SizeDistribution

__all__ = [
    "Arm",
    "NullModel",
    "Oracle",
    "Kuznetsov",
    "KuzRam",
    "GroupDiscriminant",
    "PublishedRegression",
    "RefittedRegression",
    "LADDER",
    "TIERS",
]


class Arm(ABC):
    """One rung of the ladder.

    ``fit`` is a no-op for the closed-form arms, which is why it has a default: a benchmark can call
    it uniformly without special-casing, and an arm that genuinely needs data overrides it.
    """

    name: str
    tier: str
    lane: str
    source: str

    def fit(self, blasts: Sequence[Blast]) -> "Arm":
        """Train on a set of blasts. Closed-form arms ignore this and return themselves."""
        return self

    @abstractmethod
    def predict_one(self, blast: Blast) -> Prediction:
        """Predict for one blast, or abstain with a reason."""

    def predict(self, blasts: Sequence[Blast]) -> list[Prediction]:
        return [self.predict_one(b) for b in blasts]

    def _abstain(self, blast: Blast, reason: str) -> Prediction:
        return Prediction(
            method=self.name, blast_id=blast.blast_id, x50_m=None, abstain_reason=reason
        )


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------

class NullModel(Arm):
    """Predict the training mean for every blast.

    The comparison every other arm is measured against. On the published hold-out it scores a root
    mean square error of 0.147 m, and the classical arm's 0.128 m improves on that by 13 percent.
    """

    name = "null"
    tier = "control"
    lane = "live"
    source = "the mean measured size of the training blasts"

    def __init__(self, mean_m: float | None = None) -> None:
        self.mean_m = mean_m

    def fit(self, blasts: Sequence[Blast]) -> "NullModel":
        sizes = [b.x50_m for b in blasts if b.x50_m is not None]
        if not sizes:
            raise ValueError("the null model needs measured sizes to average")
        self.mean_m = sum(sizes) / len(sizes)
        return self

    def predict_one(self, blast: Blast) -> Prediction:
        if self.mean_m is None:
            return self._abstain(blast, "the null model has not been fitted")
        return Prediction(method=self.name, blast_id=blast.blast_id, x50_m=self.mean_m)


class Oracle(Arm):
    """Return the measurement. The ceiling, and a check that the harness is wired correctly.

    An oracle that does not score perfectly means the scoring code is broken, not that the model is.
    """

    name = "oracle"
    tier = "control"
    lane = "offline"
    source = "the measured value itself"

    def predict_one(self, blast: Blast) -> Prediction:
        if blast.x50_m is None:
            return self._abstain(blast, "this blast has no measured size")
        return Prediction(method=self.name, blast_id=blast.blast_id, x50_m=blast.x50_m)


# ---------------------------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------------------------

class Kuznetsov(Arm):
    """The classical mean-size equation on the reconstructed pattern.

    Needs a rock volume and a charge mass, so it **abstains** on any blast whose site publishes no
    hole diameter. That refusal is the whole reason the geometry module raises instead of guessing.

    The rock factor comes from the per-site values recovered by back-solving the published
    predictions, which is the only source of factors for this corpus: both papers say the factor was
    estimated per blast and neither prints one. Those values are derived rather than published, and
    every prediction carries that in its detail.
    """

    name = "kuznetsov"
    tier = "classical"
    lane = "live"
    source = (
        "Kuznetsov 1973 with Cunningham's explosive-strength correction, as printed in Hudaverdi, "
        "Kulatilake and Kuzu 2010 doi:10.1002/nag.957 Eq. 1"
    )

    def __init__(
        self,
        rock_factors: dict[str, float] | None = None,
        *,
        timing_factor: float = 1.0,
    ) -> None:
        self.rock_factors = dict(rock_factors if rock_factors is not None else SITE_ROCK_FACTOR)
        self.timing_factor = timing_factor

    def predict_one(self, blast: Blast) -> Prediction:
        if not has_absolute_geometry(blast):
            return self._abstain(
                blast,
                f"{blast.site} publishes no hole diameter, so rock volume and charge mass cannot "
                "be resolved; this equation needs both",
            )
        factor = self.rock_factors.get(blast.site)
        if factor is None:
            return self._abstain(
                blast, f"no rock factor is available for {blast.site}"
            )
        try:
            pattern = reconstruct_pattern(blast)
        except GeometryUnavailable as exc:
            return self._abstain(blast, str(exc))
        return Prediction(
            method=self.name,
            blast_id=blast.blast_id,
            x50_m=kuznetsov_x50_m(pattern, factor, timing_factor=self.timing_factor),
            detail={
                "rock_factor": factor,
                "rock_factor_origin": "back-solved per site from the published predictions, derived",
                "rock_volume_m3": pattern.rock_volume_m3,
                "charge_mass_kg": pattern.charge_mass_kg,
                "timing_factor": self.timing_factor,
            },
        )


class KuzRam(Kuznetsov):
    """The full classical model: the mean size plus a Rosin-Rammler distribution around it.

    The mean size is identical to :class:`Kuznetsov`; what this adds is the shape, and therefore the
    percentile sizes an engineer actually specifies against.
    """

    name = "kuz-ram"
    source = (
        "Kuznetsov 1973 with the Cunningham 1987 uniformity index and the Rosin-Rammler "
        "distribution, as printed in Amoako, Jha and Zhong 2022 doi:10.3390/mining2020013 "
        "Eqs. 3, 5 and 7"
    )

    def distribution(self, blast: Blast) -> SizeDistribution | None:
        """The full size distribution for one blast, or ``None`` where the arm abstains."""
        prediction = self.predict_one(blast)
        if prediction.abstained:
            return None
        assert prediction.x50_m is not None
        pattern = reconstruct_pattern(blast)
        return rosin_rammler(prediction.x50_m, cunningham_uniformity_index(pattern))


class Swebrec(KuzRam):
    """The classical mean size with the three-parameter Swebrec distribution around it.

    The upper size limit defaults to the larger of burden and spacing, which is the closure the
    literature states. The undulation parameter is the caller's; there is no published fit for it on
    this corpus.
    """

    name = "swebrec"
    source = "Ouchterlony 2005, as printed in Amoako, Jha and Zhong 2022 Eqs. 10 and 11"

    def __init__(self, *args, undulation: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.undulation = undulation

    def distribution(self, blast: Blast) -> SizeDistribution | None:
        prediction = self.predict_one(blast)
        if prediction.abstained:
            return None
        assert prediction.x50_m is not None
        pattern = reconstruct_pattern(blast)
        x_max = max(pattern.burden_m, pattern.spacing_m)
        return swebrec(prediction.x50_m, x_max, self.undulation)


class CrushZone(KuzRam):
    """The classical coarse branch composed with a crushed-zone fines branch.

    The mechanism is sourced; the branch constants are not published anywhere held for this work and
    are the caller's. Both facts travel with every distribution this returns.
    """

    name = "crush-zone"
    tier = "semi-mechanistic"
    source = (
        "structure from Amoako, Jha and Zhong 2022 section 3 describing Kanchibotla et al. 1999 "
        "and Djordjevic 1999; branch constants caller-supplied"
    )

    def __init__(self, *args, parameters: CrushZoneParameters | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parameters = parameters or CrushZoneParameters()

    def distribution(self, blast: Blast) -> SizeDistribution | None:
        from .classical import crush_zone

        prediction = self.predict_one(blast)
        if prediction.abstained:
            return None
        assert prediction.x50_m is not None
        pattern = reconstruct_pattern(blast)
        return crush_zone(
            prediction.x50_m, cunningham_uniformity_index(pattern), self.parameters
        )


# ---------------------------------------------------------------------------------------------
# Statistical
# ---------------------------------------------------------------------------------------------

# Hudaverdi, Kulatilake and Kuzu 2010 Eq. 8. Unstandardised discriminant coefficients, canonical
# correlation 0.973. Verified in this package: it reproduces the published group membership of all
# 97 training blasts and all 14 hold-out blasts with zero errors, and the two groups are perfectly
# separated (the low-modulus maximum is 10.318 against a high-modulus minimum of 13.067).
DISCRIMINANT_COEFFICIENTS: dict[str, float] = {
    "S_over_B": 4.467,
    "H_over_B": -0.551,
    "B_over_D": -0.123,
    "T_over_B": 1.642,
    "Pf_kg_m3": -3.005,
    "XB_m": 0.309,
    "E_GPa": 0.208,
}
DISCRIMINANT_CONSTANT = 3.577

#: Midpoint between the two group centroids, measured on the shipped corpus. Any score above this
#: is the high-modulus group. Measured rather than published: the source prints the function and the
#: group memberships, from which the boundary follows.
DISCRIMINANT_BOUNDARY = 11.821


def discriminant_score(blast: Blast) -> float:
    """The published discriminant score. Above the boundary is the high-modulus group."""
    return (
        sum(coefficient * getattr(blast, name) for name, coefficient in DISCRIMINANT_COEFFICIENTS.items())
        + DISCRIMINANT_CONSTANT
    )


def assign_group(blast: Blast) -> Group:
    """Which rock-stiffness group a blast belongs to, by the published discriminant function.

    This is a real gate rather than a label lookup: the group-specific regressions and the
    group-specific neural networks are selected by it, so a blast the corpus never saw still gets
    routed the way the source authors would have routed it.
    """
    return 1 if discriminant_score(blast) > DISCRIMINANT_BOUNDARY else 2


class GroupDiscriminant(Arm):
    """Not a size predictor: the router the group-specific arms depend on.

    It is in the ladder because it is a promised method with a published equation and a measurable
    accuracy, and because its accuracy bounds everything downstream of it. It predicts a group, so it
    abstains on size and reports the group in its detail.
    """

    name = "group-discriminant"
    tier = "statistical"
    lane = "live"
    source = "Hudaverdi, Kulatilake and Kuzu 2010 doi:10.1002/nag.957 Eq. 8"

    def predict_one(self, blast: Blast) -> Prediction:
        return Prediction(
            method=self.name,
            blast_id=blast.blast_id,
            x50_m=None,
            abstain_reason="this arm assigns a rock-stiffness group rather than a fragment size",
            group=assign_group(blast),
            detail={"discriminant_score": discriminant_score(blast)},
        )

    def accuracy(self, blasts: Sequence[Blast]) -> dict[str, object]:
        """How often the published function reproduces the published group membership."""
        labelled = [b for b in blasts if b.group is not None]
        errors = [b.blast_id for b in labelled if assign_group(b) != b.group]
        return {
            "n_labelled": len(labelled),
            "n_misassigned": len(errors),
            "misassigned": errors,
            "accuracy": 1.0 - len(errors) / len(labelled) if labelled else float("nan"),
        }


@dataclass(frozen=True, slots=True)
class RegressionCoefficients:
    """A power-law regression: an intercept and one exponent per feature."""

    intercept: float
    exponents: dict[str, float]
    source: str = ""

    def predict(self, blast: Blast) -> float:
        value = self.intercept
        for name, exponent in self.exponents.items():
            value *= getattr(blast, name) ** exponent
        return value


# Hudaverdi 2010 Eqs. 9 and 10, identical to Kulatilake 2012 Eqs. 15 and 16. Transcribed verbatim.
PUBLISHED_COEFFICIENTS: dict[Group, RegressionCoefficients] = {
    1: RegressionCoefficients(
        intercept=208.0,
        exponents={
            "S_over_B": 2.788,
            "H_over_B": 0.112,
            "B_over_D": 0.027,
            "T_over_B": -0.321,
            "Pf_kg_m3": -0.360,
            "XB_m": 0.233,
            "E_GPa": -1.802,
        },
        source="Hudaverdi et al. 2010 Eq. 9, high modulus, R2 0.708 on 35 blasts",
    ),
    2: RegressionCoefficients(
        intercept=0.60,
        exponents={
            "S_over_B": 0.547,
            "H_over_B": 0.535,
            "B_over_D": 0.427,
            "T_over_B": -0.101,
            "Pf_kg_m3": -0.115,
            "XB_m": 0.434,
            "E_GPa": -1.202,
        },
        source="Hudaverdi et al. 2010 Eq. 10, low modulus, R2 0.739 on 62 blasts",
    ),
}
# The two intercepts differ by a factor of 347 while both return metres, which looks like a unit
# error and is not: the modulus exponents differ by 0.6 and the modulus spans 9.57 to 60 GPa, so the
# modulus term absorbs the gap. Verified numerically before these were accepted.


class PublishedRegression(Arm):
    """The two published group-specific power-law regressions, coefficients verbatim.

    Which equation fires is decided by the published discriminant function rather than by the
    corpus's group label, so the arm works on a blast the corpus never saw.
    """

    name = "published-regression"
    tier = "statistical"
    lane = "live"
    source = "Hudaverdi, Kulatilake and Kuzu 2010 doi:10.1002/nag.957 Eqs. 9 and 10"

    def __init__(self, coefficients: dict[Group, RegressionCoefficients] | None = None) -> None:
        self.coefficients = coefficients or PUBLISHED_COEFFICIENTS

    def predict_one(self, blast: Blast) -> Prediction:
        group = assign_group(blast)
        coefficients = self.coefficients[group]
        return Prediction(
            method=self.name,
            blast_id=blast.blast_id,
            x50_m=coefficients.predict(blast),
            group=group,
            detail={"equation": coefficients.source, "discriminant_score": discriminant_score(blast)},
        )


class RefittedRegression(Arm):
    """The same functional form, refitted from the corpus rather than taken as constants.

    Fitting a power law is a linear least squares in logarithms, so this needs no optimiser and no
    dependency beyond numpy. Comparing the refitted coefficients against the published ones is a real
    result: it says whether the published equation is the one the published data supports.
    """

    name = "refitted-regression"
    tier = "statistical"
    lane = "live"
    source = "the published functional form, coefficients refitted from the training rows supplied"

    def __init__(self) -> None:
        self.coefficients: dict[Group, RegressionCoefficients] = {}

    def fit(self, blasts: Sequence[Blast]) -> "RefittedRegression":
        import numpy as np

        from .types import FEATURES

        self.coefficients = {}
        for group in (1, 2):
            rows = [
                b for b in blasts if b.x50_m is not None and b.x50_m > 0 and assign_group(b) == group
            ]
            if len(rows) <= len(FEATURES):
                # Refusing to fit is correct here: seven exponents from seven or fewer rows is an
                # interpolation dressed as a regression.
                continue
            design = np.array(
                [[1.0] + [math.log(getattr(b, name)) for name in FEATURES] for b in rows]
            )
            target = np.array([math.log(b.x50_m) for b in rows])  # type: ignore[arg-type]
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.coefficients[group] = RegressionCoefficients(  # type: ignore[index]
                intercept=float(math.exp(solution[0])),
                exponents={name: float(c) for name, c in zip(FEATURES, solution[1:])},
                source=f"refitted from {len(rows)} training blasts in group {group}",
            )
        return self

    def predict_one(self, blast: Blast) -> Prediction:
        group = assign_group(blast)
        coefficients = self.coefficients.get(group)
        if coefficients is None:
            return self._abstain(
                blast,
                f"no refitted equation for group {group}: the training rows supplied did not "
                "contain enough of that group to fit seven exponents",
            )
        return Prediction(
            method=self.name,
            blast_id=blast.blast_id,
            x50_m=coefficients.predict(blast),
            group=group,
            detail={"equation": coefficients.source},
        )


TIERS: tuple[str, ...] = ("control", "classical", "semi-mechanistic", "statistical", "learned")

LADDER: dict[str, type[Arm]] = {
    "null": NullModel,
    "oracle": Oracle,
    "kuznetsov": Kuznetsov,
    "kuz-ram": KuzRam,
    "swebrec": Swebrec,
    "crush-zone": CrushZone,
    "group-discriminant": GroupDiscriminant,
    "published-regression": PublishedRegression,
    "refitted-regression": RefittedRegression,
}
"""The rungs implemented in the core. Learned rungs register themselves from ``blastfrag.learned``."""
