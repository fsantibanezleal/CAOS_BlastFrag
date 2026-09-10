"""The classical rungs: mean size, uniformity, and the distributions built on them.

Everything here is closed form and runs in microseconds. That is why these models are still in daily
use, and it is also why they are worth testing hard: a model this cheap tends to be believed.

The measured verdict, on the published twelve-blast hold-out, is that the classical mean-size
equation explains 0.232 of the variance about the identity line and beats predicting the training
mean by 13 percent on root-mean-square error. It is the weakest of the three arms printed in its own
source table. That is not a reason to leave it out; it is the reason to ship it beside a null model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Pattern, SizeDistribution

__all__ = [
    "kuznetsov_x50_m",
    "cunningham_uniformity_index",
    "modified_uniformity_index",
    "rosin_rammler",
    "swebrec",
    "crush_zone",
    "sieve_grid",
    "TimingFactor",
    "CrushZoneParameters",
]


# The corpus's own explosive strength: every blast in the training database used ANFO, whose relative
# weight strength is 100 on its own scale. TNT is 115, which is why 115 appears in the equation.
_TNT_RWS = 115.0

# Two primary sources print the explosive-strength exponent differently. Hudaverdi et al. 2010 Eq. 1
# writes (E/115)^(-19/30); Amoako et al. 2022 Eq. 3 writes (115/RWS)^(19/20). These are not the same:
# at a relative weight strength of 140 they differ by about 8 percent.
#
# The back-solve in blastfrag.rockfactor settles it on THIS corpus. Recovering the rock factor from
# the published predictions with the 19/30 form gives a value that is near constant within each site;
# it is the form the source authors used. So 19/30 is the default and 19/20 is a named variant.
RWS_EXPONENT_DEFAULT = -19.0 / 30.0
RWS_EXPONENT_AMOAKO = -19.0 / 20.0


def kuznetsov_x50_m(
    pattern: Pattern,
    rock_factor: float,
    *,
    timing_factor: float = 1.0,
    rock_factor_correction: float = 1.0,
    rws_exponent: float = RWS_EXPONENT_DEFAULT,
) -> float:
    """Mean fragment size in metres, from the classical equation.

    Kuznetsov 1973 with Cunningham's explosive-strength correction, as printed by Hudaverdi,
    Kulatilake and Kuzu 2010 Eq. 1 and by Amoako, Jha and Zhong 2022 Eqs. 2 and 3::

        x50 = A * (V/Q)^0.8 * Q^(1/6) * (RWS/115)^(-19/30)          [centimetres]

    ``V`` is the rock volume broken per hole and ``Q`` the explosive mass in it. Note that ``V/Q`` is
    the reciprocal of the powder factor, which is why the two published spellings of this equation,
    one in ``V/Q`` and one in the powder factor to the power ``-0.8``, agree on that term.

    ``timing_factor`` and ``rock_factor_correction`` are the two multipliers Cunningham added in 2005
    for electronic detonators, Amoako 2022 Eq. 8. **Their tabulated values are not printed in any
    primary source held for this work**, so they default to 1 and are supplied by the caller. A
    fabricated formula for them would be worse than their absence.

    The timing factor also has no spatial structure: it is a scalar on the mean size, so changing an
    initiation tie-in changes nothing here. Anything on a screen that suggests otherwise is
    choreography, and this docstring is the reason to label it as such.
    """
    if rock_factor <= 0:
        raise ValueError(f"the rock factor must be positive, got {rock_factor}")
    volume = pattern.rock_volume_m3
    charge = pattern.charge_mass_kg
    if volume <= 0 or charge <= 0:
        raise ValueError(
            f"a blast needs a positive rock volume and charge; got V={volume}, Q={charge}"
        )
    size_cm = (
        rock_factor
        * timing_factor
        * rock_factor_correction
        * (volume / charge) ** 0.8
        * charge ** (1.0 / 6.0)
        * (pattern.explosive.rws / _TNT_RWS) ** rws_exponent
    )
    return size_cm / 100.0


def cunningham_uniformity_index(pattern: Pattern) -> float:
    """The Rosin-Rammler uniformity index, Cunningham 1987.

    As printed by Amoako, Jha and Zhong 2022 Eq. 7::

        n = (2.2 - 14*B/d) * sqrt((1 + S/B)/2) * (1 - W/B)
              * (abs((BCL - CCL)/L) + 0.1)^0.1 * (L/H)

    with the burden and spacing in metres, the hole diameter in millimetres, ``W`` the standard
    deviation of drilling precision, ``L`` the charge length and ``H`` the bench height. Multiply by
    1.1 for a staggered pattern.

    The charge-distribution term needs a bottom-charge and column-charge split that the corpus does
    not publish. With a single continuous column, which is what ANFO in these patterns is, the two
    lengths are equal and the term reduces to ``0.1^0.1``, about 0.794. That reduction is applied
    rather than a split being invented.

    Amoako 2022 states the practical range: the index usually lies between 0.7 and 2, with high
    values meaning uniform sizing and low values a wide spread carrying both oversize and fines.
    """
    burden = pattern.burden_m
    diameter_mm = pattern.hole_diameter_mm
    charge_length = pattern.charge_length_m
    if burden <= 0 or diameter_mm <= 0 or pattern.bench_height_m <= 0:
        raise ValueError("burden, hole diameter and bench height must all be positive")

    # THE UNIT TRAP IN THIS EQUATION, and it is easy to fall into. The source states the burden in
    # METRES and the diameter in MILLIMETRES, so the published `B/d` is about 0.027 for a 4.5 m
    # burden on a 165 mm hole. It is NOT the dimensionless burden-to-diameter ratio the corpus
    # tabulates, which is 27.27 for the same hole and a thousand times larger. Reading it as the
    # tabulated ratio makes this term 2.2 - 382 and the index deeply negative.
    #
    # Written as an explicit mixed-unit quotient so the trap is visible rather than folded away.
    burden_over_diameter_m_per_mm = burden / diameter_mm
    geometry = 2.2 - 14.0 * burden_over_diameter_m_per_mm

    spacing_term = math.sqrt((1.0 + pattern.spacing_m / burden) / 2.0)
    deviation_term = 1.0 - pattern.drill_deviation_m / burden
    charge_term = (abs(0.0) + 0.1) ** 0.1  # single continuous column: BCL equals CCL
    length_term = charge_length / pattern.bench_height_m

    index = geometry * spacing_term * deviation_term * charge_term * length_term
    if pattern.staggered:
        index *= 1.1
    return index


def modified_uniformity_index(pattern: Pattern, *, timing_scatter_factor: float = 1.0) -> float:
    """The 2005 uniformity index, Amoako 2022 Eq. 9::

        n = n_s * sqrt(2 - 30*B/d) * sqrt((1 + S/B)/2) * (1 - W/B) * (L/H)^0.3 * C(n)

    ``n_s`` is the uniformity factor for timing scatter and ``C(n)`` a correction, neither of which
    is printed in any primary source held for this work; both default to 1 and are caller-supplied.

    The same mixed-unit reading applies as in the 1987 form: the burden is in metres and the diameter
    in millimetres, so the quotient is around 0.027 and the first square root's argument is near 1.2
    for a typical bench. Read as the dimensionless tabulated ratio instead, the argument goes
    negative and the expression is undefined, which is the tell that the reading is wrong.

    The argument is nonetheless checked, because a very small burden on a very large hole would take
    it negative legitimately. That case raises rather than returning the magnitude: a uniformity
    index recovered from a negative square root is not a number this model produced.
    """
    burden = pattern.burden_m
    if burden <= 0 or pattern.bench_height_m <= 0:
        raise ValueError("burden and bench height must be positive")
    burden_over_diameter_m_per_mm = burden / pattern.hole_diameter_mm
    inner = 2.0 - 30.0 * burden_over_diameter_m_per_mm
    if inner <= 0:
        raise ValueError(
            f"the 2005 uniformity index is undefined for this pattern: 2 - 30*B/d = {inner:.3f} with "
            f"B = {burden:.2f} m and d = {pattern.hole_diameter_mm:.0f} mm. The published form has "
            "no branch for a burden this large relative to the hole."
        )
    geometry = math.sqrt(inner)
    spacing_term = math.sqrt((1.0 + pattern.spacing_m / burden) / 2.0)
    deviation_term = 1.0 - pattern.drill_deviation_m / burden
    length_term = (pattern.charge_length_m / pattern.bench_height_m) ** 0.3
    index = timing_scatter_factor * geometry * spacing_term * deviation_term * length_term
    if pattern.staggered:
        index *= 1.1
    return index


def sieve_grid(low_m: float = 1e-4, high_m: float = 3.0, n: int = 241) -> list[float]:
    """A shared logarithmic sieve grid, 0.1 mm to 3 m by default.

    Every distribution in the package is evaluated on the same grid so curves can be overlaid and
    differenced without resampling. Logarithmic because sieve series are, and because the fines and
    the oversize branches are the two parts that matter.
    """
    if not 0 < low_m < high_m:
        raise ValueError(f"need 0 < low < high, got {low_m} and {high_m}")
    step = (math.log(high_m) - math.log(low_m)) / (n - 1)
    return [math.exp(math.log(low_m) + i * step) for i in range(n)]


def rosin_rammler(
    x50_m: float, uniformity: float, *, sizes_m: list[float] | None = None
) -> SizeDistribution:
    """The Rosin-Rammler distribution written on the mean size.

    Amoako 2022 Eq. 5, the retained fraction above a mesh::

        R(x) = exp( -0.693 * (x / x50)^n )

    so the passing fraction is one minus that. The 0.693 is ``ln 2``, which is what makes ``x50`` the
    fifty-percent-passing size rather than the characteristic size; the characteristic size, through
    which 63.2 percent passes, is ``x50 / 0.693^(1/n)``.

    Together with the mean-size equation and the uniformity index this is the classical model.
    """
    if x50_m <= 0:
        raise ValueError(f"the mean size must be positive, got {x50_m}")
    if uniformity <= 0:
        raise ValueError(f"the uniformity index must be positive, got {uniformity}")
    sizes = sizes_m if sizes_m is not None else sieve_grid()
    passing = [1.0 - math.exp(-math.log(2.0) * (x / x50_m) ** uniformity) for x in sizes]
    return SizeDistribution(
        method="rosin-rammler",
        sizes_m=sizes,
        passing=passing,
        x50_m=x50_m,
        detail={
            "uniformity": uniformity,
            "characteristic_size_m": x50_m / math.log(2.0) ** (1.0 / uniformity),
            "source": "Amoako, Jha and Zhong 2022 doi:10.3390/mining2020013 Eq. 5",
        },
    )


def swebrec(
    x50_m: float,
    x_max_m: float,
    undulation: float,
    *,
    sizes_m: list[float] | None = None,
) -> SizeDistribution:
    """The Swebrec function, the distribution half of the KCO model.

    Ouchterlony 2005, as printed by Amoako 2022 Eqs. 10 and 11 and by Babaeian 2019::

        P(x) = 1 / (1 + f(x))
        f(x) = [ ln(x_max / x) / ln(x_max / x50) ]^b

    Three parameters rather than Rosin-Rammler's two, and the third is an explicit upper size limit.
    That limit is what fixes the coarse tail, which the two-parameter form gets wrong, and the extra
    curvature is what fixes the fines branch. Amoako 2022: the function "proves to be more adaptable
    and is able to predict fines better".

    ``x_max`` is conventionally the burden or the spacing, whichever is larger, which is the closure
    Babaeian 2019 states and the one used when a caller does not supply it.

    **A counter-example ships with this model rather than after it.** Babaeian 2019, on 24 blasts at
    one bauxite mine measured by image analysis, found the opposite for that site: the Rosin-Rammler
    range was closer to the measurement than the Swebrec range, and they concluded that site's
    distribution follows Rosin-Rammler. Swebrec is better in general and not better everywhere.
    """
    if not 0 < x50_m < x_max_m:
        raise ValueError(f"need 0 < x50 < x_max, got x50={x50_m} and x_max={x_max_m}")
    if undulation <= 0:
        raise ValueError(f"the undulation parameter must be positive, got {undulation}")
    sizes = sizes_m if sizes_m is not None else sieve_grid()
    denominator = math.log(x_max_m / x50_m)

    passing = []
    for x in sizes:
        if x >= x_max_m:
            passing.append(1.0)
            continue
        f = (math.log(x_max_m / x) / denominator) ** undulation
        passing.append(1.0 / (1.0 + f))
    return SizeDistribution(
        method="swebrec",
        sizes_m=sizes,
        passing=passing,
        x50_m=x50_m,
        detail={
            "x_max_m": x_max_m,
            "undulation": undulation,
            "source": "Ouchterlony 2005, as printed in Amoako 2022 Eqs. 10 and 11",
            "caveat": (
                "Babaeian et al. 2019 found Rosin-Rammler closer to image analysis at the Jajarm "
                "bauxite mine, so this function is more adaptable in general and not everywhere"
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class CrushZoneParameters:
    """The parameters of the two-branch crush-zone composition.

    **None of these constants is printed in any primary source held for this work.** Kanchibotla,
    Valery and Morrell 1999 and Djordjevic 1999 are proceedings papers that are not held; what is
    held is Amoako 2022's description of the mechanism, which is that two failure modes act at once,
    tensile fracturing producing the coarse fragments and compressive-shear fracturing in the crushed
    zone around the hole producing the fines, and that the model predicts the coarse branch with the
    classical model and the fine branch by modifying the distribution's parameters.

    So the STRUCTURE here is sourced and the CONSTANTS are the caller's. Defaults are stated as
    plausible starting values, not as published ones, and they are labelled that way in every result.
    """

    crossover_m: float = 0.01
    """Size below which the fines branch governs. A crushed-zone scale, order centimetres."""

    fines_uniformity: float = 0.8
    """Uniformity of the fines branch. Lower than the coarse branch, which is the mechanism."""

    fines_fraction: float = 0.05
    """Mass fraction generated in the crushed zone."""

    source: str = (
        "structure from Amoako, Jha and Zhong 2022 section 3 describing Kanchibotla et al. 1999 and "
        "Djordjevic 1999; the branch constants are NOT published in any source held for this work "
        "and are supplied by the caller"
    )


def crush_zone(
    x50_m: float,
    uniformity: float,
    parameters: CrushZoneParameters | None = None,
    *,
    sizes_m: list[float] | None = None,
) -> SizeDistribution:
    """A two-branch distribution: the classical coarse branch plus a crushed-zone fines branch.

    The coarse branch is Rosin-Rammler on the classical mean size. The fines branch is a second
    Rosin-Rammler with a lower uniformity, anchored at the crossover size, and the two are blended by
    mass so that the fines branch contributes exactly its stated fraction.

    This addresses the classical model's best-documented failure. Amoako 2022 states it plainly: "A
    major shortfall of the Kuz-Ram model is the underestimation of fines."

    Every result carries the source note above, so the distinction between the sourced mechanism and
    the caller's constants travels with the number.
    """
    params = parameters or CrushZoneParameters()
    if not 0.0 <= params.fines_fraction < 1.0:
        raise ValueError(f"the fines fraction must lie in [0, 1), got {params.fines_fraction}")
    if params.crossover_m >= x50_m:
        raise ValueError(
            f"the crushed-zone crossover ({params.crossover_m} m) must be finer than the mean size "
            f"({x50_m} m); above it there is no coarse branch left to compose with"
        )

    sizes = sizes_m if sizes_m is not None else sieve_grid()
    coarse = rosin_rammler(x50_m, uniformity, sizes_m=sizes)
    fines = rosin_rammler(params.crossover_m, params.fines_uniformity, sizes_m=sizes)

    w = params.fines_fraction
    passing = [w * f + (1.0 - w) * c for f, c in zip(fines.passing, coarse.passing)]

    blended = SizeDistribution(
        method="crush-zone",
        sizes_m=sizes,
        passing=passing,
        x50_m=x50_m,
        detail={
            "coarse_uniformity": uniformity,
            "parameters": params,
            "constants_are_published": False,
            "source": params.source,
        },
    )
    # The blend moves the fifty-percent size, so the reported mean is the blended one rather than the
    # coarse branch's. Reporting the input would misdescribe the curve actually returned.
    return SizeDistribution(
        method="crush-zone",
        sizes_m=sizes,
        passing=passing,
        x50_m=blended.percentile_m(0.5),
        detail=blended.detail | {"coarse_x50_m": x50_m},
    )


@dataclass(frozen=True, slots=True)
class TimingFactor:
    """The inter-hole delay multiplier of the 2005 modified classical model.

    Cunningham introduced it "mainly as a result of the introduction of electronic delay detonators"
    (Amoako 2022 section 3). Its tabulated values are not printed in any source held for this work.

    Two things about it must reach any screen that shows an initiation sequence.

    It is a **scalar**. It multiplies the mean size and has no spatial structure, so changing a
    tie-in from row-by-row to a V-cut, or reversing the initiation direction, changes the animation
    and changes nothing in the prediction. Only the aggregate delay moves the number.

    Its **shape is a hypothesis, not a fitted curve here**. The literature reports that fragmentation
    improves with delay up to a plateau, which refutes naive stress-wave collision arguments, but no
    copy of that work is held for this build and no number from it is used. The value is therefore
    the caller's, with a documented plausible range, and it is never fitted silently.
    """

    value: float = 1.0
    plausible_range: tuple[float, float] = (0.8, 1.2)
    source: str = (
        "Amoako, Jha and Zhong 2022 doi:10.3390/mining2020013 Eq. 8 names the factor; its tabulated "
        "values are not printed there and Cunningham 2005 is not held. Caller-supplied."
    )

    def __post_init__(self) -> None:
        low, high = self.plausible_range
        if not low <= self.value <= high:
            raise ValueError(
                f"the timing factor {self.value} is outside its documented plausible range "
                f"[{low}, {high}]. Widen the range explicitly if that is intended."
            )
