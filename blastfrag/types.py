"""Typed records shared by every model in the package.

The seven dimensionless features below are the ones the published corpus uses, and they are the
common currency of every statistical and learned model here. The classical models need something
the ratios cannot supply, rock volume and charge mass per hole, so they take a :class:`Pattern`
instead. Keeping the two apart in the type system is deliberate: a model that silently invented a
burden in order to run on ratio-only data would produce a plausible number from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

FEATURES: tuple[str, ...] = (
    "S_over_B",
    "H_over_B",
    "B_over_D",
    "T_over_B",
    "Pf_kg_m3",
    "XB_m",
    "E_GPa",
)
"""The seven model inputs, in the order every published equation in this package uses."""

Group = Literal[1, 2]
"""Rock-stiffness group. 1 is high Young modulus, 2 is low. See :mod:`blastfrag.grouping`."""


@dataclass(frozen=True, slots=True)
class Blast:
    """One bench blast as the published corpus records it: seven ratios and a measured size.

    ``x50_m`` is the measured mean fragment size in metres and may be ``None`` for a design that has
    not been fired. ``site`` is the source campaign, and it is the unit a leave-one-site-out split
    holds out, because rows within a site are not independent.
    """

    blast_id: str
    site: str
    S_over_B: float
    H_over_B: float
    B_over_D: float
    T_over_B: float
    Pf_kg_m3: float
    XB_m: float
    E_GPa: float
    x50_m: float | None = None
    group: Group | None = None
    meta: dict = field(default_factory=dict)

    def features(self) -> tuple[float, ...]:
        """The seven inputs in :data:`FEATURES` order."""
        return tuple(getattr(self, name) for name in FEATURES)


@dataclass(frozen=True, slots=True)
class Explosive:
    """An explosive, characterised by density and weight strength relative to ANFO.

    ``rws`` is the relative weight strength on the scale where ANFO is 100. The whole published
    corpus used ANFO, so ``rws`` is 100 there; a user-entered design can use anything.
    """

    name: str
    rws: float = 100.0
    density_kg_m3: float = 800.0
    source: str = ""


ANFO = Explosive(
    name="ANFO",
    rws=100.0,
    density_kg_m3=800.0,
    source="Hudaverdi, Kulatilake and Kuzu 2010 doi:10.1002/nag.957: every blast in the database used ANFO",
)


@dataclass(frozen=True, slots=True)
class Pattern:
    """An absolute bench-blast pattern, in metres, with the charge that fires it.

    This is what the classical mean-size equation needs and what the ratio table cannot give.
    ``burden_m`` and ``spacing_m`` are the drilled pattern; ``bench_height_m`` sets the rock beam;
    ``stemming_m`` is the uncharged collar; ``hole_diameter_mm`` closes the geometry.

    ``powder_factor_kg_m3`` is carried explicitly rather than derived from a charge length, because
    the corpus publishes it directly and deriving it would introduce an unpublished charge geometry.
    """

    burden_m: float
    spacing_m: float
    bench_height_m: float
    hole_diameter_mm: float
    stemming_m: float
    powder_factor_kg_m3: float
    subdrill_m: float = 0.0
    explosive: Explosive = ANFO
    drill_deviation_m: float = 0.0
    staggered: bool = False
    inter_hole_delay_ms: float | None = None
    n_rows: int = 1
    n_holes_per_row: int = 1
    provenance: str = ""

    @property
    def rock_volume_m3(self) -> float:
        """Rock volume broken per hole, burden by spacing by bench height."""
        return self.burden_m * self.spacing_m * self.bench_height_m

    @property
    def charge_mass_kg(self) -> float:
        """Explosive mass per hole, from the powder factor and the rock volume it breaks."""
        return self.powder_factor_kg_m3 * self.rock_volume_m3

    @property
    def charge_length_m(self) -> float:
        """Charged column length, bench plus subdrill minus stemming, floored at zero."""
        return max(0.0, self.bench_height_m + self.subdrill_m - self.stemming_m)

    @property
    def stiffness_ratio(self) -> float:
        """Bench height over burden, the rock-beam stiffness the corpus calls ``H/B``."""
        return self.bench_height_m / self.burden_m


@dataclass(frozen=True, slots=True)
class Rock:
    """The rock mass, as the rock-factor schemes and the structural cap need it.

    Every field is optional because the published corpus gives only Young modulus and in situ block
    size, and a scheme that needs more must say so by refusing rather than by defaulting.
    """

    E_GPa: float
    in_situ_block_m: float
    ucs_mpa: float | None = None
    density_t_m3: float | None = None
    rock_mass_description: Literal["powdery", "blocky", "vertically_jointed", "massive"] | None = None
    joint_spacing_m: float | None = None
    joint_plane_orientation: Literal[
        "horizontal", "dip_out_of_face", "strike_normal_to_face", "dip_into_face"
    ] | None = None
    name: str = ""


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model's answer for one blast, or its refusal to answer.

    A model that cannot run on a row returns ``x50_m = None`` with an ``abstain_reason``. That is a
    first-class result: it is reported as an abstention, never averaged away and never replaced by a
    guess. ``extrapolated`` marks a row outside the fitted envelope.
    """

    method: str
    blast_id: str
    x50_m: float | None
    abstain_reason: str | None = None
    extrapolated: bool = False
    group: Group | None = None
    detail: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.x50_m is None

    def __post_init__(self) -> None:
        if (self.x50_m is None) != (self.abstain_reason is not None):
            raise ValueError(
                f"{self.method} on {self.blast_id}: a prediction has a value or a reason to abstain, "
                "never both and never neither"
            )


@dataclass(frozen=True, slots=True)
class SizeDistribution:
    """A cumulative fragment-size distribution on a shared sieve grid.

    ``sizes_m`` is ascending and ``passing`` is the fraction (not percent) passing each size, so it
    is non-decreasing and lies in [0, 1]. Every distribution model in the package returns this shape,
    which is what lets them be overlaid and differenced.
    """

    method: str
    sizes_m: Sequence[float]
    passing: Sequence[float]
    x50_m: float
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sizes_m) != len(self.passing):
            raise ValueError(f"{self.method}: {len(self.sizes_m)} sizes against {len(self.passing)} values")
        if any(b <= a for a, b in zip(self.sizes_m, self.sizes_m[1:])):
            raise ValueError(f"{self.method}: sieve sizes must ascend strictly")
        if any(not (0.0 <= p <= 1.0) for p in self.passing):
            raise ValueError(f"{self.method}: passing fraction outside [0, 1]")
        if any(b < a - 1e-12 for a, b in zip(self.passing, self.passing[1:])):
            raise ValueError(f"{self.method}: passing fraction must not decrease with size")

    def passing_at(self, size_m: float) -> float:
        """Fraction passing a given mesh, linearly interpolated in log size."""
        import math

        xs, ys = list(self.sizes_m), list(self.passing)
        if size_m <= xs[0]:
            return ys[0]
        if size_m >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if size_m <= xs[i]:
                t = (math.log(size_m) - math.log(xs[i - 1])) / (math.log(xs[i]) - math.log(xs[i - 1]))
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]

    def percentile_m(self, fraction: float) -> float:
        """The mesh size passing a given fraction, for example 0.8 for the P80."""
        import math

        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must lie strictly inside (0, 1), got {fraction}")
        xs, ys = list(self.sizes_m), list(self.passing)
        if fraction <= ys[0]:
            return xs[0]
        if fraction >= ys[-1]:
            return xs[-1]
        for i in range(1, len(ys)):
            if fraction <= ys[i]:
                span = ys[i] - ys[i - 1]
                t = 0.0 if span <= 0 else (fraction - ys[i - 1]) / span
                return math.exp(math.log(xs[i - 1]) + t * (math.log(xs[i]) - math.log(xs[i - 1])))
        return xs[-1]
