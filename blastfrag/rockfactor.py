"""The rock factor: three published schemes that disagree, and a fourth recovered from data.

The classical mean-size equation multiplies everything by a single dimensionless rock factor, and
that factor carries the whole of "what kind of rock is this". Its original form was a three-value
lookup, which the literature itself calls too coarse: Hudaverdi et al. 2010 note that the rock-mass
categories the equation defines "are very wide, and thus need more precision".

Cunningham's route out is a rating scheme, and this is where a formula collection would quietly pick
one. Two primary sources in the corpus print rating tables under the same attribution and they are
**not the same table**. The strength term differs by a factor of six. At a uniaxial compressive
strength of 100 MPa the two give rock factors 0.85 apart, which moves a predicted fragment size by
roughly a quarter.

So all of them ship, side by side, named, with the source on each. Presenting one as "the" rock
factor would hide the subjectivity that is the real content of this parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .types import Rock

__all__ = [
    "RockFactorScheme",
    "SCHEMES",
    "rock_factor",
    "lilly_blastability_index",
    "RMD_RATINGS",
    "JPS_RATINGS",
    "JPO_RATINGS",
    "HUSTRULID_LOOKUP",
    "SITE_ROCK_FACTOR",
    "back_solve_rock_factor",
    "derive_site_rock_factors",
    "SITE_ROCK_FACTOR_2010",
]

SchemeName = Literal["lilly-hudaverdi", "lilly-babaeian", "hustrulid", "back-solved"]


# Hudaverdi, Kulatilake and Kuzu 2010, doi:10.1002/nag.957, quoting Lilly. Transcribed verbatim.
RMD_RATINGS: dict[str, int] = {"powdery": 10, "blocky": 20, "massive": 50}
JPS_RATINGS: tuple[tuple[float, int], ...] = ((0.1, 10), (1.0, 20), (float("inf"), 50))
JPO_RATINGS: dict[str, int] = {
    "horizontal": 10,
    "dip_out_of_face": 20,
    "strike_normal_to_face": 30,
    "dip_into_face": 40,
}

# Babaeian, Ataei, Sereshki, Sotoudeh and Mohammadi 2019, doi:10.1016/j.jrmge.2018.11.006, Table 2,
# also attributed to Lilly. Its rock-mass descriptions differ ("vertically jointed" where the other
# says "blocky"), its joint-spacing bands are expressed against the pattern rather than in metres,
# and, decisively, its strength term is UCS/3 or UCS/5 by modulus where the other is 0.05 * UCS.
RMD_RATINGS_BABAEIAN: dict[str, int] = {"powdery": 10, "vertically_jointed": 20, "massive": 50}

# Hustrulid 1999, as tabulated by Babaeian 2019 Table 1: a Protodyakonov strength index mapped
# straight onto a rock factor, bypassing the rating sum entirely.
HUSTRULID_LOOKUP: tuple[tuple[str, float, float, float], ...] = (
    ("very soft", 3.0, 5.0, 3.0),
    ("soft", 5.0, 8.0, 5.0),
    ("medium soft", 8.0, 10.0, 7.0),
    ("rigid", 10.0, 14.0, 10.0),
    ("rigid and homogeneous", 12.0, 16.0, 13.0),
)


class RockFactorUnavailable(ValueError):
    """Raised when a scheme needs a rock property that was not supplied.

    Every scheme refuses rather than defaulting. A rock factor produced from an assumed uniaxial
    compressive strength is a number with no evidence behind it, and it propagates straight into a
    predicted fragment size.
    """


def _joint_spacing_rating(spacing_m: float) -> int:
    for bound, rating in JPS_RATINGS:
        if spacing_m < bound:
            return rating
    return JPS_RATINGS[-1][1]


def lilly_blastability_index(rock: Rock, *, scheme: SchemeName = "lilly-hudaverdi") -> float:
    """Lilly's blastability index, as the named source prints it.

    Hudaverdi 2010 form, its Eq. 3::

        BI = 0.5 * (RMD + JPS + JPO + RDI + S)
        RDI = 25 * density - 50
        S   = 0.05 * UCS

    Babaeian 2019 Table 2 form: the same sum, with the strength term replaced by a hardness factor
    that is ``UCS / 3`` below 50 GPa of Young modulus and ``UCS / 5`` above it.
    """
    missing = [
        name
        for name, value in (
            ("rock_mass_description", rock.rock_mass_description),
            ("joint_spacing_m", rock.joint_spacing_m),
            ("joint_plane_orientation", rock.joint_plane_orientation),
            ("density_t_m3", rock.density_t_m3),
            ("ucs_mpa", rock.ucs_mpa),
        )
        if value is None
    ]
    if missing:
        raise RockFactorUnavailable(
            f"the {scheme} scheme needs {', '.join(missing)}, which this rock does not carry. "
            "Supply them or choose a scheme that does not require them."
        )
    assert rock.density_t_m3 is not None and rock.ucs_mpa is not None
    assert rock.joint_spacing_m is not None

    if scheme == "lilly-hudaverdi":
        ratings = RMD_RATINGS
        strength = 0.05 * rock.ucs_mpa
    elif scheme == "lilly-babaeian":
        ratings = RMD_RATINGS_BABAEIAN
        strength = rock.ucs_mpa / (3.0 if rock.E_GPa < 50.0 else 5.0)
    else:
        raise ValueError(f"{scheme!r} is not a blastability-index scheme")

    if rock.rock_mass_description not in ratings:
        raise RockFactorUnavailable(
            f"the {scheme} scheme has no rating for a {rock.rock_mass_description!r} rock mass; "
            f"it recognises {sorted(ratings)}"
        )

    rmd = ratings[rock.rock_mass_description]
    jps = _joint_spacing_rating(rock.joint_spacing_m)
    jpo = JPO_RATINGS[rock.joint_plane_orientation]
    rdi = 25.0 * rock.density_t_m3 - 50.0
    return 0.5 * (rmd + jps + jpo + rdi + strength)


def _lilly_factor(rock: Rock, scheme: SchemeName) -> float:
    # Hudaverdi 2010 Eq. 2: the rock factor is 0.06 times the blastability index.
    return 0.06 * lilly_blastability_index(rock, scheme=scheme)


def _hustrulid_factor(rock: Rock) -> float:
    """Map a Protodyakonov strength index onto a rock factor.

    The Protodyakonov index is not one of the corpus's fields, so it is read from ``Rock.meta``-like
    supply through the uniaxial compressive strength using the usual ``UCS / 10`` approximation in
    megapascals. That approximation is stated here rather than hidden: it is a convention, not a
    measurement, and it is why this scheme is offered as a coarse cross-check rather than a default.
    """
    if rock.ucs_mpa is None:
        raise RockFactorUnavailable(
            "the hustrulid scheme needs a uniaxial compressive strength to reach a Protodyakonov "
            "index; this rock does not carry one"
        )
    protodyakonov = rock.ucs_mpa / 10.0
    for _label, low, high, factor in HUSTRULID_LOOKUP:
        if low <= protodyakonov <= high:
            return factor
    if protodyakonov < HUSTRULID_LOOKUP[0][1]:
        return HUSTRULID_LOOKUP[0][3]
    return HUSTRULID_LOOKUP[-1][3]


@dataclass(frozen=True, slots=True)
class RockFactorScheme:
    """One named way of turning a rock description into the classical model's rock factor."""

    name: SchemeName
    source: str
    compute: Callable[[Rock], float]
    note: str = ""


SCHEMES: dict[SchemeName, RockFactorScheme] = {
    "lilly-hudaverdi": RockFactorScheme(
        name="lilly-hudaverdi",
        source="Hudaverdi, Kulatilake and Kuzu 2010, doi:10.1002/nag.957, Eqs. 2 and 3, after Lilly",
        compute=lambda rock: _lilly_factor(rock, "lilly-hudaverdi"),
        note="strength term is 0.05 times the uniaxial compressive strength",
    ),
    "lilly-babaeian": RockFactorScheme(
        name="lilly-babaeian",
        source="Babaeian et al. 2019, doi:10.1016/j.jrmge.2018.11.006, Table 2, after Lilly 1986",
        compute=lambda rock: _lilly_factor(rock, "lilly-babaeian"),
        note=(
            "same attribution, different table: the strength term is UCS/3 below 50 GPa and UCS/5 "
            "above, which is six to ten times the other scheme's term at the same strength"
        ),
    ),
    "hustrulid": RockFactorScheme(
        name="hustrulid",
        source="Hustrulid 1999, as tabulated by Babaeian et al. 2019 Table 1",
        compute=_hustrulid_factor,
        note=(
            "a five-band lookup on the Protodyakonov index, which is reached here from the "
            "uniaxial compressive strength by the conventional division by ten; a coarse "
            "cross-check, not a default"
        ),
    ),
}


def rock_factor(rock: Rock, *, scheme: SchemeName = "lilly-hudaverdi") -> float:
    """The classical model's rock factor under a named scheme.

    Raises :class:`RockFactorUnavailable` when the scheme needs a property the rock does not carry.
    """
    if scheme == "back-solved":
        raise ValueError(
            "the back-solved factor is per site rather than per rock; use SITE_ROCK_FACTOR or "
            "back_solve_rock_factor"
        )
    if scheme not in SCHEMES:
        raise ValueError(f"{scheme!r} is not a known scheme; try {sorted(SCHEMES)}")
    return SCHEMES[scheme].compute(rock)


def compare_schemes(rock: Rock) -> dict[str, float | str]:
    """Every scheme's answer for the same rock, so the disagreement is visible rather than argued."""
    out: dict[str, float | str] = {}
    for name, scheme in SCHEMES.items():
        try:
            out[name] = scheme.compute(rock)
        except RockFactorUnavailable as exc:
            out[name] = f"unavailable: {exc}"
    return out


# ---------------------------------------------------------------------------------------------
# The fourth scheme: recovered from the data rather than from a rating table.
# ---------------------------------------------------------------------------------------------

def back_solve_rock_factor(
    published_x50_m: float,
    *,
    rock_volume_m3: float,
    charge_mass_kg: float,
    rws: float = 100.0,
) -> float:
    """Recover the rock factor implied by a published classical prediction.

    Both source papers state that the rock factor "was estimated for each blast" and neither prints a
    value. Inverting the classical mean-size equation on a published prediction recovers it::

        A = x50 / [ (V/Q)^0.8 * Q^(1/6) * (RWS/115)^(-19/30) ]

    with the size in centimetres, which is the equation's native unit.

    This is only meaningful because the recovered values turn out to be near constant within each
    site, which they are: 0.6 percent spread at Murgul, 1.6 at Ozmert, 3 at Akdaglar, all of it
    consistent with the predictions having been rounded to two decimals. That constancy is doing
    double duty. It validates the geometry reconstruction, because an error in burden or bench height
    would scatter the factor within a site, and it selects between two published spellings of the
    explosive-strength exponent.
    """
    if rock_volume_m3 <= 0 or charge_mass_kg <= 0:
        raise ValueError("rock volume and charge mass must be positive")
    base_cm = (
        (rock_volume_m3 / charge_mass_kg) ** 0.8
        * charge_mass_kg ** (1.0 / 6.0)
        * (rws / 115.0) ** (-19.0 / 30.0)
    )
    return published_x50_m * 100.0 / base_cm


def derive_site_rock_factors(
    column: str = "kuzram_2012",
) -> dict[str, tuple[float, int, float]]:
    """Recover a rock factor per site from a published prediction column.

    Returns ``site -> (mean factor, blasts used, relative spread)``.

    **Which column matters, and mixing them is a defect.** The two source papers print different
    classical predictions for the same blasts, so they imply different rock factors: Murgul comes
    out at 9.23 from the 2012 column and 10.05 from the 2010 one. The default is the 2012 column
    because that is the table carrying all three published arms, which is the table the benchmark
    reproduces. The 2010 column is available for comparison and the difference is a result, not
    noise to be averaged over.
    """
    from .datasets import load_holdout
    from .geometry import has_absolute_geometry, reconstruct_pattern

    per_site: dict[str, list[float]] = {}
    for blast in load_holdout(protocol="union"):
        published = blast.meta["published"].get(column)
        if published is None or not has_absolute_geometry(blast):
            continue
        pattern = reconstruct_pattern(blast)
        per_site.setdefault(blast.site, []).append(
            back_solve_rock_factor(
                published,
                rock_volume_m3=pattern.rock_volume_m3,
                charge_mass_kg=pattern.charge_mass_kg,
            )
        )

    out: dict[str, tuple[float, int, float]] = {}
    for site, values in per_site.items():
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / mean if len(values) > 1 else 0.0
        out[site] = (mean, len(values), spread)
    return out


# Derived from the 2012 published prediction column by `derive_site_rock_factors`, pinned here so
# the classical arm is reproducible without recomputing, and asserted against that function in the
# test suite. These are DERIVED, not published, and every prediction that uses one says so.
#
# The evidence they are right is that they barely move within a site: the widest spread is 3.7
# percent across the two Akdaglar blasts, which is what rounding the published predictions to two
# decimals produces. An error in the geometry reconstruction would scatter them.
#
# Miami has no reconstructable geometry, so no factor can be recovered for it. Absent by design, and
# the classical arm abstains there rather than borrowing a neighbour's.
SITE_ROCK_FACTOR: dict[str, float] = {
    "Reocin": 12.14,
    "Reocin-UG": 11.19,
    "Enusa": 10.97,
    "Murgul": 9.23,
    "Akdaglar": 6.72,
    "Ozmert": 6.44,
    "Mrica": 6.32,
    "Soma": 6.24,
    "Dongri-Buzurg": 3.68,
}

SITE_ROCK_FACTOR_2010: dict[str, float] = {
    # The same recovery from the earlier paper's column, for comparison. Systematically higher by
    # about 8 percent, because that paper's classical predictions are systematically larger.
    "Reocin-UG": 12.22,
    "Enusa": 11.97,
    "Murgul": 10.05,
    "Mrica": 7.11,
    "Akdaglar": 7.03,
    "Ozmert": 7.04,
    "Soma": 6.78,
    "Dongri-Buzurg": 4.14,
}
