"""Loading, validating and vouching for the published blast corpora.

Three real datasets ship with this package. Loading any of them runs the ingestion contract, and
loading the training corpus additionally runs a source-integrity gate that reproduces the source
paper's own descriptive statistics from the shipped rows.

That gate exists because it caught something. The shipped corpus had five transcription defects
against the published tables, two of them on the regression target, and the tell was that the
paper's own summary table reported a powder-factor maximum of 1.26 while the file contained 1.47.
A paper that prints a summary table has handed you a checksum; this module reads it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from typing import Iterable, Iterator, Sequence

from .types import FEATURES, Blast

__all__ = [
    "ContractViolation",
    "EnvelopeStat",
    "load_training_corpus",
    "load_holdout",
    "load_field_holdout",
    "load_all",
    "TRAINING_ENVELOPE",
    "PUBLISHED_DESCRIPTIVE_STATS",
    "DATASET_DIGEST",
    "compute_dataset_digest",
    "check_source_integrity",
    "validate_blast",
    "envelope_report",
]


class ContractViolation(ValueError):
    """Raised when a row fails the ingestion contract. Bad data is rejected, never coerced."""


@dataclass(frozen=True, slots=True)
class EnvelopeStat:
    """One feature's published descriptive statistics, used as the integrity checksum."""

    minimum: float
    maximum: float
    mean: float
    sd: float


PUBLISHED_DESCRIPTIVE_STATS: dict[str, EnvelopeStat] = {
    # Hudaverdi, Kulatilake and Kuzu 2010, doi:10.1002/nag.957, Table III. Transcribed verbatim.
    "S_over_B": EnvelopeStat(1.00, 1.75, 1.19, 0.117),
    "H_over_B": EnvelopeStat(1.33, 6.82, 3.34, 1.634),
    "B_over_D": EnvelopeStat(17.98, 39.47, 27.35, 4.838),
    "T_over_B": EnvelopeStat(0.50, 4.67, 1.26, 0.674),
    "Pf_kg_m3": EnvelopeStat(0.22, 1.26, 0.53, 0.236),
    "XB_m": EnvelopeStat(0.02, 2.35, 1.10, 0.533),
    "E_GPa": EnvelopeStat(9.57, 60.00, 29.46, 17.879),
}

# The hard rejection bounds of the ingestion contract. These are deliberately WIDER than the corpus
# envelope: a design outside the corpus is an extrapolation to be flagged, while a value outside
# these is not a blast at all and is rejected.
CONTRACT_BOUNDS: dict[str, tuple[float, float]] = {
    "S_over_B": (0.5, 3.0),
    "H_over_B": (0.5, 15.0),
    "B_over_D": (5.0, 80.0),
    "T_over_B": (0.1, 8.0),
    "Pf_kg_m3": (0.05, 3.0),
    "XB_m": (0.005, 10.0),
    "E_GPa": (0.5, 150.0),
}

X50_BOUNDS = (0.001, 5.0)
"""Measured mean fragment size must lie in these metres. Outside is a unit error, not a blast."""

DATASET_DIGEST = "9a72094d641a2557be4f22a100f4103d466dfbfe3830172dcf4e631979c56073"
"""SHA-256 of the canonical serialisation of the 97 training rows, pinned to the corrected corpus.

The descriptive-statistics gate below catches drift away from the source paper, but a mean is a weak
detector for a single-cell change: correcting one powder factor by 0.06 moves the mean by 0.0006 and
would slip through. This digest catches any change at all, and the two together are the contract:
the statistics say the file still IS the published table, the digest says it has not moved since.

Regenerate deliberately, with :func:`compute_dataset_digest`, only when the corpus is knowingly
corrected, and record why in the file's own header.
"""

TRAINING_ENVELOPE: dict[str, tuple[float, float]] = {
    name: (stat.minimum, stat.maximum) for name, stat in PUBLISHED_DESCRIPTIVE_STATS.items()
}
"""The fitted envelope. A row outside it is an extrapolation and every prediction on it is stamped."""


def _read_rows(filename: str) -> list[dict[str, str]]:
    """Read a shipped CSV, dropping the provenance comment block at the top."""
    text = resources.files(__package__).joinpath("data", filename).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines(keepends=True) if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _f(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    return None if raw == "" else float(raw)


def validate_blast(blast: Blast, *, allow_extrapolation: bool = False) -> list[str]:
    """Apply the ingestion contract to one blast.

    Returns the list of extrapolation warnings. Raises :class:`ContractViolation` on anything the
    contract rejects outright. When ``allow_extrapolation`` is false, a value outside the fitted
    envelope is also a rejection: the caller must opt in to predicting outside the data.
    """
    for name in FEATURES:
        value = getattr(blast, name)
        low, high = CONTRACT_BOUNDS[name]
        if not (low <= value <= high):
            raise ContractViolation(
                f"{blast.blast_id}: {name} = {value} is outside the contract bounds [{low}, {high}]. "
                "This is a unit or entry error, not an unusual blast."
            )
    if blast.x50_m is not None:
        low, high = X50_BOUNDS
        if not (low <= blast.x50_m <= high):
            raise ContractViolation(
                f"{blast.blast_id}: measured x50 = {blast.x50_m} m is outside [{low}, {high}] m"
            )

    warnings: list[str] = []
    for name in FEATURES:
        value = getattr(blast, name)
        low, high = TRAINING_ENVELOPE[name]
        if value < low or value > high:
            warnings.append(
                f"{name} = {value:g} is outside the fitted envelope [{low:g}, {high:g}]"
            )
    if warnings and not allow_extrapolation:
        raise ContractViolation(
            f"{blast.blast_id} lies outside the fitted envelope: "
            + "; ".join(warnings)
            + ". Pass allow_extrapolation=True to predict anyway; every result will be stamped."
        )
    return warnings


def _decimals(printed: float) -> int:
    """How many decimal places the source printed for a value, from its literal spelling."""
    text = f"{printed!r}"
    return len(text.split(".")[1]) if "." in text else 0


def _agrees_at_printed_precision(computed: float, printed: float, *, units: float = 0.5) -> bool:
    """Whether a computed statistic reproduces a printed one at the precision it was printed to.

    ``units`` is the allowed disagreement in units of the last printed digit. Half a unit is exact
    agreement under rounding; a full unit also admits truncation and a last-digit slip.
    """
    places = _decimals(printed)
    return abs(computed - printed) <= units * (10.0**-places) + 1e-9


# The source's summary table is not perfectly consistent with its own data tables, and the size of
# that inconsistency sets how tight this gate can be. Measured over the corrected corpus:
#
#   every MINIMUM and every MAXIMUM reproduces exactly, all 14 of them;
#   five of seven standard deviations reproduce exactly, the other two are one last-digit unit low
#     (powder factor 0.235455 against a printed 0.236, block size 0.532274 against 0.533);
#   five of seven means reproduce exactly, the other two are one unit high because the source
#     truncated rather than rounded (stiffness ratio 3.3452 printed as 3.34, block size 1.1067
#     printed as 1.10).
#
# So minima and maxima are gated EXACTLY, because they are facts read straight off the data tables
# and any disagreement there is a transcription defect. Means and standard deviations get one full
# last-digit unit, which is the measured inconsistency of the source itself and no more.
_STAT_TOLERANCE_UNITS = {"min": 0.5, "max": 0.5, "mean": 1.0, "sd": 1.0}


def compute_dataset_digest(blasts: Sequence[Blast]) -> str:
    """SHA-256 over the canonical serialisation of a blast list.

    Values are formatted to six decimals so the digest is stable across platforms and across a
    reformatting of the CSV, and depends only on the numbers themselves.
    """
    import hashlib

    lines = []
    for blast in sorted(blasts, key=lambda b: b.blast_id):
        cells = [blast.blast_id, blast.site, str(blast.group)]
        cells += [f"{getattr(blast, name):.6f}" for name in FEATURES]
        cells.append("" if blast.x50_m is None else f"{blast.x50_m:.6f}")
        lines.append(",".join(cells))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def check_source_integrity(blasts: Sequence[Blast], *, check_digest: bool = True) -> None:
    """Vouch for the shipped corpus in two independent ways.

    First, reproduce the source paper's own descriptive statistics from the shipped rows. The paper
    prints a minimum, maximum, mean and standard deviation per feature, and all four are recomputed
    and compared at the precision they were printed to. A mismatch means the shipped file no longer
    is the published table, which is exactly the failure this gate was written after finding: the
    file's powder-factor maximum read 1.47 where the paper prints 1.26.

    Second, compare a content digest against :data:`DATASET_DIGEST`. The statistics alone are a weak
    detector for a single-cell change, because correcting one powder factor moves a mean over 97 rows
    by less than a rounding step. The digest catches any change at all.

    Together they answer two different questions: the statistics say the file still is the published
    table, and the digest says it has not moved since it was corrected.
    """
    import math

    if len(blasts) != 97:
        raise ContractViolation(f"the training corpus is 97 blasts, got {len(blasts)}")

    problems: list[str] = []
    for name, published in PUBLISHED_DESCRIPTIVE_STATS.items():
        values = [getattr(b, name) for b in blasts]
        n = len(values)
        mean = sum(values) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        for label, got, want in (
            ("min", min(values), published.minimum),
            ("max", max(values), published.maximum),
            ("mean", mean, published.mean),
            ("sd", sd, published.sd),
        ):
            if not _agrees_at_printed_precision(got, want, units=_STAT_TOLERANCE_UNITS[label]):
                problems.append(
                    f"{name}.{label}: computed {got:.6f}, the source paper prints {want}"
                )

    if check_digest:
        digest = compute_dataset_digest(blasts)
        if digest != DATASET_DIGEST:
            problems.append(
                f"content digest {digest} does not match the pinned {DATASET_DIGEST}. The corpus has "
                "changed since it was corrected against the published tables."
            )

    if problems:
        raise ContractViolation("the shipped corpus failed its integrity gate:\n  " + "\n  ".join(problems))


def load_training_corpus(*, verify: bool = True) -> list[Blast]:
    """The 97 published bench blasts used to fit every model in this package.

    Hudaverdi, Kulatilake and Kuzu 2010, doi:10.1002/nag.957, Tables I and II.
    """
    blasts = [
        Blast(
            blast_id=row["blast_id"],
            site=row["site"],
            S_over_B=float(row["S_over_B"]),
            H_over_B=float(row["H_over_B"]),
            B_over_D=float(row["B_over_D"]),
            T_over_B=float(row["T_over_B"]),
            Pf_kg_m3=float(row["Pf_kg_m3"]),
            XB_m=float(row["XB_m"]),
            E_GPa=float(row["E_GPa"]),
            x50_m=float(row["x50_m"]),
            group=int(row["group"]),  # type: ignore[arg-type]
            meta={"source": "hudaverdi2010", "doi": "10.1002/nag.957", "role": "train"},
        )
        for row in _read_rows("hudaverdi1997.csv")
    ]
    if verify:
        check_source_integrity(blasts)
        for blast in blasts:
            validate_blast(blast, allow_extrapolation=False)
    return blasts


def load_holdout(*, protocol: str = "2012") -> list[Blast]:
    """The published validation blasts.

    ``protocol`` selects which published hold-out to reproduce:

    ``"2012"``
        The 12 blasts of Kulatilake, Hudaverdi and Wu 2012 Tables 4 and 5. Includes ``Rc1``, which
        is also a training blast: the leakage flag travels on the record.
    ``"2010"``
        The 13 blasts of Hudaverdi et al. 2010 Table VIII. Excludes ``Rc1``, includes ``Mi7`` and
        ``Ad25``.
    ``"union"``
        All 14 rows.
    ``"clean"``
        The union minus ``Rc1``, which is the only set here with no training membership at all.

    Each blast's ``meta`` carries the published competing predictions from both papers, so a
    reproduction can be scored against the reference it is reproducing.
    """
    valid = {"2012", "2010", "union", "clean"}
    if protocol not in valid:
        raise ValueError(f"protocol must be one of {sorted(valid)}, got {protocol!r}")

    out: list[Blast] = []
    for row in _read_rows("hudaverdi_holdout.csv"):
        in_2010 = row["in_2010_table8"] == "true"
        in_2012 = row["in_2012_table45"] == "true"
        leaked = row["in_training_table"] == "true"
        keep = {
            "2010": in_2010,
            "2012": in_2012,
            "union": True,
            "clean": not leaked,
        }[protocol]
        if not keep:
            continue
        out.append(
            Blast(
                blast_id=row["blast_id"],
                site=row["site"],
                S_over_B=float(row["S_over_B"]),
                H_over_B=float(row["H_over_B"]),
                B_over_D=float(row["B_over_D"]),
                T_over_B=float(row["T_over_B"]),
                Pf_kg_m3=float(row["Pf_kg_m3"]),
                XB_m=float(row["XB_m"]),
                E_GPa=float(row["E_GPa"]),
                x50_m=float(row["x50_measured_m"]),
                group=int(row["group"]),  # type: ignore[arg-type]
                meta={
                    "source": "hudaverdi2010+kulatilake2012",
                    "doi": "10.1002/nag.957, 10.1007/s10706-012-9496-3",
                    "role": "holdout",
                    "protocol": protocol,
                    "in_training_table": leaked,
                    "in_2010_table8": in_2010,
                    "in_2012_table45": in_2012,
                    "published": {
                        "regression_2010": _f(row, "x50_regression_2010_m"),
                        "kuzram_2010": _f(row, "x50_kuzram_2010_m"),
                        "regression_2012": _f(row, "x50_regression_2012_m"),
                        "kuzram_2012": _f(row, "x50_kuzram_2012_m"),
                        "nn_mean_2012": _f(row, "x50_nn_mean_2012_m"),
                    },
                },
            )
        )
    return out


def load_field_holdout() -> list[Blast]:
    """Five field blasts from a granite mine, published with their absolute pattern.

    Sui, Zhou, Zhao, Yang and Zou 2025, doi:10.3390/app15031254, CC BY 4.0, Tables 4 and 5.

    Every row here is an extrapolation: the Young modulus of 5.6 GPa sits below the corpus minimum
    of 9.57 GPa, on the feature both 2025 studies rank most important. The records carry the
    absolute geometry as well, so this set doubles as the independent check on the geometry
    reconstruction that the ratio-only corpus needs.
    """
    return [
        Blast(
            blast_id=row["blast_id"],
            site=row["site"],
            S_over_B=float(row["S_over_B"]),
            H_over_B=float(row["H_over_B"]),
            B_over_D=float(row["B_over_D"]),
            T_over_B=float(row["T_over_B"]),
            Pf_kg_m3=float(row["Pf_kg_m3"]),
            XB_m=float(row["XB_m"]),
            E_GPa=float(row["E_GPa"]),
            x50_m=float(row["x50_measured_m"]),
            group=None,
            meta={
                "source": "sui2025",
                "doi": "10.3390/app15031254",
                "licence": "CC BY 4.0",
                "role": "field-holdout",
                "extrapolated": True,
                "absolute": {
                    "burden_m": float(row["B_m"]),
                    "spacing_m": float(row["S_m"]),
                    "bench_height_m": float(row["H_m"]),
                    "hole_diameter_mm": float(row["D_mm"]),
                    "stemming_m": float(row["T_m"]),
                },
            },
        )
        for row in _read_rows("sui2025_field.csv")
    ]


def load_all() -> dict[str, list[Blast]]:
    """Every shipped dataset, keyed by role."""
    return {
        "train": load_training_corpus(),
        "holdout": load_holdout(protocol="union"),
        "field": load_field_holdout(),
    }


def envelope_report(blasts: Iterable[Blast]) -> dict[str, dict[str, float | int]]:
    """Per-feature coverage of the fitted envelope, for the coverage matrix.

    Reports how far each feature spans relative to the published corpus, so a case set that only
    exercises the middle of the envelope is visible as such rather than being described as wide.
    """
    import math

    rows = list(blasts)
    out: dict[str, dict[str, float | int]] = {}
    for name in FEATURES:
        values = [getattr(b, name) for b in rows]
        low, high = TRAINING_ENVELOPE[name]
        span = high - low
        mean = sum(values) / len(values)
        out[name] = {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "sd": math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)),
            "envelope_coverage": (max(values) - min(values)) / span if span > 0 else 0.0,
            "n_outside_envelope": sum(1 for v in values if v < low or v > high),
        }
    return out


def iter_sites(blasts: Iterable[Blast]) -> Iterator[str]:
    """The distinct sites present, in first-seen order."""
    seen: set[str] = set()
    for blast in blasts:
        if blast.site not in seen:
            seen.add(blast.site)
            yield blast.site
