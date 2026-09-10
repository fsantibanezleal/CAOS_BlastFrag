"""blastfrag: blast-fragmentation prediction from a bench pattern, with the honest statistic.

The package implements the published ladder for predicting muckpile fragment size from a
drill-and-blast design, from the 1973 classical mean-size equation to the 2025 stacking ensemble,
and it scores every one of them against the same real measured blasts under three split protocols.

Three things distinguish it from a formula collection.

**It refuses rather than guesses.** A model that needs rock volume and charge mass cannot run on a
dimensionless corpus, so it abstains with a reason instead of inventing a burden. Abstention is a
first-class result.

**It names the statistic.** Two different quantities are called ``R2`` in this literature and they
differ by a factor of two and a half on the published hold-out. Every figure this package returns
carries the name of what it is, and a null model that predicts the training mean runs beside it.

**It checks its own data against the paper it came from.** Loading the training corpus reproduces
the source's own descriptive statistics from the shipped rows. That gate found five transcription
defects, two of them on the regression target.

Quick start::

    import blastfrag as bf

    train = bf.load_training_corpus()
    holdout = bf.load_holdout(protocol="2012")

    kuzram = bf.KuzRam()
    preds = [kuzram.predict(b) for b in holdout]
    print(bf.score(holdout, preds))
"""

from __future__ import annotations

__version__ = "0.01.000"
__all__ = [
    "__version__",
    # types
    "Blast",
    "Pattern",
    "Rock",
    "Explosive",
    "ANFO",
    "Prediction",
    "SizeDistribution",
    "FEATURES",
    # data
    "load_training_corpus",
    "load_holdout",
    "load_field_holdout",
    "load_all",
    "check_source_integrity",
    "validate_blast",
    "envelope_report",
    "ContractViolation",
    "TRAINING_ENVELOPE",
    "PUBLISHED_DESCRIPTIVE_STATS",
    # geometry
    "SITE_GEOMETRY",
    "GeometryUnavailable",
    "has_absolute_geometry",
    "reconstruct_pattern",
    "verify_reconstruction",
]

from .datasets import (
    PUBLISHED_DESCRIPTIVE_STATS,
    TRAINING_ENVELOPE,
    ContractViolation,
    check_source_integrity,
    envelope_report,
    load_all,
    load_field_holdout,
    load_holdout,
    load_training_corpus,
    validate_blast,
)
from .geometry import (
    SITE_GEOMETRY,
    GeometryUnavailable,
    has_absolute_geometry,
    reconstruct_pattern,
    verify_reconstruction,
)
from .types import (
    ANFO,
    FEATURES,
    Blast,
    Explosive,
    Pattern,
    Prediction,
    Rock,
    SizeDistribution,
)
