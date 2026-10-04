"""Diagnostics that inform a reading of the benchmark without changing it.

Nothing in this module filters a row, refits an arm or edits a score. Each function reports
something about the corpus or about a fitted arm, and the caller decides what to show.

``outlier_screen``
    An Isolation Forest (Liu, Ting and Zhou 2008, doi:10.1109/ICDM.2008.17) over the corpus. Huan et
    al. 2025 (doi:10.1038/s41598-025-96005-7) ran the same kind of screen on a 105-sample superset of
    this corpus and flagged five samples. Here it is a report: on a corpus of 97 blasts from ten
    sites, the rows a density method calls unusual are mostly the rows of an unusual SITE, and
    removing them would quietly remove the hardest part of the cross-site question.

``native_importance``
    The model-internal feature importances of a fitted forest or boosting arm, for comparison with the
    ranking Sui et al. 2025 publish for the same two models on this corpus.

``grouped_resampling_importance``
    How much each input matters to an arm's predictions on a site it has not seen: under
    leave-one-site-out, the held-out site's values of one feature are replaced by values drawn from
    the training rows, and the increase in squared error is recorded. Within one site several inputs
    are constant (Young's modulus always is), so permuting inside the test fold would measure nothing;
    drawing from the training rows asks what happens when that site looked like another one.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Callable, Sequence

from .models import Arm
from .splits import leave_one_site_out
from .types import FEATURES, Blast

__all__ = [
    "PUBLISHED_IMPORTANCE",
    "outlier_screen",
    "native_importance",
    "grouped_resampling_importance",
]

PUBLISHED_IMPORTANCE: dict[str, dict[str, float]] = {
    "random-forest": {"E_GPa": 0.7129},
    "xgboost": {"E_GPa": 0.4608},
}
"""The importances Sui et al. 2025 print for the modulus, the one value per model the text gives in
full. Their ranking, in words: the modulus first for both models, then in-situ block size and the
stemming ratio, with the spacing, bench-height and burden ratios least important."""


def outlier_screen(
    blasts: Sequence[Blast],
    *,
    contamination: float = 5 / 105,
    n_estimators: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    """Score every blast with an Isolation Forest and report the most isolated ones.

    The inputs are the seven features and the logarithm of the measured size, each standardised.
    Including the target is this package's choice; the copy of Huan et al. held for this work does
    not state which columns its screen used. ``contamination`` defaults to their observed rate, five
    of 105.
    """
    import numpy as np
    from sklearn.ensemble import IsolationForest

    rows = [b for b in blasts if b.x50_m is not None and b.x50_m > 0]
    X = np.array([[*b.features(), math.log(b.x50_m)] for b in rows])  # type: ignore[arg-type]
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    model = IsolationForest(
        n_estimators=n_estimators, contamination=contamination, random_state=seed
    ).fit(Z)
    anomaly = -model.score_samples(Z)  # larger is more isolated
    flagged = [b.blast_id for b, flag in zip(rows, model.predict(Z)) if flag == -1]
    by_site: dict[str, int] = {}
    for blast in rows:
        if blast.blast_id in flagged:
            by_site[blast.site] = by_site.get(blast.site, 0) + 1
    return {
        "method": "Isolation Forest, Liu, Ting and Zhou 2008",
        "inputs": [*FEATURES, "ln x50_m"],
        "standardised": True,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "seed": seed,
        "n_blasts": len(rows),
        "flagged": flagged,
        "flagged_by_site": by_site,
        "anomaly_score": {b.blast_id: float(a) for b, a in zip(rows, anomaly)},
        "applied_as_filter": False,
    }


def native_importance(arm: Arm) -> dict[str, object] | None:
    """The fitted model's own feature importances, or ``None`` for an arm that has none.

    Forest importances are the mean decrease in impurity; XGBoost's are the library's default
    importance type, which is named in the result because the two are different quantities. For the
    stacking arm both base learners are reported.
    """
    model = getattr(arm, "model", None)
    if model is None:
        return None

    def read(estimator) -> dict[str, float] | None:
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            return None
        return {name: float(v) for name, v in zip(FEATURES, values)}

    if hasattr(model, "forest") and hasattr(model, "boosting"):
        return {
            "kind": "stacking base learners",
            "forest": read(model.forest),
            "boosting": read(model.boosting),
            "boosting_importance_type": getattr(model.boosting, "importance_type", None) or "gain",
        }
    values = read(model)
    if values is None:
        return None
    kind = "mean decrease in impurity"
    if type(model).__name__.startswith("XGB"):
        kind = f"xgboost {getattr(model, 'importance_type', None) or 'gain'}"
    return {"kind": kind, "values": values}


def grouped_resampling_importance(
    factory: Callable[[], Arm],
    blasts: Sequence[Blast],
    *,
    n_repeats: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Increase in held-out squared error when one input is resampled from the training rows.

    Leave-one-site-out, one fresh arm per fold. For each fold and each feature, the test rows' values
    of that feature are replaced by values drawn with replacement from the fold's training rows,
    ``n_repeats`` times, and the arm predicts again. A row enters the comparison only where the arm
    answered both before and after, and the counts are returned so a reader can see how many did.

    Returns, per feature, the mean increase in mean squared error over every fold and repeat, and
    each feature's share of the summed positive increases.
    """
    rng = random.Random(seed)
    base_sq: list[float] = []
    increases: dict[str, list[float]] = {name: [] for name in FEATURES}
    n_rows = 0
    for split in leave_one_site_out(blasts):
        arm = factory()
        try:
            arm.fit(split.train)
        except ValueError:
            continue
        test = [b for b in split.test if b.x50_m is not None]
        base = {p.blast_id: p.x50_m for p in arm.predict(test)}
        answered = [b for b in test if base.get(b.blast_id) is not None]
        if not answered:
            continue
        n_rows += len(answered)
        base_sq.extend((base[b.blast_id] - b.x50_m) ** 2 for b in answered)  # type: ignore[operator]
        for name in FEATURES:
            pool = [getattr(b, name) for b in split.train]
            for _ in range(n_repeats):
                perturbed = [
                    dataclasses.replace(b, **{name: rng.choice(pool)}) for b in answered
                ]
                after = {p.blast_id: p.x50_m for p in arm.predict(perturbed)}
                for blast in answered:
                    value = after.get(blast.blast_id)
                    if value is None:
                        continue
                    before = (base[blast.blast_id] - blast.x50_m) ** 2  # type: ignore[operator]
                    increases[name].append((value - blast.x50_m) ** 2 - before)  # type: ignore[operator]

    mean_increase = {
        name: (sum(v) / len(v) if v else None) for name, v in increases.items()
    }
    positive = {name: max(v or 0.0, 0.0) for name, v in mean_increase.items()}
    total = sum(positive.values())
    return {
        "method": (
            "leave-one-site-out; each feature of the held-out site resampled from the training "
            "rows; mean increase in squared error"
        ),
        "n_repeats": n_repeats,
        "seed": seed,
        "n_rows": n_rows,
        "baseline_mse": sum(base_sq) / len(base_sq) if base_sq else None,
        "mean_increase_in_mse": mean_increase,
        "share": {name: (v / total if total > 0 else None) for name, v in positive.items()},
    }
