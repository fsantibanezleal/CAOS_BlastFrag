"""Scoring predictions, with the name of every statistic attached to it.

Two different quantities are called ``R2`` in the blast-fragmentation literature, and on the
published twelve-blast hold-out they differ by a factor of two and a half for the same arm. That is
not a subtlety, it is the difference between "this model is usable" and "this model is barely better
than a constant", and the sources report the flattering one.

So this module never returns a bare ``R2``. It returns a :class:`Score` whose fields are named for
what they are, and it always computes the null model alongside, because "better than nothing" is a
claim that should be visible rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .types import Blast, Prediction

__all__ = [
    "Score",
    "score",
    "null_model_score",
    "bootstrap_interval",
    "worst_rows",
]


@dataclass(frozen=True, slots=True)
class Score:
    """Every headline number for one arm on one set of blasts, each named for what it is."""

    method: str
    n_scored: int
    n_abstained: int
    n_extrapolated: int

    pearson_r: float | None
    """Correlation between predicted and measured. Says how well the arm RANKS blasts."""

    pearson_r2: float | None
    """The square of that correlation. This is the figure the source papers report as ``R2``."""

    r2_identity: float | None
    """Variance explained about the 1:1 line. This is what a reader assumes ``R2`` means."""

    r2_parity_fit: float | None
    """Fit quality of a free straight line through the parity scatter, the sources' figure panels."""

    parity_slope: float | None
    parity_intercept: float | None

    rmse_m: float | None
    mae_m: float | None
    mape_pct: float | None
    bias_m: float | None
    """Signed mean error. A near-zero bias with a large error is exactly the case where the two
    variance statistics diverge, which is what the classical arm does on this corpus."""

    detail: dict = field(default_factory=dict)

    def summary(self) -> str:
        """A one-line reading that cannot be mistaken for a bare ``R2``."""
        if self.n_scored == 0:
            return f"{self.method}: nothing scored, {self.n_abstained} abstained"
        return (
            f"{self.method}: variance explained about identity {self.r2_identity:.3f}, "
            f"squared correlation {self.pearson_r2:.3f}, RMSE {self.rmse_m:.4f} m, "
            f"MAPE {self.mape_pct:.1f} percent, n = {self.n_scored}"
            + (f", {self.n_abstained} abstained" if self.n_abstained else "")
        )


def _paired(
    blasts: Sequence[Blast], predictions: Sequence[Prediction]
) -> tuple[list[float], list[float], list[Prediction], list[Prediction]]:
    """Line up measurements with predictions, separating out the abstentions."""
    by_id = {p.blast_id: p for p in predictions}
    measured: list[float] = []
    predicted: list[float] = []
    abstained: list[Prediction] = []
    scored: list[Prediction] = []
    for blast in blasts:
        prediction = by_id.get(blast.blast_id)
        if prediction is None:
            raise KeyError(
                f"no prediction for {blast.blast_id}. A missing cell is not the same as an "
                "abstention: return a Prediction with an abstain_reason instead of omitting it."
            )
        if blast.x50_m is None:
            continue
        if prediction.abstained:
            abstained.append(prediction)
            continue
        assert prediction.x50_m is not None
        measured.append(blast.x50_m)
        predicted.append(prediction.x50_m)
        scored.append(prediction)
    return measured, predicted, scored, abstained


def score(
    blasts: Sequence[Blast],
    predictions: Sequence[Prediction],
    *,
    method: str | None = None,
) -> Score:
    """Score one arm against measured sizes.

    Abstentions are counted and excluded from the statistics rather than being treated as errors or
    silently dropped. An arm that abstains on half the rows and scores well on the rest has not
    beaten an arm that answered everything, and the counts are on the result so that is visible.
    """
    measured, predicted, scored, abstained = _paired(blasts, predictions)
    name = method or (predictions[0].method if predictions else "unnamed")
    n_extrapolated = sum(1 for p in scored if p.extrapolated)

    if len(measured) < 2:
        return Score(
            method=name,
            n_scored=len(measured),
            n_abstained=len(abstained),
            n_extrapolated=n_extrapolated,
            pearson_r=None,
            pearson_r2=None,
            r2_identity=None,
            r2_parity_fit=None,
            parity_slope=None,
            parity_intercept=None,
            rmse_m=None,
            mae_m=None,
            mape_pct=None,
            bias_m=None,
            detail={"reason": "fewer than two scoreable rows"},
        )

    n = len(measured)
    mean_measured = sum(measured) / n
    mean_predicted = sum(predicted) / n

    cov = sum((m - mean_measured) * (p - mean_predicted) for m, p in zip(measured, predicted))
    var_m = sum((m - mean_measured) ** 2 for m in measured)
    var_p = sum((p - mean_predicted) ** 2 for p in predicted)
    r = cov / math.sqrt(var_m * var_p) if var_m > 0 and var_p > 0 else float("nan")

    ss_res = sum((m - p) ** 2 for m, p in zip(measured, predicted))
    r2_identity = 1.0 - ss_res / var_m if var_m > 0 else float("nan")

    # The parity-line fit the source figures show: regress predicted ON measured, then report how
    # much of the predicted variance that line explains.
    slope = cov / var_m if var_m > 0 else float("nan")
    intercept = mean_predicted - slope * mean_measured
    fitted = [slope * m + intercept for m in measured]
    ss_fit = sum((p - f) ** 2 for p, f in zip(predicted, fitted))
    r2_parity = 1.0 - ss_fit / var_p if var_p > 0 else float("nan")

    return Score(
        method=name,
        n_scored=n,
        n_abstained=len(abstained),
        n_extrapolated=n_extrapolated,
        pearson_r=r,
        pearson_r2=r * r,
        r2_identity=r2_identity,
        r2_parity_fit=r2_parity,
        parity_slope=slope,
        parity_intercept=intercept,
        rmse_m=math.sqrt(ss_res / n),
        mae_m=sum(abs(m - p) for m, p in zip(measured, predicted)) / n,
        mape_pct=100.0 * sum(abs((m - p) / m) for m, p in zip(measured, predicted)) / n,
        bias_m=sum(p - m for m, p in zip(measured, predicted)) / n,
        detail={
            "abstain_reasons": sorted({p.abstain_reason or "" for p in abstained}),
            "mean_measured_m": mean_measured,
        },
    )


def null_model_score(blasts: Sequence[Blast], *, training_mean_m: float) -> Score:
    """Score the arm that predicts the training mean for every blast.

    This runs on every case in this package, because "the model works" is a comparative claim and
    the comparison has to be drawn somewhere. On the published hold-out the classical arm beats this
    by 13 percent on root-mean-square error, which is the single most useful number about it.
    """
    import dataclasses

    predictions = [
        Prediction(method="null", blast_id=b.blast_id, x50_m=training_mean_m) for b in blasts
    ]
    result = score(blasts, predictions, method=f"null (predict {training_mean_m:.3f} m)")
    return dataclasses.replace(
        result, detail=result.detail | {"training_mean_m": training_mean_m}
    )


def bootstrap_interval(
    blasts: Sequence[Blast],
    predictions: Sequence[Prediction],
    statistic: str = "r2_identity",
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    unit: str = "blast",
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and a bootstrap interval, resampling the right unit.

    ``unit`` is ``"blast"`` or ``"site"``. **The choice matters and the default is not always right.**
    This corpus is heavily clustered: one quarry supplies 22 of its 97 rows, so rows are not
    independent draws and resampling them understates the uncertainty. Under a leave-one-site-out
    protocol the resampling unit must be the site, and passing ``unit="site"`` is how that is said.

    Returns ``(point, low, high)``.
    """
    import random

    if unit not in {"blast", "site"}:
        raise ValueError(f"unit must be 'blast' or 'site', got {unit!r}")

    by_id = {p.blast_id: p for p in predictions}
    point_score = score(blasts, predictions)
    point = getattr(point_score, statistic)
    if point is None:
        raise ValueError(f"the point estimate of {statistic} is undefined on these rows")

    if unit == "blast":
        groups: list[list[Blast]] = [[b] for b in blasts]
    else:
        by_site: dict[str, list[Blast]] = {}
        for blast in blasts:
            by_site.setdefault(blast.site, []).append(blast)
        groups = list(by_site.values())

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample: list[Blast] = []
        for _ in range(len(groups)):
            sample.extend(rng.choice(groups))
        try:
            value = getattr(score(sample, [by_id[b.blast_id] for b in sample]), statistic)
        except (KeyError, ZeroDivisionError):
            continue
        if value is not None and not math.isnan(value):
            draws.append(value)

    if len(draws) < n_boot // 10:
        raise ValueError(f"only {len(draws)} of {n_boot} bootstrap draws were scoreable")

    draws.sort()
    tail = (1.0 - level) / 2.0
    low = draws[max(0, int(tail * len(draws)) - 1)]
    high = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
    return point, low, high


def worst_rows(
    blasts: Sequence[Blast], predictions: Sequence[Prediction], *, n: int = 3
) -> list[dict[str, object]]:
    """The rows an arm gets most wrong, which is the part an engineer can act on.

    A summary statistic says a model is imperfect. This says where, by how much and in which
    direction, which is what turns a benchmark into a case note.
    """
    by_id = {p.blast_id: p for p in predictions}
    rows: list[dict[str, object]] = []
    for blast in blasts:
        prediction = by_id.get(blast.blast_id)
        if prediction is None or prediction.abstained or blast.x50_m is None:
            continue
        assert prediction.x50_m is not None
        error = prediction.x50_m - blast.x50_m
        rows.append(
            {
                "blast_id": blast.blast_id,
                "site": blast.site,
                "measured_m": blast.x50_m,
                "predicted_m": prediction.x50_m,
                "error_m": error,
                "error_pct": 100.0 * error / blast.x50_m,
            }
        )
    rows.sort(key=lambda r: -abs(float(r["error_m"])))
    return rows[:n]


def training_mean(blasts: Iterable[Blast]) -> float:
    """The mean measured size, which is what the null model predicts."""
    values = [b.x50_m for b in blasts if b.x50_m is not None]
    if not values:
        raise ValueError("no measured sizes to average")
    return sum(values) / len(values)
