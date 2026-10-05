"""The protocol-sensitivity benchmark: the same arms, the same rows, three ways of splitting.

This is the module the whole package exists to support. The 2025 state of the art on this corpus
reports its headline from a random 80/20 split of 97 rows, 17 of which duplicate another row's
feature vector. Nothing in the literature reports what happens under a split that holds out whole
sites.

Running all three protocols and publishing the gap is the contribution. **Either sign of the gap is
a result**, and the kill criterion is declared before the run: if leave-one-site-out does not put the
best learned arm meaningfully above a constant predictor, that is what gets reported, in those words.

What 0.3.0 changed, and why. An adversarial re-run of 0.2.2 found four ways in which the reported
effects depended on the evaluation design rather than on the models, and each now has a mechanism
here:

1. The random protocols were one draw of about 19 test rows. They are now repeated (100 draws by
   default) and every arm carries its spread; the protocol gap is computed from the median.
2. Under leave-one-site-out the classical arms abstain on the one site with no hole diameter, so they
   were scored on fewer rows than the fitted arms. Every grouped score is now reported on two
   supports: every blast, and the blasts with resolvable geometry. The criterion is evaluated on both,
   and a verdict that differs between them is reported as depending on the row set.
3. No headline figure carried an interval. Grouped scores now carry a site-resampled 95 percent
   interval, because the site, not the blast, is the independent unit.
4. Arms whose coefficients were fitted on the corpus itself were listed among the arms that transfer.
   Every arm now declares what it was fitted on (:meth:`Arm.provenance`) and in-sample arms are
   reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .geometry import has_absolute_geometry
from .metrics import Score, bootstrap_interval, score, summarise_draws, training_mean
from .models import Arm, NullModel
from .splits import REPEATED_PROTOCOLS, Split, all_protocols
from .types import Blast, Prediction

__all__ = [
    "ArmResult",
    "ProtocolResult",
    "BenchmarkResult",
    "run_benchmark",
    "run_fixed_holdout",
    "default_arms",
    "KILL_CRITERION",
    "SUPPORTS",
    "PUBLISHED_RANDOM_SPLIT_R2",
    "PUBLISHED_NETWORK_WIDTHS",
    "network_width_sweep",
]

KILL_CRITERION = (
    "The learned tier counts as generalising across sites only if the best learned arm's variance "
    "explained under leave-one-site-out is BOTH positive and at least 0.10 above the null model's. "
    "Both halves are required: a margin over a null that is itself deeply negative is not skill, it "
    "is two models failing by different amounts."
)
"""Declared before the run, so the answer counts either way.

The second half of this criterion was added after the first run, and the reason is worth recording
because it is the failure mode the criterion exists to prevent. As first written the rule asked only
for a margin over the null. The run then produced a best learned arm at -0.034 against a null at
-0.216, and the rule declared that the learned tier generalises. It does not: an arm with negative
variance explained is worse than predicting a constant, and it had simply failed less badly than a
constant fitted to a different mix of sites.

The text has not changed since. 0.3.0 changes what is reported BESIDE it (two supports, intervals),
not the rule, because editing a declared criterion after seeing the data is the failure it guards.
"""

SUPPORTS: dict[str, Callable[[Blast], bool]] = {
    "all": lambda blast: True,
    "geometry": has_absolute_geometry,
}
"""The two row sets every grouped score is reported on.

``all`` is every blast. ``geometry`` is the blasts whose pattern geometry is resolvable, which are the
only rows on which the classical arms can answer; on this corpus that removes the six Miami blasts.
Comparing a classical score over 91 rows with a fitted score over 97 is comparing different
denominators, and on this corpus that difference decides the verdict.
"""

PUBLISHED_RANDOM_SPLIT_R2: dict[str, float] = {
    "stacking": 0.943,
    "svr-poly": 0.578,
}
"""Test-set figures Sui et al. 2025 (doi:10.3390/app15031254) report from one random 80/20 split of
these 97 blasts, for the two arms whose parameters the source states without ambiguity. Each is
placed within this package's own distribution of draws of the same protocol, so a reader can see
whether it is a typical draw or a favourable one.

The source also reports 0.797 for its random forest and 0.758 for XGBoost, but it prints two
parameter sets for those learners and does not say unambiguously which produced the figures, so
placing them in a distribution of draws made with one of the sets would compare unlike things."""


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's score under one protocol, with the abstentions it declared.

    For a repeated protocol ``score`` is the first draw (the seed the caller passed, which reproduces
    a single published split) and ``detail["repeats"]`` is the spread over every draw. For
    leave-one-site-out ``score`` is the pooled score over every blast and ``detail["supports"]``,
    ``detail["per_site"]`` and ``detail["predictions"]`` carry the rest.
    """

    arm: str
    tier: str
    protocol: str
    score: Score
    n_folds: int = 1
    detail: dict = field(default_factory=dict)

    def headline_r2(self) -> float | None:
        """The figure a reader should quote: the median draw for a repeated protocol, else the score."""
        repeats = self.detail.get("repeats")
        if repeats and repeats.get("n"):
            return float(repeats["median"])
        return self.score.r2_identity


@dataclass(frozen=True, slots=True)
class ProtocolResult:
    """Every arm under one protocol."""

    protocol: str
    n_folds: int
    arms: tuple[ArmResult, ...]
    repeated: bool = False

    def by_arm(self) -> dict[str, ArmResult]:
        return {r.arm: r for r in self.arms}

    def best(self, *, tier: str | None = None) -> ArmResult | None:
        candidates = [
            r
            for r in self.arms
            if r.headline_r2() is not None
            and r.tier not in {"control"}
            and (tier is None or r.tier == tier)
        ]
        return max(candidates, key=lambda r: r.headline_r2()) if candidates else None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The whole sweep: every arm, every protocol, plus the verdict on the declared criterion."""

    protocols: tuple[ProtocolResult, ...]
    verdict: dict[str, object]
    dataset_digest: str
    provenance: dict[str, dict] = field(default_factory=dict)
    n_repeats: int = 1

    def by_protocol(self) -> dict[str, ProtocolResult]:
        return {p.protocol: p for p in self.protocols}

    def table(self) -> list[dict[str, object]]:
        """A flat table, one row per arm per protocol, for export or for a screen."""
        rows: list[dict[str, object]] = []
        for protocol in self.protocols:
            for result in protocol.arms:
                rows.append(
                    {
                        "arm": result.arm,
                        "tier": result.tier,
                        "protocol": result.protocol,
                        "n_folds": result.n_folds,
                        "headline_r2_identity": result.headline_r2(),
                        **result.score.as_dict(),
                    }
                )
        return rows


# ---------------------------------------------------------------------------------------------
# Fitting and predicting one split
# ---------------------------------------------------------------------------------------------

def _fit_predict(
    factory: Callable[[], Arm], split: Split
) -> tuple[list[Blast], list[Prediction]]:
    """Fit a fresh arm on a split's training rows and predict its test rows.

    An arm that cannot be fitted on the split abstains on every test row with the reason, so a hole
    in a table is always explained.
    """
    arm = factory()
    try:
        arm.fit(split.train)
    except ValueError as exc:
        return list(split.test), [
            Prediction(
                method=arm.name,
                blast_id=blast.blast_id,
                x50_m=None,
                abstain_reason=f"could not fit on the {split.protocol} training rows: {exc}",
            )
            for blast in split.test
        ]
    return list(split.test), arm.predict(split.test)


def _score_repeated(
    name: str, factory: Callable[[], Arm], splits: Sequence[Split], protocol: str
) -> ArmResult:
    """Score every draw on its own and report the spread; the first draw is the headline score."""
    tier = factory().tier
    first: Score | None = None
    r2: list[float] = []
    rmse: list[float] = []
    kept: list[tuple[list[Blast], list[Prediction]]] = []
    for split in splits:
        tested, predictions = _fit_predict(factory, split)
        kept.append((tested, predictions))
        result = score(tested, predictions, method=name)
        if first is None:
            first = result
        if result.r2_identity is not None:
            r2.append(result.r2_identity)
        if result.rmse_m is not None:
            rmse.append(result.rmse_m)
    assert first is not None
    return ArmResult(
        arm=name,
        tier=tier,
        protocol=protocol,
        score=first,
        n_folds=len(splits),
        detail={
            "seeds": [s.seed for s in splits],
            "repeats": summarise_draws(r2),
            "rmse_repeats": summarise_draws(rmse),
            "draws_r2_identity": r2,
            _KEPT: kept,
        },
    )


def _per_site(tested: Sequence[Blast], predictions: Sequence[Prediction]) -> dict[str, dict]:
    """Error on each held-out site, which is where a transfer failure has a name."""
    by_id = {p.blast_id: p for p in predictions}
    out: dict[str, dict] = {}
    for site in sorted({b.site for b in tested}):
        rows = [b for b in tested if b.site == site and b.x50_m is not None]
        pairs = [
            (b.x50_m, by_id[b.blast_id].x50_m)
            for b in rows
            if by_id[b.blast_id].x50_m is not None
        ]
        entry: dict[str, object] = {
            "n_blasts": len(rows),
            "n_scored": len(pairs),
            "mean_measured_m": sum(b.x50_m for b in rows) / len(rows) if rows else None,
        }
        if pairs:
            errors = [p - m for m, p in pairs]  # type: ignore[operator]
            entry.update(
                {
                    "mean_predicted_m": sum(p for _, p in pairs) / len(pairs),  # type: ignore[misc]
                    "rmse_m": (sum(e * e for e in errors) / len(errors)) ** 0.5,
                    "mae_m": sum(abs(e) for e in errors) / len(errors),
                    "bias_m": sum(errors) / len(errors),
                }
            )
        else:
            entry["abstain_reason"] = next(
                (by_id[b.blast_id].abstain_reason for b in rows if by_id[b.blast_id].abstained),
                None,
            )
        out[site] = entry
    return out


def _score_grouped(
    name: str,
    factory: Callable[[], Arm],
    splits: Sequence[Split],
    protocol: str,
    *,
    n_boot: int,
    boot_seed: int,
) -> ArmResult:
    """Pool the out-of-fold predictions, then score them on each support with a site interval.

    Pooling rather than averaging per-fold scores is deliberate. A leave-one-site-out sweep has folds
    from 6 to 22 rows, and averaging their scores would weight a six-row site the same as a
    twenty-two-row one. Pooling scores every blast exactly once, on the fold where it was held out.
    """
    tier = factory().tier
    tested: list[Blast] = []
    predictions: list[Prediction] = []
    for split in splits:
        t, p = _fit_predict(factory, split)
        tested.extend(t)
        predictions.extend(p)

    by_id = {p.blast_id: p for p in predictions}
    supports: dict[str, dict] = {}
    headline: Score | None = None
    for support, keep in SUPPORTS.items():
        rows = [b for b in tested if keep(b)]
        preds = [by_id[b.blast_id] for b in rows]
        result = score(rows, preds, method=name)
        if support == "all":
            headline = result
        interval: list[float] | None = None
        if n_boot > 0 and result.r2_identity is not None:
            try:
                _point, low, high = bootstrap_interval(
                    rows, preds, unit="site", n_boot=n_boot, seed=boot_seed
                )
                interval = [low, high]
            except ValueError:
                interval = None
        supports[support] = {
            "score": result.as_dict(),
            "interval_95": interval,
            "n_sites": len({b.site for b in rows if not by_id[b.blast_id].abstained}),
        }
    assert headline is not None
    return ArmResult(
        arm=name,
        tier=tier,
        protocol=protocol,
        score=headline,
        n_folds=len(splits),
        detail={
            "supports": supports,
            "per_site": _per_site(tested, predictions),
            "predictions": {
                p.blast_id: p.x50_m for p in predictions
            },
            _KEPT: (tested, predictions),
        },
    )


_KEPT = "_kept_predictions"
"""Where a scorer leaves its predictions for the common-support pass, which removes them."""


def _answered(predictions: Sequence[Prediction]) -> set[str]:
    return {p.blast_id for p in predictions if not p.abstained}


def _on_rows(
    tested: Sequence[Blast], predictions: Sequence[Prediction], keep: set[str]
) -> tuple[list[Blast], list[Prediction]]:
    by_id = {p.blast_id: p for p in predictions}
    rows = [b for b in tested if b.blast_id in keep]
    return rows, [by_id[b.blast_id] for b in rows]


def _add_common_support(
    results: Sequence[ArmResult], *, repeated: bool, n_boot: int, boot_seed: int
) -> None:
    """Score every arm again on the rows that every size-predicting arm answered.

    ``score`` drops each arm's abstentions, so two arms in one table are otherwise scored on different
    rows. The size-predicting arms are the ones that answer at least one row in the protocol; the
    router predicts a group, answers none, and is left out so it cannot empty the intersection. The
    declared ``SUPPORTS`` and the verdict are not touched: this is a row set reported beside them, not
    one the criterion is re-evaluated on after the run.
    """
    if not results:
        return
    if repeated:
        draws = [r.detail[_KEPT] for r in results]
        size_arms = [
            i for i, kept in enumerate(draws) if any(_answered(p) for _t, p in kept)
        ]
        names = sorted(results[i].arm for i in size_arms)
        n_draws = len(draws[0])
        common_ids = [
            set.intersection(*(_answered(draws[i][d][1]) for i in size_arms)) if size_arms else set()
            for d in range(n_draws)
        ]
        for result, kept in zip(results, draws):
            per_draw: list[float | None] = []
            rmse: list[float] = []
            for (tested, predictions), keep in zip(kept, common_ids):
                rows, preds = _on_rows(tested, predictions, keep)
                if len(rows) < 2:
                    per_draw.append(None)
                    continue
                scored = score(rows, preds, method=result.arm)
                per_draw.append(scored.r2_identity)
                if scored.rmse_m is not None:
                    rmse.append(scored.rmse_m)
            result.detail["common"] = {
                "repeats": summarise_draws([v for v in per_draw if v is not None]),
                "rmse_repeats": summarise_draws(rmse),
                "n_rows": summarise_draws([float(len(k)) for k in common_ids]),
                "draws_r2_identity": per_draw,
                "arms": names,
            }
    else:
        pooled = [r.detail[_KEPT] for r in results]
        size_arms = [i for i, (_t, p) in enumerate(pooled) if _answered(p)]
        names = sorted(results[i].arm for i in size_arms)
        keep = set.intersection(*(_answered(pooled[i][1]) for i in size_arms)) if size_arms else set()
        for result, (tested, predictions) in zip(results, pooled):
            rows, preds = _on_rows(tested, predictions, keep)
            scored = score(rows, preds, method=result.arm)
            interval: list[float] | None = None
            if n_boot > 0 and scored.r2_identity is not None:
                try:
                    _point, low, high = bootstrap_interval(
                        rows, preds, unit="site", n_boot=n_boot, seed=boot_seed
                    )
                    interval = [low, high]
                except ValueError:
                    interval = None
            result.detail["common"] = {
                "score": scored.as_dict(),
                "interval_95": interval,
                "n_rows": len(rows),
                "n_sites": len({b.site for b in rows}),
                "arms": names,
            }
    for result in results:
        result.detail.pop(_KEPT, None)


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------

def run_benchmark(
    blasts: Sequence[Blast],
    arms: dict[str, Callable[[], Arm]],
    *,
    seed: int = 0,
    n_repeats: int = 100,
    n_boot: int = 2000,
    boot_seed: int = 0,
) -> BenchmarkResult:
    """Score every arm under every protocol on the same corpus.

    ``arms`` maps a name to a **factory**, not an instance, because each fold needs a freshly
    initialised model. Reusing one instance across folds would let a fold's fit leak into the next.

    ``n_repeats`` draws of each random protocol are scored one by one, seeds ``seed`` onward;
    ``n_boot`` site resamples give the grouped intervals (0 skips them). With BLAS pinned to one thread
    the default run takes about three minutes on the default device, almost all of it the published
    network's Levenberg-Marquardt training; multi-threaded BLAS on a loaded machine can make it many
    times slower, because the network's matrices are too small to share.

    Every arm also carries ``detail["common"]``: its score on the rows every size-predicting arm
    answered, which is reported beside the declared supports and never decides the verdict.
    """
    from .datasets import compute_dataset_digest

    protocols = all_protocols(blasts, seed=seed, n_repeats=n_repeats)
    results: list[ProtocolResult] = []
    for protocol_name, splits in protocols.items():
        repeated = protocol_name in REPEATED_PROTOCOLS
        arm_results = [
            _score_repeated(name, factory, splits, protocol_name)
            if repeated
            else _score_grouped(
                name, factory, splits, protocol_name, n_boot=n_boot, boot_seed=boot_seed
            )
            for name, factory in arms.items()
        ]
        _add_common_support(arm_results, repeated=repeated, n_boot=n_boot, boot_seed=boot_seed)
        results.append(
            ProtocolResult(
                protocol=protocol_name,
                n_folds=len(splits),
                arms=tuple(arm_results),
                repeated=repeated,
            )
        )

    provenance: dict[str, dict] = {}
    for name, factory in arms.items():
        arm = factory()
        provenance[name] = {
            "tier": arm.tier,
            "lane": arm.lane,
            "source": arm.source,
            "shares_mean_size_with": arm.shares_mean_size_with,
            **arm.provenance(),
        }

    return BenchmarkResult(
        protocols=tuple(results),
        verdict=_verdict(results, provenance, blasts),
        dataset_digest=compute_dataset_digest(blasts),
        provenance=provenance,
        n_repeats=n_repeats,
    )


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------

def _criterion_on(grouped: ProtocolResult, support: str) -> dict[str, object] | None:
    """Apply the declared kill criterion to one support of the grouped protocol."""

    def r2(result: ArmResult) -> float | None:
        return result.detail["supports"][support]["score"]["r2_identity"]

    learned = [r for r in grouped.arms if r.tier == "learned" and r2(r) is not None]
    null = next((r for r in grouped.arms if r.arm == "null"), None)
    if not learned or null is None or r2(null) is None:
        return None
    best = max(learned, key=lambda r: r2(r))  # type: ignore[arg-type, return-value]
    best_r2 = float(r2(best))  # type: ignore[arg-type]
    null_r2 = float(r2(null))  # type: ignore[arg-type]
    margin = best_r2 - null_r2
    positive = best_r2 > 0.0
    return {
        "n_blasts": best.detail["supports"][support]["score"]["n_scored"]
        + best.detail["supports"][support]["score"]["n_abstained"],
        "best_learned_arm": best.arm,
        "best_learned_r2_identity": best_r2,
        "best_learned_interval_95": best.detail["supports"][support]["interval_95"],
        "null_r2_identity": null_r2,
        "null_pearson_r": null.detail["supports"][support]["score"]["pearson_r"],
        "margin_over_null": margin,
        "best_learned_is_positive": positive,
        "n_learned_arms_positive": sum(1 for r in learned if (r2(r) or 0.0) > 0.0),
        "n_learned_arms": len(learned),
        "generalises_across_sites": positive and margin >= 0.10,
    }


def _verdict(
    results: Sequence[ProtocolResult],
    provenance: dict[str, dict],
    blasts: Sequence[Blast],
) -> dict[str, object]:
    """Apply the declared kill criterion on both supports, and measure the protocol gap."""
    by_protocol = {p.protocol: p for p in results}
    grouped = by_protocol.get("leave-one-site-out")
    verdict: dict[str, object] = {"criterion": KILL_CRITERION}
    if grouped is None:
        verdict["outcome"] = "not evaluated: no leave-one-site-out protocol was run"
        return verdict

    on = {support: _criterion_on(grouped, support) for support in SUPPORTS}
    if on["all"] is None:
        verdict["outcome"] = "not evaluated: the learned tier or the null model did not score"
        return verdict
    verdict["supports"] = on
    # The top-level fields are the "all" support: that is the row set the criterion was first applied
    # to, and keeping it there keeps the record continuous across releases.
    verdict.update({k: v for k, v in on["all"].items() if k != "n_blasts"})

    depends = on["geometry"] is not None and (
        on["geometry"]["generalises_across_sites"] != on["all"]["generalises_across_sites"]
    )
    verdict["depends_on_support"] = depends
    verdict["sites_outside_geometry_support"] = sorted(
        {b.site for b in blasts if not SUPPORTS["geometry"](b)}
    )

    # Which arms, other than in-sample ones, keep a positive score on an unseen site; which arms are
    # in sample and why; and which arms have an interval that clears zero at all.
    arms = grouped.by_arm()
    in_sample = sorted(n for n, p in provenance.items() if p.get("in_sample_corpus"))
    site_constant = sorted(n for n, p in provenance.items() if p.get("uses_site_constant"))
    verdict["in_sample_arms"] = [
        [name, arms[name].score.r2_identity, provenance[name]["fitted_on"]]
        for name in in_sample
        if name in arms and arms[name].score.r2_identity is not None
    ]
    verdict["site_constant_arms"] = site_constant
    verdict["arms_with_positive_variance_explained_across_sites"] = sorted(
        (
            [r.arm, r.score.r2_identity]
            for r in grouped.arms
            if r.tier != "control"
            and r.arm not in in_sample
            and r.score.r2_identity is not None
            and r.score.r2_identity > 0
        ),
        key=lambda pair: -pair[1],
    )
    intervals = {
        r.arm: {s: r.detail["supports"][s]["interval_95"] for s in SUPPORTS}
        for r in grouped.arms
        if r.tier != "control"
    }
    verdict["intervals_95"] = intervals
    verdict["arms_with_interval_above_zero"] = sorted(
        arm
        for arm, by_support in intervals.items()
        if arm not in in_sample
        and any(iv is not None and iv[0] > 0 for iv in by_support.values())
    )

    # The protocol gap, from the median random draw, so one lucky or unlucky split cannot set it.
    random_draw = by_protocol.get("random-8020")
    dedup_draw = by_protocol.get("dedup-random")
    if random_draw is not None:
        random_arms = random_draw.by_arm()
        gaps = {
            r.arm: random_arms[r.arm].headline_r2() - r.score.r2_identity
            for r in grouped.arms
            if r.arm in random_arms
            and r.score.r2_identity is not None
            and random_arms[r.arm].headline_r2() is not None
        }
        verdict["protocol_gap_random_minus_grouped"] = gaps
        verdict["protocol_gap_basis"] = (
            f"median of {random_draw.n_folds} random 80/20 draws minus the pooled "
            "leave-one-site-out score over every blast"
        )
        learned_gaps = sorted(
            gap for arm, gap in gaps.items() if arms[arm].tier == "learned"
        )
        if learned_gaps:
            mid = len(learned_gaps) // 2
            verdict["median_protocol_gap"] = (
                learned_gaps[mid]
                if len(learned_gaps) % 2
                else 0.5 * (learned_gaps[mid - 1] + learned_gaps[mid])
            )
            verdict["median_protocol_gap_over"] = "the learned arms"
        if dedup_draw is not None:
            dedup_arms = dedup_draw.by_arm()
            verdict["dedup_minus_random_median"] = {
                arm: dedup_arms[arm].headline_r2() - random_arms[arm].headline_r2()
                for arm in random_arms
                if arm in dedup_arms
                and arms.get(arm) is not None
                and arms[arm].tier != "control"
                and dedup_arms[arm].headline_r2() is not None
                and random_arms[arm].headline_r2() is not None
            }
        published: dict[str, dict] = {}
        for arm, value in PUBLISHED_RANDOM_SPLIT_R2.items():
            draws = random_arms.get(arm)
            values = draws.detail.get("draws_r2_identity") if draws is not None else None
            if values:
                published[arm] = {
                    "published": value,
                    "share_of_draws_below": sum(1 for v in values if v < value) / len(values),
                    "median_draw": draws.headline_r2(),
                }
        verdict["published_random_split_figures"] = published

    verdict["outcome"] = _outcome(on, depends, verdict)
    return verdict


def _outcome(on: dict, depends: bool, verdict: dict) -> str:
    """One canonical English sentence group, composed from the structured fields."""
    all_ = on["all"]
    geo = on["geometry"]
    parts: list[str] = []
    if depends and geo is not None:
        parts.append(
            "THE VERDICT DEPENDS ON THE ROW SET. "
            f"Over all {all_['n_blasts']} blasts the best learned arm under leave-one-site-out, "
            f"{all_['best_learned_arm']}, scores {all_['best_learned_r2_identity']:.3f} and the "
            "learned tier "
            + ("meets" if all_["generalises_across_sites"] else "does not meet")
            + f" the criterion. Over the {geo['n_blasts']} blasts whose pattern geometry is "
            "resolvable, the rows on which the classical arms are also scored, "
            f"{geo['best_learned_arm']} scores {geo['best_learned_r2_identity']:.3f}, "
            f"{geo['margin_over_null']:.3f} above the null, and the tier "
            + ("meets" if geo["generalises_across_sites"] else "does not meet")
            + " it. The blasts that separate the two row sets come from "
            + ", ".join(verdict["sites_outside_geometry_support"])
            + "."
        )
    elif all_["generalises_across_sites"]:
        parts.append(
            f"The learned tier generalises across sites: {all_['best_learned_arm']} explains "
            f"{all_['best_learned_r2_identity']:.3f} of the variance and exceeds a constant "
            f"predictor by {all_['margin_over_null']:.3f}."
        )
    elif not all_["best_learned_is_positive"]:
        parts.append(
            "THE LEARNED TIER DOES NOT GENERALISE ACROSS SITES. Every learned arm has NEGATIVE "
            f"variance explained under leave-one-site-out; the best of them, "
            f"{all_['best_learned_arm']}, scores {all_['best_learned_r2_identity']:.3f}, which is "
            f"worse than predicting a constant. Its {all_['margin_over_null']:.3f} margin over the "
            "null is two models failing by different amounts, not skill."
        )
    else:
        parts.append(
            "THE LEARNED TIER DOES NOT GENERALISE ACROSS SITES: the best learned arm, "
            f"{all_['best_learned_arm']}, exceeds a constant predictor by only "
            f"{all_['margin_over_null']:.3f} in variance explained under leave-one-site-out."
        )

    if not verdict.get("arms_with_interval_above_zero"):
        parts.append(
            "Apart from arms whose source fitted them on this corpus, no arm has a site-resampled 95 "
            "percent interval above zero: with ten sites, none of them is distinguishable from "
            "predicting the corpus mean."
        )
    if all_.get("null_pearson_r") is not None and all_["null_pearson_r"] < 0:
        parts.append(
            f"The null's held-out predictions correlate with the measurements at "
            f"{all_['null_pearson_r']:.2f}: holding out a coarse site lowers the training mean, so "
            "any margin over the null under this protocol is larger than the skill it measures."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------
# Fixed published hold-outs
# ---------------------------------------------------------------------------------------------

def run_fixed_holdout(
    train: Sequence[Blast],
    holdout: Sequence[Blast],
    arms: dict[str, Callable[[], Arm]],
) -> list[ArmResult]:
    """Score every arm on a fixed published hold-out, which is a reproduction rather than a split.

    Used for the two published validation sets. The null model is fitted on the training rows, as it
    must be: a null fitted on the hold-out would be using the answer.

    Two cautions travel with the result. The published regression and the router were fitted by
    their source on ``train`` when ``train`` is the 97-blast corpus, so on these hold-outs they are
    genuinely out of sample. The classical arm's rock factors were back-solved from the published
    classical predictions for these same hold-out blasts, so its agreement with that published
    column is circular here; only the within-site constancy of the factors is evidence.
    """
    results: list[ArmResult] = []
    for name, factory in arms.items():
        arm = factory()
        try:
            arm.fit(train)
        except ValueError as exc:
            results.append(
                ArmResult(
                    arm=name,
                    tier=arm.tier,
                    protocol="fixed-holdout",
                    score=score(
                        holdout,
                        [
                            Prediction(
                                method=name,
                                blast_id=b.blast_id,
                                x50_m=None,
                                abstain_reason=str(exc),
                            )
                            for b in holdout
                        ],
                        method=name,
                    ),
                )
            )
            continue
        results.append(
            ArmResult(
                arm=name,
                tier=arm.tier,
                protocol="fixed-holdout",
                score=score(holdout, arm.predict(holdout), method=name),
                detail={"training_mean_m": training_mean(train)},
            )
        )
    return results


PUBLISHED_NETWORK_WIDTHS: dict[int, int] = {1: 9, 2: 7}
"""The hidden widths Kulatilake, Hudaverdi and Wu 2012 chose per rock-stiffness group on their hold-out."""


def network_width_sweep(
    corpus: Sequence[Blast],
    holdout: Sequence[Blast],
    *,
    widths: Sequence[int] = tuple(range(6, 16)),
    n_simulations: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    """The published network's hidden width, swept under the source's protocol and held out by site.

    The source swept the width over 6 to 15, with eight simulations per width, and chose 9 for the
    high-modulus group and 7 for the low on its own hold-out, by RMSE and correlation. Two questions
    follow and this answers both:

    - ``published_protocol``: does the same procedure, reproduced here (train on the corpus, choose
      per group on the 2012 hold-out), land on the published widths? It is
      :meth:`PublishedNeuralNetwork.sweep_hidden_width`.
    - ``leave_one_site_out``: how much does the width matter at a mine the network has not seen? Each
      width (the same in both groups) is scored on the pooled out-of-fold predictions on both
      ``SUPPORTS``, beside the published pair, which is the network as the benchmark runs it.

    About two minutes for the default sweep with BLAS pinned to one thread; offline only.
    """
    from .learned import PublishedNeuralNetwork
    from .splits import leave_one_site_out

    widths = tuple(int(w) for w in widths)
    published_protocol = PublishedNeuralNetwork(
        n_simulations=n_simulations, seed=seed, sweep=widths
    ).sweep_hidden_width(corpus, holdout)

    splits = list(leave_one_site_out(corpus))
    configs: list[tuple[bool, dict[int, int]]] = [(True, dict(PUBLISHED_NETWORK_WIDTHS))]
    configs += [(False, {1: w, 2: w}) for w in widths]
    table: list[dict[str, object]] = []
    for published, hidden in configs:
        tested: list[Blast] = []
        predictions: list[Prediction] = []
        for split in splits:
            arm = PublishedNeuralNetwork(hidden=hidden, n_simulations=n_simulations, seed=seed)
            arm.fit(split.train)
            tested.extend(split.test)
            predictions.extend(arm.predict(split.test))
        by_id = {p.blast_id: p for p in predictions}
        supports: dict[str, dict] = {}
        for support, keep in SUPPORTS.items():
            rows = [b for b in tested if keep(b)]
            supports[support] = score(
                rows, [by_id[b.blast_id] for b in rows], method="published-neural-net"
            ).as_dict()
        table.append({"hidden": hidden, "published": published, "supports": supports})

    return {
        "widths": list(widths),
        "n_simulations": n_simulations,
        "seed": seed,
        "published_widths": dict(PUBLISHED_NETWORK_WIDTHS),
        "published_protocol": published_protocol,
        "leave_one_site_out": table,
    }


def default_arms(*, include_learned: bool = True) -> dict[str, Callable[[], Arm]]:
    """The ladder as the benchmark runs it, controls first.

    One predictor per distinct mean size: Kuz-Ram, Swebrec and the crush-zone composition return the
    classical mean size and differ only in curve shape, so they are not benchmarked separately. The
    in-situ cap does change the mean size where the classical prediction exceeds the block, so its arm
    on the classical equation is benchmarked, as a declared choice rather than a published relation.
    """
    from .models import (
        GroupDiscriminant,
        InSituCap,
        Kuznetsov,
        KuznetsovTransfer,
        Oracle,
        PublishedRegression,
        RefittedRegression,
    )

    arms: dict[str, Callable[[], Arm]] = {
        "null": NullModel,
        "oracle": Oracle,
        "kuznetsov": Kuznetsov,
        "kuznetsov-transfer": KuznetsovTransfer,
        "kuznetsov-capped": lambda: InSituCap(Kuznetsov()),
        "group-discriminant": GroupDiscriminant,
        "published-regression": PublishedRegression,
        "refitted-regression": RefittedRegression,
    }
    if include_learned:
        from .learned import (
            GradientBoosting,
            PublishedNeuralNetwork,
            RandomForest,
            StackingEnsemble,
            SupportVectorRegression,
        )

        arms.update(
            {
                "published-neural-net": PublishedNeuralNetwork,
                "svr-rbf": lambda: SupportVectorRegression(variant="rbf-amoako"),
                "svr-poly": lambda: SupportVectorRegression(variant="poly-sui"),
                "random-forest": RandomForest,
                "xgboost": GradientBoosting,
                "stacking": StackingEnsemble,
            }
        )
    return arms
