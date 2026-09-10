"""The protocol-sensitivity benchmark: the same arms, the same rows, three ways of splitting.

This is the module the whole package exists to support. The 2025 state of the art on this corpus
reports its headline from a random 80/20 split of 97 rows, 17 of which duplicate another row's
feature vector. Nothing in the literature reports what happens under a split that does not leak.

Running all three protocols and publishing the gap is the contribution. **Either sign of the gap is
a result**, and the kill criterion is declared before the run: if leave-one-site-out does not put the
best learned arm meaningfully above a constant predictor, that is what gets reported, in those words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .metrics import Score, score, training_mean
from .models import Arm, NullModel
from .splits import Split, all_protocols
from .types import Blast, Prediction

__all__ = [
    "ArmResult",
    "ProtocolResult",
    "BenchmarkResult",
    "run_benchmark",
    "run_fixed_holdout",
    "KILL_CRITERION",
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
constant fitted to a different mix of sites. A gate that can pass on two failures is not measuring
its own subject.
"""


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's score under one protocol, with the abstentions it declared."""

    arm: str
    tier: str
    protocol: str
    score: Score
    n_folds: int = 1
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProtocolResult:
    """Every arm under one protocol."""

    protocol: str
    n_folds: int
    arms: tuple[ArmResult, ...]

    def by_arm(self) -> dict[str, ArmResult]:
        return {r.arm: r for r in self.arms}

    def best(self, *, tier: str | None = None) -> ArmResult | None:
        candidates = [
            r
            for r in self.arms
            if r.score.r2_identity is not None
            and r.tier not in {"control"}
            and (tier is None or r.tier == tier)
        ]
        return max(candidates, key=lambda r: r.score.r2_identity) if candidates else None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The whole sweep: every arm, every protocol, plus the verdict on the declared criterion."""

    protocols: tuple[ProtocolResult, ...]
    verdict: dict[str, object]
    dataset_digest: str

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
                        "n_scored": result.score.n_scored,
                        "n_abstained": result.score.n_abstained,
                        "r2_identity": result.score.r2_identity,
                        "pearson_r2": result.score.pearson_r2,
                        "rmse_m": result.score.rmse_m,
                        "mae_m": result.score.mae_m,
                        "mape_pct": result.score.mape_pct,
                        "bias_m": result.score.bias_m,
                    }
                )
        return rows


def _pooled(
    arm_factory: Callable[[], Arm], splits: Sequence[Split], protocol: str
) -> tuple[Score, int]:
    """Fit and predict over every fold, then score the pooled out-of-fold predictions.

    Pooling rather than averaging per-fold scores is deliberate. A leave-one-site-out sweep has folds
    from 6 to 22 rows, and averaging their scores would weight a six-row site the same as a
    twenty-two-row one. Pooling scores every blast exactly once, on the fold where it was held out.
    """
    tested: list[Blast] = []
    predictions: list[Prediction] = []
    for split in splits:
        arm = arm_factory()
        try:
            arm.fit(split.train)
        except ValueError as exc:
            # An arm that cannot be fitted on this fold abstains for the fold, with the reason.
            for blast in split.test:
                tested.append(blast)
                predictions.append(
                    Prediction(
                        method=arm.name,
                        blast_id=blast.blast_id,
                        x50_m=None,
                        abstain_reason=f"could not fit on the {split.protocol} training rows: {exc}",
                    )
                )
            continue
        tested.extend(split.test)
        predictions.extend(arm.predict(split.test))
    return score(tested, predictions, method=arm_factory().name), len(splits)


def run_benchmark(
    blasts: Sequence[Blast],
    arms: dict[str, Callable[[], Arm]],
    *,
    seed: int = 0,
) -> BenchmarkResult:
    """Score every arm under every protocol on the same corpus.

    ``arms`` maps a name to a **factory**, not an instance, because each fold needs a freshly
    initialised model. Reusing one instance across folds would let a fold's fit leak into the next.
    """
    from .datasets import compute_dataset_digest

    protocols = all_protocols(blasts, seed=seed)
    results: list[ProtocolResult] = []

    for protocol_name, splits in protocols.items():
        arm_results: list[ArmResult] = []
        for name, factory in arms.items():
            pooled, n_folds = _pooled(factory, splits, protocol_name)
            arm_results.append(
                ArmResult(
                    arm=name,
                    tier=factory().tier,
                    protocol=protocol_name,
                    score=pooled,
                    n_folds=n_folds,
                    detail={"seed": seed},
                )
            )
        results.append(
            ProtocolResult(
                protocol=protocol_name, n_folds=len(splits), arms=tuple(arm_results)
            )
        )

    return BenchmarkResult(
        protocols=tuple(results),
        verdict=_verdict(results),
        dataset_digest=compute_dataset_digest(blasts),
    )


def _verdict(results: Sequence[ProtocolResult]) -> dict[str, object]:
    """Apply the declared kill criterion, and measure the protocol gap."""
    by_protocol = {p.protocol: p for p in results}
    grouped = by_protocol.get("leave-one-site-out")
    random_draw = by_protocol.get("random-8020")

    verdict: dict[str, object] = {"criterion": KILL_CRITERION}
    if grouped is None:
        verdict["outcome"] = "not evaluated: no leave-one-site-out protocol was run"
        return verdict

    learned = [
        r for r in grouped.arms if r.tier == "learned" and r.score.r2_identity is not None
    ]
    null = next((r for r in grouped.arms if r.arm == "null"), None)
    if not learned or null is None or null.score.r2_identity is None:
        verdict["outcome"] = "not evaluated: the learned tier or the null model did not score"
        return verdict

    best = max(learned, key=lambda r: r.score.r2_identity)
    margin = best.score.r2_identity - null.score.r2_identity
    positive = best.score.r2_identity > 0.0
    generalises = positive and margin >= 0.10

    if generalises:
        outcome = (
            f"the learned tier generalises across sites: {best.arm} explains "
            f"{best.score.r2_identity:.3f} of the variance and exceeds a constant predictor by "
            f"{margin:.3f}"
        )
    elif not positive:
        outcome = (
            "THE LEARNED TIER DOES NOT GENERALISE ACROSS SITES. Every learned arm has NEGATIVE "
            f"variance explained under leave-one-site-out; the best of them, {best.arm}, scores "
            f"{best.score.r2_identity:.3f}, which is worse than predicting a constant. Its "
            f"{margin:.3f} margin over the null is two models failing by different amounts, not "
            "skill."
        )
    else:
        outcome = (
            "THE LEARNED TIER DOES NOT GENERALISE ACROSS SITES: the best learned arm, "
            f"{best.arm}, exceeds a constant predictor by only {margin:.3f} in variance explained "
            "under leave-one-site-out"
        )

    verdict.update(
        {
            "best_learned_arm": best.arm,
            "best_learned_r2_identity": best.score.r2_identity,
            "null_r2_identity": null.score.r2_identity,
            "margin_over_null": margin,
            "best_learned_is_positive": positive,
            "n_learned_arms_positive": sum(1 for r in learned if r.score.r2_identity > 0.0),
            "n_learned_arms": len(learned),
            "generalises_across_sites": generalises,
            "outcome": outcome,
        }
    )

    # Which arms DO transfer, if any. On this corpus the answer is the two arms whose coefficients
    # are fixed rather than fitted, which is the finding.
    transferring = sorted(
        (
            (r.arm, r.score.r2_identity)
            for r in grouped.arms
            if r.tier not in {"control"} and r.score.r2_identity is not None and r.score.r2_identity > 0
        ),
        key=lambda pair: -pair[1],
    )
    verdict["arms_with_positive_variance_explained_across_sites"] = transferring

    if random_draw is not None:
        random_arms = random_draw.by_arm()
        gaps = {
            r.arm: (random_arms[r.arm].score.r2_identity - r.score.r2_identity)
            for r in grouped.arms
            if r.arm in random_arms
            and r.score.r2_identity is not None
            and random_arms[r.arm].score.r2_identity is not None
        }
        verdict["protocol_gap_random_minus_grouped"] = gaps
        if gaps:
            verdict["median_protocol_gap"] = sorted(gaps.values())[len(gaps) // 2]
    return verdict


def run_fixed_holdout(
    train: Sequence[Blast],
    holdout: Sequence[Blast],
    arms: dict[str, Callable[[], Arm]],
) -> list[ArmResult]:
    """Score every arm on a fixed published hold-out, which is a reproduction rather than a split.

    Used for the two published validation sets. The null model is fitted on the training rows, as it
    must be: a null fitted on the hold-out would be using the answer.
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


def default_arms(*, include_learned: bool = True) -> dict[str, Callable[[], Arm]]:
    """The ladder as the benchmark runs it, controls first."""
    from .models import (
        GroupDiscriminant,
        Kuznetsov,
        Oracle,
        PublishedRegression,
        RefittedRegression,
    )

    arms: dict[str, Callable[[], Arm]] = {
        "null": NullModel,
        "oracle": Oracle,
        "kuznetsov": Kuznetsov,
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
