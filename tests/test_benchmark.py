"""The protocol-sensitivity benchmark, and the finding it produced.

Every number asserted here was measured by running the sweep. They are pinned because they are the
package's central claim and a silent change to any of them would change what the product says.
"""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.benchmark import KILL_CRITERION, default_arms, run_benchmark, run_fixed_holdout
from blastfrag.models import PLAUSIBLE_X50_M, RefittedRegression
from blastfrag.splits import leave_one_site_out

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def result():
    return run_benchmark(bf.load_training_corpus(), default_arms(), seed=0)


def test_the_kill_criterion_requires_a_positive_score_not_just_a_margin():
    """The criterion was wrong on its first writing, and the run caught it.

    As first stated it asked only for a margin over the null. The sweep then produced a best learned
    arm at -0.034 against a null at -0.216 and the rule declared success. An arm with negative
    variance explained is worse than a constant; it had simply failed less badly. Both halves are
    required now, and this test is what keeps the weaker version from coming back.
    """
    assert "positive" in KILL_CRITERION
    assert "0.10" in KILL_CRITERION


def test_the_learned_tier_does_not_generalise_across_sites(result):
    """The headline. Zero of six learned arms have positive variance explained on an unseen site."""
    verdict = result.verdict
    assert verdict["generalises_across_sites"] is False
    assert verdict["n_learned_arms_positive"] == 0
    assert verdict["n_learned_arms"] == 6
    assert verdict["best_learned_r2_identity"] < 0.0
    assert "DOES NOT GENERALISE" in verdict["outcome"]


def test_the_only_arms_that_transfer_are_the_ones_that_are_not_fitted(result):
    """The mechanism, and the reason the finding is interesting rather than merely negative.

    Two arms hold up on an unseen site and both have FIXED coefficients: the published regression,
    whose exponents are constants from a paper, and the classical equation, whose only free quantity
    is a per-site rock factor. Everything that fits itself to this corpus fails to leave it.
    """
    transferring = dict(result.verdict["arms_with_positive_variance_explained_across_sites"])
    assert set(transferring) == {"published-regression", "kuznetsov"}
    assert transferring["published-regression"] == pytest.approx(0.802, abs=0.02)
    assert transferring["kuznetsov"] == pytest.approx(0.311, abs=0.03)


def test_the_classical_arm_improves_when_held_out_by_site(result):
    """It has nothing to overfit, so a harder split does not hurt it. It helps.

    Under a random split the classical arm scores below zero; held out by site it reaches 0.311. The
    difference is not the model changing, it is the comparison changing: on a random split it is
    competing against arms that have memorised near-duplicates of the test rows.
    """
    by_protocol = result.by_protocol()
    random_draw = by_protocol["random-8020"].by_arm()["kuznetsov"].score.r2_identity
    grouped = by_protocol["leave-one-site-out"].by_arm()["kuznetsov"].score.r2_identity
    assert random_draw < 0.0 < grouped


def test_every_learned_arm_loses_ground_when_the_split_stops_leaking(result):
    """The protocol gap, per arm, in the direction the finding predicts."""
    gaps = result.verdict["protocol_gap_random_minus_grouped"]
    learned = [
        r.arm for r in result.by_protocol()["leave-one-site-out"].arms if r.tier == "learned"
    ]
    for arm in learned:
        assert gaps[arm] > 0.5, f"{arm} lost only {gaps[arm]:.3f} when the split stopped leaking"
    assert result.verdict["median_protocol_gap"] > 0.5


def test_deduplication_alone_does_not_explain_the_gap(result):
    """Worth separating, because it is the obvious first explanation and it is not the whole story.

    Collapsing duplicate feature vectors and splitting randomly gives HIGHER scores than the raw
    random split, not lower. So the duplicates are not what is propping the learned arms up; the
    shared site is.
    """
    by_protocol = result.by_protocol()
    random_arms = by_protocol["random-8020"].by_arm()
    dedup_arms = by_protocol["dedup-random"].by_arm()
    improved = [
        arm
        for arm, row in dedup_arms.items()
        if row.tier == "learned"
        and row.score.r2_identity is not None
        and random_arms[arm].score.r2_identity is not None
        and row.score.r2_identity > random_arms[arm].score.r2_identity
    ]
    assert len(improved) >= 5


def test_the_oracle_and_the_null_bracket_every_protocol(result):
    for protocol in result.protocols:
        arms = protocol.by_arm()
        assert arms["oracle"].score.r2_identity == pytest.approx(1.0)
        assert arms["null"].score.r2_identity < 0.05


def test_the_router_abstains_on_size_under_every_protocol(result):
    """It assigns a group, so it must never contribute a size to a benchmark row."""
    for protocol in result.protocols:
        row = protocol.by_arm()["group-discriminant"]
        assert row.score.n_scored == 0
        assert row.score.n_abstained > 0


def test_a_physically_impossible_prediction_is_refused_not_scored():
    """Refitting the power law with a site held out extrapolates past ten metres.

    Scoring that number rather than refusing it produced a per-fold variance explained of about
    -11000, which then swamped every other fold pooled with it. The guard turns the failure into an
    abstention, which is both truer and more informative.
    """
    blown = []
    for split in leave_one_site_out(bf.load_training_corpus()):
        arm = RefittedRegression().fit(split.train)
        blown.extend(p for p in arm.predict(split.test) if p.abstained)
    assert blown, "no fold triggered the plausibility guard, so this test proves nothing"
    assert any("outside the plausible fragment range" in (p.abstain_reason or "") for p in blown)
    assert PLAUSIBLE_X50_M == (0.001, 3.0)


def test_the_benchmark_stamps_the_dataset_it_ran_on(result):
    from blastfrag.datasets import DATASET_DIGEST

    assert result.dataset_digest == DATASET_DIGEST


def test_each_fold_gets_a_fresh_model(result):
    """Reusing one instance across folds would let a fold's fit leak into the next.

    The runner takes factories rather than instances, and this asserts the interface stayed that way.
    """
    arms = default_arms()
    first, second = arms["random-forest"](), arms["random-forest"]()
    assert first is not second
    assert first.model is None and second.model is None


def test_the_published_holdout_reproduction_runs_and_ranks_as_recorded():
    train = bf.load_training_corpus()
    holdout = bf.load_holdout(protocol="2012")
    results = {r.arm: r for r in run_fixed_holdout(train, holdout, default_arms())}
    assert results["oracle"].score.r2_identity == pytest.approx(1.0)
    assert results["null"].score.r2_identity < 0.0
    # On the fixed published hold-out the learned arms DO beat the classical one. That is the
    # protocol the literature reports, and it is exactly why the grouped protocol matters.
    assert results["random-forest"].score.r2_identity > results["kuznetsov"].score.r2_identity
