"""The protocol-sensitivity benchmark, and what it does and does not establish.

Every number asserted here was measured by running the sweep. They are pinned because they are the
package's central claims, and a silent change to any of them would change what a product built on
the package says.

0.3.0 withdrew two claims that 0.2.x pinned in this file. "The classical arm improves when held out
by site" and "deduplication raises the learned scores" were both read off a single random draw
(seed 0); repeated over 100 draws, neither holds. The slow tests at the bottom pin the repeated
figures so they cannot come back.
"""

from __future__ import annotations

import hashlib

import pytest

import blastfrag as bf
from blastfrag.benchmark import (
    KILL_CRITERION,
    PUBLISHED_RANDOM_SPLIT_R2,
    default_arms,
    run_benchmark,
    run_fixed_holdout,
)
from blastfrag.models import PLAUSIBLE_X50_M, KuznetsovTransfer, RefittedRegression
from blastfrag.splits import leave_one_site_out

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def result():
    # Few draws and few resamples keep this module to seconds; the grouped protocol, which carries
    # every leave-one-site-out figure, does not depend on either.
    return run_benchmark(bf.load_training_corpus(), default_arms(), seed=0, n_repeats=5, n_boot=300)


def grouped(result, arm, support="all"):
    return result.by_protocol()["leave-one-site-out"].by_arm()[arm].detail["supports"][support]


# ---------------------------------------------------------------------------------------------
# The criterion, as declared
# ---------------------------------------------------------------------------------------------

def test_the_kill_criterion_text_has_not_been_edited_since_it_was_declared():
    """Changing a declared criterion after seeing the data is the failure it guards against.

    0.3.0 changes what is reported beside the criterion, never the criterion. The hash pins the
    sentence as it stood after its one recorded amendment (the positivity half, added after the
    first run caught the margin-only version passing two failures).
    """
    assert "positive" in KILL_CRITERION and "0.10" in KILL_CRITERION
    assert hashlib.sha256(KILL_CRITERION.encode()).hexdigest()[:12] == "a13f11385ef8"


def test_the_verdict_depends_on_which_rows_are_scored(result):
    """The finding 0.3.0 exists to report.

    Over all 97 blasts the best learned arm is below zero. Over the 91 blasts whose geometry is
    resolvable, the rows the classical arms also answer, the same arm is above zero and more than
    0.10 above the null, which meets the criterion as written. The six blasts that separate the two
    are one site.
    """
    verdict = result.verdict
    on_all, on_geometry = verdict["supports"]["all"], verdict["supports"]["geometry"]
    assert on_all["n_blasts"] == 97 and on_geometry["n_blasts"] == 91
    # Gradient boosting and the stacked model it dominates (see the learned tests) tie to the third
    # decimal, so which of the two leads a support is not the finding; their level is.
    assert on_all["best_learned_arm"] == "xgboost"
    assert on_geometry["best_learned_arm"] in {"xgboost", "stacking"}
    assert on_all["best_learned_r2_identity"] == pytest.approx(-0.034, abs=0.005)
    assert on_geometry["best_learned_r2_identity"] == pytest.approx(0.034, abs=0.005)
    assert on_all["generalises_across_sites"] is False
    assert on_geometry["generalises_across_sites"] is True
    assert verdict["depends_on_support"] is True
    assert verdict["sites_outside_geometry_support"] == ["Miami"]
    assert verdict["outcome"].startswith("THE VERDICT DEPENDS ON THE ROW SET")


def test_no_arm_fitted_without_the_corpus_has_an_interval_above_zero(result):
    """With ten sites, a site-resampled interval spans zero for every arm that is not in sample."""
    verdict = result.verdict
    assert verdict["arms_with_interval_above_zero"] == []
    for arm in ("kuznetsov", "kuznetsov-transfer", "xgboost", "random-forest", "stacking"):
        for support in ("all", "geometry"):
            low, high = grouped(result, arm, support)["interval_95"]
            assert low < 0.0 < high or high < 0.0, (arm, support, low, high)
    # The in-sample regression is the one arm whose interval clears zero, and it is in sample.
    low, _ = grouped(result, "published-regression")["interval_95"]
    assert low > 0.0


def test_the_null_is_anti_correlated_with_the_truth_under_this_protocol(result):
    """Holding out a coarse site lowers the training mean, so the null predicts low exactly where the
    truth is high. Any margin over it is inflated, and the outcome sentence says so."""
    r = grouped(result, "null")["score"]["pearson_r"]
    assert r < -0.7
    assert "correlate with the measurements at -0.79" in result.verdict["outcome"]


# ---------------------------------------------------------------------------------------------
# What each arm was fitted on
# ---------------------------------------------------------------------------------------------

def test_in_sample_arms_are_reported_apart_from_the_arms_that_transfer(result):
    """The published regression's coefficients were fitted on these 97 blasts by its source.

    0.2.x listed it among the arms positive across sites and described it as fixed rather than
    fitted. It is now reported as in sample with the reason, and never in the transfer list.
    """
    verdict = result.verdict
    in_sample = {row[0]: row for row in verdict["in_sample_arms"]}
    assert set(in_sample) == {"published-regression"}
    assert in_sample["published-regression"][1] == pytest.approx(0.802, abs=0.005)
    assert "all 97 blasts" in in_sample["published-regression"][2]
    positive = dict(verdict["arms_with_positive_variance_explained_across_sites"])
    assert "published-regression" not in positive
    assert set(positive) == {"kuznetsov", "kuznetsov-transfer", "kuznetsov-capped"}
    assert result.provenance["group-discriminant"]["in_sample_corpus"] is True
    assert result.provenance["refitted-regression"]["router_in_sample"] is True
    assert result.provenance["published-neural-net"]["router_in_sample"] is True
    # The cap inherits the site factor of the arm it caps, so it reads a site constant too.
    assert verdict["site_constant_arms"] == ["kuznetsov", "kuznetsov-capped"]


def test_the_classical_result_does_not_depend_on_the_held_out_sites_rock_factor(result):
    """The shipped classical arm reads a factor back-solved from the held-out site's own published
    predictions. The transfer arm predicts it from Young's modulus over the training sites only and
    loses 0.013, so the classical arm's cross-site score is not borrowed."""
    shipped = grouped(result, "kuznetsov")["score"]["r2_identity"]
    transfer = grouped(result, "kuznetsov-transfer")["score"]["r2_identity"]
    assert shipped == pytest.approx(0.311, abs=0.005)
    assert transfer == pytest.approx(0.298, abs=0.005)


def test_the_transfer_arm_never_sees_the_held_out_site():
    for split in leave_one_site_out(bf.load_training_corpus()):
        held_out = split.test[0].site
        arm = KuznetsovTransfer().fit(split.train)
        assert held_out not in arm.fit_sites
        assert len(arm.fit_sites) >= KuznetsovTransfer.MIN_SITES


def test_the_distribution_arms_are_not_benchmarked_as_separate_predictors():
    """Kuz-Ram, Swebrec and the crush-zone composition return the classical mean size."""
    arms = default_arms()
    assert not {"kuz-ram", "swebrec", "crush-zone"} & set(arms)
    for cls in (bf.KuzRam, bf.Swebrec, bf.CrushZone):
        assert cls.shares_mean_size_with == "kuznetsov"


# ---------------------------------------------------------------------------------------------
# Structure of the repeated and grouped protocols
# ---------------------------------------------------------------------------------------------

def test_random_protocols_are_repeated_and_report_the_median(result):
    for protocol in ("random-8020", "dedup-random"):
        block = result.by_protocol()[protocol]
        assert block.repeated and block.n_folds == 5
        forest = block.by_arm()["random-forest"]
        assert forest.detail["seeds"] == [0, 1, 2, 3, 4]
        repeats = forest.detail["repeats"]
        assert repeats["n"] == 5
        assert forest.headline_r2() == repeats["median"]
        # The headline score object is still the seed-0 draw, the reproduction of one split.
        assert forest.score.r2_identity == forest.detail["draws_r2_identity"][0]


def test_every_site_has_a_per_site_entry_and_refusals_carry_reasons(result):
    per_site = result.by_protocol()["leave-one-site-out"].by_arm()["kuznetsov"].detail["per_site"]
    assert set(per_site) == {b.site for b in bf.load_training_corpus()}
    assert per_site["Miami"]["n_scored"] == 0
    assert "hole diameter" in per_site["Miami"]["abstain_reason"]
    assert per_site["Murgul"]["rmse_m"] < 0.1


def test_the_oracle_and_the_null_bracket_every_protocol(result):
    for protocol in result.protocols:
        arms = protocol.by_arm()
        assert arms["oracle"].score.r2_identity == pytest.approx(1.0)
        assert arms["null"].headline_r2() < 0.05


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


def test_each_fold_gets_a_fresh_model():
    """Reusing one instance across folds would let a fold's fit leak into the next."""
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
    # On the fixed published hold-out, whose sites the training rows include, the learned arms DO
    # beat the classical one. That is the protocol the literature reports.
    assert results["random-forest"].score.r2_identity > results["kuznetsov"].score.r2_identity


# ---------------------------------------------------------------------------------------------
# The repeated figures, pinned (slow: about four minutes, most of it the network)
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def repeated():
    return run_benchmark(bf.load_training_corpus(), default_arms(), seed=0, n_repeats=100, n_boot=0)


@pytest.mark.slow
def test_the_classical_arm_scores_the_same_under_every_protocol(repeated):
    """A fixed-coefficient arm predicts the same value for a blast under every protocol, so only the
    scored rows change. Its median random draw sits where its site-held-out score does."""
    random_median = repeated.by_protocol()["random-8020"].by_arm()["kuznetsov"].headline_r2()
    held_out = repeated.by_protocol()["leave-one-site-out"].by_arm()["kuznetsov"].score.r2_identity
    assert random_median == pytest.approx(0.303, abs=0.01)
    assert abs(random_median - held_out) < 0.02


@pytest.mark.slow
def test_deduplication_moves_no_learned_median_by_more_than_a_few_hundredths(repeated):
    shifts = repeated.verdict["dedup_minus_random_median"]
    learned = [a for a in ("random-forest", "xgboost", "stacking", "svr-rbf")]
    for arm in learned:
        assert abs(shifts[arm]) < 0.05, (arm, shifts[arm])


@pytest.mark.slow
def test_every_learned_arm_loses_ground_when_whole_sites_are_held_out(repeated):
    gaps = repeated.verdict["protocol_gap_random_minus_grouped"]
    for arm in ("published-neural-net", "svr-rbf", "random-forest", "xgboost", "stacking"):
        assert gaps[arm] > 0.5, (arm, gaps[arm])
    assert repeated.verdict["median_protocol_gap"] > 0.7


@pytest.mark.slow
def test_the_published_stacking_figure_is_a_favourable_draw(repeated):
    figures = repeated.verdict["published_random_split_figures"]
    assert set(figures) == set(PUBLISHED_RANDOM_SPLIT_R2)
    assert figures["stacking"]["share_of_draws_below"] >= 0.95
