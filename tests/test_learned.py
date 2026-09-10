"""The learned rungs, and the reproduction result that came out of building them.

The headline here is a partial refutation, and it is the point of the module. The published network
is fully specified, so it can be reproduced exactly rather than approximately. Reproduced, it does
not reach the score its paper reports, and the shortfall is concentrated in the blasts the paper
itself flags as unstable.
"""

from __future__ import annotations

import statistics

import pytest

import blastfrag as bf
from blastfrag.learned import (
    BackPropNetwork,
    GradientBoosting,
    MinMaxScaler,
    PublishedNeuralNetwork,
    RandomForest,
    StackingEnsemble,
    SupportVectorRegression,
)
from blastfrag.metrics import score
from blastfrag.models import PublishedRegression
from blastfrag.types import Prediction

sklearn = pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def train():
    return bf.load_training_corpus()


@pytest.fixture(scope="module")
def holdout():
    return bf.load_holdout(protocol="2012")


# ---------------------------------------------------------------------------------------------
# The optimiser itself
# ---------------------------------------------------------------------------------------------

def test_the_jacobian_matches_a_finite_difference():
    """Levenberg-Marquardt is hand-written, so its derivative has to be checked, not assumed.

    A wrong Jacobian still trains, just slowly and to a worse optimum, which is exactly the kind of
    defect that ships looking fine.
    """
    import numpy as np

    net = BackPropNetwork(5, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(bf.FEATURES)))
    weights = rng.normal(0.0, 0.5, size=net._n_parameters())

    analytic = net._jacobian(weights, X)
    numeric = np.zeros_like(analytic)
    eps = 1e-6
    for k in range(len(weights)):
        up, down = weights.copy(), weights.copy()
        up[k] += eps
        down[k] -= eps
        numeric[:, k] = (net._forward(up, X)[0] - net._forward(down, X)[0]) / (2 * eps)

    assert np.abs(analytic - numeric).max() < 1e-7


def test_the_optimiser_reduces_the_loss_monotonically():
    """Damped Gauss-Newton accepts a step only when it improves, so the history cannot rise."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(40, len(bf.FEATURES)))
    y = X[:, 0] * 0.5 + X[:, 4] * 0.3 + 0.1
    net = BackPropNetwork(4, seed=0).fit(X, y)
    assert all(b <= a + 1e-12 for a, b in zip(net.history, net.history[1:]))
    assert net.history[-1] < net.history[0]


def test_the_seed_is_drawn_before_training_so_fits_are_order_independent():
    """A network whose weights depend on how many models were fitted earlier is not reproducible."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(40, len(bf.FEATURES)))
    y = X[:, 0] * 0.5 + 0.1

    first = BackPropNetwork(4, seed=7).fit(X, y).weights
    _decoy = BackPropNetwork(6, seed=99).fit(X, y)
    second = BackPropNetwork(4, seed=7).fit(X, y).weights
    assert first == second


def test_min_max_normalisation_maps_the_training_range_onto_the_unit_interval():
    scaler = MinMaxScaler().fit([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    assert scaler.transform([0.0, 10.0]) == [0.0, 0.0]
    assert scaler.transform([4.0, 30.0]) == [1.0, 1.0]
    assert scaler.transform([2.0, 20.0]) == [0.5, 0.5]
    # A constant column has no range, and dividing by its zero span would be a silent infinity.
    flat = MinMaxScaler().fit([[5.0], [5.0]])
    assert flat.transform([5.0]) == [0.5]


# ---------------------------------------------------------------------------------------------
# The reproduction result
# ---------------------------------------------------------------------------------------------

def test_the_network_fits_its_training_rows_almost_exactly(train):
    """Which is the problem, not the achievement.

    The published architecture puts 64 free parameters on the 62 low-modulus blasts and 82 on the
    35 high-modulus ones. A fit this good on training data is what over-parameterisation looks like.
    """
    arm = PublishedNeuralNetwork(seed=1).fit(train)
    assert arm.n_inputs_times(2) == 64
    assert arm.n_inputs_times(1) == 82
    assert sum(1 for b in train if bf.assign_group(b) == 2) == 62
    assert sum(1 for b in train if bf.assign_group(b) == 1) == 35

    result = score(train, arm.predict(train))
    assert result.r2_identity > 0.95, "the network should nearly interpolate its own training rows"


def test_most_holdout_rows_reproduce_the_published_network_closely(train, holdout):
    """Ten of twelve land within 0.06 m of the published mean, which is the reproduction working."""
    arm = PublishedNeuralNetwork(seed=1).fit(train)
    predictions = {p.blast_id: p for p in arm.predict(holdout)}
    close = [
        b.blast_id
        for b in holdout
        if abs(predictions[b.blast_id].x50_m - b.meta["published"]["nn_mean_2012"]) <= 0.06
    ]
    assert len(close) >= 10
    # And the two that do not are the two the SOURCE flags as its own most unstable rows.
    apart = {b.blast_id for b in holdout} - set(close)
    assert apart <= {"Ru7", "Db10"}


@pytest.mark.slow
def test_the_published_network_score_is_not_reachable_across_seeds(train, holdout):
    """The refutation, measured rather than asserted.

    Across thirty seeds the reproduction's variance explained on the published hold-out runs from
    0.167 to 0.636 with a median near 0.34. The published 0.910 is above every one of them, and no
    seed reaches even 0.7.

    Marked slow because it trains 480 networks. It is the module's most important test.
    """
    values = []
    for seed in range(30):
        arm = PublishedNeuralNetwork(seed=seed).fit(train)
        values.append(score(holdout, arm.predict(holdout)).r2_identity)

    assert max(values) < 0.7, f"a seed reached {max(values):.3f}, which the recorded run did not"
    assert 0.1 < statistics.median(values) < 0.6
    published = score(
        holdout,
        [
            Prediction(method="pub", blast_id=b.blast_id, x50_m=b.meta["published"]["nn_mean_2012"])
            for b in holdout
        ],
    ).r2_identity
    assert published == pytest.approx(0.910, abs=0.005)
    assert published > max(values)


def test_the_instability_is_reported_on_every_prediction(train, holdout):
    """The source reports a coefficient of variation across its simulations, reaching 0.76.

    Carrying it per prediction is what turns "the model is uncertain here" from a footnote into
    something a caller can act on.
    """
    arm = PublishedNeuralNetwork(seed=1).fit(train)
    for prediction in arm.predict(holdout):
        assert "coefficient_of_variation" in prediction.detail
        assert len(prediction.detail["simulations_m"]) == 8
    worst = max(
        arm.predict(holdout), key=lambda p: p.detail["coefficient_of_variation"]
    )
    assert worst.blast_id == "Ru7", "the source flags this same blast as its most variable"
    assert worst.detail["coefficient_of_variation"] > 0.4


def test_predictions_are_clamped_to_the_target_range_the_network_was_fitted_on(train):
    """A min-max target confines the network's claim to the interval it saw.

    Without this the wildest simulations drag one blast's eight-run mean to zero and take the arm's
    variance explained from positive to below -1 on that row alone.
    """
    arm = PublishedNeuralNetwork(seed=1).fit(train)
    low, high = arm.target_range[2]
    for prediction in arm.predict(bf.load_holdout(protocol="2012")):
        for value in prediction.detail["simulations_m"]:
            group_low, group_high = arm.target_range[prediction.group]
            assert group_low - 1e-9 <= value <= group_high + 1e-9
    assert low > 0


def test_db10_is_irreproducible_by_both_published_models_and_no_single_input_fixes_it(train):
    """A convergent inconsistency in one published row, and a hypothesis that did not survive.

    Db10's published inputs give 0.324 m from the published regression equation where both papers
    print 0.16, and our network reproduction predicts 0.6 to 0.76 across every seed where the paper
    prints 0.33. Two independent models disagree with the source in the same direction on the same
    row.

    The natural hypothesis was that one input cell is wrong. It was tested by scanning each of the
    seven inputs for the value that reconciles the regression, and then checking whether that same
    value also reconciles the network. **None does.** The closest, a stiffness ratio of 1.16, is
    below the corpus minimum of 1.33 and still leaves the network at 0.26 against a published 0.33.

    So the row is carried with its published inputs and its inconsistency recorded, rather than
    adjusted to fit. This test pins the refuted hypothesis so nobody re-derives it.
    """
    import dataclasses

    db10 = next(b for b in bf.load_holdout(protocol="union") if b.blast_id == "Db10")
    regression = PublishedRegression()
    assert regression.predict_one(db10).x50_m == pytest.approx(0.324, abs=0.002)
    assert db10.meta["published"]["regression_2010"] == 0.16
    assert db10.meta["published"]["nn_mean_2012"] == 0.33

    nets = [PublishedNeuralNetwork(seed=s).fit(train) for s in range(3)]
    network_value = statistics.fmean(n.predict_one(db10).x50_m for n in nets)
    assert network_value > 0.55, "the network reproduction should sit well above the published 0.33"

    # The refuted hypothesis: no single input value reconciles both models at once.
    reconciled_both = []
    for feature in bf.FEATURES:
        original = getattr(db10, feature)
        for k in range(400):
            value = (original * 0.05) * ((20 / 0.05) ** (k / 399))
            candidate = dataclasses.replace(db10, **{feature: value})
            regressed = regression.predict_one(candidate).x50_m
            if regressed is None or abs(regressed - 0.16) > 0.005:
                continue
            values = [n.predict_one(candidate).x50_m for n in nets]
            if any(v is None for v in values):
                continue
            network = statistics.fmean(values)
            if abs(network - 0.33) < 0.05:
                reconciled_both.append((feature, value))
    assert not reconciled_both, f"a single-input fix was found after all: {reconciled_both}"


# ---------------------------------------------------------------------------------------------
# The 2025 arms
# ---------------------------------------------------------------------------------------------

def test_the_two_published_kernels_disagree_and_both_ship(train, holdout):
    """One source chose a radial kernel and called it better; another chose polynomial and reported
    this arm as its worst. Averaging them would erase the disagreement."""
    radial = score(holdout, SupportVectorRegression(variant="rbf-amoako").fit(train).predict(holdout))
    polynomial = score(holdout, SupportVectorRegression(variant="poly-sui").fit(train).predict(holdout))
    assert radial.r2_identity > polynomial.r2_identity + 0.15
    assert radial.method != polynomial.method


def test_the_tree_arms_beat_the_null_and_the_classical_arm(train, holdout):
    from blastfrag.models import Kuznetsov, NullModel

    floor = score(holdout, NullModel().fit(train).predict(holdout)).rmse_m
    classical = score(holdout, Kuznetsov().predict(holdout)).rmse_m
    for arm in (RandomForest(), GradientBoosting()):
        result = score(holdout, arm.fit(train).predict(holdout))
        assert result.rmse_m < classical < floor, arm.name


def test_the_boosting_arm_overfits_as_its_source_reports(train, holdout):
    """The source states its boosting model overfits at the learning rate it published.

    Reproduced as published rather than tuned away, and the overfitting is visible as a large gap
    between the training and hold-out fit.
    """
    arm = GradientBoosting().fit(train)
    on_train = score(train, arm.predict(train)).r2_identity
    on_holdout = score(holdout, arm.predict(holdout)).r2_identity
    assert on_train > 0.98
    assert on_train - on_holdout > 0.05


def test_the_stacking_arm_carries_its_sources_removed_cross_validation():
    assert "cross-validation" in StackingEnsemble.__doc__
    assert "protocol" in StackingEnsemble.__doc__


def test_every_learned_arm_refuses_to_fit_on_too_few_rows(train):
    for arm in (RandomForest(), GradientBoosting(), SupportVectorRegression()):
        with pytest.raises(ValueError, match="at least 10 measured blasts"):
            arm.fit(train[:5])


def test_an_unfitted_arm_abstains_rather_than_raising(train):
    prediction = RandomForest().predict_one(train[0])
    assert prediction.abstained
    assert "has not been fitted" in (prediction.abstain_reason or "")


def test_the_standard_score_and_the_min_max_normalisers_are_kept_apart(train):
    """The 2012 network uses min-max and the 2025 ensembles a standard score.

    Silently unifying them would change every learned number, so the two live in different places
    and this test says so.
    """
    forest = RandomForest().fit(train)
    assert forest.mean and forest.sd
    network = PublishedNeuralNetwork(seed=0).fit(train)
    assert network.scalers[2].minimum and network.scalers[2].maximum
