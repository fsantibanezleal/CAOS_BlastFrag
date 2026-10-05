"""Common support: every arm also scored on the rows every size-predicting arm answered."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.benchmark import SUPPORTS
from blastfrag.metrics import score
from blastfrag.splits import all_protocols


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


class _ConstantLearned(bf.NullModel):
    """A stand-in for the learned tier that needs no training, so the verdict is evaluated."""

    name = "constant-learned"
    tier = "learned"


@pytest.fixture(scope="module")
def arms():
    # The closed forms, the router and the regressions: every kind of abstention the benchmark meets
    # (no geometry, a group rather than a size, a refit leaving the plausible range), and no training.
    return bf.default_arms(include_learned=False) | {"constant-learned": _ConstantLearned}


@pytest.fixture(scope="module")
def result(corpus, arms):
    return bf.run_benchmark(corpus, arms, n_repeats=5, n_boot=100)


def test_every_draw_is_also_scored_on_the_rows_every_arm_answered(corpus, arms, result):
    random = result.by_protocol()["random-8020"]
    first_split = all_protocols(corpus, seed=0, n_repeats=5)["random-8020"][0]
    common = random.arms[0].detail["common"]["arms"]
    predictions = {}
    for name in common:
        arm = arms[name]()
        arm.fit(first_split.train)
        predictions[name] = arm.predict(first_split.test)
    answered = [{p.blast_id for p in preds if not p.abstained} for preds in predictions.values()]
    keep = set.intersection(*answered)
    for name in common:
        rows = [b for b in first_split.test if b.blast_id in keep]
        by_id = {p.blast_id: p for p in predictions[name]}
        expected = score(rows, [by_id[b.blast_id] for b in rows], method=name).r2_identity
        found = random.by_arm()[name].detail["common"]["draws_r2_identity"][0]
        assert found == pytest.approx(expected, rel=1e-12), name
    assert all(len(r.detail["common"]["draws_r2_identity"]) == 5 for r in random.arms)


def test_the_common_rows_are_exactly_the_intersection(result):
    held_out = result.by_protocol()["leave-one-site-out"]
    answered = {
        r.arm: {blast for blast, value in r.detail["predictions"].items() if value is not None}
        for r in held_out.arms
    }
    size_arms = sorted(name for name, rows in answered.items() if rows)
    keep = set.intersection(*(answered[name] for name in size_arms))
    for r in held_out.arms:
        assert r.detail["common"]["arms"] == size_arms
        assert r.detail["common"]["n_rows"] == len(keep)
    # Every size-predicting arm is scored on exactly those rows, with no abstention among them.
    for name in size_arms:
        block = held_out.by_arm()[name].detail["common"]["score"]
        assert block["n_scored"] == len(keep) and block["n_abstained"] == 0, name


def test_the_held_out_result_carries_a_common_support_score(result):
    held_out = result.by_protocol()["leave-one-site-out"]
    for r in held_out.arms:
        common = r.detail["common"]
        assert set(common) == {"score", "interval_95", "n_rows", "n_sites", "arms"}
    kuznetsov = held_out.by_arm()["kuznetsov"].detail["common"]
    # The six Miami blasts have no geometry and the refit leaves the plausible range on some rows, so
    # the common rows are fewer than the 91 the classical arm answers.
    assert 0 < kuznetsov["n_rows"] <= 91
    assert kuznetsov["interval_95"] is not None and len(kuznetsov["interval_95"]) == 2
    assert kuznetsov["interval_95"][0] < kuznetsov["score"]["r2_identity"] < kuznetsov["interval_95"][1]
    assert not any("_kept_predictions" in r.detail for p in result.protocols for r in p.arms)


def test_the_criterion_is_not_evaluated_on_the_common_rows(result):
    assert set(SUPPORTS) == {"all", "geometry"}
    assert set(result.verdict["supports"]) == {"all", "geometry"}
    assert all(set(by) == {"all", "geometry"} for by in result.verdict["intervals_95"].values())


def test_an_arm_that_never_answers_does_not_empty_the_common_rows(result):
    for protocol in result.protocols:
        names = protocol.arms[0].detail["common"]["arms"]
        assert "group-discriminant" not in names
        assert "kuznetsov" in names and "null" in names
    held_out = result.by_protocol()["leave-one-site-out"]
    assert held_out.by_arm()["group-discriminant"].detail["common"]["n_rows"] > 0
