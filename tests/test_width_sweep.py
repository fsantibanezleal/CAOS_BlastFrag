"""The network width sweep: the source's own protocol, and every width held out by site."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.learned import PublishedNeuralNetwork

pytestmark = pytest.mark.trains

WIDTHS = (6,)
SIMULATIONS = 1


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


@pytest.fixture(scope="module")
def holdout():
    return bf.load_holdout(protocol="2012")


@pytest.fixture(scope="module")
def sweep(corpus, holdout):
    return bf.network_width_sweep(corpus, holdout, widths=WIDTHS, n_simulations=SIMULATIONS, seed=0)


def test_the_sweep_fits_every_width_with_its_simulations(corpus, holdout, monkeypatch):
    seen: list[tuple[dict, int]] = []
    original = PublishedNeuralNetwork.__init__

    def recording(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append((dict(self.hidden), self.n_simulations))

    monkeypatch.setattr(PublishedNeuralNetwork, "__init__", recording)
    result = bf.network_width_sweep(corpus, holdout, widths=(6, 7), n_simulations=2, seed=0)
    held_out = [{1: 6, 2: 6}, {1: 7, 2: 7}, {1: 9, 2: 7}]
    for hidden in held_out:
        # One network per fold, ten folds, every one with the requested simulations.
        assert sum(1 for h, n in seen if h == hidden and n == 2) >= 10, hidden
    assert [row["hidden"] for row in result["leave_one_site_out"]] == [{1: 9, 2: 7}, {1: 6, 2: 6}, {1: 7, 2: 7}]
    assert result["widths"] == [6, 7] and result["n_simulations"] == 2


def test_the_published_protocol_reports_its_choice_beside_the_published_optimum(sweep):
    protocol = sweep["published_protocol"]
    assert set(protocol) == {1, 2}
    for group, entry in protocol.items():
        assert entry["published_optimum"] == {1: 9, 2: 7}[group]
        assert entry["best_hidden"] in WIDTHS
        assert [row["hidden"] for row in entry["table"]] == list(WIDTHS)
        assert all({"rmse", "correlation"} <= set(row) for row in entry["table"])


def test_every_width_is_scored_held_out_by_site_on_both_row_sets(sweep):
    for row in sweep["leave_one_site_out"]:
        supports = row["supports"]
        assert set(supports) == {"all", "geometry"}
        assert supports["all"]["n_scored"] + supports["all"]["n_abstained"] == 97
        assert supports["geometry"]["n_scored"] + supports["geometry"]["n_abstained"] == 91
        assert supports["all"]["r2_identity"] is not None


def test_no_fold_of_the_sweep_sees_its_held_out_site(corpus, holdout, monkeypatch):
    trained_on: list[set[str]] = []
    asked_for: list[set[str]] = []
    fit, predict = PublishedNeuralNetwork.fit, PublishedNeuralNetwork.predict

    def recording_fit(self, blasts):
        trained_on.append({b.site for b in blasts})
        return fit(self, blasts)

    def recording_predict(self, blasts):
        asked_for.append({b.site for b in blasts})
        return predict(self, blasts)

    monkeypatch.setattr(PublishedNeuralNetwork, "fit", recording_fit)
    monkeypatch.setattr(PublishedNeuralNetwork, "predict", recording_predict)
    bf.network_width_sweep(corpus, holdout, widths=WIDTHS, n_simulations=SIMULATIONS, seed=0)
    held_out_pairs = [(t, a) for t, a in zip(trained_on[-len(asked_for):], asked_for) if len(a) == 1]
    assert len(held_out_pairs) == 20  # ten sites, for the published pair and for width 6
    assert all(not (train & test) for train, test in held_out_pairs)


def test_the_sweep_is_reproducible_for_a_seed(corpus, holdout, sweep):
    again = bf.network_width_sweep(corpus, holdout, widths=WIDTHS, n_simulations=SIMULATIONS, seed=0)
    assert again == sweep


def test_the_published_widths_are_marked(sweep):
    published = [row for row in sweep["leave_one_site_out"] if row["published"]]
    assert len(published) == 1 and published[0]["hidden"] == {1: 9, 2: 7}
    assert sweep["published_widths"] == {1: 9, 2: 7}
