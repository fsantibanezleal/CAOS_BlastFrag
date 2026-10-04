"""Diagnostics report on the corpus and on fitted arms; none of them changes a row or a score."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.diagnostics import (
    PUBLISHED_IMPORTANCE,
    grouped_resampling_importance,
    native_importance,
    outlier_screen,
)
from blastfrag.models import KuznetsovTransfer, PublishedRegression


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


def test_the_outlier_screen_reports_and_never_filters(corpus):
    pytest.importorskip("sklearn")
    report = outlier_screen(corpus)
    assert report["applied_as_filter"] is False
    assert report["n_blasts"] == 97
    assert len(report["anomaly_score"]) == 97
    assert set(report["flagged"]) <= {b.blast_id for b in corpus}
    # The default contamination is the rate Huan et al. observed, five of 105.
    assert len(report["flagged"]) == round(97 * 5 / 105)
    assert sum(report["flagged_by_site"].values()) == len(report["flagged"])


def test_the_forest_ranks_the_modulus_first_as_the_2025_study_does(corpus):
    pytest.importorskip("sklearn")
    arm = bf.default_arms()["random-forest"]().fit(corpus)
    values = native_importance(arm)["values"]
    assert max(values, key=values.get) == "E_GPa"
    assert "E_GPa" in PUBLISHED_IMPORTANCE["random-forest"]


def test_native_importance_is_none_for_an_arm_without_one(corpus):
    assert native_importance(PublishedRegression()) is None


def test_resampling_importance_runs_under_leave_one_site_out(corpus):
    """Numpy-only arm, so this runs without the learned extras."""
    report = grouped_resampling_importance(KuznetsovTransfer, corpus, n_repeats=3, seed=0)
    assert report["n_rows"] == 91  # Miami has no geometry, so the arm abstains there
    shares = [v for v in report["share"].values() if v is not None]
    assert sum(shares) == pytest.approx(1.0)
    # The rock factor is predicted from the modulus, so resampling it must cost something.
    assert report["mean_increase_in_mse"]["E_GPa"] > 0
