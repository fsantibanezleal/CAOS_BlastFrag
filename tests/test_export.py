"""The portable export, held to the original arms to the last bit.

A browser runs the learned tier from these documents, and its own tests compare against
:func:`predict_portable`. That only means something if the reference agrees with the fitted arm, so
this module checks every exportable arm on every blast the package ships, abstentions included.
"""

from __future__ import annotations

import json
import math

import pytest

import blastfrag as bf
from blastfrag.export import PORTABLE_SCHEMA, PortableError, export_arm, f32, predict_portable
from blastfrag.models import KuznetsovTransfer, RefittedRegression


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


@pytest.fixture(scope="module")
def every_blast(corpus):
    return corpus + bf.load_holdout(protocol="union") + bf.load_field_holdout()


def _agree(arm, document, blasts, *, rel):
    document = json.loads(json.dumps(document))  # what a consumer reads is the serialised form
    worst = 0.0
    for blast in blasts:
        expected = arm.predict_one(blast)
        value, reason = predict_portable(document, blast)
        assert (value is None) == expected.abstained, blast.blast_id
        if value is None:
            assert reason
            continue
        worst = max(worst, abs(value - expected.x50_m) / abs(expected.x50_m))
    assert worst <= rel, worst


def test_the_refitted_power_law_round_trips_exactly(corpus, every_blast):
    arm = RefittedRegression().fit(corpus)
    _agree(arm, export_arm(arm), every_blast, rel=0.0)


@pytest.mark.trains
def test_the_published_network_round_trips(corpus, every_blast):
    from blastfrag.learned import PublishedNeuralNetwork

    arm = PublishedNeuralNetwork(seed=0).fit(corpus)
    document = export_arm(arm)
    assert document["kind"] == "network"
    assert {g["n_hidden"] for g in document["groups"].values()} == {9, 7}
    _agree(arm, document, every_blast, rel=1e-12)


@pytest.mark.parametrize(
    "name, rel",
    [("svr-rbf", 1e-12), ("svr-poly", 1e-12), ("random-forest", 0.0), ("xgboost", 0.0), ("stacking", 0.0)],
)
def test_every_scikit_and_boosting_arm_round_trips(corpus, every_blast, name, rel):
    """Trees agree to the last bit only because the walker rounds inputs, split values and the
    boosting sum to 32 bits, exactly as the libraries do."""
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    arm = bf.default_arms()[name]().fit(corpus)
    _agree(arm, export_arm(arm), every_blast, rel=rel)


def test_the_transfer_arm_exports_its_rock_factor_fit(corpus):
    arm = KuznetsovTransfer().fit(corpus)
    document = export_arm(arm)
    blast = next(b for b in corpus if b.site == "Murgul")
    value, _ = predict_portable(document, blast)
    assert value == pytest.approx(arm.rock_factor_for(blast.E_GPa), rel=1e-15)
    assert document["rock_factor"]["fit_sites"] == list(arm.fit_sites)


def test_an_unfitted_or_unsupported_arm_is_refused():
    with pytest.raises(PortableError):
        export_arm(RefittedRegression())
    with pytest.raises(PortableError):
        export_arm(bf.Kuznetsov())


def test_every_document_carries_the_schema_and_the_input_order(corpus):
    document = export_arm(RefittedRegression().fit(corpus))
    assert document["schema"] == PORTABLE_SCHEMA
    assert document["features"] == list(bf.FEATURES)
    assert document["plausible_x50_m"] == [0.001, 3.0]
    with pytest.raises(PortableError):
        predict_portable(document | {"schema": "something/else"}, corpus[0])


def test_f32_matches_ieee_single_precision():
    assert f32(0.1) == 0.10000000149011612
    assert f32(1.0) == 1.0
    assert math.isinf(f32(1e39))
