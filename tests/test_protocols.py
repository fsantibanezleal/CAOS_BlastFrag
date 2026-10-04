"""Split protocols and arm provenance: numpy-only, so these run in CI."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.metrics import quantile, summarise_draws
from blastfrag.models import KuznetsovTransfer
from blastfrag.splits import (
    REPEATED_PROTOCOLS,
    all_protocols,
    leave_one_site_out,
    repeated_deduplicated_splits,
    repeated_random_splits,
)


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


def test_repeated_draws_are_seeded_distinct_and_reproducible(corpus):
    first = repeated_random_splits(corpus, n_repeats=4, seed=10)
    again = repeated_random_splits(corpus, n_repeats=4, seed=10)
    assert [s.seed for s in first] == [10, 11, 12, 13]
    assert [tuple(b.blast_id for b in s.test) for s in first] == [
        tuple(b.blast_id for b in s.test) for s in again
    ]
    assert len({tuple(b.blast_id for b in s.test) for s in first}) == 4
    for split in repeated_deduplicated_splits(corpus, n_repeats=3):
        train = {b.features() for b in split.train}
        assert not any(b.features() in train for b in split.test)


def test_all_protocols_returns_the_repeats_and_the_ten_folds(corpus):
    protocols = all_protocols(corpus, n_repeats=3)
    assert len(protocols["random-8020"]) == len(protocols["dedup-random"]) == 3
    assert len(protocols["leave-one-site-out"]) == 10
    assert REPEATED_PROTOCOLS == {"random-8020", "dedup-random"}
    with pytest.raises(ValueError):
        repeated_random_splits(corpus, n_repeats=0)


def test_the_quantile_is_the_interpolating_definition():
    values = [4.0, 1.0, 3.0, 2.0]
    assert quantile(values, 0.5) == 2.5
    assert quantile(values, 0.0) == 1.0 and quantile(values, 1.0) == 4.0
    summary = summarise_draws(values)
    assert summary["n"] == 4 and summary["median"] == 2.5
    assert summarise_draws([]) == {"n": 0}


def test_the_transfer_fit_is_one_point_per_training_site(corpus):
    arm = KuznetsovTransfer().fit(corpus)
    assert arm.fit_sites == tuple(sorted(bf.SITE_ROCK_FACTOR))
    # Stiffer rock gets a larger factor on this corpus, so the slope is positive.
    assert arm.slope > 0
    with pytest.raises(ValueError, match="at least 3"):
        KuznetsovTransfer().fit([b for b in corpus if b.site in {"Murgul", "Soma"}])


def test_the_transfer_arm_abstains_where_there_is_no_geometry(corpus):
    split = next(s for s in leave_one_site_out(corpus) if s.test[0].site == "Miami")
    arm = KuznetsovTransfer().fit(split.train)
    assert all(p.abstained for p in arm.predict(split.test))


def test_every_arm_declares_what_it_was_fitted_on():
    from blastfrag.models import LADDER

    for name, cls in LADDER.items():
        provenance = cls.provenance(cls.__new__(cls))
        assert set(provenance) == {
            "fitted_on", "in_sample_corpus", "uses_site_constant", "router_in_sample"
        }, name
    assert bf.PublishedRegression.in_sample_corpus and bf.GroupDiscriminant.in_sample_corpus
    assert bf.Kuznetsov.uses_site_constant and not KuznetsovTransfer.uses_site_constant
