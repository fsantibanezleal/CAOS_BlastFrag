"""The in-situ block cap: a declared choice that changes a prediction only where it must."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.classical import IN_SITU_CAP_SOURCE, cap_at_in_situ_block, rosin_rammler, sieve_grid
from blastfrag.models import InSituCap, Kuznetsov


@pytest.fixture(scope="module")
def corpus():
    return bf.load_training_corpus()


@pytest.fixture(scope="module")
def pairs(corpus):
    """Every corpus blast with its uncapped and capped classical predictions."""
    base, capped = Kuznetsov(), InSituCap(Kuznetsov())
    return [(b, base.predict_one(b), capped.predict_one(b)) for b in corpus]


def test_no_capped_prediction_exceeds_the_in_situ_block(pairs):
    answered = [(b, c) for b, _, c in pairs if not c.abstained]
    assert len(answered) == 91
    assert all(c.x50_m <= b.XB_m for b, c in answered)


def test_the_cap_changes_nothing_where_it_does_not_bind(pairs):
    for blast, uncapped, capped in pairs:
        if uncapped.abstained or uncapped.x50_m <= blast.XB_m:
            assert capped.x50_m == uncapped.x50_m, blast.blast_id


def test_a_binding_cap_is_recorded_on_the_prediction(pairs):
    for blast, uncapped, capped in pairs:
        if capped.abstained:
            continue
        assert capped.detail["capped_from_m"] == uncapped.x50_m
        assert capped.detail["in_situ_block_m"] == blast.XB_m
        assert capped.detail["cap_binds"] == (uncapped.x50_m > blast.XB_m)


def test_the_capped_curve_passes_everything_at_the_block_size():
    sizes = sieve_grid()
    curve = rosin_rammler(0.5, 1.2, sizes_m=sizes)
    block = 0.4
    capped = cap_at_in_situ_block(curve, block)
    for x, before, after in zip(sizes, curve.passing, capped.passing):
        assert after == (before if x < block else 1.0)
    assert capped.x50_m == block
    assert capped.detail["cap_binds"] is True and capped.detail["uncapped_x50_m"] == 0.5
    # A block far above the mean size changes nothing below it, and the mean size stays.
    loose = cap_at_in_situ_block(curve, 2.5)
    assert loose.x50_m == 0.5 and loose.detail["cap_binds"] is False
    with pytest.raises(ValueError, match="positive"):
        cap_at_in_situ_block(curve, 0.0)


def test_the_cap_declares_that_it_is_not_published():
    arm = InSituCap(Kuznetsov())
    assert arm.provenance()["declared_not_published"] is True
    assert arm.provenance()["caps"] == "kuznetsov"
    assert "not a published relation" in arm.source and IN_SITU_CAP_SOURCE in arm.source


def test_the_cap_binds_on_three_reocin_blasts_and_no_other(pairs):
    binding = sorted(b.blast_id for b, _, c in pairs if not c.abstained and c.detail["cap_binds"])
    assert binding == ["Rc1", "Rc2", "Rc3"]
    # No measured mean size anywhere exceeds its block, which is the physical premise of the cap.
    assert all(b.x50_m <= b.XB_m for b, _, _ in pairs)


def test_the_capped_arm_is_in_the_default_ladder_with_its_provenance():
    arms = bf.default_arms(include_learned=False)
    assert "kuznetsov-capped" in arms
    arm = arms["kuznetsov-capped"]()
    base = Kuznetsov()
    assert arm.name == "kuznetsov-capped" and arm.tier == base.tier and arm.lane == base.lane
    assert {k: arm.provenance()[k] for k in base.provenance()} == base.provenance()
    assert arm.provenance()["uses_site_constant"] is True


def test_the_capped_arm_abstains_where_its_base_does(pairs):
    abstained = [(u, c) for _, u, c in pairs if u.abstained]
    assert len(abstained) == 6
    assert all(c.abstained and c.abstain_reason == u.abstain_reason for u, c in abstained)
