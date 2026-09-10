"""The geometry reconstruction must agree with everything the source says about each site.

This is the module that turns a dimensionless corpus back into patterns with rock volumes and charge
masses, which is what lets the classical models run on real blasts at all. Its correctness rests
entirely on the assertions below: the source states dimensions in its prose, and the reconstruction
has to reproduce them.
"""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.geometry import SITE_GEOMETRY


def test_every_site_in_every_shipped_dataset_resolves_through_the_registry():
    sites = set()
    for blasts in bf.load_all().values():
        sites |= {b.site for b in blasts}
    assert sites <= set(SITE_GEOMETRY), f"unregistered sites: {sorted(sites - set(SITE_GEOMETRY))}"


def test_the_reconstruction_reproduces_every_published_narrative_constraint():
    """The whole claim, in one assertion.

    ``verify_reconstruction`` raises if any reconstructed dimension leaves the range the source
    states for that site, after allowing for the two-decimal printing of the ratios.
    """
    report = bf.verify_reconstruction(bf.load_training_corpus())
    checked = [c for site in report.values() for c in site["checks"]]
    assert len(checked) == 15, "the number of published constraints being asserted"
    assert all(c["ok"] for c in checked)


def test_murgul_is_the_decisive_check():
    """Three independent constraints from one paragraph, all three reproduced.

    An error in the arithmetic, in the diameter or in the shipped ratios would break at least one of
    these, which is what makes the reconstruction a result rather than an assumption.
    """
    report = bf.verify_reconstruction(bf.load_training_corpus())["Murgul"]
    quantities = {c["quantity"]: c for c in report["checks"]}
    assert set(quantities) == {"bench_height_m", "burden_m", "spacing_m"}
    assert all(c["ok"] for c in quantities.values())
    assert report["hole_diameter_mm"] == 165.0
    assert quantities["burden_m"]["reconstructed"] == (4.5, 5.0)
    assert quantities["spacing_m"]["reconstructed"][0] == 4.5


def test_dongri_buzurg_is_the_second_check():
    report = bf.verify_reconstruction(bf.load_training_corpus())["Dongri-Buzurg"]
    quantities = {c["quantity"]: c for c in report["checks"]}
    assert set(quantities) == {"bench_height_m", "burden_m", "spacing_m"}
    assert all(c["ok"] for c in quantities.values())


def test_the_derived_diameter_is_the_same_on_every_row_of_its_site():
    """Reocin underground publishes no diameter; inverting its stated bench height supplies one.

    That inversion is only trustworthy because all six rows agree, so the agreement is the test.
    """
    rows = [b for b in bf.load_training_corpus() if b.site == "Reocin-UG"]
    assert len(rows) == 6
    implied = [(18.0 / b.H_over_B) / b.B_over_D * 1000.0 for b in rows]
    assert max(implied) - min(implied) < 0.05, implied
    assert implied[0] == pytest.approx(SITE_GEOMETRY["Reocin-UG"].hole_diameter_mm, abs=0.05)


def test_miami_abstains_rather_than_guessing():
    """The geometry negative control.

    Nothing in the source fixes Miami's scale, so a reconstruction there would be an invention. Six
    rows must refuse, and the refusal must say why.
    """
    miami = [b for b in bf.load_training_corpus() if b.site == "Miami"]
    assert len(miami) == 6
    for blast in miami:
        assert not bf.has_absolute_geometry(blast)
        with pytest.raises(bf.GeometryUnavailable, match="publishes no hole diameter"):
            bf.reconstruct_pattern(blast)


def test_ninety_one_of_ninety_seven_blasts_are_reconstructable():
    train = bf.load_training_corpus()
    assert sum(1 for b in train if bf.has_absolute_geometry(b)) == 91


def test_soma_carries_its_unresolved_source_conflict():
    """The source states both a 5 m burden and a 21 cm diameter, and the ratios cannot hold both.

    The registry takes the diameter, which reproduces the stated spacing and bench height exactly,
    and does NOT assert the burden. If someone later adds a burden assertion here it will fail, and
    that failure is the correct outcome rather than a reason to change the diameter.
    """
    site = SITE_GEOMETRY["Soma"]
    asserted = {q for q, _lo, _hi in site.narrative}
    assert "burden_m" not in asserted
    assert asserted == {"bench_height_m", "spacing_m"}
    assert "UNRESOLVED SOURCE CONFLICT" in site.note

    rows = [b for b in bf.load_training_corpus() if b.site == "Soma"]
    burdens = {round(bf.reconstruct_pattern(b).burden_m, 2) for b in rows}
    assert burdens == {6.0}, "taking the published diameter gives 6.00 m, not the stated 5 m"


def test_the_field_set_checks_the_reconstruction_rather_than_using_it():
    """The 2025 field blasts publish their absolute pattern AND their ratios.

    So they are the one place the reconstruction arithmetic can be checked against a published
    answer instead of against a narrative range.
    """
    for blast in bf.load_field_holdout():
        absolute = blast.meta["absolute"]
        pattern = bf.reconstruct_pattern(blast)
        assert pattern.burden_m == pytest.approx(absolute["burden_m"], rel=2e-4)
        assert pattern.spacing_m == pytest.approx(absolute["spacing_m"], rel=2e-4)
        assert pattern.bench_height_m == pytest.approx(absolute["bench_height_m"], rel=2e-4)
        assert pattern.stemming_m == pytest.approx(absolute["stemming_m"], rel=2e-4)
        assert pattern.hole_diameter_mm == pytest.approx(absolute["hole_diameter_mm"])


def test_volume_and_charge_follow_from_the_pattern():
    blast = next(b for b in bf.load_training_corpus() if b.blast_id == "En1")
    pattern = bf.reconstruct_pattern(blast)
    assert pattern.rock_volume_m3 == pytest.approx(
        pattern.burden_m * pattern.spacing_m * pattern.bench_height_m
    )
    assert pattern.charge_mass_kg == pytest.approx(
        blast.Pf_kg_m3 * pattern.rock_volume_m3
    )
    assert pattern.stiffness_ratio == pytest.approx(blast.H_over_B, rel=1e-9)


def test_a_broken_reconstruction_is_caught():
    """The gate must fail when the arithmetic changes, or it is decoration.

    Perturbing one site's diameter by 10 percent moves every dimension there by 10 percent, which is
    far outside any published range.
    """
    import dataclasses

    original = SITE_GEOMETRY["Murgul"]
    SITE_GEOMETRY["Murgul"] = dataclasses.replace(original, hole_diameter_mm=181.5)
    try:
        with pytest.raises(AssertionError, match="contradicts the published narrative"):
            bf.verify_reconstruction(bf.load_training_corpus())
    finally:
        SITE_GEOMETRY["Murgul"] = original
