"""The shipped corpora must be the published tables, and must stay that way."""

from __future__ import annotations

import pytest

import blastfrag as bf
from blastfrag.datasets import (
    DATASET_DIGEST,
    PUBLISHED_DESCRIPTIVE_STATS,
    check_source_integrity,
    compute_dataset_digest,
)


def test_training_corpus_is_97_blasts_over_10_sites():
    train = bf.load_training_corpus()
    assert len(train) == 97
    assert len({b.site for b in train}) == 10
    # The published group split, Hudaverdi 2010 section 4.
    assert sum(1 for b in train if b.group == 1) == 35
    assert sum(1 for b in train if b.group == 2) == 62


def test_every_training_blast_has_a_measured_target():
    assert all(b.x50_m is not None for b in bf.load_training_corpus())


def test_source_integrity_gate_passes_on_the_shipped_corpus():
    check_source_integrity(bf.load_training_corpus(verify=False))


def test_dataset_digest_is_pinned():
    assert compute_dataset_digest(bf.load_training_corpus(verify=False)) == DATASET_DIGEST


def test_source_integrity_gate_catches_the_defect_it_was_written_for():
    """The original defect: one row's block size copied into its powder-factor column.

    The corpus shipped with a powder factor of 1.47 on Ad19 where the paper prints 1.26, and the
    tell was the paper's own descriptive-statistics table giving a maximum of 1.26. Reintroducing
    that exact defect must fail the gate, or the gate is decoration.
    """
    import dataclasses

    train = bf.load_training_corpus(verify=False)
    corrupted = [
        dataclasses.replace(b, Pf_kg_m3=1.47) if b.blast_id == "Ad19" else b for b in train
    ]
    with pytest.raises(bf.ContractViolation) as excinfo:
        check_source_integrity(corrupted)
    message = str(excinfo.value)
    assert "Pf_kg_m3.max" in message
    assert "1.26" in message


def test_source_integrity_gate_catches_a_single_cell_change_the_statistics_would_miss():
    """A one-row correction moves a mean over 97 rows by less than a rounding step.

    The digest is what catches it, which is why both halves of the gate exist.
    """
    import dataclasses

    train = bf.load_training_corpus(verify=False)
    corrupted = [
        dataclasses.replace(b, Pf_kg_m3=0.33) if b.blast_id == "Db5" else b for b in train
    ]
    # The statistics alone would let this through, which is the point.
    check_source_integrity(corrupted, check_digest=False)
    with pytest.raises(bf.ContractViolation, match="content digest"):
        check_source_integrity(corrupted)


def test_published_minima_and_maxima_reproduce_exactly():
    train = bf.load_training_corpus()
    for name, published in PUBLISHED_DESCRIPTIVE_STATS.items():
        values = [getattr(b, name) for b in train]
        assert min(values) == pytest.approx(published.minimum, abs=1e-9)
        assert max(values) == pytest.approx(published.maximum, abs=1e-9)


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [("2012", 12), ("2010", 13), ("union", 14), ("clean", 13)],
)
def test_holdout_protocols_have_their_published_sizes(protocol, expected):
    assert len(bf.load_holdout(protocol=protocol)) == expected


def test_the_leaked_blast_is_present_and_flagged_in_the_2012_protocol():
    """Rc1 is in the 2012 validation set and also in the 35-blast training table.

    It is carried rather than quietly dropped, because reproducing the published protocol means
    reproducing its defect. What must never happen is carrying it without the flag.
    """
    holdout = {b.blast_id: b for b in bf.load_holdout(protocol="2012")}
    assert "Rc1" in holdout
    assert holdout["Rc1"].meta["in_training_table"] is True
    assert all(
        b.meta["in_training_table"] is False for b in holdout.values() if b.blast_id != "Rc1"
    )

    training_ids = {b.blast_id for b in bf.load_training_corpus()}
    assert "Rc1" in training_ids, "the leakage claim depends on Rc1 being in the training table"

    assert "Rc1" not in {b.blast_id for b in bf.load_holdout(protocol="clean")}


def test_the_2010_and_2012_holdouts_differ_exactly_as_recorded():
    ids_2010 = {b.blast_id for b in bf.load_holdout(protocol="2010")}
    ids_2012 = {b.blast_id for b in bf.load_holdout(protocol="2012")}
    assert ids_2012 - ids_2010 == {"Rc1"}
    assert ids_2010 - ids_2012 == {"Mi7", "Ad25"}


def test_the_two_papers_disagree_on_their_own_regression_predictions():
    """Both papers print the same regression equations and different answers on five rows.

    Recorded as a test so that a future edit to the shipped table cannot quietly erase the
    disagreement, which is one of the product's findings. Four of the five resolve in the 2010
    paper's favour when the published equation is recomputed; Ad24 matches neither, which is
    checked in the regression test module.
    """
    disagree = set()
    for blast in bf.load_holdout(protocol="union"):
        published = blast.meta["published"]
        a, b = published["regression_2010"], published["regression_2012"]
        if a is not None and b is not None and a != b:
            disagree.add(blast.blast_id)
    assert disagree == {"Mr12", "Sm8", "Oz9", "Ad23", "Ad24"}


def test_the_two_papers_disagree_on_their_classical_predictions_too():
    disagree = {
        b.blast_id
        for b in bf.load_holdout(protocol="union")
        if (p := b.meta["published"])["kuzram_2010"] is not None
        and p["kuzram_2012"] is not None
        and p["kuzram_2010"] != p["kuzram_2012"]
    }
    assert disagree == {"En13", "Ru7", "Mg8", "Mg9", "Mr12", "Db10", "Sm8", "Oz8", "Oz9", "Ad23"}


def test_the_holdout_shares_no_blast_id_with_training_except_the_known_leak():
    training_ids = {b.blast_id for b in bf.load_training_corpus()}
    overlap = {b.blast_id for b in bf.load_holdout(protocol="union")} & training_ids
    assert overlap == {"Rc1"}


def test_duplicate_feature_vectors_in_the_corpus_are_exactly_those_recorded():
    """17 of 97 rows duplicate another row's feature vector, in 7 groups.

    This is the fact that makes a random split leak, and it is the basis of the protocol
    sensitivity experiment, so it is pinned.
    """
    from collections import Counter

    train = bf.load_training_corpus()
    counts = Counter(b.features() for b in train)
    duplicated = {k: v for k, v in counts.items() if v > 1}
    assert len(duplicated) == 7
    assert sum(duplicated.values()) == 17

    groups = sorted(
        sorted(b.blast_id for b in train if b.features() == key) for key in duplicated
    )
    assert groups == [
        ["En1", "En2"],
        ["En11", "En12"],
        ["En3", "En5"],
        ["En8", "En9"],
        ["Mr1", "Mr3"],
        ["Sm1", "Sm2", "Sm3"],
        ["Sm4", "Sm5", "Sm6", "Sm7"],
    ]


def test_field_holdout_is_an_extrapolation_and_says_so():
    field = bf.load_field_holdout()
    assert len(field) == 5
    low, _high = bf.TRAINING_ENVELOPE["E_GPa"]
    assert all(b.E_GPa < low for b in field), "the whole point of this set is that it is outside"
    assert all(b.meta["extrapolated"] for b in field)
    # It publishes its absolute pattern, so it can check the reconstruction path rather than use it.
    assert all("absolute" in b.meta for b in field)


def test_the_contract_refuses_an_out_of_envelope_row_unless_asked():
    field = bf.load_field_holdout()[0]
    with pytest.raises(bf.ContractViolation, match="outside the fitted envelope"):
        bf.validate_blast(field, allow_extrapolation=False)
    warnings = bf.validate_blast(field, allow_extrapolation=True)
    assert any("E_GPa" in w for w in warnings)


def test_the_contract_rejects_a_unit_error_outright():
    """A modulus in MPa rather than GPa is not an unusual blast, it is a broken row."""
    import dataclasses

    blast = dataclasses.replace(bf.load_training_corpus()[0], E_GPa=60000.0)
    with pytest.raises(bf.ContractViolation, match="outside the contract bounds"):
        bf.validate_blast(blast, allow_extrapolation=True)


def test_envelope_report_covers_every_feature():
    report = bf.envelope_report(bf.load_training_corpus())
    assert set(report) == set(bf.FEATURES)
    # The training corpus is the envelope, so it covers all of it and sits none outside.
    for name, stats in report.items():
        assert stats["envelope_coverage"] == pytest.approx(1.0, abs=1e-9), name
        assert stats["n_outside_envelope"] == 0, name
