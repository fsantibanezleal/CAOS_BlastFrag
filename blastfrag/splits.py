"""Split protocols, and the leakage guards that make them mean something.

The 2025 state of the art on this corpus reports its headline from a **random** 80/20 split of 97
rows, 17 of which duplicate another row's feature vector. A random draw puts duplicates on both sides
by construction. The same paper records that cross-validation was tried and then removed because "the
cross-validated model had a poor prediction effect on the test set", which is the symptom this
predicts.

Rather than argue about which protocol is right, this module implements three and the package scores
every learned arm under all of them. The effect size between protocols is the finding, and either
sign of it is worth reporting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .types import Blast

__all__ = [
    "Split",
    "LeakageError",
    "random_split",
    "deduplicated_split",
    "leave_one_site_out",
    "all_protocols",
    "assert_no_leakage",
    "duplicate_groups",
]


class LeakageError(AssertionError):
    """Raised when a split puts the same information on both sides."""


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition, carrying the protocol that produced it."""

    protocol: str
    train: tuple[Blast, ...]
    test: tuple[Blast, ...]
    seed: int | None = None
    note: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train or not self.test:
            raise ValueError(f"{self.protocol}: a split needs rows on both sides")


def duplicate_groups(blasts: Sequence[Blast]) -> list[list[str]]:
    """Blast identifiers grouped by identical feature vector, groups of two or more only.

    On the shipped corpus this returns seven groups covering seventeen rows. That fact is the reason
    the deduplicated protocol exists.
    """
    by_features: dict[tuple[float, ...], list[str]] = {}
    for blast in blasts:
        by_features.setdefault(blast.features(), []).append(blast.blast_id)
    return sorted((ids for ids in by_features.values() if len(ids) > 1), key=lambda g: g[0])


def assert_no_leakage(split: Split, *, allow_duplicate_features: bool = True) -> None:
    """Check a split for the three ways information crosses the partition.

    Identifier overlap is always an error. Feature-vector overlap is an error only when the protocol
    claims to have removed it, which is why ``allow_duplicate_features`` defaults to permitting it:
    the random protocol reproduces a published method and its duplicates are the point.

    Site overlap is checked by the leave-one-site-out builder itself.
    """
    train_ids = {b.blast_id for b in split.train}
    test_ids = {b.blast_id for b in split.test}
    shared_ids = train_ids & test_ids
    if shared_ids:
        raise LeakageError(
            f"{split.protocol}: {len(shared_ids)} blast(s) on both sides: {sorted(shared_ids)}"
        )

    if not allow_duplicate_features:
        train_features = {b.features() for b in split.train}
        crossing = [b.blast_id for b in split.test if b.features() in train_features]
        if crossing:
            raise LeakageError(
                f"{split.protocol} claims to remove duplicate feature vectors, but "
                f"{len(crossing)} test row(s) have inputs identical to a training row: "
                f"{sorted(crossing)}"
            )


def random_split(
    blasts: Sequence[Blast], *, test_fraction: float = 0.2, seed: int = 0
) -> Split:
    """A seeded random partition, reproducing the published protocol.

    This one **does** leak, and deliberately: it is here so the published result can be reproduced
    and so the gap to the leakage-free protocols can be measured rather than asserted. The note on
    the returned split says how many test rows share a feature vector with a training row.
    """
    rows = list(blasts)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(round(len(rows) * test_fraction)))
    test, train = rows[:cut], rows[cut:]

    train_features = {b.features() for b in train}
    crossing = [b.blast_id for b in test if b.features() in train_features]

    split = Split(
        protocol="random-8020",
        train=tuple(train),
        test=tuple(test),
        seed=seed,
        note=(
            f"{len(crossing)} of {len(test)} test rows share a feature vector with a training row"
            if crossing
            else "no duplicate feature vector crossed this particular draw"
        ),
        detail={"crossing_ids": sorted(crossing), "test_fraction": test_fraction},
    )
    assert_no_leakage(split, allow_duplicate_features=True)
    return split


def deduplicated_split(
    blasts: Sequence[Blast], *, test_fraction: float = 0.2, seed: int = 0
) -> Split:
    """A random partition over collapsed duplicates, so no feature vector crosses.

    Duplicate rows are collapsed to one representative before splitting, and the representative's
    measured size is the mean of its group. That averaging is a choice worth stating: duplicated
    inputs in this corpus carry genuinely different measured outcomes (four Soma blasts share one
    feature vector and their sizes range over 0.22 to 0.28 m), which is measurement scatter rather
    than a modelling target, and averaging it keeps one row per distinct design.
    """
    import dataclasses

    by_features: dict[tuple[float, ...], list[Blast]] = {}
    for blast in blasts:
        by_features.setdefault(blast.features(), []).append(blast)

    collapsed: list[Blast] = []
    for group in by_features.values():
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        sizes = [b.x50_m for b in group if b.x50_m is not None]
        collapsed.append(
            dataclasses.replace(
                group[0],
                x50_m=sum(sizes) / len(sizes) if sizes else None,
                blast_id="+".join(b.blast_id for b in group),
                meta=group[0].meta | {"collapsed_from": [b.blast_id for b in group]},
            )
        )

    rows = list(collapsed)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(round(len(rows) * test_fraction)))
    split = Split(
        protocol="dedup-random",
        train=tuple(rows[cut:]),
        test=tuple(rows[:cut]),
        seed=seed,
        note=(
            f"{len(blasts)} rows collapsed to {len(collapsed)} distinct feature vectors before "
            "splitting"
        ),
        detail={"n_collapsed": len(blasts) - len(collapsed), "test_fraction": test_fraction},
    )
    assert_no_leakage(split, allow_duplicate_features=False)
    return split


def leave_one_site_out(blasts: Sequence[Blast]) -> Iterator[Split]:
    """One split per source site, holding that whole site out.

    This is the protocol that respects the dependence in this corpus, and the one a benchmark should
    lead with. Rows within a campaign share a rock mass, a drilling rig, an explosive supply and a
    measurement operator, so they are not independent draws. One quarry supplies 22 of the 97 rows.

    It is also the protocol that answers the question a practitioner actually has, which is not "how
    well does this model interpolate between blasts I have already measured" but "how well does it
    transfer to my mine".
    """
    by_site: dict[str, list[Blast]] = {}
    for blast in blasts:
        by_site.setdefault(blast.site, []).append(blast)

    for site in sorted(by_site):
        test = by_site[site]
        train = [b for b in blasts if b.site != site]
        if not train or not test:
            continue
        split = Split(
            protocol=f"leave-out-{site}",
            train=tuple(train),
            test=tuple(test),
            seed=None,
            note=f"{len(test)} blasts from {site} held out; no row from that site is in training",
            detail={"held_out_site": site, "n_sites_in_train": len({b.site for b in train})},
        )
        assert_no_leakage(split, allow_duplicate_features=False)
        if {b.site for b in split.train} & {site}:
            raise LeakageError(f"{split.protocol}: the held-out site appears in training")
        yield split


def all_protocols(blasts: Sequence[Blast], *, seed: int = 0) -> dict[str, list[Split]]:
    """Every protocol, so a benchmark shows the gap rather than choosing a side."""
    return {
        "random-8020": [random_split(blasts, seed=seed)],
        "dedup-random": [deduplicated_split(blasts, seed=seed)],
        "leave-one-site-out": list(leave_one_site_out(blasts)),
    }
