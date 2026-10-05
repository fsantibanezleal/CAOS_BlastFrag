# Design: the data and the ingestion contract

Retroactive. `blastfrag/datasets.py` reads the shipped CSVs (package data, each with a provenance header), builds
`Blast` records and runs `check_source_integrity` against `PUBLISHED_DESCRIPTIVE_STATS` on every load.
`validate_blast` applies `CONTRACT_BOUNDS` (reject) and `TRAINING_ENVELOPE` (reject unless
`allow_extrapolation`). `blastfrag/geometry.py` turns the corpus ratios into an absolute pattern from the per-site
diameter registry `SITE_GEOMETRY`, and `verify_reconstruction` checks every narrative constraint. Data flow: CSV,
integrity gate, `Blast`, contract, geometry, arms. The user-facing page is `docs/data.md`.
