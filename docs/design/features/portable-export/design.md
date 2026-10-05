# Design: the portable export

Retroactive. `export_arm` (`blastfrag/export.py`) writes a fitted arm as a JSON document of schema
`blastfrag.portable/v1`: the input order, the scalers, and the model itself (network weights, support vectors and
coefficients, or each tree as four arrays: left, right, feature, threshold or leaf). `predict_portable` is a
reference walker with no dependency beyond the standard library, so a consumer in another language has a
specification to hold its own walker to. Tree inputs are rounded to single precision because XGBoost compares in
32 bits. The page is `docs/data/03_portable-export.md`.
