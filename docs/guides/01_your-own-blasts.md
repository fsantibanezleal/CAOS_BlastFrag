# Predicting your own blasts

How to put a bench design of your own through the ladder, and how to read what comes back.

---

## 1. Describe the blast

The statistical and learned arms take the seven ratios the corpus uses; the classical arms also need
the absolute pattern. A `Blast` carries the ratios, and `reconstruct_pattern` turns them into metres
when the hole diameter is known.

```python
import blastfrag as bf

blast = bf.Blast(
    blast_id="my-001", site="my-mine",
    S_over_B=1.20, H_over_B=3.0, B_over_D=27.0, T_over_B=1.0,
    Pf_kg_m3=0.55, XB_m=1.0, E_GPa=30.0,
    meta={"hole_diameter_mm": 165.0},
)
warnings = bf.validate_blast(blast, allow_extrapolation=True)   # [] inside the corpus envelope
```

`validate_blast` rejects a value that cannot be a blast (outside the contract bounds) and, unless you
pass `allow_extrapolation=True`, a value outside the published envelope of the 97 training blasts. The
envelope is the training range of each feature: `S/B` 1.00 to 1.75, `H/B` 1.33 to 6.82, `B/D` 17.98 to
39.47, `T/B` 0.50 to 4.67, `Pf` 0.22 to 1.26 kg/m3, `XB` 0.02 to 2.35 m, `E` 9.57 to 60 GPa.

## 2. Choose the arms that fit your question

| You want | Use | Why |
|---|---|---|
| a mean size at a site none of the corpus campaigns resembles | `KuznetsovTransfer` fitted on the corpus | its only fitted quantity is a rock-factor line over nine sites, and it scored 0.298 held out by site |
| a full passing curve | `KuzRam` or `Swebrec` | the shape around the classical mean size; unvalidated, see [distributions](../methods/02_distributions.md) |
| the published regression's answer | `PublishedRegression` | in sample for the corpus; its out-of-sample evidence is the two published hold-outs from the same sites |
| a learned prediction | any learned arm, fitted on the corpus | their random-split scores (0.66 to 0.75) do not survive a site hold-out; treat them as interpolators between the corpus campaigns |
| a reference | `NullModel` fitted on the corpus | predicts 0.304 m for everything; any arm that cannot beat it on your data is not helping |

```python
train = bf.load_training_corpus()
transfer = bf.KuznetsovTransfer().fit(train)
prediction = transfer.predict_one(blast)
prediction.x50_m, prediction.detail["rock_factor"]
```

## 3. Read the answer

- A `Prediction` has a value or an `abstain_reason`, never both. A refusal says which input was
  missing or which range was left.
- `extrapolated` is set when the blast lies outside the training envelope.
- The `detail` of the classical arms carries the rock factor and where it came from, the rock volume
  and the charge mass.
- Every number this package returns as a score names its statistic. "Variance explained" is $R^2$
  about the identity line, not a squared correlation.

## 4. Score your own campaign

If your blasts are measured, score every arm on them and compare with the null:

```python
mine = [...]   # your Blast objects with x50_m set
for name, factory in bf.default_arms().items():
    arm = factory().fit(train)
    print(name, bf.score(mine, arm.predict(mine)).summary())
```

With one campaign, the uncertainty is the uncertainty of one site: `bf.bootstrap_interval(...,
unit="blast")` over your rows understates it, because nothing in one campaign tells you how a second
campaign would differ.
