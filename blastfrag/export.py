"""A portable form of every fitted arm, and a reference predictor that reads it.

The learned arms are fitted with numpy, scikit-learn and xgboost. A browser has none of those, and
ONNX Runtime Web has no tree-ensemble kernel, so the only faithful way to run a fitted forest in a
browser is to walk its trees. This module writes each fitted arm as plain JSON: the network's
weights, the support vectors and dual coefficients, and every tree as four flat arrays. It also
provides :func:`predict_portable`, a dependency-free Python walker that reads that JSON, so a
consumer in another language can be tested against a reference that is itself tested against the
original arm.

Two numeric details decide whether a port agrees with the original to the last digit:

* scikit-learn casts inputs to 32-bit floats before comparing them with a tree's 64-bit
  thresholds, and XGBoost compares 32-bit inputs with 32-bit split values and accumulates 32-bit
  leaf values. A port must round the (standardised) input to 32 bits before every tree comparison,
  and for XGBoost accumulate in 32 bits; :func:`f32` is that rounding.
* The plausibility guard and the network's output clamp are part of the arm, so they are part of
  the export, and the walker applies them.

Schema ``blastfrag.portable/v1``. Every document carries ``schema``, ``arm``, ``kind``,
``engine_version``, ``features`` (the input order) and ``plausible_x50_m``; the rest depends on
``kind``.
"""

from __future__ import annotations

import json
import math
import re
import struct
from typing import Mapping, Sequence

from .models import (
    DISCRIMINANT_BOUNDARY,
    DISCRIMINANT_COEFFICIENTS,
    DISCRIMINANT_CONSTANT,
    PLAUSIBLE_X50_M,
    Arm,
    KuznetsovTransfer,
    RefittedRegression,
)
from .types import FEATURES, Blast

__all__ = ["PORTABLE_SCHEMA", "export_arm", "predict_portable", "f32", "PortableError"]

PORTABLE_SCHEMA = "blastfrag.portable/v1"


class PortableError(ValueError):
    """Raised when an arm cannot be exported, or a document cannot be read."""


def f32(value: float) -> float:
    """Round a Python float to the nearest 32-bit float, as ``Math.fround`` does in a browser.

    Like ``Math.fround``, a magnitude beyond the 32-bit range becomes an infinity rather than an
    error, and a NaN stays a NaN.
    """
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

def _router() -> dict[str, object]:
    return {
        "coefficients": [DISCRIMINANT_COEFFICIENTS[name] for name in FEATURES],
        "constant": DISCRIMINANT_CONSTANT,
        "boundary": DISCRIMINANT_BOUNDARY,
        "rule": "group 1 when the score is strictly above the boundary, else group 2",
    }


def _sklearn_tree(tree) -> dict[str, list]:
    """One fitted scikit-learn regression tree as four flat arrays.

    ``t`` holds the threshold at an internal node and the leaf value at a leaf; a node is a leaf
    when ``left`` is -1.
    """
    t = tree.tree_
    left = [int(v) for v in t.children_left]
    right = [int(v) for v in t.children_right]
    feature = [int(v) if left[i] != -1 else -1 for i, v in enumerate(t.feature)]
    values = [
        float(t.value[i][0][0]) if left[i] == -1 else float(t.threshold[i])
        for i in range(t.node_count)
    ]
    return {"left": left, "right": right, "feature": feature, "t": values}


def _forest(model) -> dict[str, object]:
    return {"trees": [_sklearn_tree(estimator) for estimator in model.estimators_]}


_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _xgboost(model) -> dict[str, object]:
    """One fitted XGBoost regressor as flat arrays, read from the library's own JSON model.

    In that format a leaf's value is stored in ``split_conditions`` at the leaf's index, and the
    model's starting value is ``base_score``, which recent releases write as a bracketed string.
    """
    booster = model.get_booster()
    document = json.loads(booster.save_raw(raw_format="json"))
    learner = document["learner"]
    objective = learner.get("objective", {}).get("name")
    if objective not in {"reg:squarederror", None}:
        raise PortableError(f"only squared-error regression is supported, got {objective!r}")
    match = _FLOAT.search(str(learner["learner_model_param"]["base_score"]))
    if match is None:
        raise PortableError("the boosting model carries no readable base score")
    trees = []
    for tree in learner["gradient_booster"]["model"]["trees"]:
        left = [int(v) for v in tree["left_children"]]
        right = [int(v) for v in tree["right_children"]]
        feature = [int(v) if left[i] != -1 else -1 for i, v in enumerate(tree["split_indices"])]
        trees.append(
            {
                "left": left,
                "right": right,
                "feature": feature,
                # The JSON prints each 32-bit value as a short decimal, which parses to a nearby
                # 64-bit float. Rounding it back is what makes an input equal to a split value land
                # on the same side as it does inside the library: without it, 3 of 100 trees took
                # the other branch on one blast.
                "t": [f32(float(v)) for v in tree["split_conditions"]],
            }
        )
    return {"base_score": f32(float(match.group(0))), "trees": trees}


def _standardise(arm) -> dict[str, list[float]]:
    return {"mean": list(arm.mean), "sd": list(arm.sd)}


def export_arm(arm: Arm, *, engine_version: str | None = None) -> dict[str, object]:
    """The portable JSON document for one FITTED arm.

    Supported: the published network, both support-vector variants, the random forest, gradient
    boosting, the stacking ensemble, the refitted power law and the transfer form of the classical
    equation. Raises :class:`PortableError` for anything else or for an arm that was never fitted.
    """
    from . import __version__
    from .learned import (
        GradientBoosting,
        PublishedNeuralNetwork,
        RandomForest,
        StackingEnsemble,
        SupportVectorRegression,
    )

    base: dict[str, object] = {
        "schema": PORTABLE_SCHEMA,
        "arm": arm.name,
        "engine_version": engine_version or __version__,
        "features": list(FEATURES),
        "plausible_x50_m": list(PLAUSIBLE_X50_M),
        "provenance": arm.provenance(),
    }

    if isinstance(arm, PublishedNeuralNetwork):
        if not arm.networks:
            raise PortableError("the network has not been fitted")
        groups = {}
        for group, networks in arm.networks.items():
            low, high = arm.target_range[group]
            groups[str(group)] = {
                "minimum": list(arm.scalers[group].minimum),
                "maximum": list(arm.scalers[group].maximum),
                "target_low": low,
                "target_high": high,
                "n_hidden": arm.hidden[group],
                "networks": [list(map(float, net.weights)) for net in networks],
            }
        return base | {
            "kind": "network",
            "router": _router(),
            "activation": "logistic",
            "layout": "W1 row-major (inputs by hidden), b1, W2, b2",
            "output": "each network's output clamped to [0, 1], rescaled to the target range, averaged",
            "groups": groups,
        }

    if isinstance(arm, (SupportVectorRegression, RandomForest, GradientBoosting, StackingEnsemble)):
        if arm.model is None:
            raise PortableError(f"{arm.name} has not been fitted")
        common = base | {"standardise": _standardise(arm)}
        if isinstance(arm, SupportVectorRegression):
            model = arm.model
            return common | {
                "kind": "svr",
                "kernel": model.kernel,
                "gamma": float(model._gamma),
                "degree": int(model.degree),
                "coef0": float(model.coef0),
                "support_vectors": [list(map(float, row)) for row in model.support_vectors_],
                "dual_coef": [float(v) for v in model.dual_coef_[0]],
                "intercept": float(model.intercept_[0]),
            }
        if isinstance(arm, RandomForest):
            return common | {"kind": "forest", **_forest(arm.model)}
        if isinstance(arm, GradientBoosting):
            return common | {"kind": "xgboost", **_xgboost(arm.model)}
        stack = arm.model
        return common | {
            "kind": "stacking",
            "forest": _forest(stack.forest),
            "boosting": _xgboost(stack.boosting),
            "meta": {
                "coef": [float(v) for v in stack.meta.coef_],
                "intercept": float(stack.meta.intercept_),
                "inputs": ["forest", "boosting"],
            },
        }

    if isinstance(arm, RefittedRegression):
        if not arm.coefficients:
            raise PortableError("the refitted regression has not been fitted")
        return base | {
            "kind": "power-law",
            "router": _router(),
            "groups": {
                str(group): {
                    "intercept": c.intercept,
                    "exponents": [c.exponents[name] for name in FEATURES],
                }
                for group, c in arm.coefficients.items()
            },
        }

    if isinstance(arm, KuznetsovTransfer):
        if arm.intercept is None:
            raise PortableError("the transfer rock factor has not been fitted")
        return base | {
            "kind": "kuznetsov-transfer",
            "rock_factor": {
                "intercept": arm.intercept,
                "slope": arm.slope,
                "form": "ln A = intercept + slope * ln E",
                "fit_sites": list(arm.fit_sites),
            },
            "timing_factor": arm.timing_factor,
            "note": (
                "needs absolute geometry; the portable reader returns the rock factor and leaves the "
                "mean-size equation to the consumer's own classical implementation"
            ),
        }

    raise PortableError(f"{arm.name} has no portable form")


# ---------------------------------------------------------------------------------------------
# Reference predictor
# ---------------------------------------------------------------------------------------------

def _features(blast_or_features: Blast | Mapping[str, float] | Sequence[float]) -> list[float]:
    if isinstance(blast_or_features, Blast):
        return list(blast_or_features.features())
    if isinstance(blast_or_features, Mapping):
        return [float(blast_or_features[name]) for name in FEATURES]
    values = [float(v) for v in blast_or_features]
    if len(values) != len(FEATURES):
        raise PortableError(f"expected {len(FEATURES)} features, got {len(values)}")
    return values


def _group(router: Mapping, x: Sequence[float]) -> int:
    score = sum(c * v for c, v in zip(router["coefficients"], x)) + router["constant"]
    return 1 if score > router["boundary"] else 2


def _walk(tree: Mapping, z32: Sequence[float]) -> float:
    left, right, feature, t = tree["left"], tree["right"], tree["feature"], tree["t"]
    node = 0
    while left[node] != -1:
        node = left[node] if z32[feature[node]] <= t[node] else right[node]
    return t[node]


def _walk_xgb(tree: Mapping, z32: Sequence[float]) -> float:
    left, right, feature, t = tree["left"], tree["right"], tree["feature"], tree["t"]
    node = 0
    while left[node] != -1:
        node = left[node] if z32[feature[node]] < t[node] else right[node]
    return t[node]


def _forest_value(forest: Mapping, z32: Sequence[float]) -> float:
    total = 0.0
    for tree in forest["trees"]:
        total += _walk(tree, z32)
    return total / len(forest["trees"])


def _xgb_value(boosting: Mapping, z32: Sequence[float]) -> float:
    total = f32(boosting["base_score"])
    for tree in boosting["trees"]:
        total = f32(total + _walk_xgb(tree, z32))
    return total


def _logistic(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def predict_portable(
    document: Mapping, blast_or_features: Blast | Mapping[str, float] | Sequence[float]
) -> tuple[float | None, str | None]:
    """Predict the mean fragment size, in metres, from a portable document.

    Returns ``(value, None)``, or ``(None, reason)`` where the arm would have abstained. For a
    ``kuznetsov-transfer`` document the value returned is the ROCK FACTOR, not a size: the classical
    equation needs absolute geometry, which the seven ratios do not carry.
    """
    if document.get("schema") != PORTABLE_SCHEMA:
        raise PortableError(f"unknown schema {document.get('schema')!r}")
    x = _features(blast_or_features)
    kind = document["kind"]

    if kind == "kuznetsov-transfer":
        rf = document["rock_factor"]
        return math.exp(rf["intercept"] + rf["slope"] * math.log(x[FEATURES.index("E_GPa")])), None

    if kind == "power-law":
        group = _group(document["router"], x)
        coefficients = document["groups"].get(str(group))
        if coefficients is None:
            return None, f"no refitted equation for group {group}"
        value = coefficients["intercept"]
        for v, exponent in zip(x, coefficients["exponents"]):
            value *= v ** exponent
    elif kind == "network":
        group = _group(document["router"], x)
        spec = document["groups"].get(str(group))
        if spec is None:
            return None, f"no network was trained for group {group}"
        scaled = [
            0.5 if hi == lo else (v - lo) / (hi - lo)
            for v, lo, hi in zip(x, spec["minimum"], spec["maximum"])
        ]
        n_in, n_h = len(x), spec["n_hidden"]
        cut = n_in * n_h
        outputs = []
        for w in spec["networks"]:
            hidden = [
                _logistic(sum(scaled[i] * w[i * n_h + j] for i in range(n_in)) + w[cut + j])
                for j in range(n_h)
            ]
            raw = sum(h * w[cut + n_h + j] for j, h in enumerate(hidden)) + w[cut + 2 * n_h]
            clamped = min(1.0, max(0.0, raw))
            outputs.append(clamped * (spec["target_high"] - spec["target_low"]) + spec["target_low"])
        value = sum(outputs) / len(outputs)
    else:
        mean, sd = document["standardise"]["mean"], document["standardise"]["sd"]
        z = [(v - m) / (s if s > 0 else 1.0) for v, m, s in zip(x, mean, sd)]
        z32 = [f32(v) for v in z]
        if kind == "svr":
            gamma, degree, coef0 = document["gamma"], document["degree"], document["coef0"]
            total = document["intercept"]
            for coefficient, sv in zip(document["dual_coef"], document["support_vectors"]):
                if document["kernel"] == "rbf":
                    k = math.exp(-gamma * sum((a - b) ** 2 for a, b in zip(sv, z)))
                elif document["kernel"] == "poly":
                    k = (gamma * sum(a * b for a, b in zip(sv, z)) + coef0) ** degree
                else:
                    raise PortableError(f"unsupported kernel {document['kernel']!r}")
                total += coefficient * k
            value = total
        elif kind == "forest":
            value = _forest_value(document, z32)
        elif kind == "xgboost":
            value = _xgb_value(document, z32)
        elif kind == "stacking":
            forest = _forest_value(document["forest"], z32)
            boosting = _xgb_value(document["boosting"], z32)
            meta = document["meta"]
            value = meta["coef"][0] * forest + meta["coef"][1] * boosting + meta["intercept"]
        else:
            raise PortableError(f"unknown kind {kind!r}")

    low, high = document["plausible_x50_m"]
    if not math.isfinite(value) or not low <= value <= high:
        return None, (
            f"the model returned {value:.4g} m, which is outside the plausible fragment range "
            f"{low} to {high} m"
        )
    return value, None
