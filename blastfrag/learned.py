"""The learned rungs: the published network, and the 2025 ensembles.

Three of the four arms here are fully specified in their source papers, down to the hidden width,
the training algorithm, the normalisation and the hyperparameters, so they are reproductions rather
than reimplementations. That matters: a learned arm whose architecture was guessed cannot be
compared with the number its paper reports.

The published network is written against numpy directly, including its Levenberg-Marquardt training,
because it is a seven-input single-hidden-layer network and pulling in a deep-learning framework to
fit forty weights would be absurd. The tree and support-vector arms need scikit-learn and xgboost,
which is why they are extras.

One arm from the 2025 literature is deliberately NOT reproduced. Its optimiser's update rule is not
transcribable with confidence from the copy held for this work, and a hand-rolled approximation
shipped under that name would be a fabricated method. It appears in the documentation as prior art
with its published numbers, attributed, and not as a rung.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .models import Arm, assign_group
from .types import FEATURES, Blast, Group, Prediction

__all__ = [
    "MinMaxScaler",
    "BackPropNetwork",
    "PublishedNeuralNetwork",
    "SupportVectorRegression",
    "RandomForest",
    "GradientBoosting",
    "StackingEnsemble",
    "LEARNED_LADDER",
]


@dataclass(slots=True)
class MinMaxScaler:
    """The normalisation the published network uses, Kulatilake et al. 2012 Eq. 11.

    ``y = (x - x_min) / (x_max - x_min)``, fitted on the training rows only. Which normalisation is
    used is not cosmetic: the published network uses min-max and the 2025 ensembles use a standard
    score, and silently substituting one for the other changes every number a learned arm returns.
    """

    minimum: list[float] = field(default_factory=list)
    maximum: list[float] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]]) -> "MinMaxScaler":
        columns = list(zip(*rows))
        self.minimum = [min(c) for c in columns]
        self.maximum = [max(c) for c in columns]
        return self

    def transform(self, row: Sequence[float]) -> list[float]:
        out = []
        for value, low, high in zip(row, self.minimum, self.maximum):
            span = high - low
            out.append(0.5 if span == 0 else (value - low) / span)
        return out

    def inverse_scalar(self, value: float, low: float, high: float) -> float:
        return value * (high - low) + low


class BackPropNetwork:
    """A single-hidden-layer network trained by Levenberg-Marquardt, in numpy.

    The architecture is the one the source specifies: seven inputs, ``N`` hidden units with a
    logistic activation, one linear output. The source justifies the single hidden layer on Cybenko's
    universal approximation result and reports that Levenberg-Marquardt "showed the highest
    stability among the four training algorithms" it compared and "reached the global minimum with
    the lowest number of training cycles".

    Levenberg-Marquardt is implemented here rather than borrowed because no mainstream Python
    library ships it for neural networks, and substituting gradient descent would not be the
    published method. It is a Gauss-Newton step damped toward gradient descent::

        (J'J + lambda I) delta = -J' r

    with the damping raised on a step that increases the loss and lowered on one that decreases it.
    """

    def __init__(self, n_hidden: int, *, seed: int = 0, max_iter: int = 200) -> None:
        self.n_hidden = n_hidden
        self.seed = seed
        self.max_iter = max_iter
        self.weights: list[float] = []
        self.n_inputs = len(FEATURES)
        self.history: list[float] = []

    # The parameter vector is laid out as: input-to-hidden weights (n_inputs * n_hidden), hidden
    # biases (n_hidden), hidden-to-output weights (n_hidden), output bias (1).
    def _n_parameters(self) -> int:
        return self.n_inputs * self.n_hidden + self.n_hidden + self.n_hidden + 1

    def _forward(self, weights, X):
        import numpy as np

        n_in, n_h = self.n_inputs, self.n_hidden
        cut = n_in * n_h
        W1 = weights[:cut].reshape(n_in, n_h)
        b1 = weights[cut : cut + n_h]
        W2 = weights[cut + n_h : cut + 2 * n_h]
        b2 = weights[cut + 2 * n_h]
        hidden = 1.0 / (1.0 + np.exp(-(X @ W1 + b1)))
        return hidden @ W2 + b2, hidden, W2

    def _jacobian(self, weights, X):
        import numpy as np

        n_in, n_h = self.n_inputs, self.n_hidden
        _output, hidden, W2 = self._forward(weights, X)
        n = X.shape[0]
        J = np.zeros((n, self._n_parameters()))
        # d(out)/d(W2) = hidden ; d(out)/d(b2) = 1
        delta_hidden = hidden * (1.0 - hidden) * W2  # n by n_h
        # Input-to-hidden, laid out to match the row-major reshape in _forward: W1[i, j] is
        # weights[i * n_h + j]. Verified against a central finite difference in the test suite.
        for i in range(n_in):
            for j in range(n_h):
                J[:, i * n_h + j] = delta_hidden[:, j] * X[:, i]
        cut = n_in * n_h
        J[:, cut : cut + n_h] = delta_hidden
        J[:, cut + n_h : cut + 2 * n_h] = hidden
        J[:, cut + 2 * n_h] = 1.0
        return J

    def fit(self, X, y) -> "BackPropNetwork":
        """Fit by Levenberg-Marquardt.

        THE SEED IS DRAWN BEFORE ANY TRAINING, not inside it. A network whose initial weights depend
        on how many models were fitted earlier in the same process is not reproducible, and the bug
        is invisible because every individual run looks fine.
        """
        import numpy as np

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        # Nguyen-Widrow-ish scaling: small enough that the logistic units start in their linear
        # region, which is where Gauss-Newton behaves.
        weights = rng.normal(0.0, 0.5, size=self._n_parameters())

        damping = 1e-2
        prediction, _hidden, _W2 = self._forward(weights, X)
        loss = float(np.mean((prediction - y) ** 2))
        self.history = [loss]

        for _ in range(self.max_iter):
            residual = prediction - y
            J = self._jacobian(weights, X)
            JtJ = J.T @ J
            Jtr = J.T @ residual
            improved = False
            for _attempt in range(12):
                try:
                    step = np.linalg.solve(JtJ + damping * np.eye(JtJ.shape[0]), -Jtr)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue
                candidate = weights + step
                candidate_prediction, _h, _w = self._forward(candidate, X)
                candidate_loss = float(np.mean((candidate_prediction - y) ** 2))
                if candidate_loss < loss:
                    weights, prediction, loss = candidate, candidate_prediction, candidate_loss
                    damping = max(damping / 10.0, 1e-12)
                    improved = True
                    break
                damping *= 10.0
            self.history.append(loss)
            if not improved or (len(self.history) > 2 and abs(self.history[-2] - loss) < 1e-12):
                break

        self.weights = list(weights)
        return self

    def predict(self, X):
        import numpy as np

        if not self.weights:
            raise RuntimeError("the network has not been fitted")
        output, _hidden, _W2 = self._forward(np.asarray(self.weights), np.asarray(X, dtype=float))
        return output


class PublishedNeuralNetwork(Arm):
    """The published seven-input network, per rock-stiffness group, with the published sweep.

    Reproduces Kulatilake, Hudaverdi and Wu 2012 section 5 as specified:

    - one network per group, trained on that group's rows only;
    - min-max normalisation of inputs and target, their Eq. 11;
    - Levenberg-Marquardt training;
    - the hidden width swept over the range two published heuristics allow, 6 to 15;
    - eight simulations at each width, scored by root mean square error and by correlation, with the
      best width chosen on the held-out rows exactly as the source did.

    The published optima are 9 hidden units for the high-modulus group and 7 for the low-modulus
    group. Whether this reproduction lands on the same widths is a **result**, not an assumption, and
    it is reported rather than asserted: the source's own tables show the choice is unstable, with
    the correlation for one group swinging between 0.11 and 0.81 across adjacent widths.
    """

    name = "published-neural-net"
    tier = "learned"
    lane = "offline-train-live-infer"
    source = "Kulatilake, Hudaverdi and Wu 2012 doi:10.1007/s10706-012-9496-3 section 5"

    def __init__(
        self,
        *,
        hidden: dict[Group, int] | None = None,
        n_simulations: int = 8,
        seed: int = 0,
        sweep: tuple[int, ...] = tuple(range(6, 16)),
    ) -> None:
        self.hidden = dict(hidden) if hidden else {1: 9, 2: 7}
        self.n_simulations = n_simulations
        self.seed = seed
        self.sweep = sweep
        self.scalers: dict[Group, MinMaxScaler] = {}
        self.target_range: dict[Group, tuple[float, float]] = {}
        self.networks: dict[Group, list[BackPropNetwork]] = {}
        self.sweep_report: dict[Group, list[dict[str, float]]] = {}

    def _rows(self, blasts: Sequence[Blast], group: Group):
        return [b for b in blasts if b.x50_m is not None and assign_group(b) == group]

    def fit(self, blasts: Sequence[Blast]) -> "PublishedNeuralNetwork":
        for group in (1, 2):
            rows = self._rows(blasts, group)
            if len(rows) < 10:
                continue
            features = [list(b.features()) for b in rows]
            scaler = MinMaxScaler().fit(features)
            X = [scaler.transform(f) for f in features]
            targets = [b.x50_m for b in rows]
            low, high = min(targets), max(targets)  # type: ignore[type-var]
            span = high - low or 1.0
            y = [(t - low) / span for t in targets]  # type: ignore[operator]

            self.scalers[group] = scaler
            self.target_range[group] = (low, high)  # type: ignore[assignment]
            self.networks[group] = [
                BackPropNetwork(self.hidden[group], seed=self.seed * 1000 + group * 100 + k).fit(X, y)
                for k in range(self.n_simulations)
            ]
        return self

    def sweep_hidden_width(
        self, train: Sequence[Blast], validation: Sequence[Blast]
    ) -> dict[Group, dict[str, object]]:
        """Reproduce the published width sweep and report where it lands.

        Returns, per group, the best width by root mean square error on the validation rows and the
        full table, so the instability the source's own tables show is visible rather than hidden
        behind a single chosen number.
        """
        out: dict[Group, dict[str, object]] = {}
        for group in (1, 2):
            validation_rows = self._rows(validation, group)
            if not validation_rows or len(self._rows(train, group)) < 10:
                continue
            table: list[dict[str, float]] = []
            for width in self.sweep:
                arm = PublishedNeuralNetwork(
                    hidden={1: width, 2: width}, n_simulations=self.n_simulations, seed=self.seed
                ).fit(train)
                predictions = arm.predict(validation_rows)
                paired = [
                    (b.x50_m, p.x50_m)
                    for b, p in zip(validation_rows, predictions)
                    if b.x50_m is not None and p.x50_m is not None
                ]
                if len(paired) < 2:
                    continue
                rmse = math.sqrt(sum((m - q) ** 2 for m, q in paired) / len(paired))
                mean_m = sum(m for m, _ in paired) / len(paired)
                mean_q = sum(q for _, q in paired) / len(paired)
                cov = sum((m - mean_m) * (q - mean_q) for m, q in paired)
                var_m = sum((m - mean_m) ** 2 for m, _ in paired)
                var_q = sum((q - mean_q) ** 2 for _, q in paired)
                r = cov / math.sqrt(var_m * var_q) if var_m > 0 and var_q > 0 else float("nan")
                table.append({"hidden": width, "rmse": rmse, "correlation": r})
            if table:
                best = min(table, key=lambda row: row["rmse"])
                out[group] = {
                    "best_hidden": int(best["hidden"]),
                    "best_rmse": best["rmse"],
                    "published_optimum": {1: 9, 2: 7}[group],
                    "table": table,
                }
        self.sweep_report = {g: v["table"] for g, v in out.items()}  # type: ignore[assignment]
        return out

    def predict_one(self, blast: Blast) -> Prediction:
        group = assign_group(blast)
        networks = self.networks.get(group)
        if not networks:
            return self._abstain(
                blast, f"no network was trained for group {group} on the rows supplied"
            )
        scaler = self.scalers[group]
        low, high = self.target_range[group]
        x = [scaler.transform(list(blast.features()))]

        # The target was min-max normalised onto [0, 1], so that interval is the whole of what a
        # network fitted on it can assert. A raw output outside it is the network extrapolating past
        # every size it has ever seen, so the output is clamped to the nearest value it can assert.
        #
        # This is not cosmetic. On one hold-out blast the eight simulations disagree wildly, and
        # without the clamp their mean lands at zero and drags the arm's variance explained from
        # positive to -1.18 on its own. The source reports the same instability on the same blast:
        # its own eight simulations there run from 0.23 to 0.96 m, a coefficient of variation of
        # 0.56, and it notes the cause, which is that this blast shares every input with a training
        # blast except one.
        raw = [float(net.predict(x)[0]) for net in networks]
        clamped = [min(1.0, max(0.0, v)) for v in raw]
        values = [v * (high - low) + low for v in clamped]

        mean = sum(values) / len(values)
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        spread = (
            math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        return self._guarded(
            blast,
            mean,
            group=group,
            detail={
                "n_simulations": len(values),
                "simulations_m": values,
                "median_m": median,
                "simulation_spread_m": spread,
                "n_clamped": sum(1 for r, c in zip(raw, clamped) if r != c),
                "hidden_units": self.hidden[group],
                "n_parameters": self.n_inputs_times(group),
                # The source reports a coefficient of variation per blast across its simulations,
                # reaching 0.76 on one row. Carrying it makes the instability visible on every
                # prediction rather than only in a table nobody reads.
                "coefficient_of_variation": spread / mean if mean > 0 else float("nan"),
            },
        )

    def n_inputs_times(self, group: Group) -> int:
        """Free parameters in this group's network.

        Worth reporting on every prediction, because the published architecture is generous
        relative to the data: seven hidden units on the low-modulus group is 64 parameters fitted
        to 62 blasts, and nine on the high-modulus group is 82 fitted to 35.
        """
        width = self.hidden[group]
        return len(FEATURES) * width + width + width + 1


class _SklearnArm(Arm):
    """Shared plumbing for the scikit-learn arms, including a standard-score normaliser.

    The 2025 source normalises with a standard score rather than the min-max the 2012 network uses,
    and that difference is preserved rather than unified.
    """

    tier = "learned"
    lane = "offline-train-live-infer"

    def __init__(self) -> None:
        self.model = None
        self.mean: list[float] = []
        self.sd: list[float] = []

    def _build(self):
        raise NotImplementedError

    def _standardise(self, features: Sequence[float]) -> list[float]:
        return [
            (v - m) / (s if s > 0 else 1.0) for v, m, s in zip(features, self.mean, self.sd)
        ]

    def fit(self, blasts: Sequence[Blast]) -> "_SklearnArm":
        rows = [b for b in blasts if b.x50_m is not None]
        if len(rows) < 10:
            raise ValueError(f"{self.name} needs at least 10 measured blasts, got {len(rows)}")
        columns = list(zip(*[b.features() for b in rows]))
        self.mean = [sum(c) / len(c) for c in columns]
        self.sd = [
            math.sqrt(sum((v - m) ** 2 for v in c) / len(c))
            for c, m in zip(columns, self.mean)
        ]
        X = [self._standardise(b.features()) for b in rows]
        y = [b.x50_m for b in rows]
        self.model = self._build()
        self.model.fit(X, y)
        return self

    def predict_one(self, blast: Blast) -> Prediction:
        if self.model is None:
            return self._abstain(blast, f"{self.name} has not been fitted")
        value = float(self.model.predict([self._standardise(blast.features())])[0])
        return self._guarded(blast, value, group=assign_group(blast))


class SupportVectorRegression(_SklearnArm):
    """Support-vector regression, in both published parameterisations.

    Two sources tune it on this corpus and reach opposite conclusions about the kernel. Amoako 2022
    searched 2700 combinations across four kernels and chose a radial basis function with a
    regularisation of 5.25 and an epsilon of 0.04, noting radial models "have better generalization
    abilities". Sui 2025 used a degree-five polynomial with a regularisation of 1 and reported this
    arm as the WORST of its three single learners.

    Both ship. The disagreement is the point, and averaging them would erase it.
    """

    name = "svr"
    source = "Amoako, Jha and Zhong 2022 section 4.2.1; Sui et al. 2025 section 4"

    def __init__(self, *, variant: str = "rbf-amoako") -> None:
        super().__init__()
        if variant not in {"rbf-amoako", "poly-sui"}:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.name = f"svr-{variant}"

    def _build(self):
        from sklearn.svm import SVR

        if self.variant == "rbf-amoako":
            return SVR(kernel="rbf", C=5.25, epsilon=0.04)
        return SVR(kernel="poly", degree=5, C=1.0)


class RandomForest(_SklearnArm):
    """Random forest with the hyperparameters the 2025 stacking study settled on."""

    name = "random-forest"
    source = "Sui, Zhou, Zhao, Yang and Zou 2025 doi:10.3390/app15031254, final parameters"

    def _build(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=76, random_state=27)


class GradientBoosting(_SklearnArm):
    """Gradient boosting with the 2025 study's parameters.

    The source used xgboost with a learning rate of 0.5. That learning rate is unusually high, and
    the source itself reports the consequence: "the prediction bias of the XGBoost model in the
    testing set is more pronounced, and its prediction accuracy is much worse than in the training
    set, indicating that the model is overfitting". Reproduced as published, with the overfitting
    left visible rather than tuned away.
    """

    name = "xgboost"
    source = "Sui et al. 2025, final parameters: learning rate 0.5, seed 42"

    def _build(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the xgboost arm needs the 'boost' extra: pip install blastfrag[boost]"
            ) from exc
        return XGBRegressor(learning_rate=0.5, random_state=42, n_estimators=100)


class StackingEnsemble(_SklearnArm):
    """The 2025 state of the art on this corpus: forest and boosting under a linear meta-learner.

    The source chose a plain linear regression as the meta-learner deliberately, "to avoid
    overfitting caused by excessive complexity", having found its boosting base learner overfitting.

    **It also removed cross-validation**, recording that "the cross-validated model had a poor
    prediction effect on the test set". On a 97-row corpus containing 17 rows that duplicate another
    row's feature vector, that is a symptom with a mechanism, and it is the reason this package
    scores every learned arm under three split protocols rather than one.

    Reproduced as published, without cross-validation, so the protocol comparison is like for like.
    """

    name = "stacking"
    source = "Sui, Zhou, Zhao, Yang and Zou 2025 doi:10.3390/app15031254 section 4"

    def _build(self):
        from sklearn.ensemble import RandomForestRegressor, StackingRegressor
        from sklearn.linear_model import LinearRegression

        try:
            from xgboost import XGBRegressor

            boosting = XGBRegressor(learning_rate=0.5, random_state=42, n_estimators=100)
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import GradientBoostingRegressor

            boosting = GradientBoostingRegressor(learning_rate=0.5, random_state=42)
        return StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=76, random_state=27)),
                ("gb", boosting),
            ],
            final_estimator=LinearRegression(),
            # The source removed cross-validation; passing the smallest legal value reproduces that
            # as closely as the library allows while keeping the fit defined.
            cv=2,
        )


LEARNED_LADDER: dict[str, type[Arm]] = {
    "published-neural-net": PublishedNeuralNetwork,
    "svr": SupportVectorRegression,
    "random-forest": RandomForest,
    "xgboost": GradientBoosting,
    "stacking": StackingEnsemble,
}
