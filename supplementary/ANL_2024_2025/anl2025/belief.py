from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _entropy(weights: Iterable[float]) -> float:
    return -sum(value * math.log(max(value, 1e-15)) for value in weights)


def _value_key(value: Any) -> str:
    return repr(value)


@dataclass
class EdgePreferenceBelief:
    """Online preference/acceptance belief for one ANL 2025 edge.

    Standard modes never receive the edge utility function. Preference scores
    are learned from issue values in edge proposals, while a separate posterior
    represents the edge's latent minimum acceptable preference score. Oracle
    and fixed modes are evaluator-only causal interventions.
    """

    outcomes: tuple[Any, ...]
    mode: str = "continuous"
    oracle_ufun: Any | None = None
    threshold_grid_size: int = 51
    concession_exponents: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0)
    threshold_grid: list[float] = field(init=False)
    threshold_weights: list[float] = field(init=False)
    value_counts: list[dict[str, float]] = field(init=False)
    offer_observations: int = 0
    response_observations: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    calibration_records: list[dict[str, Any]] = field(default_factory=list)
    _oracle_min: float = field(init=False, default=0.0)
    _oracle_max: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        if self.mode not in {"continuous", "static", "oracle", "fixed"}:
            raise ValueError(f"Unknown edge belief mode: {self.mode}")
        if self.mode in {"oracle", "fixed"} and self.oracle_ufun is None:
            raise ValueError(f"{self.mode} mode requires evaluator-supplied ufun")
        self.threshold_grid = [
            index / (self.threshold_grid_size - 1)
            for index in range(self.threshold_grid_size)
        ]
        self.threshold_weights = [1.0 / self.threshold_grid_size] * self.threshold_grid_size
        nissues = max((len(outcome) for outcome in self.outcomes if outcome is not None), default=1)
        self.value_counts = [dict() for _ in range(nissues)]
        for outcome in self.outcomes:
            if outcome is None:
                continue
            for issue, value in enumerate(outcome):
                self.value_counts[issue].setdefault(_value_key(value), 1.0)
        if self.oracle_ufun is not None:
            try:
                self._oracle_min, self._oracle_max = (
                    float(value) for value in self.oracle_ufun.minmax()
                )
            except Exception:
                values = [float(self.oracle_ufun(outcome)) for outcome in self.outcomes]
                self._oracle_min, self._oracle_max = min(values), max(values)
            normalized_reservation = self._normalize_oracle(
                float(self.oracle_ufun.reserved_value)
            )
            raw = [
                math.exp(-0.5 * ((point - normalized_reservation) / 0.025) ** 2)
                for point in self.threshold_grid
            ]
            self._replace_threshold_weights(raw)

    def _normalize_oracle(self, utility: float) -> float:
        span = max(self._oracle_max - self._oracle_min, 1e-9)
        return _clip((utility - self._oracle_min) / span)

    def _replace_threshold_weights(self, raw: Iterable[float]) -> None:
        values = [max(float(value), 1e-12) for value in raw]
        total = sum(values)
        if not math.isfinite(total) or total <= 0.0:
            self.threshold_weights = [1.0 / self.threshold_grid_size] * self.threshold_grid_size
            return
        self.threshold_weights = [value / total for value in values]

    def preference_score(self, outcome: Any) -> float:
        if outcome is None:
            return 0.0
        if self.oracle_ufun is not None:
            try:
                return self._normalize_oracle(float(self.oracle_ufun(outcome)))
            except Exception:
                return 0.5
        scores: list[float] = []
        for issue, value in enumerate(outcome):
            if issue >= len(self.value_counts):
                scores.append(0.5)
                continue
            counts = self.value_counts[issue]
            if not counts:
                scores.append(0.5)
                continue
            values = list(counts.values())
            low, high = min(values), max(values)
            if high - low <= 1e-9:
                scores.append(0.5)
                continue
            count = counts.get(_value_key(value), low)
            scores.append(_clip((count - low) / (high - low)))
        return sum(scores) / len(scores) if scores else 0.5

    def observe_edge_offer(self, outcome: Any, relative_time: float) -> None:
        time = _clip(relative_time)
        before = self.preference_score(outcome)
        if self.mode == "continuous" and outcome is not None:
            evidence_weight = 1.0 + 0.5 * (1.0 - time)
            for issue, value in enumerate(outcome):
                if issue >= len(self.value_counts):
                    continue
                key = _value_key(value)
                self.value_counts[issue][key] = (
                    self.value_counts[issue].get(key, 1.0) + evidence_weight
                )
        score = self.preference_score(outcome)
        if self.mode == "continuous":
            feasibility = [
                _sigmoid((score - threshold) / 0.08)
                for threshold in self.threshold_grid
            ]
            self._replace_threshold_weights(
                prior * (0.20 + 0.80 * likelihood) ** 0.20
                for prior, likelihood in zip(
                    self.threshold_weights, feasibility, strict=True
                )
            )
        self.offer_observations += 1
        self.evidence.append(
            {
                "kind": "edge_offer",
                "relative_time": round(time, 6),
                "score_before": round(before, 6),
                "score_after": round(score, 6),
                "threshold_mean": round(self.threshold_mean, 6),
            }
        )

    def acceptance_probability(self, outcome: Any, relative_time: float) -> float:
        score = self.preference_score(outcome)
        time = _clip(relative_time)
        probability = 0.0
        exponent_weight = 1.0 / len(self.concession_exponents)
        for threshold, threshold_weight in zip(
            self.threshold_grid, self.threshold_weights, strict=True
        ):
            conditional = 0.0
            for exponent in self.concession_exponents:
                aspiration = threshold + (1.0 - threshold) * (1.0 - time**exponent)
                conditional += exponent_weight * _sigmoid((score - aspiration) / 0.06)
            probability += threshold_weight * conditional
        return _clip(probability)

    def observe_response(
        self,
        outcome: Any,
        relative_time: float,
        *,
        accepted: bool,
        predicted_probability: float | None = None,
    ) -> None:
        time = _clip(relative_time)
        score = self.preference_score(outcome)
        if predicted_probability is None:
            predicted_probability = self.acceptance_probability(outcome, time)
        if self.mode == "continuous":
            likelihoods: list[float] = []
            for threshold in self.threshold_grid:
                conditional = 0.0
                for exponent in self.concession_exponents:
                    aspiration = threshold + (1.0 - threshold) * (1.0 - time**exponent)
                    conditional += _sigmoid((score - aspiration) / 0.06)
                p_accept = conditional / len(self.concession_exponents)
                likelihoods.append(p_accept if accepted else 1.0 - p_accept)
            self._replace_threshold_weights(
                prior * (0.05 + 0.95 * likelihood) ** 0.35
                for prior, likelihood in zip(
                    self.threshold_weights, likelihoods, strict=True
                )
            )
        self.response_observations += 1
        self.calibration_records.append(
            {
                "predicted": _clip(predicted_probability),
                "accepted": bool(accepted),
            }
        )
        self.evidence.append(
            {
                "kind": "our_offer_response",
                "relative_time": round(time, 6),
                "preference_score": round(score, 6),
                "predicted_acceptance": round(_clip(predicted_probability), 6),
                "accepted": bool(accepted),
                "threshold_mean": round(self.threshold_mean, 6),
            }
        )

    def expected_information_gain(self, outcome: Any, relative_time: float) -> float:
        p_accept = self.acceptance_probability(outcome, relative_time)
        if p_accept <= 1e-9 or p_accept >= 1.0 - 1e-9:
            return 0.0
        score = self.preference_score(outcome)
        time = _clip(relative_time)
        accept_raw: list[float] = []
        reject_raw: list[float] = []
        for threshold, prior in zip(
            self.threshold_grid, self.threshold_weights, strict=True
        ):
            conditional = sum(
                _sigmoid(
                    (
                        score
                        - (
                            threshold
                            + (1.0 - threshold) * (1.0 - time**exponent)
                        )
                    )
                    / 0.06
                )
                for exponent in self.concession_exponents
            ) / len(self.concession_exponents)
            accept_raw.append(prior * conditional)
            reject_raw.append(prior * (1.0 - conditional))
        accept_total, reject_total = sum(accept_raw), sum(reject_raw)
        if accept_total <= 1e-12 or reject_total <= 1e-12:
            return 0.0
        accept_post = [value / accept_total for value in accept_raw]
        reject_post = [value / reject_total for value in reject_raw]
        after = p_accept * _entropy(accept_post) + (1.0 - p_accept) * _entropy(reject_post)
        return max(0.0, _entropy(self.threshold_weights) - after)

    @property
    def threshold_mean(self) -> float:
        return sum(
            point * weight
            for point, weight in zip(
                self.threshold_grid, self.threshold_weights, strict=True
            )
        )

    @property
    def normalized_entropy(self) -> float:
        return _entropy(self.threshold_weights) / math.log(len(self.threshold_weights))

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "edge_preference_acceptance_belief",
            "mode": self.mode,
            "threshold_mean": round(self.threshold_mean, 6),
            "threshold_entropy": round(self.normalized_entropy, 6),
            "offer_observations": self.offer_observations,
            "response_observations": self.response_observations,
            "issue_value_counts": [
                {key: round(value, 4) for key, value in counts.items()}
                for counts in self.value_counts
            ],
            "evidence": self.evidence[-12:],
        }
