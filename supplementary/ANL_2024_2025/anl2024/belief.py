from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _entropy(weights: Iterable[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-15)) for p in weights)


@dataclass
class ReservationBelief:
    """Discrete posterior over the opponent reservation value.

    ANL 2024 reveals the opponent's utility function shape but hides its
    reservation value.  This class never receives the hidden truth in normal
    operation.  It updates from two public observations:

    * an opponent offer is unlikely to be below its own reservation value;
    * rejection of our offer is evidence about the opponent's current
      acceptance threshold.

    ``mode=static`` keeps the prior fixed. ``mode=oracle`` is only for a causal
    upper-bound intervention and requires an explicitly supplied oracle value.
    """

    grid_size: int = 101
    prior_low: float = 0.0
    prior_high: float = 1.0
    mode: str = "continuous"
    oracle_value: float | None = None
    likelihood_floor: float = 1e-6
    grid: list[float] = field(init=False)
    weights: list[float] = field(init=False)
    offer_observations: int = 0
    rejection_observations: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    first_offer_utility: float | None = None
    concession_exponents: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.mode not in {"continuous", "static", "oracle", "fixed"}:
            raise ValueError(f"Unknown belief mode: {self.mode}")
        self.prior_low = _clip(self.prior_low)
        self.prior_high = _clip(self.prior_high)
        if self.prior_high <= self.prior_low:
            raise ValueError("prior_high must exceed prior_low")
        step = (self.prior_high - self.prior_low) / (self.grid_size - 1)
        self.grid = [self.prior_low + i * step for i in range(self.grid_size)]
        self.weights = [1.0 / self.grid_size] * self.grid_size
        if self.mode in {"oracle", "fixed"}:
            if self.oracle_value is None:
                raise ValueError(f"{self.mode} belief requires oracle_value")
            self._set_concentrated(self.oracle_value)

    def _set_concentrated(self, value: float, sigma: float = 0.008) -> None:
        value = _clip(value, self.prior_low, self.prior_high)
        raw = [math.exp(-0.5 * ((point - value) / sigma) ** 2) for point in self.grid]
        self._replace_weights(raw)

    def _replace_weights(self, raw: Iterable[float]) -> None:
        values = [max(float(value), self.likelihood_floor) for value in raw]
        total = sum(values)
        if not math.isfinite(total) or total <= 0:
            self.weights = [1.0 / self.grid_size] * self.grid_size
            return
        self.weights = [value / total for value in values]

    def _bayes_update(self, likelihoods: Iterable[float]) -> None:
        if self.mode != "continuous":
            return
        self._replace_weights(
            prior * max(float(likelihood), self.likelihood_floor)
            for prior, likelihood in zip(self.weights, likelihoods, strict=True)
        )

    def observe_opponent_offer(self, opponent_utility: float, relative_time: float) -> None:
        """Update from an offer made by the opponent.

        Rational offers should be above the opponent reservation value. A soft
        feasibility likelihood tolerates noisy and non-standard agents.
        """

        utility = _clip(opponent_utility)
        time = _clip(relative_time)
        before = self.mean
        feasibility = [_sigmoid((utility - rv) / 0.025) for rv in self.grid]
        if self.first_offer_utility is None:
            self.first_offer_utility = utility
            likelihoods = feasibility
        else:
            # Marginalize conservatively over common conceder/linear/Boulware
            # aspiration exponents. Offers are the primary RV evidence in ANL
            # 2024; using a maximum rather than a product prevents a wrong
            # behavioral-family assumption from creating false confidence.
            first = self.first_offer_utility
            curve_likelihoods = []
            for rv in self.grid:
                best = 0.0
                for exponent in self.concession_exponents:
                    predicted = rv + (first - rv) * (1.0 - time**exponent)
                    error = (utility - predicted) / 0.075
                    best = max(best, math.exp(-0.5 * error * error))
                curve_likelihoods.append(0.10 + 0.90 * best)
            likelihoods = [
                feasible_value * curve_value
                for feasible_value, curve_value in zip(
                    feasibility, curve_likelihoods, strict=True
                )
            ]
        self._bayes_update(likelihoods)
        self.offer_observations += 1
        self.evidence.append(
            {
                "kind": "opponent_offer",
                "relative_time": round(time, 6),
                "opponent_utility": round(utility, 6),
                "mean_before": round(before, 6),
                "mean_after": round(self.mean, 6),
            }
        )

    def response_likelihood(self, opponent_utility: float, relative_time: float) -> list[float]:
        """Return P(accept | utility, time, reservation-grid-point)."""

        utility = _clip(opponent_utility)
        time = _clip(relative_time)
        # Unknown opponents may be conceders, linear, or Boulware. Marginalize
        # acceptance over these behavioral families instead of interpreting an
        # early rejection as direct evidence for an extremely high RV.
        response_exponents = (0.2, 1.0, 4.0)
        response_weights = (0.25, 0.30, 0.45)
        values = []
        for rv in self.grid:
            probability = 0.0
            for exponent, weight in zip(
                response_exponents, response_weights, strict=True
            ):
                aspiration = rv + (1.0 - rv) * (1.0 - time**exponent)
                probability += weight * _sigmoid((utility - aspiration) / 0.055)
            values.append(probability)
        return values

    def observe_rejection(self, opponent_utility: float, relative_time: float) -> None:
        utility = _clip(opponent_utility)
        time = _clip(relative_time)
        before = self.mean
        accept_likelihood = self.response_likelihood(utility, time)
        # A rejection primarily identifies aspiration/concession behavior, not
        # RV. Apply a fractional Bayes update so it remains weak evidence.
        self._bayes_update(
            max(1.0 - value, self.likelihood_floor) ** 0.15
            for value in accept_likelihood
        )
        self.rejection_observations += 1
        self.evidence.append(
            {
                "kind": "rejected_our_offer",
                "relative_time": round(time, 6),
                "opponent_utility": round(utility, 6),
                "mean_before": round(before, 6),
                "mean_after": round(self.mean, 6),
            }
        )

    def acceptance_probability(self, opponent_utility: float, relative_time: float) -> float:
        likelihood = self.response_likelihood(opponent_utility, relative_time)
        return sum(p * q for p, q in zip(self.weights, likelihood, strict=True))

    def expected_information_gain(self, opponent_utility: float, relative_time: float) -> float:
        likelihood = self.response_likelihood(opponent_utility, relative_time)
        p_accept = sum(p * q for p, q in zip(self.weights, likelihood, strict=True))
        if p_accept <= 1e-12 or p_accept >= 1.0 - 1e-12:
            return 0.0
        accept_raw = [p * q for p, q in zip(self.weights, likelihood, strict=True)]
        reject_raw = [p * (1.0 - q) for p, q in zip(self.weights, likelihood, strict=True)]
        accept_total = sum(accept_raw)
        reject_total = sum(reject_raw)
        accept_post = [value / accept_total for value in accept_raw]
        reject_post = [value / reject_total for value in reject_raw]
        expected_after = p_accept * _entropy(accept_post) + (1.0 - p_accept) * _entropy(reject_post)
        return max(0.0, _entropy(self.weights) - expected_after)

    def cdf(self, value: float) -> float:
        return sum(weight for point, weight in zip(self.grid, self.weights, strict=True) if point <= value)

    def quantile(self, probability: float) -> float:
        target = _clip(probability)
        total = 0.0
        for point, weight in zip(self.grid, self.weights, strict=True):
            total += weight
            if total >= target:
                return point
        return self.grid[-1]

    @property
    def mean(self) -> float:
        return sum(point * weight for point, weight in zip(self.grid, self.weights, strict=True))

    @property
    def variance(self) -> float:
        mean = self.mean
        return sum(weight * (point - mean) ** 2 for point, weight in zip(self.grid, self.weights, strict=True))

    @property
    def normalized_entropy(self) -> float:
        return _entropy(self.weights) / math.log(len(self.weights))

    @property
    def confidence(self) -> float:
        return _clip(1.0 - self.normalized_entropy)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "reservation_posterior",
            "mode": self.mode,
            "mean": round(self.mean, 6),
            "std": round(math.sqrt(self.variance), 6),
            "q10": round(self.quantile(0.10), 6),
            "q50": round(self.quantile(0.50), 6),
            "q90": round(self.quantile(0.90), 6),
            "confidence": round(self.confidence, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "offer_observations": self.offer_observations,
            "rejection_observations": self.rejection_observations,
            "evidence": self.evidence[-12:],
        }
