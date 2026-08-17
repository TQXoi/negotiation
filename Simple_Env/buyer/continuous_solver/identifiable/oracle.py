"""Evaluator-side seller behavior oracle used to verify identifiability.

The oracle is not exposed to the deployed buyer.  It asks whether two hidden
utility/policy profiles imply observably different responses to at least one
budget-safe buyer offer.  This prevents a change benchmark whose latent state
changes on paper while every rational public action stays effectively equal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


OUTCOMES = ("accept", "counter", "quit")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(key, 0.0))) for key in OUTCOMES)
    if total <= 0:
        return {key: 1.0 / len(OUTCOMES) for key in OUTCOMES}
    return {key: max(0.0, float(values.get(key, 0.0))) / total for key in OUTCOMES}


def categorical_js_divergence(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1]."""

    p, q = _normalize(left), _normalize(right)
    middle = {key: 0.5 * (p[key] + q[key]) for key in OUTCOMES}

    def kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return sum(
            a[key] * math.log2(a[key] / b[key])
            for key in OUTCOMES
            if a[key] > 0 and b[key] > 0
        )

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


@dataclass(frozen=True)
class SellerBehaviorModel:
    """Compact latent seller profile used only by suite construction/evaluation."""

    reservation_price: float
    rationality_tau: float
    counter_propensity: float = 0.78
    quit_bias: float = 0.08
    finality: float = 0.25

    def response_distribution(
        self, offer: float, *, round_id: int = 1, max_turns: int = 6
    ) -> dict[str, float]:
        progress = max(0.0, min(1.0, round_id / max(1, max_turns)))
        tau = max(0.25, float(self.rationality_tau))
        accept = _sigmoid((float(offer) - self.reservation_price) / tau)
        remaining = 1.0 - accept
        quit_share = max(
            0.01,
            min(0.95, self.quit_bias + 0.32 * progress + 0.22 * self.finality),
        )
        counter_share = max(
            0.01,
            min(0.99 - quit_share, self.counter_propensity * (1.0 - 0.35 * progress)),
        )
        return _normalize(
            {
                "accept": accept,
                "counter": remaining * counter_share,
                "quit": remaining * (1.0 - counter_share),
            }
        )
