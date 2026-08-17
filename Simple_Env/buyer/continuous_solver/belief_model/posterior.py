"""Posterior belief models for seller reservation price."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable, List, Optional, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario

from .base import AcceptancePoint, BeliefState, ReservationBelief, clamp


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


@dataclass
class ReservationPosterior:
    """Discrete posterior over seller reservation prices.

    The grid is deliberately simple. It gives us a mathematical object for
    Bayesian-style updates, entropy, quantiles, and acceptance probabilities.
    """

    grid: List[float]
    probs: List[float]
    tau_accept: float

    @classmethod
    def from_scenario(
        cls,
        scenario: RLVRScenario,
        *,
        num_grid: int = 81,
        low_ratio: float = 0.30,
        high_ratio: float = 1.02,
        prior_mean_ratio: float = 0.65,
        prior_std_ratio: float = 0.18,
    ) -> "ReservationPosterior":
        ref = max(float(scenario.reference_price), 1.0)
        low = max(0.01, low_ratio * ref)
        high = max(low + 1.0, high_ratio * ref)
        step = (high - low) / max(1, num_grid - 1)
        grid = [low + i * step for i in range(num_grid)]
        mean = prior_mean_ratio * ref
        std = max(1.0, prior_std_ratio * ref)
        probs = [exp(-0.5 * ((x - mean) / std) ** 2) for x in grid]
        tau_accept = max(1.0, 0.035 * ref)
        posterior = cls(grid=grid, probs=probs, tau_accept=tau_accept)
        posterior.normalize()
        return posterior

    def normalize(self) -> None:
        total = sum(self.probs)
        if total <= 0:
            self.probs = [1.0 / len(self.probs)] * len(self.probs)
        else:
            self.probs = [p / total for p in self.probs]

    def copy(self) -> "ReservationPosterior":
        return ReservationPosterior(list(self.grid), list(self.probs), self.tau_accept)

    def update_likelihood(self, likelihood: Sequence[float], *, mix_floor: float = 1e-4) -> None:
        self.probs = [p * max(float(l), mix_floor) for p, l in zip(self.probs, likelihood)]
        self.normalize()

    def p_accept(self, price: float) -> float:
        return sum(p * _sigmoid((float(price) - r) / self.tau_accept) for r, p in zip(self.grid, self.probs))

    def update_accept(self, price: float) -> None:
        self.update_likelihood([_sigmoid((float(price) - r) / self.tau_accept) for r in self.grid])

    def update_reject(self, price: float) -> None:
        self.update_likelihood([1.0 - _sigmoid((float(price) - r) / self.tau_accept) for r in self.grid])

    def update_counter(
        self,
        buyer_offer: Optional[float],
        seller_offer: float,
        *,
        round_pressure: float,
        buyer_budget: Optional[float] = None,
    ) -> None:
        """Update after seller rejected buyer offer and countered at seller_offer."""

        if buyer_offer is not None:
            self.update_reject(buyer_offer)
        # A seller counteroffer is a strategic ask, not a cost disclosure. The
        # early asks in RLVR are often high anchors, so infer reservation from a
        # discounted quote and only gradually trust repeated counters.
        markup = 0.16 + 0.16 * (1.0 - round_pressure)
        if buyer_budget is not None and seller_offer >= 0.90 * float(buyer_budget):
            markup += 0.06
        markup = max(0.08, min(0.42, markup))
        sigma = max(1.0, 0.14 * max(seller_offer, self.grid[-1]))
        center = float(seller_offer) * (1.0 - markup)
        likelihood = [exp(-0.5 * ((r - center) / sigma) ** 2) for r in self.grid]
        # Seller countering at y makes r >> y less likely, but do not let a
        # high opening anchor collapse all mass near the ask.
        likelihood = [l * (1.0 if r <= seller_offer * 1.02 else 0.35) for r, l in zip(self.grid, likelihood)]
        # Mix with the previous posterior so one dramatic quote cannot dominate
        # the belief state. Later rounds get slightly more weight.
        quote_weight = 0.45 + 0.25 * round_pressure
        mixed = [(1.0 - quote_weight) + quote_weight * l for l in likelihood]
        self.update_likelihood(mixed)

    def quantile(self, q: float) -> float:
        target = clamp(q)
        cumulative = 0.0
        for r, p in zip(self.grid, self.probs):
            cumulative += p
            if cumulative >= target:
                return r
        return self.grid[-1]

    def mean(self) -> float:
        return sum(r * p for r, p in zip(self.grid, self.probs))

    def std(self) -> float:
        mu = self.mean()
        return sqrt(max(0.0, sum(p * (r - mu) ** 2 for r, p in zip(self.grid, self.probs))))

    def entropy(self) -> float:
        return -sum(p * (0.0 if p <= 0 else __import__("math").log(p)) for p in self.probs)

    def summary(self) -> ReservationBelief:
        std = self.std()
        mean = self.mean()
        spread = max(self.grid[-1] - self.grid[0], 1.0)
        confidence = clamp(1.0 - std / spread)
        return ReservationBelief(
            mean=mean,
            std=std,
            p10=self.quantile(0.10),
            p50=self.quantile(0.50),
            p90=self.quantile(0.90),
            confidence=confidence,
        )

    def acceptance_curve(self, prices: Iterable[float]) -> List[AcceptancePoint]:
        std = self.std()
        spread = max(self.grid[-1] - self.grid[0], 1.0)
        confidence = clamp(1.0 - std / spread)
        return [
            AcceptancePoint(
                price=round(float(price), 2),
                p_accept=clamp(self.p_accept(float(price))),
                confidence=confidence,
                evidence="computed from reservation posterior",
            )
            for price in prices
        ]


class PosteriorBeliefModel:
    """Stateful belief model that owns a ReservationPosterior."""

    def __init__(self, scenario: RLVRScenario, *, num_grid: int = 81):
        self.scenario = scenario
        self.posterior = ReservationPosterior.from_scenario(scenario, num_grid=num_grid)
        self.state = BeliefState(
            item_id=scenario.item_id,
            buyer_budget=float(scenario.buyer_budget),
            reference_price=float(scenario.reference_price),
            reservation=self.posterior.summary(),
            metadata={
                "belief_type": "posterior",
                "prior": "rule_based_reference_price_gaussian",
                "posterior_entropy": self.posterior.entropy(),
            },
        )

    def refresh_summary(self, candidate_prices: Iterable[float]) -> BeliefState:
        self.state.reservation = self.posterior.summary()
        self.state.acceptance_curve = self.posterior.acceptance_curve(candidate_prices)
        self.state.metadata["posterior_entropy"] = self.posterior.entropy()
        return self.state
