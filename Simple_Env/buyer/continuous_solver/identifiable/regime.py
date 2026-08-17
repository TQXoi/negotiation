"""Old/new regime mixture for online seller response observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .oracle import OUTCOMES, SellerBehaviorModel


@dataclass(frozen=True)
class RegimeUpdate:
    outcome: str
    old_probability: float
    new_probability: float
    change_probability: float
    predictive_probability: float


class RegimeMixtureBelief:
    """Two-regime Bayesian filter with an explicit transition hazard.

    `old` represents the persistent pre-change hypothesis. `new` represents the
    reset/change hypothesis. Unlike a blind uniform reset, the new regime makes
    concrete action predictions and must earn posterior mass from observations.
    """

    def __init__(
        self,
        *,
        old: SellerBehaviorModel,
        new: SellerBehaviorModel,
        hazard: float = 0.08,
        initial_change_probability: float = 0.02,
    ):
        self.old = old
        self.new = new
        self.hazard = max(0.0, min(1.0, float(hazard)))
        self.change_probability = max(0.0, min(1.0, float(initial_change_probability)))
        self.history: list[RegimeUpdate] = []

    def predict_change_probability(self) -> float:
        return self.change_probability + (1.0 - self.change_probability) * self.hazard

    def update(
        self,
        *,
        offer: float,
        outcome: str,
        round_id: int = 1,
        max_turns: int = 6,
    ) -> RegimeUpdate:
        if outcome == "walk_away":
            normalized = "quit"
        elif outcome == "reject":
            # In Simple Env [REJECT] keeps bargaining alive; it is non-accept
            # evidence, not a terminal quit observation.
            normalized = "counter"
        else:
            normalized = outcome
        if normalized not in OUTCOMES:
            raise ValueError(f"unsupported seller outcome: {outcome}")
        prior_new = self.predict_change_probability()
        prior_old = 1.0 - prior_new
        old_likelihood = self.old.response_distribution(
            offer, round_id=round_id, max_turns=max_turns
        )[normalized]
        new_likelihood = self.new.response_distribution(
            offer, round_id=round_id, max_turns=max_turns
        )[normalized]
        predictive = prior_old * old_likelihood + prior_new * new_likelihood
        posterior_new = prior_new * new_likelihood / max(predictive, 1e-12)
        self.change_probability = max(0.0, min(1.0, posterior_new))
        row = RegimeUpdate(
            outcome=normalized,
            old_probability=1.0 - self.change_probability,
            new_probability=self.change_probability,
            change_probability=self.change_probability,
            predictive_probability=predictive,
        )
        self.history.append(row)
        return row

    def response_distribution(
        self, offer: float, *, round_id: int = 1, max_turns: int = 6
    ) -> Mapping[str, float]:
        weight = self.predict_change_probability()
        old = self.old.response_distribution(offer, round_id=round_id, max_turns=max_turns)
        new = self.new.response_distribution(offer, round_id=round_id, max_turns=max_turns)
        return {key: (1.0 - weight) * old[key] + weight * new[key] for key in OUTCOMES}
