"""Active probe selection under a frozen downstream planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .oracle import SellerBehaviorModel, categorical_js_divergence
from .adaptive import ChangePointBehaviorBelief, expected_information_gain_bits


@dataclass(frozen=True)
class ProbeScore:
    offer: float
    information_gain_bits: float
    surplus_cost: float
    score: float
    old_response: dict[str, float]
    new_response: dict[str, float]


class ActiveProbePlanner:
    """Select a behavior-separating offer without changing reward weights."""

    def __init__(self, *, information_weight: float = 1.0, surplus_cost_weight: float = 0.08):
        self.information_weight = float(information_weight)
        self.surplus_cost_weight = float(surplus_cost_weight)

    def choose(
        self,
        *,
        old: SellerBehaviorModel,
        new: SellerBehaviorModel,
        candidate_offers: Iterable[float],
        buyer_budget: float,
        round_id: int = 1,
        max_turns: int = 6,
    ) -> ProbeScore:
        rows = []
        for offer in candidate_offers:
            if offer <= 0 or offer > buyer_budget:
                continue
            left = old.response_distribution(offer, round_id=round_id, max_turns=max_turns)
            right = new.response_distribution(offer, round_id=round_id, max_turns=max_turns)
            info = categorical_js_divergence(left, right)
            cost = offer / max(buyer_budget, 1e-9)
            rows.append(
                ProbeScore(
                    offer=float(offer),
                    information_gain_bits=info,
                    surplus_cost=cost,
                    score=self.information_weight * info - self.surplus_cost_weight * cost,
                    old_response=left,
                    new_response=right,
                )
            )
        if not rows:
            raise ValueError("no positive budget-safe probe offers")
        return max(rows, key=lambda row: (row.score, row.information_gain_bits, -row.offer))

    def choose_from_belief(
        self,
        *,
        belief: ChangePointBehaviorBelief,
        candidate_offers: Iterable[float],
        buyer_budget: float,
        round_id: int = 1,
        max_turns: int = 6,
    ) -> ProbeScore:
        """Select a probe from the deployed posterior, without oracle truth."""

        rows = []
        for offer in candidate_offers:
            if offer <= 0 or offer > buyer_budget:
                continue
            info = expected_information_gain_bits(
                belief,
                offer=float(offer),
                round_id=round_id,
                max_turns=max_turns,
            )
            cost = float(offer) / max(float(buyer_budget), 1e-9)
            predictive = dict(
                belief.response_distribution(
                    float(offer), round_id=round_id, max_turns=max_turns
                )
            )
            rows.append(
                ProbeScore(
                    offer=float(offer),
                    information_gain_bits=info,
                    surplus_cost=cost,
                    score=self.information_weight * info - self.surplus_cost_weight * cost,
                    old_response=predictive,
                    new_response=predictive,
                )
            )
        if not rows:
            raise ValueError("no positive budget-safe probe offers")
        return max(rows, key=lambda row: (row.score, row.information_gain_bits, -row.offer))
