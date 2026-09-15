"""Typed-action buyer policies."""

from __future__ import annotations

from typing import Optional

from .schemas import BeliefState, TypedAction


class RuleBasedTypedBuyerPolicy:
    """Simple concession policy used before RL training."""

    def __init__(self, buyer_max_price: float, opening_fraction: float = 0.62):
        self.buyer_max_price = buyer_max_price
        self.opening_fraction = opening_fraction

    def act(
        self,
        round_index: int,
        max_rounds: int,
        last_buyer_price: Optional[float],
        last_seller_price: Optional[float],
        belief_state: Optional[BeliefState] = None,
    ) -> TypedAction:
        if last_seller_price is not None and last_seller_price <= self.buyer_max_price:
            if last_buyer_price is not None and last_seller_price <= last_buyer_price:
                return TypedAction(act="accept", target_price=last_seller_price, rationale="seller crossed buyer offer")
            if round_index >= max_rounds - 1:
                return TypedAction(act="accept", target_price=min(last_seller_price, self.buyer_max_price), rationale="final round")

        if last_buyer_price is None:
            target = self.buyer_max_price * self.opening_fraction
            return TypedAction(act="anchor_low", target_price=target, rationale="initial anchor")

        progress = min(1.0, (round_index + 1) / max(1, max_rounds))
        concession_budget = self.buyer_max_price - last_buyer_price
        step = concession_budget * min(0.35, 0.1 + 0.25 * progress)
        target = min(self.buyer_max_price, last_buyer_price + max(1.0, step))

        if last_seller_price is not None and target >= last_seller_price:
            target = min(self.buyer_max_price, last_seller_price)
            return TypedAction(act="concede_medium", target_price=target, rationale="move near seller ask")

        return TypedAction(act="concede_small", target_price=target, rationale="gradual concession")
