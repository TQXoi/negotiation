"""Legacy conservative EV planner baseline.

This approximates the earlier continuous-rule EV behavior that opened much
closer to the buyer budget and was therefore safer for deal rate but weaker on
buyer surplus. Keep it as a fixed baseline when evaluating learned planners.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario, last_price

from ..belief_model.base import BeliefState
from .ev_planner import EVPlanner


class ConservativeEVPlanner(EVPlanner):
    """Prior conservative schedule: higher opening offer and quicker concession."""

    def candidate_prices(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> List[float]:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        last_buyer = last_price(history, "buyer")
        seller_lowest = self._lowest_seller_offer(history)
        cap = min(budget, budget * min(0.98, 0.60 + 0.065 * max(0, round_id - 1)))
        if seller_lowest is not None:
            cap = min(cap, seller_lowest * 0.995)
        floor = max(0.01, min(budget * 0.30, ref * 0.24))
        if last_buyer is not None:
            floor = max(floor, last_buyer + budget * 0.035)
        floor = min(floor, cap)
        cap = max(floor, min(cap, budget))

        prices = {
            min(cap, max(floor, budget * 0.60)),
            min(cap, max(floor, budget * (0.62 + 0.07 * max(0, round_id - 1)))),
        }
        if belief.reservation.p10 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p10) * 1.04)))
        if belief.reservation.p50 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p50) * 1.02)))
        if belief.reservation.p90 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p90) * 0.88)))
        last_seller = last_price(history, "seller")
        if last_seller is not None:
            for ratio in [0.82, 0.88, 0.94]:
                prices.add(min(cap, max(floor, last_seller * ratio)))
        if last_buyer is not None:
            prices.add(min(cap, max(floor, last_buyer + 0.04 * budget)))

        clean = sorted({round(max(0.01, min(budget, p)), 2) for p in prices})
        if len(clean) <= max(1, self.candidate_k):
            return clean
        selected = {clean[0], clean[-1], clean[len(clean) // 2]}
        for price in clean:
            if len(selected) >= self.candidate_k:
                break
            selected.add(price)
        return sorted(selected)
