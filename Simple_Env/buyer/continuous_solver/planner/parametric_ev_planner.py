"""CEM-tunable EV planner for continual resolving."""

from __future__ import annotations

from typing import Dict, List, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario, last_price

from ..belief_model.base import BeliefState
from .ev_planner import EVPlanner
from .params import PlannerParams


class ParametricEVPlanner(EVPlanner):
    """EV planner whose concession schedule is controlled by PlannerParams."""

    def __init__(self, *, candidate_k: int = 5, params: PlannerParams | None = None):
        self.params = (params or PlannerParams()).clipped()
        super().__init__(
            candidate_k=candidate_k,
            quit_penalty=self.params.quit_penalty,
            future_value_weight=self.params.future_value_weight,
        )

    def candidate_prices(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> List[float]:
        params = self.params
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        last_buyer = last_price(history, "buyer")
        last_seller = last_price(history, "seller")
        seller_lowest = self._lowest_seller_offer(history)
        progress = max(0.0, min(1.0, (round_id - 1) / max(1, max_turns - 1)))

        concession_rate = (
            params.early_concession_rate
            if round_id <= max(2, max_turns // 2)
            else params.late_concession_rate
        )
        schedule_ratio = params.first_anchor_ratio + concession_rate * max(0, round_id - 1)
        cap = min(budget, budget * min(0.98, schedule_ratio + 0.12 * progress))
        if seller_lowest is not None:
            seller_discount = params.seller_discount_early * (1.0 - progress) + params.seller_discount_late * progress
            cap = min(cap, seller_lowest * min(0.995, seller_discount + 0.08))
        floor = max(0.01, min(budget * 0.18, ref * 0.16))
        if last_buyer is not None:
            floor = max(floor, last_buyer + params.min_increment_ratio * budget)
        floor = min(floor, cap)
        cap = max(floor, min(cap, budget))

        prices = {
            min(cap, max(floor, budget * params.first_anchor_ratio)),
            min(cap, max(floor, budget * schedule_ratio)),
        }
        if belief.reservation.p10 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p10) * 1.02)))
        if belief.reservation.p50 is not None:
            p50 = float(belief.reservation.p50)
            prices.add(min(cap, max(floor, p50 * params.p50_low_mult)))
            prices.add(min(cap, max(floor, p50 * params.p50_mid_mult)))
            prices.add(min(cap, max(floor, p50 * params.p50_high_mult)))
        if belief.reservation.mean is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.mean) * params.mean_mult)))
        if belief.reservation.p90 is not None:
            prices.add(min(cap, max(floor, float(belief.reservation.p90) * 0.72)))
        if last_seller is not None:
            seller_discount = params.seller_discount_early * (1.0 - progress) + params.seller_discount_late * progress
            for delta in [-0.08, 0.0, 0.06]:
                prices.add(min(cap, max(floor, last_seller * max(0.45, seller_discount + delta))))
        if last_buyer is not None:
            prices.add(min(cap, max(floor, last_buyer + params.min_increment_ratio * budget)))

        clean = sorted({round(max(0.01, min(budget, p)), 2) for p in prices})
        if len(clean) <= max(1, self.candidate_k):
            return clean
        center = float(belief.reservation.p50 or belief.reservation.mean or clean[len(clean) // 2])
        selected = set()
        for target in [
            center * params.p50_low_mult,
            center * params.p50_mid_mult,
            center * params.p50_high_mult,
            float(belief.reservation.mean or center) * params.mean_mult,
        ]:
            if len(selected) >= self.candidate_k:
                break
            selected.add(min(clean, key=lambda p: abs(p - target)))
        for price in sorted(clean, key=lambda p: abs(p - center)):
            if len(selected) >= self.candidate_k:
                break
            selected.add(price)
        return sorted(selected)

    def _score_offer(self, price, scenario, belief, posterior, round_id, max_turns):
        candidate = super()._score_offer(price, scenario, belief, posterior, round_id, max_turns)
        if belief.reservation.p10 is not None and price < float(belief.reservation.p10):
            gap = (float(belief.reservation.p10) - price) / max(float(scenario.buyer_budget), 1.0)
            candidate.ev -= max(0.0, self.params.lowball_penalty_weight - 0.20) * gap
        candidate.rationale += "; concession schedule controlled by learned planner params"
        return candidate

    def _should_consider_accept(self, *, scenario, history, belief, last_seller, round_id, max_turns, offer_candidates):
        budget = float(scenario.buyer_budget)
        if last_seller > budget + 1e-9:
            return False
        seller_ratio = float(last_seller) / max(budget, 1.0)
        reward = self._reward_if_deal(last_seller, scenario)
        remaining = max_turns - round_id
        best_counter_ev = max((c.ev for c in offer_candidates if c.action_type == "offer"), default=-1.0)
        if remaining >= 2 and seller_ratio > self.params.accept_seller_ratio_threshold:
            return False
        if reward < self.params.accept_reward_threshold and round_id < max_turns:
            return False
        if best_counter_ev > reward - 0.04 and round_id < max_turns:
            return False
        return round_id >= max_turns or reward >= self.params.accept_reward_threshold
