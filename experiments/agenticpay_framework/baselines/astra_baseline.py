"""Lightweight ASTRA-like buyer baseline.

This is an engineering baseline inspired by opponent modeling plus offer
optimization. It is not the original ASTRA implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase
from experiments.agenticpay_framework.baselines.helpers import (
    belief_floor,
    buyer_max_from_context,
    last_seller_price,
    round_number,
    rule_target_price,
    seller_guardrails,
)
from experiments.agenticpay_framework.schemas import BeliefState, StrategicPlan


class ASTRABaselineBuyerAgent(BaselineBuyerBase):
    variant = "astra_baseline"
    uses_belief = True
    prompt_guidance = (
        "BASELINE: ASTRA-like opponent modeling plus offer optimization. Follow the provided "
        "belief-conditioned offer plan exactly; this is a lightweight prompt-only approximation, "
        "not the original ASTRA system."
    )

    def build_plan(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        assert belief is not None
        buyer_max = buyer_max_from_context(self.buyer_max_price, self.context)
        last_offer = belief.last_seller_offer or last_seller_price(conversation_history)
        floor = belief_floor(belief)
        round_id = round_number(current_state)
        deal_risk = belief.deal_risk
        target = self._target_price(buyer_max, floor, last_offer, round_id, deal_risk)
        strategic_act = "concede_small"
        if buyer_max is not None and floor is not None and floor > buyer_max:
            strategic_act = "walk_away" if belief.seller_reservation_confidence >= 0.55 or deal_risk >= 0.7 else "ask_info"
            target = buyer_max
        elif last_offer is not None and buyer_max is not None and last_offer <= buyer_max:
            strategic_act = "accept"
            target = last_offer
        elif round_id <= 1:
            strategic_act = "anchor_low"
        elif deal_risk >= 0.7:
            strategic_act = "concede_medium"
        return StrategicPlan(
            strategic_act=strategic_act,
            target_price=target,
            reservation_guardrail=buyer_max,
            concession_size="medium" if strategic_act == "concede_medium" else "small",
            accept_if_at_or_below=buyer_max,
            contract_priorities=["price", "acceptance probability", "buyer-side utility", "seller feasibility"],
            feasibility_checks=["price <= buyer max", "buyer-side contract utility >= 0", "seller likely utility >= 0"],
            seller_feasible_price_floor=floor,
            seller_feasibility_risk=deal_risk,
            seller_feasible_non_price_terms=belief.contract_term_preferences,
            seller_feasibility_guardrails=seller_guardrails(floor, buyer_max),
            language_style="friendly" if belief.seller_friendliness_range is None else "patient",
            private_rationale="ASTRA-like baseline: estimate opponent feasibility, then choose an offer in the buyer/seller overlap.",
        )

    @staticmethod
    def _target_price(
        buyer_max: Optional[float],
        floor: Optional[float],
        last_offer: Optional[float],
        round_id: int,
        deal_risk: float,
    ) -> Optional[float]:
        if buyer_max is None:
            return None
        rule_target = rule_target_price(buyer_max, last_offer, round_id) or buyer_max
        if floor is None:
            return rule_target
        if floor > buyer_max:
            return buyer_max
        risk_weight = min(0.9, max(0.25, deal_risk + 0.1 * (round_id / 10.0)))
        overlap_target = floor + (buyer_max - floor) * risk_weight
        if last_offer is not None and last_offer <= buyer_max:
            overlap_target = min(overlap_target, last_offer)
        return round(min(max(rule_target, overlap_target), buyer_max), 4)
