"""Rule offer-generator plus native naturalizer baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase
from experiments.agenticpay_framework.baselines.helpers import (
    buyer_max_from_context,
    last_seller_price,
    round_number,
    rule_target_price,
)
from experiments.agenticpay_framework.schemas import BeliefState, StrategicPlan


class RuleOfferGeneratorBuyerAgent(BaselineBuyerBase):
    variant = "rule_offer_generator"
    prompt_guidance = (
        "BASELINE: Rule offer generator plus naturalizer. Follow the provided high-level plan exactly; "
        "the plan is produced by a deterministic concession rule, not by an LLM planner."
    )

    def build_plan(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        buyer_max = buyer_max_from_context(self.buyer_max_price, self.context)
        last_offer = last_seller_price(conversation_history)
        round_id = round_number(current_state)
        target = rule_target_price(buyer_max, last_offer, round_id)
        accept_at = buyer_max
        strategic_act = "concede_small"
        if last_offer is not None and buyer_max is not None and last_offer <= buyer_max:
            strategic_act = "accept"
            target = last_offer
            accept_at = last_offer
        elif round_id <= 1:
            strategic_act = "anchor_low"
        elif round_id >= 8:
            strategic_act = "concede_medium"
        return StrategicPlan(
            strategic_act=strategic_act,
            target_price=target,
            reservation_guardrail=buyer_max,
            concession_size="small" if strategic_act != "concede_medium" else "medium",
            accept_if_at_or_below=accept_at,
            contract_priorities=["price", "buyer-side utility", "required non-price terms"],
            feasibility_checks=["price <= buyer max", "buyer-side contract utility >= 0"],
            language_style="friendly",
            private_rationale="Deterministic concession schedule baseline.",
        )
