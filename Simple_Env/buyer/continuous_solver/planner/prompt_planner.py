"""Compatibility planner that reuses the existing prompt strategic planner."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from experiments.agenticpay_framework.components import PromptStrategicPlanner
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    RLVRScenario,
    history_to_agenticpay,
    last_price,
)
from experiments.model_clients import ModelClient

from ..belief_model.base import BeliefState
from .base import PlannerResult


class PromptBeliefPlanner:
    """Use persistent belief state as input to the existing prompt planner."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 1200):
        self.planner = PromptStrategicPlanner(client, max_tokens=max_tokens)

    def plan(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> PlannerResult:
        plan = self.planner.plan(
            context={
                "task": "RLVR negotiation continual-resolving prompt planner",
                "item": scenario.title,
                "reference_price": scenario.reference_price,
                "max_turns": max_turns,
                "learned_bargaining_policy": [
                    "Treat the buyer budget as a hard limit, never as a target price.",
                    "Use anchor-and-concession: open aggressively low to test seller flexibility, then concede in small calculated steps.",
                    "Maintain bargaining pressure through persuasive but concise language; do not capitulate after one seller counter.",
                    "Prefer concrete counteroffers over defensive rejection in early and middle rounds.",
                    "Use tactical finality only as language pressure, e.g. 'this is my best offer', without revealing the true budget.",
                    "Accept seller offers only when the opportunity cost of another counter is low or the negotiation is near the final round.",
                    "A seller's verbal finality is evidence, not proof; rely on repeated behavior and the persistent belief state.",
                ],
            },
            conversation_history=history_to_agenticpay(history),
            current_state={
                "round": round_id,
                "last_seller_offer": last_price(history, "seller"),
                "last_buyer_offer": last_price(history, "buyer"),
                "persistent_belief": belief.to_dict(),
            },
            buyer_max_price=scenario.buyer_budget,
            belief=belief.to_agenticpay_belief(),
        )
        return PlannerResult(
            plan=plan,
            planner_type="prompt_belief_planner",
            diagnostics={"planner_raw": self.planner.last_raw},
        )
