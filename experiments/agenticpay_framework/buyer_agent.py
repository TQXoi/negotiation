"""AgenticPay buyer agent that composes modular prompt-only components."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agenticpay.agents.base_agent import BaseAgent
from experiments.agenticpay_framework.components import (
    NativeBuyerNaturalizer,
    PromptOpponentBeliefModel,
    PromptStrategicPlanner,
    ResponseValidator,
)
from experiments.agenticpay_framework.schemas import FrameworkTrace
from experiments.model_clients import ModelClient


class ModularBuyerAgent(BaseAgent):
    """Composable buyer agent for framework ablations.

    Supported variants:
    - belief_prompt: belief model -> naturalizer.
    - planner_generator: planner -> naturalizer.
    - full_framework: belief model -> planner -> naturalizer.

    The naturalizer is currently AgenticPay's native BuyerAgent with a private
    prompt suffix, so this remains compatible with existing benchmark parsers.
    """

    def __init__(
        self,
        model: ModelClient,
        variant: str,
        buyer_max_price: Optional[float],
        name: str = "Buyer",
        role_description: str = "You are a buyer looking for a good deal.",
        system_prompt_suffix: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, role_description=role_description, name=name)
        self.variant = variant
        self.buyer_max_price = buyer_max_price
        self.belief_model = PromptOpponentBeliefModel(model=model, max_tokens=max_tokens)
        self.planner = PromptStrategicPlanner(model=model, max_tokens=max_tokens)
        self.naturalizer = NativeBuyerNaturalizer(
            model=model,
            name=name,
            role_description=role_description,
            buyer_max_price=buyer_max_price,
            system_prompt_suffix=system_prompt_suffix,
        )
        self.validator = ResponseValidator(
            model=model,
            buyer_max_price=buyer_max_price,
            max_tokens=max_tokens,
        )
        self.traces: List[FrameworkTrace] = []
        self.last_selected_seller: Optional[int] = None

    def initialize(self, context: Dict[str, Any]) -> None:
        super().initialize(context)
        self.naturalizer.initialize(context)

    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")

        belief = None
        if self.variant in {"belief_prompt", "full_framework"}:
            belief = self.belief_model.predict(
                context=self.context,
                conversation_history=conversation_history,
                current_state=current_state,
            )

        plan = None
        if self.variant in {"planner_generator", "full_framework"}:
            plan = self.planner.plan(
                context=self.context,
                conversation_history=conversation_history,
                current_state=current_state,
                buyer_max_price=self.buyer_max_price,
                belief=belief,
            )

        naturalization = self.naturalizer.generate(
            conversation_history=conversation_history,
            current_state=current_state,
            belief=belief,
            plan=plan,
        )
        self.last_selected_seller = getattr(self.naturalizer, "last_selected_seller", None)
        naturalization = self.validator.validate_and_revise(
            response=naturalization.final_response,
            context=self.context,
            current_state=current_state,
            conversation_history=conversation_history,
            belief=belief,
            plan=plan,
            selected_seller_id=self.last_selected_seller,
        )
        trace = FrameworkTrace(
            variant=self.variant,
            round_id=current_state.get("round") or current_state.get("round_index"),
            belief=belief,
            plan=plan,
            naturalization=naturalization,
        )
        self.traces.append(trace)
        print("FRAMEWORK_TRACE " + json.dumps(trace.to_dict(), ensure_ascii=False))
        return naturalization.final_response
