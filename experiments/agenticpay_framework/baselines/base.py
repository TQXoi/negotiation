"""Base classes for report baselines.

Each concrete baseline lives in its own file. This base class keeps the common
AgenticPay-compatible naturalization, validation, and trace plumbing in one
place so the per-baseline files stay readable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agenticpay.agents.base_agent import BaseAgent
from experiments.agenticpay_framework.components import (
    NativeBuyerNaturalizer,
    PromptOpponentBeliefModel,
    ResponseValidator,
)
from experiments.agenticpay_framework.schemas import BeliefState, FrameworkTrace, StrategicPlan
from experiments.model_clients import ModelClient


class BaselineBuyerBase(BaseAgent):
    """Shared wrapper around native AgenticPay generation and validation."""

    variant = "baseline"
    prompt_guidance = "BASELINE: Native AgenticPay buyer behavior."
    uses_belief = False

    def __init__(
        self,
        model: ModelClient,
        buyer_max_price: Optional[float],
        name: str = "Buyer",
        role_description: str = "You are a buyer looking for a good deal.",
        system_prompt_suffix: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, role_description=role_description, name=name)
        self.buyer_max_price = buyer_max_price
        self.belief_model = PromptOpponentBeliefModel(model=model, max_tokens=max_tokens)
        self.naturalizer = NativeBuyerNaturalizer(
            model=model,
            name=name,
            role_description=role_description,
            buyer_max_price=buyer_max_price,
            system_prompt_suffix=self._combined_suffix(system_prompt_suffix),
        )
        self.validator = ResponseValidator(model=model, buyer_max_price=buyer_max_price, max_tokens=max_tokens)
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

        belief = self._build_belief(conversation_history, current_state)
        plan = self.build_plan(conversation_history, current_state, belief)
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
        print("BASELINE_TRACE " + json.dumps(trace.to_dict(), ensure_ascii=False))
        return naturalization.final_response

    def build_plan(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        belief: Optional[BeliefState],
    ) -> Optional[StrategicPlan]:
        return None

    def _build_belief(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> Optional[BeliefState]:
        if not self.uses_belief:
            return None
        return self.belief_model.predict(
            context=self.context,
            conversation_history=conversation_history,
            current_state=current_state,
        )

    def _combined_suffix(self, base_suffix: Optional[str]) -> str:
        return "\n\n".join(part for part in [base_suffix, self.prompt_guidance] if part)
