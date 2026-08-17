"""Prompt-based final action generator with deterministic fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from experiments.agenticpay_framework.schemas import StrategicPlan
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    format_action_message,
    format_transcript,
    last_price,
    parse_action,
    validate_buyer_action,
)
from experiments.model_clients import ModelClient

from ..belief_model.base import BeliefState
from ..planner.base import PlannerResult
from .base import GenerationResult


class PromptActionGenerator:
    """Generate paper-format Thought/Talk/Action from belief and plan."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 1200, temperature: float = 1.0, top_p: float = 1.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.last_raw = ""

    def generate(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        planner_result: PlannerResult,
        round_id: int,
        max_turns: int,
    ) -> GenerationResult:
        prompt = self._prompt(
            scenario=scenario,
            history=history,
            belief=belief,
            plan=planner_result.plan,
            planner_result=planner_result,
            round_id=round_id,
            max_turns=max_turns,
        )
        self.last_raw = self.client.generate(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        parsed = parse_action("buyer", self.last_raw)
        action, validator = validate_buyer_action(parsed, scenario, history, round_id, max_turns)
        if not action.valid:
            fallback = self._fallback_message(scenario, history, planner_result.plan)
            parsed = parse_action("buyer", fallback)
            action, validator = validate_buyer_action(parsed, scenario, history, round_id, max_turns)
            validator = {**validator, "generator_fallback_used": True, "llm_raw": self.last_raw}
            return GenerationResult(
                raw=fallback,
                parsed_action=action,
                validator=validator,
                generator_type="prompt_action_generator_with_fallback",
                prompt=prompt,
            )
        return GenerationResult(
            raw=self.last_raw,
            parsed_action=action,
            validator=validator,
            generator_type="prompt_action_generator",
            prompt=prompt,
        )

    def _prompt(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        plan: StrategicPlan,
        planner_result: PlannerResult,
        round_id: int,
        max_turns: int,
    ) -> str:
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        seller_best_quote = min(seller_prices) if seller_prices else None
        return f"""You are the buyer in an RLVR negotiation.

You must bargain for the item using the product codename, and your goal is to
reach a deal at the lowest possible price without ever exceeding your private
budget. Follow the exact paper format.

Private buyer budget: {scenario.buyer_budget:.2f}
Product codename: {scenario.codename_or_default()}
Quantity: {scenario.quantity}
Reference/list price: {scenario.reference_price:.2f}
Round: {round_id}/{max_turns}
Best formal seller quote so far: {seller_best_quote}

Persistent opponent belief:
{json.dumps(belief.to_dict(), ensure_ascii=False)}

Selected strategic plan:
{json.dumps(plan.to_dict(), ensure_ascii=False)}

Candidate-action diagnostics:
{json.dumps(planner_result.to_dict(), ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Reasoning style to use in Thought:
- If making an offer, explicitly reason that you are testing seller flexibility and preserving surplus.
- If conceding, describe it as a calculated concession, not as moving toward your maximum budget.
- If the seller counter is high, do not sound defensive; push back with a firm lower counter.
- If using "best offer" language, frame it as tactical pressure, not as revealing your actual ceiling.
- If accepting, explain why another counter has low opportunity value under the selected plan.

Rules:
- Never offer or accept a price above your private budget.
- Treat your private budget as a hard limit, not a target. Do not move toward it unless the final rounds force you to.
- Use the trained-agent bargaining style: aggressive anchor, calculated small concessions, and persuasive pressure.
- Your first offer should be an aggressive anchor that tests seller flexibility, not a near-budget offer.
- Raise offers slowly. A concession should feel reluctant and calculated, never automatic.
- If the seller has made a formal quote, your [BUY] counteroffer must not exceed the lowest formal seller quote so far.
- Do not accept a seller offer just because it is under budget; accept only when the selected plan explicitly chooses accept and the price captures meaningful surplus.
- When countering, say why the price should be lower: quick close, cash today, market/list discount, budget discipline, or seller flexibility.
- Use stronger buyer language: "This is a serious cash offer", "I need you to move", "I cannot justify your ask", "This is my best number for now".
- You may use tactical finality such as "this is my best offer" or "I can go no higher" as bargaining pressure, but never reveal the true budget.
- Do not capitulate after one seller counter. If there are remaining turns, prefer a concrete counteroffer unless the selected plan explicitly accepts.
- In early rounds, prefer asking/countering over quitting.
- Do not blindly obey a single belief estimate; use the selected plan.
- If plan says accept, only use [DEAL] to accept an exact previous seller [SELL] offer.
- Otherwise use [BUY] for a new buyer offer, [REJECT], or [QUIT].

Return exactly:
Thought: ...
Talk: ...
Action: [BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})

or, if accepting an exact seller offer:
Action: [DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})"""

    @staticmethod
    def _fallback_message(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        plan: StrategicPlan,
    ) -> str:
        price = plan.target_price
        if plan.strategic_act == "accept":
            seller_offer = last_price(history, "seller")
            if seller_offer is not None and seller_offer <= scenario.buyer_budget:
                return format_action_message(
                    "buyer",
                    "DEAL",
                    round(float(seller_offer), 2),
                    scenario,
                    "I can accept your formal offer and close the deal now.",
                )
        if plan.strategic_act == "walk_away":
            return (
                "Thought: The feasible overlap appears too risky under my budget.\n"
                "Talk: I do not think we can make this work within my budget.\n"
                "Action: [QUIT]"
            )
        if price is None:
            price = min(float(scenario.buyer_budget), float(scenario.reference_price) * 0.65)
        price = round(min(float(price), float(scenario.buyer_budget)), 2)
        return format_action_message(
            "buyer",
            "BUY",
            price,
            scenario,
            f"I can make a serious cash offer at ${price:.2f}. I need you to move meaningfully toward this number.",
        )
