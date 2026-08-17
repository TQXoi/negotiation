"""Strict final generator for posterior-scored candidate actions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

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
from ..planner.base import CandidateAction, PlannerResult
from .base import GenerationResult


class StrictScoredCandidateGenerator:
    """Choose from scored candidates and realize a parser-safe buyer message."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 1400, temperature: float = 0.7, top_p: float = 1.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.last_raw = ""
        self.last_selected: Optional[CandidateAction] = None

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
        selected = self._match_candidate(action, planner_result)
        if not action.valid or selected is None or not self._selection_allowed(action, scenario, history, round_id, max_turns):
            fallback = self._fallback_message(scenario, history, planner_result, round_id, max_turns)
            parsed = parse_action("buyer", fallback)
            action, validator = validate_buyer_action(parsed, scenario, history, round_id, max_turns)
            selected = self._match_candidate(action, planner_result) or planner_result.selected
            validator = {
                **validator,
                "generator_fallback_used": True,
                "llm_raw": self.last_raw,
                "fallback_reason": "invalid_or_non_candidate_or_policy_violating_action",
            }
            self.last_selected = selected
            return GenerationResult(
                raw=fallback,
                parsed_action=action,
                validator={**validator, "selected_candidate": selected.to_dict() if selected else None},
                generator_type="strict_scored_candidate_generator_with_fallback",
                prompt=prompt,
            )
        self.last_selected = selected
        return GenerationResult(
            raw=self.last_raw,
            parsed_action=action,
            validator={**validator, "selected_candidate": selected.to_dict()},
            generator_type="strict_scored_candidate_generator",
            prompt=prompt,
        )

    def _prompt(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        belief: BeliefState,
        planner_result: PlannerResult,
        round_id: int,
        max_turns: int,
    ) -> str:
        candidates = [self._compact_candidate(c) for c in planner_result.candidates[:12]]
        seller_lowest = self._lowest_seller_offer(history)
        return f"""You are the final buyer generator in an RLVR negotiation.

You must choose exactly one candidate action from the scored candidate table and
write the final Thought/Talk/Action message in the paper format.

Private buyer budget: {scenario.buyer_budget:.2f}
Product codename: {scenario.codename_or_default()}
Quantity: {scenario.quantity}
Reference/list price: {scenario.reference_price:.2f}
Round: {round_id}/{max_turns}
Lowest formal seller quote so far: {seller_lowest}

Bayesian belief summary:
{json.dumps(self._compact_belief_for_prompt(belief), ensure_ascii=False)}

Scored candidate table:
{json.dumps(candidates, ensure_ascii=False)}

Advisor-selected candidate:
{json.dumps(planner_result.selected.to_dict() if planner_result.selected else None, ensure_ascii=False)}

Selection rules:
- Choose from the candidate table. You may choose a different candidate than the
  advisor if your private rationale explains why, but the final Action must
  match one candidate's action_type and price.
- Rounds 1-2: low P_accept is allowed. Prefer an aggressive anchor/information
  probe if it preserves buyer surplus and does not exceed budget.
- When belief confidence is low, do not let low P_accept alone force a high
  offer or acceptance.
- Later rounds: give more weight to p_accept, p_quit, and closing expected value.
- Never offer or accept above the private buyer budget.
- A [BUY] counteroffer must be below the lowest formal seller quote so far.
- A [DEAL] must copy an exact previous seller [SELL] offer. Do not use [DEAL]
  to propose a new price.
- Do not quit in early rounds unless every candidate is invalid.
- Do not accept merely because a price is under budget; accept only when it
  still gives meaningful buyer surplus or this is the final round.

Language rules:
- Be concise, firm, and buyer-favorable.
- Use anchor-and-concession language: serious offer, quick close, cannot justify
  the ask, need you to move, best number for now.
- Do not reveal the true budget.
- Do not mention probabilities, posterior, belief model, EV, or candidate table.

Return exactly:
Thought: ...
Talk: ...
Action: [BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})

or, when accepting an exact previous seller offer:
Action: [DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})
"""

    @staticmethod
    def _compact_belief_for_prompt(belief: BeliefState) -> Dict[str, Any]:
        data = belief.to_dict()
        return {
            "reservation": data.get("reservation"),
            "acceptance_curve": [
                {
                    "price": point.get("price"),
                    "p_accept": point.get("p_accept"),
                    "confidence": point.get("confidence"),
                }
                for point in (data.get("acceptance_curve") or [])[:8]
                if isinstance(point, dict)
            ],
            "concession": data.get("concession"),
            "interaction": data.get("interaction"),
            "recent_evidence": [
                {
                    "round_id": item.get("round_id"),
                    "source": item.get("source"),
                    "type": item.get("evidence_type"),
                    "effect": item.get("effect"),
                    "confidence": item.get("confidence"),
                }
                for item in (data.get("evidence") or [])[-4:]
                if isinstance(item, dict)
            ],
            "posterior_entropy": (data.get("metadata") or {}).get("posterior_entropy"),
        }

    @staticmethod
    def _compact_candidate(candidate: CandidateAction) -> Dict[str, Any]:
        data = candidate.to_dict()
        rationale = str(data.get("rationale") or "")
        return {
            "action_type": data.get("action_type"),
            "price": data.get("price"),
            "p_accept": data.get("p_accept"),
            "p_quit": data.get("p_quit"),
            "reward_if_deal": data.get("reward_if_deal"),
            "future_value": data.get("future_value"),
            "ev": data.get("ev"),
            "rationale": rationale[:220],
        }

    def _fallback_message(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        planner_result: PlannerResult,
        round_id: int,
        max_turns: int,
    ) -> str:
        selected = planner_result.selected or self._best_valid_candidate(planner_result, scenario, history, round_id, max_turns)
        if selected is None:
            price = min(float(scenario.buyer_budget), float(scenario.reference_price) * 0.42)
            return format_action_message(
                "buyer",
                "BUY",
                round(price, 2),
                scenario,
                f"I can make a serious cash offer at ${price:.2f}. I need you to move meaningfully.",
            )
        if selected.action_type == "accept":
            seller_offer = last_price(history, "seller")
            if seller_offer is not None and seller_offer <= scenario.buyer_budget:
                return format_action_message(
                    "buyer",
                    "DEAL",
                    round(float(seller_offer), 2),
                    scenario,
                    "I can close at your formal offer now.",
                )
        if selected.action_type == "reject":
            return (
                "Thought: I need more information and should not move toward my budget yet.\n"
                "Talk: I cannot justify that ask. Give me a sharper number and I can move quickly.\n"
                "Action: [REJECT]"
            )
        if selected.action_type == "quit" and round_id >= max_turns:
            return (
                "Thought: The remaining overlap is too weak within my budget.\n"
                "Talk: I do not think we can make this work at a sensible price for me.\n"
                "Action: [QUIT]"
            )
        price = selected.price
        if price is None:
            best = self._best_valid_candidate(planner_result, scenario, history, round_id, max_turns)
            price = best.price if best and best.price is not None else min(scenario.buyer_budget, scenario.reference_price * 0.48)
        price = self._safe_offer_price(float(price), scenario, history)
        return format_action_message(
            "buyer",
            "BUY",
            round(float(price), 2),
            scenario,
            f"${price:.2f} is a serious cash offer. I need you to move toward this if we are closing today.",
        )

    def _selection_allowed(
        self,
        action: ParsedAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> bool:
        if action.action in {"buy", "deal"} and action.price is not None and action.price > scenario.buyer_budget:
            return False
        seller_lowest = self._lowest_seller_offer(history)
        if action.action == "buy" and seller_lowest is not None and action.price is not None:
            if action.price >= seller_lowest:
                return False
        if action.action == "deal":
            seller_prices = self._seller_offer_prices(history)
            return action.price is not None and any(abs(action.price - x) <= 1e-6 for x in seller_prices)
        if action.action == "quit" and round_id < max_turns:
            return False
        return True

    @staticmethod
    def _match_candidate(action: ParsedAction, planner_result: PlannerResult) -> Optional[CandidateAction]:
        normalized = "offer" if action.action == "buy" else "accept" if action.action == "deal" else action.action
        for candidate in planner_result.candidates:
            if candidate.action_type != normalized:
                continue
            if candidate.price is None and action.price is None:
                return candidate
            if candidate.price is not None and action.price is not None and abs(candidate.price - action.price) <= 0.02:
                return candidate
        return None

    def _best_valid_candidate(
        self,
        planner_result: PlannerResult,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> Optional[CandidateAction]:
        for candidate in sorted(planner_result.candidates, key=lambda x: x.ev, reverse=True):
            action_name = "buy" if candidate.action_type == "offer" else "deal" if candidate.action_type == "accept" else candidate.action_type
            parsed = ParsedAction(
                role="buyer",
                action=action_name,
                price=candidate.price,
                raw="",
                valid=True,
                item_spec=f"{scenario.quantity}x {scenario.codename_or_default()}",
            )
            if self._selection_allowed(parsed, scenario, history, round_id, max_turns):
                return candidate
        return None

    def _safe_offer_price(self, price: float, scenario: RLVRScenario, history: Sequence[Dict[str, Any]]) -> float:
        cap = float(scenario.buyer_budget)
        seller_lowest = self._lowest_seller_offer(history)
        if seller_lowest is not None:
            cap = min(cap, seller_lowest * 0.995)
        return max(0.01, min(float(price), cap))

    @staticmethod
    def _seller_offer_prices(history: Sequence[Dict[str, Any]]) -> list[float]:
        return [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]

    def _lowest_seller_offer(self, history: Sequence[Dict[str, Any]]) -> Optional[float]:
        prices = self._seller_offer_prices(history)
        return min(prices) if prices else None
