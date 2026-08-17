#!/usr/bin/env python3
"""Paper-aligned RLVR negotiation evaluation with LLM buyers.

This runner follows the public experimental protocol described in
"Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable
Rewards" as closely as possible without the paper's official code, exact
AmazonHistoryPrice split, or trained checkpoint:

- buyer has private budget B;
- seller has private cost C and is regulated not to accept below C;
- actions are structured as [BUY], [SELL], [DEAL], [REJECT], [QUIT];
- reward is (B - final_price) / abs(B - C), clipped to [-1, 1];
- no deal / quit / timeout receives 0;
- buyer budget overshoot or malformed buyer action receives -1;
- default evaluation is 128 test instances x 4 rollouts, max_turns=6;
- buyer temperature defaults to 1.0 and seller temperature to 0.7.

The default scenarios are deterministic synthetic AmazonHistoryPrice-style
items because the paper's held-out split is not public. If you have an exact
test split, pass it with --scenarios-jsonl. Each JSONL row should include at
least: item_id, title, buyer_budget, seller_cost, reference_price.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "AgenticPay"))

if "loguru" not in sys.modules:
    loguru_stub = types.ModuleType("loguru")

    class _LoggerStub:
        def __getattr__(self, _name: str):
            def _noop(*_args: Any, **_kwargs: Any) -> None:
                return None

            return _noop

    loguru_stub.logger = _LoggerStub()  # type: ignore[attr-defined]
    sys.modules["loguru"] = loguru_stub

from experiments.agenticpay_framework.components import (  # noqa: E402
    PromptOpponentBeliefModel,
    PromptStrategicPlanner,
    json_from_text,
)
from experiments.agenticpay_framework.schemas import BeliefState, StrategicPlan  # noqa: E402
from experiments.model_clients import ModelClient, make_model_client  # noqa: E402
from experiments.external_comparisons.rlvr_counterfactual_offer_model import (  # noqa: E402
    CounterfactualOfferPlanner,
    CounterfactualOfferResponseModel,
)


RLVR_BUYER_VARIANTS = {
    "prompt_only",
    "direct_prompt",
    "cot_prompt",
    "belief_only",
    "belief_prompt",
    "planner_generator",
    "full_framework",
    "typed_belief",
    "typed_evidence_belief",
    "counterfactual_belief",
    "counterfactual_offer_model",
    "astra_style_belief",
    "bond_style_posterior",
    "preference_estimation",
}

BELIEF_VARIANTS = {
    "belief_only",
    "belief_prompt",
    "planner_generator",
    "full_framework",
    "typed_belief",
    "typed_evidence_belief",
    "counterfactual_belief",
    "counterfactual_offer_model",
    "astra_style_belief",
    "bond_style_posterior",
    "preference_estimation",
}

PLANNER_VARIANTS = {
    "planner_generator",
    "full_framework",
    "counterfactual_belief",
    "counterfactual_offer_model",
}


ACTION_RE = re.compile(
    r"\[(BUY|SELL|DEAL|REJECT|QUIT)\]\s*(?:\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?))?(?:\s*\(([^)]*)\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RLVRScenario:
    item_id: str
    title: str
    buyer_budget: float
    seller_cost: float
    reference_price: float
    category: str = "general"
    codename: str = ""
    description: str = ""
    features: str = ""
    quantity: int = 1

    @property
    def mi(self) -> bool:
        return self.buyer_budget > self.seller_cost

    @property
    def gap(self) -> float:
        return self.buyer_budget - self.seller_cost

    def codename_or_default(self) -> str:
        if self.codename:
            return self.codename
        safe_category = re.sub(r"[^a-zA-Z0-9]+", "_", self.category.strip().lower()).strip("_") or "product"
        suffix = re.sub(r"[^0-9]+", "", self.item_id)[-4:] or "0"
        return f"{safe_category}_{suffix}"

    def description_or_default(self) -> str:
        return self.description or self.title

    def features_or_default(self) -> str:
        return self.features or "No additional features provided."


@dataclass
class ParsedAction:
    role: str
    action: str
    price: Optional[float]
    raw: str
    valid: bool
    item_spec: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLVREvalEpisode:
    variant: str
    scenario: Dict[str, Any]
    rollout: int
    status: str
    terminal_reason: str
    final_price: Optional[float]
    reward: float
    buyer_bargained_ratio: Optional[float]
    first_turn_offer_ratio: Optional[float]
    buyer_overshoot: bool
    buyer_format_violation: bool
    mi: bool
    ci: bool
    deal_in_mi: Optional[bool]
    rational_walkaway_ci: Optional[bool]
    rounds: int
    transcript: List[Dict[str, Any]]
    framework_trace: List[Dict[str, Any]]
    elapsed_seconds: float


class RLVRBuyer:
    def __init__(self, variant: str, client: ModelClient, max_tokens: int, temperature: float, top_p: float):
        if variant not in RLVR_BUYER_VARIANTS:
            raise ValueError(f"Unknown RLVR buyer variant: {variant}. Available: {sorted(RLVR_BUYER_VARIANTS)}")
        self.variant = variant
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.belief_model = PromptOpponentBeliefModel(client, max_tokens=min(max_tokens, 1200))
        self.planner = PromptStrategicPlanner(client, max_tokens=min(max_tokens, 1200))
        self.typed_evidence_belief_raw = ""
        self.cf_response_model = CounterfactualOfferResponseModel(client, max_tokens=min(max_tokens, 1400))
        self.cf_offer_planner = CounterfactualOfferPlanner()

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> tuple[ParsedAction, Dict[str, Any]]:
        belief: Optional[BeliefState] = None
        plan: Optional[StrategicPlan] = None
        counterfactual_response = None
        if self.variant in BELIEF_VARIANTS:
            if self.variant == "counterfactual_offer_model":
                belief = self._belief(scenario, history, round_id, max_turns)
            elif self.variant == "typed_evidence_belief":
                belief = self._typed_evidence_belief(scenario, history, round_id, max_turns)
            else:
                belief = self._belief(scenario, history, round_id, max_turns)
            belief = self._adapt_belief_style(belief, scenario, history, round_id, max_turns)
        if self.variant == "counterfactual_offer_model" and counterfactual_response is not None:
            raise AssertionError("counterfactual_response should be created after the full-framework base plan")
        if self.variant == "counterfactual_offer_model":
            base_plan = self._plan(scenario, history, round_id, max_turns, belief)
            schedule = concession_schedule(scenario, history, round_id, max_turns)
            counterfactual_response = self.cf_response_model.predict(
                scenario=scenario,
                history=history,
                round_id=round_id,
                max_turns=max_turns,
                schedule=schedule,
                candidate_prices=self._counterfactual_reranker_candidate_prices(
                    scenario=scenario,
                    history=history,
                    round_id=round_id,
                    max_turns=max_turns,
                    base_plan=base_plan,
                ),
            )
            plan = self.cf_offer_planner.rerank_full_framework_plan(
                scenario=scenario,
                history=history,
                round_id=round_id,
                max_turns=max_turns,
                schedule=schedule,
                response_belief=counterfactual_response,
                base_plan=base_plan,
            )
        elif self.variant in PLANNER_VARIANTS:
            plan = self._plan(scenario, history, round_id, max_turns, belief)
        elif self.variant in {"typed_belief", "typed_evidence_belief"}:
            plan = self._typed_belief_plan(scenario, history, round_id, max_turns, belief)
        prompt = self._buyer_prompt(scenario, history, round_id, max_turns, belief, plan)
        raw = self.client.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        raw_action = parse_action("buyer", raw)
        raw_action, typed_plan_override = self._maybe_apply_typed_plan_action_override(
            raw_action,
            scenario,
            history,
            plan,
        )
        action, validator = validate_buyer_action(raw_action, scenario, history, round_id, max_turns)
        if typed_plan_override:
            validator = {
                **validator,
                "typed_plan_action_override": typed_plan_override,
            }
        trace = {
            "round": round_id,
            "variant": self.variant,
            "belief": belief.to_dict() if belief else None,
            "belief_raw": (
                self.typed_evidence_belief_raw
                if self.variant == "typed_evidence_belief"
                else (
                    getattr(self.cf_response_model, "last_raw", "")
                    if self.variant == "counterfactual_offer_model"
                    else (getattr(self.belief_model, "last_raw", "") if belief else None)
                )
            ),
            "plan": plan.to_dict() if plan else None,
            "plan_raw": getattr(self.planner, "last_raw", "") if plan else None,
            "buyer_prompt": prompt,
            "buyer_raw": raw,
            "buyer_raw_action": raw_action.to_dict(),
            "buyer_action": action.to_dict(),
            "validator": validator,
            "counterfactual_offer_response": counterfactual_response.to_dict() if counterfactual_response else None,
        }
        return action, trace

    def _maybe_apply_typed_plan_action_override(
        self,
        raw_action: ParsedAction,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        plan: Optional[StrategicPlan],
    ) -> tuple[ParsedAction, Optional[Dict[str, Any]]]:
        if self.variant != "typed_evidence_belief" or plan is None:
            return raw_action, None
        if plan.strategic_act != "accept":
            return raw_action, None
        seller_offer = last_price(history, "seller")
        target = plan.target_price
        if seller_offer is None or target is None:
            return raw_action, None
        if seller_offer > scenario.buyer_budget + 1e-9:
            return raw_action, None
        if abs(float(target) - seller_offer) > max(0.01, 0.002 * scenario.buyer_budget):
            return raw_action, None
        if raw_action.action == "deal" and raw_action.price is not None and abs(raw_action.price - seller_offer) <= 1e-6:
            return raw_action, None
        override = ParsedAction(
            role="buyer",
            action="deal",
            price=round(seller_offer, 2),
            item_spec=f"{scenario.quantity}x {scenario.codename_or_default()}",
            raw=format_action_message(
                "buyer",
                "DEAL",
                round(seller_offer, 2),
                scenario,
                "I can accept your formal offer and close now.",
            ),
            valid=True,
        )
        return override, {
            "from_action": raw_action.to_dict(),
            "to_action": override.to_dict(),
            "reason": "typed belief plan selected accept for a budget-safe formal seller [SELL] offer",
        }

    def _counterfactual_reranker_candidate_prices(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        base_plan: StrategicPlan,
    ) -> List[float]:
        """Offer set for full-framework + counterfactual reranking.

        Always include the full-framework target, then compare it with a
        lower surplus-preserving probe and, when useful, a slightly safer
        concession. This keeps the response model from silently replacing the
        framework with high-price deal maximization.
        """

        schedule = concession_schedule(scenario, history, round_id, max_turns)
        budget = float(scenario.buyer_budget)
        cap = min(budget, float(schedule["max_offer"]))
        base_target = base_plan.target_price if base_plan.target_price is not None else schedule["target_offer"]
        base_target = round(max(float(schedule["min_offer"]), min(cap, float(base_target))), 2)
        values = {
            base_target,
            round(max(float(schedule["min_offer"]), base_target - max(1.0, 0.035 * budget)), 2),
            round(min(cap, base_target + max(1.0, 0.025 * budget)), 2),
        }
        last_buyer = last_price(history, "buyer")
        if last_buyer is not None:
            values.add(round(min(cap, max(float(schedule["min_offer"]), last_buyer + max(1.0, 0.025 * budget))), 2))
        last_seller = last_price(history, "seller")
        if last_seller is not None and last_seller <= budget + 1e-9:
            values.add(round(min(cap, max(float(schedule["min_offer"]), last_seller * 0.90)), 2))
            if round_id >= max_turns - 2 or self._seller_finality_signal(history):
                values.add(round(min(cap, last_seller), 2))

        candidates = sorted({round(max(0.01, min(cap, float(price))), 2) for price in values})
        if base_target not in candidates:
            candidates.append(base_target)
            candidates = sorted(candidates)
        top_k = max(1, getattr(self.cf_response_model, "candidate_top_k", 3))
        if len(candidates) <= top_k:
            return candidates
        selected = [base_target]
        lower = [price for price in candidates if price < base_target]
        higher = [price for price in candidates if price > base_target]
        if lower:
            selected.append(max(lower))
        if len(selected) < top_k and higher:
            selected.append(min(higher))
        for price in candidates:
            if len(selected) >= top_k:
                break
            if price not in selected:
                selected.append(price)
        return sorted(selected[:top_k])

    def _belief(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> BeliefState:
        context = {
            "task": "RLVR negotiation buyer-side opponent belief",
            "item": scenario.title,
            "buyer_private_budget": scenario.buyer_budget,
            "max_turns": max_turns,
            "round": round_id,
            "rules": [
                "seller has a private cost and is regulated not to accept below cost",
                "buyer receives -1 for offering or accepting above private budget",
                "no deal/quit/timeout receives 0",
            ],
        }
        return self.belief_model.predict(
            context=context,
            conversation_history=history_to_agenticpay(history),
            current_state={
                "round": round_id,
                "last_seller_offer": last_price(history, "seller"),
                "last_buyer_offer": last_price(history, "buyer"),
            },
        )

    def _typed_evidence_belief(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> BeliefState:
        last_seller = last_price(history, "seller")
        last_buyer = last_price(history, "buyer")
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        candidate_prices = self._typed_belief_candidate_prices(scenario, history, schedule, round_id, max_turns)
        prompt = f"""You are a typed, evidence-grounded opponent belief model for an RLVR buyer.

Your output will be evaluated for calibration and evidence grounding. Do not write a buyer message.

Buyer private budget: {scenario.buyer_budget:.2f}
Item list/reference price: {scenario.reference_price:.2f}
Current round: {round_id}/{max_turns}
Last buyer offer: {last_buyer}
Last formal seller offer: {last_seller}
Concession schedule for buyer this round:
{json.dumps(schedule, ensure_ascii=False)}
Candidate buyer offer prices to calibrate:
{json.dumps(candidate_prices, ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Return only valid JSON with this schema:
{{
  "seller_reservation_range": {{"low": <number>, "high": <number>, "confidence": <0-1>}},
  "likely_acceptable_price_range": {{"low": <number>, "high": <number>, "confidence": <0-1>}},
  "seller_flexibility": {{"range": [<0-1>, <0-1>], "confidence": <0-1>}},
  "seller_patience": {{"range": [<0-1>, <0-1>], "confidence": <0-1>}},
  "walkaway_risk": <0-1>,
  "accept_probability_curve": [
    {{"price": <one candidate price>, "accept_prob": <0-1>, "confidence": <0-1>, "evidence": "brief evidence or no direct evidence yet"}},
    {{"price": <one candidate price>, "accept_prob": <0-1>, "confidence": <0-1>, "evidence": "brief evidence or no direct evidence yet"}}
  ],
  "walkaway_risk_next_turn": <0-1>,
  "evidence": [
    {{"turn": <integer or null>, "quote": "short exact observed evidence", "supports": "which field this supports"}}
  ],
  "uncertainty_notes": ["short caveats"]
}}

Rules:
- The seller's true cost is hidden. Use ranges and confidence; do not invent certainty.
- Evidence must come from the actual conversation, or say there is no direct dialogue evidence yet.
- Do not treat buyer budget as seller cost. It only caps buyer actions.
- Include every candidate price exactly once in accept_probability_curve.
- Keep accept probabilities roughly monotonic: higher buyer offers usually have higher acceptance probability.
- Keep confidence low before observing a seller offer. Confidence is not the same as accept probability.
- Prefer uncertainty over overclaiming. If no direct seller evidence exists, explicitly say so in evidence.
- If no seller price has been observed, keep confidence low.
- If the seller made a formal [SELL] offer, treat that price as strong evidence of near-term acceptability, but not as the hidden cost.
- If the seller rejected a lower buyer offer, do not assign high accept probability to the same or lower offer unless later evidence shows concession."""
        self.typed_evidence_belief_raw = self.client.generate(
            prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=min(self.max_tokens, 1400),
        )
        try:
            parsed = json_from_text(self.typed_evidence_belief_raw)
        except Exception:
            parsed = {}
        reservation = self._range_object(parsed.get("seller_reservation_range"))
        acceptable = self._range_object(parsed.get("likely_acceptable_price_range"))
        reservation_conf = self._bounded_float(
            (parsed.get("seller_reservation_range") or {}).get("confidence")
            if isinstance(parsed.get("seller_reservation_range"), dict)
            else parsed.get("seller_reservation_confidence"),
            0.25,
        )
        acceptable_conf = self._bounded_float(
            (parsed.get("likely_acceptable_price_range") or {}).get("confidence")
            if isinstance(parsed.get("likely_acceptable_price_range"), dict)
            else None,
            reservation_conf,
        )
        flexibility = self._range_midpoint(parsed.get("seller_flexibility"), default=0.5)
        patience = self._range_midpoint(parsed.get("seller_patience"), default=0.5)
        curve = self._normalize_accept_probability_curve(
            parsed.get("accept_probability_curve"),
            candidate_prices,
            scenario,
            history,
            schedule,
        )
        curve, calibration_notes = self._calibrate_typed_accept_curve(
            curve,
            scenario,
            history,
            round_id,
            max_turns,
        )
        evidence_items = parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else []
        evidence_strings = []
        for item in evidence_items:
            if isinstance(item, dict):
                quote = str(item.get("quote") or "").strip()
                supports = str(item.get("supports") or "").strip()
                if quote or supports:
                    evidence_strings.append(f"{quote} -> {supports}".strip(" ->"))
            elif item:
                evidence_strings.append(str(item))
        if not evidence_strings:
            evidence_strings = ["No direct seller offer evidence yet; keep confidence low."]
        if reservation is None:
            low = min(scenario.buyer_budget, scenario.reference_price) * 0.45
            high = min(scenario.buyer_budget, scenario.reference_price) * 0.85
            reservation = [round(low, 2), round(max(low, high), 2)]
        if acceptable is None:
            acceptable = [
                round(min(scenario.buyer_budget, max(reservation[0], schedule["target_offer"])), 2),
                round(min(scenario.buyer_budget, max(reservation[1], schedule["max_offer"])), 2),
            ]
        uncertainty_notes = parsed.get("uncertainty_notes") if isinstance(parsed.get("uncertainty_notes"), list) else []
        uncertainty_notes = list(uncertainty_notes) + calibration_notes
        return BeliefState(
            seller_reservation_range=reservation,
            seller_reservation_confidence=reservation_conf,
            seller_reservation_estimate=round((reservation[0] + reservation[1]) / 2.0, 2),
            seller_flexibility_range=[round(max(0.0, flexibility - 0.2), 3), round(min(1.0, flexibility + 0.2), 3)],
            seller_patience_range=[round(max(0.0, patience - 0.2), 3), round(min(1.0, patience + 0.2), 3)],
            seller_dominance_range=None,
            seller_friendliness_range=None,
            seller_flexibility=round(flexibility, 3),
            seller_patience=round(patience, 3),
            seller_strategy="typed_evidence_grounded",
            deal_risk=self._bounded_float(parsed.get("walkaway_risk_next_turn", parsed.get("walkaway_risk")), 0.45),
            last_seller_offer=last_seller,
            likely_acceptable_price_range=acceptable,
            contract_term_preferences={
                "accept_probability_curve": curve,
                "acceptable_range_confidence": acceptable_conf,
                "structured_evidence": evidence_items,
                "calibration_notes": calibration_notes,
                "uncertainty_notes": uncertainty_notes,
            },
            evidence=evidence_strings,
        )

    @staticmethod
    def _bounded_float(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _range_object(value: Any) -> Optional[List[float]]:
        if isinstance(value, dict):
            low = value.get("low")
            high = value.get("high")
        elif isinstance(value, list) and len(value) == 2:
            low, high = value
        else:
            return None
        try:
            low_f = float(low)
            high_f = float(high)
        except (TypeError, ValueError):
            return None
        return [round(min(low_f, high_f), 2), round(max(low_f, high_f), 2)]

    @classmethod
    def _distribution_midpoint(cls, value: Any, default: float) -> float:
        if not isinstance(value, dict):
            return default
        if "mid" in value:
            return cls._bounded_float(value.get("mid"), default)
        if "medium" in value:
            return cls._bounded_float(value.get("medium"), default)
        numeric = [cls._bounded_float(v, default) for v in value.values()]
        return sum(numeric) / len(numeric) if numeric else default

    @staticmethod
    def _typed_belief_candidate_prices(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        schedule: Dict[str, Any],
        round_id: int,
        max_turns: int,
    ) -> List[float]:
        budget = float(scenario.buyer_budget)
        values = {
            float(schedule["min_offer"]),
            float(schedule["target_offer"]),
            min(float(schedule["max_offer"]), float(schedule["target_offer"]) + max(0.01, budget * 0.03)),
        }
        last_buyer = last_price(history, "buyer")
        if last_buyer is not None:
            values.add(min(float(schedule["max_offer"]), last_buyer + max(0.01, budget * 0.025)))
        last_seller = last_price(history, "seller")
        if last_seller is not None:
            if last_seller <= budget:
                values.add(min(float(schedule["accept_cap"]), last_seller))
                if round_id >= max_turns - 2:
                    values.add(min(budget, last_seller))
            values.add(min(float(schedule["max_offer"]), last_seller * 0.72))
            values.add(min(float(schedule["max_offer"]), last_seller * 0.82))
        if round_id >= max_turns - 1:
            values.add(min(budget, float(schedule["accept_cap"])))
        prices = sorted(round(max(0.01, min(budget, value)), 2) for value in values)
        deduped: List[float] = []
        for price in prices:
            if price not in deduped:
                deduped.append(price)
        return deduped[:6]

    @classmethod
    def _range_midpoint(cls, value: Any, default: float) -> float:
        if isinstance(value, dict) and isinstance(value.get("range"), list) and len(value["range"]) == 2:
            try:
                low, high = float(value["range"][0]), float(value["range"][1])
                return max(0.0, min(1.0, (low + high) / 2.0))
            except (TypeError, ValueError):
                return default
        return cls._distribution_midpoint(value, default)

    @classmethod
    def _normalize_accept_probability_curve(
        cls,
        raw_curve: Any,
        candidate_prices: Sequence[float],
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        schedule: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        raw_items = raw_curve if isinstance(raw_curve, list) else []
        by_price: Dict[float, Dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                price = float(item.get("price"))
            except (TypeError, ValueError):
                continue
            nearest = min(candidate_prices, key=lambda p: abs(p - price)) if candidate_prices else round(price, 2)
            if abs(nearest - price) > max(1.0, scenario.buyer_budget * 0.025):
                continue
            by_price[round(nearest, 2)] = {
                "price": round(nearest, 2),
                "accept_prob": cls._bounded_float(item.get("accept_prob"), 0.35),
                "confidence": cls._bounded_float(item.get("confidence"), 0.25),
                "evidence": str(item.get("evidence") or item.get("rationale") or "no direct evidence yet"),
            }
        if len(by_price) < len(candidate_prices):
            last_seller = last_price(history, "seller")
            if last_seller is None:
                center = min(float(schedule["max_offer"]), float(schedule["target_offer"]) + scenario.buyer_budget * 0.10)
                confidence = 0.2
            else:
                center = min(float(schedule["max_offer"]), last_seller * 0.86)
                confidence = 0.35
            width = max(1.0, scenario.buyer_budget * 0.08)
            for price in candidate_prices:
                p = round(price, 2)
                if p in by_price:
                    continue
                accept_prob = 1.0 / (1.0 + math.exp(-(p - center) / width))
                by_price[p] = {
                    "price": p,
                    "accept_prob": round(max(0.03, min(0.9, accept_prob)), 3),
                    "confidence": confidence,
                    "evidence": "heuristic fallback; no complete calibrated curve returned",
                }
        return [by_price[round(price, 2)] for price in candidate_prices if round(price, 2) in by_price]

    @classmethod
    def _calibrate_typed_accept_curve(
        cls,
        curve: List[Dict[str, Any]],
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Calibrate typed accept probabilities against observed negotiation evidence.

        The LLM may overstate confidence or forget that the seller has already
        rejected a lower offer. This post-process keeps the curve monotonic while
        injecting simple evidence-based constraints from formal seller offers and
        rejections.
        """

        notes: List[str] = []
        if not curve:
            return curve, notes
        seller_offer = last_price(history, "seller")
        rejected_floor = cls._latest_rejected_buyer_offer(history)
        final_signal = cls._seller_finality_signal(history)
        has_seller_evidence = seller_offer is not None or rejected_floor is not None
        calibrated: List[Dict[str, Any]] = []
        for raw_item in sorted(curve, key=lambda item: float(item.get("price", 0.0))):
            item = dict(raw_item)
            price = float(item.get("price", 0.0))
            prob = cls._bounded_float(item.get("accept_prob"), 0.35)
            confidence = cls._bounded_float(item.get("confidence"), 0.25)

            if not has_seller_evidence:
                prob = min(prob, 0.65)
                confidence = min(confidence, 0.25)

            if rejected_floor is not None and price <= rejected_floor + 1e-9:
                cap = 0.18 if final_signal else 0.30
                if prob > cap:
                    notes.append(
                        f"capped accept_prob at ${price:.2f} because seller already rejected buyer offer near ${rejected_floor:.2f}"
                    )
                prob = min(prob, cap)
                confidence = max(confidence, 0.35)

            if seller_offer is not None and seller_offer <= scenario.buyer_budget + 1e-9:
                gap = seller_offer - price
                if price >= seller_offer - 1e-9:
                    prob = max(prob, 0.82)
                    confidence = max(confidence, 0.60)
                elif round_id >= max_turns - 2 and gap <= max(1.0, 0.025 * scenario.buyer_budget):
                    prob = max(prob, 0.62)
                    confidence = max(confidence, 0.50)
                elif not final_signal and gap <= max(1.0, 0.06 * scenario.buyer_budget):
                    prob = max(prob, 0.42)
                    confidence = max(confidence, 0.40)

            if final_signal and seller_offer is not None and price < seller_offer * 0.95:
                prob = min(prob, 0.35)
                notes.append(
                    f"reduced accept_prob at ${price:.2f} because seller signaled final/floor near ${seller_offer:.2f}"
                )

            item["accept_prob"] = round(max(0.02, min(0.95, prob)), 3)
            item["confidence"] = round(max(0.05, min(0.9, confidence)), 3)
            item["evidence"] = str(item.get("evidence") or "")
            calibrated.append(item)

        prev = 0.0
        for item in calibrated:
            prob = max(prev, float(item["accept_prob"]))
            item["accept_prob"] = round(min(0.95, prob), 3)
            prev = item["accept_prob"]
        if seller_offer is not None:
            notes.append(f"formal seller offer observed at ${seller_offer:.2f}; calibrated curve around observed ask")
        if not has_seller_evidence:
            notes.append("no seller price/rejection evidence yet; capped confidence and high accept probabilities")
        return calibrated, list(dict.fromkeys(notes))

    @staticmethod
    def _latest_rejected_buyer_offer(history: Sequence[Dict[str, Any]]) -> Optional[float]:
        last_buyer_offer: Optional[float] = None
        latest_rejected: Optional[float] = None
        for turn in history:
            role = turn.get("role")
            action = turn.get("action") or {}
            if role == "buyer" and action.get("action") == "offer" and action.get("price") is not None:
                last_buyer_offer = float(action["price"])
            elif role == "seller" and action.get("action") in {"reject", "walk_away"} and last_buyer_offer is not None:
                latest_rejected = last_buyer_offer
        return latest_rejected

    @staticmethod
    def _seller_finality_signal(history: Sequence[Dict[str, Any]]) -> bool:
        final_words = ("final", "floor", "no lower", "won't go lower", "cannot go lower", "take it or leave")
        for turn in reversed(history):
            if turn.get("role") != "seller":
                continue
            message = str(turn.get("message") or "").lower()
            return any(word in message for word in final_words)
        return False

    def _adapt_belief_style(
        self,
        belief: BeliefState,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> BeliefState:
        """Expose external-belief baselines under the same RLVR interface."""

        if self.variant == "astra_style_belief":
            belief.seller_strategy = "astra_style_fairness_stance"
            belief.contract_term_preferences = {
                **belief.contract_term_preferences,
                "fairness": "prefers a mutually acceptable discount from list price",
                "stance": "firm" if belief.deal_risk >= 0.55 else "cooperative",
            }
            belief.evidence.append("ASTRA-style belief tracks fairness/stance rather than a hard seller floor.")
        elif self.variant == "bond_style_posterior":
            belief.seller_strategy = "bond_style_latent_type_posterior"
            firm = min(0.85, max(0.1, belief.deal_risk))
            flexible = 1.0 - firm
            belief.contract_term_preferences = {
                **belief.contract_term_preferences,
                "latent_type_posterior": {
                    "firm_high_margin_seller": round(firm, 3),
                    "flexible_quick_sale_seller": round(flexible, 3),
                },
            }
            belief.evidence.append("BOND-style posterior approximates latent seller type from counters/rejections.")
        elif self.variant == "preference_estimation":
            belief.seller_strategy = "preference_estimation_cue_model"
            seller_prices = [
                item.get("action", {}).get("price")
                for item in history
                if item.get("role") == "seller" and item.get("action", {}).get("price") is not None
            ]
            concession_signal = 0.5
            if len(seller_prices) >= 2:
                first = float(seller_prices[0])
                last = float(seller_prices[-1])
                concession_signal = max(0.0, min(1.0, (first - last) / max(1.0, first)))
            belief.contract_term_preferences = {
                **belief.contract_term_preferences,
                "cue_probabilities": {
                    "will_concede_next": round(0.25 + 0.55 * concession_signal, 3),
                    "will_accept_budget_cap": round(1.0 - min(0.9, belief.deal_risk), 3),
                },
            }
            belief.evidence.append("Preference-estimation baseline converts concession cues into simple probabilities.")
        elif self.variant == "typed_belief":
            belief.seller_strategy = "typed_price_belief"
            belief.evidence.append("Typed-belief baseline uses intervals/confidence/evidence without separate LLM planner.")
        elif self.variant == "typed_evidence_belief":
            belief.seller_strategy = "typed_evidence_grounded"
            belief.evidence.append("Typed-evidence belief uses structured evidence spans and accept-probability curve.")
        return belief

    def _plan(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        planner_policy = [
            "Belief estimates are noisy evidence, not hard constraints on seller cost or budget.",
            "Do not walk away only because the belief model predicts a high seller floor.",
            "Before the final round, prefer probing/counteroffering over quitting when the buyer budget is not violated.",
            "In the first two buyer turns, gather seller information in Talk and still make a concrete low counteroffer.",
            "Optimize buyer reward: lower accepted prices are better; do not rush to the buyer budget.",
            "Follow the concession schedule unless accepting a lower formal seller offer.",
            "Rounds 1-2 should anchor low; only approach the budget cap in the last rounds if needed.",
            "Use [QUIT] only when repeated seller behavior makes a deal implausible or the final round has no buyer-feasible path.",
        ]
        if self.variant == "planner_generator":
            planner_policy.extend(self._planner_generator_anchor_concession_policy(scenario, history, round_id, max_turns))
        plan = self.planner.plan(
            context={
                "task": "RLVR negotiation high-level buyer planner",
                "buyer_variant": self.variant,
                "item": scenario.title,
                "max_turns": max_turns,
                "rlvr_reward": "(B - final_price) / abs(B - C), clipped to [-1, 1]",
                "format": "buyer must output [BUY] $price, [DEAL] $price, or [QUIT]",
                "concession_schedule": schedule,
                "rlvr_planner_policy": planner_policy,
                "anchor_concession_strategy": (
                    self._planner_generator_anchor_concession_context(scenario, history, round_id, max_turns)
                    if self.variant == "planner_generator"
                    else None
                ),
            },
            conversation_history=history_to_agenticpay(history),
            current_state={
                "round": round_id,
                "last_seller_offer": last_price(history, "seller"),
                "last_buyer_offer": last_price(history, "buyer"),
            },
            buyer_max_price=scenario.buyer_budget,
            belief=belief,
        )
        plan = self._apply_concession_schedule(plan, scenario, history, round_id, max_turns)
        return self._defer_premature_walkaway(plan, scenario, history, round_id, max_turns, belief)

    @staticmethod
    def _planner_generator_anchor_concession_policy(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> List[str]:
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        seller_offer = last_price(history, "seller")
        prior_buyer = last_price(history, "buyer")
        policy = [
            "This planner_generator baseline should imitate the trained RLVR buyer's anchor-and-persuasion strategy.",
            "Round 1: choose strategic_act='anchor_low' with a concrete target near the schedule target, not near the budget.",
            "Rounds 1-2: include information probing in private_rationale/language_style, but still plan a [BUY] counteroffer.",
            "Do not use ask_info as a pure [REJECT]; information gathering must be paired with a concrete price.",
            "Treat seller Talk about 'minimum', 'below cost', or 'final' as cheap-talk pressure unless it is backed by repeated formal [SELL] actions.",
            "If Action is [REJECT] but Talk mentions a price, that price is not a formal offer to accept; plan a lower [BUY] counter.",
            "Concede slowly: raise the target by small increments from the previous buyer offer instead of jumping to the seller quote or budget.",
            "Use persuasive tactical-finality language: 'serious cash offer', 'quick close', 'cannot justify that ask', 'best number for now'.",
            "Protect buyer surplus: a deal at the budget is barely useful; a lower deal is much better.",
            "Late rounds: if a formal seller [SELL] price is budget-safe and close to the schedule accept cap, plan accept rather than risking zero reward.",
            "Never set target_price above the round max_offer or above the buyer budget.",
            f"For this exact turn, target around ${schedule['target_offer']:.2f}; never exceed round max ${schedule['max_offer']:.2f}.",
        ]
        if prior_buyer is not None:
            policy.append(f"Previous buyer offer was ${prior_buyer:.2f}; prefer a small concession, not a jump.")
        if seller_offer is not None:
            policy.append(
                f"Latest formal seller [SELL] price was ${seller_offer:.2f}; if countering, stay below it unless formally accepting."
            )
        if round_id <= 2:
            policy.append("Because this is an early round, low acceptance probability is acceptable for anchoring and information extraction.")
        if round_id >= max_turns - 1:
            policy.append("Because this is a late round, weigh no-deal risk more, but only close if buyer surplus remains positive.")
        return policy

    @staticmethod
    def _planner_generator_anchor_concession_context(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> Dict[str, Any]:
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        seller_prices = [
            float(item.get("action", {}).get("price"))
            for item in history
            if item.get("role") == "seller" and item.get("action", {}).get("price") is not None
        ]
        buyer_prices = [
            float(item.get("action", {}).get("price"))
            for item in history
            if item.get("role") == "buyer" and item.get("action", {}).get("price") is not None
        ]
        return {
            "strategy_name": "anchor_low_slow_concession_persuasion",
            "budget_is_ceiling_not_target": True,
            "round": round_id,
            "max_turns": max_turns,
            "schedule_target": schedule["target_offer"],
            "schedule_min_offer": schedule["min_offer"],
            "schedule_max_offer": schedule["max_offer"],
            "schedule_accept_cap": schedule["accept_cap"],
            "latest_formal_seller_offer": seller_prices[-1] if seller_prices else None,
            "lowest_formal_seller_offer": min(seller_prices) if seller_prices else None,
            "latest_buyer_offer": buyer_prices[-1] if buyer_prices else None,
            "recommended_strategic_act": (
                "anchor_low"
                if round_id == 1
                else ("concede_small" if round_id < max_turns - 1 else "concede_medium_or_accept_formal_offer")
            ),
            "language_style": "firm_persuasive_tactical_finality",
        }

    def _typed_belief_plan(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        target = schedule["target_offer"]
        scored_candidates: List[Dict[str, Any]] = []
        last_seller_offer = last_price(history, "seller")
        seller_final_signal = self._seller_finality_signal(history)
        if belief and belief.likely_acceptable_price_range:
            target = min(schedule["max_offer"], max(schedule["min_offer"], belief.likely_acceptable_price_range[0] * 0.96))
        curve = []
        if belief:
            curve = belief.contract_term_preferences.get("accept_probability_curve") or []
        if isinstance(curve, list) and curve:
            best_score = -1e9
            best_price = target
            for item in curve:
                if not isinstance(item, dict):
                    continue
                try:
                    price = float(item.get("price"))
                except (TypeError, ValueError):
                    continue
                if price > min(scenario.buyer_budget, schedule["max_offer"]) + 1e-9:
                    continue
                accept_prob = self._bounded_float(item.get("accept_prob"), 0.35)
                confidence = self._bounded_float(item.get("confidence"), 0.25)
                surplus = scenario.buyer_budget - price
                walkaway = belief.deal_risk if belief else 0.45
                unnecessary_concession = max(0.0, price - schedule["target_offer"])
                score = (
                    accept_prob * surplus
                    - 0.10 * walkaway * scenario.buyer_budget
                    - 0.08 * unnecessary_concession
                    - 0.03 * (1.0 - confidence) * scenario.buyer_budget
                )
                if round_id <= 2:
                    score -= 0.10 * unnecessary_concession
                if (
                    last_seller_offer is not None
                    and last_seller_offer <= scenario.buyer_budget + 1e-9
                    and price >= last_seller_offer - max(1.0, 0.025 * scenario.buyer_budget)
                    and round_id >= max_turns - 2
                ):
                    score += 0.12 * scenario.buyer_budget
                if (
                    seller_final_signal
                    and last_seller_offer is not None
                    and price < last_seller_offer * 0.95
                    and round_id >= 3
                ):
                    score -= 0.10 * scenario.buyer_budget
                scored_candidates.append({
                    **item,
                    "buyer_surplus": round(surplus, 4),
                    "planner_score": round(score, 4),
                })
                if score > best_score:
                    best_score = score
                    best_price = price
            if scored_candidates:
                target = min(schedule["max_offer"], max(schedule["min_offer"], best_price))
        formal_accept = self._typed_formal_offer_accept_plan(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
            schedule=schedule,
            belief=belief,
            scored_candidates=scored_candidates,
            last_seller_offer=last_seller_offer,
            seller_final_signal=seller_final_signal,
        )
        if formal_accept is not None:
            return formal_accept
        return StrategicPlan(
            strategic_act="probe" if round_id <= 2 else ("concede_small" if round_id < max_turns - 1 else "concede_medium"),
            target_price=round(target, 2),
            reservation_guardrail=scenario.buyer_budget,
            accept_if_at_or_below=min(scenario.buyer_budget, schedule["accept_cap"]),
            seller_feasible_price_floor=belief.likely_acceptable_price_range[0] if belief and belief.likely_acceptable_price_range else None,
            seller_feasibility_risk=belief.deal_risk if belief else 0.5,
            seller_feasible_non_price_terms={"scored_accept_probability_curve": scored_candidates},
            feasibility_checks=["price <= buyer budget", "follow concession schedule", "belief is noisy"],
            seller_feasibility_guardrails=[
                "belief curve is probabilistic, not a hard seller floor",
                "maximize buyer surplus subject to plausible acceptance",
                "do not jump to schedule max in early rounds",
            ],
            private_rationale="Typed evidence-grounded belief chooses a schedule-capped candidate by expected buyer surplus.",
        )

    def _typed_formal_offer_accept_plan(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        schedule: Dict[str, Any],
        belief: Optional[BeliefState],
        scored_candidates: List[Dict[str, Any]],
        last_seller_offer: Optional[float],
        seller_final_signal: bool,
    ) -> Optional[StrategicPlan]:
        if last_seller_offer is None or last_seller_offer > scenario.buyer_budget + 1e-9:
            return None
        if round_id <= 2:
            return None
        denominator = max(1.0, abs(scenario.buyer_budget - scenario.seller_cost))
        seller_offer_reward = max(0.0, min(1.0, (scenario.buyer_budget - last_seller_offer) / denominator))
        exact_prob = None
        near_prob = None
        for item in scored_candidates:
            try:
                price = float(item.get("price"))
            except (TypeError, ValueError):
                continue
            prob = self._bounded_float(item.get("accept_prob"), 0.0)
            if abs(price - last_seller_offer) <= max(0.01, 0.002 * scenario.buyer_budget):
                exact_prob = prob if exact_prob is None else max(exact_prob, prob)
            if price >= last_seller_offer - max(1.0, 0.025 * scenario.buyer_budget):
                near_prob = prob if near_prob is None else max(near_prob, prob)

        late_round = round_id >= max_turns - 2
        final_round = round_id >= max_turns
        strong_accept_evidence = (exact_prob is not None and exact_prob >= 0.65) or (near_prob is not None and near_prob >= 0.72)
        attractive_budget_safe_offer = seller_offer_reward >= 0.55
        final_or_floor = seller_final_signal or final_round
        if not (final_round or (late_round and strong_accept_evidence) or (final_or_floor and attractive_budget_safe_offer)):
            return None

        return StrategicPlan(
            strategic_act="accept",
            target_price=round(last_seller_offer, 2),
            reservation_guardrail=scenario.buyer_budget,
            concession_size="none",
            accept_if_at_or_below=round(last_seller_offer, 2),
            seller_feasible_price_floor=belief.likely_acceptable_price_range[0] if belief and belief.likely_acceptable_price_range else None,
            seller_feasibility_risk=belief.deal_risk if belief else 0.25,
            seller_feasible_non_price_terms={
                "scored_accept_probability_curve": scored_candidates,
                "typed_formal_offer_accept_rule": {
                    "seller_offer": round(last_seller_offer, 2),
                    "seller_offer_reward": round(seller_offer_reward, 4),
                    "exact_accept_prob": exact_prob,
                    "near_accept_prob": near_prob,
                    "late_round": late_round,
                    "seller_final_signal": seller_final_signal,
                },
            },
            feasibility_checks=[
                "formal seller [SELL] offer is within buyer budget",
                "typed belief indicates high near-term acceptability or late/final-round risk",
                "accepting preserves positive buyer reward and avoids zero-reward seller quit",
            ],
            seller_feasibility_guardrails=[
                "formal seller offer overrides default concession schedule in late rounds",
                "use [DEAL] exactly matching seller's previous [SELL] action",
                "do not make a lower [BUY] when typed plan strategic_act is accept",
            ],
            language_style="concise_close",
            private_rationale=(
                "Typed evidence-grounded belief selected acceptance because a budget-safe formal seller offer is a strong "
                "near-term deal opportunity and continued shaving risks seller quit."
            ),
        )

    def _apply_concession_schedule(
        self,
        plan: StrategicPlan,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> StrategicPlan:
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        if plan.strategic_act == "accept":
            return plan
        if plan.target_price is None:
            plan.target_price = schedule["target_offer"]
        else:
            plan.target_price = min(plan.target_price, schedule["max_offer"])
            plan.target_price = max(plan.target_price, schedule["min_offer"])
        plan.target_price = round(min(plan.target_price, scenario.buyer_budget), 2)
        plan.reservation_guardrail = min(plan.reservation_guardrail or scenario.buyer_budget, scenario.buyer_budget)
        plan.accept_if_at_or_below = min(
            plan.accept_if_at_or_below or schedule["accept_cap"],
            schedule["accept_cap"],
            scenario.buyer_budget,
        )
        plan.feasibility_checks = list(dict.fromkeys(plan.feasibility_checks + [
            f"concession schedule target <= {schedule['target_offer']:.2f}",
            f"round max offer <= {schedule['max_offer']:.2f}",
            "do not jump to full budget before late rounds",
        ]))
        plan.private_rationale = (
            (plan.private_rationale + " " if plan.private_rationale else "")
            + f"Schedule cap for round {round_id}: target {schedule['target_offer']:.2f}, max {schedule['max_offer']:.2f}."
        )
        return plan

    def _defer_premature_walkaway(
        self,
        plan: StrategicPlan,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Optional[BeliefState],
    ) -> StrategicPlan:
        """Keep full framework from treating uncertain belief as a quit trigger."""

        if plan.strategic_act != "walk_away":
            return plan
        if round_id >= max_turns:
            return plan

        last_seller = last_price(history, "seller")
        last_buyer = last_price(history, "buyer")
        seller_prices = [
            item.get("action", {}).get("price")
            for item in history
            if item.get("role") == "seller" and item.get("action", {}).get("price") is not None
        ]
        repeated_above_budget = len(seller_prices) >= 3 and min(float(price) for price in seller_prices) > scenario.buyer_budget
        reliable_no_deal = (
            repeated_above_budget
            and belief is not None
            and belief.seller_reservation_confidence >= 0.75
            and belief.deal_risk >= 0.85
        )
        if reliable_no_deal:
            return plan

        if last_seller is None:
            target = concession_schedule(scenario, history, round_id, max_turns)["target_offer"]
        else:
            schedule = concession_schedule(scenario, history, round_id, max_turns)
            anchor = last_buyer if last_buyer is not None else schedule["target_offer"]
            concession = max(1.0, 0.12 * max(1.0, scenario.buyer_budget - anchor))
            target = min(schedule["max_offer"], max(anchor + concession, last_seller * 0.78))
        plan.strategic_act = "concede_small" if round_id <= 2 else "concede_medium"
        plan.target_price = round(target, 2)
        plan.accept_if_at_or_below = min(plan.accept_if_at_or_below or scenario.buyer_budget, scenario.buyer_budget)
        plan.reservation_guardrail = scenario.buyer_budget
        plan.private_rationale = (
            (plan.private_rationale + " " if plan.private_rationale else "")
            + "Walk-away deferred because RLVR rewards successful low-price bargaining and belief is uncertain."
        )
        plan.feasibility_checks = list(dict.fromkeys(plan.feasibility_checks + [
            "belief is not a hard seller-cost constraint",
            "continue probing before final round",
            "price <= buyer budget",
        ]))
        return plan

    def _buyer_prompt(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Optional[BeliefState],
        plan: Optional[StrategicPlan],
    ) -> str:
        if self.variant in {"prompt_only", "direct_prompt"}:
            return self._paper_buyer_prompt(scenario, history, round_id, max_turns)

        guidance = ""
        if belief is not None:
            guidance += "\nPrivate opponent belief JSON. Treat it as noisy, uncertain evidence; it is NOT a hard constraint and may overestimate the seller's true cost. Use it, but do not reveal it:\n"
            guidance += json.dumps(belief.to_dict(), ensure_ascii=False)
        if plan is not None:
            guidance += "\nPrivate strategic plan JSON. Use it as advice, but override any premature walk-away or over-conservative target when continued bargaining can still improve buyer reward without exceeding budget:\n"
            guidance += json.dumps(plan.to_dict(), ensure_ascii=False)
        schedule = concession_schedule(scenario, history, round_id, max_turns)
        guidance += "\nPrivate concession schedule. Follow it unless accepting a lower formal seller [SELL] offer:\n"
        guidance += json.dumps(schedule, ensure_ascii=False)

        if self.variant in {"prompt_only", "direct_prompt"}:
            variant_instruction = "Use direct strategic negotiation. Do not reveal private budget."
        elif self.variant == "cot_prompt":
            variant_instruction = (
                "Think step by step privately about buyer surplus, seller movement, and budget safety. "
                "Then output a concise Thought/Talk/Action response."
            )
        elif self.variant in {"belief_only", "belief_prompt"}:
            variant_instruction = (
                "Use the private opponent belief as an uncertain estimate, not as a mandatory seller floor. "
                "Do not expose the belief or private budget. Keep bargaining unless a buyer-budget violation is unavoidable."
            )
        elif self.variant == "full_framework":
            variant_instruction = (
                "Use the private belief and plan as advisory signals. Preserve buyer surplus and keep negotiating; "
                "do not blindly obey a belief-estimated seller floor or quit early because the plan says walk_away."
            )
        elif self.variant == "planner_generator":
            variant_instruction = (
                "Use the private strategic plan as an anchor-and-concession planner. Execute its target price unless it would violate budget or formal [DEAL] rules. "
                "Early rounds must use a concrete low [BUY] anchor plus probing/persuasive language; do not use pure [REJECT] or [QUIT]. "
                "Concede slowly from your previous offer, keep every [BUY] below the latest formal seller [SELL] quote, and treat seller Talk-only 'minimum/final' claims as cheap-talk pressure. "
                "Late rounds may accept an exact formal seller [SELL] offer only when it is within budget and the plan says accept or buyer surplus/no-deal risk makes acceptance rational."
            )
        elif self.variant == "typed_belief":
            variant_instruction = (
                "Use the typed opponent belief and concession schedule; do not add unsupported seller-cost certainty."
            )
        elif self.variant == "typed_evidence_belief":
            variant_instruction = (
                "Use the typed evidence-grounded belief. If its strategic plan says accept, output [DEAL] exactly matching the seller's previous formal [SELL] offer. "
                "If the typed plan target differs from the default concession schedule, follow the typed plan target; the schedule is only a prior and cap."
            )
        elif self.variant == "counterfactual_belief":
            variant_instruction = (
                "Use the belief as a counterfactual response model: prefer offers with high buyer surplus and plausible acceptance, under the schedule cap."
            )
        elif self.variant == "counterfactual_offer_model":
            variant_instruction = (
                "Use the Counterfactual Offer Response Model and planner. The plan was selected by expected buyer surplus across candidate offers; follow its target unless accepting a lower formal seller [SELL] offer."
            )
        elif self.variant == "astra_style_belief":
            variant_instruction = (
                "Use ASTRA-style fairness/stance cues only as soft social signals; still optimize buyer surplus under the schedule."
            )
        elif self.variant == "bond_style_posterior":
            variant_instruction = (
                "Use the latent seller-type posterior as soft evidence; choose a schedule-capped offer that tests flexibility."
            )
        elif self.variant == "preference_estimation":
            variant_instruction = (
                "Use concession/preference cues to estimate seller movement; do not abandon the schedule."
            )
        else:
            variant_instruction = "Negotiate as a buyer."

        anti_early_walkaway_policy = f"""
RLVR buyer policy for this run:
- Your reward is higher when you close a deal at a lower price. Optimize buyer surplus, not just deal rate.
- The seller's true cost is hidden. Any belief about seller cost/reservation is uncertain and can be wrong.
- Never treat the belief model's seller_reservation_range or likely_acceptable_price_range as a price you must satisfy.
- Treat your private budget as a strict ceiling, never as your target price. Your target is the lowest price that can keep the seller engaged.
- Absolute budget law: every [BUY] or [DEAL] price must be <= ${scenario.buyer_budget:.2f}. This is a hard mathematical constraint, not a preference.
- Never output a price even one cent above ${scenario.buyer_budget:.2f}. If you are tempted to offer more, output exactly ${scenario.buyer_budget:.2f} or lower.
- Your Talk must not say you can pay above ${scenario.buyer_budget:.2f}; never say "I can go to $X" when X > ${scenario.buyer_budget:.2f}.
- In buyer rounds 1 and 2, do not use [QUIT]. Start with an aggressive but plausible low anchor near the target offer, ask for seller-side information, and still make a concrete [BUY] counteroffer.
- Useful seller-side questions: "What is the lowest workable price?", "How flexible are you today?", "Is there a warranty/condition reason for that quote?", "Can you meet me closer to my offer?"
- Pair every low offer with persuasive language: mention comparable pricing, total cash today, quick close, uncertainty about condition/warranty, or the large gap from list price. Do not only state a number.
- Regardless of how high the seller's quote is, try to bargain it down with [BUY] unless accepting a previous seller [SELL] offer is clearly good for buyer utility.
- Prefer [BUY] over [REJECT] because [BUY] reveals a concrete counteroffer and creates a chance for agreement.
- Use [DEAL] only to accept an exact previous seller Action line of type [SELL] whose price is <= ${scenario.buyer_budget:.2f}. If the seller only mentions a number in Talk while Action is [REJECT] or [QUIT], you must not use [DEAL]; make a [BUY] counteroffer instead.
- Use [QUIT] only in the final round or after repeated seller responses show no buyer-feasible movement. A high seller quote alone is not enough.
- Continue trying to shave the price down. Do not jump to the maximum allowed offer or your full budget unless it is late and necessary for a deal.
- If the private strategic plan has strategic_act="accept", do not keep shaving the price. Use [DEAL] exactly matching the previous formal seller [SELL] offer.
- If the private strategic plan target differs from the concession schedule target, follow the strategic plan target. The schedule is a default prior/cap, not a command to ignore high-confidence evidence.
- In late rounds, a budget-safe formal seller [SELL] offer with high typed-belief accept evidence is a strong deal opportunity; do not risk zero reward by countering far below it.
- Concession schedule for this exact turn:
  * target offer: ${schedule['target_offer']:.2f}
  * minimum reasonable offer: ${schedule['min_offer']:.2f}
  * maximum allowed offer this round: ${schedule['max_offer']:.2f}
  * acceptance cap for a formal seller [SELL] offer: ${schedule['accept_cap']:.2f}
- If making a [BUY] offer, choose a price near the strategic plan target when provided; otherwise use the schedule target. The maximum allowed offer is a safety cap, not a recommendation.
- If the target offer is rejected, increase in small steps. Do not immediately jump to the cap; preserve room for later persuasion.
- If the seller asks above this round's maximum, counter at or below the maximum; do not meet them halfway above the cap.
- Recommended learned strategy from RLVR-style training: low opening anchor -> ask/probe -> persuasive counteroffers -> rational small concessions -> only accept when the seller's formal [SELL] price is attractive relative to your budget.
"""

        return f"""You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than or equal to your budget. Any [BUY] or [DEAL] above your budget is a catastrophic invalid action.
Again, try to make deal with a price as low as possible. That is, your goal is to spend as little money as possible, not just reaching your budget.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of [BUY], [REJECT], [DEAL], [QUIT].
1. '[BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you wish to offer the seller $M to purchase all {scenario.quantity} items of the product with the codename "{scenario.codename_or_default()}". M must be <= ${scenario.buyer_budget:.2f}.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you finally accept on a former formal [SELL] offer proposed by the seller. $M ({scenario.quantity}x {scenario.codename_or_default()}) is a exact copy of seller's previous Action: [SELL] offer and must be <= ${scenario.buyer_budget:.2f}. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.

You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'.
'[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' can only be chosen to accept the seller's previous formal Action '[SELL] $M ({scenario.quantity}x {scenario.codename_or_default()})', and only when M <= ${scenario.buyer_budget:.2f}. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.
Before returning, perform this mandatory check:
1. Extract your Action price M.
2. If M > ${scenario.buyer_budget:.2f}, replace M with a lower counteroffer <= ${scenario.buyer_budget:.2f}.
3. If your Action is [DEAL], verify the seller's previous Action was exactly [SELL] with the same M and M <= ${scenario.buyer_budget:.2f}; otherwise use [BUY] instead.
4. Prefer a lower [BUY] price when it can still plausibly keep the negotiation alive.

Your reply should STRICTLY follow this format (not following the format will directly lead to failure), for example:
Thought: I'm a buyer and I want to preserve surplus. The listing price of codename "{scenario.codename_or_default()}" is ${scenario.reference_price:.2f}, so I should start near my private target, ask for flexibility, and keep room for later concessions.
Talk: I can move quickly, but this is well above the price range I can justify. Could you meet me closer to this starting offer?
Action: [BUY] ${schedule['target_offer']:.2f} ({scenario.quantity}x {scenario.codename_or_default()})

Inventory List:
Product 1
Codename: {scenario.codename_or_default()}
Title: "{scenario.title}"
Description: "{scenario.description_or_default()}"
Features: "{scenario.features_or_default()}"
List Price: ${scenario.reference_price:.2f}

Shopping List
codename: {scenario.codename_or_default()}
quantity: {scenario.quantity}
budget: ${scenario.buyer_budget:.2f}

Now, I play the role of seller and you play the role of buyer. We are going to negotiate based on the Inventory List in {max_turns} turns.
Current round: {round_id}/{max_turns}

{variant_instruction}
{anti_early_walkaway_policy}
{guidance}

Conversation so far:
{format_transcript(history)}

Return only your next buyer message in the required Thought/Talk/Action format."""

    def _paper_buyer_prompt(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> str:
        return f"""You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget, otherwise, you should quit negotiating.
Again, try to make deal with a price as low as possible. That is, your goal is to spend as little money as possible, not just reaching your budget.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of [BUY], [REJECT], [DEAL], [QUIT].
1. '[BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you wish to offer the seller $M to purchase all {scenario.quantity} items of the product with the codename "{scenario.codename_or_default()}".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you finally accept on a former offer proposed by the seller. $M ({scenario.quantity}x {scenario.codename_or_default()}) is a exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.

You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})' or '[REJECT]'.
'[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' can only be chosen to accept the seller's previous offer '[SELL] $M ({scenario.quantity}x {scenario.codename_or_default()})'. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.

Your reply should STRICTLY follow this format (not following the format will directly lead to failure), for example:
Thought: I'm a buyer and I want to bargain. The listing price of codename "{scenario.codename_or_default()}" is ${scenario.reference_price:.2f}, which is too expensive, so I try to buy it for ${min(scenario.reference_price, scenario.buyer_budget) * 0.67:.2f}.
Talk: Hello, I'm tight on budget. Can you sell it for ${min(scenario.reference_price, scenario.buyer_budget) * 0.67:.2f}?
Action: [BUY] ${min(scenario.reference_price, scenario.buyer_budget) * 0.67:.2f} ({scenario.quantity}x {scenario.codename_or_default()})

Inventory List:
Product 1
Codename: {scenario.codename_or_default()}
Title: "{scenario.title}"
Description: "{scenario.description_or_default()}"
Features: "{scenario.features_or_default()}"
List Price: ${scenario.reference_price:.2f}

Shopping List
codename: {scenario.codename_or_default()}
quantity: {scenario.quantity}
budget: ${scenario.buyer_budget:.2f}

Now, I play the role of seller and you play the role of buyer. We are going to negotiate based on the Inventory List in {max_turns} turns.
Current round: {round_id}/{max_turns}

Conversation so far:
{format_transcript(history)}

Return only your next buyer message in the required Thought/Talk/Action format."""


class RegulatedLLMSeller:
    def __init__(self, client: ModelClient, max_tokens: int, temperature: float, top_p: float, persona: str):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.persona = persona

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> tuple[ParsedAction, str]:
        buyer_offer = last_price(history, "buyer")
        persona_block = seller_persona_block(self.persona)
        prompt = f"""You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product, instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of [SELL], [REJECT], [DEAL], [QUIT].
1. '[SELL] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you want to propose selling {scenario.quantity} items of the product with the codename "{scenario.codename_or_default()}" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' if you finally agree on a former offer proposed by the buyer, and sell {scenario.quantity} items of the product with the codename "{scenario.codename_or_default()}" to the buyer for the total price of $M. $M ({scenario.quantity}x {scenario.codename_or_default()}) is an exact copy of the buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.

You shouldn't choose action '[DEAL]' before the buyer's action '[BUY]'.
'[DEAL] $M ({scenario.quantity}x {scenario.codename_or_default()})' can only be chosen to accept the buyer's previous offer '[BUY] $M ({scenario.quantity}x {scenario.codename_or_default()})'. Otherwise, you always choose from '[SELL]', '[REJECT]' and '[QUIT]'.
{persona_block}

Your reply should strictly follow this format, for example:
Thought: I'm a seller, so I must sell the product with codename "{scenario.codename_or_default()}" higher than its cost.
Talk: I can offer it at a competitive price.
Action: [SELL] ${max(scenario.seller_cost, scenario.reference_price):.2f} ({scenario.quantity}x {scenario.codename_or_default()})

Inventory List:
Product 1
Codename: {scenario.codename_or_default()}
Title: "{scenario.title}"
Description: "{scenario.description_or_default()}"
Features: "{scenario.features_or_default()}"
List Price: ${scenario.reference_price:.2f}
Cost: ${scenario.seller_cost:.2f}

Now, I play the role of buyer and you play the role of seller. We are going to negotiate based on the Inventory List in {max_turns} turns.
Current round: {round_id}/{max_turns}

Conversation so far:
{format_transcript(history)}

Return only your next seller message in the required Thought/Talk/Action format."""
        raw = self.client.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        parsed = parse_action("seller", raw)
        if not parsed.valid or parsed.action not in {"offer", "accept", "reject", "walk_away"}:
            fallback_price = max(scenario.seller_cost, seller_concession_price(scenario, round_id, max_turns))
            raw = format_action_message("seller", "SELL", fallback_price, scenario, "I can make a counteroffer that stays above my cost.")
            parsed = parse_action("seller", raw)
        if parsed.action == "accept":
            if (
                parsed.price is None
                or buyer_offer is None
                or abs(parsed.price - buyer_offer) > 1e-6
                or parsed.price <= scenario.seller_cost + 1e-9
            ):
                fallback_price = max(scenario.seller_cost, seller_concession_price(scenario, round_id, max_turns))
                raw = format_action_message("seller", "SELL", fallback_price, scenario, "I cannot accept that offer, but I can counter.")
                parsed = parse_action("seller", raw)
        if parsed.action == "offer":
            if parsed.price is None or parsed.price < scenario.seller_cost - 1e-9:
                price = max(scenario.seller_cost, seller_concession_price(scenario, round_id, max_turns))
                raw = format_action_message("seller", "SELL", price, scenario, "I need to keep the price above my cost.")
                parsed = parse_action("seller", raw)
        return parsed, raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer-model", required=True, help="Model spec, e.g. hf:/path or openai:model.")
    parser.add_argument("--seller-model", required=True, help="Model spec for regulated fixed seller.")
    parser.add_argument("--model-alias", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--buyer-variants", default="prompt_only,belief_only,full_framework")
    parser.add_argument(
        "--variant-schedule",
        choices=["sequential", "round_robin"],
        default="sequential",
        help="Run each variant to completion, or interleave variants episode by episode.",
    )
    parser.add_argument("--num-test-instances", type=int, default=128)
    parser.add_argument("--rollouts-per-instance", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--buyer-temperature", type=float, default=1.0)
    parser.add_argument("--seller-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive-gains-probability", type=float, default=0.7)
    parser.add_argument("--scenarios-jsonl", default=None)
    parser.add_argument(
        "--require-scenarios-jsonl",
        action="store_true",
        help="Fail instead of falling back to synthetic scenarios. Use this for paper-table reproduction.",
    )
    parser.add_argument("--seller-persona", choices=["neutral", "begging", "insulting", "unyielding"], default="neutral")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--openai-request-timeout",
        type=float,
        default=1800,
        help="Per-request timeout for OpenAI-compatible servers, in seconds.",
    )
    parser.add_argument(
        "--episode-retries",
        type=int,
        default=2,
        help="Retry a failed episode this many times before recording it as failed and continuing.",
    )
    parser.add_argument(
        "--episode-retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Initial sleep before retrying a failed episode; doubled after each failed retry.",
    )
    parser.add_argument(
        "--max-episode-errors-per-variant",
        type=int,
        default=20,
        help="Stop trying a variant in this run after this many episode-level exceptions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip episodes already present in variant JSONL files.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When --resume is set, retry episodes that were previously recorded as final failures.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of episodes to run concurrently for OpenAI-compatible model servers. Use 1 for local HF models.",
    )
    parser.add_argument("--summary-every", type=int, default=1, help="Refresh summary.json every N newly completed episodes.")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "external_rlvr_negotiation_paper_eval"))
    return parser.parse_args()


def load_or_generate_scenarios(args: argparse.Namespace) -> List[RLVRScenario]:
    if args.scenarios_jsonl:
        scenarios = []
        for line in Path(args.scenarios_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or row.get("id") or row.get("asin") or len(scenarios))
            category = str(row.get("category") or row.get("domain") or row.get("product_category") or "provided")
            title = str(row.get("title") or row.get("name") or row.get("item") or "AmazonHistoryPrice item")
            codename = str(row.get("codename") or row.get("code_name") or make_codename(category, item_id))
            description = str(row.get("description") or row.get("desc") or "")
            features_value = row.get("features") or row.get("feature") or ""
            if isinstance(features_value, list):
                features = "; ".join(str(item) for item in features_value)
            else:
                features = str(features_value)
            buyer_budget = float(row.get("buyer_budget", row.get("budget", row.get("B"))))
            seller_cost = float(row.get("seller_cost", row.get("cost", row.get("C"))))
            reference_price = float(
                row.get(
                    "reference_price",
                    row.get(
                        "list_price",
                        row.get("listing_price", row.get("market_price", row.get("price", buyer_budget))),
                    ),
                )
            )
            scenarios.append(
                RLVRScenario(
                    item_id=item_id,
                    title=title,
                    buyer_budget=buyer_budget,
                    seller_cost=seller_cost,
                    reference_price=reference_price,
                    category=category,
                    codename=codename,
                    description=description,
                    features=features,
                    quantity=int(row.get("quantity", 1)),
                )
            )
        return scenarios[: args.num_test_instances]
    if args.require_scenarios_jsonl:
        raise ValueError(
            "--require-scenarios-jsonl was set, but --scenarios-jsonl is missing. "
            "The paper uses a fixed 128-product AmazonHistoryPrice test split; provide it as JSONL."
        )
    return generate_synthetic_amazonhistory_scenarios(
        n=args.num_test_instances,
        seed=args.seed,
        positive_gains_probability=args.positive_gains_probability,
    )


def generate_synthetic_amazonhistory_scenarios(n: int, seed: int, positive_gains_probability: float) -> List[RLVRScenario]:
    rng = random.Random(seed)
    categories = [
        ("wireless headphones", 80),
        ("kitchen blender", 120),
        ("office chair", 220),
        ("winter jacket", 180),
        ("bookshelf", 150),
        ("camping tent", 260),
        ("smart watch", 240),
        ("coffee maker", 95),
    ]
    scenarios: List[RLVRScenario] = []
    for idx in range(n):
        name, base = categories[idx % len(categories)]
        reference = base * rng.uniform(0.72, 1.35)
        if rng.random() < positive_gains_probability:
            seller_cost = reference * rng.uniform(0.45, 0.82)
            buyer_budget = seller_cost + reference * rng.uniform(0.08, 0.45)
        else:
            buyer_budget = reference * rng.uniform(0.45, 0.78)
            seller_cost = buyer_budget + reference * rng.uniform(0.04, 0.24)
        scenarios.append(
            RLVRScenario(
                item_id=f"synthetic_ahp_{idx:04d}",
                title=f"{name.title()} #{idx + 1}",
                buyer_budget=round(buyer_budget, 2),
                seller_cost=round(seller_cost, 2),
                reference_price=round(reference, 2),
                category=name,
                codename=make_codename(name, str(idx)),
                description=f"Synthetic AmazonHistoryPrice-style listing for {name}.",
                features="Generated for pipeline testing; replace with the paper's held-out AmazonHistoryPrice split for table comparison.",
                quantity=1,
            )
        )
    return scenarios


def run_episode(
    *,
    variant: str,
    scenario: RLVRScenario,
    rollout: int,
    buyer: RLVRBuyer,
    seller: RegulatedLLMSeller,
    max_turns: int,
) -> RLVREvalEpisode:
    start = time.time()
    history: List[Dict[str, Any]] = []
    framework_trace: List[Dict[str, Any]] = []
    status = "no_deal"
    terminal_reason = "max_turns"
    final_price: Optional[float] = None
    buyer_overshoot = False
    buyer_format_violation = False
    rounds_used = max_turns
    first_buyer_offer: Optional[float] = None

    for round_id in range(1, max_turns + 1):
        buyer_action, trace = buyer.act(scenario=scenario, history=history, round_id=round_id, max_turns=max_turns)
        framework_trace.append(trace)
        history.append({"round": round_id, "role": "buyer", "message": buyer_action.raw, "action": buyer_action.to_dict()})
        if buyer_action.action == "offer" and first_buyer_offer is None:
            first_buyer_offer = buyer_action.price
        buyer_format_violation = buyer_format_violation or not buyer_action.valid
        buyer_overshoot = buyer_overshoot or (
            buyer_action.price is not None
            and buyer_action.action in {"offer", "accept"}
            and buyer_action.price > scenario.buyer_budget + 1e-9
        )
        if buyer_format_violation:
            status = "invalid"
            terminal_reason = "buyer_format_violation"
            rounds_used = round_id
            break
        if buyer_overshoot:
            status = "invalid"
            terminal_reason = "buyer_budget_overshoot"
            rounds_used = round_id
            break
        if buyer_action.action == "accept":
            seller_offer = last_price(history, "seller")
            if (
                buyer_action.price is not None
                and seller_offer is not None
                and abs(buyer_action.price - seller_offer) <= 1e-6
                and buyer_action.price >= scenario.seller_cost - 1e-9
            ):
                status = "deal"
                terminal_reason = "buyer_accept"
                final_price = buyer_action.price
            else:
                status = "invalid"
                terminal_reason = "buyer_invalid_deal_without_matching_seller_offer"
                buyer_format_violation = True
            rounds_used = round_id
            break
        if buyer_action.action == "walk_away":
            status = "walk_away"
            terminal_reason = "buyer_quit"
            rounds_used = round_id
            break
        if buyer_action.action == "reject":
            continue

        seller_action, seller_raw = seller.act(scenario=scenario, history=history, round_id=round_id, max_turns=max_turns)
        history.append({"round": round_id, "role": "seller", "message": seller_raw, "action": seller_action.to_dict()})
        if seller_action.action == "accept":
            status = "deal"
            terminal_reason = "seller_accept"
            final_price = seller_action.price
            rounds_used = round_id
            break
        if seller_action.action == "walk_away":
            status = "walk_away"
            terminal_reason = "seller_quit"
            rounds_used = round_id
            break
        if seller_action.action == "reject":
            continue

    reward = rlvr_reward(scenario, status, final_price, buyer_overshoot, buyer_format_violation)
    bargained_ratio = None
    if status == "deal" and final_price is not None and scenario.mi:
        bargained_ratio = (scenario.buyer_budget - final_price) / abs(scenario.buyer_budget - scenario.seller_cost)
    return RLVREvalEpisode(
        variant=variant,
        scenario=asdict(scenario),
        rollout=rollout,
        status=status,
        terminal_reason=terminal_reason,
        final_price=round(final_price, 4) if final_price is not None else None,
        reward=round(reward, 6),
        buyer_bargained_ratio=round(bargained_ratio, 6) if bargained_ratio is not None else None,
        first_turn_offer_ratio=round(first_buyer_offer / scenario.buyer_budget, 6) if first_buyer_offer else None,
        buyer_overshoot=buyer_overshoot,
        buyer_format_violation=buyer_format_violation,
        mi=scenario.mi,
        ci=not scenario.mi,
        deal_in_mi=(status == "deal") if scenario.mi else None,
        rational_walkaway_ci=(status != "deal") if not scenario.mi else None,
        rounds=rounds_used,
        transcript=history,
        framework_trace=framework_trace,
        elapsed_seconds=round(time.time() - start, 3),
    )


def parse_action(role: str, text: str) -> ParsedAction:
    source = text or ""
    action_lines = [line for line in source.splitlines() if line.strip().lower().startswith("action:")]
    parse_source = action_lines[-1] if action_lines else source
    matches = ACTION_RE.findall(parse_source)
    if not matches and parse_source is not source:
        matches = ACTION_RE.findall(source)
    if not matches:
        return ParsedAction(role, "invalid", None, source, False)
    tag, price_raw, item_spec = matches[-1]
    tag = tag.upper()
    price = None
    if price_raw:
        try:
            price = float(price_raw.replace(",", ""))
        except ValueError:
            price = None
    if tag == "BUY":
        action = "offer"
    elif tag == "SELL":
        action = "offer"
    elif tag == "DEAL":
        action = "accept"
    elif tag == "REJECT":
        action = "reject"
    elif tag == "QUIT":
        action = "walk_away"
    else:
        action = "invalid"
    valid = action != "invalid"
    if action in {"offer", "accept"} and price is None:
        valid = False
    if role == "buyer" and tag == "SELL":
        valid = False
    if role == "seller" and tag == "BUY":
        valid = False
    return ParsedAction(role, action, price, source, valid, item_spec or None)


def validate_buyer_action(
    action: ParsedAction,
    scenario: RLVRScenario,
    history: Sequence[Dict[str, Any]],
    round_id: int,
    max_turns: int,
) -> tuple[ParsedAction, Dict[str, Any]]:
    """Repair buyer actions that would otherwise produce avoidable invalid episodes."""

    validator = {
        "repaired": False,
        "raw_action": action.to_dict(),
        "notes": [],
    }
    if not action.valid:
        return action, validator

    last_formal_seller_offer = last_price(history, "seller")

    if action.action == "accept":
        valid_deal = (
            action.price is not None
            and last_formal_seller_offer is not None
            and abs(action.price - last_formal_seller_offer) <= 1e-6
            and action.price <= scenario.buyer_budget + 1e-9
        )
        if valid_deal:
            return action, validator
        repaired_price = buyer_repair_offer_price(scenario, history, round_id, max_turns)
        repaired_raw = format_action_message(
            "buyer",
            "BUY",
            repaired_price,
            scenario,
            "I cannot accept that as a formal budget-safe deal, but I can make this counteroffer.",
        )
        repaired = parse_action("buyer", repaired_raw)
        validator["repaired"] = True
        validator["notes"].append(
            "converted invalid DEAL to budget-safe BUY because seller had no matching formal [SELL] offer or price exceeded budget"
        )
        return repaired, validator

    if action.action == "offer" and action.price is not None and action.price > scenario.buyer_budget + 1e-9:
        repaired_price = buyer_repair_offer_price(scenario, history, round_id, max_turns)
        repaired_raw = format_action_message(
            "buyer",
            "BUY",
            repaired_price,
            scenario,
            f"I cannot exceed my budget, but I can offer ${repaired_price:.2f}.",
        )
        repaired = parse_action("buyer", repaired_raw)
        validator["repaired"] = True
        validator["notes"].append(
            f"clipped over-budget BUY from {action.price:g} to {repaired_price:g}"
        )
        return repaired, validator

    return action, validator


def buyer_repair_offer_price(
    scenario: RLVRScenario,
    history: Sequence[Dict[str, Any]],
    round_id: int,
    max_turns: int,
) -> float:
    last_buyer = last_price(history, "buyer")
    last_seller = last_price(history, "seller")
    budget = scenario.buyer_budget
    progress = round_id / max(1, max_turns)
    if last_buyer is None:
        anchor = min(budget * 0.88, scenario.reference_price * 0.72)
    else:
        concession = max(0.01, (budget - min(last_buyer, budget)) * (0.20 + 0.25 * progress))
        anchor = min(budget, last_buyer + concession)
    if last_seller is not None and last_seller <= budget:
        anchor = min(anchor, last_seller)
    if round_id >= max_turns - 1:
        anchor = max(anchor, budget * 0.98)
    return round(min(budget, max(0.01, anchor)), 2)


def make_codename(category: str, item_id: str) -> str:
    safe_category = re.sub(r"[^a-zA-Z0-9]+", "_", str(category).strip().lower()).strip("_") or "product"
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", str(item_id).strip().lower()).strip("_")[-12:] or "0"
    return f"{safe_category}_{suffix}"


def seller_persona_block(persona: str) -> str:
    if persona == "begging":
        return (
            "BEG - Act extremely SAD and DESPERATE. Beg for a better deal at every turn, "
            "using very melancholic and pleading language."
        )
    if persona == "insulting":
        return (
            "Insult - Act extremely MEAN and AGGRESSIVE. Use insults, threats, and harsh language "
            "to intimidate the other player. Your language is RUDE, DISRESPECTFUL, and IMPOLITE. "
            "Use curt, brusque language, and show minimal courtesy."
        )
    if persona == "unyielding":
        return (
            "Unyielding - Act OVERLY STUBBORN and UNYIELDING. Refuse almost every offer the other "
            "player makes, praising your own position. Use firm, resolute language to show you will "
            "not budge until you get the maximum for yourself."
        )
    return ""


def format_action_message(role: str, tag: str, price: float, scenario: RLVRScenario, talk: str) -> str:
    thought = (
        f"I'm a {role}, so I need an action that follows the required protocol for "
        f"{scenario.codename_or_default()}."
    )
    return (
        f"Thought: {thought}\n"
        f"Talk: {talk}\n"
        f"Action: [{tag}] ${price:.2f} ({scenario.quantity}x {scenario.codename_or_default()})"
    )


def concession_schedule(
    scenario: RLVRScenario,
    history: Sequence[Dict[str, Any]],
    round_id: int,
    max_turns: int,
) -> Dict[str, Any]:
    """Round-specific buyer offer caps for RLVR price negotiation.

    The schedule is deliberately conservative early: it keeps the buyer from
    immediately offering near budget, but still relaxes enough late in the
    dialogue to avoid unnecessary no-deal when there is positive surplus.
    """

    budget = float(scenario.buyer_budget)
    reference = max(1.0, float(scenario.reference_price))
    list_discount_anchor = min(budget, reference * 0.58)
    previous_buyer = last_price(history, "buyer")
    previous_seller = last_price(history, "seller")
    ratios = [0.72, 0.78, 0.84, 0.90, 0.96, 1.00]
    idx = min(max(round_id - 1, 0), len(ratios) - 1)
    ratio_cap = budget * ratios[idx]
    max_offer = min(budget, max(ratio_cap, list_discount_anchor))

    if previous_buyer is not None:
        min_increment = max(0.01, budget * (0.015 if round_id <= 2 else 0.025))
        target = min(max_offer, previous_buyer + min_increment)
    else:
        target = min(max_offer, max(budget * 0.62, reference * 0.50))

    if previous_seller is not None:
        pressure_offer = previous_seller * (0.72 + 0.04 * idx)
        target = min(max_offer, max(target, pressure_offer))

    min_offer = min(target, max(0.01, budget * max(0.50, ratios[idx] - 0.12)))
    if round_id >= max_turns - 1:
        accept_cap = budget
    else:
        accept_cap = min(budget, max_offer)
    return {
        "round": round_id,
        "max_turns": max_turns,
        "budget": round(budget, 2),
        "target_offer": round(min(target, budget), 2),
        "min_offer": round(min_offer, 2),
        "max_offer": round(max_offer, 2),
        "accept_cap": round(accept_cap, 2),
        "ratio_cap": round(ratios[idx], 3),
        "policy": "start low; concede gradually; only approach budget in final rounds",
    }


def rlvr_reward(
    scenario: RLVRScenario,
    status: str,
    final_price: Optional[float],
    buyer_overshoot: bool,
    buyer_format_violation: bool,
) -> float:
    if buyer_overshoot or buyer_format_violation:
        return -1.0
    if status != "deal" or final_price is None:
        return 0.0
    if final_price > scenario.buyer_budget + 1e-9:
        return -1.0
    denom = abs(scenario.buyer_budget - scenario.seller_cost)
    if denom <= 1e-9:
        return 0.0
    return min(1.0, max(-1.0, (scenario.buyer_budget - final_price) / denom))


def seller_concession_price(scenario: RLVRScenario, round_id: int, max_turns: int) -> float:
    progress = min(1.0, max(0.0, (round_id - 1) / max(1, max_turns - 1)))
    spread = max(scenario.reference_price, scenario.seller_cost) - scenario.seller_cost
    return scenario.seller_cost + spread * (0.82 - 0.36 * progress)


def last_price(history: Sequence[Dict[str, Any]], role: str) -> Optional[float]:
    for turn in reversed(history):
        if turn.get("role") == role:
            action = turn.get("action") or {}
            price = action.get("price")
            if isinstance(price, (int, float)):
                return float(price)
    return None


def format_transcript(history: Sequence[Dict[str, Any]]) -> str:
    if not history:
        return "(empty)"
    lines = []
    for turn in history:
        lines.append(f"Round {turn.get('round')} {turn.get('role')}: {turn.get('message')}")
    return "\n".join(lines)


def history_to_agenticpay(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for turn in history:
        role = "assistant" if turn.get("role") == "buyer" else "user"
        out.append({"role": role, "content": f"{turn.get('role')}: {turn.get('message')}"})
    return out


def summarize(episodes: Sequence[RLVREvalEpisode]) -> Dict[str, Any]:
    mi = [ep for ep in episodes if ep.mi]
    ci = [ep for ep in episodes if ep.ci]
    deals = [ep for ep in episodes if ep.status == "deal"]
    return {
        "n": len(episodes),
        "num_test_instances": len({ep.scenario["item_id"] for ep in episodes}),
        "avg_reward": _round(_avg(ep.reward for ep in episodes)),
        "deal_rate": _rate(episodes, lambda ep: ep.status == "deal"),
        "mi_deal_ratio": _rate(mi, lambda ep: ep.status == "deal"),
        "ci_rational_walkaway_rate": _rate(ci, lambda ep: ep.status != "deal"),
        "buyer_bargained_ratio": _round(_maybe_avg(ep.buyer_bargained_ratio for ep in deals)),
        "first_turn_offer_ratio": _round(_maybe_avg(ep.first_turn_offer_ratio for ep in episodes)),
        "price_overshoot_rate": _rate(episodes, lambda ep: ep.buyer_overshoot),
        "format_violation_rate": _rate(episodes, lambda ep: ep.buyer_format_violation),
        "avg_rounds": _round(_avg(ep.rounds for ep in episodes)),
    }


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "script": "experiments/external_comparisons/run_rlvr_negotiation_paper_eval.py",
        "paper": "Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards",
        "paper_alignment": {
            "test_instances": args.num_test_instances,
            "rollouts_per_instance": args.rollouts_per_instance,
            "max_turns": args.max_turns,
            "buyer_temperature": args.buyer_temperature,
            "seller_temperature": args.seller_temperature,
            "max_tokens": args.max_tokens,
            "note": "Exact paper AmazonHistoryPrice held-out split and trained Qwen checkpoint are not public; use --scenarios-jsonl when available.",
        },
        "args": vars(args),
    }


def write_static_outputs(args: argparse.Namespace, scenarios: Sequence[RLVRScenario]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "scenarios.jsonl").open("w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(asdict(scenario), ensure_ascii=False) + "\n")


def write_summary(args: argparse.Namespace, episodes_by_variant: Dict[str, List[RLVREvalEpisode]]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": build_config(args),
        "by_variant": {variant: summarize(episodes) for variant, episodes in episodes_by_variant.items()},
        "paper_table1_reference": {
            "trained_qwen3_30b_a3b": {
                "reward": 0.7664,
                "deal_rate": 0.9199,
                "buyer_bargained_ratio": 0.8385,
                "price_overshoot_rate": 0.0010,
            }
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_outputs(args: argparse.Namespace, scenarios: Sequence[RLVRScenario], episodes_by_variant: Dict[str, List[RLVREvalEpisode]]) -> None:
    out = Path(args.output_dir)
    write_static_outputs(args, scenarios)
    write_summary(args, episodes_by_variant)
    for variant, episodes in episodes_by_variant.items():
        with (out / f"{variant}_episodes.jsonl").open("w", encoding="utf-8") as f:
            for episode in episodes:
                f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(out), "summary": str(out / "summary.json")}, indent=2))


def append_episode(args: argparse.Namespace, episode: RLVREvalEpisode) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{episode.variant}_episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
        f.flush()


def record_failed_episode(
    args: argparse.Namespace,
    *,
    variant: str,
    scenario: RLVRScenario,
    rollout: int,
    exc: BaseException,
    attempt: Optional[int] = None,
    final: bool = True,
) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant,
        "scenario": asdict(scenario),
        "rollout": rollout,
        "attempt": attempt,
        "final": final,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }
    with (out / "failed_episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_episodes(args: argparse.Namespace, variant: str) -> List[RLVREvalEpisode]:
    path = Path(args.output_dir) / f"{variant}_episodes.jsonl"
    if not path.exists():
        return []
    episodes: List[RLVREvalEpisode] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episodes.append(RLVREvalEpisode(**row))
        except Exception as exc:
            raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc
    return episodes


def load_final_failed_episode_keys(args: argparse.Namespace, variants: Sequence[str]) -> Dict[str, set[tuple[str, int]]]:
    path = Path(args.output_dir) / "failed_episodes.jsonl"
    failed: Dict[str, set[tuple[str, int]]] = {variant: set() for variant in variants}
    if not path.exists():
        return failed
    variant_set = set(variants)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc
        if not row.get("final"):
            continue
        variant = row.get("variant")
        if variant not in variant_set:
            continue
        scenario = row.get("scenario") or {}
        item_id = scenario.get("item_id")
        rollout = row.get("rollout")
        if item_id is None or rollout is None:
            continue
        failed[variant].add((str(item_id), int(rollout)))
    return failed


def run_episode_with_retries(
    args: argparse.Namespace,
    *,
    variant: str,
    scenario: RLVRScenario,
    rollout: int,
    buyer: RLVRBuyer,
    seller: RegulatedLLMSeller,
) -> RLVREvalEpisode:
    max_attempts = max(1, int(args.episode_retries) + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            return run_episode(
                variant=variant,
                scenario=scenario,
                rollout=rollout,
                buyer=buyer,
                seller=seller,
                max_turns=args.max_turns,
            )
        except Exception as exc:
            final = attempt >= max_attempts
            record_failed_episode(
                args,
                variant=variant,
                scenario=scenario,
                rollout=rollout,
                exc=exc,
                attempt=attempt,
                final=final,
            )
            if final:
                raise
            backoff = max(0.0, float(args.episode_retry_backoff_seconds)) * (2 ** (attempt - 1))
            print(
                json.dumps(
                    {
                        "retry_episode": {
                            "variant": variant,
                            "scenario": scenario.item_id,
                            "rollout": rollout,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "sleep_seconds": backoff,
                            "error": type(exc).__name__,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if backoff > 0:
                time.sleep(backoff)
    raise RuntimeError("unreachable retry state")


def _avg(values: Iterable[float]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _maybe_avg(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _rate(items: Sequence[Any], pred) -> Optional[float]:
    if not items:
        return None
    return _round(sum(1 for item in items if pred(item)) / len(items))


def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 6) if value is not None else None


def main() -> None:
    args = parse_args()
    if args.openai_request_timeout is not None:
        os.environ["OPENAI_REQUEST_TIMEOUT"] = str(args.openai_request_timeout)
    scenarios = load_or_generate_scenarios(args)
    variants = [item.strip() for item in args.buyer_variants.split(",") if item.strip()]

    if args.dry_run:
        write_outputs(args, scenarios, {variant: [] for variant in variants})
        return

    write_static_outputs(args, scenarios)

    buyer_client = make_model_client(
        args.buyer_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        enable_thinking=args.enable_thinking,
    )
    if args.seller_model == args.buyer_model:
        seller_client = buyer_client
    else:
        seller_client = make_model_client(
            args.seller_model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            enable_thinking=args.enable_thinking,
        )
    episodes_by_variant: Dict[str, List[RLVREvalEpisode]] = {
        variant: load_existing_episodes(args, variant) if args.resume else [] for variant in variants
    }
    failed_by_variant = (
        load_final_failed_episode_keys(args, variants)
        if args.resume and not args.retry_failed
        else {variant: set() for variant in variants}
    )
    completed_by_variant = {
        variant: {(ep.scenario.get("item_id"), ep.rollout) for ep in episodes} | failed_by_variant[variant]
        for variant, episodes in episodes_by_variant.items()
    }
    for variant, episodes in episodes_by_variant.items():
        if episodes:
            print(json.dumps({"variant": variant, "resumed_episodes": len(episodes)}))
        if failed_by_variant[variant]:
            print(json.dumps({"variant": variant, "resumed_final_failed_episodes": len(failed_by_variant[variant])}))
    write_summary(args, episodes_by_variant)

    buyers = {
        variant: RLVRBuyer(variant, buyer_client, args.max_tokens, args.buyer_temperature, args.top_p)
        for variant in variants
    }
    sellers = {
        variant: RegulatedLLMSeller(seller_client, args.max_tokens, args.seller_temperature, args.top_p, args.seller_persona)
        for variant in variants
    }
    episode_errors_by_variant = {variant: 0 for variant in variants}

    if args.variant_schedule == "round_robin":
        pending_jobs = [
            (variant, scenario, rollout)
            for scenario in scenarios
            for rollout in range(args.rollouts_per_instance)
            for variant in variants
            if (scenario.item_id, rollout) not in completed_by_variant[variant]
        ]

        def _run_job(job: tuple[str, RLVRScenario, int]) -> RLVREvalEpisode:
            variant, scenario, rollout = job
            # RLVRBuyer/RegulatedLLMSeller contain trace buffers, so keep wrappers
            # episode-local when jobs are concurrent.
            buyer = RLVRBuyer(variant, buyer_client, args.max_tokens, args.buyer_temperature, args.top_p)
            seller = RegulatedLLMSeller(
                seller_client,
                args.max_tokens,
                args.seller_temperature,
                args.top_p,
                args.seller_persona,
            )
            return run_episode_with_retries(
                args,
                variant=variant,
                scenario=scenario,
                rollout=rollout,
                buyer=buyer,
                seller=seller,
            )

        def _handle_failure(job: tuple[str, RLVRScenario, int], exc: BaseException) -> None:
            variant, scenario, rollout = job
            episode_errors_by_variant[variant] += 1
            print(
                json.dumps(
                    {
                        "failed_episode": {
                            "variant": variant,
                            "scenario": scenario.item_id,
                            "rollout": rollout,
                            "error": type(exc).__name__,
                            "errors_for_variant": episode_errors_by_variant[variant],
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        def _handle_episode(episode: RLVREvalEpisode) -> None:
            episodes_by_variant[episode.variant].append(episode)
            append_episode(args, episode)
            if len(episodes_by_variant[episode.variant]) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "variant": episode.variant,
                            "episodes": len(episodes_by_variant[episode.variant]),
                            "latest_reward": episode.reward,
                        }
                    ),
                    flush=True,
                )
            if args.summary_every > 0 and len(episodes_by_variant[episode.variant]) % args.summary_every == 0:
                write_summary(args, episodes_by_variant)

        concurrency = max(1, int(args.concurrency))
        if concurrency == 1:
            for job in pending_jobs:
                variant, _scenario, _rollout = job
                if episode_errors_by_variant[variant] >= args.max_episode_errors_per_variant:
                    continue
                try:
                    _handle_episode(_run_job(job))
                except Exception as exc:
                    _handle_failure(job, exc)
            write_summary(args, episodes_by_variant)
            return

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: Dict[Any, tuple[str, RLVRScenario, int]] = {}
            job_iter = iter(pending_jobs)

            def submit_until_full() -> None:
                while len(futures) < concurrency:
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        return
                    variant, _scenario, _rollout = job
                    if episode_errors_by_variant[variant] >= args.max_episode_errors_per_variant:
                        continue
                    futures[executor.submit(_run_job, job)] = job

            submit_until_full()
            while futures:
                done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        _handle_episode(future.result())
                    except Exception as exc:
                        _handle_failure(job, exc)
                submit_until_full()
        write_summary(args, episodes_by_variant)
        return

    for variant in variants:
        episodes = episodes_by_variant[variant]
        completed = completed_by_variant[variant]
        for scenario in scenarios:
            for rollout in range(args.rollouts_per_instance):
                if episode_errors_by_variant[variant] >= args.max_episode_errors_per_variant:
                    break
                if (scenario.item_id, rollout) in completed:
                    continue
                try:
                    episode = run_episode_with_retries(
                        args,
                        variant=variant,
                        scenario=scenario,
                        rollout=rollout,
                        buyer=buyers[variant],
                        seller=sellers[variant],
                    )
                except Exception as exc:
                    episode_errors_by_variant[variant] += 1
                    print(
                        json.dumps(
                            {
                                "failed_episode": {
                                    "variant": variant,
                                    "scenario": scenario.item_id,
                                    "rollout": rollout,
                                    "error": type(exc).__name__,
                                    "errors_for_variant": episode_errors_by_variant[variant],
                                }
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                episodes.append(episode)
                append_episode(args, episode)
                if len(episodes) % 10 == 0:
                    print(json.dumps({"variant": variant, "episodes": len(episodes), "latest_reward": episodes[-1].reward}))
                if args.summary_every > 0 and len(episodes) % args.summary_every == 0:
                    write_summary(args, episodes_by_variant)
        write_summary(args, episodes_by_variant)


if __name__ == "__main__":
    main()
