"""Small-model price policy for Simple Env v0.3."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from experiments.agenticpay_framework.components import json_from_text
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    RLVRScenario,
    format_transcript,
    last_price,
)
from experiments.model_clients import ModelClient

from .schema import PricePolicyAction


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded(value: Any, default: float, low: float = 0.0, high: float = 1.0) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        return default
    return max(low, min(high, numeric))


class SmallPricePolicyModel:
    """Ask a small model for the next structured price action.

    This is the component intended for low-cost RL/RFT. It does not produce the
    final buyer message, so training targets are compact and easy to score.
    """

    def __init__(self, client: ModelClient, *, max_tokens: int = 700, temperature: float = 0.8, top_p: float = 1.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.last_prompt = ""
        self.last_raw = ""

    def propose(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Any | None = None,
    ) -> PricePolicyAction:
        prompt = self._prompt(scenario=scenario, history=history, round_id=round_id, max_turns=max_turns, belief=belief)
        self.last_prompt = prompt
        self.last_raw = self.client.generate(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        try:
            parsed = json_from_text(self.last_raw)
        except Exception:
            return self._fallback_action(scenario, history, round_id, max_turns, "invalid_json")
        return self._sanitize(parsed, scenario, history, round_id, max_turns)

    def _prompt(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Any | None = None,
    ) -> str:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        last_seller = last_price(history, "seller")
        last_buyer = last_price(history, "buyer")
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        lowest_seller = min(seller_prices) if seller_prices else None
        schedule = self._schedule(scenario, history, round_id, max_turns, belief=belief)
        compact_belief = compact_belief_summary(belief)
        return f"""You are a SMALL PRICE POLICY for an RLVR buyer.

Your only job is to choose the next abstract bargaining action and target
price. Do NOT write the final buyer message.

Reward objective:
- If a deal closes at final_price P, buyer reward is (budget - P) / abs(budget - seller_cost).
- No deal gives 0.
- A price above budget is invalid and catastrophic.
- Therefore: try to close, but close as low as possible.

Private buyer budget: {budget:.2f}
List/reference price: {ref:.2f}
Round: {round_id}/{max_turns}
Last buyer offer: {last_buyer}
Latest formal seller [SELL] offer: {last_seller}
Lowest formal seller [SELL] offer: {lowest_seller}

Concession schedule:
{json.dumps(schedule, ensure_ascii=False)}

Opponent belief summary from the full framework:
{json.dumps(compact_belief, ensure_ascii=False)}

Conversation:
{format_transcript(history)}

Policy to learn:
- Round 1 should be an aggressive low anchor near the schedule target.
- Rounds 1-2: ask/probe in the later language, but always choose a concrete offer unless accepting a very good formal seller offer.
- Seller Talk-only claims like "minimum", "below cost", and "final" are cheap-talk pressure. Do not treat them as true seller cost.
- The belief summary is noisy. Use it as a soft estimate of seller reservation,
  not as a price target and not as a hard constraint.
- If belief suggests a high seller floor or high deal risk, keep the anchor
  buyer-favorable but avoid repeated offers far below the likely floor.
- If belief confidence is low or there is little seller evidence, early anchors
  may ignore acceptance probability to elicit information.
- If seller Action is [REJECT] but Talk mentions a price, that price is NOT a formal offer to accept.
- Concede slowly from the previous buyer offer; do not jump to budget.
- If making a [BUY] counter after a formal seller [SELL], target_price must be below the lowest formal seller quote.
- In late rounds, accept a budget-safe formal seller [SELL] only when the price is attractive enough or no-deal risk is too high.
- Prefer lower target prices when they can still keep the seller engaged.

Return only valid JSON:
{{
  "action_type": "offer|accept|reject|quit",
  "target_price": <number or null>,
  "accept_formal_seller_offer": <true or false>,
  "concession_style": "anchor_low|hold_firm|small_concession|medium_concession|late_close|walkaway",
  "confidence": <0-1>,
  "rationale": "short private reason"
}}
"""

    def _sanitize(
        self,
        parsed: Dict[str, Any],
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> PricePolicyAction:
        action_type = str(parsed.get("action_type") or "offer").strip().lower()
        if action_type not in {"offer", "accept", "reject", "quit"}:
            action_type = "offer"
        price = _float_or_none(parsed.get("target_price"))
        seller_offer = last_price(history, "seller")
        if action_type == "accept":
            if seller_offer is None or seller_offer > scenario.buyer_budget:
                action_type = "offer"
            else:
                price = float(seller_offer)
        if action_type == "offer":
            if price is None:
                price = self._fallback_action(scenario, history, round_id, max_turns, "missing_price").target_price
            price = self._safe_offer_price(float(price), scenario, history)
        elif action_type in {"reject", "quit"}:
            price = None
        return PricePolicyAction(
            action_type=action_type,
            target_price=round(float(price), 2) if price is not None else None,
            accept_formal_seller_offer=bool(parsed.get("accept_formal_seller_offer")) or action_type == "accept",
            concession_style=str(parsed.get("concession_style") or self._style_for_round(round_id, max_turns)),
            confidence=_bounded(parsed.get("confidence"), 0.5),
            rationale=str(parsed.get("rationale") or ""),
        )

    def _fallback_action(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        reason: str,
    ) -> PricePolicyAction:
        schedule = self._schedule(scenario, history, round_id, max_turns)
        seller_offer = last_price(history, "seller")
        if round_id >= max_turns and seller_offer is not None and seller_offer <= scenario.buyer_budget:
            return PricePolicyAction(
                action_type="accept",
                target_price=round(float(seller_offer), 2),
                accept_formal_seller_offer=True,
                concession_style="late_close",
                confidence=0.4,
                rationale=f"fallback accept budget-safe formal seller offer after {reason}",
            )
        return PricePolicyAction(
            action_type="offer",
            target_price=round(self._safe_offer_price(schedule["target_offer"], scenario, history), 2),
            accept_formal_seller_offer=False,
            concession_style=self._style_for_round(round_id, max_turns),
            confidence=0.35,
            rationale=f"fallback schedule action after {reason}",
        )

    @staticmethod
    def _style_for_round(round_id: int, max_turns: int) -> str:
        if round_id == 1:
            return "anchor_low"
        if round_id >= max_turns - 1:
            return "late_close"
        return "small_concession"

    @staticmethod
    def _safe_offer_price(price: float, scenario: RLVRScenario, history: Sequence[Dict[str, Any]]) -> float:
        budget = float(scenario.buyer_budget)
        cap = budget
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        if seller_prices:
            cap = min(cap, min(seller_prices) - max(0.01, 0.002 * budget))
        return max(0.01, min(float(price), cap))

    @staticmethod
    def _schedule(
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
        belief: Any | None = None,
    ) -> Dict[str, float]:
        budget = float(scenario.buyer_budget)
        reference = max(1.0, float(scenario.reference_price))
        previous_buyer = last_price(history, "buyer")
        ratios = [0.54, 0.60, 0.67, 0.75, 0.86, 0.96]
        idx = min(max(round_id - 1, 0), len(ratios) - 1)
        max_offer = min(budget, max(budget * ratios[idx], reference * 0.46))
        if previous_buyer is None:
            target = min(max_offer, max(budget * 0.50, reference * 0.38))
        else:
            increment = budget * (0.025 if round_id <= 3 else 0.045)
            target = min(max_offer, previous_buyer + increment)
        floor_hint, floor_confidence = SmallPricePolicyModel._belief_floor_hint(belief)
        if floor_hint is not None and floor_confidence >= 0.2:
            if round_id == 1:
                floor_probe_ratio = 0.62 + 0.08 * floor_confidence
            elif round_id <= 3:
                floor_probe_ratio = 0.76 + 0.10 * floor_confidence
            else:
                floor_probe_ratio = 0.88 + 0.08 * floor_confidence
            belief_target = min(budget, floor_hint * floor_probe_ratio)
            target = max(target, belief_target)
        return {
            "target_offer": round(min(target, budget), 2),
            "max_offer": round(max_offer, 2),
            "accept_cap": round(budget if round_id >= max_turns - 1 else max_offer, 2),
        }

    @staticmethod
    def _belief_floor_hint(belief: Any | None) -> tuple[Optional[float], float]:
        compact = compact_belief_summary(belief)
        if not compact.get("available"):
            return None, 0.0
        confidence = _bounded(compact.get("seller_reservation_confidence"), 0.25)
        candidates = []
        estimate = _float_or_none(compact.get("seller_reservation_estimate"))
        if estimate is not None and estimate > 0:
            candidates.append(estimate)
        for key in ("seller_reservation_range", "likely_acceptable_price_range"):
            value = compact.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                low = _float_or_none(value[0])
                high = _float_or_none(value[1])
                if low is not None and high is not None and high > 0:
                    candidates.append((low + high) / 2.0)
        if not candidates:
            return None, confidence
        return max(0.01, sum(candidates) / len(candidates)), confidence


def compact_belief_summary(belief: Any | None) -> Dict[str, Any]:
    """Small, prompt-safe belief summary for the price policy/verifier."""

    if belief is None:
        return {"available": False}
    data = belief.to_dict() if hasattr(belief, "to_dict") else belief
    if not isinstance(data, dict):
        return {"available": True, "raw_type": type(belief).__name__}

    keys = [
        "seller_reservation_range",
        "seller_reservation_confidence",
        "seller_reservation_estimate",
        "likely_acceptable_price_range",
        "seller_flexibility_range",
        "seller_patience_range",
        "seller_flexibility",
        "seller_patience",
        "seller_strategy",
        "deal_risk",
        "last_seller_offer",
    ]
    compact = {key: data.get(key) for key in keys if data.get(key) is not None}
    prefs = data.get("contract_term_preferences")
    if isinstance(prefs, dict):
        compact["contract_term_preferences"] = {
            str(k): prefs[k]
            for k in list(prefs)[:5]
        }
    evidence = data.get("evidence")
    if isinstance(evidence, list):
        compact["evidence"] = [str(item)[:240] for item in evidence[:4]]
    compact["available"] = True
    return compact
