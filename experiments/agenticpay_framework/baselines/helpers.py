"""Shared helpers for AgenticPay baseline buyer agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.agenticpay_framework.components import extract_buyer_price, extract_contract
from experiments.agenticpay_framework.schemas import BeliefState


def buyer_max_from_context(buyer_max_price: Optional[float], context: Dict[str, Any]) -> Optional[float]:
    if buyer_max_price is not None:
        return float(buyer_max_price)
    value = context.get("max_price")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_number(current_state: Dict[str, Any]) -> int:
    raw = current_state.get("round") or current_state.get("round_index") or 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def last_seller_price(conversation_history: List[Dict[str, Any]]) -> Optional[float]:
    for msg in reversed(conversation_history):
        role = str(msg.get("role", "")).lower()
        if "seller" not in role:
            continue
        content = str(msg.get("content", ""))
        contract = extract_contract(content)
        if contract is not None:
            try:
                return float(contract.get("price"))
            except (TypeError, ValueError):
                pass
        # Reuse the buyer-price parser after normalizing the tag name.
        price = extract_buyer_price(content.replace("SELLER_PRICE", "BUYER_PRICE"))
        if price is not None:
            return price
    return None


def rule_target_price(
    buyer_max: Optional[float],
    last_offer: Optional[float],
    round_id: int,
) -> Optional[float]:
    if buyer_max is None:
        return None
    start = buyer_max * 0.72
    final = buyer_max * 0.98
    schedule = min(1.0, max(0.0, (round_id - 1) / 8.0))
    scheduled_price = start + (final - start) * schedule
    if last_offer is not None:
        scheduled_price = min(scheduled_price, max(buyer_max * 0.65, last_offer * 0.93))
    return round(min(scheduled_price, buyer_max), 4)


def belief_floor(belief: BeliefState) -> Optional[float]:
    if belief.likely_acceptable_price_range:
        return belief.likely_acceptable_price_range[0]
    if belief.seller_reservation_range and belief.seller_reservation_confidence >= 0.45:
        return belief.seller_reservation_range[0]
    if belief.last_seller_offer is not None and belief.seller_flexibility <= 0.35:
        return belief.last_seller_offer * 0.97
    return None


def seller_guardrails(floor: Optional[float], buyer_max: Optional[float]) -> List[str]:
    guardrails = []
    if floor is not None:
        guardrails.append(f"avoid offers below likely seller-feasible floor {floor:g}")
    if floor is not None and buyer_max is not None and floor > buyer_max:
        guardrails.append("seller floor appears above buyer max; ask for more information or walk away")
    return guardrails
