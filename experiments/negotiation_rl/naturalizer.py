"""Natural-language realization for typed negotiation actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import TypedAction


@dataclass
class NaturalizerContext:
    """Context needed to verbalize a typed buyer action."""

    product_name: str = "the product"
    buyer_max_price: Optional[float] = None
    last_seller_price: Optional[float] = None
    last_buyer_price: Optional[float] = None
    round_index: int = 0


class TemplateNaturalizer:
    """Format-safe buyer message generator for AgenticPay price tasks."""

    def naturalize(self, action: TypedAction, context: NaturalizerContext) -> str:
        if action.act == "walk_away":
            price = self._safe_price(action.target_price, context.last_buyer_price, context.buyer_max_price)
            return (
                f"I do not think this deal works for me at the current level. "
                f"My final position is ### BUYER_PRICE(${price:.2f}) ###."
            )

        if action.act == "accept":
            price = self._safe_price(action.target_price, context.last_seller_price, context.buyer_max_price)
            return (
                f"That is acceptable for {context.product_name}. "
                f"I can close at ### BUYER_PRICE(${price:.2f}) ###. MAKE_DEAL"
            )

        price = self._safe_price(action.target_price, context.last_buyer_price, context.buyer_max_price)

        if action.act == "anchor_low":
            prefix = "I like the product, but I am comparing alternatives."
        elif action.act == "concede_small":
            prefix = "I can improve my offer a little to keep this moving."
        elif action.act == "concede_medium":
            prefix = "I can make a stronger move if we can settle soon."
        elif action.act == "hold":
            prefix = "I am still not convinced the current price is competitive."
        elif action.act == "ask_info":
            prefix = "Could you clarify the value you are including at this price?"
        else:
            prefix = "Here is my current offer."

        return f"{prefix} I can offer ### BUYER_PRICE(${price:.2f}) ### for {context.product_name}."

    @staticmethod
    def _safe_price(*candidates: Optional[float]) -> float:
        for candidate in candidates:
            if candidate is not None and candidate > 0:
                return float(candidate)
        return 1.0
