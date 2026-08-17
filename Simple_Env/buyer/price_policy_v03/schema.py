"""Structured action schema for v0.3 small price-policy buyers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PricePolicyAction:
    """Low-dimensional buyer action produced by a trainable small model.

    The small policy should only decide strategy and price. Natural language is
    intentionally left to a larger verifier/generator.
    """

    action_type: str = "offer"
    target_price: Optional[float] = None
    accept_formal_seller_offer: bool = False
    concession_style: str = "anchor_low"
    confidence: float = 0.5
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedPolicyAction:
    """30B verifier output after checking policy price/action sanity."""

    approved: bool
    action: PricePolicyAction
    revision_reason: str = ""
    language_plan: str = "firm, concise, buyer-favorable"
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "action": self.action.to_dict(),
            "revision_reason": self.revision_reason,
            "language_plan": self.language_plan,
            "raw": self.raw,
        }
