"""Structured interfaces between modular AgenticPay buyer components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BeliefState:
    """Buyer's private estimate of the seller and negotiation state."""

    seller_reservation_range: Optional[List[float]] = None
    seller_reservation_confidence: float = 0.3
    seller_reservation_estimate: Optional[float] = None
    seller_flexibility_range: Optional[List[float]] = None
    seller_patience_range: Optional[List[float]] = None
    seller_dominance_range: Optional[List[float]] = None
    seller_friendliness_range: Optional[List[float]] = None
    seller_flexibility: float = 0.5
    seller_patience: float = 0.5
    seller_strategy: str = "unknown"
    deal_risk: float = 0.5
    last_seller_offer: Optional[float] = None
    likely_acceptable_price_range: Optional[List[float]] = None
    contract_term_preferences: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategicPlan:
    """High-level bargaining intent before language realization."""

    strategic_act: str = "concede_small"
    target_price: Optional[float] = None
    reservation_guardrail: Optional[float] = None
    concession_size: str = "small"
    accept_if_at_or_below: Optional[float] = None
    contract_priorities: List[str] = field(default_factory=lambda: ["price"])
    required_non_price_terms: Dict[str, Any] = field(default_factory=dict)
    avoid_non_price_terms: Dict[str, Any] = field(default_factory=dict)
    feasibility_checks: List[str] = field(default_factory=list)
    seller_feasible_price_floor: Optional[float] = None
    seller_feasibility_risk: float = 0.5
    seller_feasible_non_price_terms: Dict[str, Any] = field(default_factory=dict)
    seller_feasibility_guardrails: List[str] = field(default_factory=list)
    language_style: str = "friendly"
    private_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NaturalizationResult:
    """Final low-level buyer message plus optional diagnostics."""

    final_response: str
    raw_response: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameworkTrace:
    """One turn of module-level diagnostics for trajectory analysis."""

    variant: str
    round_id: Optional[Any]
    belief: Optional[BeliefState]
    plan: Optional[StrategicPlan]
    naturalization: NaturalizationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "round": self.round_id,
            "belief": self.belief.to_dict() if self.belief else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "naturalization": self.naturalization.to_dict(),
        }
