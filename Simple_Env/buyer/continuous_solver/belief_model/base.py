"""Base persistent opponent belief state.

The central design choice is that the belief lives across turns. LLMs may help
extract evidence, but they should not rewrite the full state from scratch each
round. This makes uncertainty, calibration, and planner behavior inspectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from experiments.agenticpay_framework.schemas import BeliefState as AgenticPayBeliefState


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class BeliefEvidence:
    """One piece of evidence supporting a belief update.

    Attributes:
        round_id: Negotiation round where the evidence was observed.
        source: "rule", "llm", or "hybrid".
        evidence_type: Semantic/action type, e.g. seller_counter, finality_cue.
        text: Short human-readable evidence description.
        effect: How the evidence changed the belief.
        confidence: Local confidence in this evidence item, not global belief confidence.
    """

    round_id: int
    source: str
    evidence_type: str
    text: str
    effect: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReservationBelief:
    """Summary of seller reservation price uncertainty.

    mean/std/p10/p50/p90 are derived from a posterior when available. They also
    serve as a fallback for simpler belief variants that do not maintain a full
    distribution.
    """

    mean: Optional[float] = None
    std: Optional[float] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    confidence: float = 0.2

    def interval(self) -> Optional[List[float]]:
        if self.p10 is None or self.p90 is None:
            return None
        return [float(self.p10), float(self.p90)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AcceptancePoint:
    """Estimated seller acceptance probability at a candidate buyer offer."""

    price: float
    p_accept: float
    confidence: float = 0.3
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConcessionState:
    """Behavioral estimate of seller concession dynamics.

    slope: EMA of relative seller price drops. Higher means more flexible.
    last_seller_offer: Last formal seller [SELL] price.
    previous_seller_offer: Previous formal seller [SELL] price.
    num_price_drops: Number of observed seller downward concessions.
    """

    slope: float = 0.0
    last_seller_offer: Optional[float] = None
    previous_seller_offer: Optional[float] = None
    num_price_drops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionState:
    """Non-price opponent state used for risk-aware planning.

    finality_prob: Probability seller is near a real final offer.
    quit_risk: Probability seller may quit after another low offer.
    patience: Seller willingness to keep bargaining.
    friendliness/dominance: Language/stance estimates for generator style.
    language_reliability: Whether seller's verbal finality matches behavior.
    """

    finality_prob: float = 0.15
    quit_risk: float = 0.10
    patience: float = 0.70
    friendliness: float = 0.50
    dominance: float = 0.40
    language_reliability: float = 0.50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeliefState:
    """Persistent belief state for one negotiation episode."""

    item_id: str
    buyer_budget: float
    reference_price: float
    round_id: int = 0
    reservation: ReservationBelief = field(default_factory=ReservationBelief)
    acceptance_curve: List[AcceptancePoint] = field(default_factory=list)
    concession: ConcessionState = field(default_factory=ConcessionState)
    interaction: InteractionState = field(default_factory=InteractionState)
    evidence: List[BeliefEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: BeliefEvidence) -> None:
        self.evidence.append(evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "buyer_budget": self.buyer_budget,
            "reference_price": self.reference_price,
            "round_id": self.round_id,
            "reservation": self.reservation.to_dict(),
            "acceptance_curve": [x.to_dict() for x in self.acceptance_curve],
            "concession": self.concession.to_dict(),
            "interaction": self.interaction.to_dict(),
            "evidence": [x.to_dict() for x in self.evidence],
            "metadata": self.metadata,
        }

    def to_agenticpay_belief(self) -> AgenticPayBeliefState:
        """Convert to the existing framework schema for prompt planner reuse."""

        interval = self.reservation.interval()
        likely_range = None
        if self.acceptance_curve:
            feasible = [p.price for p in self.acceptance_curve if p.p_accept >= 0.45]
            if feasible:
                likely_range = [min(feasible), max(feasible)]
        return AgenticPayBeliefState(
            seller_reservation_range=interval,
            seller_reservation_confidence=self.reservation.confidence,
            seller_reservation_estimate=self.reservation.mean,
            seller_flexibility_range=[
                clamp(self.concession.slope * 2.0),
                clamp(self.concession.slope * 2.0 + 0.25),
            ],
            seller_patience_range=[
                clamp(self.interaction.patience - 0.15),
                clamp(self.interaction.patience + 0.15),
            ],
            seller_flexibility=clamp(self.concession.slope * 2.0),
            seller_patience=clamp(self.interaction.patience),
            seller_strategy="persistent_belief",
            deal_risk=clamp(self.interaction.quit_risk),
            last_seller_offer=self.concession.last_seller_offer,
            likely_acceptable_price_range=likely_range or interval,
            evidence=[f"{e.evidence_type}: {e.text} -> {e.effect}" for e in self.evidence[-8:]],
        )
