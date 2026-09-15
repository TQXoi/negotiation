"""Shared schemas for AgenticPay RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


ACTION_NAMES = (
    "anchor_low",
    "concede_small",
    "concede_medium",
    "hold",
    "ask_info",
    "accept",
    "walk_away",
)


ActionName = Literal[
    "anchor_low",
    "concede_small",
    "concede_medium",
    "hold",
    "ask_info",
    "accept",
    "walk_away",
]


@dataclass
class TypedAction:
    """A compact strategic action chosen by the planner or RL policy."""

    act: ActionName
    target_price: Optional[float] = None
    warmth: float = 0.5
    dominance: float = 0.5
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeliefState:
    """Opponent-belief features available to the strategic policy."""

    seller_reservation_mean: Optional[float] = None
    seller_reservation_std: Optional[float] = None
    accept_prob_at_target: Optional[float] = None
    patience: Optional[float] = None
    warmth: Optional[float] = None
    dominance: Optional[float] = None
    uncertainty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundRecord:
    """One buyer-seller exchange inside an episode."""

    round_index: int
    observation: Dict[str, Any]
    typed_action: Optional[TypedAction]
    belief_state: Optional[BeliefState]
    buyer_message: str
    seller_message: str
    buyer_price: Optional[float] = None
    seller_price: Optional[float] = None
    agreed_price: Optional[float] = None
    step_buyer_reward: Optional[float] = None
    step_seller_reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.typed_action is not None:
            data["typed_action"] = self.typed_action.to_dict()
        if self.belief_state is not None:
            data["belief_state"] = self.belief_state.to_dict()
        return data


@dataclass
class EpisodeRecord:
    """A full AgenticPay negotiation trajectory."""

    episode_id: str
    scenario_id: str
    seller_character: str
    buyer_variant: str
    seed: int
    model_id: Optional[str] = None
    rounds: List[RoundRecord] = field(default_factory=list)
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    hidden_state_for_training_only: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "seller_character": self.seller_character,
            "buyer_variant": self.buyer_variant,
            "seed": self.seed,
            "model_id": self.model_id,
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "final_metrics": self.final_metrics,
            "hidden_state_for_training_only": self.hidden_state_for_training_only,
        }
