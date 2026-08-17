"""Context-conditioned planner parameters for Simple Env continuous CEM."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario, last_price

from ..belief_model.base import BeliefState, clamp
from .params import PlannerParams


FEATURE_NAMES = [
    "budget_to_ref_low",
    "belief_floor_to_budget_high",
    "belief_uncertain",
    "quit_risk_high",
    "seller_offer_to_budget_high",
    "seller_concession_low",
]

ADAPTIVE_PARAM_NAMES = [
    "first_anchor_ratio",
    "early_concession_rate",
    "late_concession_rate",
    "seller_discount_early",
    "seller_discount_late",
    "accept_reward_threshold",
    "accept_seller_ratio_threshold",
    "lowball_penalty_weight",
    "quit_penalty",
    "future_value_weight",
]


@dataclass
class ContextualPlannerPolicy:
    """Base planner params plus small linear contextual offsets."""

    base_params: PlannerParams = field(default_factory=PlannerParams)
    feature_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    additive_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "contextual_planner_policy_v0.2",
            "base_params": self.base_params.to_dict(),
            "feature_names": FEATURE_NAMES,
            "adaptive_param_names": ADAPTIVE_PARAM_NAMES,
            "feature_weights": self.feature_weights,
            "additive_weights": self.additive_weights,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextualPlannerPolicy":
        if "contextual_policy" in data and isinstance(data["contextual_policy"], Mapping):
            data = data["contextual_policy"]
        base_data = data.get("base_params") or data.get("best_params") or data.get("params") or {}
        return cls(
            base_params=PlannerParams.from_dict(dict(base_data)),
            feature_weights={
                str(param): {str(feat): float(value) for feat, value in weights.items()}
                for param, weights in dict(data.get("feature_weights") or {}).items()
            },
            additive_weights={
                str(param): {str(feat): float(value) for feat, value in weights.items()}
                for param, weights in dict(data.get("additive_weights") or {}).items()
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ContextualPlannerPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def contextualize(
        self,
        *,
        scenario: RLVRScenario,
        history: Any,
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> tuple[PlannerParams, Dict[str, float]]:
        features = planner_features(scenario=scenario, history=history, belief=belief, round_id=round_id, max_turns=max_turns)
        values = self.base_params.to_dict()
        for param in ADAPTIVE_PARAM_NAMES:
            base = values[param]
            mult_delta = sum(self.feature_weights.get(param, {}).get(name, 0.0) * value for name, value in features.items())
            add_delta = sum(self.additive_weights.get(param, {}).get(name, 0.0) * value for name, value in features.items())
            values[param] = base * (1.0 + mult_delta) + add_delta
        return PlannerParams.from_dict(values), features


def planner_features(
    *,
    scenario: RLVRScenario,
    history: Any,
    belief: BeliefState,
    round_id: int,
    max_turns: int,
) -> Dict[str, float]:
    budget = max(float(scenario.buyer_budget), 1.0)
    ref = max(float(scenario.reference_price), 1.0)
    floor = float(belief.reservation.p10 or belief.reservation.p50 or belief.reservation.mean or budget * 0.55)
    seller = last_price(history, "seller")
    first_seller = _first_seller_price(history)
    seller_offer_ratio = float(seller) / budget if seller is not None else 0.0
    concession_slope = float(belief.concession.slope or 0.0)
    if first_seller and seller:
        concession_slope = max(concession_slope, (float(first_seller) - float(seller)) / max(float(first_seller), 1.0))
    progress = round_id / max(1, max_turns - 1)
    return {
        "budget_to_ref_low": clamp(1.0 - budget / ref, 0.0, 1.0),
        "belief_floor_to_budget_high": clamp(floor / budget, 0.0, 1.5) / 1.5,
        "belief_uncertain": clamp(1.0 - float(belief.reservation.confidence or 0.0), 0.0, 1.0),
        "quit_risk_high": clamp(float(belief.interaction.quit_risk or 0.0), 0.0, 1.0),
        "seller_offer_to_budget_high": clamp(seller_offer_ratio, 0.0, 1.4) / 1.4,
        "seller_concession_low": clamp(1.0 - concession_slope * 4.0, 0.0, 1.0) * clamp(progress + 0.25, 0.0, 1.0),
    }


def _first_seller_price(history: Any) -> float | None:
    for item in history:
        if item.get("role") == "seller":
            price = last_price([item], "seller")
            if price is not None:
                return float(price)
    return None
