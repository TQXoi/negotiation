from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class OpponentBelief:
    option_logits: Dict[str, List[float]]
    threshold_ratio: float = 0.50
    flexibility: float = 0.50
    adversarial_probability: float = 0.10
    evidence: List[str] = field(default_factory=list)

    def option_posteriors(self) -> Dict[str, List[float]]:
        result = {}
        for issue, logits in self.option_logits.items():
            anchor = max(logits)
            exps = [math.exp(value - anchor) for value in logits]
            total = sum(exps) or 1.0
            result[issue] = [round(value / total, 5) for value in exps]
        return result

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["option_posteriors"] = self.option_posteriors()
        return data


class MultiPartyBeliefModel:
    """Transparent continuous updater; it never reads hidden opponent scores."""

    def update(self, game: Any, focal_player: str, history: List[Dict[str, Any]]) -> Dict[str, OpponentBelief]:
        beliefs = {
            player: OpponentBelief({issue: [0.0] * count for issue, count in game.issues.items()})
            for player in game.players
            if player != focal_player
        }
        previous_deal: Dict[str, Dict[str, int]] = {}
        for row in history:
            actor = row["actor"]
            if actor == focal_player or actor not in beliefs:
                continue
            action = row["action"]
            belief = beliefs[actor]
            deal = action.get("deal")
            action_type = action.get("type")
            if deal:
                strength = 0.42 if action_type in {"propose", "support"} else -0.22
                for issue, option in deal.items():
                    belief.option_logits[issue][int(option) - 1] += strength
                belief.evidence.append(f"turn={row['turn']}:{action_type}:{deal}")
                if actor in previous_deal and previous_deal[actor] != deal:
                    belief.flexibility = min(1.0, belief.flexibility + 0.05)
                previous_deal[actor] = deal
            if action_type == "oppose":
                belief.threshold_ratio = min(0.9, belief.threshold_ratio + 0.025)
                belief.flexibility = max(0.0, belief.flexibility - 0.025)
            elif action_type == "support":
                belief.threshold_ratio = max(0.2, belief.threshold_ratio - 0.015)
        return beliefs
