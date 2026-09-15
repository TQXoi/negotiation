from __future__ import annotations

from typing import Any, Dict, List

from astra_integration.framework.planner_v2 import CoalitionAwarePlannerV2


class RobustCoalitionPlannerV3(CoalitionAwarePlannerV2):
    """V3: rerank V2's final shortlist by coalition margin and veto safety."""

    version = "robust_coalition_v3"

    def __init__(
        self, candidate_top_k: int = 8, *, veto_gate: float = 0.65,
        robust_support_gate: float = 0.65, history_bonus: float = 0.30,
        self_bonus: float = 0.05, shortfall_penalty: float = 0.50,
    ):
        super().__init__(candidate_top_k=candidate_top_k)
        self.veto_gate = veto_gate
        self.robust_support_gate = robust_support_gate
        self.history_bonus = history_bonus
        self.self_bonus = self_bonus
        self.shortfall_penalty = shortfall_penalty

    def generate(
        self, game: Any, focal: str, beliefs: Dict[str, Any], history: List[Dict[str, Any]],
        turn_id: int, max_turns: int, final_vote: bool,
    ) -> List[Dict[str, Any]]:
        candidates = super().generate(game, focal, beliefs, history, turn_id, max_turns, final_vote)
        for row in candidates:
            probabilities = list(row["acceptance_by_opponent"].values())
            robust_count = sum(value >= self.robust_support_gate for value in probabilities)
            shortfall = sum(max(0.0, self.robust_support_gate - value) for value in probabilities)
            veto_safe = row["p2_accept_prob"] >= self.veto_gate
            robust_score = (
                row["coalition_prob"]
                + self.history_bonus * row["history_consensus"]
                + self.self_bonus * row["self_norm"]
                - self.shortfall_penalty * shortfall
            )
            row.update({
                "veto_safe": veto_safe,
                "robust_support_gate": self.robust_support_gate,
                "robust_predicted_supporters": robust_count,
                "acceptance_shortfall": round(shortfall, 5),
                "robust_final_score": round(robust_score, 5),
                "planner_version": self.version,
            })
        if final_vote:
            candidates.sort(key=lambda row: (
                row["veto_safe"], row["robust_predicted_supporters"],
                row["robust_final_score"], row["p2_accept_prob"], row["self_score"],
            ), reverse=True)
        return candidates

    @classmethod
    def choose(cls, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {"action_type": "abstain", "deal": None, "planner_version": cls.version}
        return {**candidates[0], "action_type": "propose"}
