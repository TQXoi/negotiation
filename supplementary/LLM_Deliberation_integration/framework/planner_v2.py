from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from astra_integration.tools.deals import enumerate_deals, normalized_score, score_deal


class CoalitionAwarePlannerV2:
    """V2: veto/coalition feasibility first, own utility second."""

    version = "coalition_aware_v2"

    def __init__(
        self, candidate_top_k: int = 8, acceptance_gate: float = 0.55,
        acceptance_bias: float = 0.05, alignment_weight: float = 0.78,
        flexibility_weight: float = 0.12, threshold_weight: float = 0.10,
        adversarial_weight: float = 0.10,
        final_coalition_weight: float = 0.62, final_history_weight: float = 0.23,
        final_self_weight: float = 0.15, discussion_coalition_weight: float = 0.45,
        discussion_self_weight: float = 0.25, discussion_history_weight: float = 0.20,
        discussion_info_weight: float = 0.10,
    ):
        self.candidate_top_k = candidate_top_k
        self.acceptance_gate = acceptance_gate
        self.acceptance_bias = acceptance_bias
        self.alignment_weight = alignment_weight
        self.flexibility_weight = flexibility_weight
        self.threshold_weight = threshold_weight
        self.adversarial_weight = adversarial_weight
        self.final_weights = (final_coalition_weight, final_history_weight, final_self_weight)
        self.discussion_weights = (
            discussion_coalition_weight, discussion_self_weight,
            discussion_history_weight, discussion_info_weight,
        )

    @staticmethod
    def _history_consensus(history: List[Dict[str, Any]], issues: Dict[str, int]) -> Dict[str, Counter]:
        counts = {issue: Counter() for issue in issues}
        for row in history:
            deal = row["action"].get("deal")
            if not deal:
                continue
            for issue, option in deal.items():
                counts[issue][int(option)] += 1
        return counts

    def generate(
        self, game: Any, focal: str, beliefs: Dict[str, Any], history: List[Dict[str, Any]],
        turn_id: int, max_turns: int, final_vote: bool,
    ) -> List[Dict[str, Any]]:
        consensus = self._history_consensus(history, game.issues)
        rows = []
        for deal in enumerate_deals(game.issues):
            own_score = score_deal(game.players[focal].scores, deal)
            self_norm = normalized_score(game.players[focal].scores, deal)
            if own_score < game.players[focal].threshold:
                continue  # A final agreement rejected by focal is never useful.
            acceptance: Dict[str, float] = {}
            uncertainties = []
            for opponent, belief in beliefs.items():
                posteriors = belief.option_posteriors()
                # Compare each selected option with that issue's posterior mode.
                # This removes the domain-size bias of averaging raw probabilities.
                alignment = sum(
                    posteriors[issue][deal[issue] - 1] / max(max(posteriors[issue]), 1e-9)
                    for issue in game.issues
                ) / len(game.issues)
                probability = (
                    self.acceptance_bias + self.alignment_weight * alignment
                    + self.flexibility_weight * belief.flexibility
                    - self.threshold_weight * belief.threshold_ratio
                    - self.adversarial_weight * belief.adversarial_probability
                )
                acceptance[opponent] = max(0.01, min(0.99, probability))
                uncertainties.append(
                    sum(1.0 - max(values) for values in posteriors.values()) / len(posteriors)
                )
            p2_prob = acceptance.get(game.p2, 0.0)
            other_probs = sorted(
                (value for name, value in acceptance.items() if name != game.p2), reverse=True
            )
            others_needed = max(len(game.players) - 3, 0)
            coalition_floor = other_probs[others_needed - 1] if others_needed else 1.0
            coalition_prob = min(p2_prob, coalition_floor)
            predicted_opponent_supporters = sum(value >= self.acceptance_gate for value in acceptance.values())
            coalition_feasible = (
                p2_prob >= self.acceptance_gate
                and predicted_opponent_supporters >= len(game.players) - 2
            )
            history_terms = []
            for issue in game.issues:
                total = sum(consensus[issue].values())
                history_terms.append(consensus[issue][deal[issue]] / total if total else 0.0)
            history_score = sum(history_terms) / len(history_terms)
            time_ratio = turn_id / max(max_turns, 1)
            info_gain = (sum(uncertainties) / max(len(uncertainties), 1)) * (1.0 - time_ratio)
            if final_vote:
                # Exploration is worthless on the official final proposal.
                cw, hw, sw = self.final_weights
                planner_score = cw * coalition_prob + hw * history_score + sw * self_norm
            else:
                cw, sw, hw, iw = self.discussion_weights
                planner_score = (
                    cw * coalition_prob + sw * self_norm
                    + hw * history_score + iw * info_gain
                )
            rows.append({
                "deal": deal,
                "self_score": own_score,
                "self_norm": round(self_norm, 5),
                "p2_accept_prob": round(p2_prob, 5),
                "coalition_prob": round(coalition_prob, 5),
                "predicted_opponent_supporters": predicted_opponent_supporters,
                "coalition_feasible": coalition_feasible,
                "history_consensus": round(history_score, 5),
                "info_gain": round(0.0 if final_vote else info_gain, 5),
                "planner_score": round(planner_score, 5),
                "acceptance_by_opponent": {key: round(value, 5) for key, value in acceptance.items()},
                "planner_version": self.version,
            })
        if final_vote:
            rows.sort(key=lambda row: (
                row["coalition_feasible"], row["coalition_prob"],
                row["history_consensus"], row["self_score"],
            ), reverse=True)
        else:
            rows.sort(key=lambda row: (row["planner_score"], row["coalition_prob"], row["self_score"]), reverse=True)
        return rows[: self.candidate_top_k]

    @staticmethod
    def choose(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {"action_type": "abstain", "deal": None, "planner_version": "coalition_aware_v2"}
        return {**candidates[0], "action_type": "propose"}
