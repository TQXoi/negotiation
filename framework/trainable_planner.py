"""Trainable candidate-Q planner with environment-independent features.

The language model is deliberately outside this module.  Environment adapters
enumerate feasible actions, the frozen belief model supplies a public-history
posterior, and this planner only learns which normalized candidate should cross
the decision boundary.  Checkpoints are plain JSON so inference does not
depend on torch and remains easy to audit in every benchmark integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .planner import BehavioralFrontierPlanner
from .schemas import CandidateAction, CanonicalState, OpponentBelief, ScoredCandidate


FEATURE_NAMES = (
    "action_offer", "action_accept", "action_reject", "action_quit", "action_ask",
    "own_utility", "opponent_proxy", "adapter_acceptance", "information_gain",
    "feasibility_margin", "action_novelty",
    "base_p_accept", "base_p_quit", "base_exploitation", "base_exploration",
    "base_risk", "base_score", "base_top_gap",
    "progress", "remaining_turn_fraction",
    "reservation_low", "reservation_mean", "reservation_high", "reservation_width",
    "preference_reliability", "counter_rate", "quit_rate", "patience",
    "policy_reliability", "direct_response_count", "rejection_streak",
    "rejected_frontier", "frontier_distance", "behind_frontier", "frontier_jump",
    "aspiration_mean", "aspiration_width", "aspiration_reliability",
    "regime_change_probability", "posterior_entropy", "accept_regret", "probe_value",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def candidate_features(
    state: CanonicalState,
    belief: OpponentBelief,
    row: ScoredCandidate,
    *,
    base_top_score: float,
    best_offer_utility: float,
) -> list[float]:
    """Encode one feasible action without benchmark units or private truth."""

    candidate = row.candidate
    policy = belief.response_policy
    pref = belief.preference
    action = candidate.action_type
    frontier = _clamp(policy.rejected_compatibility_max)
    proxy = _clamp(candidate.opponent_value_proxy)
    entropy = belief.latent_profile_entropy_bits
    if entropy is None:
        entropy = 0.0
    uncertainty = 1.0 - 0.5 * (_clamp(pref.reliability) + _clamp(policy.reliability))
    accept_regret = (
        max(0.0, best_offer_utility - candidate.own_utility_normalized)
        if action == "accept" else 0.0
    )
    values = {
        "action_offer": action == "offer",
        "action_accept": action == "accept",
        "action_reject": action == "reject",
        "action_quit": action == "quit",
        "action_ask": action == "ask",
        "own_utility": candidate.own_utility_normalized,
        "opponent_proxy": proxy,
        "adapter_acceptance": _clamp(candidate.base_acceptance),
        "information_gain": _clamp(candidate.information_gain),
        "feasibility_margin": candidate.feasibility_margin,
        "action_novelty": _clamp(candidate.metadata.get("action_novelty", 1.0)),
        "base_p_accept": row.p_accept,
        "base_p_quit": row.p_quit,
        "base_exploitation": row.exploitation_value,
        "base_exploration": row.exploration_value,
        "base_risk": row.risk_penalty,
        "base_score": row.score,
        "base_top_gap": base_top_score - row.score,
        "progress": state.progress,
        "remaining_turn_fraction": 1.0 - state.progress,
        "reservation_low": pref.reservation_ratio_low,
        "reservation_mean": pref.reservation_ratio_mean,
        "reservation_high": pref.reservation_ratio_high,
        "reservation_width": pref.reservation_ratio_high - pref.reservation_ratio_low,
        "preference_reliability": pref.reliability,
        "counter_rate": policy.counter_rate,
        "quit_rate": policy.quit_rate,
        "patience": policy.patience,
        "policy_reliability": policy.reliability,
        "direct_response_count": min(1.0, belief.direct_response_count / 5.0),
        "rejection_streak": min(1.0, policy.repeated_rejection_streak / 4.0),
        "rejected_frontier": frontier,
        "frontier_distance": proxy - frontier,
        "behind_frontier": belief.direct_response_count > 0 and proxy <= frontier + 1e-9,
        "frontier_jump": max(0.0, proxy - frontier),
        "aspiration_mean": policy.aspiration_ratio_mean,
        "aspiration_width": policy.aspiration_ratio_high - policy.aspiration_ratio_low,
        "aspiration_reliability": policy.aspiration_reliability,
        "regime_change_probability": belief.regime_change_probability,
        "posterior_entropy": min(1.0, max(0.0, float(entropy)) / 6.0),
        "accept_regret": accept_regret,
        "probe_value": (
            _clamp(candidate.information_gain) * uncertainty * (1.0 - state.progress)
            if action in {"offer", "ask"} else 0.0
        ),
    }
    return [float(values[name]) for name in FEATURE_NAMES]


@dataclass(frozen=True)
class TrainablePlannerCheckpoint:
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    layers: tuple[dict, ...]
    base_residual_weight: float = 0.05

    @classmethod
    def load(cls, path: str | Path) -> "TrainablePlannerCheckpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "trainable_decision_boundary_planner_v2":
            raise ValueError(f"Unsupported planner checkpoint: {payload.get('schema_version')}")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Planner checkpoint feature schema does not match runtime.")
        return cls(
            feature_mean=tuple(float(x) for x in payload["normalization"]["mean"]),
            feature_std=tuple(float(x) for x in payload["normalization"]["std"]),
            layers=tuple(payload["layers"]),
            base_residual_weight=float(payload.get("base_residual_weight", 0.05)),
        )

    def predict(self, features: Sequence[float]) -> float:
        hidden = [
            (float(value) - mean) / max(1e-6, std)
            for value, mean, std in zip(features, self.feature_mean, self.feature_std)
        ]
        for layer_id, layer in enumerate(self.layers):
            weights = layer["weight"]
            bias = layer["bias"]
            output = [
                float(bias[row_id])
                + sum(float(weight) * value for weight, value in zip(weights[row_id], hidden))
                for row_id in range(len(weights))
            ]
            hidden = [math.tanh(value) for value in output] if layer_id + 1 < len(self.layers) else output
        return float(hidden[0])


class TrainableDecisionBoundaryPlanner:
    """Fine-tuned listwise candidate policy with a frozen safety/base scorer."""

    def __init__(
        self,
        checkpoint: str | Path | TrainablePlannerCheckpoint,
        base_planner: BehavioralFrontierPlanner | None = None,
        *,
        require_direct_evidence_for_override: bool = False,
        minimum_override_advantage: float = 0.0,
        minimum_accept_override_advantage: float | None = None,
        minimum_continue_over_accept_advantage: float | None = None,
        max_initial_proxy_jump: float | None = None,
    ):
        self.checkpoint = (
            checkpoint if isinstance(checkpoint, TrainablePlannerCheckpoint)
            else TrainablePlannerCheckpoint.load(checkpoint)
        )
        self.base_planner = base_planner or BehavioralFrontierPlanner()
        self.require_direct_evidence_for_override = bool(require_direct_evidence_for_override)
        self.minimum_override_advantage = float(minimum_override_advantage)
        self.minimum_accept_override_advantage = (
            None if minimum_accept_override_advantage is None
            else float(minimum_accept_override_advantage)
        )
        self.minimum_continue_over_accept_advantage = (
            None if minimum_continue_over_accept_advantage is None
            else float(minimum_continue_over_accept_advantage)
        )
        self.max_initial_proxy_jump = (
            None if max_initial_proxy_jump is None else float(max_initial_proxy_jump)
        )

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> list[ScoredCandidate]:
        ranked = self.base_planner.rank(state, belief, list(candidates))
        base_winner = ranked[0]
        base_top_score = max(row.score for row in ranked)
        best_offer_utility = max(
            (row.candidate.own_utility_normalized for row in ranked if row.candidate.action_type == "offer"),
            default=0.0,
        )
        for row in ranked:
            features = candidate_features(
                state, belief, row,
                base_top_score=base_top_score,
                best_offer_utility=best_offer_utility,
            )
            learned_q = self.checkpoint.predict(features)
            base_score = row.score
            row.score = learned_q + self.checkpoint.base_residual_weight * base_score
            row.diagnostics.update({
                "trainable_planner_v2": True,
                "learned_candidate_q": learned_q,
                "pre_finetune_base_score": base_score,
                "base_residual_weight": self.checkpoint.base_residual_weight,
            })
        learned_rank = sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )
        selected = learned_rank[0]
        guard_reason = None
        if belief.direct_response_count <= 0 and self.require_direct_evidence_for_override:
            selected = base_winner
            guard_reason = "no_direct_evidence_use_base_probe"
        elif belief.direct_response_count <= 0 and self.max_initial_proxy_jump is not None:
            cap = base_winner.candidate.opponent_value_proxy + self.max_initial_proxy_jump
            allowed = [
                row for row in learned_rank
                if row.candidate.action_type != "offer"
                or row.candidate.opponent_value_proxy <= cap + 1e-9
            ]
            if allowed:
                selected = allowed[0]
                if selected is not learned_rank[0]:
                    guard_reason = "initial_probe_jump_cap"
        learned_advantage = (
            selected.diagnostics["learned_candidate_q"]
            - base_winner.diagnostics["learned_candidate_q"]
        )
        required_advantage = self.minimum_override_advantage
        if (
            selected.candidate.action_type == "accept"
            and self.minimum_accept_override_advantage is not None
        ):
            required_advantage = self.minimum_accept_override_advantage
        if (
            base_winner.candidate.action_type == "accept"
            and selected.candidate.action_type != "accept"
            and self.minimum_continue_over_accept_advantage is not None
        ):
            required_advantage = self.minimum_continue_over_accept_advantage
        if selected is not base_winner and learned_advantage < required_advantage:
            selected = base_winner
            guard_reason = "learned_advantage_below_margin"
        selected.diagnostics.update({
            "deployment_guard_reason": guard_reason,
            "learned_advantage_over_base": learned_advantage,
            "base_winner_candidate_id": base_winner.candidate.candidate_id,
            "minimum_override_advantage": self.minimum_override_advantage,
            "required_override_advantage": required_advantage,
        })
        if selected is learned_rank[0]:
            return learned_rank
        # Preserve a rank-consistent trace after a safety/residual override.
        selected.score = max(row.score for row in learned_rank) + 1e-6
        return [selected] + [row for row in learned_rank if row is not selected]
