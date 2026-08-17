"""Conservative offline-RL residual planner for cross-environment negotiation.

The policy does not replace the hand-written belief-usable planner.  It learns
the *advantage over that planner's action* and may override it only when a
bootstrap lower confidence bound is positive.  This design makes abstention
(falling back to the base planner) the default on uncertain or out-of-domain
states.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Sequence

from .planner import BehavioralFrontierPlanner
from .schemas import CandidateAction, CanonicalState, OpponentBelief, ScoredCandidate
from .trainable_planner import FEATURE_NAMES, candidate_features


AWR_CONTEXT_FEATURE_NAMES = (
    "delta_own_utility_vs_base",
    "delta_opponent_proxy_vs_base",
    "delta_adapter_acceptance_vs_base",
    "delta_information_gain_vs_base",
    "delta_base_score_vs_base",
    "issue_count_scaled",
    "multi_issue",
    "continuous_issue_fraction",
    "discrete_issue_fraction",
    "offer_completeness",
    "term_tradeoff_candidate",
    "repeated_opponent",
    "cross_session_progress",
    "old_regime_weight",
    "new_regime_weight",
    "counterparty_count_scaled",
)
AWR_FEATURE_NAMES = FEATURE_NAMES + AWR_CONTEXT_FEATURE_NAMES


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def awr_candidate_features(
    state: CanonicalState,
    belief: OpponentBelief,
    row: ScoredCandidate,
    *,
    base_winner: ScoredCandidate,
    base_top_score: float,
    best_offer_utility: float,
) -> list[float]:
    """Encode a candidate in normalized units, including base-relative terms."""

    values = candidate_features(
        state,
        belief,
        row,
        base_top_score=base_top_score,
        best_offer_utility=best_offer_utility,
    )
    candidate = row.candidate
    base = base_winner.candidate
    issues = list(state.issues)
    issue_count = max(1, len(issues))
    continuous_count = sum(issue.kind == "continuous" for issue in issues)
    discrete_count = sum(issue.kind == "discrete" for issue in issues)
    offer = candidate.offer
    supplied = 0
    if offer is not None:
        supplied = int(offer.price is not None)
        supplied += len(offer.continuous_terms)
        supplied += len(offer.discrete_terms)
        supplied += len(offer.allocations)
    source = str(candidate.metadata.get("term_profile", ""))
    extras = {
        "delta_own_utility_vs_base": candidate.own_utility_normalized - base.own_utility_normalized,
        "delta_opponent_proxy_vs_base": candidate.opponent_value_proxy - base.opponent_value_proxy,
        "delta_adapter_acceptance_vs_base": candidate.base_acceptance - base.base_acceptance,
        "delta_information_gain_vs_base": candidate.information_gain - base.information_gain,
        "delta_base_score_vs_base": row.score - base_winner.score,
        "issue_count_scaled": min(1.0, len(issues) / 8.0),
        "multi_issue": len(issues) > 1,
        "continuous_issue_fraction": continuous_count / issue_count,
        "discrete_issue_fraction": discrete_count / issue_count,
        "offer_completeness": _clamp(supplied / issue_count) if offer is not None else 0.0,
        "term_tradeoff_candidate": source.startswith("tradeoff_"),
        "repeated_opponent": bool(state.metadata.get("repeated_opponent", False)),
        "cross_session_progress": _clamp(state.metadata.get("cross_session_progress", 0.0)),
        "old_regime_weight": _clamp(belief.old_regime_weight),
        "new_regime_weight": _clamp(belief.new_regime_weight),
        "counterparty_count_scaled": min(
            1.0, float(state.metadata.get("counterparty_count", 1)) / 5.0
        ),
    }
    values.extend(float(extras[name]) for name in AWR_CONTEXT_FEATURE_NAMES)
    return values


def _predict_layers(layers: Sequence[dict], values: Sequence[float]) -> float:
    hidden = [float(value) for value in values]
    for layer_id, layer in enumerate(layers):
        weights = layer["weight"]
        bias = layer["bias"]
        output = [
            float(bias[row_id])
            + sum(float(weight) * value for weight, value in zip(weights[row_id], hidden))
            for row_id in range(len(weights))
        ]
        hidden = (
            [math.tanh(value) for value in output]
            if layer_id + 1 < len(layers)
            else output
        )
    return float(hidden[0])


@dataclass(frozen=True)
class ConservativeAWRCheckpoint:
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    members: tuple[tuple[dict, ...], ...]
    uncertainty_scale: float = 1.0
    minimum_override_lcb: float = 0.015
    initial_override_lcb: float = 0.035
    accept_override_lcb: float = 0.025
    continue_over_accept_lcb: float = 0.035
    ood_z_threshold: float = 3.5
    ood_penalty: float = 0.02
    max_initial_proxy_jump: float = 0.12

    @classmethod
    def load(cls, path: str | Path) -> "ConservativeAWRCheckpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "conservative_cross_environment_awr_v1":
            raise ValueError(f"Unsupported AWR checkpoint: {payload.get('schema_version')}")
        if tuple(payload["feature_names"]) != AWR_FEATURE_NAMES:
            raise ValueError("AWR checkpoint feature schema does not match runtime.")
        deployment = payload.get("deployment", {})
        return cls(
            feature_mean=tuple(float(x) for x in payload["normalization"]["mean"]),
            feature_std=tuple(float(x) for x in payload["normalization"]["std"]),
            members=tuple(tuple(member["layers"]) for member in payload["ensemble"]),
            uncertainty_scale=float(deployment.get("uncertainty_scale", 1.0)),
            minimum_override_lcb=float(deployment.get("minimum_override_lcb", 0.015)),
            initial_override_lcb=float(deployment.get("initial_override_lcb", 0.035)),
            accept_override_lcb=float(deployment.get("accept_override_lcb", 0.025)),
            continue_over_accept_lcb=float(deployment.get("continue_over_accept_lcb", 0.035)),
            ood_z_threshold=float(deployment.get("ood_z_threshold", 3.5)),
            ood_penalty=float(deployment.get("ood_penalty", 0.02)),
            max_initial_proxy_jump=float(deployment.get("max_initial_proxy_jump", 0.12)),
        )

    def normalize(self, features: Sequence[float]) -> tuple[list[float], float]:
        normalized = [
            (float(value) - mean) / max(1e-6, std)
            for value, mean, std in zip(features, self.feature_mean, self.feature_std)
        ]
        z_rms = math.sqrt(sum(value * value for value in normalized) / max(1, len(normalized)))
        return normalized, z_rms

    def predict(self, features: Sequence[float]) -> tuple[float, float, float]:
        normalized, z_rms = self.normalize(features)
        predictions = [_predict_layers(member, normalized) for member in self.members]
        mean = statistics.fmean(predictions)
        std = statistics.pstdev(predictions) if len(predictions) > 1 else 0.0
        return mean, std, z_rms


class ConservativeAWRPlanner:
    """Safe residual policy trained with advantage-weighted regression.

    A learned action is used only if its predicted lower confidence bound
    clears an action-dependent threshold. Otherwise the base action is returned
    unchanged. No benchmark identity is used by the policy.
    """

    def __init__(
        self,
        checkpoint: str | Path | ConservativeAWRCheckpoint,
        base_planner: BehavioralFrontierPlanner | None = None,
        safe_improvement_gate: bool = False,
        require_direct_evidence: bool = False,
    ):
        self.checkpoint = (
            checkpoint
            if isinstance(checkpoint, ConservativeAWRCheckpoint)
            else ConservativeAWRCheckpoint.load(checkpoint)
        )
        self.base_planner = base_planner or BehavioralFrontierPlanner()
        # This gate is deliberately opt-in so historical V6.3/V2 checkpoints
        # retain bit-for-bit deployment semantics.  The reward-labelled V12
        # policy enables it as a second line of defence after the ensemble LCB.
        self.safe_improvement_gate = bool(safe_improvement_gate)
        # Opt-in to preserve every historical AWR variant.  When enabled, the
        # residual learner may rank candidates before any opponent response,
        # but it cannot override the frozen base action from prior alone.
        self.require_direct_evidence = bool(require_direct_evidence)

    @staticmethod
    def _passes_safe_improvement_gate(
        state: CanonicalState,
        base: ScoredCandidate,
        candidate: ScoredCandidate,
    ) -> tuple[bool, dict[str, float | bool]]:
        """Constrain learned residuals to a local offer-to-offer trust region.

        Terminal decisions remain owned by the frozen base planner.  A lower
        buyer-utility offer must buy enough estimated acceptance probability;
        a higher-utility offer may lose only a small amount of acceptance.
        """

        score_gap = max(0.0, float(base.diagnostics.get("base_pre_awr_score", base.score))
                        - float(candidate.diagnostics.get("base_pre_awr_score", candidate.score)))
        utility_loss = (
            base.candidate.own_utility_normalized
            - candidate.candidate.own_utility_normalized
        )
        acceptance_gain = candidate.p_accept - base.p_accept
        allowed_score_gap = 0.01 + 0.01 * state.progress
        allowed_utility_loss = 0.025 + 0.025 * state.progress
        checks = {
            "terminal_base_protected": base.candidate.action_type not in {"accept", "quit"},
            "offer_to_offer": (
                base.candidate.action_type == "offer"
                and candidate.candidate.action_type == "offer"
            ),
            "score_gap_ok": score_gap <= allowed_score_gap + 1e-12,
            "utility_loss_ok": utility_loss <= allowed_utility_loss + 1e-12,
            "concession_has_acceptance_gain": (
                utility_loss <= 0.0 or acceptance_gain >= max(0.04, utility_loss)
            ),
            "utility_gain_preserves_acceptance": (
                utility_loss > 0.0 or acceptance_gain >= -0.03
            ),
        }
        diagnostics: dict[str, float | bool] = {
            **checks,
            "safe_improvement_score_gap": score_gap,
            "safe_improvement_allowed_score_gap": allowed_score_gap,
            "safe_improvement_utility_loss": utility_loss,
            "safe_improvement_allowed_utility_loss": allowed_utility_loss,
            "safe_improvement_acceptance_gain": acceptance_gain,
        }
        return all(checks.values()), diagnostics

    def rank(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        candidates: Iterable[CandidateAction],
    ) -> list[ScoredCandidate]:
        ranked = self.base_planner.rank(state, belief, list(candidates))
        base_winner = ranked[0]
        base_top_score = base_winner.score
        best_offer_utility = max(
            (
                row.candidate.own_utility_normalized
                for row in ranked
                if row.candidate.action_type == "offer"
            ),
            default=0.0,
        )
        eligible: list[ScoredCandidate] = []
        for row in ranked:
            original_score = row.score
            features = awr_candidate_features(
                state,
                belief,
                row,
                base_winner=base_winner,
                base_top_score=base_top_score,
                best_offer_utility=best_offer_utility,
            )
            mean, std, z_rms = self.checkpoint.predict(features)
            if row is base_winner:
                mean, std, lcb = 0.0, 0.0, 0.0
            else:
                lcb = mean - self.checkpoint.uncertainty_scale * std
                lcb -= self.checkpoint.ood_penalty * max(
                    0.0, z_rms - self.checkpoint.ood_z_threshold
                )
            threshold = self.checkpoint.minimum_override_lcb
            if belief.direct_response_count <= 0:
                threshold = max(threshold, self.checkpoint.initial_override_lcb)
            if row.candidate.action_type == "accept":
                threshold = max(threshold, self.checkpoint.accept_override_lcb)
            if (
                base_winner.candidate.action_type == "accept"
                and row.candidate.action_type != "accept"
            ):
                threshold = max(threshold, self.checkpoint.continue_over_accept_lcb)
            initial_jump_ok = not (
                belief.direct_response_count <= 0
                and row.candidate.action_type == "offer"
                and row.candidate.opponent_value_proxy
                > base_winner.candidate.opponent_value_proxy
                + self.checkpoint.max_initial_proxy_jump
            )
            evidence_gate_ok = not (
                self.require_direct_evidence
                and belief.direct_response_count <= 0
                and row is not base_winner
            )
            override_eligible = (
                row is not base_winner
                and lcb >= threshold
                and initial_jump_ok
                and evidence_gate_ok
            )
            safe_gate_ok = True
            safe_gate_diagnostics: dict[str, float | bool] = {}
            if override_eligible and self.safe_improvement_gate:
                safe_gate_ok, safe_gate_diagnostics = self._passes_safe_improvement_gate(
                    state, base_winner, row
                )
                override_eligible = override_eligible and safe_gate_ok
            row.score = lcb
            row.diagnostics.update(
                {
                    "conservative_awr": True,
                    "base_pre_awr_score": original_score,
                    "predicted_advantage_mean": mean,
                    "predicted_advantage_std": std,
                    "predicted_advantage_lcb": lcb,
                    "override_threshold": threshold,
                    "override_eligible": override_eligible,
                    "initial_proxy_jump_ok": initial_jump_ok,
                    "feature_z_rms": z_rms,
                    "base_winner_candidate_id": base_winner.candidate.candidate_id,
                    "safe_improvement_gate_enabled": self.safe_improvement_gate,
                    "safe_improvement_gate_passed": safe_gate_ok,
                    "direct_evidence_gate_enabled": self.require_direct_evidence,
                    "direct_evidence_gate_passed": evidence_gate_ok,
                    **safe_gate_diagnostics,
                }
            )
            if override_eligible:
                eligible.append(row)

        selected = max(
            eligible,
            key=lambda item: (item.score, item.candidate.own_utility),
            default=base_winner,
        )
        selected.diagnostics["awr_selected_override"] = selected is not base_winner
        selected.diagnostics["awr_abstain_reason"] = (
            "no_direct_response"
            if selected is base_winner
            and self.require_direct_evidence
            and belief.direct_response_count <= 0
            else None
        )
        ordered = sorted(
            ranked,
            key=lambda item: (item.score, item.candidate.own_utility),
            reverse=True,
        )
        return [selected] + [row for row in ordered if row is not selected]
