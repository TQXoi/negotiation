"""Core orchestration with structured-action locking."""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .belief import BeliefStore, SemanticBeliefUpdater, StructuredBeliefUpdater, TextGenerator
from .planner import BeliefUsablePlanner
from .schemas import CandidateAction, CanonicalState, FrameworkDecision, OpponentBelief, ScoredCandidate


class EnvironmentAdapter(Protocol):
    """Thin interface implemented once per environment family."""

    def candidates(self, state: CanonicalState, belief: OpponentBelief) -> List[CandidateAction]: ...

    def render_locked(
        self,
        state: CanonicalState,
        belief: OpponentBelief,
        selected: ScoredCandidate,
        language_client: Optional[TextGenerator],
    ) -> Any: ...

    def validate_locked(
        self,
        state: CanonicalState,
        selected: ScoredCandidate,
        rendered_action: Any,
    ) -> Tuple[Any, Dict[str, Any]]: ...


class UniversalNegotiationEngine:
    """Belief update -> candidate planning -> locked protocol realization."""

    def __init__(
        self,
        *,
        planner: Optional[BeliefUsablePlanner] = None,
        belief_store: Optional[BeliefStore] = None,
        semantic_client: Optional[TextGenerator] = None,
        semantic_updater: Optional[SemanticBeliefUpdater] = None,
        structured_updater: Optional[StructuredBeliefUpdater] = None,
        language_client: Optional[TextGenerator] = None,
        belief_mode: str = "learned",
        seed: int = 0,
    ):
        self.planner = planner or BeliefUsablePlanner()
        self.store = belief_store or BeliefStore()
        self.structured_updater = structured_updater or StructuredBeliefUpdater()
        self.semantic_updater = semantic_updater or (
            SemanticBeliefUpdater(semantic_client) if semantic_client else None
        )
        self.language_client = language_client
        self.belief_mode = belief_mode
        self.random = random.Random(seed)
        self.last_decision: Optional[FrameworkDecision] = None

    def decide(self, state: CanonicalState, adapter: EnvironmentAdapter) -> FrameworkDecision:
        stored = self.store.get(state.session_id, state.counterparty_id)
        unseen = self.store.unseen(state, state.observations)
        if self.belief_mode not in {"frozen", "no_update"}:
            self.structured_updater.update(stored, state, unseen)
        semantic_update = None
        if self.semantic_updater is not None and unseen and self.belief_mode not in {"frozen", "no_update"}:
            semantic_update = self.semantic_updater.update(stored, state, unseen)

        decision_belief = self._intervene(stored, state)
        candidates = adapter.candidates(state, decision_belief)
        ranked = self.planner.rank(state, decision_belief, candidates)
        selected = ranked[0]
        rendered = adapter.render_locked(state, decision_belief, selected, self.language_client)
        rendered, validation = adapter.validate_locked(state, selected, rendered)
        decision = FrameworkDecision(
            state=state,
            belief=copy.deepcopy(decision_belief),
            ranked_candidates=ranked,
            selected=selected,
            rendered_action=rendered,
            validation=validation,
            semantic_update=semantic_update,
        )
        self.last_decision = decision
        return decision

    def _intervene(self, belief: OpponentBelief, state: CanonicalState) -> OpponentBelief:
        result = copy.deepcopy(belief)
        if self.belief_mode in {"learned", "frozen", "no_update"}:
            return result
        if self.belief_mode == "wrong":
            preference = result.preference
            old_low, old_high = preference.reservation_ratio_low, preference.reservation_ratio_high
            preference.reservation_ratio_low = max(0.0, 1.0 - old_high)
            preference.reservation_ratio_high = min(1.5, 1.0 - old_low)
            preference.reservation_ratio_mean = (
                preference.reservation_ratio_low + preference.reservation_ratio_high
            ) / 2.0
            preference.reliability = 0.95
            return result
        if self.belief_mode == "shuffled":
            for scores in result.preference.issue_option_scores.values():
                keys = list(scores)
                values = list(scores.values())
                self.random.shuffle(values)
                scores.clear()
                scores.update(zip(keys, values))
            result.preference.reservation_ratio_mean = self.random.uniform(0.30, 0.95)
            result.preference.reliability = 0.90
            return result
        if self.belief_mode in {
            "wrong_policy",
            "wrong_policy_after_evidence",
            "wrong_policy_frontier_after_evidence",
        }:
            if self.belief_mode.endswith("after_evidence") and result.direct_response_count == 0:
                return result
            policy = result.response_policy
            policy.accept_alpha, policy.accept_beta = (
                list(policy.accept_beta),
                list(policy.accept_alpha),
            )
            policy.counter_rate = 1.0 - policy.counter_rate
            policy.quit_rate = 1.0 - policy.quit_rate
            policy.patience = 1.0 - policy.patience
            policy.reliability = 1.0
            if self.belief_mode == "wrong_policy_frontier_after_evidence":
                policy.rejected_compatibility_max = max(
                    0.05, 1.0 - policy.rejected_compatibility_max
                )
            return result
        if self.belief_mode in {
            "shuffled_full",
            "shuffled_full_after_evidence",
            "shuffled_full_frontier_after_evidence",
        }:
            if self.belief_mode.endswith("after_evidence") and result.direct_response_count == 0:
                return result
            preference = result.preference
            preference.reservation_ratio_mean = self.random.uniform(0.10, 1.10)
            preference.reliability = 0.95
            policy = result.response_policy
            combined = list(zip(policy.accept_alpha, policy.accept_beta))
            self.random.shuffle(combined)
            policy.accept_alpha = [beta for _, beta in combined]
            policy.accept_beta = [alpha for alpha, _ in combined]
            policy.quit_rate = self.random.uniform(0.0, 1.0)
            policy.patience = self.random.uniform(0.0, 1.0)
            policy.reliability = 1.0
            if self.belief_mode == "shuffled_full_frontier_after_evidence":
                policy.rejected_compatibility_max = self.random.uniform(0.05, 0.95)
            return result
        if self.belief_mode == "no_frontier_after_evidence":
            result.response_policy.rejected_compatibility_max = 0.0
            result.response_policy.last_rejected_compatibility = None
            result.response_policy.repeated_rejection_streak = 0
            return result
        if self.belief_mode == "shuffled_latent_profile_after_evidence":
            if result.direct_response_count == 0:
                return result
            weights = result.latent_profile_weights
            ratios = result.latent_profile_reservation_ratios
            if weights and set(weights) == set(ratios):
                values = list(weights.values())
                self.random.shuffle(values)
                result.latent_profile_weights = dict(zip(weights, values))
                ratio_mass: Dict[float, float] = {}
                for profile_id, weight in result.latent_profile_weights.items():
                    ratio = float(ratios[profile_id])
                    ratio_mass[ratio] = ratio_mass.get(ratio, 0.0) + float(weight)
                preference = result.preference
                preference.reservation_ratio_mean = sum(
                    ratio * mass for ratio, mass in ratio_mass.items()
                )
                cumulative = 0.0
                low_quantile = min(ratio_mass)
                high_quantile = max(ratio_mass)
                low_set = False
                for ratio in sorted(ratio_mass):
                    cumulative += ratio_mass[ratio]
                    if cumulative >= 0.10 and not low_set:
                        low_quantile = ratio
                        low_set = True
                    if cumulative >= 0.90:
                        high_quantile = ratio
                        break
                preference.reservation_ratio_low = low_quantile
                preference.reservation_ratio_high = high_quantile
                preference.reliability = 0.95
            policy = result.response_policy
            policy.quit_rate = self.random.uniform(0.0, 1.0)
            policy.patience = self.random.uniform(0.0, 1.0)
            policy.rejected_compatibility_max = self.random.uniform(0.05, 0.95)
            policy.reliability = 1.0
            return result
        if self.belief_mode in {"oracle", "oracle_utility"}:
            oracle = state.metadata.get("oracle_belief")
            if not isinstance(oracle, dict):
                raise ValueError("belief_mode=oracle requires state.metadata['oracle_belief'].")
            preference = result.preference
            ratio = oracle.get("reservation_ratio")
            if isinstance(ratio, (int, float)):
                preference.reservation_ratio_low = float(ratio)
                preference.reservation_ratio_mean = float(ratio)
                preference.reservation_ratio_high = float(ratio)
            option_scores = oracle.get("issue_option_scores")
            if isinstance(option_scores, dict):
                preference.issue_option_scores = copy.deepcopy(option_scores)
            preference.reliability = 1.0
            result.response_policy.reliability = 1.0
            return result
        raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

    def trace_json(self) -> str:
        if self.last_decision is None:
            return "{}"
        return json.dumps(self.last_decision.trace(), ensure_ascii=False, default=str)
