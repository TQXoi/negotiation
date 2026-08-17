"""Contextual CEM planner for Simple Env continuous resolving."""

from __future__ import annotations

from typing import Dict, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario

from ..belief_model.base import BeliefState
from .contextual_params import ContextualPlannerPolicy
from .parametric_ev_planner import ParametricEVPlanner


class ContextualParametricEVPlanner(ParametricEVPlanner):
    """Parametric planner that adapts its parameters from scenario/belief features."""

    def __init__(self, *, candidate_k: int = 5, policy: ContextualPlannerPolicy | None = None):
        self.policy = policy or ContextualPlannerPolicy()
        self._last_features = {}
        self._last_contextual_params = self.policy.base_params
        super().__init__(candidate_k=candidate_k, params=self.policy.base_params)

    def _bind_context(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict],
        belief: BeliefState,
        round_id: int,
        max_turns: int,
    ) -> None:
        params, features = self.policy.contextualize(
            scenario=scenario,
            history=history,
            belief=belief,
            round_id=round_id,
            max_turns=max_turns,
        )
        self.params = params
        self.quit_penalty = params.quit_penalty
        self.future_value_weight = params.future_value_weight
        self._last_features = features
        self._last_contextual_params = params

    def candidate_prices(self, *, scenario, history, belief, round_id, max_turns):
        self._bind_context(scenario=scenario, history=history, belief=belief, round_id=round_id, max_turns=max_turns)
        return super().candidate_prices(
            scenario=scenario,
            history=history,
            belief=belief,
            round_id=round_id,
            max_turns=max_turns,
        )

    def _score_offer(self, price, scenario, belief, posterior, round_id, max_turns):
        candidate = super()._score_offer(price, scenario, belief, posterior, round_id, max_turns)
        candidate.rationale += f"; contextual_features={self._last_features}"
        return candidate

    def plan(self, *, scenario, history, belief, posterior, round_id, max_turns):
        result = super().plan(
            scenario=scenario,
            history=history,
            belief=belief,
            posterior=posterior,
            round_id=round_id,
            max_turns=max_turns,
        )
        result.diagnostics["contextual_features"] = self._last_features
        result.diagnostics["contextual_params"] = self._last_contextual_params.to_dict()
        return result
