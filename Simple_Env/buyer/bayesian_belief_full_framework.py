"""Full-framework ablation with only the belief model replaced by Bayesian posterior."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from experiments.agenticpay_framework.schemas import BeliefState as AgenticPayBeliefState
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    parse_action,
)
from experiments.model_clients import ModelClient

from .base import RLVRBuyer
from .continuous_solver.belief_model.posterior import PosteriorBeliefModel
from .continuous_solver.update import HybridBeliefUpdater, LLMAssistedUpdater, RuleBasedBeliefUpdater


class BayesianBeliefFullFrameworkBuyer(RLVRBuyer):
    """Keep full_framework planner/generator, replace only belief with posterior.

    The base RLVR runner only knows the paper/framework variant names. We
    therefore initialize as `full_framework`, call the original full-framework
    act path, and relabel the returned trace for analysis.
    """

    variant_name = "full_framework"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 3,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        super().__init__(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_offer_k=candidate_offer_k,
            planner_client=planner_client,
            continuous_planner_params_json=continuous_planner_params_json,
        )
        self.output_variant = "bayesian_belief_full_framework"
        self._belief_model: Optional[PosteriorBeliefModel] = None
        self._scenario_key: Optional[str] = None
        self._processed_history_len = 0
        self._rule_updater = RuleBasedBeliefUpdater()
        self._llm_updater = LLMAssistedUpdater(client, max_tokens=min(max_tokens, 900))
        self._hybrid_updater = HybridBeliefUpdater(self._rule_updater, self._llm_updater)
        self._last_persistent_belief: Optional[Dict[str, Any]] = None

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ):
        original_variant = self.variant
        self.variant = "full_framework"
        try:
            action, trace = super().act(
                scenario=scenario,
                history=history,
                round_id=round_id,
                max_turns=max_turns,
            )
        finally:
            self.variant = original_variant
        trace["variant"] = self.output_variant
        trace["belief_model_override"] = "persistent_bayesian_reservation_posterior"
        if self._last_persistent_belief is not None:
            trace["persistent_bayesian_belief"] = self._last_persistent_belief
        return action, trace

    def _belief(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> AgenticPayBeliefState:
        self._ensure_episode(scenario)
        assert self._belief_model is not None
        self._update_from_new_history(scenario, history, round_id, max_turns)
        candidate_prices = self._candidate_prices(scenario, history)
        persistent = self._belief_model.refresh_summary(candidate_prices)
        self._last_persistent_belief = persistent.to_dict()
        compact = self._compact_model_belief(persistent)
        belief = persistent.to_agenticpay_belief()
        belief.seller_strategy = "persistent_bayesian_reservation_posterior"
        belief.contract_term_preferences = {
            **belief.contract_term_preferences,
            "compact_bayesian_posterior": compact,
            "acceptance_curve": compact["acceptance_curve"],
            "posterior_entropy": compact["posterior_entropy"],
            "update_mode": "hybrid_rule_plus_llm_semantic_delta",
        }
        belief.evidence = [
            *belief.evidence[-4:],
            "Compact Bayesian posterior summary shown to planner; full posterior saved only in trace/debug.",
            "Planner/generator are otherwise unchanged from full_framework.",
        ]
        return belief

    @staticmethod
    def _compact_model_belief(persistent) -> Dict[str, Any]:
        """Small model-facing summary; full posterior remains in trace only."""

        data = persistent.to_dict()
        evidence = []
        for item in (data.get("evidence") or [])[-4:]:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "round_id": item.get("round_id"),
                    "source": item.get("source"),
                    "type": item.get("evidence_type"),
                    "effect": item.get("effect"),
                    "confidence": item.get("confidence"),
                }
            )
        curve = []
        for point in (data.get("acceptance_curve") or [])[:8]:
            if not isinstance(point, dict):
                continue
            curve.append(
                {
                    "price": point.get("price"),
                    "p_accept": point.get("p_accept"),
                    "confidence": point.get("confidence"),
                }
            )
        return {
            "reservation": data.get("reservation"),
            "acceptance_curve": curve,
            "concession": data.get("concession"),
            "interaction": data.get("interaction"),
            "recent_evidence": evidence,
            "posterior_entropy": (data.get("metadata") or {}).get("posterior_entropy"),
            "debug_note": "full grid/probs/evidence are saved in trace.persistent_bayesian_belief",
        }

    def _ensure_episode(self, scenario: RLVRScenario) -> None:
        if self._scenario_key == scenario.item_id and self._belief_model is not None:
            return
        self._scenario_key = scenario.item_id
        self._belief_model = PosteriorBeliefModel(scenario)
        self._processed_history_len = 0
        self._last_persistent_belief = None

    def _update_from_new_history(
        self,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> None:
        assert self._belief_model is not None
        start = min(self._processed_history_len, len(history))
        for idx in range(start, len(history)):
            item = history[idx]
            if item.get("role") != "seller":
                continue
            seller_action = self._parsed_from_history_item(item)
            buyer_action = self._nearest_prior_buyer_action(history, idx)
            self._hybrid_updater.update(
                belief_model=self._belief_model,
                scenario=scenario,
                history=history[: idx + 1],
                buyer_action=buyer_action,
                seller_action=seller_action,
                round_id=int(item.get("round") or round_id),
                max_turns=max_turns,
                candidate_prices=self._candidate_prices(scenario, history[: idx + 1]),
            )
        self._processed_history_len = len(history)

    @staticmethod
    def _parsed_from_history_item(item: Dict[str, Any]) -> ParsedAction:
        action = item.get("action")
        if isinstance(action, dict):
            return ParsedAction(
                role=str(action.get("role") or item.get("role") or ""),
                action=str(action.get("action") or ""),
                price=action.get("price"),
                raw=str(action.get("raw") or item.get("message") or ""),
                valid=bool(action.get("valid", True)),
                item_spec=action.get("item_spec"),
            )
        return parse_action(str(item.get("role") or ""), str(item.get("message") or ""))

    def _nearest_prior_buyer_action(
        self,
        history: Sequence[Dict[str, Any]],
        seller_idx: int,
    ) -> Optional[ParsedAction]:
        for idx in range(seller_idx - 1, -1, -1):
            if history[idx].get("role") == "buyer":
                return self._parsed_from_history_item(history[idx])
        return None

    @staticmethod
    def _candidate_prices(scenario: RLVRScenario, history: Sequence[Dict[str, Any]]) -> list[float]:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        prices = {
            round(min(budget, max(0.01, ratio * ref)), 2)
            for ratio in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        }
        seller_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "seller" and (turn.get("action") or {}).get("price") is not None
        ]
        for price in seller_prices:
            prices.add(round(min(budget, price * 0.75), 2))
            prices.add(round(min(budget, price * 0.85), 2))
            prices.add(round(min(budget, price * 0.95), 2))
        buyer_prices = [
            float((turn.get("action") or {}).get("price"))
            for turn in history
            if turn.get("role") == "buyer" and (turn.get("action") or {}).get("price") is not None
        ]
        if buyer_prices:
            last = buyer_prices[-1]
            prices.add(round(min(budget, last + 0.03 * budget), 2))
            prices.add(round(min(budget, last + 0.06 * budget), 2))
        return sorted({p for p in prices if 0 < p <= budget})
