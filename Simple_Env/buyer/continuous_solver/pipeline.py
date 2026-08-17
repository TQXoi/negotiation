"""Pipeline that connects persistent belief, updater, planner, and generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    parse_action,
)
from experiments.model_clients import ModelClient

from .belief_model.posterior import PosteriorBeliefModel
from .generator import GenerationResult, PromptActionGenerator, StrictScoredCandidateGenerator
from .planner import (
    BayesianLLMCandidatePlanner,
    ConservativeEVPlanner,
    ContextualParametricEVPlanner,
    ContextualPlannerPolicy,
    EVPlanner,
    ParametricEVPlanner,
    PlannerParams,
    PlannerResult,
    PromptBeliefPlanner,
)
from .update import HybridBeliefUpdater, LLMAssistedUpdater, RuleBasedBeliefUpdater


@dataclass
class PipelineResult:
    """One buyer action plus module diagnostics."""

    action: ParsedAction
    trace: Dict[str, Any]


class ContinualResolvingPipeline:
    """Persistent-belief buyer pipeline for RLVR Simple Env.

    The pipeline is deliberately configurable so experiments can swap one axis
    at a time:

    - belief model: posterior now; future variants can add particles/value nets
    - update: rule-only, LLM-only, or hybrid
    - planner: prompt planner or EV continual resolver
    - generator: paper-format prompt generator
    """

    def __init__(
        self,
        *,
        client: ModelClient,
        planner_client: Optional[ModelClient] = None,
        update_mode: str = "hybrid",
        planner_mode: str = "ev",
        candidate_offer_k: int = 5,
        planner_params: PlannerParams | None = None,
        contextual_policy: ContextualPlannerPolicy | None = None,
        max_tokens: int = 1200,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.client = client
        self.planner_client = planner_client or client
        self.update_mode = update_mode
        self.planner_mode = planner_mode
        self.candidate_offer_k = candidate_offer_k
        self.planner_params = planner_params
        self.contextual_policy = contextual_policy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.rule_updater = RuleBasedBeliefUpdater()
        self.llm_updater = (
            LLMAssistedUpdater(client, max_tokens=min(max_tokens, 900)) if update_mode in {"llm", "hybrid"} else None
        )
        self.hybrid_updater = HybridBeliefUpdater(self.rule_updater, self.llm_updater)
        if planner_mode == "ev_conservative":
            self.ev_planner = ConservativeEVPlanner(candidate_k=candidate_offer_k)
        elif planner_mode == "ev_contextual_parametric":
            self.ev_planner = ContextualParametricEVPlanner(candidate_k=candidate_offer_k, policy=contextual_policy)
        elif planner_mode == "ev_parametric":
            self.ev_planner = ParametricEVPlanner(candidate_k=candidate_offer_k, params=planner_params)
        else:
            self.ev_planner = EVPlanner(candidate_k=candidate_offer_k)
        self.prompt_planner = PromptBeliefPlanner(self.planner_client, max_tokens=min(max_tokens, 1200))
        self.bayesian_candidate_planner = BayesianLLMCandidatePlanner(
            self.planner_client,
            candidate_k=candidate_offer_k,
            max_tokens=min(max_tokens, 1400),
        )
        self.generator = PromptActionGenerator(
            client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.strict_candidate_generator = StrictScoredCandidateGenerator(
            client,
            max_tokens=max_tokens,
            temperature=min(0.8, temperature),
            top_p=top_p,
        )

        self._belief_model: Optional[PosteriorBeliefModel] = None
        self._scenario_key: Optional[str] = None
        self._processed_history_len = 0

    def reset(self) -> None:
        self._belief_model = None
        self._scenario_key = None
        self._processed_history_len = 0

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> PipelineResult:
        self._ensure_episode(scenario)
        assert self._belief_model is not None
        self._update_from_new_history(scenario=scenario, history=history, round_id=round_id, max_turns=max_turns)

        # Refresh curve before planning with planner-generated candidate prices.
        belief = self._belief_model.refresh_summary(self._default_candidate_prices(scenario))
        if self.planner_mode == "prompt":
            planner_result = self.prompt_planner.plan(
                scenario=scenario,
                history=history,
                belief=belief,
                round_id=round_id,
                max_turns=max_turns,
            )
        elif self.planner_mode == "bayesian_llm_candidates":
            planner_result = self.bayesian_candidate_planner.plan(
                scenario=scenario,
                history=history,
                belief=belief,
                posterior=self._belief_model.posterior,
                round_id=round_id,
                max_turns=max_turns,
            )
        elif self.planner_mode in {"ev", "ev_conservative", "ev_parametric", "ev_contextual_parametric"}:
            planner_result = self.ev_planner.plan(
                scenario=scenario,
                history=history,
                belief=belief,
                posterior=self._belief_model.posterior,
                round_id=round_id,
                max_turns=max_turns,
            )
        else:
            raise ValueError(f"Unknown continual solver planner_mode: {self.planner_mode}")

        generator = (
            self.strict_candidate_generator
            if self.planner_mode == "bayesian_llm_candidates"
            else self.generator
        )
        generation = generator.generate(
            scenario=scenario,
            history=history,
            belief=belief,
            planner_result=planner_result,
            round_id=round_id,
            max_turns=max_turns,
        )
        trace = self._trace(
            belief=belief,
            planner_result=planner_result,
            generation=generation,
            round_id=round_id,
        )
        return PipelineResult(action=generation.parsed_action, trace=trace)

    def _ensure_episode(self, scenario: RLVRScenario) -> None:
        if self._scenario_key == scenario.item_id and self._belief_model is not None:
            return
        self._scenario_key = scenario.item_id
        self._belief_model = PosteriorBeliefModel(scenario)
        self._processed_history_len = 0

    def _update_from_new_history(
        self,
        *,
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
            candidate_prices = self._default_candidate_prices(scenario)
            if self.update_mode == "rule":
                self.rule_updater.update(
                    belief_model=self._belief_model,
                    scenario=scenario,
                    history=history[: idx + 1],
                    buyer_action=buyer_action,
                    seller_action=seller_action,
                    round_id=int(item.get("round") or round_id),
                    max_turns=max_turns,
                    candidate_prices=candidate_prices,
                )
            elif self.update_mode in {"llm", "hybrid"}:
                self.hybrid_updater.update(
                    belief_model=self._belief_model,
                    scenario=scenario,
                    history=history[: idx + 1],
                    buyer_action=buyer_action,
                    seller_action=seller_action,
                    round_id=int(item.get("round") or round_id),
                    max_turns=max_turns,
                    candidate_prices=candidate_prices,
                )
            else:
                raise ValueError(f"Unknown continual solver update_mode: {self.update_mode}")
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

    def _nearest_prior_buyer_action(self, history: Sequence[Dict[str, Any]], seller_idx: int) -> Optional[ParsedAction]:
        for idx in range(seller_idx - 1, -1, -1):
            if history[idx].get("role") == "buyer":
                return self._parsed_from_history_item(history[idx])
        return None

    def _default_candidate_prices(self, scenario: RLVRScenario) -> list[float]:
        budget = float(scenario.buyer_budget)
        ref = float(scenario.reference_price)
        return sorted(
            {
                round(min(budget, max(0.01, ratio * ref)), 2)
                for ratio in [0.45, 0.55, 0.65, 0.75, 0.85]
            }
        )

    def _trace(
        self,
        *,
        belief,
        planner_result: PlannerResult,
        generation: GenerationResult,
        round_id: int,
    ) -> Dict[str, Any]:
        assert self._belief_model is not None
        return {
            "round": round_id,
            "variant": "continual_resolving",
            "update_mode": self.update_mode,
                "planner_mode": self.planner_mode,
                "planner_params": self.planner_params.to_dict() if self.planner_params is not None else None,
                "contextual_policy": self.contextual_policy.to_dict() if self.contextual_policy is not None else None,
            "belief": belief.to_dict(),
            "posterior": {
                "grid": self._belief_model.posterior.grid,
                "probs": self._belief_model.posterior.probs,
                "entropy": self._belief_model.posterior.entropy(),
            },
            "llm_update": (
                self.hybrid_updater.last_llm_update.to_dict()
                if self.hybrid_updater.last_llm_update is not None
                else None
            ),
            "planner": planner_result.to_dict(),
            "generation": generation.to_dict(),
        }
