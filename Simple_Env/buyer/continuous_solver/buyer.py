"""Simple_Env buyer wrapper for continual-resolving experiments."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import ParsedAction, RLVRScenario
from experiments.model_clients import ModelClient

from .pipeline import ContinualResolvingPipeline
from .planner import ContextualPlannerPolicy, PlannerParams


class ContinuousSolverBuyer:
    """Buyer interface compatible with Simple_Env.factory/eval."""

    update_mode = "hybrid"
    planner_mode = "ev"

    def __init__(
        self,
        client: ModelClient,
        max_tokens: int,
        temperature: float,
        top_p: float,
        candidate_offer_k: int = 5,
        planner_client: ModelClient | None = None,
        continuous_planner_params_json: str | None = None,
    ):
        self.variant = f"continuous_solver_{self.update_mode}_{self.planner_mode}"
        params_path = continuous_planner_params_json or os.environ.get("SIMPLE_ENV_CONTINUOUS_PLANNER_PARAMS_JSON")
        planner_params = PlannerParams.from_json(params_path) if params_path and self.planner_mode == "ev_parametric" else None
        contextual_policy = (
            ContextualPlannerPolicy.from_json(params_path)
            if params_path and self.planner_mode == "ev_contextual_parametric"
            else None
        )
        self.pipeline = ContinualResolvingPipeline(
            client=client,
            planner_client=planner_client,
            update_mode=self.update_mode,
            planner_mode=self.planner_mode,
            candidate_offer_k=candidate_offer_k,
            planner_params=planner_params,
            contextual_policy=contextual_policy,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ) -> tuple[ParsedAction, Dict[str, Any]]:
        result = self.pipeline.act(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
        )
        return result.action, result.trace


class ContinuousRuleEVBuyer(ContinuousSolverBuyer):
    update_mode = "rule"
    planner_mode = "ev"


class ContinuousRuleConservativeEVBuyer(ContinuousSolverBuyer):
    update_mode = "rule"
    planner_mode = "ev_conservative"


class ContinuousRuleCEMEVBuyer(ContinuousSolverBuyer):
    update_mode = "rule"
    planner_mode = "ev_parametric"


class ContinuousRuleContextualCEMEVBuyer(ContinuousSolverBuyer):
    update_mode = "rule"
    planner_mode = "ev_contextual_parametric"


class ContinuousHybridEVBuyer(ContinuousSolverBuyer):
    update_mode = "hybrid"
    planner_mode = "ev"


class ContinuousHybridPromptPlannerBuyer(ContinuousSolverBuyer):
    update_mode = "hybrid"
    planner_mode = "prompt"


class BayesianBeliefLLMPlannerBuyer(ContinuousSolverBuyer):
    """Bayesian posterior belief + LLM candidate planner + strict generator."""

    update_mode = "hybrid"
    planner_mode = "bayesian_llm_candidates"
