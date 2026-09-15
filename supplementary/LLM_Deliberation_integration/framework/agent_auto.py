from __future__ import annotations

from typing import Any

from astra_integration.framework.agent_v2 import BeliefPlannerGeneratorRoleV2
from astra_integration.framework.auto_config import load_auto_config
from astra_integration.framework.planner_v2 import CoalitionAwarePlannerV2


class AutoResearchBeliefPlannerRole(BeliefPlannerGeneratorRoleV2):
    name = "belief_planner_generator_auto"

    def __init__(self, model: Any | None = None, *, candidate_top_k: int = 8, temperature: float = 0.0, max_tokens: int = 1600):
        super().__init__(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
        self.auto_config = load_auto_config()
        self.planner = CoalitionAwarePlannerV2(candidate_top_k, **self.auto_config)
