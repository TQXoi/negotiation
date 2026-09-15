from __future__ import annotations

from typing import Any

from astra_integration.framework.agent_v2 import BeliefPlannerGeneratorRoleV2
from astra_integration.framework.planner_v3 import RobustCoalitionPlannerV3


class BeliefPlannerGeneratorRoleV3(BeliefPlannerGeneratorRoleV2):
    """Continuous belief + V2 generator protocol + robust final planner V3."""

    name = "belief_planner_generator_v3"

    def __init__(self, model: Any | None = None, *, candidate_top_k: int = 8, temperature: float = 0.0, max_tokens: int = 1600):
        super().__init__(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
        self.planner = RobustCoalitionPlannerV3(candidate_top_k=candidate_top_k)
