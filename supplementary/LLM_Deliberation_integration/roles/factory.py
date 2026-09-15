from __future__ import annotations

from astra_integration.framework.agent import BeliefPlannerGeneratorRole
from astra_integration.framework.agent_v2 import BeliefPlannerGeneratorRoleV2
from astra_integration.framework.agent_v3 import BeliefPlannerGeneratorRoleV3
from astra_integration.framework.agent_auto import AutoResearchBeliefPlannerRole
from astra_integration.roles.direct import DirectPromptRole
from astra_integration.roles.paper_structured import PaperStructuredRole
from astra_integration.roles.published_baselines import RepeatedRuleBasedRole
from astra_integration.roles.scripted import ScriptedRole


def make_role(name: str, model=None, *, candidate_top_k: int = 8, temperature: float = 0.0, max_tokens: int = 700, discussion_turns: int = 24, history_window: int = 6):
    if name == "scripted_threshold":
        return ScriptedRole()
    if name == "repeated_rule_based":
        return RepeatedRuleBasedRole()
    if name == "direct_prompt":
        if model is None:
            raise ValueError("direct_prompt requires a model")
        return DirectPromptRole(model, temperature=temperature, max_tokens=max_tokens)
    if name == "paper_structured":
        return PaperStructuredRole(
            model, discussion_turns=discussion_turns, history_window=history_window,
            temperature=temperature, max_tokens=max_tokens,
        )
    if name == "belief_planner_generator":
        return BeliefPlannerGeneratorRole(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
    if name == "belief_planner_generator_v2":
        return BeliefPlannerGeneratorRoleV2(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
    if name == "belief_planner_generator_v3":
        return BeliefPlannerGeneratorRoleV3(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
    if name == "belief_planner_generator_auto":
        return AutoResearchBeliefPlannerRole(model, candidate_top_k=candidate_top_k, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown role variant: {name}")
