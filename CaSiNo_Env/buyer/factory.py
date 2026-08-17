from __future__ import annotations

from experiments.model_clients import ModelClient

from CaSiNo_Env.buyer.belief_usable_planner.buyer import BeliefUsableLLMChooserBuyer, BeliefUsablePlannerBuyer
from CaSiNo_Env.buyer.belief_planner import ASTRABeliefPlannerBuyer
from CaSiNo_Env.buyer.direct_prompt import DirectPromptBuyer


def make_buyer(name: str, model: ModelClient, **kwargs):
    normalized = name.strip().lower()
    if normalized in {"direct", "direct_prompt", "prompt"}:
        return DirectPromptBuyer(model, **kwargs)
    if normalized in {"astra_style_belief_planner", "belief_planner", "full_framework", "our_framework"}:
        return ASTRABeliefPlannerBuyer(model, **kwargs)
    if normalized in {"belief_usable_planner", "belief_usable", "weighted_belief_planner"}:
        return BeliefUsablePlannerBuyer(model, **kwargs)
    if normalized in {"belief_usable_llm_chooser", "belief_llm_chooser", "weighted_belief_llm_chooser"}:
        return BeliefUsableLLMChooserBuyer(model, **kwargs)
    raise ValueError(f"Unknown CaSiNo_Env buyer variant: {name}")
