"""Buyer variants supported by the modular AgenticPay wrapper.

The concrete AgenticPay agent classes live in
`experiments.agenticpay_framework` and the upstream AgenticPay repository. This
module keeps the experiment-facing registry small and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BuyerVariantSpec:
    name: str
    family: str
    source: str
    description: str


BUYER_VARIANTS: Dict[str, BuyerVariantSpec] = {
    "repo_native": BuyerVariantSpec(
        name="repo_native",
        family="native",
        source="AgenticPay paper/repo native BuyerAgent",
        description="Original AgenticPay buyer prompt and behavior.",
    ),
    "direct_prompt": BuyerVariantSpec(
        name="direct_prompt",
        family="prompt",
        source="direct prompting baseline",
        description="Native buyer with a minimal direct bargaining reminder.",
    ),
    "cot_prompt": BuyerVariantSpec(
        name="cot_prompt",
        family="prompt",
        source="CoT / Pro-CoT-style baseline",
        description="Prompt-only baseline with private self-check reasoning.",
    ),
    "warm_prompt": BuyerVariantSpec(
        name="warm_prompt",
        family="prompt",
        source="style prompting baseline",
        description="Cooperative/warm bargaining style prompt.",
    ),
    "dominant_prompt": BuyerVariantSpec(
        name="dominant_prompt",
        family="prompt",
        source="style prompting / anchoring baseline",
        description="Firm/dominant anchoring style prompt.",
    ),
    "rule_offer_generator": BuyerVariantSpec(
        name="rule_offer_generator",
        family="planner",
        source="deterministic concession-schedule baseline",
        description="Rule offer planner plus native naturalizer/validator.",
    ),
    "astra_baseline": BuyerVariantSpec(
        name="astra_baseline",
        family="belief_planner",
        source="ASTRA-inspired belief + offer optimization",
        description="Belief estimate plus heuristic overlap offer optimizer.",
    ),
    "belief_prompt": BuyerVariantSpec(
        name="belief_prompt",
        family="our_ablation",
        source="our framework ablation",
        description="Opponent belief model injected into native buyer naturalizer.",
    ),
    "planner_generator": BuyerVariantSpec(
        name="planner_generator",
        family="our_ablation",
        source="our framework ablation",
        description="Strategic planner plus native language generator, without belief.",
    ),
    "full_framework": BuyerVariantSpec(
        name="full_framework",
        family="our_framework",
        source="our A+B+C framework",
        description="Belief model -> strategic planner -> naturalizer -> validator.",
    ),
    "universal_framework_v1": BuyerVariantSpec(
        name="universal_framework_v1",
        family="our_framework",
        source="cross-environment universal belief-usable planner",
        description="Persistent per-opponent belief, normalized candidate planner, action lock, and thin AgenticPay adapter.",
    ),
    "universal_framework_v1_frozen": BuyerVariantSpec(
        name="universal_framework_v1_frozen",
        family="our_ablation",
        source="universal framework causal control",
        description="Same planner and adapter with a frozen prior belief.",
    ),
    "universal_framework_v1_wrong": BuyerVariantSpec(
        name="universal_framework_v1_wrong",
        family="our_ablation",
        source="universal framework causal control",
        description="Same planner with a deliberately inverted high-confidence reservation belief.",
    ),
    "universal_framework_v1_shuffled": BuyerVariantSpec(
        name="universal_framework_v1_shuffled",
        family="our_ablation",
        source="universal framework causal control",
        description="Same planner with shuffled opponent preferences.",
    ),
    "universal_framework_v2_conservative_awr": BuyerVariantSpec(
        name="universal_framework_v2_conservative_awr",
        family="our_framework",
        source="cross-environment conservative offline RL / AWR",
        description="Bootstrap lower-confidence residual planner with repeated-opponent memory and safe base fallback.",
    ),
    "universal_framework_v3_action_consistent": BuyerVariantSpec(
        name="universal_framework_v3_action_consistent",
        family="our_framework",
        source="cross-environment framework protocol-correctness iteration 001",
        description="V1 belief/planner with deterministic action-aware rendering and semantic validation.",
    ),
    "universal_framework_v4_terminal_accept_guard": BuyerVariantSpec(
        name="universal_framework_v4_terminal_accept_guard",
        family="our_framework",
        source="cross-environment framework reward iteration 002",
        description="V3 plus a public-proposal-backed conservative terminal accept/continue boundary.",
    ),
    "universal_framework_v5_verified_response_frontier": BuyerVariantSpec(
        name="universal_framework_v5_verified_response_frontier",
        family="our_framework",
        source="cross-environment framework reward iteration 003",
        description="V4 plus a verified public-response frontier that avoids repeating formally rejected offers.",
    ),
    "universal_framework_v6_public_proposal_feasibility_guard": BuyerVariantSpec(
        name="universal_framework_v6_public_proposal_feasibility_guard",
        family="our_framework",
        source="cross-environment framework reward iteration 004",
        description="V5 plus an early exact-public-proposal feasibility guard using repeated public commitments.",
    ),
    "universal_framework_v7_action_locked_strategic_naturalization": BuyerVariantSpec(
        name="universal_framework_v7_action_locked_strategic_naturalization",
        family="our_framework",
        source="cross-environment framework reward iteration 006",
        description="Frozen V4 planner plus public-context strategic OFFER language under exact action/contract locking.",
    ),
    "universal_framework_v8_constrained_rhetorical_selector": BuyerVariantSpec(
        name="universal_framework_v8_constrained_rhetorical_selector",
        family="our_framework",
        source="cross-environment framework reward iteration 007",
        description="Frozen V4 planner plus an LLM-selected finite rhetorical act rendered by trusted templates.",
    ),
    "universal_framework_v9_llm_proposal_candidate": BuyerVariantSpec(
        name="universal_framework_v9_llm_proposal_candidate",
        family="our_framework",
        source="cross-environment framework reward iteration 008",
        description="Frozen V4 plus one public-only structured LLM proposal candidate under common planner arbitration and action locking.",
    ),
    "universal_framework_v10_belief_grounded_candidate_arbitrator": BuyerVariantSpec(
        name="universal_framework_v10_belief_grounded_candidate_arbitrator",
        family="our_framework",
        source="cross-environment framework reward iteration 009",
        description="V9 candidate pool plus a stalled-state belief-grounded constrained candidate-ID arbitrator.",
    ),
    "universal_framework_v11_safe_improvement_arbitrator": BuyerVariantSpec(
        name="universal_framework_v11_safe_improvement_arbitrator",
        family="our_framework",
        source="cross-environment framework reward iteration 010",
        description="V10 residual candidate suggestion gated by a frozen-reference score, utility, acceptance, and terminal-action trust region.",
    ),
    "universal_framework_v12_reward_labeled_residual_awr": BuyerVariantSpec(
        name="universal_framework_v12_reward_labeled_residual_awr",
        family="our_framework",
        source="cross-environment reward-labelled conservative offline RL / AWR iteration 011",
        description="Ensemble-LCB residual policy trained on official reward labels over the frozen V4 planner, with a local safe-improvement trust region.",
    ),
    "universal_framework_v13_reward_labeled_lcb": BuyerVariantSpec(
        name="universal_framework_v13_reward_labeled_lcb",
        family="our_framework",
        source="cross-environment reward-labelled conservative offline RL / AWR iteration 012",
        description="V12 with the conflicting hand-written trust region removed; retains the frozen V4 base, ensemble LCB, OOD penalty, initial-jump guard, and exact fallback.",
    ),
    "universal_framework_v14_evidence_gated_lcb": BuyerVariantSpec(
        name="universal_framework_v14_evidence_gated_lcb",
        family="our_framework",
        source="cross-environment reward-first iteration 013",
        description="V13 with a universal abstention rule that forbids residual overrides before any direct opponent response.",
    ),
}


DEFAULT_BASELINES: List[str] = [
    "repo_native",
    "direct_prompt",
    "cot_prompt",
    "rule_offer_generator",
    "astra_baseline",
    "belief_prompt",
    "planner_generator",
    "full_framework",
    "universal_framework_v1",
]

FRAMEWORK_VARIANTS: List[str] = [
    "belief_prompt",
    "planner_generator",
    "full_framework",
    "universal_framework_v1",
    "universal_framework_v1_frozen",
    "universal_framework_v1_wrong",
    "universal_framework_v1_shuffled",
    "universal_framework_v2_conservative_awr",
    "universal_framework_v3_action_consistent",
    "universal_framework_v4_terminal_accept_guard",
    "universal_framework_v5_verified_response_frontier",
    "universal_framework_v6_public_proposal_feasibility_guard",
    "universal_framework_v7_action_locked_strategic_naturalization",
    "universal_framework_v8_constrained_rhetorical_selector",
    "universal_framework_v9_llm_proposal_candidate",
    "universal_framework_v10_belief_grounded_candidate_arbitrator",
    "universal_framework_v11_safe_improvement_arbitrator",
    "universal_framework_v12_reward_labeled_residual_awr",
    "universal_framework_v13_reward_labeled_lcb",
    "universal_framework_v14_evidence_gated_lcb",
]


def universal_belief_mode(variant: str) -> str | None:
    return {
        "universal_framework_v1": "learned",
        "universal_framework_v1_frozen": "frozen",
        "universal_framework_v1_wrong": "wrong",
        "universal_framework_v1_shuffled": "shuffled",
        "universal_framework_v2_conservative_awr": "learned",
        "universal_framework_v3_action_consistent": "learned",
        "universal_framework_v4_terminal_accept_guard": "learned",
        "universal_framework_v5_verified_response_frontier": "learned",
        "universal_framework_v6_public_proposal_feasibility_guard": "learned",
        "universal_framework_v7_action_locked_strategic_naturalization": "learned",
        "universal_framework_v8_constrained_rhetorical_selector": "learned",
        "universal_framework_v9_llm_proposal_candidate": "learned",
        "universal_framework_v10_belief_grounded_candidate_arbitrator": "learned",
        "universal_framework_v11_safe_improvement_arbitrator": "learned",
        "universal_framework_v12_reward_labeled_residual_awr": "learned",
        "universal_framework_v13_reward_labeled_lcb": "learned",
        "universal_framework_v14_evidence_gated_lcb": "learned",
    }.get(variant)


def parse_buyer_variants(raw: str | None) -> List[str]:
    if not raw or raw.strip().lower() in {"default", "all"}:
        return list(DEFAULT_BASELINES)
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(BUYER_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown AgenticPay buyer variants: {unknown}. Available: {sorted(BUYER_VARIANTS)}")
    return variants
