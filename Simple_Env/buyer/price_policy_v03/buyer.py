"""v0.3 buyer: small price policy + large verifier/generator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario
from experiments.model_clients import ModelClient

from ..base import RLVRBuyer
from .policy_model import SmallPricePolicyModel, compact_belief_summary
from .verifier_generator import PolicyVerifierGenerator


class SmallPolicyV03Buyer(RLVRBuyer):
    """Two-model buyer for cheap RL over price decisions.

    - planner_client: small trainable policy model, e.g. Qwen3-14B.
    - client: larger verifier/generator model, e.g. Qwen3-30B.
    """

    variant_name = "direct_prompt"

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
        self.variant = "small_policy_v03"
        self.use_full_framework_belief = False
        self.policy_client = planner_client or client
        self.price_policy = SmallPricePolicyModel(
            self.policy_client,
            max_tokens=min(max_tokens, 700),
            temperature=min(0.9, max(0.2, temperature)),
            top_p=top_p,
        )
        self.verifier_generator = PolicyVerifierGenerator(
            client,
            max_tokens=min(max_tokens, 1600),
            temperature=min(0.9, temperature),
            top_p=top_p,
        )

    def act(
        self,
        *,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        round_id: int,
        max_turns: int,
    ):
        belief = None
        if self.use_full_framework_belief:
            belief = self._belief(scenario, history, round_id, max_turns)
            belief = self._adapt_belief_style(belief, scenario, history, round_id, max_turns)
        policy_action = self.price_policy.propose(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
            belief=belief,
        )
        generation = self.verifier_generator.generate(
            scenario=scenario,
            history=history,
            round_id=round_id,
            max_turns=max_turns,
            policy_action=policy_action,
            belief=belief,
        )
        trace = {
            "round": round_id,
            "variant": self.variant,
            "architecture": "small_price_policy_plus_large_verifier_generator",
            "policy_model": getattr(self.policy_client, "model_id", ""),
            "generator_model": getattr(self.client, "model_id", ""),
            "belief": belief.to_dict() if belief else None,
            "belief_compact": compact_belief_summary(belief),
            "belief_raw": getattr(self.belief_model, "last_raw", "") if belief else None,
            "small_policy": {
                "prompt": self.price_policy.last_prompt,
                "raw": self.price_policy.last_raw,
                "parsed_action": policy_action.to_dict(),
            },
            "verifier": (
                self.verifier_generator.last_verified.to_dict()
                if self.verifier_generator.last_verified is not None
                else None
            ),
            "generation": generation.to_dict(),
            "buyer_action": generation.parsed_action.to_dict(),
            "validator": generation.validator,
        }
        return generation.parsed_action, trace


class SmallPolicyBeliefV03Buyer(SmallPolicyV03Buyer):
    """Full-framework belief + small price policy + 30B verifier/generator."""

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
        self.variant = "small_policy_belief_v03"
        self.use_full_framework_belief = True
