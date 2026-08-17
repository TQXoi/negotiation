from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRBuyer as _PaperRLVRBuyer
from experiments.model_clients import ModelClient


class RLVRBuyer(_PaperRLVRBuyer):
    """Base buyer wrapper for the simple RLVR environment.

    Subclasses only choose a buyer variant. The core action/parsing behavior is
    shared with the paper-aligned runner so metrics remain directly comparable.
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
            self.variant_name,
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if planner_client is not None:
            from experiments.agenticpay_framework.components import PromptStrategicPlanner

            self.planner = PromptStrategicPlanner(planner_client, max_tokens=min(max_tokens, 1200))
        if hasattr(self, "cf_response_model"):
            self.cf_response_model.candidate_top_k = max(1, int(candidate_offer_k))
