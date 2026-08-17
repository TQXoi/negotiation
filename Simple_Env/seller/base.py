from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    RegulatedLLMSeller as _PaperRegulatedLLMSeller,
)
from experiments.model_clients import ModelClient


class RLVRSeller(_PaperRegulatedLLMSeller):
    """Base fixed seller wrapper for the simple RLVR environment."""

    seller_name = "regulated_llm_seller"

    def __init__(self, client: ModelClient, max_tokens: int, temperature: float, top_p: float, persona: str):
        super().__init__(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            persona=persona,
        )
