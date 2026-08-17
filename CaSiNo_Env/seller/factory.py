from __future__ import annotations

from experiments.model_clients import ModelClient

from CaSiNo_Env.seller.llm_partner import LLMPartnerSeller


def make_seller(name: str, model: ModelClient, **kwargs):
    normalized = name.strip().lower()
    if normalized in {"llm_partner", "partner", "base"}:
        return LLMPartnerSeller(model, **kwargs)
    raise ValueError(f"Unknown CaSiNo_Env seller variant: {name}")
