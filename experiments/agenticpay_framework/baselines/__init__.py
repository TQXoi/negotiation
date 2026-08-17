"""Buyer baseline registry for AgenticPay experiments."""

from __future__ import annotations

from typing import Any, Dict, Type

from experiments.agenticpay_framework.baselines.astra_baseline import ASTRABaselineBuyerAgent
from experiments.agenticpay_framework.baselines.cot_prompt import CoTPromptBuyerAgent
from experiments.agenticpay_framework.baselines.direct_prompt import DirectPromptBuyerAgent
from experiments.agenticpay_framework.baselines.dominant_prompt import DominantPromptBuyerAgent
from experiments.agenticpay_framework.baselines.rule_offer_generator import RuleOfferGeneratorBuyerAgent
from experiments.agenticpay_framework.baselines.warm_prompt import WarmPromptBuyerAgent
from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase


BASELINE_BUYER_AGENTS: Dict[str, Type[BaselineBuyerBase]] = {
    "direct_prompt": DirectPromptBuyerAgent,
    "cot_prompt": CoTPromptBuyerAgent,
    "warm_prompt": WarmPromptBuyerAgent,
    "dominant_prompt": DominantPromptBuyerAgent,
    "rule_offer_generator": RuleOfferGeneratorBuyerAgent,
    "astra_baseline": ASTRABaselineBuyerAgent,
}

BASELINE_BUYER_VARIANTS = set(BASELINE_BUYER_AGENTS)


def make_baseline_buyer_agent(variant: str, **kwargs: Any) -> BaselineBuyerBase:
    try:
        cls = BASELINE_BUYER_AGENTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline buyer variant: {variant}") from exc
    return cls(**kwargs)


__all__ = [
    "ASTRABaselineBuyerAgent",
    "BASELINE_BUYER_AGENTS",
    "BASELINE_BUYER_VARIANTS",
    "CoTPromptBuyerAgent",
    "DirectPromptBuyerAgent",
    "DominantPromptBuyerAgent",
    "RuleOfferGeneratorBuyerAgent",
    "WarmPromptBuyerAgent",
    "make_baseline_buyer_agent",
]
