"""Modular AgenticPay buyer framework with lazy upstream imports.

The lightweight schemas and Simple Env utilities remain importable before the
optional AgenticPay repository is bootstrapped. AgenticPay-specific classes are
loaded only when requested.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "BeliefState",
    "BASELINE_BUYER_VARIANTS",
    "FrameworkTrace",
    "ModularBuyerAgent",
    "NativeBuyerNaturalizer",
    "NaturalizationResult",
    "PromptOpponentBeliefModel",
    "PromptStrategicPlanner",
    "ResponseValidator",
    "StrategicPlan",
    "make_baseline_buyer_agent",
]


def __getattr__(name: str) -> Any:
    if name in {"BeliefState", "FrameworkTrace", "NaturalizationResult", "StrategicPlan"}:
        from experiments.agenticpay_framework import schemas

        return getattr(schemas, name)
    if name in {
        "NativeBuyerNaturalizer",
        "PromptOpponentBeliefModel",
        "PromptStrategicPlanner",
        "ResponseValidator",
    }:
        from experiments.agenticpay_framework import components

        return getattr(components, name)
    if name == "ModularBuyerAgent":
        from experiments.agenticpay_framework.buyer_agent import ModularBuyerAgent

        return ModularBuyerAgent
    if name in {"BASELINE_BUYER_VARIANTS", "make_baseline_buyer_agent"}:
        from experiments.agenticpay_framework import baselines

        return getattr(baselines, name)
    raise AttributeError(name)
