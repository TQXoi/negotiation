"""Warm cooperative prompt buyer baseline."""

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase


class WarmPromptBuyerAgent(BaselineBuyerBase):
    variant = "warm_prompt"
    prompt_guidance = (
        "BASELINE: Warm cooperative prompting. Be friendly, collaborative, and relationship-preserving, "
        "while still protecting buyer utility and never exceeding buyer max/value."
    )
