"""Dominant firm prompt buyer baseline."""

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase


class DominantPromptBuyerAgent(BaselineBuyerBase):
    variant = "dominant_prompt"
    prompt_guidance = (
        "BASELINE: Dominant firm prompting. Anchor firmly, resist unnecessary concessions, and use concise "
        "pressure, while remaining polite and never exceeding buyer max/value."
    )
