"""Direct-prompt buyer baseline."""

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase


class DirectPromptBuyerAgent(BaselineBuyerBase):
    variant = "direct_prompt"
    prompt_guidance = (
        "BASELINE: Direct prompting. Use the native AgenticPay instructions with no explicit belief model "
        "or planner. Optimize buyer utility while satisfying all output rules."
    )
