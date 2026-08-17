"""Private chain-of-thought prompt buyer baseline."""

from experiments.agenticpay_framework.baselines.base import BaselineBuyerBase


class CoTPromptBuyerAgent(BaselineBuyerBase):
    variant = "cot_prompt"
    prompt_guidance = (
        "BASELINE: Private chain-of-thought prompting. Before writing the final message, silently check "
        "buyer max/value, latest seller offer, deal risk, non-price terms, and required output format. "
        "Do not reveal this reasoning; output only the final negotiation message."
    )
