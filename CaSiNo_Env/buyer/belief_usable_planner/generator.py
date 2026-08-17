from __future__ import annotations

import json
from typing import Any, Dict, List

from experiments.model_clients import ModelClient

from CaSiNo_Env.environment.actions import NegotiationAction, action_format_instructions, parse_action
from CaSiNo_Env.environment.episode import visible_history


class FinalDialogueGenerator:
    """Use the LLM only to turn a chosen action into natural language."""

    def __init__(self, model: ModelClient, *, temperature: float, top_p: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        scenario: Any,
        history: List[Dict[str, Any]],
        belief: Dict[str, Any],
        chosen: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        round_id: int,
        max_rounds: int,
    ) -> tuple[NegotiationAction, str]:
        prompt = f"""You are participant P1 in a CaSiNo camping negotiation.

The planner has already selected the strategic action. Your job is to generate concise persuasive dialogue while preserving the selected action.

{scenario.public_context_for_p1()}

Round: {round_id}/{max_rounds}
Dialogue so far:
{visible_history(history)}

Compact belief about P2:
{json.dumps(belief, ensure_ascii=False, indent=2)}

Chosen action candidate:
{json.dumps(chosen, ensure_ascii=False, indent=2)}

Top scored candidates for context:
{json.dumps(candidates[:5], ensure_ascii=False, indent=2)}

Generation rules:
- Preserve the chosen candidate allocation exactly.
- Do not walk away unless the chosen candidate explicitly says walk_away.
- Mention a plausible tradeoff: concede lower-priority supplies while protecting your own important supplies.
- If early, use firm anchoring plus a light question about P2 priorities.
- If late, sound decisive and agreement-oriented.

{action_format_instructions("buyer/P1")}
"""
        raw = self.model.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        action = parse_action(raw)
        if chosen.get("action_type") == "accept":
            action.type = "accept"
            action.allocation = None
            if not action.message:
                action.message = "I can accept that split. Let's finalize it."
            return action, raw
        if chosen.get("allocation") is not None:
            # Keep the learned/math-selected action fixed even if the LLM drifts.
            action.allocation = chosen["allocation"]
            if action.type not in {"offer", "accept"}:
                action.type = "offer"
                action.parse_error = (action.parse_error or "") + "|generator_type_overridden_to_offer"
        return action, raw
