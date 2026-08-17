from __future__ import annotations

from typing import Any, Dict, List, Tuple

from CaSiNo_Env.environment.actions import NegotiationAction, action_format_instructions, parse_action
from CaSiNo_Env.environment.episode import visible_history
from CaSiNo_Env.seller.base import ASTRASeller


PERSONA_BLOCKS = {
    "base": "Negotiate naturally. Seek a beneficial agreement while keeping enough value for yourself.",
    "greedy": "Act tough and self-interested. Ask for more of your high-value supplies and concede slowly.",
    "fair": "Act cooperative and fairness-oriented. Try to find a balanced split that both sides can accept.",
    "procot": "Think step by step about priorities, feasible compromises, and how to persuade P1.",
}


class LLMPartnerSeller(ASTRASeller):
    name = "llm_partner"

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        persona = PERSONA_BLOCKS.get(self.personality, PERSONA_BLOCKS["base"])
        prompt = f"""You are participant P2 in a CaSiNo camping negotiation.

{scenario.public_context_for_p2()}

Persona:
{persona}

Round: {round_id}/{max_rounds}
Dialogue so far:
{visible_history(history)}

Policy:
- Reach an agreement if possible, but protect your high-value issues.
- If P1 makes a reasonable offer for your priorities, accept it.
- If P1 asks about your priorities, answer briefly without revealing exact point values.
- Avoid walking away until late rounds unless P1 repeatedly offers you very little value.

{action_format_instructions("seller/P2")}
"""
        raw = self.model.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        action = parse_action(raw)
        return action, {"prompt_type": self.name, "personality": self.personality, "raw": raw}
