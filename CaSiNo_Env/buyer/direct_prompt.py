from __future__ import annotations

from typing import Any, Dict, List, Tuple

from CaSiNo_Env.buyer.base import ASTRABuyer
from CaSiNo_Env.environment.actions import NegotiationAction, action_format_instructions, parse_action
from CaSiNo_Env.environment.episode import visible_history


class DirectPromptBuyer(ASTRABuyer):
    name = "direct_prompt"

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        prompt = f"""You are participant P1 in a CaSiNo camping negotiation.

Goal: reach an agreement that gives you as many points as possible while remaining plausible enough for P2 to accept.

{scenario.public_context_for_p1()}

Round: {round_id}/{max_rounds}
Dialogue so far:
{visible_history(history)}

Policy:
- Early rounds: ask for P2's priorities if they are unclear, but still make concrete offers.
- Do not walk away unless no agreement is possible near the final round.
- Prefer offers that keep more of your high-value issues and concede lower-value issues.
- If P2 made an offer that is acceptable and time is almost over, accept it.

{action_format_instructions("buyer/P1")}
"""
        raw = self.model.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        action = parse_action(raw)
        return action, {"prompt_type": self.name, "raw": raw}
