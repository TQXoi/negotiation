from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from experiments.model_clients import ModelClient

from CaSiNo_Env.environment.episode import visible_history


class LLMFinalChooser:
    """Choose among already-scored candidates with an LLM.

    The chooser cannot invent a new offer. It returns a candidate_id from the
    supplied table plus a short rationale. This makes it a clean comparison
    against deterministic belief_score selection.
    """

    def __init__(self, model: ModelClient, *, temperature: float, top_p: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = min(max_tokens, 900)

    def choose(
        self,
        *,
        scenario: Any,
        history: List[Dict[str, Any]],
        belief: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        deterministic_choice: Dict[str, Any],
        round_id: int,
        max_rounds: int,
    ) -> Dict[str, Any]:
        indexed = []
        for idx, candidate in enumerate(candidates):
            row = dict(candidate)
            row["candidate_id"] = idx
            indexed.append(row)
        if deterministic_choice.get("candidate_type") == "accept_last_seller_offer":
            accept_row = dict(deterministic_choice)
            accept_row["candidate_id"] = len(indexed)
            indexed.append(accept_row)

        prompt = f"""You are the strategic final chooser for participant P1 in a CaSiNo multi-issue negotiation.

You must choose exactly one candidate_id from the table. Do not invent a new allocation.

{scenario.public_context_for_p1()}

Round: {round_id}/{max_rounds}
Dialogue so far:
{visible_history(history)}

Compact mathematical belief about P2:
{json.dumps(belief, ensure_ascii=False, indent=2)}

Candidate table:
{json.dumps(indexed, ensure_ascii=False, indent=2)}

Deterministic belief-score choice:
{json.dumps(deterministic_choice, ensure_ascii=False, indent=2)}

Choosing policy:
- Early rounds: belief can be inaccurate. Prefer candidates that keep strong P1 value, preserve bargaining room, and reveal information, unless walkaway risk is very high.
- Middle rounds: use belief to find issue tradeoffs. Prefer candidates with good self_score and credible estimated partner utility.
- Late rounds: prioritize agreement probability while preserving reasonable P1 score.
- Avoid choosing a candidate that gives P2 almost nothing.
- Accept P2's offer only if it gives P1 a strong score or time is nearly over.

Reply with exactly one JSON object:
{{"candidate_id": 0, "rationale": "short reason"}}
"""
        raw = self.model.generate(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)
        choice = self._parse_choice(raw)
        candidate_id = choice.get("candidate_id")
        if not isinstance(candidate_id, int) or candidate_id < 0 or candidate_id >= len(indexed):
            chosen = deterministic_choice
            chosen = dict(chosen)
            chosen["llm_choose_error"] = "invalid_candidate_id"
            chosen["llm_raw_choice"] = raw
            return chosen
        chosen = dict(indexed[candidate_id])
        chosen["llm_rationale"] = str(choice.get("rationale", "")).strip()
        chosen["llm_raw_choice"] = raw
        return chosen

    @staticmethod
    def _parse_choice(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            data = json.loads(text)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
