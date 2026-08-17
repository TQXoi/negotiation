from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from CaSiNo_Env.environment.casino import ITEMS, allocation_from_any


VALID_ACTIONS = {"offer", "accept", "walk_away", "ask_preference"}


@dataclass
class NegotiationAction:
    type: str
    message: str
    allocation: Optional[Dict[str, int]]
    raw_text: str
    parse_error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        obj = json.loads(cleaned)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_action(text: str) -> NegotiationAction:
    obj = extract_json_object(text)
    if obj is None:
        return NegotiationAction("walk_away", "", None, text, "no_json_object")
    action_type = str(obj.get("type", "")).strip().lower()
    message = str(obj.get("message", "")).strip()
    allocation = allocation_from_any(obj.get("allocation"))
    if action_type not in VALID_ACTIONS:
        return NegotiationAction("walk_away", message, allocation, text, f"invalid_action_type:{action_type}")
    if action_type == "offer" and allocation is None:
        return NegotiationAction("walk_away", message, None, text, "offer_missing_valid_allocation")
    if action_type == "accept" and allocation is not None:
        allocation = {item: int(allocation[item]) for item in ITEMS}
    return NegotiationAction(action_type, message, allocation, text, None)


def action_format_instructions(role: str) -> str:
    return (
        "Reply with exactly one JSON object and no extra text.\n"
        "Schema:\n"
        '{"type":"offer|accept|walk_away|ask_preference","message":"short natural-language message",'
        '"allocation":{"Food":0-3,"Water":0-3,"Firewood":0-3}}\n'
        f"For an offer, allocation means how many units YOU ({role}) keep. "
        "For accept, copy the previous offer allocation if it was yours to accept; allocation may also be null. "
        "For ask_preference or walk_away, allocation may be null."
    )
