"""Environment-agnostic extraction of auditable public negotiation events.

Formal protocol actions and normalized offer features are verified evidence.
Natural-language statements such as "below cost" are retained only as
unverified claims with a public-text anchor; they are never utility labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Literal, Optional

from .schemas import CanonicalOffer, CanonicalState, NegotiationObservation


VerifiedOutcome = Literal["accept", "counter", "quit", "message"]


def _clamp(value: float, low: float = 0.0, high: float = 1.5) -> float:
    return max(low, min(high, float(value)))


def public_talk_and_action(message: str) -> str:
    """Remove private scratchpad text and retain only public Talk/Action."""
    raw = str(message or "")
    talk = re.search(r"(?:^|\n)Talk:\s*(.*?)(?=\nAction:|\Z)", raw, re.DOTALL)
    action = re.search(r"(?:^|\n)Action:\s*(.*)", raw, re.DOTALL)
    parts = []
    if talk:
        parts.append("Talk: " + talk.group(1).strip())
    if action:
        parts.append("Action: " + action.group(1).strip())
    # Environments without a Thought/Talk/Action protocol already expose a
    # public message, so preserve it. Never fall back to a known Thought block.
    if not parts and not re.search(r"(?:^|\n)Thought:\s*", raw):
        return raw.strip()
    return "\n".join(parts)


def _ratio(offer: Optional[CanonicalOffer], scale: float) -> Optional[float]:
    if offer is None or offer.price is None or scale <= 0:
        return None
    return _clamp(float(offer.price) / float(scale))


def _metadata_number(metadata: Dict[str, Any], key: str) -> Optional[float]:
    value = metadata.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _claims(public_text: str) -> Dict[str, Dict[str, Any]]:
    patterns = {
        "cost_floor": r"\b(?:below|under)\s+(?:my\s+)?cost\b|\bcost\s+(?:floor|price)\b",
        "final_offer": r"\b(?:final|best|lowest)\s+(?:price|offer)\b|\b(?:cannot|can't|won't)\s+(?:go|move)\s+(?:lower|further)\b",
        "urgency": r"\b(?:today|now|immediately|last chance|deadline)\b",
    }
    result: Dict[str, Dict[str, Any]] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, public_text, re.IGNORECASE)
        result[name] = {
            "present": bool(match),
            "public_span": match.group(0) if match else None,
            "truth_status": "unverified_claim" if match else "not_claimed",
        }
    return result


@dataclass(frozen=True)
class VerifiedNegotiationEvent:
    observation_id: str
    environment_id: str
    session_id: str
    actor_id: str
    counterparty_id: str
    turn: int
    progress: float
    formal_response: str
    outcome: VerifiedOutcome
    tested_compatibility: Optional[float]
    counter_compatibility: Optional[float]
    counter_concession: Optional[float]
    terminal: bool
    public_text: str
    claims: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verification: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_verified_event(
    state: CanonicalState,
    observation: NegotiationObservation,
    *,
    previous_counter_compatibility: Optional[float] = None,
) -> VerifiedNegotiationEvent:
    """Convert one canonical observation into a provenance-aware event."""
    response = str(observation.response_type)
    if response == "accept":
        outcome: VerifiedOutcome = "accept"
    elif response in {"counter", "reject"}:
        outcome = "counter"
    elif response == "quit":
        outcome = "quit"
    else:
        outcome = "message"

    metadata = dict(observation.metadata or {})
    tested = _metadata_number(metadata, "tested_compatibility")
    if tested is None:
        tested = _ratio(observation.response_to_offer, state.own_value_scale)
    counter = _metadata_number(metadata, "counter_compatibility")
    if counter is None:
        counter = _ratio(observation.offer, state.own_value_scale)
    concession = None
    if previous_counter_compatibility is not None and counter is not None:
        concession = float(previous_counter_compatibility) - counter
    text = public_talk_and_action(observation.text)
    formal_verified = response in {"accept", "counter", "reject", "quit"}
    return VerifiedNegotiationEvent(
        observation_id=observation.observation_id,
        environment_id=state.environment_id,
        session_id=state.session_id,
        actor_id=observation.actor_id,
        counterparty_id=observation.counterparty_id,
        turn=observation.turn,
        progress=max(0.0, min(1.0, observation.turn / max(1, state.max_turns))),
        formal_response=response,
        outcome=outcome,
        tested_compatibility=tested,
        counter_compatibility=counter,
        counter_concession=concession,
        terminal=outcome in {"accept", "quit"},
        public_text=text,
        claims=_claims(text),
        verification={
            "formal_response": formal_verified,
            "tested_offer_observed": tested is not None,
            "counter_offer_observed": counter is not None,
            "private_scratchpad_removed": "Thought:" not in text,
            "claim_truth_used_as_label": False,
        },
    )
