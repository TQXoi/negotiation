"""LLM-assisted semantic evidence extraction for belief updates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from experiments.agenticpay_framework.components import json_from_text
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import (
    ParsedAction,
    RLVRScenario,
    format_transcript,
)
from experiments.model_clients import ModelClient

from ..belief_model.base import BeliefEvidence, BeliefState, clamp


@dataclass
class LLMEvidenceUpdate:
    """Bounded semantic deltas proposed by an LLM.

    Deltas are intentionally small and clipped by the updater. This preserves
    the persistent state and prevents one persuasive seller utterance from
    overwriting all previous evidence.
    """

    finality_delta: float = 0.0
    quit_risk_delta: float = 0.0
    patience_delta: float = 0.0
    friendliness_delta: float = 0.0
    dominance_delta: float = 0.0
    confidence_delta: float = 0.0
    evidence: List[BeliefEvidence] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finality_delta": self.finality_delta,
            "quit_risk_delta": self.quit_risk_delta,
            "patience_delta": self.patience_delta,
            "friendliness_delta": self.friendliness_delta,
            "dominance_delta": self.dominance_delta,
            "confidence_delta": self.confidence_delta,
            "evidence": [x.to_dict() for x in self.evidence],
            "raw": self.raw,
        }


class LLMAssistedUpdater:
    """Extract semantic evidence and apply bounded scalar updates."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 900, max_abs_delta: float = 0.08):
        self.client = client
        self.max_tokens = max_tokens
        self.max_abs_delta = max_abs_delta
        self.last_raw = ""

    def extract(
        self,
        *,
        state: BeliefState,
        scenario: RLVRScenario,
        history: Sequence[Dict[str, Any]],
        buyer_action: Optional[ParsedAction],
        seller_action: Optional[ParsedAction],
        round_id: int,
        max_turns: int,
    ) -> LLMEvidenceUpdate:
        prompt = f"""You are a semantic evidence extractor for a persistent negotiation belief state.

Do not decide the buyer's next action. Do not overwrite the whole belief.
Only extract bounded evidence from the latest seller behavior.
Be conservative: seller language such as "final", "minimum", or "cannot go
lower" is weak evidence unless paired with repeated non-concession or quitting.
Do not increase quit risk just because the seller rejects one offer.

Buyer budget: {scenario.buyer_budget:.2f}
Reference/list price: {scenario.reference_price:.2f}
Round: {round_id}/{max_turns}
Previous belief state:
{json.dumps(state.to_dict(), ensure_ascii=False)}

Latest buyer action:
{buyer_action.to_dict() if buyer_action else None}

Latest seller action:
{seller_action.to_dict() if seller_action else None}

Conversation:
{format_transcript(history)}

Return only valid JSON:
{{
  "evidence": [
    {{"type": "finality_cue|flexibility_cue|patience_cue|dominance_cue|friendliness_cue|cost_pressure_cue|other",
      "quote": "short quote or action evidence",
      "effect": "short explanation",
      "confidence": <0-1>}}
  ],
  "suggested_deltas": {{
    "finality_delta": <-0.08 to 0.08>,
    "quit_risk_delta": <-0.08 to 0.08>,
    "patience_delta": <-0.08 to 0.08>,
    "friendliness_delta": <-0.08 to 0.08>,
    "dominance_delta": <-0.08 to 0.08>,
    "confidence_delta": <-0.1 to 0.1>
  }}
}}"""
        self.last_raw = self.client.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
        return self._parse(self.last_raw, round_id)

    def apply(self, state: BeliefState, update: LLMEvidenceUpdate) -> BeliefState:
        state.interaction.finality_prob = clamp(
            state.interaction.finality_prob + self._clip_delta(update.finality_delta, limit=self.max_abs_delta)
        )
        state.interaction.quit_risk = clamp(
            state.interaction.quit_risk + self._clip_delta(update.quit_risk_delta, limit=self.max_abs_delta * 0.75)
        )
        state.interaction.patience = clamp(state.interaction.patience + self._clip_delta(update.patience_delta))
        state.interaction.friendliness = clamp(
            state.interaction.friendliness + self._clip_delta(update.friendliness_delta)
        )
        state.interaction.dominance = clamp(state.interaction.dominance + self._clip_delta(update.dominance_delta))
        state.reservation.confidence = clamp(
            state.reservation.confidence + max(-0.10, min(0.10, update.confidence_delta))
        )
        for evidence in update.evidence:
            state.add_evidence(evidence)
        state.metadata["llm_evidence_raw"] = update.raw
        return state

    def _clip_delta(self, value: float, *, limit: Optional[float] = None) -> float:
        bound = self.max_abs_delta if limit is None else float(limit)
        return max(-bound, min(bound, float(value)))

    @staticmethod
    def _parse(raw: str, round_id: int) -> LLMEvidenceUpdate:
        try:
            parsed = json_from_text(raw)
        except Exception:
            parsed = {}
        deltas = parsed.get("suggested_deltas") if isinstance(parsed.get("suggested_deltas"), dict) else {}
        evidence_items = []
        for item in parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence_items.append(
                BeliefEvidence(
                    round_id=round_id,
                    source="llm",
                    evidence_type=str(item.get("type") or "semantic_evidence"),
                    text=str(item.get("quote") or ""),
                    effect=str(item.get("effect") or ""),
                    confidence=clamp(float(item.get("confidence") or 0.5)),
                )
            )
        return LLMEvidenceUpdate(
            finality_delta=float(deltas.get("finality_delta") or 0.0),
            quit_risk_delta=float(deltas.get("quit_risk_delta") or 0.0),
            patience_delta=float(deltas.get("patience_delta") or 0.0),
            friendliness_delta=float(deltas.get("friendliness_delta") or 0.0),
            dominance_delta=float(deltas.get("dominance_delta") or 0.0),
            confidence_delta=float(deltas.get("confidence_delta") or 0.0),
            evidence=evidence_items,
            raw=raw,
        )
