from __future__ import annotations

from typing import Any, Dict, List, Tuple

from experiments.model_clients import ModelClient

from CaSiNo_Env.environment.actions import NegotiationAction


class ASTRABuyer:
    name = "base_buyer"

    def __init__(
        self,
        model: ModelClient,
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1200,
        candidate_top_k: int = 8,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.candidate_top_k = candidate_top_k

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        raise NotImplementedError
