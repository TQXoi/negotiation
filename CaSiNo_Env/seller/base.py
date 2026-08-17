from __future__ import annotations

from typing import Any, Dict, List, Tuple

from experiments.model_clients import ModelClient

from CaSiNo_Env.environment.actions import NegotiationAction


class ASTRASeller:
    name = "base_seller"

    def __init__(
        self,
        model: ModelClient,
        *,
        personality: str = "base",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1200,
    ):
        self.model = model
        self.personality = personality
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def act(self, *, scenario: Any, history: List[Dict[str, Any]], round_id: int, max_rounds: int) -> Tuple[NegotiationAction, Dict[str, Any]]:
        raise NotImplementedError
