from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from astra_integration.environment.actions import DeliberationAction


class DeliberationRole(Protocol):
    name: str

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int, max_turns: int, final_vote: bool
    ) -> Tuple[DeliberationAction, Dict[str, Any]]: ...
