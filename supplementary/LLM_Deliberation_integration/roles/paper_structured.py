from __future__ import annotations

from typing import Any, Dict, List, Tuple

from initial_prompts import InitialPrompt
from rounds import RoundPrompts

from astra_integration.environment.actions import DeliberationAction, parse_paper_response


class PaperStructuredRole:
    """The upstream NeurIPS structured prompt, using the shared ModelClient."""

    name = "paper_structured"

    def __init__(self, model: Any, *, discussion_turns: int = 24, history_window: int = 6, temperature: float = 0.0, max_tokens: int = 1600):
        if model is None:
            raise ValueError("paper_structured requires a model")
        self.model = model
        self.discussion_turns = discussion_turns
        self.history_window = history_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.previous_plans: List[str] = []

    def act(
        self, *, game: Any, player: str, history: List[Dict[str, Any]], turn_id: int, max_turns: int, final_vote: bool
    ) -> Tuple[DeliberationAction, Dict[str, Any]]:
        spec = game.players[player]
        initial = InitialPrompt(
            game.game_dir, player, spec.file_name, game.p1, game.p2,
            num_issues=len(game.issues), num_agents=len(game.players), incentive=spec.incentive,
        ).return_initial_prompt()
        round_builder = RoundPrompts(
            player, game.p1, ",".join(f"{key}{value}" for key, value in game.initial_deal.items()),
            incentive=spec.incentive, window_size=self.history_window,
            rounds_num=self.discussion_turns, agents_num=len(game.players),
        )
        upstream_history = {
            "rounds": [
                {"agent": row["actor"], "public_answer": row["action"].get("message", "")}
                for row in history
            ],
            "plan": {player: self.previous_plans} if self.previous_plans else {},
        }
        paper_round_idx = self.discussion_turns if final_vote else turn_id
        slot_prompt = round_builder.build_slot_prompt(upstream_history, paper_round_idx)
        result = self.model.generate_messages(
            [{"role": "user", "content": initial}, {"role": "user", "content": slot_prompt}],
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=1.0,
        )
        action, plan = parse_paper_response(result.text, game.issues)
        if plan:
            self.previous_plans.append(plan)
        return action, {
            "prompt_type": self.name,
            "initial_prompt": initial,
            "slot_prompt": slot_prompt,
            "private_plan": plan,
            "raw": result.text,
            "final_vote": final_vote,
        }
