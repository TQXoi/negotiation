"""NegotiationArena games matching the Opponent Simulation paper settings.

中文阅读导引：本文件只负责环境规则和 evaluator reward，不包含我们的 belief。
``PaperRepeatedBuySellGame`` 支持切换价格游戏先手；``PaperRepeatedTradingGame``
固定 RED 先手并计算资源交换后的真实 private utility。Resource-first/second 的差异
来自 runner 选择 RED 或 BLUE 为 focal，而不是使用了两套不同资源环境。
"""

from __future__ import annotations

from negotiationarena.constants import (
    ACCEPTING_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    REJECTION_TAG,
)
from negotiationarena.game_objects.goal import Goal
from negotiationarena.game_objects.trade import Trade
from negotiationarena.game_objects.valuation import Valuation
from games.buy_sell_game.game import BuySellGame
from games.trading_game.game import TradingGame


class PaperRepeatedBuySellGame(BuySellGame):
    """Buy/sell protocol with configurable first mover and robust termination."""

    def __init__(self, *args, start_turn: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = start_turn

    def game_over(self):
        state = self.game_state[-1]
        if not state:
            return False
        answer = state["player_public_info_dict"].get(PLAYER_ANSWER_TAG, "NONE")
        return answer in {ACCEPTING_TAG, REJECTION_TAG} or state.get("current_iteration") == self.iterations


class ValueMaximisationGoal(Goal):
    """Private additive utility over the change in resource holdings."""

    def __init__(self, initial_resources, valuation: Valuation):
        self.initial_resources = initial_resources
        self.valuation = valuation

    def __str__(self):
        values = ", ".join(
            f"each unit of {name} is worth {value} utility"
            for name, value in self.valuation.valuation_dict.items()
        )
        return f"Maximize the value of my final holdings; {values}."

    def __repr__(self):
        return str(self)

    def goal_reached(self, final_resources):
        # 资源 reward 是成交后相对初始持有的价值增量；没有成交时增量为 0。
        return self.valuation.value(final_resources - self.initial_resources)

    def json(self):
        return {
            "_type": "value_maximisation_goal",
            "_value": {
                "initial_resources": self.initial_resources.resource_dict,
                "valuation": self.valuation.valuation_dict,
            },
        }


class PaperRepeatedTradingGame(TradingGame):
    """Resource exchange with scalar utilities and configurable first mover."""

    def __init__(self, *args, start_turn: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = start_turn

    def after_game_ends(self):
        settings = self.game_state[0]["settings"]
        initial_resources = settings["player_initial_resources"]
        player_goals = settings["player_goals"]
        end_state = self.game_state[-1]
        answer = end_state["player_public_info_dict"].get(PLAYER_ANSWER_TAG, "NONE")
        proposed_trade = None
        agreement = answer == ACCEPTING_TAG
        if agreement:
            # ACCEPT applies only to the immediately preceding opponent action.
            # Searching farther back can make a player accept its own stale
            # counteroffer after the opponent has explicitly WAITed, producing
            # impossible negative-payoff "agreements".
            previous_state = self.game_state[-2] if len(self.game_state) >= 2 else {}
            previous_public = previous_state.get("player_public_info_dict", {})
            candidate = previous_public.get(PROPOSED_TRADE_TAG)
            previous_turn = previous_state.get("turn")
            accepting_turn = end_state.get("turn")
            if isinstance(candidate, Trade) and previous_turn != accepting_turn:
                proposed_trade = candidate
            agreement = proposed_trade is not None
        if agreement:
            final_resources = [
                proposed_trade.execute_trade(resources, idx)
                for idx, resources in enumerate(initial_resources)
            ]
        else:
            final_resources = initial_resources
        outcomes = [
            goal.goal_reached(final)
            for goal, final in zip(player_goals, final_resources)
        ]
        final_response = ACCEPTING_TAG if agreement else (
            "NONE" if answer == ACCEPTING_TAG else answer
        )
        self.game_state.append(
            {
                "current_iteration": "END",
                "turn": "None",
                "summary": {
                    "player_goals": player_goals,
                    "initial_resources": initial_resources,
                    "proposed_trade": proposed_trade,
                    "final_response": final_response,
                    "final_resources": final_resources,
                    "player_outcome": outcomes,
                },
            }
        )
