"""Run the repeated NegotiationArena settings from arXiv:2602.19309.

The harness compares a repeated direct baseline, the paper's black-box
Opponent Simulation abstraction, and our dual-timescale belief planner under
the same game, opponent policy, first-mover position, model, and random seed.

中文阅读导引
------------
本文件是所有正式 NegotiationArena repeated-game 实验的统一入口：

* ``parse_args`` 注册方法名和四个 focal setting；
* ``make_agent`` 把 ``--method`` 映射到 Direct、Opponent Simulation 或 V2--V8；
* ``focal_index`` 决定哪一方使用被测试方法；
* ``resource_valuations`` 只在 evaluator 侧定义 RED/BLUE 私有效用；
* ``make_game`` 构造 Buyer--Seller 或 Resource Exchange，并固定先后手；
* ``summarize`` 汇总 focal reward、agreement、joint reward 和 belief diagnostics。

这里的 ``focal`` 不是第三个玩家，而是“当前被评估、其方法被替换的一方”。
价格游戏中 focal buyer/seller 都是后手；资源游戏中 ``resource_first`` 是
focal RED 先手，``resource_second`` 是 focal BLUE 后手。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

from arena_integration.repeated_agents import (
    DualTimescaleBeliefPlannerAgent,
    OpponentSimulationAgent,
    ProtocolFormatError,
    RepeatedLanguageAgent,
)
from arena_integration.decision_calibrated_agent import (
    ActionSpaceAdaptiveCommitmentBeliefPlannerAgent,
    CrossEpisodeInformationPlannerAgent,
    DecisionRelevantInformationPlannerAgent,
    EndogeneityCalibratedInformationPlannerAgent,
    FactorizedPreferencePolicyBeliefPlannerAgent,
    ReliabilityGatedFactorizedBeliefPlannerAgent,
    DecisionCalibratedBeliefPlannerAgent,
    ReciprocalCommitmentBeliefPlannerAgent,
    ScopedCommitmentBeliefPlannerAgent,
    UncertaintySafeguardedBeliefPlannerAgent,
)
from arena_integration.paper_aligned_opponent_simulation import PaperAlignedOpponentSimulationAgent
from arena_integration.repeated_games import (
    PaperRepeatedBuySellGame,
    PaperRepeatedTradingGame,
    ValueMaximisationGoal,
)
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.trade import Trade
from negotiationarena.game_objects.valuation import Valuation


BUY_SELL_RULE = "buyer reward = 63 - deal price; seller reward = deal price - 43"
RESOURCE_RULE = (
    "reward is the change in private additive resource value; RED values X=0.5,Y=2.5 "
    "and BLUE values X=2.5,Y=0.5"
)
def strategic_brainstorming_prompt(game_rule: str) -> str:
    """Prompt released by the Opponent Simulation authors (commit recorded in config)."""
    return f"""You should brainstorm and explicitly enumerate 5 high-level negotiation strategies or tactics,
and you must follow the strict format:
[(1). Strategy 1: xxx, yyy; (2). Strategy 2: xxx, yyy; ...], where xxx is the strategy name
and yyy is the strategy explanation.
You must brainstorm at every iteration/turn. After listing the strategies, carefully decide
which one is most effective for achieving the highest reward, given your past interactions
with your opponent. Justify your choice clearly.
Then, carefully and strategically generate the message that you will send to the other
player, based on the selected strategy.
Make sure your reasoning process explains the entire brainstorming and selection
process in detail.
Your reward is calculated as:
[{game_rule}]
In any episode where no deal is made, your reward is 0 (which is still better than accepting a
negative payoff).
Always aim to maximize your reward.
In the last line of your reasoning, you must declare your chosen strategy using this exact
format:
<strategy declaration> I have chosen Strategy x: yyy </strategy>"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "direct",
            "opponent_simulation",
            "opponent_simulation_paper",
            "framework",
            "framework_frozen",
            "framework_v2",
            "framework_v2_frozen",
            "framework_v3",
            "framework_v3_frozen",
            "framework_v3_1",
            "framework_v3_1_frozen",
            "framework_v3_2",
            "framework_v3_2_frozen",
            "framework_v3_3",
            "framework_v3_3_frozen",
            "framework_v4",
            "framework_v4_frozen",
            "framework_v4_wrong_confident",
            "framework_v4_shuffled",
            "framework_v4_oracle",
            "framework_v5",
            "framework_v5_frozen",
            "framework_v6",
            "framework_v6_frozen",
            "framework_v6_wrong_confident",
            "framework_v6_shuffled",
            "framework_v6_oracle",
            "framework_v7",
            "framework_v7_frozen",
            "framework_v7_wrong_confident",
            "framework_v7_shuffled",
            "framework_v7_oracle",
            "framework_v7_policy_uniform",
            "framework_v7_policy_shuffled",
            "framework_v8",
            "framework_v8_frozen",
            "framework_v8_wrong_confident",
            "framework_v8_shuffled",
            "framework_v8_oracle",
            "framework_v8_policy_uniform",
            "framework_v8_policy_shuffled",
            "framework_v8_no_cross_episode",
        ],
        required=True,
    )
    parser.add_argument(
        "--setting",
        choices=["buyer", "seller", "resource_first", "resource_second"],
        required=True,
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--episodes-per-run", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507-base")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--opponent-model")
    parser.add_argument("--opponent-base-url")
    parser.add_argument("--opponent-api-key-env")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--history-window", type=int, default=20)
    parser.add_argument("--context-char-limit", type=int, default=38000)
    parser.add_argument(
        "--protocol-mode",
        choices=["raw", "normalize", "normalize_retry"],
        default="raw",
        help="raw preserves the paper protocol; parser-safe modes are versioned diagnostics",
    )
    parser.add_argument("--protocol-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--no-language-realizer",
        action="store_true",
        help="use a deterministic public message while preserving the V2 locked action",
    )
    parser.add_argument(
        "--no-semantic-belief",
        action="store_true",
        help="ablate the bounded LLM extractor; retain structured continuous belief updates",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--restart-error-runs",
        action="store_true",
        help="when resuming, rerun an entire repeated-game run if any episode in it failed",
    )
    parser.add_argument(
        "--fail-fast-transport",
        action="store_true",
        help="stop the cell after persisting a model-service transport failure",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=["strategic_brainstorming", "direct"],
        default="strategic_brainstorming",
    )
    parser.add_argument(
        "--opponent-switch-episode",
        type=int,
        default=0,
        help="replace the opponent with a fresh instance at this episode; 0 disables",
    )
    parser.add_argument(
        "--opponent-switch-policy",
        choices=["strategic_brainstorming", "direct"],
        default="direct",
    )
    parser.add_argument(
        "--opponent-preference-switch-episode",
        type=int,
        default=0,
        help=(
            "resource games only: from this episode, change only the opponent's "
            "private X:Y valuation (5:1 to 2:1, or 1:5 to 1:2); 0 disables"
        ),
    )
    parser.add_argument(
        "--opponent-preference-switch-profile",
        choices=["toward_prior", "away_from_prior"],
        default="toward_prior",
        help="direction of the controlled resource-valuation change",
    )
    return parser.parse_args()


def make_agent(args, *, name, focal, seed, trace_dir):
    """根据实验方法名构造 agent；这是阅读各 variant 对应关系的首要入口。

    非 focal 一律使用 matched ``RepeatedLanguageAgent``。focal 才会被替换为
    Direct、Opponent Simulation 或我们的 framework。``wrong/shuffled/oracle``
    不是新 planner class，而是向同一 class 传入不同 ``belief_mode`` 的因果干预。
    """
    common = dict(
        agent_name=name,
        model=args.model,
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=seed,
        trace_dir=str(trace_dir),
        history_window=args.history_window,
        context_char_limit=args.context_char_limit,
        protocol_mode=args.protocol_mode,
        protocol_repair_attempts=args.protocol_repair_attempts,
        game_turn_limit=args.max_turns,
    )
    if not focal or args.method == "direct":
        if not focal:
            common.update(
                model=args.opponent_model or args.model,
                base_url=args.opponent_base_url or args.base_url,
                api_key=os.environ.get(args.opponent_api_key_env or args.api_key_env),
            )
        return RepeatedLanguageAgent(**common)
    if args.method == "opponent_simulation":
        return OpponentSimulationAgent(**common, candidate_count=args.candidate_count)
    if args.method == "opponent_simulation_paper":
        return PaperAlignedOpponentSimulationAgent(**common, candidate_count=args.candidate_count)
    if args.method in {"framework_v2", "framework_v2_frozen"}:
        return DecisionCalibratedBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v2_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {"framework_v3", "framework_v3_frozen"}:
        return UncertaintySafeguardedBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v3_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {"framework_v3_1", "framework_v3_1_frozen"}:
        return ReciprocalCommitmentBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v3_1_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {"framework_v3_2", "framework_v3_2_frozen"}:
        return ScopedCommitmentBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v3_2_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {"framework_v3_3", "framework_v3_3_frozen"}:
        return ActionSpaceAdaptiveCommitmentBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v3_3_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {
        "framework_v4", "framework_v4_frozen", "framework_v4_wrong_confident",
        "framework_v4_shuffled", "framework_v4_oracle",
    }:
        belief_mode = {
            "framework_v4_frozen": "frozen",
            "framework_v4_wrong_confident": "wrong_confident",
            "framework_v4_shuffled": "shuffled",
            "framework_v4_oracle": "oracle",
        }.get(args.method, "continuous")
        return CrossEpisodeInformationPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode=belief_mode,
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {"framework_v5", "framework_v5_frozen"}:
        return EndogeneityCalibratedInformationPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode="frozen" if args.method == "framework_v5_frozen" else "continuous",
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {
        "framework_v6", "framework_v6_frozen", "framework_v6_wrong_confident",
        "framework_v6_shuffled", "framework_v6_oracle",
    }:
        belief_mode = {
            "framework_v6_frozen": "frozen",
            "framework_v6_wrong_confident": "wrong_confident",
            "framework_v6_shuffled": "shuffled",
            "framework_v6_oracle": "oracle",
        }.get(args.method, "continuous")
        return DecisionRelevantInformationPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode=belief_mode,
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {
        "framework_v7", "framework_v7_frozen", "framework_v7_wrong_confident",
        "framework_v7_shuffled", "framework_v7_oracle",
        "framework_v7_policy_uniform", "framework_v7_policy_shuffled",
    }:
        belief_mode = {
            "framework_v7_frozen": "frozen",
            "framework_v7_wrong_confident": "wrong_confident",
            "framework_v7_shuffled": "shuffled",
            "framework_v7_oracle": "oracle",
            "framework_v7_policy_uniform": "policy_uniform",
            "framework_v7_policy_shuffled": "policy_shuffled",
        }.get(args.method, "continuous")
        return FactorizedPreferencePolicyBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode=belief_mode,
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    if args.method in {
        "framework_v8", "framework_v8_frozen", "framework_v8_wrong_confident",
        "framework_v8_shuffled", "framework_v8_oracle",
        "framework_v8_policy_uniform", "framework_v8_policy_shuffled",
        "framework_v8_no_cross_episode",
    }:
        belief_mode = {
            "framework_v8_frozen": "frozen",
            "framework_v8_wrong_confident": "wrong_confident",
            "framework_v8_shuffled": "shuffled",
            "framework_v8_oracle": "oracle",
            "framework_v8_policy_uniform": "policy_uniform",
            "framework_v8_policy_shuffled": "policy_shuffled",
            "framework_v8_no_cross_episode": "no_cross_episode",
        }.get(args.method, "continuous")
        return ReliabilityGatedFactorizedBeliefPlannerAgent(
            **common,
            candidate_count=args.candidate_count,
            belief_mode=belief_mode,
            language_realizer=not args.no_language_realizer,
            semantic_belief=not args.no_semantic_belief,
        )
    return DualTimescaleBeliefPlannerAgent(
        **common,
        candidate_count=args.candidate_count,
        belief_mode="frozen" if args.method == "framework_frozen" else "continuous",
    )


def focal_index(setting: str) -> int:
    # RED/AGENT_ONE 的 index 为 0，BLUE/AGENT_TWO 的 index 为 1。
    # buyer 与 resource_second 测 BLUE；seller 与 resource_first 测 RED。
    return 1 if setting in {"buyer", "resource_second"} else 0


def runs_requiring_restart(latest: dict[tuple[int, int], dict], enabled: bool) -> set[int]:
    """Return repeated-game runs that cannot be resumed episode-by-episode.

    The framework carries posterior state across the 20 episodes in a run.  If
    one episode failed, merely filling that episode later would condition it on
    a different history.  We therefore append a fresh copy of the whole run;
    the latest (run, episode) record remains the authoritative result while the
    older records stay in JSONL as an audit trail.
    """
    if not enabled:
        return set()
    return {int(row["run"]) for row in latest.values() if row.get("error")}


def is_transport_failure(row: dict) -> bool:
    error = str(row.get("error") or "")
    return any(
        marker in error
        for marker in (
            "URLError",
            "Connection refused",
            "Connection reset",
            "Remote end closed connection",
            "timed out",
            "HTTP 5",
        )
    )


def resource_valuations(args, episode):
    """Return actual private utilities for a controlled, moderate switch.

    The focal utility is invariant.  The opponent keeps the same issue ordering
    so the game remains complementary, but its normalized X weight changes
    5/6 -> 2/3 (BLUE) or 1/6 -> 1/3 (RED).
    """
    # index 0=RED：有较多 X、但 Y 更值钱；index 1=BLUE：有较多 Y、但 X 更值钱。
    # 这种互补偏好保证存在双方都能获得正效用的交换空间。
    valuations = [{"X": 0.5, "Y": 2.5}, {"X": 2.5, "Y": 0.5}]
    active = bool(
        args.opponent_preference_switch_episode > 0
        and episode >= args.opponent_preference_switch_episode
    )
    if active:
        opponent = 1 - focal_index(args.setting)
        if args.opponent_preference_switch_profile == "away_from_prior":
            valuations[opponent] = (
                {"X": 0.15, "Y": 2.85}
                if opponent == 0
                else {"X": 2.85, "Y": 0.15}
            )
        else:
            valuations[opponent] = (
                {"X": 1.0, "Y": 2.0}
                if opponent == 0
                else {"X": 2.0, "Y": 1.0}
            )
    return valuations


def prepare_agents(args, players, episode):
    if args.setting == "buyer":
        objectives = ["seller reward is deal price minus private production cost 43", "buyer reward is private WTP 63 minus deal price"]
    elif args.setting == "seller":
        objectives = ["seller reward is deal price minus private production cost 43", "buyer reward is private WTP 63 minus deal price"]
    else:
        values = resource_valuations(args, episode)
        objectives = [
            f"maximize net resource value with private values X={row['X']} and Y={row['Y']}"
            for row in values
        ]
    start_turn = 0
    if args.setting == "seller":
        start_turn = 1
    for idx, player in enumerate(players):
        player.prepare_episode(
            episode_index=episode,
            total_episodes=args.episodes_per_run,
            game_kind="buyer_seller" if args.setting in {"buyer", "seller"} else "resource_exchange",
            private_objective=objectives[idx],
            starts_episode=idx == start_turn,
        )
    if args.setting in {"resource_first", "resource_second"}:
        values = resource_valuations(args, episode)
        opponent = 1 - focal_index(args.setting)
        truth = values[opponent]["X"] / (values[opponent]["X"] + values[opponent]["Y"])
        focal_player = players[focal_index(args.setting)]
        if hasattr(focal_player, "evaluation_opponent_truth"):
            focal_player.evaluation_opponent_truth = truth


def make_game(args, players, out, run_index, episode):
    """构造一局并固定 focal role 与 first/second mover。

    Buyer--Seller 中角色恒定为 RED=seller、BLUE=buyer，但通过 ``start_turn``
    让被测试的 focal 一方处于后手。Resource Exchange 中始终 RED 先手；改变
    ``focal_index`` 得到 focal RED/first 与 focal BLUE/second 两个 setting。
    """
    focal = focal_index(args.setting)
    social = ["", ""]
    if args.opponent_policy == "strategic_brainstorming":
        prompt = strategic_brainstorming_prompt(
            BUY_SELL_RULE if args.setting in {"buyer", "seller"} else RESOURCE_RULE
        )
        # The released logs apply strategic brainstorming to both players.  A
        # previous local runner applied it only to the actual opponent, which
        # made the direct focal baseline an unmatched weaker prompt.
        social = [prompt, prompt]
    switch_active = bool(
        args.opponent_switch_episode > 0 and episode >= args.opponent_switch_episode
    )
    if switch_active and args.opponent_switch_policy == "direct":
        social[1 - focal] = ""
    log_path = out / "arena_logs" / f"run_{run_index:02d}" / f"episode_{episode:02d}"
    if args.setting in {"buyer", "seller"}:
        # The focal method starts second in both roles.  Roles and private goals
        # remain RED=seller, BLUE=buyer; only the initial turn changes.
        start_turn = 0 if args.setting == "buyer" else 1
        return PaperRepeatedBuySellGame(
            players=players,
            iterations=args.max_turns,
            start_turn=start_turn,
            player_goals=[
                SellerGoal(Valuation({"X": 43})),
                BuyerGoal(Valuation({"X": 63})),
            ],
            player_starting_resources=[Resources({"X": 1}), Resources({MONEY_TOKEN: 1000})],
            player_conversation_roles=[f"You are {AGENT_ONE}.", f"You are {AGENT_TWO}."],
            player_social_behaviour=social,
            log_path=str(log_path),
            log_dir=str(log_path.parent),
        )
    initial = [Resources({"X": 25, "Y": 5}), Resources({"X": 5, "Y": 25})]
    valuations = [Valuation(row) for row in resource_valuations(args, episode)]
    start_turn = 0  # RED first; selecting focal BLUE yields resource_second.
    return PaperRepeatedTradingGame(
        players=players,
        iterations=args.max_turns,
        start_turn=start_turn,
        resources_support_set=Resources({"X": 0, "Y": 0}),
        player_goals=[ValueMaximisationGoal(initial[i], valuations[i]) for i in range(2)],
        player_initial_resources=initial,
        player_social_behaviour=social,
        player_roles=[f"You are {AGENT_ONE}.", f"You are {AGENT_TWO}."],
        log_path=str(log_path),
        log_dir=str(log_path.parent),
    )


def serialize_deal(value):
    if isinstance(value, Trade):
        return {
            "from_red": value.resources_from_first_agent.resource_dict,
            "from_blue": value.resources_from_second_agent.resource_dict,
        }
    return None


def protocol_counts(player):
    return {
        "invalid_raw": player.protocol_invalid_raw_count,
        "deterministic_repairs": player.protocol_deterministic_repair_count,
        "retry_calls": player.protocol_retry_count,
        "repair_failures": player.protocol_repair_failure_count,
    }


def protocol_deltas(players, before):
    return [
        {key: protocol_counts(player)[key] - before[index][key] for key in before[index]}
        for index, player in enumerate(players)
    ]


def episode_result(args, run_index, episode, game, players, elapsed, calls_before, protocol_before):
    summary = game.game_state[-1].get("summary", {})
    outcomes = summary.get("player_outcome", [0.0, 0.0])
    agreement = summary.get("final_response") == "ACCEPT"
    trade = summary.get("proposed_trade")
    deal = serialize_deal(trade)
    price = None
    if agreement and isinstance(trade, Trade) and args.setting in {"buyer", "seller"}:
        price = trade.resources_from_second_agent.resource_dict.get(MONEY_TOKEN)
        deal = price
    transcript = game.game_state[1:-1]
    for idx, player in enumerate(players):
        player.record_episode(
            transcript=transcript,
            own_reward=float(outcomes[idx]),
            agreement=agreement,
            deal=deal,
        )
    focal = focal_index(args.setting)
    protocol_stats = protocol_deltas(players, protocol_before)
    diagnostics = (
        players[focal].episode_diagnostics()
        if hasattr(players[focal], "episode_diagnostics")
        else None
    )
    return {
        "method": args.method,
        "setting": args.setting,
        "run": run_index,
        "episode": episode,
        "agreement": agreement,
        "deal": deal,
        "price": price,
        "focal_reward": float(outcomes[focal]),
        "opponent_reward": float(outcomes[1 - focal]),
        "joint_reward": float(sum(outcomes)),
        "turns": len(transcript),
        "focal_model_calls": players[focal].call_index - calls_before[focal],
        "opponent_model_calls": players[1 - focal].call_index - calls_before[1 - focal],
        "focal_protocol": protocol_stats[focal],
        "opponent_protocol": protocol_stats[1 - focal],
        "focal_diagnostics": diagnostics,
        "opponent_switched": bool(
            args.opponent_switch_episode > 0 and episode >= args.opponent_switch_episode
        ),
        "opponent_preference_switched": bool(
            args.opponent_preference_switch_episode > 0
            and episode >= args.opponent_preference_switch_episode
        ),
        "elapsed_seconds": round(elapsed, 3),
        "error": None,
    }


def policy_protocol_failure_result(
    args, run_index, episode, game, players, elapsed, calls_before,
    protocol_before, exc,
):
    """Record an invalid model action as a zero-reward policy outcome.

    A format-repair failure is not missing evaluation data: the deployed policy
    failed to emit an executable action.  Treating it as an infrastructure error
    causes deterministic seeds to be retried forever and can silently select a
    lucky rollout.  We preserve the failure, assign no deal/zero reward, and
    reserve ``error`` for transport/API/harness failures that require a rerun.
    """
    focal = focal_index(args.setting)
    protocol_stats = protocol_deltas(players, protocol_before)
    actor = "focal" if protocol_stats[focal].get("repair_failures", 0) else "opponent"
    transcript = list(getattr(game, "game_state", [])[1:])
    for player in players:
        player.record_episode(
            transcript=transcript,
            own_reward=0.0,
            agreement=False,
            deal=None,
        )
    diagnostics = (
        players[focal].episode_diagnostics()
        if hasattr(players[focal], "episode_diagnostics")
        else None
    )
    return {
        "method": args.method,
        "setting": args.setting,
        "run": run_index,
        "episode": episode,
        "agreement": False,
        "deal": None,
        "price": None,
        "focal_reward": 0.0,
        "opponent_reward": 0.0,
        "joint_reward": 0.0,
        "turns": len(transcript),
        "focal_model_calls": players[focal].call_index - calls_before[focal],
        "opponent_model_calls": players[1 - focal].call_index - calls_before[1 - focal],
        "focal_protocol": protocol_stats[focal],
        "opponent_protocol": protocol_stats[1 - focal],
        "focal_diagnostics": diagnostics,
        "opponent_switched": bool(
            args.opponent_switch_episode > 0
            and episode >= args.opponent_switch_episode
        ),
        "opponent_preference_switched": bool(
            args.opponent_preference_switch_episode > 0
            and episode >= args.opponent_preference_switch_episode
        ),
        "elapsed_seconds": round(elapsed, 3),
        "policy_protocol_failure": {
            "type": type(exc).__name__,
            "actor": actor,
            "message": str(exc),
        },
        "policy_failure_traceback": traceback.format_exc(),
        "error": None,
    }


def summarize(rows):
    valid = [row for row in rows if not row.get("error")]
    policy_failures = [row for row in valid if row.get("policy_protocol_failure")]
    by_run: dict[int, list[dict[str, Any]]] = {}
    for row in valid:
        by_run.setdefault(int(row["run"]), []).append(row)
    run_metrics = []
    for run, items in sorted(by_run.items()):
        items.sort(key=lambda row: row["episode"])
        run_metrics.append(
            {
                "run": run,
                "episodes": len(items),
                "agreement_rate": mean(float(row["agreement"]) for row in items),
                "mean_focal_reward": mean(row["focal_reward"] for row in items),
                "early_reward_ep1_5": mean(row["focal_reward"] for row in items[:5]),
                "late_reward_ep16_20": mean(row["focal_reward"] for row in items[-5:]),
                "pre_switch_reward": mean(
                    row["focal_reward"] for row in items
                    if not (row.get("opponent_preference_switched") or row.get("opponent_switched"))
                ) if any(
                    not (row.get("opponent_preference_switched") or row.get("opponent_switched"))
                    for row in items
                ) else None,
                "post_switch_reward": mean(
                    row["focal_reward"] for row in items
                    if row.get("opponent_preference_switched") or row.get("opponent_switched")
                ) if any(
                    row.get("opponent_preference_switched") or row.get("opponent_switched")
                    for row in items
                ) else None,
            }
        )
    calibration = [
        item
        for row in valid
        for item in (row.get("focal_diagnostics") or {}).get("calibration", [])
    ]
    ece = None
    if calibration:
        weighted_error = 0.0
        for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
            upper = lower + 0.2
            bucket = [
                item for item in calibration
                if lower <= float(item["predicted_accept"]) < upper
                or (upper == 1.0 and float(item["predicted_accept"]) == 1.0)
            ]
            if bucket:
                confidence = mean(float(item["predicted_accept"]) for item in bucket)
                accuracy = mean(float(item["accepted"]) for item in bucket)
                weighted_error += len(bucket) / len(calibration) * abs(confidence - accuracy)
        ece = weighted_error
    diagnostics = [row.get("focal_diagnostics") for row in valid if row.get("focal_diagnostics")]
    flip_modes = (
        "frozen", "wrong_confident", "shuffled", "oracle",
        "policy_uniform", "policy_shuffled", "ungated", "anchor",
    )
    total_decisions = sum(int(item.get("action_decisions", 0)) for item in diagnostics)
    flip_opportunities = sum(
        int(item.get("action_flip_opportunities", 0)) for item in diagnostics
    )
    return {
        "episodes": len(rows),
        "valid_episodes": len(valid),
        "errors": len(rows) - len(valid),
        "infrastructure_errors": len(rows) - len(valid),
        "policy_protocol_failures": len(policy_failures),
        "format_success_rate": (
            (len(valid) - len(policy_failures)) / len(rows) if rows else 0.0
        ),
        "all_episode_agreement_rate": sum(float(row.get("agreement", False)) for row in valid) / len(rows)
        if rows else 0.0,
        "all_episode_mean_focal_reward": sum(float(row.get("focal_reward", 0.0)) for row in valid) / len(rows)
        if rows else None,
        "agreement_rate": mean(float(row["agreement"]) for row in valid) if valid else 0.0,
        "mean_focal_reward": mean(row["focal_reward"] for row in valid) if valid else None,
        "mean_opponent_reward": mean(row["opponent_reward"] for row in valid) if valid else None,
        "mean_joint_reward": mean(row["joint_reward"] for row in valid) if valid else None,
        "mean_turns": mean(row["turns"] for row in valid) if valid else None,
        "mean_focal_model_calls_per_episode": mean(row["focal_model_calls"] for row in valid) if valid else None,
        "mean_opponent_model_calls_per_episode": mean(row["opponent_model_calls"] for row in valid) if valid else None,
        "total_focal_invalid_raw": sum(row.get("focal_protocol", {}).get("invalid_raw", 0) for row in rows),
        "total_opponent_invalid_raw": sum(row.get("opponent_protocol", {}).get("invalid_raw", 0) for row in rows),
        "total_focal_deterministic_repairs": sum(row.get("focal_protocol", {}).get("deterministic_repairs", 0) for row in rows),
        "total_opponent_deterministic_repairs": sum(row.get("opponent_protocol", {}).get("deterministic_repairs", 0) for row in rows),
        "total_focal_protocol_retry_calls": sum(row.get("focal_protocol", {}).get("retry_calls", 0) for row in rows),
        "total_opponent_protocol_retry_calls": sum(row.get("opponent_protocol", {}).get("retry_calls", 0) for row in rows),
        "total_focal_protocol_repair_failures": sum(row.get("focal_protocol", {}).get("repair_failures", 0) for row in rows),
        "total_opponent_protocol_repair_failures": sum(row.get("opponent_protocol", {}).get("repair_failures", 0) for row in rows),
        "mean_price_on_agreement": mean(row["price"] for row in valid if row.get("price") is not None)
        if any(row.get("price") is not None for row in valid)
        else None,
        "belief_calibration_count": len(calibration),
        "belief_accept_brier": mean(float(item["brier"]) for item in calibration) if calibration else None,
        "belief_accept_nll": mean(float(item["nll"]) for item in calibration) if calibration else None,
        "belief_accept_ece_5bin": ece,
        "mean_belief_abs_error": mean(float(item["belief_abs_error"]) for item in diagnostics) if diagnostics else None,
        "belief_q10_q90_coverage": mean(float(item["belief_q10_q90_covered"]) for item in diagnostics) if diagnostics else None,
        "action_decisions": total_decisions,
        "action_flip_opportunities": flip_opportunities,
        "action_flip_rate": {
            mode: (
                sum(int(item.get("action_flip_counts", {}).get(mode, 0)) for item in diagnostics)
                / flip_opportunities
                if flip_opportunities else None
            )
            for mode in flip_modes
        },
        "mean_theta_trust": mean(
            float(item["mean_theta_trust"])
            for item in diagnostics if item.get("mean_theta_trust") is not None
        ) if any(item.get("mean_theta_trust") is not None for item in diagnostics) else None,
        "mean_policy_trust": mean(
            float(item["mean_policy_trust"])
            for item in diagnostics if item.get("mean_policy_trust") is not None
        ) if any(item.get("mean_policy_trust") is not None for item in diagnostics) else None,
        "run_metrics": run_metrics,
        "paper_reported_qwen3_improvement_over_direct": {
            "buyer": "+10.04 ± 2.03",
            "seller": "+18.54 ± 2.46",
            "resource_first": "+8.95 ± 6.23",
            "resource_second": "+29.65 ± 0.33",
        },
        "important": "Paper numbers are deltas over its own direct baseline, not absolute rewards.",
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "paper": "Scaling Inference-Time Computation via Opponent Simulation, arXiv:2602.19309v2",
            "paper_prompt_repo_commit": "40b91861899586cb388d6b05c723959e2f5be4c8",
            "protocol": "10 runs x 20 repeated episodes x 10 turns; cost=43; WTP=63; BoN=5",
            "method_semantics": {
                "direct": "paper zero-shot repeated baseline with compact public outcome memory",
                "opponent_simulation": (
                    "compact paper-aligned reimplementation using one joint future-rollout call; "
                    "not an exact reproduction of the unreleased Algorithm-1 runner"
                ),
                "opponent_simulation_paper": (
                    "paper-aligned structured brainstorming plus independent serialized rollout per candidate; "
                    "authors did not release the exact Algorithm-1 runner"
                ),
                "framework": "historical prompt-only dual-timescale framework V1",
                "framework_frozen": "historical V1 with frozen within-episode belief",
                "framework_v2": (
                    "structured action-conditional posterior, reciprocal contingent planner, action lock"
                ),
                "framework_v2_frozen": "V2 planner with first-update-only within-episode belief",
                "framework_v3": (
                    "V2 posterior/candidates plus adverse-tail response forecast, outside-option "
                    "dominance, and agreement-regret safeguards"
                ),
                "framework_v3_frozen": (
                    "V3 uncertainty-safeguarded planner with first-update-only within-episode belief"
                ),
                "framework_v3_1": (
                    "V3 safeguards plus a recent realized-payoff commitment floor and separate "
                    "opening exploit versus safe response-support frontiers"
                ),
                "framework_v3_1_frozen": (
                    "V3.1 reciprocal commitment planner with first-update-only within-episode belief"
                ),
                "framework_v3_2": (
                    "V3.1 with ungated exploit frontier scoped only to resource first-mover openings"
                ),
                "framework_v3_2_frozen": (
                    "V3.2 scoped commitment planner with first-update-only within-episode belief"
                ),
                "framework_v3_3": (
                    "V3.1 safeguards and commitment with proposal breadth adapted to scalar "
                    "versus combinatorial action spaces"
                ),
                "framework_v3_3_frozen": (
                    "V3.3 action-space-adaptive planner with first-update-only within-episode belief"
                ),
                "framework_v4": (
                    "V3.3 plus finite-horizon cross-episode value of information in "
                    "accept-versus-probe planning"
                ),
                "framework_v4_frozen": (
                    "V4 cross-episode information planner with first-update-only within-episode belief"
                ),
                "framework_v4_wrong_confident": (
                    "V4 planner driven by a deliberately wrong point-mass opponent belief"
                ),
                "framework_v4_shuffled": (
                    "V4 planner driven by a deterministic permutation of the learned posterior"
                ),
                "framework_v4_oracle": (
                    "V4 planner driven by the evaluator's true opponent utility parameter"
                ),
                "framework_v5": (
                    "V4 plus source-aware likelihood tempering and repeated-evidence discounting"
                ),
                "framework_v5_frozen": (
                    "V5 endogeneity-calibrated planner with first-update-only within-episode belief"
                ),
                "framework_v6": (
                    "V5 posterior plus decision-relevant, opportunity-cost-bounded value of information"
                ),
                "framework_v6_frozen": (
                    "V6 planner with first-update-only within-episode belief"
                ),
                "framework_v6_wrong_confident": (
                    "V6 planner driven by a deliberately wrong point-mass opponent belief"
                ),
                "framework_v6_shuffled": (
                    "V6 planner driven by a deterministic permutation of the learned posterior"
                ),
                "framework_v6_oracle": (
                    "V6 planner driven by the evaluator's true opponent utility parameter"
                ),
                "framework_v7": (
                    "factorized preference-response-policy belief with action-level posterior regret"
                ),
                "framework_v7_frozen": (
                    "V7 planner with first-update-only joint belief"
                ),
                "framework_v7_wrong_confident": (
                    "V7 with deliberately wrong point-mass preference and learned response policy"
                ),
                "framework_v7_shuffled": (
                    "V7 with shuffled preference marginal and learned response policy"
                ),
                "framework_v7_oracle": (
                    "V7 with evaluator preference truth and learned response policy"
                ),
                "framework_v7_policy_uniform": (
                    "V7 with learned preference and uniform response-policy belief"
                ),
                "framework_v7_policy_shuffled": (
                    "V7 with learned preference and deterministically shuffled response-policy belief"
                ),
                "framework_v8": (
                    "V7 joint belief mediated by evidence-channel reliability and an episode anchor"
                ),
                "framework_v8_frozen": (
                    "V8 planner with first-update-only learned belief before reliability mediation"
                ),
                "framework_v8_wrong_confident": (
                    "V8 safety gate applied to a deliberately wrong point-mass preference"
                ),
                "framework_v8_shuffled": (
                    "V8 safety gate applied to a shuffled preference marginal"
                ),
                "framework_v8_oracle": (
                    "V8 diagnostic with evaluator preference truth bypassing only the theta gate"
                ),
                "framework_v8_policy_uniform": (
                    "V8 reliability gate applied to a uniform response-policy intervention"
                ),
                "framework_v8_policy_shuffled": (
                    "V8 reliability gate applied to a shuffled response-policy intervention"
                ),
                "framework_v8_no_cross_episode": (
                    "V8 with identical within-episode inference/planner but a uniform reset "
                    "that prevents all cross-episode belief carryover"
                ),
            }[args.method],
            "resource_endowments": [{"X": 25, "Y": 5}, {"X": 5, "Y": 25}],
            "resource_values": [{"X": 0.5, "Y": 2.5}, {"X": 2.5, "Y": 0.5}],
            "resolved_actor_model": args.model,
            "resolved_actor_base_url": args.base_url,
            "resolved_opponent_model": args.opponent_model or args.model,
            "resolved_opponent_base_url": args.opponent_base_url or args.base_url,
            "paper_exact_external_opponent": (
                (args.opponent_model or args.model).lower() == "gemini-2.5-flash"
            ),
        }
    )
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    results_path = out / "episodes.jsonl"
    existing = []
    if args.resume and results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {(row["run"], row["episode"]): row for row in existing}
    restart_runs = runs_requiring_restart(latest, args.restart_error_runs)
    with results_path.open("a", encoding="utf-8") as stream:
        for run_index in range(1, args.runs + 1):
            run_trace = out / "model_traces" / f"run_{run_index:02d}"
            focal = focal_index(args.setting)
            agents = [None, None]
            agents[focal] = make_agent(
                args, name=AGENT_ONE if focal == 0 else AGENT_TWO, focal=True,
                seed=args.seed + run_index * 1009 + focal, trace_dir=run_trace,
            )
            agents[1 - focal] = make_agent(
                args, name=AGENT_ONE if 1 - focal == 0 else AGENT_TWO, focal=False,
                seed=args.seed + run_index * 1009 + 1 - focal, trace_dir=run_trace,
            )
            for episode in range(1, args.episodes_per_run + 1):
                key = (run_index, episode)
                if (
                    args.resume
                    and run_index not in restart_runs
                    and key in latest
                    and not latest[key].get("error")
                ):
                    # Reconstruct public repeated memory before continuing. Full private
                    # call state is intentionally not inferred from a summary.
                    row = latest[key]
                    for idx, agent in enumerate(agents):
                        own = row["focal_reward"] if idx == focal else row["opponent_reward"]
                        agent.episode_records.append({
                            "episode": episode, "agreement": row["agreement"],
                            "deal": row.get("deal"), "own_reward": own, "public_transcript": [],
                        })
                    continue
                if args.opponent_switch_episode > 0 and episode == args.opponent_switch_episode:
                    agents[1 - focal] = make_agent(
                        args,
                        name=AGENT_ONE if 1 - focal == 0 else AGENT_TWO,
                        focal=False,
                        seed=args.seed + run_index * 1009 + 1 - focal + 1000003,
                        trace_dir=run_trace / "switched_opponent",
                    )
                prepare_agents(args, agents, episode)
                started = time.time()
                calls_before = [agent.call_index for agent in agents]
                protocol_before = [protocol_counts(agent) for agent in agents]
                try:
                    game = make_game(args, agents, out, run_index, episode)
                    game.run()
                    row = episode_result(
                        args, run_index, episode, game, agents,
                        time.time() - started, calls_before, protocol_before,
                    )
                except ProtocolFormatError as exc:
                    row = policy_protocol_failure_result(
                        args, run_index, episode, game, agents,
                        time.time() - started, calls_before, protocol_before, exc,
                    )
                except Exception as exc:
                    protocol_stats = protocol_deltas(agents, protocol_before)
                    row = {
                        "method": args.method, "setting": args.setting, "run": run_index,
                        "episode": episode, "agreement": False,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "focal_protocol": protocol_stats[focal],
                        "opponent_protocol": protocol_stats[1 - focal],
                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                    }
                latest[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                current = summarize(list(latest.values()))
                (out / "summary.json").write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"episode_result": row}, ensure_ascii=False), flush=True)
                if args.fail_fast_transport and is_transport_failure(row):
                    raise RuntimeError(
                        "model service transport failure; cell stopped after persisting the episode"
                    )
    final = summarize(list(latest.values()))
    print(json.dumps({"summary": final, "output_dir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
