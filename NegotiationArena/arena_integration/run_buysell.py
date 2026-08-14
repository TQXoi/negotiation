from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from arena_integration.framework_agent import BeliefPlannerAgent
from games.buy_sell_game.game import BuySellGame
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation


class PaperFaithfulBuySellGame(BuySellGame):
    """Main-branch game with the paper branch's ACCEPT/REJECT termination."""

    def game_over(self):
        state = self.game_state[-1]
        if not state:
            return False
        response = state["player_public_info_dict"].get("player answer", "NONE")
        return response in {"ACCEPT", "REJECT"} or state.get("current_iteration", 0) == self.iterations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "framework_seller", "framework_buyer"], required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seller-model")
    parser.add_argument("--buyer-model")
    parser.add_argument("--seller-base-url")
    parser.add_argument("--buyer-base-url")
    parser.add_argument("--seller-api-key-env")
    parser.add_argument("--buyer-api-key-env")
    parser.add_argument("--seller-cost", type=int, default=40)
    parser.add_argument("--buyer-wtp", type=int, default=60)
    parser.add_argument("--money-cap", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def endpoint(args, role):
    return {
        "model": getattr(args, f"{role}_model") or args.model,
        "base_url": getattr(args, f"{role}_base_url") or args.base_url,
        "api_key_env": getattr(args, f"{role}_api_key_env") or args.api_key_env,
    }


def direct_agent(args, name, role, seed):
    spec = endpoint(args, role)
    if not spec["model"]:
        raise ValueError(f"No model configured for {role}")
    return ChatGPTAgent(
        agent_name=name,
        model=spec["model"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=seed,
        base_url=spec["base_url"],
        api_key=os.environ.get(spec["api_key_env"]),
    )


def framework_agent(args, name, role, value, seed, trace_path):
    spec = endpoint(args, role)
    return BeliefPlannerAgent(
        agent_name=name,
        focal_role=role,
        private_value=value,
        money_cap=args.money_cap,
        model=spec["model"],
        base_url=spec["base_url"],
        api_key=os.environ.get(spec["api_key_env"]),
        temperature=0.2,
        seed=seed,
        trace_path=str(trace_path),
    )


def make_players(args, rollout, out):
    seed = args.seed + rollout * 17
    seller = direct_agent(args, AGENT_ONE, "seller", seed)
    buyer = direct_agent(args, AGENT_TWO, "buyer", seed + 1)
    if args.mode == "framework_seller":
        seller = framework_agent(args, AGENT_ONE, "seller", args.seller_cost, seed, out / "traces" / f"{rollout:04d}_seller.json")
    elif args.mode == "framework_buyer":
        buyer = framework_agent(args, AGENT_TWO, "buyer", args.buyer_wtp, seed + 1, out / "traces" / f"{rollout:04d}_buyer.json")
    return [seller, buyer]


def result_from_game(args, rollout, game, elapsed):
    summary = game.game_state[-1].get("summary", {})
    outcomes = summary.get("player_outcome", [0, 0])
    response = summary.get("final_response", "ERROR")
    trade = summary.get("proposed_trade")
    price = None
    if trade is not None:
        try:
            price = trade.resources_from_second_agent.resource_dict[MONEY_TOKEN]
        except Exception:
            pass
    agreement = response == "ACCEPT"
    return {
        "mode": args.mode,
        "rollout": rollout,
        "agreement": agreement,
        "final_response": response,
        "price": price if agreement else None,
        "seller_payoff": outcomes[0] if outcomes else 0,
        "buyer_payoff": outcomes[1] if outcomes else 0,
        "total_payoff": sum(outcomes) if outcomes else 0,
        "elapsed_seconds": round(elapsed, 3),
        "error": None,
    }


def summarize(rows):
    valid = [row for row in rows if not row["error"]]
    agreed = [row for row in valid if row["agreement"]]
    n = len(rows) or 1
    return {
        "episodes": len(rows),
        "valid_episodes": len(valid),
        "errors": len(rows) - len(valid),
        "agreement_rate": len(agreed) / n,
        "mean_price_on_agreement": sum(row["price"] for row in agreed) / len(agreed) if agreed else None,
        "mean_seller_payoff": sum(row["seller_payoff"] for row in valid) / len(valid) if valid else None,
        "mean_buyer_payoff": sum(row["buyer_payoff"] for row in valid) / len(valid) if valid else None,
        "mean_total_payoff": sum(row["total_payoff"] for row in valid) / len(valid) if valid else None,
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["paper_code_commit"] = "d35a7a3aa0d94c2d49f1d6ac13c5f931851abf12"
    config["protocol"] = "paper_buy_sell_cost40_wtp60_10turn_accept_reject"
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results_path = out / "episodes.jsonl"
    existing = []
    if args.resume and results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    # JSONL is an immutable attempt ledger. On resume, retry failed rollouts
    # but summarize only the latest attempt for each rollout.
    latest = {row["rollout"]: row for row in existing}
    done = {rollout for rollout, row in latest.items() if not row.get("error")}
    rows = list(latest.values())
    with results_path.open("a", encoding="utf-8") as stream:
        for rollout in range(args.episodes):
            if rollout in done:
                continue
            start = time.time()
            try:
                players = make_players(args, rollout, out)
                game = PaperFaithfulBuySellGame(
                    players=players,
                    iterations=10,
                    player_goals=[
                        SellerGoal(Valuation({"X": args.seller_cost})),
                        BuyerGoal(Valuation({"X": args.buyer_wtp})),
                    ],
                    player_starting_resources=[Resources({"X": 1}), Resources({MONEY_TOKEN: args.money_cap})],
                    player_conversation_roles=[f"You are {AGENT_ONE}.", f"You are {AGENT_TWO}."],
                    player_social_behaviour=["", ""],
                    log_dir=str(out / "arena_logs"),
                )
                game.run()
                row = result_from_game(args, rollout, game, time.time() - start)
            except Exception as exc:
                row = {
                    "mode": args.mode,
                    "rollout": rollout,
                    "agreement": False,
                    "elapsed_seconds": round(time.time() - start, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            latest[rollout] = row
            rows = list(latest.values())
            print(json.dumps({"episode_result": row}, ensure_ascii=False), flush=True)
            (out / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    print(json.dumps({"summary": summarize(rows), "output_dir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
