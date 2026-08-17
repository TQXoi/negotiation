#!/usr/bin/env python3
"""Low-cost CEM search for continuous-rule EV planner parameters.

This script treats the LLM buyer/seller as black-box environment components and
optimizes only a small interpretable planner parameter vector. It is intended as
the cheapest RL step before LoRA/SFT/GRPO.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.model_clients import make_model_client  # noqa: E402
from Simple_Env.buyer.factory import make_buyer  # noqa: E402
from Simple_Env.buyer.continuous_solver.planner.params import PlannerParams  # noqa: E402
from Simple_Env.environment import run_episode  # noqa: E402
from Simple_Env.environment.data import load_or_generate_scenarios  # noqa: E402
from Simple_Env.seller import make_seller  # noqa: E402


PARAM_NAMES = list(PlannerParams.__dataclass_fields__.keys())  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buyer-model", required=True)
    parser.add_argument("--seller-model", required=True)
    parser.add_argument("--seller-type", default="default")
    parser.add_argument("--seller-persona", choices=["neutral", "begging", "insulting", "unyielding"], default="neutral")
    parser.add_argument(
        "--scenarios-jsonl",
        default=str(ROOT / "Simple_Env" / "research_splits" / "universal_v1" / "development_000_023.jsonl"),
    )
    parser.add_argument("--require-scenarios-jsonl", action="store_true")
    parser.add_argument("--num-train-instances", type=int, default=8)
    parser.add_argument("--rollouts-per-instance", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--buyer-temperature", type=float, default=1.0)
    parser.add_argument("--seller-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--candidate-offer-k", type=int, default=5)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument(
        "--init-params-json",
        default=None,
        help="Warm-start CEM mean from an existing best_planner_params.json or params JSON.",
    )
    parser.add_argument(
        "--init-std-scale",
        type=float,
        default=1.0,
        help="Multiply the default initial std by this factor when warm-starting/refining.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--openai-request-timeout", type=float, default=1800.0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--positive-gains-probability", type=float, default=0.7)
    parser.add_argument(
        "--print-episode-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print one JSON line after every rollout inside each CEM candidate evaluation.",
    )
    parser.add_argument(
        "--save-episode-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode metrics to episode_metrics.jsonl even when not printing them.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=32,
        help="Number of recent episodes used for rolling training-progress metrics.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "simple" / "cem_planner_search"))
    return parser.parse_args()


def initial_mean_std() -> Tuple[Dict[str, float], Dict[str, float]]:
    base = PlannerParams().to_dict()
    std = {
        "first_anchor_ratio": 0.08,
        "early_concession_rate": 0.025,
        "late_concession_rate": 0.035,
        "p50_low_mult": 0.055,
        "p50_mid_mult": 0.040,
        "p50_high_mult": 0.050,
        "mean_mult": 0.045,
        "seller_discount_early": 0.075,
        "seller_discount_late": 0.060,
        "min_increment_ratio": 0.012,
        "accept_reward_threshold": 0.055,
        "accept_seller_ratio_threshold": 0.055,
        "lowball_penalty_weight": 0.080,
        "quit_penalty": 0.080,
        "future_value_weight": 0.100,
    }
    return base, std


def initialize_distribution(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean, std = initial_mean_std()
    if args.init_params_json:
        params = PlannerParams.from_json(args.init_params_json)
        mean = params.to_dict()
    scale = max(0.01, float(args.init_std_scale))
    std = {name: value * scale for name, value in std.items()}
    return mean, std


def sample_params(mean: Dict[str, float], std: Dict[str, float], rng: random.Random) -> PlannerParams:
    return PlannerParams.from_dict({name: rng.gauss(mean[name], max(std[name], 1e-6)) for name in PARAM_NAMES})


def update_distribution(elites: List[PlannerParams], old_std: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    rows = [p.to_dict() for p in elites]
    mean = {name: sum(row[name] for row in rows) / len(rows) for name in PARAM_NAMES}
    std = {}
    for name in PARAM_NAMES:
        var = sum((row[name] - mean[name]) ** 2 for row in rows) / max(1, len(rows))
        # Keep a small exploration floor and smooth std so the search does not
        # collapse after one noisy elite set.
        std[name] = max(0.20 * old_std[name], min(old_std[name], var**0.5 + 1e-6))
    return mean, std


def shaped_score(episode: Any) -> float:
    score = float(episode.reward or 0.0)
    if episode.status == "deal":
        score += 0.05
    if episode.terminal_reason == "seller_quit":
        score -= 0.10
    if episode.terminal_reason == "buyer_quit":
        score -= 0.06
    if episode.buyer_overshoot:
        score -= 0.50
    if episode.buyer_format_violation:
        score -= 0.30
    return score


def summarize_episodes(episodes: Iterable[Any], errors: int) -> Dict[str, float]:
    rows = list(episodes)
    n = len(rows)
    if not rows:
        return {"n": 0, "errors": errors, "score": -1.0}
    deals = [ep for ep in rows if ep.status == "deal"]
    bargains = [float(ep.buyer_bargained_ratio) for ep in deals if ep.buyer_bargained_ratio is not None]
    return {
        "n": n,
        "errors": errors,
        "score": sum(shaped_score(ep) for ep in rows) / n - 0.03 * errors,
        "avg_reward": sum(float(ep.reward or 0.0) for ep in rows) / n,
        "deal_rate": len(deals) / n,
        "buyer_bargained_ratio": sum(bargains) / len(bargains) if bargains else 0.0,
        "seller_quit_rate": sum(ep.terminal_reason == "seller_quit" for ep in rows) / n,
        "overshoot_rate": sum(bool(ep.buyer_overshoot) for ep in rows) / n,
    }


def rolling_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    items = list(rows)
    if not items:
        return {
            "rolling_n": 0,
            "rolling_avg_reward": 0.0,
            "rolling_avg_shaped_score": 0.0,
            "rolling_deal_rate": 0.0,
            "rolling_bargain_ratio": 0.0,
            "rolling_seller_quit_rate": 0.0,
        }
    bargains = [float(row["buyer_bargained_ratio"]) for row in items if row.get("buyer_bargained_ratio") is not None]
    return {
        "rolling_n": len(items),
        "rolling_avg_reward": sum(float(row.get("reward") or 0.0) for row in items) / len(items),
        "rolling_avg_shaped_score": sum(float(row.get("shaped_score") or 0.0) for row in items) / len(items),
        "rolling_deal_rate": sum(row.get("status") == "deal" for row in items) / len(items),
        "rolling_bargain_ratio": sum(bargains) / len(bargains) if bargains else 0.0,
        "rolling_seller_quit_rate": sum(row.get("terminal_reason") == "seller_quit" for row in items) / len(items),
    }


def evaluate_params(
    *,
    args: argparse.Namespace,
    params: PlannerParams,
    param_path: Path,
    scenarios: List[Any],
    buyer_client: Any,
    seller_client: Any,
    rolling_history: deque | None = None,
    iteration: int | None = None,
    sample_id: int | None = None,
    episode_log_path: Path | None = None,
) -> Dict[str, Any]:
    episodes = []
    errors = 0
    variant = "continuous_rule_ev_cem"
    for scenario in scenarios:
        for rollout in range(args.rollouts_per_instance):
            for attempt in range(max(1, args.episode_retries + 1)):
                try:
                    buyer = make_buyer(
                        variant,
                        buyer_client,
                        args.max_tokens,
                        args.buyer_temperature,
                        args.top_p,
                        candidate_offer_k=args.candidate_offer_k,
                        continuous_planner_params_json=str(param_path),
                    )
                    seller = make_seller(
                        args.seller_type,
                        seller_client,
                        args.max_tokens,
                        args.seller_temperature,
                        args.top_p,
                        args.seller_persona,
                    )
                    episode = run_episode(
                        variant=variant,
                        scenario=scenario,
                        rollout=rollout,
                        buyer=buyer,
                        seller=seller,
                        max_turns=args.max_turns,
                    )
                    episodes.append(episode)
                    episode_row = {
                        "iteration": iteration,
                        "sample_id": sample_id,
                        "scenario": scenario.item_id,
                        "rollout": rollout,
                        "status": episode.status,
                        "terminal_reason": episode.terminal_reason,
                        "reward": float(episode.reward or 0.0),
                        "shaped_score": round(shaped_score(episode), 6),
                        "final_price": episode.final_price,
                        "buyer_budget": scenario.buyer_budget,
                        "seller_cost": scenario.seller_cost,
                        "buyer_bargained_ratio": episode.buyer_bargained_ratio,
                        "rounds": episode.rounds,
                    }
                    if rolling_history is not None:
                        rolling_history.append(episode_row)
                    recent = rolling_metrics(rolling_history or [])
                    episode_row.update(
                        {
                            "rolling_window": int(args.rolling_window),
                            **{
                                key: round(float(value), 6)
                                for key, value in recent.items()
                                if key != "rolling_n"
                            },
                            "rolling_n": int(recent["rolling_n"]),
                        }
                    )
                    if args.save_episode_metrics and episode_log_path is not None:
                        append_jsonl(episode_log_path, episode_row)
                    if args.print_episode_results:
                        running = summarize_episodes(episodes, errors)
                        print(
                            json.dumps(
                                {
                                    "cem_episode": {
                                        "iteration": iteration,
                                        "sample_id": sample_id,
                                        "scenario": episode_row["scenario"],
                                        "rollout": episode_row["rollout"],
                                        "attempt": attempt + 1,
                                        "status": episode_row["status"],
                                        "terminal_reason": episode_row["terminal_reason"],
                                        "reward": episode_row["reward"],
                                        "shaped_score": episode_row["shaped_score"],
                                        "final_price": episode_row["final_price"],
                                        "buyer_budget": episode_row["buyer_budget"],
                                        "seller_cost": episode_row["seller_cost"],
                                        "buyer_bargained_ratio": episode_row["buyer_bargained_ratio"],
                                        "rounds": episode_row["rounds"],
                                        "running_score": round(float(running.get("score", 0.0)), 6),
                                        "running_avg_reward": round(float(running.get("avg_reward", 0.0)), 6),
                                        "running_deal_rate": round(float(running.get("deal_rate", 0.0)), 6),
                                        "rolling_window": episode_row["rolling_window"],
                                        "rolling_avg_reward": episode_row["rolling_avg_reward"],
                                        "rolling_avg_shaped_score": episode_row["rolling_avg_shaped_score"],
                                        "rolling_deal_rate": episode_row["rolling_deal_rate"],
                                        "rolling_bargain_ratio": episode_row["rolling_bargain_ratio"],
                                        "rolling_seller_quit_rate": episode_row["rolling_seller_quit_rate"],
                                        "rolling_n": episode_row["rolling_n"],
                                    }
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    break
                except Exception:
                    errors += 1
                    if attempt >= args.episode_retries:
                        break
                    time.sleep(max(0.0, args.retry_backoff_seconds) * (2**attempt))
    metrics = summarize_episodes(episodes, errors)
    return {"params": params.to_dict(), "metrics": metrics}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    os.environ["OPENAI_REQUEST_TIMEOUT"] = str(args.openai_request_timeout)
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))

    args.num_test_instances = args.num_train_instances
    scenarios = load_or_generate_scenarios(args)[: args.num_train_instances]
    buyer_client = make_model_client(
        args.buyer_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        enable_thinking=args.enable_thinking,
    )
    seller_client = buyer_client if args.seller_model == args.buyer_model else make_model_client(
        args.seller_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        enable_thinking=args.enable_thinking,
    )

    mean, std = initialize_distribution(args)
    best: Dict[str, Any] | None = None
    elite_count = max(1, int(round(args.population_size * args.elite_frac)))
    rolling_history = deque(maxlen=max(1, int(args.rolling_window)))

    for iteration in range(args.cem_iters):
        candidates = []
        print(json.dumps({"cem_iteration": iteration, "mean": mean, "std": std}, ensure_ascii=False), flush=True)
        for sample_id in range(args.population_size):
            params = sample_params(mean, std, rng)
            param_path = out / "candidate_params" / f"iter_{iteration:03d}_sample_{sample_id:03d}.json"
            write_json(param_path, {"params": params.to_dict(), "iteration": iteration, "sample_id": sample_id})
            result = evaluate_params(
                args=args,
                params=params,
                param_path=param_path,
                scenarios=scenarios,
                buyer_client=buyer_client,
                seller_client=seller_client,
                rolling_history=rolling_history,
                iteration=iteration,
                sample_id=sample_id,
                episode_log_path=out / "episode_metrics.jsonl",
            )
            result.update({"iteration": iteration, "sample_id": sample_id, "param_path": str(param_path)})
            candidates.append(result)
            append_jsonl(out / "candidate_metrics.jsonl", result)
            print(
                json.dumps(
                    {
                        "cem_candidate": {
                            "iteration": iteration,
                            "sample_id": sample_id,
                            "score": round(float(result["metrics"].get("score", 0.0)), 6),
                            "avg_reward": round(float(result["metrics"].get("avg_reward", 0.0)), 6),
                            "deal_rate": round(float(result["metrics"].get("deal_rate", 0.0)), 6),
                            "buyer_bargained_ratio": round(
                                float(result["metrics"].get("buyer_bargained_ratio", 0.0)), 6
                            ),
                            "seller_quit_rate": round(float(result["metrics"].get("seller_quit_rate", 0.0)), 6),
                            "errors": int(result["metrics"].get("errors", 0)),
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if best is None or result["metrics"]["score"] > best["metrics"]["score"]:
                best = result
                write_json(out / "best_planner_params.json", {"best_params": best["params"], "metrics": best["metrics"]})

        candidates.sort(key=lambda row: row["metrics"]["score"], reverse=True)
        elites = [PlannerParams.from_dict(row["params"]) for row in candidates[:elite_count]]
        mean, std = update_distribution(elites, std)
        write_json(
            out / f"iteration_{iteration:03d}_summary.json",
            {
                "iteration": iteration,
                "elite_count": elite_count,
                "best_this_iter": candidates[0],
                "global_best": best,
                "next_mean": mean,
                "next_std": std,
            },
        )

    assert best is not None
    write_json(out / "best_planner_params.json", {"best_params": best["params"], "metrics": best["metrics"]})
    print(json.dumps({"best": best, "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
