#!/usr/bin/env python3
"""CEM v0.2: train/dev contextual planner search for Simple Env.

This keeps RL cheap by learning only a small contextual planner policy:

    context features -> parameter offsets -> EV planner action

The LLM buyer/seller are treated as black-box environment components.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.model_clients import make_model_client  # noqa: E402
from Simple_Env.buyer.factory import make_buyer  # noqa: E402
from Simple_Env.buyer.continuous_solver.planner.contextual_params import (  # noqa: E402
    ADAPTIVE_PARAM_NAMES,
    FEATURE_NAMES,
    ContextualPlannerPolicy,
)
from Simple_Env.buyer.continuous_solver.planner.params import PlannerParams  # noqa: E402
from Simple_Env.environment import run_episode  # noqa: E402
from Simple_Env.environment.data import load_or_generate_scenarios  # noqa: E402
from Simple_Env.seller import make_seller  # noqa: E402


BASE_PARAM_NAMES = list(PlannerParams.__dataclass_fields__.keys())  # type: ignore[attr-defined]
WEIGHT_NAMES = [f"w::{param}::{feat}" for param in ADAPTIVE_PARAM_NAMES for feat in FEATURE_NAMES]
VECTOR_NAMES = BASE_PARAM_NAMES + WEIGHT_NAMES


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
    parser.add_argument("--num-train-instances", type=int, default=64)
    parser.add_argument("--num-dev-instances", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=20260728)
    parser.add_argument("--rollouts-per-instance", type=int, default=1)
    parser.add_argument("--dev-rollouts-per-instance", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--buyer-temperature", type=float, default=1.0)
    parser.add_argument("--seller-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--candidate-offer-k", type=int, default=5)
    parser.add_argument("--cem-iters", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--dev-eval-top-k", type=int, default=3)
    parser.add_argument("--init-params-json", default=None)
    parser.add_argument("--init-std-scale", type=float, default=1.0)
    parser.add_argument("--weight-std", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--openai-request-timeout", type=float, default=1800.0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--positive-gains-probability", type=float, default=0.7)
    parser.add_argument("--rolling-window", type=int, default=64)
    parser.add_argument("--save-episode-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed candidate and episode rows in the output directory.")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "simple" / "cem_planner_v02_contextual"))
    return parser.parse_args()


def initial_mean_std(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float]]:
    base = PlannerParams().to_dict()
    if args.init_params_json:
        try:
            policy = ContextualPlannerPolicy.from_json(args.init_params_json)
            base = policy.base_params.to_dict()
            initial_weights = flatten_policy(policy)
        except Exception:
            base = PlannerParams.from_json(args.init_params_json).to_dict()
            initial_weights = {}
    else:
        initial_weights = {}
    param_std = {
        "first_anchor_ratio": 0.07,
        "early_concession_rate": 0.020,
        "late_concession_rate": 0.035,
        "p50_low_mult": 0.050,
        "p50_mid_mult": 0.035,
        "p50_high_mult": 0.045,
        "mean_mult": 0.040,
        "seller_discount_early": 0.065,
        "seller_discount_late": 0.055,
        "min_increment_ratio": 0.010,
        "accept_reward_threshold": 0.050,
        "accept_seller_ratio_threshold": 0.045,
        "lowball_penalty_weight": 0.075,
        "quit_penalty": 0.075,
        "future_value_weight": 0.090,
    }
    scale = max(0.01, float(args.init_std_scale))
    mean = {**base}
    std = {key: param_std[key] * scale for key in BASE_PARAM_NAMES}
    for name in WEIGHT_NAMES:
        mean[name] = float(initial_weights.get(name, 0.0))
        std[name] = float(args.weight_std) * scale
    return mean, std


def flatten_policy(policy: ContextualPlannerPolicy) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for param, weights in policy.feature_weights.items():
        for feat, value in weights.items():
            flat[f"w::{param}::{feat}"] = float(value)
    return flat


def vector_to_policy(vector: Dict[str, float]) -> ContextualPlannerPolicy:
    base = PlannerParams.from_dict({key: vector[key] for key in BASE_PARAM_NAMES})
    weights: Dict[str, Dict[str, float]] = {}
    for name in WEIGHT_NAMES:
        _, param, feat = name.split("::", 2)
        weights.setdefault(param, {})[feat] = float(max(-0.45, min(0.45, vector[name])))
    return ContextualPlannerPolicy(base_params=base, feature_weights=weights)


def policy_to_vector(policy: ContextualPlannerPolicy) -> Dict[str, float]:
    vector = policy.base_params.to_dict()
    flat_weights = flatten_policy(policy)
    for name in WEIGHT_NAMES:
        vector[name] = float(flat_weights.get(name, 0.0))
    return vector


def load_policy_payload(path: Path) -> Tuple[ContextualPlannerPolicy, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = ContextualPlannerPolicy.from_dict(payload.get("contextual_policy", payload))
    return policy, policy_to_vector(policy)


def sample_vector(mean: Dict[str, float], std: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    return {name: rng.gauss(mean[name], max(std[name], 1e-6)) for name in VECTOR_NAMES}


def update_distribution(elite_vectors: List[Dict[str, float]], old_std: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean = {name: statistics.mean(row[name] for row in elite_vectors) for name in VECTOR_NAMES}
    std = {}
    for name in VECTOR_NAMES:
        values = [row[name] for row in elite_vectors]
        var = statistics.mean((value - mean[name]) ** 2 for value in values) if len(values) > 1 else 0.0
        floor = 0.15 * old_std[name]
        std[name] = max(floor, min(old_std[name], var**0.5 + 1e-6))
    return mean, std


def split_scenarios(args: argparse.Namespace) -> Tuple[List[Any], List[Any], List[Any]]:
    args.num_test_instances = max(args.num_train_instances + args.num_dev_instances, args.num_train_instances)
    scenarios = load_or_generate_scenarios(args)
    rng = random.Random(args.split_seed)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)
    train = shuffled[: args.num_train_instances]
    dev = shuffled[args.num_train_instances : args.num_train_instances + args.num_dev_instances]
    return scenarios, train, dev


def run_policy_eval(
    *,
    args: argparse.Namespace,
    policy_path: Path,
    split_name: str,
    scenarios: List[Any],
    rollouts_per_instance: int,
    buyer_client: Any,
    seller_client: Any,
    iteration: int,
    sample_id: int,
    rolling_history: deque,
    episode_log_path: Path | None,
) -> Dict[str, Any]:
    rows = []
    errors = 0
    variant = "continuous_rule_ev_contextual_cem"
    completed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if args.resume and episode_log_path is not None:
        completed = load_completed_episode_rows(
            episode_log_path,
            split_name=split_name,
            iteration=iteration,
            sample_id=sample_id,
        )
    for scenario in scenarios:
        for rollout in range(rollouts_per_instance):
            key = (scenario_id(scenario), rollout)
            if key in completed:
                rows.append(completed[key])
                continue
            for attempt in range(max(1, args.episode_retries + 1)):
                try:
                    buyer = make_buyer(
                        variant,
                        buyer_client,
                        args.max_tokens,
                        args.buyer_temperature,
                        args.top_p,
                        candidate_offer_k=args.candidate_offer_k,
                        continuous_planner_params_json=str(policy_path),
                    )
                    seller = make_seller(
                        args.seller_type,
                        seller_client,
                        args.max_tokens,
                        args.seller_temperature,
                        args.top_p,
                        args.seller_persona,
                    )
                    ep = run_episode(
                        variant=variant,
                        scenario=scenario,
                        rollout=rollout,
                        buyer=buyer,
                        seller=seller,
                        max_turns=args.max_turns,
                    )
                    row = episode_row(ep, split_name=split_name, iteration=iteration, sample_id=sample_id)
                    rows.append(row)
                    rolling_history.append(row)
                    row.update(rolling_metrics(rolling_history))
                    if args.save_episode_metrics and episode_log_path is not None:
                        append_jsonl(episode_log_path, row)
                    break
                except Exception as exc:
                    errors += 1
                    if attempt >= args.episode_retries:
                        if episode_log_path is not None:
                            append_jsonl(
                                episode_log_path,
                                {
                                    "split": split_name,
                                    "iteration": iteration,
                                    "sample_id": sample_id,
                                    "scenario": scenario_id(scenario),
                                    "rollout": rollout,
                                    "error": exc.__class__.__name__,
                                    "message": str(exc),
                                },
                            )
                        break
                    time.sleep(max(0.0, args.retry_backoff_seconds) * (2**attempt))
    return summarize_episode_rows(rows, errors)


def episode_row(ep: Any, *, split_name: str, iteration: int, sample_id: int) -> Dict[str, Any]:
    first_offer = first_buyer_offer(ep)
    repeated_far_below = repeated_far_below_cost(ep)
    accept_too_early = int(ep.terminal_reason == "buyer_accept" and ep.rounds <= 2 and float(ep.reward or 0.0) < 0.70)
    return {
        "split": split_name,
        "iteration": iteration,
        "sample_id": sample_id,
        "scenario": scenario_id(ep.scenario),
        "rollout": ep.rollout,
        "status": ep.status,
        "terminal_reason": ep.terminal_reason,
        "reward": float(ep.reward or 0.0),
        "shaped_score": shaped_score(ep),
        "final_price": ep.final_price,
        "buyer_budget": scenario_value(ep.scenario, "buyer_budget"),
        "seller_cost": scenario_value(ep.scenario, "seller_cost"),
        "buyer_bargained_ratio": ep.buyer_bargained_ratio,
        "rounds": ep.rounds,
        "first_offer": first_offer,
        "first_offer_to_cost": first_offer / scenario_value(ep.scenario, "seller_cost") if first_offer and scenario_value(ep.scenario, "seller_cost") else None,
        "first_far_below_cost": int(first_offer is not None and first_offer < 0.75 * scenario_value(ep.scenario, "seller_cost")),
        "repeated_far_below_cost": repeated_far_below,
        "accept_too_early": accept_too_early,
    }


def shaped_score(ep: Any) -> float:
    reward = float(ep.reward or 0.0)
    bargain = float(ep.buyer_bargained_ratio or 0.0) if ep.status == "deal" else 0.0
    score = reward + 0.08 * int(ep.status == "deal") + 0.05 * bargain
    score -= 0.22 * int(ep.terminal_reason == "seller_quit" and ep.mi)
    score -= 0.10 * int(ep.terminal_reason == "buyer_quit" and ep.mi)
    score -= 0.16 * int(ep.terminal_reason == "buyer_accept" and ep.rounds <= 2 and reward < 0.70)
    score -= 0.10 * repeated_far_below_cost(ep)
    score -= 1.0 * int(bool(ep.buyer_overshoot))
    score -= 0.4 * int(bool(ep.buyer_format_violation))
    return round(score, 6)


def first_buyer_offer(ep: Any) -> float | None:
    for turn in ep.transcript:
        if turn.get("role") == "buyer":
            action = turn.get("action") or {}
            if action.get("price") is not None:
                return float(action["price"])
    return None


def scenario_value(scenario: Any, key: str, default: Any = None) -> Any:
    if isinstance(scenario, dict):
        return scenario.get(key, default)
    return getattr(scenario, key, default)


def scenario_id(scenario: Any) -> str:
    return str(scenario_value(scenario, "item_id", scenario_value(scenario, "scenario_id", "unknown")))


def repeated_far_below_cost(ep: Any) -> int:
    count = 0
    threshold = 0.75 * float(scenario_value(ep.scenario, "seller_cost", 0.0))
    for turn in ep.transcript:
        if turn.get("role") == "buyer":
            price = (turn.get("action") or {}).get("price")
            if price is not None and float(price) < threshold:
                count += 1
    return int(count >= 2)


def summarize_episodes(episodes: Iterable[Any], errors: int) -> Dict[str, Any]:
    rows = list(episodes)
    if not rows:
        return {"n": 0, "errors": errors, "score": -999.0}
    deals = [ep for ep in rows if ep.status == "deal"]
    bargains = [float(ep.buyer_bargained_ratio) for ep in deals if ep.buyer_bargained_ratio is not None]
    rewards = [float(ep.reward or 0.0) for ep in rows]
    shaped = [shaped_score(ep) for ep in rows]
    return {
        "n": len(rows),
        "errors": errors,
        "score": round(statistics.mean(shaped) - 0.03 * errors, 6),
        "avg_reward": round(statistics.mean(rewards), 6),
        "deal_rate": round(len(deals) / len(rows), 6),
        "buyer_bargained_ratio": round(statistics.mean(bargains), 6) if bargains else 0.0,
        "seller_quit_rate": round(sum(ep.terminal_reason == "seller_quit" for ep in rows) / len(rows), 6),
        "accept_too_early_rate": round(
            sum(ep.terminal_reason == "buyer_accept" and ep.rounds <= 2 and float(ep.reward or 0.0) < 0.70 for ep in rows)
            / len(rows),
            6,
        ),
        "repeated_far_below_cost_rate": round(sum(repeated_far_below_cost(ep) for ep in rows) / len(rows), 6),
        "first_far_below_cost_rate": round(
            sum((first_buyer_offer(ep) or 0.0) < 0.75 * float(scenario_value(ep.scenario, "seller_cost", 0.0)) for ep in rows) / len(rows),
            6,
        ),
        "overshoot_rate": round(sum(bool(ep.buyer_overshoot) for ep in rows) / len(rows), 6),
    }


def summarize_episode_rows(rows: Iterable[Dict[str, Any]], errors: int) -> Dict[str, Any]:
    items = [row for row in rows if row.get("status")]
    if not items:
        return {"n": 0, "errors": errors, "score": -999.0}
    deals = [row for row in items if row.get("status") == "deal"]
    bargains = [float(row["buyer_bargained_ratio"]) for row in deals if row.get("buyer_bargained_ratio") is not None]
    rewards = [float(row.get("reward") or 0.0) for row in items]
    shaped = [float(row.get("shaped_score") or 0.0) for row in items]
    return {
        "n": len(items),
        "errors": errors,
        "score": round(statistics.mean(shaped) - 0.03 * errors, 6),
        "avg_reward": round(statistics.mean(rewards), 6),
        "deal_rate": round(len(deals) / len(items), 6),
        "buyer_bargained_ratio": round(statistics.mean(bargains), 6) if bargains else 0.0,
        "seller_quit_rate": round(sum(row.get("terminal_reason") == "seller_quit" for row in items) / len(items), 6),
        "accept_too_early_rate": round(sum(int(row.get("accept_too_early") or 0) for row in items) / len(items), 6),
        "repeated_far_below_cost_rate": round(sum(int(row.get("repeated_far_below_cost") or 0) for row in items) / len(items), 6),
        "first_far_below_cost_rate": round(sum(int(row.get("first_far_below_cost") or 0) for row in items) / len(items), 6),
        "overshoot_rate": round(sum(bool(row.get("buyer_overshoot")) for row in items) / len(items), 6),
    }


def rolling_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    if not items:
        return {"rolling_n": 0}
    bargains = [float(row["buyer_bargained_ratio"]) for row in items if row.get("buyer_bargained_ratio") is not None]
    return {
        "rolling_n": len(items),
        "rolling_reward": round(statistics.mean(float(row.get("reward") or 0.0) for row in items), 6),
        "rolling_shaped_score": round(statistics.mean(float(row.get("shaped_score") or 0.0) for row in items), 6),
        "rolling_deal_rate": round(sum(row.get("status") == "deal" for row in items) / len(items), 6),
        "rolling_bargained_ratio": round(statistics.mean(bargains), 6) if bargains else 0.0,
        "rolling_seller_quit_rate": round(sum(row.get("terminal_reason") == "seller_quit" for row in items) / len(items), 6),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_completed_episode_rows(
    path: Path,
    *,
    split_name: str,
    iteration: int,
    sample_id: int,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    completed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("split") != split_name:
            continue
        if int(row.get("iteration", -1)) != int(iteration) or int(row.get("sample_id", -1)) != int(sample_id):
            continue
        if row.get("error") or not row.get("status"):
            continue
        key = (str(row.get("scenario")), int(row.get("rollout", 0)))
        completed[key] = row
    return completed


def load_candidate_rows(path: Path, *, metric_key: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    completed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if metric_key not in row:
            continue
        key = (int(row.get("iteration", -1)), int(row.get("sample_id", -1)))
        completed[key] = row
    return completed


def restore_latest_distribution(out: Path, args: argparse.Namespace, default_mean: Dict[str, float], default_std: Dict[str, float]) -> Tuple[int, Dict[str, float], Dict[str, float], Dict[str, Any] | None]:
    if not args.resume:
        return 0, default_mean, default_std, None
    start_iteration = 0
    mean = default_mean
    std = default_std
    best = None
    for iteration in range(args.cem_iters):
        summary_path = out / f"iteration_{iteration:03d}_summary.json"
        if not summary_path.exists():
            break
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            break
        mean = payload.get("next_mean") or mean
        std = payload.get("next_std") or std
        best = payload.get("global_best_dev") or best
        start_iteration = iteration + 1
    if start_iteration:
        print(json.dumps({"resume_from_iteration": start_iteration}, ensure_ascii=False), flush=True)
    return start_iteration, mean, std, best


def main() -> None:
    args = parse_args()
    os.environ["OPENAI_REQUEST_TIMEOUT"] = str(args.openai_request_timeout)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_scenarios, train_scenarios, dev_scenarios = split_scenarios(args)
    write_json(
        out / "config.json",
        {
            **vars(args),
            "variant": "cem-planner-v0.2-contextual",
            "all_scenario_count": len(all_scenarios),
            "train_scenarios": [scenario_id(s) for s in train_scenarios],
            "dev_scenarios": [scenario_id(s) for s in dev_scenarios],
            "feature_names": FEATURE_NAMES,
            "adaptive_param_names": ADAPTIVE_PARAM_NAMES,
        },
    )

    rng = random.Random(args.seed)
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

    mean, std = initial_mean_std(args)
    elite_count = max(1, int(round(args.population_size * args.elite_frac)))
    dev_top_k = max(1, int(args.dev_eval_top_k))
    rolling = deque(maxlen=max(1, int(args.rolling_window)))
    start_iteration, mean, std, best = restore_latest_distribution(out, args, mean, std)
    completed_train = load_candidate_rows(out / "candidate_train_metrics.jsonl", metric_key="train_metrics") if args.resume else {}
    completed_dev = load_candidate_rows(out / "candidate_dev_metrics.jsonl", metric_key="dev_metrics") if args.resume else {}

    if start_iteration >= args.cem_iters and best is not None:
        write_json(out / "best_contextual_planner_policy.json", best)
        write_json(out / "best_contextual_policy_for_eval.json", {"contextual_policy": best["contextual_policy"]})
        print(json.dumps({"already_complete": True, "best_dev": best["dev_metrics"], "output_dir": str(out)}, ensure_ascii=False), flush=True)
        return

    for iteration in range(start_iteration, args.cem_iters):
        print(
            json.dumps(
                {
                    "cem_v02_iteration": iteration,
                    "train_n": len(train_scenarios) * args.rollouts_per_instance,
                    "dev_n_per_eval": len(dev_scenarios) * args.dev_rollouts_per_instance,
                    "population": args.population_size,
                    "elite_count": elite_count,
                }
            ),
            flush=True,
        )
        candidates = []
        for sample_id in range(args.population_size):
            policy_path = out / "candidate_policies" / f"iter_{iteration:03d}_sample_{sample_id:03d}.json"
            completed_row = completed_train.get((iteration, sample_id))
            if completed_row is not None and policy_path.exists():
                candidates.append(completed_row)
                print(
                    json.dumps(
                        {
                            "cem_v02_resume_train_candidate": {
                                "iteration": iteration,
                                "sample_id": sample_id,
                                **completed_row.get("train_metrics", {}),
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            if policy_path.exists():
                policy, vector = load_policy_payload(policy_path)
            else:
                vector = sample_vector(mean, std, rng)
                policy = vector_to_policy(vector)
                write_json(policy_path, {"contextual_policy": policy.to_dict(), "iteration": iteration, "sample_id": sample_id})
            train_metrics = run_policy_eval(
                args=args,
                policy_path=policy_path,
                split_name="train",
                scenarios=train_scenarios,
                rollouts_per_instance=args.rollouts_per_instance,
                buyer_client=buyer_client,
                seller_client=seller_client,
                iteration=iteration,
                sample_id=sample_id,
                rolling_history=rolling,
                episode_log_path=out / "episode_metrics.jsonl",
            )
            row = {
                "iteration": iteration,
                "sample_id": sample_id,
                "policy_path": str(policy_path),
                "vector": vector,
                "contextual_policy": policy.to_dict(),
                "train_metrics": train_metrics,
            }
            candidates.append(row)
            append_jsonl(out / "candidate_train_metrics.jsonl", row)
            print(
                json.dumps(
                    {
                        "cem_v02_train_candidate": {
                            "iteration": iteration,
                            "sample_id": sample_id,
                            **train_metrics,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        candidates.sort(key=lambda item: item["train_metrics"]["score"], reverse=True)
        dev_candidates = []
        for row in candidates[:dev_top_k]:
            completed_dev_row = completed_dev.get((iteration, row["sample_id"]))
            if completed_dev_row is not None:
                row = completed_dev_row
                dev_metrics = row["dev_metrics"]
                print(
                    json.dumps(
                        {
                            "cem_v02_resume_dev_candidate": {
                                "iteration": iteration,
                                "sample_id": row["sample_id"],
                                **dev_metrics,
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                dev_metrics = run_policy_eval(
                    args=args,
                    policy_path=Path(row["policy_path"]),
                    split_name="dev",
                    scenarios=dev_scenarios,
                    rollouts_per_instance=args.dev_rollouts_per_instance,
                    buyer_client=buyer_client,
                    seller_client=seller_client,
                    iteration=iteration,
                    sample_id=row["sample_id"],
                    rolling_history=rolling,
                    episode_log_path=out / "episode_metrics.jsonl",
                )
                row = {**row, "dev_metrics": dev_metrics}
                append_jsonl(out / "candidate_dev_metrics.jsonl", row)
            dev_candidates.append(row)
            print(
                json.dumps(
                    {
                        "cem_v02_dev_candidate": {
                            "iteration": iteration,
                            "sample_id": row["sample_id"],
                            **dev_metrics,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if best is None or dev_metrics["score"] > best["dev_metrics"]["score"]:
                best = row
                write_json(out / "best_contextual_planner_policy.json", best)

        elite_vectors = [row["vector"] for row in candidates[:elite_count]]
        mean, std = update_distribution(elite_vectors, std)
        write_json(
            out / f"iteration_{iteration:03d}_summary.json",
            {
                "iteration": iteration,
                "best_train": candidates[0],
                "best_dev_this_iter": max(dev_candidates, key=lambda item: item["dev_metrics"]["score"]) if dev_candidates else None,
                "global_best_dev": best,
                "next_mean": mean,
                "next_std": std,
            },
        )

    assert best is not None
    write_json(out / "best_contextual_planner_policy.json", best)
    write_json(out / "best_contextual_policy_for_eval.json", {"contextual_policy": best["contextual_policy"]})
    print(json.dumps({"best_dev": best["dev_metrics"], "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
