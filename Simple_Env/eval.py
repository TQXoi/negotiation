#!/usr/bin/env python3
"""Simplified paper-aligned RLVR negotiation evaluation.

This entrypoint wires together Simple_Env.buyer, Simple_Env.seller, and
Simple_Env.environment. It intentionally prints one compact JSON line after
every completed episode so partial runs are easy to inspect.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

POLICY_MODEL_BUYER_VARIANTS = {
    "small_policy_v03",
    "price_policy_v03",
    "rl_price_policy_v03",
    "small_policy_belief_v03",
    "small_policy_v03_belief",
    "full_framework_small_policy_v03",
}
sys.path.insert(0, str(ROOT))

from experiments.model_clients import make_model_client  # noqa: E402
from Simple_Env.buyer import make_buyer  # noqa: E402
from Simple_Env.environment import (  # noqa: E402
    RLVREvalEpisode,
    append_episode,
    load_existing_episodes,
    load_final_failed_episode_keys,
    print_episode_result,
    record_failed_episode,
    run_episode,
    write_static_outputs,
    write_summary,
)
from Simple_Env.environment.data import load_or_generate_scenarios  # noqa: E402
from Simple_Env.seller import make_seller  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer-model", required=True, help="Model spec, e.g. openai:Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument(
        "--planner-model",
        default=None,
        help=(
            "Optional model spec used only by full-framework planner calls. "
            "Belief and final Thought/Talk/Action generation still use --buyer-model."
        ),
    )
    parser.add_argument(
        "--policy-model",
        default=None,
        help=(
            "Optional small policy model spec used by price-policy variants such as small_policy_v03. "
            "This does not affect full_framework/planner_generator baselines."
        ),
    )
    parser.add_argument("--seller-model", required=True, help="Fixed seller model spec.")
    parser.add_argument("--buyer-variants", default="direct_prompt,cot_prompt,belief_prompt,full_framework")
    parser.add_argument("--seller-type", default="default")
    parser.add_argument("--seller-persona", choices=["neutral", "begging", "insulting", "unyielding"], default="neutral")
    parser.add_argument("--variant-schedule", choices=["sequential", "round_robin"], default="round_robin")
    parser.add_argument("--num-test-instances", type=int, default=128)
    parser.add_argument(
        "--scenario-offset",
        type=int,
        default=0,
        help="Skip this many ordered JSONL scenarios before evaluation; useful for a held-out pilot.",
    )
    parser.add_argument("--rollouts-per-instance", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--buyer-temperature", type=float, default=1.0)
    parser.add_argument("--seller-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument(
        "--candidate-offer-k",
        type=int,
        default=3,
        help="Number of candidate offers used by counterfactual-response buyer variants.",
    )
    parser.add_argument(
        "--continuous-planner-params-json",
        default=None,
        help="JSON file with CEM-tuned PlannerParams for continuous_rule_ev_cem.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive-gains-probability", type=float, default=0.7)
    parser.add_argument("--scenarios-jsonl", default=None)
    parser.add_argument("--require-scenarios-jsonl", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--openai-request-timeout", type=float, default=1800)
    parser.add_argument("--episode-retries", type=int, default=2)
    parser.add_argument("--episode-retry-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--max-episode-errors-per-variant", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "simple_env_runs" / "simple_env_rlvr_eval"))
    return parser.parse_args()


def run_episode_with_retries(
    args: argparse.Namespace,
    *,
    variant: str,
    scenario: Any,
    rollout: int,
    buyer_client: Any,
    seller_client: Any,
    planner_client: Any = None,
    policy_client: Any = None,
) -> RLVREvalEpisode:
    max_attempts = max(1, int(args.episode_retries) + 1)
    for attempt in range(1, max_attempts + 1):
        buyer = make_buyer(
            variant,
            buyer_client,
            args.max_tokens,
            args.buyer_temperature,
            args.top_p,
            candidate_offer_k=args.candidate_offer_k,
            planner_client=policy_client if variant in POLICY_MODEL_BUYER_VARIANTS else planner_client,
            continuous_planner_params_json=args.continuous_planner_params_json,
        )
        seller = make_seller(
            args.seller_type,
            seller_client,
            args.max_tokens,
            args.seller_temperature,
            args.top_p,
            args.seller_persona,
        )
        try:
            return run_episode(
                variant=variant,
                scenario=scenario,
                rollout=rollout,
                buyer=buyer,
                seller=seller,
                max_turns=args.max_turns,
            )
        except Exception as exc:
            final = attempt >= max_attempts
            record_failed_episode(
                args,
                variant=variant,
                scenario=scenario,
                rollout=rollout,
                exc=exc,
                attempt=attempt,
                final=final,
            )
            if final:
                raise
            backoff = max(0.0, float(args.episode_retry_backoff_seconds)) * (2 ** (attempt - 1))
            print(
                json.dumps(
                    {
                        "retry_episode": {
                            "buyer": variant,
                            "seller": args.seller_type,
                            "scenario": scenario.item_id,
                            "rollout": rollout,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "sleep_seconds": backoff,
                            "error": type(exc).__name__,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if backoff > 0:
                time.sleep(backoff)
    raise RuntimeError("unreachable retry state")


def handle_completed_episode(
    args: argparse.Namespace,
    episodes_by_variant: Dict[str, List[RLVREvalEpisode]],
    episode: RLVREvalEpisode,
) -> None:
    episodes_by_variant[episode.variant].append(episode)
    append_episode(args, episode)
    print_episode_result(episode, args.seller_type, args.seller_model)
    if args.summary_every > 0 and len(episodes_by_variant[episode.variant]) % args.summary_every == 0:
        write_summary(args, episodes_by_variant)


def handle_failed_episode(
    args: argparse.Namespace,
    errors_by_variant: Dict[str, int],
    job: tuple[str, Any, int],
    exc: BaseException,
) -> None:
    variant, scenario, rollout = job
    errors_by_variant[variant] += 1
    print(
        json.dumps(
            {
                "failed_episode": {
                    "buyer": variant,
                    "seller": args.seller_type,
                    "scenario": scenario.item_id,
                    "rollout": rollout,
                    "error": type(exc).__name__,
                    "errors_for_buyer": errors_by_variant[variant],
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.openai_request_timeout is not None:
        os.environ["OPENAI_REQUEST_TIMEOUT"] = str(args.openai_request_timeout)

    variants = [item.strip() for item in args.buyer_variants.split(",") if item.strip()]
    requested_instances = args.num_test_instances
    scenario_offset = max(0, int(args.scenario_offset))
    if scenario_offset:
        # The shared loader truncates to num_test_instances, so temporarily
        # request the prefix plus held-out slice and restore the reported N.
        args.num_test_instances = requested_instances + scenario_offset
    scenarios = load_or_generate_scenarios(args)
    scenarios = scenarios[scenario_offset : scenario_offset + requested_instances]
    args.num_test_instances = requested_instances
    if len(scenarios) != requested_instances:
        raise ValueError(
            f"Requested {requested_instances} scenarios at offset {scenario_offset}, "
            f"but only loaded {len(scenarios)}."
        )
    write_static_outputs(args, scenarios)

    if args.dry_run:
        write_summary(args, {variant: [] for variant in variants})
        print(json.dumps({"dry_run": True, "output_dir": args.output_dir, "variants": variants}, ensure_ascii=False))
        return

    buyer_client = make_model_client(
        args.buyer_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        enable_thinking=args.enable_thinking,
    )
    planner_client = None
    if args.planner_model:
        if args.planner_model == args.buyer_model:
            planner_client = buyer_client
        else:
            planner_client = make_model_client(
                args.planner_model,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                cache_dir=args.cache_dir,
                enable_thinking=args.enable_thinking,
            )
    policy_client = None
    if args.policy_model:
        if args.policy_model == args.buyer_model:
            policy_client = buyer_client
        elif args.policy_model == args.planner_model and planner_client is not None:
            policy_client = planner_client
        else:
            policy_client = make_model_client(
                args.policy_model,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                cache_dir=args.cache_dir,
                enable_thinking=args.enable_thinking,
            )
    if args.seller_model == args.buyer_model:
        seller_client = buyer_client
    else:
        seller_client = make_model_client(
            args.seller_model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            enable_thinking=args.enable_thinking,
        )

    episodes_by_variant: Dict[str, List[RLVREvalEpisode]] = {
        variant: load_existing_episodes(args, variant) if args.resume else [] for variant in variants
    }
    failed_by_variant = (
        load_final_failed_episode_keys(args, variants)
        if args.resume and not args.retry_failed
        else {variant: set() for variant in variants}
    )
    completed_by_variant = {
        variant: {(ep.scenario.get("item_id"), ep.rollout) for ep in episodes} | failed_by_variant[variant]
        for variant, episodes in episodes_by_variant.items()
    }
    for variant, episodes in episodes_by_variant.items():
        if episodes:
            print(json.dumps({"buyer": variant, "resumed_episodes": len(episodes)}, ensure_ascii=False), flush=True)
        if failed_by_variant[variant]:
            print(
                json.dumps(
                    {"buyer": variant, "resumed_final_failed_episodes": len(failed_by_variant[variant])},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_summary(args, episodes_by_variant)

    errors_by_variant = {variant: 0 for variant in variants}
    if args.variant_schedule == "round_robin":
        jobs = [
            (variant, scenario, rollout)
            for scenario in scenarios
            for rollout in range(args.rollouts_per_instance)
            for variant in variants
            if (scenario.item_id, rollout) not in completed_by_variant[variant]
        ]
    else:
        jobs = [
            (variant, scenario, rollout)
            for variant in variants
            for scenario in scenarios
            for rollout in range(args.rollouts_per_instance)
            if (scenario.item_id, rollout) not in completed_by_variant[variant]
        ]

    def run_job(job: tuple[str, Any, int]) -> RLVREvalEpisode:
        variant, scenario, rollout = job
        return run_episode_with_retries(
            args,
            variant=variant,
            scenario=scenario,
            rollout=rollout,
            buyer_client=buyer_client,
            seller_client=seller_client,
            planner_client=planner_client,
            policy_client=policy_client,
        )

    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        for job in jobs:
            variant, _scenario, _rollout = job
            if errors_by_variant[variant] >= args.max_episode_errors_per_variant:
                continue
            try:
                handle_completed_episode(args, episodes_by_variant, run_job(job))
            except Exception as exc:
                handle_failed_episode(args, errors_by_variant, job, exc)
        write_summary(args, episodes_by_variant)
        return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: Dict[Any, tuple[str, Any, int]] = {}
        job_iter = iter(jobs)

        def submit_until_full() -> None:
            while len(futures) < concurrency:
                try:
                    job = next(job_iter)
                except StopIteration:
                    return
                variant, _scenario, _rollout = job
                if errors_by_variant[variant] >= args.max_episode_errors_per_variant:
                    continue
                futures[executor.submit(run_job, job)] = job

        submit_until_full()
        while futures:
            done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    handle_completed_episode(args, episodes_by_variant, future.result())
                except Exception as exc:
                    handle_failed_episode(args, errors_by_variant, job, exc)
            submit_until_full()
    write_summary(args, episodes_by_variant)


if __name__ == "__main__":
    main()
