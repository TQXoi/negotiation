#!/usr/bin/env python3
"""Modular AgenticPay evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from AgenticPay_Env.buyer.variants import DEFAULT_BASELINES, parse_buyer_variants
from AgenticPay_Env.environment.single28 import Single28RunConfig, run_single28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="single28", choices=["single28", "multi_agent", "agenticpay_all_tasks"])
    parser.add_argument(
        "--multi-agent-suite",
        default="1bms",
        choices=["1bms", "mb1s", "mbms"],
        help="AgenticPay multi-agent shape when --suite multi_agent.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--buyer-model", default=None, help="Optional buyer model spec. Defaults to --model.")
    parser.add_argument("--seller-model", default=None, help="Optional seller model spec. Defaults to --model.")
    parser.add_argument("--buyer-model-alias", default=None)
    parser.add_argument("--seller-model-alias", default=None)
    parser.add_argument("--buyer-variants", default=",".join(DEFAULT_BASELINES))
    parser.add_argument(
        "--agenticpay-task-suites",
        default="all",
        help="Comma-separated AgenticPay example suites for --suite agenticpay_all_tasks, or all.",
    )
    parser.add_argument("--seller-variant", default="native", choices=["native", "validated"])
    parser.add_argument(
        "--focal-buyer-index",
        type=int,
        default=None,
        help=(
            "1-based BuyerAgent construction index to replace in agenticpay_all_tasks. "
            "All other buyers remain the upstream native BuyerAgent. Omit to replace all buyers."
        ),
    )
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per buyer variant for multi_agent suite.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--resume-from-jsonl",
        default=None,
        help="Optional prior Single28 JSONL; selected completed rows are reused.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=str(WORKSPACE / "runs" / "agenticpay" / "single28"))
    parser.add_argument("--planner-checkpoint", default=None, help="Checkpoint for learned residual planner variants.")
    return parser.parse_args()


def default_model_alias(model: str) -> str:
    if model.startswith("openai:"):
        return model.split(":", 1)[1]
    return Path(model.removeprefix("hf:")).name or "model"


def main() -> None:
    args = parse_args()
    variants = parse_buyer_variants(args.buyer_variants)
    if args.suite == "multi_agent":
        from AgenticPay_Env.environment.multi_agent import MultiAgentRunConfig, run_multi_agent

        config = MultiAgentRunConfig(
            suite=args.multi_agent_suite,
            buyer_model=args.buyer_model or args.model,
            seller_model=args.seller_model or args.model,
            buyer_model_alias=args.buyer_model_alias or args.model_alias or default_model_alias(args.buyer_model or args.model),
            seller_model_alias=args.seller_model_alias or args.model_alias or default_model_alias(args.seller_model or args.model),
            output_dir=Path(args.output_dir),
            buyer_variants=variants,
            seller_variant=args.seller_variant,
            repeats=args.repeats,
            limit=args.limit,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            resume=not args.no_resume,
        )
        result = run_multi_agent(config, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.suite == "agenticpay_all_tasks":
        from AgenticPay_Env.environment.all_tasks import AllTasksRunConfig, run_all_tasks

        task_suites = [item.strip() for item in args.agenticpay_task_suites.split(",") if item.strip()]
        config = AllTasksRunConfig(
            buyer_model=args.buyer_model or args.model,
            seller_model=args.seller_model or args.model,
            buyer_model_alias=args.buyer_model_alias or args.model_alias or default_model_alias(args.buyer_model or args.model),
            seller_model_alias=args.seller_model_alias or args.model_alias or default_model_alias(args.seller_model or args.model),
            output_dir=Path(args.output_dir),
            buyer_variants=variants,
            task_suites=task_suites,
            seller_variant=args.seller_variant,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            planner_checkpoint=args.planner_checkpoint,
            tasks=args.tasks,
            limit=args.limit,
            resume=not args.no_resume,
            focal_buyer_index=args.focal_buyer_index,
        )
        result = run_all_tasks(config, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    config = Single28RunConfig(
        model=args.model,
        model_alias=args.model_alias or default_model_alias(args.model),
        buyer_model=args.buyer_model,
        seller_model=args.seller_model,
        buyer_model_alias=args.buyer_model_alias,
        seller_model_alias=args.seller_model_alias,
        output_dir=Path(args.output_dir),
        buyer_variants=variants,
        seller_variant=args.seller_variant,
        tasks=args.tasks,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        gpu_id=args.gpu_id,
        require_cuda=args.require_cuda,
        limit=args.limit,
        planner_checkpoint=args.planner_checkpoint,
        resume=not args.no_resume,
        resume_from_jsonl=(
            Path(args.resume_from_jsonl) if args.resume_from_jsonl else None
        ),
    )
    result = run_single28(config, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
