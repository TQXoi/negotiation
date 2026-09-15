#!/usr/bin/env python3
"""Run a minimal AgenticPay buyer-seller evaluation with a Hugging Face causal LM.

The default setup follows AgenticPay's original Task1 basic price negotiation
configuration: buyer_max_price=150, seller_min_price=80, initial_seller_price=150,
and the winter jacket scenario.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
AGENTICPAY_ROOT = ROOT / "benchmarks" / "AgenticPay"
sys.path.insert(0, str(AGENTICPAY_ROOT))

from agenticpay import make  # noqa: E402
from agenticpay.agents.buyer_agent import BuyerAgent  # noqa: E402
from agenticpay.agents.seller_agent import SellerAgent  # noqa: E402
from agenticpay.models.base_llm import BaseLLM  # noqa: E402


@dataclass
class HFModelConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    torch_dtype: str
    device_map: str
    cache_dir: Optional[str]


class HuggingFaceCausalLM(BaseLLM):
    """Small AgenticPay-compatible wrapper around transformers generation."""

    def __init__(self, config: HFModelConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.model_id = config.model_id
        dtype = self._resolve_dtype(torch, config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=config.device_map,
            cache_dir=config.cache_dir,
        )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return text

    def __repr__(self) -> str:
        return f"HuggingFaceCausalLM(model_id={self.model_id})"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def default_task_config(max_rounds: int) -> Dict[str, Any]:
    return {
        "buyer_max_price": 150.0,
        "seller_min_price": 80.0,
        "initial_seller_price": 150.0,
        "max_rounds": max_rounds,
        "price_tolerance": 0.0,
        "reward_weights": {
            "buyer_savings": 1.0,
            "seller_profit": 1.0,
            "time_cost": 0.1,
        },
        "environment_info": {
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        "user_requirement": "I need a high-quality winter jacket for cold weather",
        "user_profile": (
            "User prefers business/professional style and likes to compare prices before "
            "making purchases. In negotiations, they may mention comparing other options "
            "and seek better deals."
        ),
        "product_info": {
            "name": "Premium Winter Jacket",
            "brand": "Mountain Gear",
            "price": 180.0,
            "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
            "condition": "New",
            "material": "Gore-Tex",
        },
    }


def run_episode(model: BaseLLM, seed: int, max_rounds: int) -> Dict[str, Any]:
    set_seed(seed)
    cfg = default_task_config(max_rounds=max_rounds)

    buyer = BuyerAgent(
        model=model,
        buyer_max_price=cfg["buyer_max_price"],
        system_prompt_suffix=(
            "You are being evaluated as the buyer. Be concise, strategic, and always "
            "include the required price tag. Do not reveal your maximum acceptable price."
        ),
    )
    seller = SellerAgent(
        model=model,
        seller_min_price=cfg["seller_min_price"],
        system_prompt_suffix=(
            "Keep the original seller behavior: professional, value-focused, and willing "
            "to negotiate without going below your confidential minimum acceptable price."
        ),
    )

    env = make(
        "Task1_basic_price_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=cfg["max_rounds"],
        initial_seller_price=cfg["initial_seller_price"],
        buyer_max_price=cfg["buyer_max_price"],
        seller_min_price=cfg["seller_min_price"],
        environment_info=cfg["environment_info"],
        price_tolerance=cfg["price_tolerance"],
        reward_weights=cfg["reward_weights"],
    )

    observation, info = env.reset(
        user_requirement=cfg["user_requirement"],
        product_info=cfg["product_info"],
        user_profile=cfg["user_profile"],
    )

    start = time.time()
    done = False
    rounds: List[Dict[str, Any]] = []
    final_reward = 0.0
    while not done:
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )
        updated_history = observation["conversation_history"].copy()
        updated_history.append(
            {
                "role": "buyer",
                "content": buyer_action,
                "round": observation.get("current_round", 0),
            }
        )
        seller_action = seller.respond(
            conversation_history=updated_history,
            current_state=observation,
        )
        observation, reward, terminated, truncated, info = env.step(
            buyer_action=buyer_action,
            seller_action=seller_action,
        )
        final_reward = reward
        done = terminated or truncated
        rounds.append(
            {
                "round": observation.get("current_round", len(rounds)),
                "buyer_action": buyer_action,
                "seller_action": seller_action,
                "buyer_price": observation.get("buyer_price"),
                "seller_price": observation.get("seller_price"),
                "agreed_price": observation.get("agreed_price"),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    status = str(getattr(info.get("negotiation_info"), "status", "unknown"))
    result = {
        "task": "Task1_basic_price_negotiation",
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": repr(model),
        "status": status,
        "termination_reason": info.get("termination_reason"),
        "success": info.get("termination_reason") == "agreed",
        "reward": final_reward,
        "buyer_reward": info.get("buyer_reward"),
        "seller_reward": info.get("seller_reward"),
        "global_score": info.get("global_score"),
        "buyer_score": info.get("buyer_score"),
        "seller_score": info.get("seller_score"),
        "round_count": observation.get("current_round"),
        "buyer_price": observation.get("buyer_price"),
        "seller_price": observation.get("seller_price"),
        "agreed_price": observation.get("agreed_price"),
        "elapsed_seconds": round(time.time() - start, 3),
        "config": cfg,
        "rounds": rounds,
    }
    return result


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key: str) -> Optional[float]:
        vals = [r.get(key) for r in results if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    return {
        "episodes": len(results),
        "deal_rate": sum(1 for r in results if r.get("success")) / max(1, len(results)),
        "avg_buyer_score": mean("buyer_score"),
        "avg_seller_score": mean("seller_score"),
        "avg_global_score": mean("global_score"),
        "avg_round_count": mean("round_count"),
        "avg_agreed_price": mean("agreed_price"),
        "avg_elapsed_seconds": mean("elapsed_seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--gpu-id",
        default=None,
        help=(
            "Restrict this run to one physical GPU id by setting CUDA_VISIBLE_DEVICES. "
            "Example: --gpu-id 3 makes the selected GPU appear as cuda:0 inside the process."
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not visible after applying --gpu-id.",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "huggingface"))
    parser.add_argument("--out", default=str(ROOT / "results" / "agenticpay_qwen_task1.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                "CUDA is not available. "
                f"CUDA_VISIBLE_DEVICES={visible}. "
                "Run from a GPU-visible session or choose a valid --gpu-id."
            )

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = HuggingFaceCausalLM(
        HFModelConfig(
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
        )
    )

    results = []
    with out_path.open("a", encoding="utf-8") as f:
        for seed in seeds:
            result = run_episode(model=model, seed=seed, max_rounds=args.max_rounds)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ["seed", "success", "buyer_score", "seller_score", "global_score", "round_count", "agreed_price"]}, ensure_ascii=False))

    print("SUMMARY")
    print(json.dumps(aggregate(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
