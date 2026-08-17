"""AgenticPay multi-agent task wrapper for 1BMS, MB1S, and MBMS.

This runner keeps AgenticPay's original environment classes and score
calculation, but replaces the buyer side with the same modular framework used
in the single-buyer/seller experiments. It currently covers the parallel Task1
variants:

- 1BMS: one buyer, two sellers
- MB1S: two buyers, one seller
- MBMS: two buyers, two sellers
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

WORKSPACE = Path(__file__).resolve().parents[2]
AGENTICPAY_ROOT = WORKSPACE / "benchmarks" / "AgenticPay"
for path in [WORKSPACE, AGENTICPAY_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.examples.config import (
    buyer_reward_aggregation,
    max_rounds,
    price_tolerance,
    reward_weights,
    seller_reward_aggregation,
)
from agenticpay.envs.multi_buyer_multi_seller.Task1_parallel_two_buyer_two_seller_negotiation import (
    Task1ParallelTwoBuyerTwoSellerNegotiation,
)
from agenticpay.envs.only_multi_buyer.Task1_parallel_two_buyer_negotiation import (
    Task1ParallelTwoBuyerNegotiation,
)
from agenticpay.envs.only_multi_seller.Task1_parallel_two_seller_negotiation import (
    Task1ParallelTwoSellerNegotiation,
)

from experiments.agenticpay_framework import BASELINE_BUYER_VARIANTS, ModularBuyerAgent, make_baseline_buyer_agent
from experiments.model_clients import ModelClient, make_model_client
from AgenticPay_Env.buyer.universal_framework import UniversalAgenticPayBuyerAgent
from AgenticPay_Env.buyer.variants import universal_belief_mode


PRODUCT_INFO: Dict[str, Any] = {
    "name": "Premium Winter Jacket",
    "brand": "Mountain Gear",
    "price": 180.0,
    "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
    "condition": "New",
    "material": "Gore-Tex",
}

USER_REQUIREMENT = "I need a high-quality winter jacket for cold weather"
USER_PROFILE = (
    "User prefers business/professional style and likes to compare prices before making purchases. "
    "In negotiations, they may mention comparing other options and seek better deals."
)
ENVIRONMENT_INFO = {"temperature": "warm", "season": "summer", "weather": "sunny"}


@dataclass
class MultiAgentRunConfig:
    suite: str
    buyer_model: str
    seller_model: str
    buyer_model_alias: str
    seller_model_alias: str
    buyer_variants: List[str]
    output_dir: Path
    seller_variant: str = "native"
    repeats: int = 1
    limit: Optional[int] = None
    seed: int = 0
    max_new_tokens: int = 1024
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    resume: bool = True


def run_multi_agent(config: MultiAgentRunConfig, *, dry_run: bool = False) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    command_payload = {"config": _jsonable_config(config)}
    (config.output_dir / "agenticpay_multi_agent_config.json").write_text(
        json.dumps(command_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        return {"dry_run": True, **command_payload}

    buyer_client = make_model_client(
        config.buyer_model,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        cache_dir=config.cache_dir,
    )
    seller_client = (
        buyer_client
        if config.seller_model == config.buyer_model
        else make_model_client(
            config.seller_model,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            cache_dir=config.cache_dir,
        )
    )

    records_path = config.output_dir / "episodes.jsonl"
    completed = _load_completed(records_path) if config.resume else set()
    all_records: List[Dict[str, Any]] = []
    if config.resume and records_path.exists():
        all_records.extend(_read_jsonl(records_path))

    planned_jobs = _planned_jobs(config)
    for variant, repeat_id in planned_jobs:
        key = _episode_key(config.suite, variant, repeat_id, config.buyer_model, config.seller_model)
        if key in completed:
            continue
        start = time.time()
        try:
            record = _run_one_episode(
                suite=config.suite,
                buyer_variant=variant,
                buyer_client=buyer_client,
                seller_client=seller_client,
                buyer_model_spec=config.buyer_model,
                seller_model_spec=config.seller_model,
                buyer_model_alias=config.buyer_model_alias,
                seller_model_alias=config.seller_model_alias,
                seller_variant=config.seller_variant,
                max_tokens=config.max_new_tokens,
                repeat_id=repeat_id,
            )
        except Exception as exc:  # Keep long runs resumable.
            record = {
                "suite": config.suite,
                "buyer_variant": variant,
                "repeat_id": repeat_id,
                "buyer_model_spec": config.buyer_model,
                "seller_model_spec": config.seller_model,
                "status": "error",
                "success": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        record["elapsed_time"] = round(time.time() - start, 3)
        all_records.append(record)
        _append_jsonl(records_path, record)
        print(json.dumps({"episode_result": _compact_record(record)}, ensure_ascii=False), flush=True)

    summary = summarize_records(all_records)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"summary": str(config.output_dir / "summary.json"), "summary_data": summary, **command_payload}


def _planned_jobs(config: MultiAgentRunConfig) -> List[Tuple[str, int]]:
    jobs = [(variant, i) for i in range(config.repeats) for variant in config.buyer_variants]
    if config.limit is not None:
        jobs = jobs[: config.limit]
    return jobs


def _run_one_episode(
    *,
    suite: str,
    buyer_variant: str,
    buyer_client: ModelClient,
    seller_client: ModelClient,
    buyer_model_spec: str,
    seller_model_spec: str,
    buyer_model_alias: str,
    seller_model_alias: str,
    seller_variant: str,
    max_tokens: int,
    repeat_id: int,
) -> Dict[str, Any]:
    agents = _build_agents(suite, buyer_variant, buyer_client, seller_client, seller_variant, max_tokens)
    env = _build_env(suite, agents)
    observation, _info = env.reset(
        user_requirement=USER_REQUIREMENT,
        product_info=PRODUCT_INFO,
        user_profile=USER_PROFILE,
    )
    turns: List[Dict[str, Any]] = []
    done = False
    reward = 0.0
    info: Dict[str, Any] = {}

    while not done:
        if suite == "1bms":
            actions, traces = _act_1bms(agents, observation)
            observation, reward, terminated, truncated, info = env.step(**actions)
        elif suite == "mb1s":
            actions, traces = _act_mb1s(agents, observation)
            observation, reward, terminated, truncated, info = env.step(**actions)
        elif suite == "mbms":
            actions, traces = _act_mbms(agents, observation)
            observation, reward, terminated, truncated, info = env.step(**actions)
        else:
            raise ValueError(f"Unsupported suite: {suite}")
        done = bool(terminated or truncated)
        turns.append(
            {
                "round": info.get("round"),
                "actions": actions,
                "reward": reward,
                "info": _jsonable(info),
                "framework_traces": traces,
            }
        )

    env.close()
    status = _status_value(info.get("status", "unknown"))
    record = {
        "suite": suite,
        "task": _task_name(suite),
        "buyer_variant": buyer_variant,
        "seller_variant": seller_variant,
        "repeat_id": repeat_id,
        "buyer_model_alias": buyer_model_alias,
        "seller_model_alias": seller_model_alias,
        "buyer_model_spec": buyer_model_spec,
        "seller_model_spec": seller_model_spec,
        "status": status,
        "success": bool(status == "agreed" or info.get("final_deal_price") is not None),
        "termination_reason": info.get("termination_reason"),
        "selected_buyer": info.get("selected_buyer"),
        "selected_seller": info.get("selected_seller"),
        "final_deal_price": info.get("final_deal_price"),
        "total_rounds": info.get("round"),
        "total_reward": float(reward) if reward is not None else None,
        "buyer_score": info.get("buyer_score"),
        "seller_score": info.get("seller_score"),
        "global_score": info.get("global_score"),
        "buyer_reward": info.get("buyer_reward"),
        "seller_reward": info.get("seller_reward"),
        "buyer1_reward": info.get("buyer1_reward"),
        "buyer2_reward": info.get("buyer2_reward"),
        "seller1_reward": info.get("seller1_reward"),
        "seller2_reward": info.get("seller2_reward"),
        "buyer1_max_price": getattr(env, "buyer1_max_price", None),
        "buyer2_max_price": getattr(env, "buyer2_max_price", None),
        "buyer_max_price": getattr(env, "buyer_max_price", None),
        "seller_min_price": getattr(env, "seller_min_price", None),
        "seller1_min_price": getattr(env, "seller1_min_price", None),
        "seller2_min_price": getattr(env, "seller2_min_price", None),
        "product_info": PRODUCT_INFO,
        "final_info": _jsonable(info),
        "turns": turns,
    }
    return record


def _build_agents(
    suite: str,
    buyer_variant: str,
    buyer_client: ModelClient,
    seller_client: ModelClient,
    seller_variant: str,
    max_tokens: int,
) -> Dict[str, Any]:
    if seller_variant != "native":
        raise ValueError("multi_agent runner currently supports seller_variant=native only.")

    def buyer(name: str, buyer_max_price: float) -> Any:
        kwargs = {
            "model": buyer_client,
            "name": name,
            "role_description": "You are a buyer looking for a good deal.",
            "buyer_max_price": buyer_max_price,
            "max_tokens": max_tokens,
        }
        if buyer_variant == "repo_native":
            return BuyerAgent(
                model=buyer_client,
                name=name,
                role_description=kwargs["role_description"],
                buyer_max_price=buyer_max_price,
            )
        if buyer_variant in BASELINE_BUYER_VARIANTS:
            return make_baseline_buyer_agent(variant=buyer_variant, **kwargs)
        belief_mode = universal_belief_mode(buyer_variant)
        if belief_mode is not None:
            return UniversalAgenticPayBuyerAgent(
                **kwargs,
                belief_mode=belief_mode,
                persistent_opponent_memory=(buyer_variant == "universal_framework_v2_conservative_awr"),
            )
        return ModularBuyerAgent(variant=buyer_variant, **kwargs)

    def seller(name: str, seller_min_price: float) -> SellerAgent:
        return SellerAgent(
            model=seller_client,
            name=name,
            role_description="You are a seller looking to make a good deal.",
            seller_min_price=seller_min_price,
        )

    if suite == "1bms":
        return {"buyer": buyer("Buyer", 150.0), "seller1": seller("Seller1", 80.0), "seller2": seller("Seller2", 85.0)}
    if suite == "mb1s":
        return {"buyer1": buyer("Buyer1", 150.0), "buyer2": buyer("Buyer2", 160.0), "seller": seller("Seller", 80.0)}
    if suite == "mbms":
        return {
            "buyer1": buyer("Buyer1", 150.0),
            "buyer2": buyer("Buyer2", 160.0),
            "seller1": seller("Seller1", 80.0),
            "seller2": seller("Seller2", 85.0),
        }
    raise ValueError(f"Unsupported suite: {suite}")


def _build_env(suite: str, agents: Dict[str, Any]) -> Any:
    common = {
        "max_rounds": max_rounds,
        "environment_info": ENVIRONMENT_INFO,
        "price_tolerance": price_tolerance,
        "reward_weights": reward_weights,
        "buyer_reward_aggregation": buyer_reward_aggregation,
        "seller_reward_aggregation": seller_reward_aggregation,
    }
    if suite == "1bms":
        return Task1ParallelTwoSellerNegotiation(
            buyer_agent=agents["buyer"],
            seller1_agent=agents["seller1"],
            seller2_agent=agents["seller2"],
            initial_seller1_price=150.0,
            initial_seller2_price=160.0,
            buyer_max_price=150.0,
            seller1_min_price=80.0,
            seller2_min_price=85.0,
            **common,
        )
    if suite == "mb1s":
        return Task1ParallelTwoBuyerNegotiation(
            buyer1_agent=agents["buyer1"],
            buyer2_agent=agents["buyer2"],
            seller_agent=agents["seller"],
            initial_seller_price=150.0,
            buyer1_max_price=150.0,
            buyer2_max_price=160.0,
            seller_min_price=80.0,
            **common,
        )
    if suite == "mbms":
        return Task1ParallelTwoBuyerTwoSellerNegotiation(
            buyer1_agent=agents["buyer1"],
            buyer2_agent=agents["buyer2"],
            seller1_agent=agents["seller1"],
            seller2_agent=agents["seller2"],
            initial_seller1_price=150.0,
            initial_seller2_price=160.0,
            buyer1_max_price=150.0,
            buyer2_max_price=160.0,
            seller1_min_price=80.0,
            seller2_min_price=85.0,
            **common,
        )
    raise ValueError(f"Unsupported suite: {suite}")


def _act_1bms(agents: Dict[str, Any], observation: Dict[str, Any]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    buyer = agents["buyer"]
    b1 = buyer.respond(observation["conversation_history_seller1"], _for_counterparty(observation, "seller1"))
    t1 = _last_trace(buyer, "buyer_to_seller1")
    b2 = buyer.respond(observation["conversation_history_seller2"], _for_counterparty(observation, "seller2"))
    t2 = _last_trace(buyer, "buyer_to_seller2")
    h1 = _with_action(observation["conversation_history_seller1"], "buyer", b1, observation.get("current_round", 0))
    h2 = _with_action(observation["conversation_history_seller2"], "buyer", b2, observation.get("current_round", 0))
    s1 = agents["seller1"].respond(h1, observation)
    s2 = agents["seller2"].respond(h2, observation)
    return (
        {
            "buyer_action_seller1": b1,
            "buyer_action_seller2": b2,
            "seller1_action": s1,
            "seller2_action": s2,
        },
        [t for t in [t1, t2] if t],
    )


def _act_mb1s(agents: Dict[str, Any], observation: Dict[str, Any]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    b1 = agents["buyer1"].respond(observation["conversation_history_buyer1"], _for_counterparty(observation, "seller"))
    t1 = _last_trace(agents["buyer1"], "buyer1_to_seller")
    b2 = agents["buyer2"].respond(observation["conversation_history_buyer2"], _for_counterparty(observation, "seller"))
    t2 = _last_trace(agents["buyer2"], "buyer2_to_seller")
    h1 = _with_action(observation["conversation_history_buyer1"], "buyer", b1, observation.get("current_round", 0))
    h2 = _with_action(observation["conversation_history_buyer2"], "buyer", b2, observation.get("current_round", 0))
    s1 = agents["seller"].respond(h1, observation)
    s2 = agents["seller"].respond(h2, observation)
    return (
        {
            "buyer1_action": b1,
            "buyer2_action": b2,
            "seller_action_buyer1": s1,
            "seller_action_buyer2": s2,
        },
        [t for t in [t1, t2] if t],
    )


def _act_mbms(agents: Dict[str, Any], observation: Dict[str, Any]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    b1s1 = agents["buyer1"].respond(observation["conversation_history_b1s1"], _for_counterparty(observation, "seller1"))
    t1 = _last_trace(agents["buyer1"], "buyer1_to_seller1")
    b1s2 = agents["buyer1"].respond(observation["conversation_history_b1s2"], _for_counterparty(observation, "seller2"))
    t2 = _last_trace(agents["buyer1"], "buyer1_to_seller2")
    b2s1 = agents["buyer2"].respond(observation["conversation_history_b2s1"], _for_counterparty(observation, "seller1"))
    t3 = _last_trace(agents["buyer2"], "buyer2_to_seller1")
    b2s2 = agents["buyer2"].respond(observation["conversation_history_b2s2"], _for_counterparty(observation, "seller2"))
    t4 = _last_trace(agents["buyer2"], "buyer2_to_seller2")

    r = observation.get("current_round", 0)
    h_b1s1 = _with_action(observation["conversation_history_b1s1"], "buyer", b1s1, r)
    h_b1s2 = _with_action(observation["conversation_history_b1s2"], "buyer", b1s2, r)
    h_b2s1 = _with_action(observation["conversation_history_b2s1"], "buyer", b2s1, r)
    h_b2s2 = _with_action(observation["conversation_history_b2s2"], "buyer", b2s2, r)
    s1b1 = agents["seller1"].respond(h_b1s1, observation)
    s1b2 = agents["seller1"].respond(h_b2s1, observation)
    s2b1 = agents["seller2"].respond(h_b1s2, observation)
    s2b2 = agents["seller2"].respond(h_b2s2, observation)
    return (
        {
            "buyer1_action_seller1": b1s1,
            "buyer1_action_seller2": b1s2,
            "buyer2_action_seller1": b2s1,
            "buyer2_action_seller2": b2s2,
            "seller1_action_buyer1": s1b1,
            "seller1_action_buyer2": s1b2,
            "seller2_action_buyer1": s2b1,
            "seller2_action_buyer2": s2b2,
        },
        [t for t in [t1, t2, t3, t4] if t],
    )


def _last_trace(agent: Any, edge: str) -> Optional[Dict[str, Any]]:
    traces = getattr(agent, "traces", None)
    if not traces:
        return None
    trace = traces[-1]
    if hasattr(trace, "to_dict"):
        payload = trace.to_dict()
    else:
        payload = _jsonable(trace)
    payload["edge"] = edge
    return payload


def _for_counterparty(observation: Dict[str, Any], counterparty_id: str) -> Dict[str, Any]:
    """Attach an explicit edge id without mutating the environment observation."""

    state = dict(observation)
    state["_framework_counterparty_id"] = counterparty_id
    return state


def _with_action(history: List[Dict[str, Any]], role: str, content: str, round_id: int) -> List[Dict[str, Any]]:
    updated = list(history)
    if content:
        updated.append({"role": role, "content": content, "round": round_id})
    return updated


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if r.get("status") != "error"]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in valid:
        grouped.setdefault(f"{record.get('suite')}::{record.get('buyer_variant')}", []).append(record)
    return {
        "records": len(records),
        "valid_records": len(valid),
        "failures": len(records) - len(valid),
        "by_suite_variant": {key: _summarize_group(rows) for key, rows in sorted(grouped.items())},
    }


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "deal_rate": _mean(bool(r.get("success")) for r in rows),
        "avg_buyer_score": _mean(r.get("buyer_score") for r in rows),
        "avg_seller_score": _mean(r.get("seller_score") for r in rows),
        "avg_global_score": _mean(r.get("global_score") for r in rows),
        "avg_total_reward": _mean(r.get("total_reward") for r in rows),
        "avg_rounds": _mean(r.get("total_rounds") for r in rows),
        "termination_reasons": _counts(r.get("termination_reason") or r.get("status") for r in rows),
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float, bool))]
    if not cleaned:
        return None
    return round(statistics.mean(cleaned), 6)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "suite",
        "buyer_variant",
        "repeat_id",
        "status",
        "success",
        "termination_reason",
        "selected_buyer",
        "selected_seller",
        "final_deal_price",
        "buyer_score",
        "seller_score",
        "global_score",
        "total_rounds",
        "elapsed_time",
        "error_type",
    ]
    return {key: record.get(key) for key in keys if key in record}


def _status_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _task_name(suite: str) -> str:
    return {
        "1bms": "Task1_parallel_two_seller_negotiation",
        "mb1s": "Task1_parallel_two_buyer_negotiation",
        "mbms": "Task1_parallel_two_buyer_two_seller_negotiation",
    }[suite]


def _episode_key(suite: str, variant: str, repeat_id: int, buyer_model: str, seller_model: str) -> str:
    return json.dumps([suite, variant, repeat_id, buyer_model, seller_model], ensure_ascii=False)


def _load_completed(path: Path) -> set[str]:
    completed = set()
    for record in _read_jsonl(path):
        if record.get("status") == "error":
            continue
        completed.add(
            _episode_key(
                str(record.get("suite")),
                str(record.get("buyer_variant")),
                int(record.get("repeat_id", 0)),
                str(record.get("buyer_model_spec")),
                str(record.get("seller_model_spec")),
            )
        )
    return completed


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _jsonable_config(config: MultiAgentRunConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonable(v) for v in value]
        return str(value)
