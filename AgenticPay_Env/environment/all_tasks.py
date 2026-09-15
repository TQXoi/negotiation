"""Generic AgenticPay example runner over all task families.

Unlike `multi_agent.py`, this module does not reconstruct environment loops.
It imports the original scripts under `agenticpay/examples/*/Task*.py`,
monkeypatches their BuyerAgent/SellerAgent/model factories, runs `main(...)`,
and then copies the original per-task summary into a unified JSONL.

This is the safest path to cover the complete AgenticPay benchmark because the
example scripts encode many task-specific product, contract, and multi-agent
settings.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import re
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORKSPACE = Path(__file__).resolve().parents[2]
AGENTICPAY_ROOT = WORKSPACE / "benchmarks" / "AgenticPay"
EXAMPLES_ROOT = AGENTICPAY_ROOT / "agenticpay" / "examples"
RESULTS_ROOT = AGENTICPAY_ROOT / "agenticpay" / "results"
for path in [WORKSPACE, AGENTICPAY_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AgenticPay_Env.buyer.variants import parse_buyer_variants
from experiments.agenticpay_qwen_eval import set_seed
from experiments.model_clients import make_model_client
from experiments.run_agenticpay_single28_framework import parse_framework_traces, patch_module


TASK_FAMILIES = [
    "single_buyer_product_seller",
    "only_multi_products",
    "only_multi_seller",
    "only_multi_buyer",
    "multi_products_multi_seller",
    "multi_buyer_multi_seller",
    "multi_buyer_multi_products",
    "multi_buyer_multi_products_multi_seller",
]


@dataclass
class AllTasksRunConfig:
    buyer_model: str
    seller_model: str
    buyer_model_alias: str
    seller_model_alias: str
    output_dir: Path
    buyer_variants: List[str]
    task_suites: List[str]
    seller_variant: str = "native"
    seed: int = 0
    max_new_tokens: int = 1024
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    planner_checkpoint: Optional[str] = None
    tasks: str = "all"
    limit: Optional[int] = None
    resume: bool = True
    focal_buyer_index: Optional[int] = None


def run_all_tasks(config: AllTasksRunConfig, *, dry_run: bool = False) -> Dict[str, Any]:
    # The upstream task scripts rely on global Python/NumPy/Torch RNG state.
    # Reset it once per run so migrations and later paired comparisons are
    # reproducible instead of silently ignoring AgenticPay_Env.eval --seed.
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    task_paths = filter_task_paths(
        selected_task_paths(config.task_suites), config.tasks
    )
    if config.limit is not None:
        task_paths = task_paths[: config.limit]
    config.buyer_variants = parse_buyer_variants(",".join(config.buyer_variants))
    payload = {
        "config": _jsonable_config(config),
        "num_tasks": len(task_paths),
        "tasks": [str(path.relative_to(EXAMPLES_ROOT)) for path in task_paths],
    }
    (config.output_dir / "agenticpay_all_tasks_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        return {"dry_run": True, **payload}

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
    out_jsonl = config.output_dir / "task_results.jsonl"
    logs_dir = config.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(out_jsonl) if config.resume else []
    completed = _completed_keys(records)

    with out_jsonl.open("a", encoding="utf-8") as handle:
        for variant in config.buyer_variants:
            for idx, task_path in enumerate(task_paths):
                rel = str(task_path.relative_to(EXAMPLES_ROOT))
                key = _record_key(
                    variant,
                    rel,
                    config.buyer_model,
                    config.seller_model,
                    config.seller_variant,
                    config.focal_buyer_index,
                )
                if key in completed:
                    continue
                start_time = time.time()
                log_buffer = StringIO()
                try:
                    module = _import_task_module(task_path, variant)
                    trajectory_capture: Dict[str, Any] = {"rounds": [], "env": {}}
                    patch_module(
                        module,
                        client=buyer_client,
                        variant=variant,
                        seller_variant=config.seller_variant,
                        max_tokens=config.max_new_tokens,
                        model_alias=config.buyer_model_alias,
                        buyer_client=buyer_client,
                        seller_client=seller_client,
                        seller_model_alias=config.seller_model_alias,
                        planner_checkpoint=config.planner_checkpoint,
                        trajectory_capture=trajectory_capture,
                        focal_buyer_index=config.focal_buyer_index,
                    )
                    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                        module.main(model_name=config.buyer_model_alias)
                    buyer_patch_audit = dict(
                        getattr(module, "_negotiation_buyer_patch_audit", {})
                    )
                    if config.focal_buyer_index is not None and (
                        buyer_patch_audit.get("variant_buyer_instances") != 1
                    ):
                        raise RuntimeError(
                            "Focal-only buyer replacement audit failed: "
                            f"{buyer_patch_audit}"
                        )
                    summary_path = _latest_summary_after(task_path.parent.name, start_time)
                    summary_data = _normalize_upstream_summary(
                        _read_json(summary_path) if summary_path else {}
                    )
                    focal_metrics = _focal_buyer_metrics(
                        summary_data, config.focal_buyer_index
                    )
                    record = {
                        **summary_data,
                        **focal_metrics,
                        "task_path": rel,
                        "task_family": task_path.parent.name,
                        "task_index": idx,
                        "buyer_variant": variant,
                        "seller_variant": config.seller_variant,
                        "buyer_model_spec": config.buyer_model,
                        "seller_model_spec": config.seller_model,
                        "buyer_model_alias": config.buyer_model_alias,
                        "seller_model_alias": config.seller_model_alias,
                        "focal_buyer_index": config.focal_buyer_index,
                        "buyer_population_policy": (
                            "focal_only"
                            if config.focal_buyer_index is not None
                            else "all_buyers"
                        ),
                        "buyer_patch_audit": buyer_patch_audit,
                        "source_summary_path": str(summary_path) if summary_path else None,
                        "framework_traces": parse_framework_traces(log_buffer.getvalue()),
                        "framework_trace_version": "stdout_FRAMEWORK_TRACE_v1",
                        "elapsed_time": round(time.time() - start_time, 3),
                        "error": None,
                    }
                    # Observation-only instrumentation.  This mirrors the
                    # single28 runner and gives later offline training an exact
                    # public action/response sequence without changing the
                    # benchmark environment, seller, scorer, or chosen action.
                    for key, value in (trajectory_capture.get("env") or {}).items():
                        if value is not None and record.get(key) is None:
                            record[key] = value
                    if trajectory_capture.get("rounds"):
                        record["rounds"] = trajectory_capture["rounds"]
                        record["trajectory_capture_version"] = (
                            "agenticpay_all_tasks_env_step_v1"
                        )
                except Exception as exc:
                    record = {
                        "task_path": rel,
                        "task_family": task_path.parent.name,
                        "task_index": idx,
                        "buyer_variant": variant,
                        "seller_variant": config.seller_variant,
                        "buyer_model_spec": config.buyer_model,
                        "seller_model_spec": config.seller_model,
                        "buyer_model_alias": config.buyer_model_alias,
                        "seller_model_alias": config.seller_model_alias,
                        "focal_buyer_index": config.focal_buyer_index,
                        "buyer_population_policy": (
                            "focal_only"
                            if config.focal_buyer_index is not None
                            else "all_buyers"
                        ),
                        "status": "error",
                        "success": False,
                        "error_type": exc.__class__.__name__,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "elapsed_time": round(time.time() - start_time, 3),
                    }
                log_path = logs_dir / variant / task_path.parent.name / f"{task_path.stem}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(log_buffer.getvalue(), encoding="utf-8")
                record["log_path"] = str(log_path)
                records.append(record)
                completed.add(key)
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                _write_summary(config.output_dir / "summary.json", records)
                print(json.dumps({"task_result": _compact(record)}, ensure_ascii=False), flush=True)

    summary = summarize(records)
    return {"summary": str(config.output_dir / "summary.json"), "summary_data": summary, **payload}


def selected_task_paths(task_suites: List[str]) -> List[Path]:
    suites = TASK_FAMILIES if not task_suites or task_suites == ["all"] else task_suites
    unknown = sorted(set(suites) - set(TASK_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown AgenticPay task suites: {unknown}. Available: {TASK_FAMILIES}")
    paths: List[Path] = []
    for suite in suites:
        paths.extend(
            path
            for path in (EXAMPLES_ROOT / suite).glob("Task*.py")
            if not path.name.endswith("_example.py")
        )
    return sorted(paths, key=lambda p: (TASK_FAMILIES.index(p.parent.name), _task_sort_key(p)))


def filter_task_paths(paths: List[Path], raw: str) -> List[Path]:
    """Filter task paths by exact stem or ``TaskN`` alias.

    The filter is applied after suite selection.  It exists for inexpensive
    paired development gates; ``tasks=all`` remains the benchmark default.
    """

    if not raw or raw.strip().lower() == "all":
        return paths
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    selected: List[Path] = []
    for path in paths:
        number_match = re.match(r"Task(\d+)", path.name)
        aliases = {path.stem, path.name, str(path.relative_to(EXAMPLES_ROOT))}
        if number_match:
            aliases.add(f"Task{number_match.group(1)}")
        if aliases & wanted:
            selected.append(path)
    return selected


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if r.get("status") != "error"]
    return {
        "records": len(records),
        "valid_records": len(valid),
        "failures": len(records) - len(valid),
        "overall": _summarize_group(valid),
        "by_task_family": _grouped_summary(valid, "task_family"),
        "by_buyer_variant": _grouped_summary(valid, "buyer_variant"),
        "by_task_family_and_variant": _grouped_summary(valid, "task_family", "buyer_variant"),
        "failure_examples": [r for r in records if r.get("status") == "error"][:20],
    }


def _normalize_upstream_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Expose comparable top-level metrics for upstream nested summaries.

    `only_multi_products` reports one result per product while the other
    AgenticPay families already provide episode-level fields.  Preserve the
    original `product_results`, and add transparent macro score / additive
    reward fields only when the corresponding top-level value is absent.
    """

    output = dict(summary)
    product_rows = [
        row
        for row in output.get("product_results") or []
        if isinstance(row, dict)
    ]
    if not product_rows:
        return output
    for key in ("buyer_score", "seller_score", "global_score"):
        values = [float(row[key]) for row in product_rows if row.get(key) is not None]
        if output.get(key) is None and values:
            output[key] = statistics.fmean(values)
    for key in ("buyer_reward", "seller_reward", "rounds"):
        values = [float(row[key]) for row in product_rows if row.get(key) is not None]
        target = "total_rounds" if key == "rounds" else key
        if output.get(target) is None and values:
            output[target] = sum(values)
    output["score_aggregation"] = "macro_mean_over_product_results"
    output["reward_aggregation"] = "sum_over_product_results"
    return output


def _focal_buyer_metrics(
    summary: Dict[str, Any], focal_buyer_index: Optional[int]
) -> Dict[str, Any]:
    """Expose focal-buyer outcomes without redefining AgenticPay BuyerScore.

    Multi-buyer task summaries retain the official aggregate ``buyer_score``
    and additionally report one reward per buyer.  For causal focal-only
    evaluation, copy the selected buyer's native reward and selection event to
    explicit fields.  We deliberately do not synthesize a focal BuyerScore,
    because that would silently change the benchmark's scoring definition.
    """

    if focal_buyer_index is None:
        return {}
    reward = summary.get(f"buyer{focal_buyer_index}_reward")
    selected = summary.get("selected_buyer")
    selected_flag: Optional[bool]
    if selected is None:
        selected_flag = None
    else:
        match = re.search(r"\d+", str(selected))
        selected_flag = bool(
            match and int(match.group(0)) == focal_buyer_index
        )
    return {
        "focal_buyer_reward": reward,
        "focal_buyer_selected": selected_flag,
        "focal_buyer_max_price": summary.get(
            f"buyer{focal_buyer_index}_max_price"
        ),
    }


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "deal_rate": _mean(_success_value(r) for r in rows),
        "avg_buyer_score": _mean(r.get("buyer_score") for r in rows),
        "avg_seller_score": _mean(r.get("seller_score") for r in rows),
        "avg_global_score": _mean(r.get("global_score") for r in rows),
        "avg_buyer_reward": _mean(_first_present(r, ["buyer_reward", "buyer1_reward", "buyer_score"]) for r in rows),
        "avg_seller_reward": _mean(_first_present(r, ["seller_reward", "seller1_reward", "seller_score"]) for r in rows),
        "avg_rounds": _mean(r.get("total_rounds") for r in rows),
        "termination_reasons": _counts(r.get("termination_reason") or r.get("status") for r in rows),
    }


def _grouped_summary(rows: List[Dict[str, Any]], *keys: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = "::".join(str(row.get(k)) for k in keys)
        grouped.setdefault(key, []).append(row)
    return {key: _summarize_group(value) for key, value in sorted(grouped.items())}


def _success_value(record: Dict[str, Any]) -> bool:
    if isinstance(record.get("success"), bool):
        return bool(record["success"])
    status = str(record.get("status", "")).lower()
    return status in {"agreed", "success", "deal"}


def _first_present(record: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record.get(key)
    return None


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


def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "task_path",
        "buyer_variant",
        "status",
        "success",
        "termination_reason",
        "buyer_score",
        "seller_score",
        "global_score",
        "total_rounds",
        "elapsed_time",
        "error_type",
    ]
    return {key: record.get(key) for key in keys if key in record}


def _task_sort_key(path: Path) -> Tuple[int, str]:
    match = re.match(r"Task(\d+)", path.name)
    return (int(match.group(1)) if match else 9999, path.name)


def _import_task_module(path: Path, variant: str) -> Any:
    module_name = f"agenticpay_alltasks_{variant}_{path.parent.name}_{path.stem}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _latest_summary_after(task_family: str, start_time: float) -> Optional[Path]:
    root = RESULTS_ROOT / task_family
    candidates = [path for path in root.rglob("summary.json") if path.stat().st_mtime >= start_time - 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_summary(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_key(
    variant: str,
    task_path: str,
    buyer_model: str,
    seller_model: str,
    seller_variant: str,
    focal_buyer_index: Optional[int],
) -> str:
    return json.dumps(
        [
            variant,
            task_path,
            buyer_model,
            seller_model,
            seller_variant,
            focal_buyer_index,
        ],
        ensure_ascii=False,
    )


def _completed_keys(records: List[Dict[str, Any]]) -> set[str]:
    completed = set()
    for record in records:
        if record.get("status") == "error":
            continue
        completed.add(
            _record_key(
                str(record.get("buyer_variant")),
                str(record.get("task_path")),
                str(record.get("buyer_model_spec")),
                str(record.get("seller_model_spec")),
                str(record.get("seller_variant", "native")),
                record.get("focal_buyer_index"),
            )
        )
    return completed


def _jsonable_config(config: AllTasksRunConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload
