"""Single-buyer/product/seller AgenticPay runner wrapper.

This module deliberately delegates episode execution to
`experiments/run_agenticpay_single28_framework.py`. That runner already preserves
the upstream AgenticPay example scripts and contains the fixed-seller buyer
metric correction used in earlier experiments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from AgenticPay_Env.buyer.variants import parse_buyer_variants
from AgenticPay_Env.seller.variants import validate_seller_variant


WORKSPACE = Path(__file__).resolve().parents[2]
LEGACY_RUNNER = WORKSPACE / "experiments" / "run_agenticpay_single28_framework.py"


@dataclass
class Single28RunConfig:
    model: str
    model_alias: str
    output_dir: Path
    buyer_variants: List[str]
    buyer_model: Optional[str] = None
    seller_model: Optional[str] = None
    buyer_model_alias: Optional[str] = None
    seller_model_alias: Optional[str] = None
    seller_variant: str = "native"
    tasks: str = "all"
    seed: int = 0
    max_new_tokens: int = 1024
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    gpu_id: Optional[str] = None
    require_cuda: bool = False
    limit: Optional[int] = None
    planner_checkpoint: Optional[str] = None
    resume: bool = True
    resume_from_jsonl: Optional[Path] = None
    experiment_name: str = "agenticpay_env_single28"


def build_command(config: Single28RunConfig) -> List[str]:
    seller_variant = validate_seller_variant(config.seller_variant)
    command = [
        sys.executable,
        str(LEGACY_RUNNER),
        "--model",
        config.model,
        "--model-alias",
        config.model_alias,
        "--buyer-variants",
        ",".join(config.buyer_variants),
        "--seller-variant",
        seller_variant,
        "--tasks",
        config.tasks,
        "--seed",
        str(config.seed),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--torch-dtype",
        config.torch_dtype,
        "--device-map",
        config.device_map,
        "--experiment-name",
        config.experiment_name,
        "--run-dir",
        str(config.output_dir),
    ]
    if config.cache_dir:
        command.extend(["--cache-dir", config.cache_dir])
    if config.buyer_model:
        command.extend(["--buyer-model", config.buyer_model])
    if config.seller_model:
        command.extend(["--seller-model", config.seller_model])
    if config.buyer_model_alias:
        command.extend(["--buyer-model-alias", config.buyer_model_alias])
    if config.seller_model_alias:
        command.extend(["--seller-model-alias", config.seller_model_alias])
    if config.gpu_id is not None:
        command.extend(["--gpu-id", str(config.gpu_id)])
    if config.require_cuda:
        command.append("--require-cuda")
    if config.limit is not None:
        command.extend(["--limit", str(config.limit)])
    if config.planner_checkpoint:
        command.extend(["--planner-checkpoint", config.planner_checkpoint])
    resume_jsonl = config.resume_from_jsonl or (
        config.output_dir / "task_results.jsonl"
    )
    if config.resume and resume_jsonl.exists():
        command.extend(["--resume-from-jsonl", str(resume_jsonl)])
    return command


def run_single28(config: Single28RunConfig, *, dry_run: bool = False) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.buyer_variants = parse_buyer_variants(",".join(config.buyer_variants))
    command = build_command(config)
    payload = {"config": _jsonable_config(config), "cmd": command}
    (config.output_dir / "agenticpay_env_command.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        return {"dry_run": True, **payload}
    completed = subprocess.run(command, cwd=str(WORKSPACE), check=False)
    if completed.returncode != 0:
        return {"returncode": completed.returncode, **payload}
    summary_path = config.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"returncode": completed.returncode, "summary": str(summary_path), "summary_data": summary, **payload}


def _jsonable_config(config: Single28RunConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    if config.resume_from_jsonl is not None:
        payload["resume_from_jsonl"] = str(config.resume_from_jsonl)
    return payload
