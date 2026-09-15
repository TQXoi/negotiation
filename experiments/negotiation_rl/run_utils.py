"""Small helpers for per-experiment output directories."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: Path, experiment_name: str, run_dir: Optional[str] = None) -> Path:
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = root / path
    else:
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in experiment_name)
        path = root / "runs" / safe_name / utc_timestamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def namespace_to_jsonable(args: Namespace) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


AGENTICPAY_EVAL_FLAGS = {
    "deal_flag": "termination_reason == 'agreed'",
    "quality_success_flag": "termination_reason == 'agreed' and buyer_score >= min_acceptable_buyer_score",
    "utility_fields": ["buyer_score", "seller_score", "global_score", "agreed_price"],
    "reward_fields": ["env_reward", "step_buyer_reward", "step_seller_reward"],
    "efficiency_fields": ["round_count", "termination_reason"],
}
