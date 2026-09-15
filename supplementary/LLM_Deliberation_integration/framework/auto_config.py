from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_AUTO_CONFIG = {
    "acceptance_gate": 0.55,
    "acceptance_bias": 0.05,
    "alignment_weight": 0.78,
    "flexibility_weight": 0.12,
    "threshold_weight": 0.10,
    "adversarial_weight": 0.10,
    "final_coalition_weight": 0.62,
    "final_history_weight": 0.23,
    "final_self_weight": 0.15,
    "discussion_coalition_weight": 0.45,
    "discussion_self_weight": 0.25,
    "discussion_history_weight": 0.20,
    "discussion_info_weight": 0.10,
}

RANGES = {
    "acceptance_gate": (0.30, 0.90),
    "acceptance_bias": (-0.20, 0.30),
    "alignment_weight": (0.20, 1.20),
    "flexibility_weight": (0.00, 0.40),
    "threshold_weight": (0.00, 0.40),
    "adversarial_weight": (0.00, 0.40),
    **{key: (0.0, 1.0) for key in DEFAULT_AUTO_CONFIG if key.endswith("_weight") and key not in {
        "alignment_weight", "flexibility_weight", "threshold_weight", "adversarial_weight"
    }},
}


def validate_auto_config(value: dict[str, Any]) -> dict[str, float]:
    if set(value) != set(DEFAULT_AUTO_CONFIG):
        raise ValueError(f"Auto config keys must exactly match {sorted(DEFAULT_AUTO_CONFIG)}")
    result = {}
    for key, bounds in RANGES.items():
        number = float(value[key])
        if not bounds[0] <= number <= bounds[1]:
            raise ValueError(f"{key}={number} outside {bounds}")
        result[key] = number
    for prefix in ("final", "discussion"):
        keys = [key for key in result if key.startswith(prefix) and key.endswith("_weight")]
        if abs(sum(result[key] for key in keys) - 1.0) > 1e-6:
            raise ValueError(f"{prefix} weights must sum to 1: {keys}")
    return result


def load_auto_config() -> dict[str, float]:
    path = os.environ.get("ASTRA_AUTO_PLANNER_CONFIG")
    if not path:
        return validate_auto_config(dict(DEFAULT_AUTO_CONFIG))
    return validate_auto_config(json.loads(Path(path).read_text()))
