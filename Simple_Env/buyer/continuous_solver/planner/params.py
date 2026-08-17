"""Small, learnable parameter set for continual EV planners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class PlannerParams:
    """Concession schedule parameters optimized by low-cost CEM.

    These are deliberately interpretable. CEM can tune them without training
    the LLM, and the resulting policy can be inspected like a normal planner.
    """

    first_anchor_ratio: float = 0.45
    early_concession_rate: float = 0.055
    late_concession_rate: float = 0.095
    p50_low_mult: float = 0.92
    p50_mid_mult: float = 1.00
    p50_high_mult: float = 1.04
    mean_mult: float = 1.00
    seller_discount_early: float = 0.76
    seller_discount_late: float = 0.86
    min_increment_ratio: float = 0.020
    accept_reward_threshold: float = 0.18
    accept_seller_ratio_threshold: float = 0.86
    lowball_penalty_weight: float = 0.20
    quit_penalty: float = 0.25
    future_value_weight: float = 0.55

    def clipped(self) -> "PlannerParams":
        return PlannerParams(
            first_anchor_ratio=_clip(self.first_anchor_ratio, 0.25, 0.70),
            early_concession_rate=_clip(self.early_concession_rate, 0.015, 0.120),
            late_concession_rate=_clip(self.late_concession_rate, 0.030, 0.180),
            p50_low_mult=_clip(self.p50_low_mult, 0.75, 1.05),
            p50_mid_mult=_clip(self.p50_mid_mult, 0.88, 1.12),
            p50_high_mult=_clip(self.p50_high_mult, 0.95, 1.22),
            mean_mult=_clip(self.mean_mult, 0.85, 1.16),
            seller_discount_early=_clip(self.seller_discount_early, 0.55, 0.92),
            seller_discount_late=_clip(self.seller_discount_late, 0.65, 0.98),
            min_increment_ratio=_clip(self.min_increment_ratio, 0.005, 0.060),
            accept_reward_threshold=_clip(self.accept_reward_threshold, 0.05, 0.45),
            accept_seller_ratio_threshold=_clip(self.accept_seller_ratio_threshold, 0.68, 0.98),
            lowball_penalty_weight=_clip(self.lowball_penalty_weight, 0.05, 0.60),
            quit_penalty=_clip(self.quit_penalty, 0.05, 0.70),
            future_value_weight=_clip(self.future_value_weight, 0.15, 0.90),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self.clipped())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerParams":
        valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in data.items() if k in valid}).clipped()

    @classmethod
    def from_json(cls, path: str | Path) -> "PlannerParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "params" in data and isinstance(data["params"], dict):
            data = data["params"]
        elif "best_params" in data and isinstance(data["best_params"], dict):
            data = data["best_params"]
        elif "contextual_policy" in data and isinstance(data["contextual_policy"], dict):
            data = data["contextual_policy"].get("base_params", {})
        elif "base_params" in data and isinstance(data["base_params"], dict):
            data = data["base_params"]
        return cls.from_dict(data)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
