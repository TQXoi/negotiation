"""Planner interfaces and candidate action diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from experiments.agenticpay_framework.schemas import StrategicPlan


@dataclass
class CandidateAction:
    """A candidate buyer action scored under the current belief state."""

    action_type: str
    price: Optional[float]
    p_accept: float = 0.0
    p_quit: float = 0.0
    reward_if_deal: float = 0.0
    future_value: float = 0.0
    ev: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerResult:
    """Planner output plus all candidates for analysis."""

    plan: StrategicPlan
    candidates: List[CandidateAction] = field(default_factory=list)
    selected: Optional[CandidateAction] = None
    planner_type: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "candidates": [x.to_dict() for x in self.candidates],
            "selected": self.selected.to_dict() if self.selected else None,
            "planner_type": self.planner_type,
            "diagnostics": self.diagnostics,
        }
