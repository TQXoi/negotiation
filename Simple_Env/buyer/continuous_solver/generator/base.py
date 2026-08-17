"""Generator interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import ParsedAction


@dataclass
class GenerationResult:
    """Final natural language realization and parsed action."""

    raw: str
    parsed_action: ParsedAction
    validator: Dict[str, Any]
    generator_type: str
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parsed_action"] = self.parsed_action.to_dict()
        return data
