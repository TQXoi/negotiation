from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ITEMS = ("Food", "Water", "Firewood")
POOL = {item: 3 for item in ITEMS}
VALUE_RANK_TO_POINTS = {"Low": 3, "Medium": 4, "High": 5}
POINTS_TO_VALUE_RANK = {v: k for k, v in VALUE_RANK_TO_POINTS.items()}


@dataclass(frozen=True)
class CasinoScenario:
    scenario_id: str
    dialogue_id: int
    split: str
    p1_id: str
    p2_id: str
    pool: Dict[str, int]
    p1_values: Dict[str, int]
    p2_values: Dict[str, int]
    p1_reasons: Dict[str, str]
    p2_reasons: Dict[str, str]
    human_points_p1: Optional[float]
    human_points_p2: Optional[float]
    raw: Dict[str, Any]

    @property
    def max_score(self) -> int:
        return score_allocation(self.pool, self.p1_values)

    def public_context_for_p1(self) -> str:
        lines = [
            "You and the other participant are camping neighbors negotiating how to split shared supplies.",
            "There are exactly 3 units of each issue: Food, Water, and Firewood.",
            "Your private priority values per unit are:",
        ]
        for item in ITEMS:
            lines.append(f"- {item}: {self.p1_values[item]} points/unit")
        lines.append("Your private reasons:")
        for item in ITEMS:
            reason = self.p1_reasons.get(item, "")
            if reason:
                lines.append(f"- {item}: {reason}")
        lines.append("The other participant has their own private priorities. Do not assume they match yours.")
        return "\n".join(lines)

    def public_context_for_p2(self) -> str:
        lines = [
            "You and the other participant are camping neighbors negotiating how to split shared supplies.",
            "There are exactly 3 units of each issue: Food, Water, and Firewood.",
            "Your private priority values per unit are:",
        ]
        for item in ITEMS:
            lines.append(f"- {item}: {self.p2_values[item]} points/unit")
        lines.append("Your private reasons:")
        for item in ITEMS:
            reason = self.p2_reasons.get(item, "")
            if reason:
                lines.append(f"- {item}: {reason}")
        lines.append("The other participant has their own private priorities. Do not assume they match yours.")
        return "\n".join(lines)


def values_from_participant(info: Mapping[str, Any]) -> Dict[str, int]:
    value2issue = info.get("value2issue") or {}
    out = {item: 0 for item in ITEMS}
    for rank, issue in value2issue.items():
        if issue in out:
            out[issue] = VALUE_RANK_TO_POINTS.get(rank, 0)
    missing = [item for item, value in out.items() if value <= 0]
    if missing:
        raise ValueError(f"Missing CaSiNo issue values for {missing}: {value2issue}")
    return out


def reasons_from_participant(info: Mapping[str, Any]) -> Dict[str, str]:
    value2issue = info.get("value2issue") or {}
    value2reason = info.get("value2reason") or {}
    reasons = {item: "" for item in ITEMS}
    for rank, issue in value2issue.items():
        if issue in reasons:
            reasons[issue] = str(value2reason.get(rank, "")).strip()
    return reasons


def load_casino_split(path: str | Path, split: Optional[str] = None, limit: Optional[int] = None) -> List[CasinoScenario]:
    path = Path(path)
    with path.open() as f:
        records = json.load(f)
    scenarios = []
    split_name = split or path.stem.replace("casino_", "")
    for idx, record in enumerate(records):
        participant_info = record["participant_info"]
        ids = sorted(participant_info)
        p1_id, p2_id = ids[0], ids[1]
        p1_info, p2_info = participant_info[p1_id], participant_info[p2_id]
        scenarios.append(
            CasinoScenario(
                scenario_id=f"{split_name}_{idx:04d}_dialogue_{record.get('dialogue_id', idx)}",
                dialogue_id=int(record.get("dialogue_id", idx)),
                split=split_name,
                p1_id=p1_id,
                p2_id=p2_id,
                pool=dict(POOL),
                p1_values=values_from_participant(p1_info),
                p2_values=values_from_participant(p2_info),
                p1_reasons=reasons_from_participant(p1_info),
                p2_reasons=reasons_from_participant(p2_info),
                human_points_p1=(p1_info.get("outcomes") or {}).get("points_scored"),
                human_points_p2=(p2_info.get("outcomes") or {}).get("points_scored"),
                raw=record,
            )
        )
        if limit is not None and len(scenarios) >= limit:
            break
    return scenarios


def allocation_from_any(data: Mapping[str, Any] | None) -> Optional[Dict[str, int]]:
    if not data:
        return None
    out: Dict[str, int] = {}
    for item in ITEMS:
        raw = data.get(item, data.get(item.lower()))
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value < 0 or value > POOL[item]:
            return None
        out[item] = value
    return out


def complement_allocation(allocation: Mapping[str, int], pool: Mapping[str, int] = POOL) -> Dict[str, int]:
    return {item: int(pool[item]) - int(allocation[item]) for item in ITEMS}


def score_allocation(allocation: Mapping[str, int], values: Mapping[str, int]) -> int:
    return sum(int(allocation.get(item, 0)) * int(values[item]) for item in ITEMS)


def enumerate_allocations(pool: Mapping[str, int] = POOL) -> Iterable[Dict[str, int]]:
    for food in range(int(pool["Food"]) + 1):
        for water in range(int(pool["Water"]) + 1):
            for firewood in range(int(pool["Firewood"]) + 1):
                yield {"Food": food, "Water": water, "Firewood": firewood}


def normalize_allocation(allocation: Mapping[str, int]) -> Dict[str, int]:
    return {item: int(allocation.get(item, 0)) for item in ITEMS}


def scenario_to_json(scenario: CasinoScenario) -> Dict[str, Any]:
    return asdict(scenario)
