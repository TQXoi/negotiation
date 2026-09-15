from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable


def summarize_episodes(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = list(rows)
    if not episodes:
        return {"episodes": 0}
    return {
        "episodes": len(episodes),
        "agreement_rate_n_minus_1": round(mean(row["agreement_n_minus_1"] for row in episodes), 6),
        "unanimous_rate": round(mean(row["agreement_unanimous"] for row in episodes), 6),
        "any_success_rate": round(mean(row.get("any_success", False) for row in episodes), 6),
        "mean_p1_score": round(mean(row["scores"].get(row["p1"], 0) for row in episodes), 6),
        "mean_collective_score": round(mean(sum(row["scores"].values()) for row in episodes), 6),
        "parse_errors": sum(row["parse_errors"] for row in episodes),
        "support_count_distribution": dict(Counter(str(row["support_count"]) for row in episodes)),
    }
