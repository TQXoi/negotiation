from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from CaSiNo_Env.environment.casino import ITEMS


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def paired_t_statistic(diffs: List[float]) -> float:
    if len(diffs) < 2:
        return 0.0
    avg = mean(diffs)
    var = sum((x - avg) ** 2 for x in diffs) / (len(diffs) - 1)
    if var <= 0:
        return 0.0
    return avg / (math.sqrt(var) / math.sqrt(len(diffs)))


def _true_high_issue(row: Dict[str, Any]) -> Optional[str]:
    scenario = row.get("scenario_context") or {}
    p2_values = scenario.get("p2_values") or {}
    values = {item: p2_values.get(item) for item in ITEMS if isinstance(p2_values.get(item), (int, float))}
    if not values:
        return None
    return max(values, key=values.get)


def _final_belief(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for turn in reversed(row.get("history") or []):
        trace = turn.get("trace") or {}
        belief = trace.get("belief")
        if isinstance(belief, dict) and isinstance(belief.get("issue_priority_posterior"), dict):
            return belief
        if isinstance(belief, dict) and isinstance(belief.get("partner_issue_priority_posterior"), dict):
            converted = dict(belief)
            converted["issue_priority_posterior"] = belief["partner_issue_priority_posterior"]
            return converted
    return None


def _belief_accuracy_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = []
    for row in rows:
        true_high = _true_high_issue(row)
        belief = _final_belief(row)
        if not true_high or not belief:
            continue
        posterior = belief.get("issue_priority_posterior") or {}
        probs = {item: float(posterior.get(item, 0.0)) for item in ITEMS}
        if not any(probs.values()):
            continue
        pred_high = max(probs, key=probs.get)
        entropy = belief.get("entropy")
        if not isinstance(entropy, (int, float)):
            entropy = -sum(p * math.log(max(p, 1e-8)) for p in probs.values())
        evaluated.append(
            {
                "correct": pred_high == true_high,
                "true_high_prob": probs.get(true_high, 0.0),
                "entropy": float(entropy),
                "confidence_gap": max(probs.values()) - min(probs.values()),
            }
        )
    n = len(evaluated)
    return {
        "belief_eval_n": n,
        "belief_top1_accuracy": round(mean(item["correct"] for item in evaluated), 6) if n else None,
        "belief_true_high_prob": round(mean(item["true_high_prob"] for item in evaluated), 6) if n else None,
        "belief_entropy": round(mean(item["entropy"] for item in evaluated), 6) if n else None,
        "belief_confidence_gap": round(mean(item["confidence_gap"] for item in evaluated), 6) if n else None,
    }


def summarize_episodes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    agreements = [row for row in rows if row.get("status") == "agreement"]
    walkaways = [row for row in rows if row.get("status") == "walk_away"]
    p1_all = [float(row.get("p1_score") or 0.0) for row in rows]
    p2_all = [float(row.get("p2_score") or 0.0) for row in rows]
    p1_agreement = [float(row.get("p1_score") or 0.0) for row in agreements]
    p2_agreement = [float(row.get("p2_score") or 0.0) for row in agreements]
    diffs = [float(row.get("p1_score") or 0.0) - float(row.get("p2_score") or 0.0) for row in agreements]
    summary = {
        "n": n,
        "agreements": len(agreements),
        "walk_aways": len(walkaways),
        "agreement_rate": round(len(agreements) / n, 6) if n else 0.0,
        "walk_away_rate": round(len(walkaways) / n, 6) if n else 0.0,
        "avg_score_all_p1": round(mean(p1_all), 6),
        "avg_score_all_p2": round(mean(p2_all), 6),
        "avg_score_agreement_p1": round(mean(p1_agreement), 6),
        "avg_score_agreement_p2": round(mean(p2_agreement), 6),
        "t_statistic_agreement_p1_minus_p2": round(paired_t_statistic(diffs), 6),
    }
    summary.update(_belief_accuracy_summary(rows))
    return summary
