from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_universal_framework_runs.py"
SPEC = importlib.util.spec_from_file_location("analyze_universal_framework_runs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(scenario: str, rollout: int, reward: float, deal: bool) -> dict:
    return {
        "scenario": {"item_id": scenario},
        "rollout": rollout,
        "reward": reward,
        "status": "deal" if deal else "no_deal",
        "mi": True,
        "transcript": [],
    }


def test_cluster_paired_averages_rollouts_before_bootstrap() -> None:
    reference = [
        row("a", 0, 1.0, True),
        row("a", 1, 1.0, True),
        row("a", 2, 1.0, True),
        row("b", 0, 0.0, False),
    ]
    comparator = [
        row("a", 0, 0.0, False),
        row("a", 1, 0.0, False),
        row("a", 2, 0.0, False),
        row("b", 0, 1.0, True),
    ]

    result = MODULE.cluster_paired(reference, comparator, samples=1000)

    # Episode-level mean would be +0.5. Equal-weighted scenario clusters are 0.
    assert result["scenario_clusters"] == 2
    assert result["paired_episodes"] == 4
    assert result["mean_reference_reward_delta"] == 0.0
    assert result["mean_reference_agreement_rate_delta"] == 0.0
    assert result["rollouts_per_cluster_min"] == 1
    assert result["rollouts_per_cluster_max"] == 3
