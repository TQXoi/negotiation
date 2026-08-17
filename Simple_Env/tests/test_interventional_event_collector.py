from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "collect_interventional_verified_events.py"
SPEC = importlib.util.spec_from_file_location("interventional_collector", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scenario_split_is_disjoint_and_exhaustive() -> None:
    ids = [f"s{index}" for index in range(12)]
    split = MODULE.split_scenarios(ids, seed=7, train_n=7, calibration_n=2)
    assert len(split["train"]) == 7
    assert len(split["calibration"]) == 2
    assert len(split["test"]) == 3
    assert not (split["train"] & split["calibration"])
    assert not (split["train"] & split["test"])
    assert set().union(*split.values()) == set(ids)


def test_controlled_rounds_are_deterministic_per_scenario() -> None:
    first = MODULE.controlled_rounds("scenario-a", 6, 17)
    second = MODULE.controlled_rounds("scenario-a", 6, 17)
    assert first == second
    assert sorted(first) == [1, 2, 3, 4, 5, 6]
