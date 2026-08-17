from __future__ import annotations

import json

from framework import (
    BehavioralFrontierPlanner,
    CandidateAction,
    CanonicalOffer,
    CanonicalState,
    FEATURE_NAMES,
    OpponentBelief,
    TrainableDecisionBoundaryPlanner,
)
from Simple_Env.buyer.continuous_solver.identifiable import UtilityProfile
from Simple_Env.tools.train_decision_boundary_planner_v2 import (
    make_belief,
    make_candidates,
    make_state,
)
import random


def state() -> CanonicalState:
    return CanonicalState(
        session_id="test", environment_id="unit", self_id="buyer", role="buyer",
        counterparty_id="seller", turn=2, max_turns=6, issues=[], own_value_scale=100.0,
    )


def candidates() -> list[CandidateAction]:
    return [
        CandidateAction(
            candidate_id="offer", action_type="offer", counterparty_id="seller",
            offer=CanonicalOffer(price=40.0), own_utility=60.0,
            own_utility_normalized=0.6, opponent_value_proxy=0.4,
            base_acceptance=0.2, information_gain=0.8, feasibility_margin=0.6,
        ),
        CandidateAction(
            candidate_id="accept", action_type="accept", counterparty_id="seller",
            offer=CanonicalOffer(price=90.0), own_utility=10.0,
            own_utility_normalized=0.10, opponent_value_proxy=1.0,
            base_acceptance=1.0, feasibility_margin=0.10,
        ),
    ]


def write_accept_checkpoint(path) -> None:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("action_accept")] = 2.0
    path.write_text(json.dumps({
        "schema_version": "trainable_decision_boundary_planner_v2",
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "mean": [0.0] * len(FEATURE_NAMES),
            "std": [1.0] * len(FEATURE_NAMES),
        },
        "layers": [{"weight": [weights], "bias": [0.0]}],
        "base_residual_weight": 0.0,
    }))


def test_trainable_planner_crosses_accept_continue_boundary(tmp_path):
    checkpoint = tmp_path / "planner.json"
    write_accept_checkpoint(checkpoint)
    base = BehavioralFrontierPlanner().rank(state(), OpponentBelief("seller"), candidates())
    learned = TrainableDecisionBoundaryPlanner(checkpoint).rank(
        state(), OpponentBelief("seller"), candidates()
    )
    assert base[0].candidate.candidate_id == "offer"
    assert learned[0].candidate.candidate_id == "accept"
    assert learned[0].diagnostics["trainable_planner_v2"] is True


def test_checkpoint_feature_schema_is_strict(tmp_path):
    checkpoint = tmp_path / "planner.json"
    write_accept_checkpoint(checkpoint)
    payload = json.loads(checkpoint.read_text())
    payload["feature_names"] = payload["feature_names"][:-1]
    checkpoint.write_text(json.dumps(payload))
    try:
        TrainableDecisionBoundaryPlanner(checkpoint)
    except ValueError as exc:
        assert "feature schema" in str(exc)
    else:
        raise AssertionError("schema mismatch should fail closed")


def test_zero_evidence_prior_does_not_leak_hidden_profile():
    low = UtilityProfile("low", 0.30)
    high = UtilityProfile("high", 0.85)
    left = make_belief(low, turn=1, evidence=0, rng=random.Random(1))
    right = make_belief(high, turn=1, evidence=0, rng=random.Random(2))
    assert left.to_dict() == right.to_dict()


def test_first_turn_has_no_impossible_accept_candidate():
    profile = UtilityProfile("profile", 0.60)
    belief = make_belief(profile, turn=1, evidence=0, rng=random.Random(1))
    actions = make_candidates(make_state(profile, 0, 1), belief, random.Random(1))
    assert "accept" not in {candidate.action_type for candidate in actions}
