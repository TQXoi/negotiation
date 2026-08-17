from __future__ import annotations

from framework import (
    AWR_FEATURE_NAMES,
    BehavioralFrontierPlanner,
    CandidateAction,
    CanonicalOffer,
    CanonicalState,
    ConservativeAWRCheckpoint,
    ConservativeAWRPlanner,
    IssueSpec,
    OpponentBelief,
    ScoredCandidate,
    awr_candidate_features,
)


def _state() -> CanonicalState:
    return CanonicalState(
        session_id="test",
        environment_id="unit",
        self_id="buyer",
        role="buyer",
        counterparty_id="seller",
        turn=3,
        max_turns=6,
        issues=[
            IssueSpec("price", "price"),
            IssueSpec("delivery", "continuous", domain={"min": 1, "max": 7}),
        ],
        own_value_scale=100.0,
        metadata={"repeated_opponent": True, "cross_session_progress": 0.5},
    )


def _candidates():
    return [
        CandidateAction(
            candidate_id="low",
            action_type="offer",
            counterparty_id="seller",
            offer=CanonicalOffer(price=50, continuous_terms={"delivery": 1}),
            own_utility=50,
            own_utility_normalized=0.50,
            opponent_value_proxy=0.50,
            base_acceptance=0.25,
            information_gain=0.60,
            feasibility_margin=0.50,
        ),
        CandidateAction(
            candidate_id="high",
            action_type="offer",
            counterparty_id="seller",
            offer=CanonicalOffer(price=65, continuous_terms={"delivery": 4}),
            own_utility=35,
            own_utility_normalized=0.35,
            opponent_value_proxy=0.70,
            base_acceptance=0.70,
            information_gain=0.25,
            feasibility_margin=0.35,
        ),
        CandidateAction(
            candidate_id="quit",
            action_type="quit",
            counterparty_id="seller",
            offer=None,
            own_utility=0,
            own_utility_normalized=0,
            opponent_value_proxy=0,
            base_acceptance=0,
        ),
    ]


def _checkpoint(bias: float) -> ConservativeAWRCheckpoint:
    layer = {
        "weight": [[0.0] * len(AWR_FEATURE_NAMES)],
        "bias": [bias],
    }
    return ConservativeAWRCheckpoint(
        feature_mean=(0.0,) * len(AWR_FEATURE_NAMES),
        feature_std=(1.0,) * len(AWR_FEATURE_NAMES),
        members=((layer,), (layer,), (layer,)),
        minimum_override_lcb=0.01,
        initial_override_lcb=0.02,
        ood_z_threshold=100.0,
    )


def test_cross_environment_features_include_multi_issue_and_regime_context():
    state = _state()
    belief = OpponentBelief("seller", direct_response_count=2, old_regime_weight=0.35, new_regime_weight=0.65)
    ranked = BehavioralFrontierPlanner().rank(state, belief, _candidates())
    features = awr_candidate_features(
        state,
        belief,
        ranked[0],
        base_winner=ranked[0],
        base_top_score=ranked[0].score,
        best_offer_utility=0.50,
    )
    assert len(features) == len(AWR_FEATURE_NAMES)
    indexed = dict(zip(AWR_FEATURE_NAMES, features))
    assert indexed["multi_issue"] == 1.0
    assert indexed["repeated_opponent"] == 1.0
    assert indexed["new_regime_weight"] == 0.65


def test_negative_lower_bound_abstains_to_exact_base_action():
    state, belief, candidates = _state(), OpponentBelief("seller", direct_response_count=2), _candidates()
    base_id = BehavioralFrontierPlanner().rank(state, belief, candidates)[0].candidate.candidate_id
    selected = ConservativeAWRPlanner(_checkpoint(-0.10)).rank(state, belief, candidates)[0]
    assert selected.candidate.candidate_id == base_id
    assert selected.diagnostics["awr_selected_override"] is False


def test_positive_lower_bound_can_override_after_direct_evidence():
    state, belief, candidates = _state(), OpponentBelief("seller", direct_response_count=2), _candidates()
    base_id = BehavioralFrontierPlanner().rank(state, belief, candidates)[0].candidate.candidate_id
    selected = ConservativeAWRPlanner(_checkpoint(0.10)).rank(state, belief, candidates)[0]
    assert selected.candidate.candidate_id != base_id
    assert selected.diagnostics["awr_selected_override"] is True


def test_direct_evidence_gate_abstains_from_high_lcb_prior_override():
    state, belief, candidates = _state(), OpponentBelief("seller", direct_response_count=0), _candidates()
    base_id = BehavioralFrontierPlanner().rank(state, belief, candidates)[0].candidate.candidate_id
    selected = ConservativeAWRPlanner(
        _checkpoint(0.10),
        require_direct_evidence=True,
    ).rank(state, belief, candidates)[0]
    assert selected.candidate.candidate_id == base_id
    assert selected.diagnostics["awr_selected_override"] is False
    assert selected.diagnostics["direct_evidence_gate_enabled"] is True
    assert selected.diagnostics["awr_abstain_reason"] == "no_direct_response"


def test_direct_evidence_gate_releases_after_observed_response():
    state, belief, candidates = _state(), OpponentBelief("seller", direct_response_count=1), _candidates()
    base_id = BehavioralFrontierPlanner().rank(state, belief, candidates)[0].candidate.candidate_id
    selected = ConservativeAWRPlanner(
        _checkpoint(0.10),
        require_direct_evidence=True,
    ).rank(state, belief, candidates)[0]
    assert selected.candidate.candidate_id != base_id
    assert selected.diagnostics["awr_selected_override"] is True
    assert selected.diagnostics["direct_evidence_gate_passed"] is True
    assert selected.diagnostics["awr_abstain_reason"] is None


def test_initial_proxy_jump_guard_blocks_unsafe_large_concession():
    state, belief, candidates = _state(), OpponentBelief("seller", direct_response_count=0), _candidates()
    base = BehavioralFrontierPlanner().rank(state, belief, candidates)[0]
    checkpoint = _checkpoint(0.10)
    checkpoint = ConservativeAWRCheckpoint(
        **{**checkpoint.__dict__, "max_initial_proxy_jump": 0.0}
    )
    selected = ConservativeAWRPlanner(checkpoint).rank(state, belief, candidates)[0]
    if selected.candidate.action_type == "offer":
        assert selected.candidate.opponent_value_proxy <= base.candidate.opponent_value_proxy


def test_safe_improvement_gate_protects_terminal_base_action():
    state = _state()
    belief = OpponentBelief("seller", direct_response_count=2)
    accept = CandidateAction(
        candidate_id="accept_now",
        action_type="accept",
        counterparty_id="seller",
        offer=CanonicalOffer(price=55),
        own_utility=45,
        own_utility_normalized=0.45,
        opponent_value_proxy=0.60,
        base_acceptance=1.0,
    )
    offer = _candidates()[0]

    class TerminalBase:
        def rank(self, _state, _belief, _candidates):
            return [
                ScoredCandidate(accept, 1.0, 0.0, 0.45, 0.0, 0.0, 0.8),
                ScoredCandidate(offer, 0.25, 0.0, 0.50, 0.0, 0.0, 0.79),
            ]

    selected = ConservativeAWRPlanner(
        _checkpoint(0.10),
        base_planner=TerminalBase(),
        safe_improvement_gate=True,
    ).rank(state, belief, [accept, offer])[0]
    assert selected.candidate.candidate_id == "accept_now"
    assert selected.diagnostics["awr_selected_override"] is False


def test_safe_improvement_gate_rejects_concession_without_acceptance_gain():
    state = _state()
    belief = OpponentBelief("seller", direct_response_count=2)
    base, concession = _candidates()[:2]
    concession.base_acceptance = base.base_acceptance

    class OfferBase:
        def rank(self, _state, _belief, _candidates):
            return [
                ScoredCandidate(base, base.base_acceptance, 0.0, 0.50, 0.0, 0.0, 0.8),
                ScoredCandidate(concession, concession.base_acceptance, 0.0, 0.35, 0.0, 0.0, 0.795),
            ]

    selected = ConservativeAWRPlanner(
        _checkpoint(0.10),
        base_planner=OfferBase(),
        safe_improvement_gate=True,
    ).rank(state, belief, [base, concession])[0]
    assert selected.candidate.candidate_id == "low"
    assert selected.diagnostics["awr_selected_override"] is False
