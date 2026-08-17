from __future__ import annotations

from framework.events import extract_verified_event, public_talk_and_action
from framework.belief import StrategicReadinessMixtureBeliefUpdater
from framework.schemas import (
    CanonicalOffer,
    CanonicalState,
    NegotiationObservation,
)


def state() -> CanonicalState:
    return CanonicalState(
        session_id="s1",
        environment_id="test",
        self_id="buyer",
        role="buyer",
        counterparty_id="seller",
        turn=2,
        max_turns=6,
        issues=[],
        own_value_scale=100.0,
    )


def test_private_thought_is_removed_and_claim_is_not_truth() -> None:
    event = extract_verified_event(
        state(),
        NegotiationObservation(
            observation_id="o1",
            turn=2,
            actor_id="seller",
            counterparty_id="seller",
            response_type="reject",
            response_to_offer=CanonicalOffer(price=55.0),
            text=(
                "Thought: My actual cost is 20.\n"
                "Talk: That is below my cost and this is my final offer.\n"
                "Action: [REJECT]"
            ),
        ),
    )
    assert "actual cost" not in event.public_text
    assert event.outcome == "counter"
    assert event.tested_compatibility == 0.55
    assert event.claims["cost_floor"]["present"] is True
    assert event.claims["cost_floor"]["truth_status"] == "unverified_claim"
    assert event.verification["claim_truth_used_as_label"] is False


def test_adapter_supplied_compatibility_supports_non_price_environments() -> None:
    event = extract_verified_event(
        state(),
        NegotiationObservation(
            observation_id="o2",
            turn=3,
            actor_id="seller",
            counterparty_id="seller",
            response_type="accept",
            response_to_offer=CanonicalOffer(discrete_terms={"delivery": "fast"}),
            text="Agreed.",
            metadata={"tested_compatibility": 0.73},
        ),
    )
    assert event.outcome == "accept"
    assert event.tested_compatibility == 0.73
    assert event.terminal is True


def test_thought_only_message_never_falls_back_to_private_text() -> None:
    assert public_talk_and_action("Thought: secret utility is 0.2") == ""


def test_default_calibration_parameters_preserve_v5_9_distribution() -> None:
    updater = StrategicReadinessMixtureBeliefUpdater()
    result = updater._response_distribution(0.70, 0.65, 0.78, 0.08, 0.25, 0.50, 0.50)
    # Old V5.9 equation: sigmoid(1) * .75, followed by the fixed quit/counter shares.
    assert abs(result["accept"] - 0.5482939339725037) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_calibration_artifact_loading_is_allowlisted(tmp_path) -> None:
    artifact = tmp_path / "params.json"
    artifact.write_text(
        '{"params":{"likelihood_tau":0.12,"readiness_terminal_fraction":0.4,"ignored":9},'
        '"likelihood_power":0.75}',
        encoding="utf-8",
    )
    updater = StrategicReadinessMixtureBeliefUpdater.from_calibration_json(artifact)
    assert updater.likelihood_tau == 0.12
    assert updater.readiness_terminal_fraction == 0.4
    assert updater.likelihood_power == 0.75
    assert not hasattr(updater, "ignored")
