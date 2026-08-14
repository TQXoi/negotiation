from arena_integration.belief import ReservationBelief
from arena_integration.framework_agent import BeliefPlannerAgent
import json


def test_buyer_offer_tightens_wtp_lower_bound():
    belief = ReservationBelief.prior("buyer", 100)
    assert belief.update("<newly proposed trade> Player RED Gives X: 1 | Player BLUE Gives ZUP: 42 </newly proposed trade>") == 42
    assert belief.lower == 42


def test_seller_offer_tightens_cost_upper_bound():
    belief = ReservationBelief.prior("seller", 100)
    belief.update("Player RED Gives X: 1 | Player BLUE Gives ZUP: 65")
    assert belief.upper == 65


def test_framework_action_is_parser_compatible(monkeypatch):
    agent = BeliefPlannerAgent(
        agent_name="Player RED",
        focal_role="seller",
        private_value=40,
        money_cap=100,
        model="mock",
        base_url="http://127.0.0.1:1/v1",
    )
    agent.init_agent("rules", "role")
    monkeypatch.setattr(agent, "_generate_message", lambda action, price: f"Offer {price}.")
    response = agent.step("")
    assert "<player answer> PROPOSAL </player answer>" in response
    assert "Player RED Gives X: 1 | Player BLUE Gives ZUP:" in response


def test_framework_state_is_json_serializable(monkeypatch):
    agent = BeliefPlannerAgent(
        agent_name="Player BLUE",
        focal_role="buyer",
        private_value=60,
        money_cap=100,
        model="mock",
        base_url="http://127.0.0.1:1/v1",
    )
    agent.init_agent("rules", "role")
    json.dumps(agent.get_state())
