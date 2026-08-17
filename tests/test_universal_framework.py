from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
AGENTICPAY_ROOT = WORKSPACE / "benchmarks" / "AgenticPay"
for path in [WORKSPACE, AGENTICPAY_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AgenticPay_Env.buyer.universal_framework import AgenticPayAdapter
from Simple_Env.buyer.universal_framework import SimpleEnvAdapter
from experiments.external_comparisons.run_rlvr_negotiation_paper_eval import RLVRScenario
from framework import (
    ActiveFrontierPlanner,
    AspirationAwareFrontierPlanner,
    AspirationProbeFrontierPlanner,
    BehavioralFrontierPlanner,
    BeliefStore,
    BroadPriorBeliefStore,
    CensoredBehaviorBeliefUpdater,
    CensoredReservationAspirationBeliefUpdater,
    CenteredSemanticBeliefUpdater,
    DeadlineAwareBeliefUsablePlanner,
    DominanceAndOptionPlanner,
    LookaheadBeliefUsablePlanner,
    LatentProfileMixtureBeliefUpdater,
    OpponentBelief,
    PosteriorIntegratedFrontierPlanner,
    ReservationAspirationMixtureBeliefUpdater,
    ShadowSemanticBeliefUpdater,
    StrategicCensoredBeliefUpdater,
    StrategicReadinessMixtureBeliefUpdater,
    UniversalNegotiationEngine,
    VerifiedSemanticBeliefUpdater,
    TerminalSafeLookaheadPlanner,
)
from framework.belief import StructuredBeliefUpdater


class MockClient:
    def generate(self, prompt, temperature=0.0, max_tokens=None, top_p=1.0, **kwargs):
        if "extract bounded opponent-preference evidence" in prompt:
            return json.dumps(
                {
                    "reservation_ratio_range": [0.55, 0.82],
                    "issue_signals": [],
                    "patience": 0.7,
                    "quit_risk": 0.1,
                    "evidence": ["seller made a formal counteroffer"],
                }
            )
        return "I can proceed promptly if we can align."


class UniversalFrameworkTests(unittest.TestCase):
    def test_belief_store_is_counterparty_scoped(self):
        store = BeliefStore()
        first = store.get("session", "seller1")
        second = store.get("session", "seller2")
        first.preference.reservation_ratio_mean = 0.8
        self.assertNotEqual(first.preference.reservation_ratio_mean, second.preference.reservation_ratio_mean)

    def test_simple_adapter_runs_same_core_and_locks_action(self):
        scenario = RLVRScenario(
            item_id="item-1",
            title="test",
            buyer_budget=100.0,
            seller_cost=55.0,
            reference_price=120.0,
            codename="test_item",
        )
        history = [
            {
                "role": "buyer",
                "round": 1,
                "message": "Thought: probe\nTalk: offer\nAction: [BUY] $45 (1x test_item)",
            },
            {
                "role": "seller",
                "round": 1,
                "message": "Thought: counter\nTalk: move\nAction: [SELL] $82 (1x test_item)",
            },
        ]
        adapter = SimpleEnvAdapter(scenario, history)
        state = adapter.state(round_id=2, max_turns=6, belief_mode="learned")
        engine = UniversalNegotiationEngine(semantic_client=MockClient(), language_client=MockClient())
        decision = engine.decide(state, adapter)
        self.assertTrue(decision.rendered_action.valid)
        self.assertTrue(decision.validation["action_lock_ok"])
        self.assertLessEqual(decision.rendered_action.price or 0.0, scenario.buyer_budget)
        self.assertGreater(decision.belief.evidence_count, 0)

    def test_agenticpay_contract_candidates_enforce_buyer_ir(self):
        context = {
            "max_price": 100.0,
            "product_info": {"name": "service"},
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"delivery_days": -2.0},
                    "discrete_weights": {"warranty": {"full": 10.0, "none": -30.0}},
                },
            },
        }
        seller_contract = {
            "price": 95.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = [
            {
                "role": "seller",
                "round": 1,
                "content": "<contract>\n" + json.dumps(seller_contract) + "\n</contract>",
            }
        ]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 2, "max_rounds": 10},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller"))
        self.assertNotIn("accept_exact_seller_offer", {item.candidate_id for item in candidates})
        self.assertTrue(all(item.own_utility >= 0 for item in candidates if item.action_type != "quit"))
        decision = UniversalNegotiationEngine(language_client=MockClient()).decide(state, adapter)
        self.assertTrue(decision.validation["ok"])
        self.assertTrue(decision.validation["buyer_ir_ok"])
        self.assertTrue(decision.validation["action_lock_ok"])

    def test_structured_evidence_dominates_semantic_confidence(self):
        scenario = RLVRScenario(
            item_id="item-2",
            title="test",
            buyer_budget=100.0,
            seller_cost=60.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $60 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [SELL] $75 (1x item)"},
        ]
        adapter = SimpleEnvAdapter(scenario, history)
        state = adapter.state(2, 6, "learned")
        belief = OpponentBelief("seller")
        StructuredBeliefUpdater().update(belief, state, state.observations)
        self.assertGreater(belief.preference.reliability, 0.1)
        self.assertGreaterEqual(belief.preference.reservation_ratio_high, belief.preference.reservation_ratio_low)
        self.assertLessEqual(belief.response_policy.reliability, 0.92)

    def test_v2_semantic_interval_updates_planner_mean_with_bounded_gate(self):
        scenario = RLVRScenario(
            item_id="item-v2-mean",
            title="test",
            buyer_budget=100.0,
            seller_cost=60.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $50 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v2").state(2, 6, "learned")
        belief = OpponentBelief("seller")
        StructuredBeliefUpdater().update(belief, state, state.observations)
        before = belief.preference.reservation_ratio_mean
        result = CenteredSemanticBeliefUpdater(MockClient()).update(
            belief, state, state.observations
        )
        self.assertTrue(result and result["parsed"])
        self.assertGreater(belief.preference.reservation_ratio_mean, before)
        self.assertLess(belief.preference.reservation_ratio_mean, 0.685)

    def test_v2_penalizes_a_directly_rejected_offer_region(self):
        scenario = RLVRScenario(
            item_id="item-v2-reject",
            title="test",
            buyer_budget=100.0,
            seller_cost=70.0,
            reference_price=120.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $50 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $50 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v2")
        state = adapter.state(5, 6, "learned")
        decision = UniversalNegotiationEngine(
            planner=DeadlineAwareBeliefUsablePlanner(),
        ).decide(state, adapter)
        selected_price = decision.selected.candidate.offer.price
        self.assertIsNotNone(selected_price)
        self.assertGreater(selected_price, 50.0)
        rejected_candidates = [
            item for item in decision.ranked_candidates
            if item.candidate.opponent_value_proxy <= 0.5
            and item.candidate.action_type == "offer"
        ]
        self.assertTrue(rejected_candidates)
        self.assertGreater(
            rejected_candidates[0].diagnostics["rejected_region_penalty"], 0.0
        )

    def test_v3_semantic_proposal_is_logged_without_mutating_posterior(self):
        scenario = RLVRScenario(
            item_id="item-v3-shadow",
            title="test",
            buyer_budget=100.0,
            seller_cost=60.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $50 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v3").state(2, 6, "learned")
        belief = OpponentBelief("seller")
        StructuredBeliefUpdater().update(belief, state, state.observations)
        before = belief.to_dict()
        result = ShadowSemanticBeliefUpdater(MockClient()).update(
            belief, state, state.observations
        )
        self.assertTrue(result and result["parsed"])
        self.assertFalse(result["applied_to_planner_belief"])
        self.assertEqual(before, belief.to_dict())

    def test_v5_redacts_private_thought_from_belief_observation(self):
        scenario = RLVRScenario(
            item_id="item-v5-private",
            title="test",
            buyer_budget=100.0,
            seller_cost=42.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $45 (1x item)"},
            {
                "role": "seller",
                "round": 1,
                "message": (
                    "Thought: My private cost is exactly $42.\n"
                    "Talk: Please improve the offer.\nAction: [REJECT]"
                ),
            },
        ]
        state = SimpleEnvAdapter(scenario, history, "v5").state(2, 6, "learned")
        self.assertEqual(len(state.observations), 1)
        self.assertNotIn("private cost", state.observations[0].text)
        self.assertIn("Please improve", state.observations[0].text)

    def test_v5_censored_rejection_is_not_a_hard_reservation_bound(self):
        scenario = RLVRScenario(
            item_id="item-v5-censored",
            title="test",
            buyer_budget=100.0,
            seller_cost=45.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $70 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [SELL] $90 (1x item)"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v5").state(2, 6, "learned")
        belief = OpponentBelief("seller")
        CensoredBehaviorBeliefUpdater().update(belief, state, state.observations)
        self.assertLess(belief.preference.reservation_ratio_low, 0.70)
        self.assertLess(belief.preference.reservation_ratio_mean, 0.70)
        self.assertEqual(belief.response_policy.minimum_observed_ask, 0.90)

    def test_v5_planner_records_feasibility_and_lookahead(self):
        scenario = RLVRScenario(
            item_id="item-v5-plan",
            title="test",
            buyer_budget=100.0,
            seller_cost=60.0,
            reference_price=110.0,
            codename="item",
        )
        adapter = SimpleEnvAdapter(scenario, [], "v5")
        state = adapter.state(1, 6, "learned")
        belief = OpponentBelief("seller")
        ranked = LookaheadBeliefUsablePlanner().rank(
            state, belief, adapter.candidates(state, belief)
        )
        offer_rows = [row for row in ranked if row.candidate.action_type == "offer"]
        self.assertTrue(offer_rows)
        self.assertTrue(all("utility_feasibility_probability" in row.diagnostics for row in offer_rows))
        self.assertTrue(all("lookahead_continuation" in row.diagnostics for row in offer_rows))

    def test_v5_1_belief_candidates_cannot_reverse_a_rejected_concession(self):
        scenario = RLVRScenario(
            item_id="item-v5-1-monotone",
            title="test",
            buyer_budget=100.0,
            seller_cost=75.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $65 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_1")
        state = adapter.state(2, 6, "learned")
        belief = OpponentBelief("seller")
        CensoredBehaviorBeliefUpdater().update(belief, state, state.observations)
        offers = [
            item.offer.price
            for item in adapter.candidates(state, belief)
            if item.action_type == "offer" and item.offer is not None
        ]
        self.assertTrue(offers)
        self.assertGreaterEqual(min(offers), 65.0)

    def test_v5_verified_semantic_overlap_applies_without_error(self):
        scenario = RLVRScenario(
            item_id="item-v5-verify",
            title="test",
            buyer_budget=100.0,
            seller_cost=60.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $50 (1x item)"},
            {"role": "seller", "round": 1, "message": "Talk: Too low.\nAction: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $55 (1x item)"},
            {"role": "seller", "round": 2, "message": "Talk: Improve it.\nAction: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v5_1").state(3, 6, "learned")
        belief = OpponentBelief("seller")
        CensoredBehaviorBeliefUpdater().update(belief, state, state.observations)
        result = VerifiedSemanticBeliefUpdater(MockClient()).update(
            belief, state, state.observations
        )
        self.assertTrue(result and result["parsed"])
        self.assertTrue(result["applied_to_planner_belief"])

    def test_v5_2_final_nonnegative_offer_dominates_quit(self):
        scenario = RLVRScenario(
            item_id="item-v5-2-terminal",
            title="test",
            buyer_budget=100.0,
            seller_cost=80.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 5, "message": "Action: [BUY] $75 (1x item)"},
            {"role": "seller", "round": 5, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_2")
        state = adapter.state(6, 6, "learned")
        belief = OpponentBelief("seller")
        CensoredBehaviorBeliefUpdater().update(belief, state, state.observations)
        ranked = TerminalSafeLookaheadPlanner().rank(state, belief, adapter.candidates(state, belief))
        self.assertEqual(ranked[0].candidate.action_type, "offer")
        self.assertTrue(ranked[0].diagnostics["terminal_dominance_applied"])

    def test_v5_3_nonnegative_offer_dominates_quit_before_final(self):
        scenario = RLVRScenario(
            item_id="item-v5-3-dominance",
            title="test",
            buyer_budget=100.0,
            seller_cost=80.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 4, "message": "Action: [BUY] $75 (1x item)"},
            {"role": "seller", "round": 4, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_3")
        state = adapter.state(5, 6, "learned")
        belief = OpponentBelief("seller")
        CensoredBehaviorBeliefUpdater().update(belief, state, state.observations)
        ranked = DominanceAndOptionPlanner().rank(state, belief, adapter.candidates(state, belief))
        self.assertEqual(ranked[0].candidate.action_type, "offer")
        self.assertTrue(ranked[0].diagnostics["offer_dominance_floor_applied"])

    def test_v5_4_strategic_reject_does_not_move_utility_interval(self):
        scenario = RLVRScenario(
            item_id="item-v5-4-strategic",
            title="test",
            buyer_budget=100.0,
            seller_cost=20.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $55 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_4")
        state = adapter.state(2, 6, "learned")
        belief = BroadPriorBeliefStore().get("session", "seller")
        before = (
            belief.preference.reservation_ratio_low,
            belief.preference.reservation_ratio_mean,
            belief.preference.reservation_ratio_high,
        )
        StrategicCensoredBeliefUpdater().update(belief, state, state.observations)
        after = (
            belief.preference.reservation_ratio_low,
            belief.preference.reservation_ratio_mean,
            belief.preference.reservation_ratio_high,
        )
        self.assertEqual(before, after)
        self.assertAlmostEqual(belief.response_policy.accept_beta[2], 1.35)

    def test_v5_5_frontier_penalizes_repeating_a_formally_rejected_offer(self):
        scenario = RLVRScenario(
            item_id="item-v5-5-frontier",
            title="test",
            buyer_budget=100.0,
            seller_cost=70.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $55 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $62 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_5")
        state = adapter.state(4, 6, "learned")
        belief = BroadPriorBeliefStore().get("session", "seller")
        StrategicCensoredBeliefUpdater().update(belief, state, state.observations)
        utility_before = (
            belief.preference.reservation_ratio_low,
            belief.preference.reservation_ratio_mean,
            belief.preference.reservation_ratio_high,
        )
        ranked = BehavioralFrontierPlanner().rank(
            state, belief, adapter.candidates(state, belief)
        )
        selected = ranked[0]
        self.assertEqual(selected.candidate.action_type, "offer")
        self.assertGreater(selected.candidate.opponent_value_proxy, 0.62)
        rejected_rows = [
            row for row in ranked
            if row.candidate.action_type == "offer"
            and row.candidate.opponent_value_proxy <= 0.62 + 1e-9
        ]
        self.assertTrue(rejected_rows)
        self.assertTrue(all(row.diagnostics["behavioral_frontier_penalty"] > 0 for row in rejected_rows))
        self.assertEqual(
            utility_before,
            (
                belief.preference.reservation_ratio_low,
                belief.preference.reservation_ratio_mean,
                belief.preference.reservation_ratio_high,
            ),
        )

    def test_v5_6_active_probe_separation_grows_near_deadline(self):
        scenario = RLVRScenario(
            item_id="item-v5-6-active",
            title="test",
            buyer_budget=100.0,
            seller_cost=82.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $55 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $62 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
        ]
        adapter = SimpleEnvAdapter(scenario, history, "v5_6")
        state = adapter.state(4, 6, "learned")
        belief = BroadPriorBeliefStore().get("session", "seller")
        StrategicCensoredBeliefUpdater().update(belief, state, state.observations)
        ranked = ActiveFrontierPlanner().rank(
            state, belief, adapter.candidates(state, belief)
        )
        selected = ranked[0]
        required = selected.diagnostics["active_probe_minimum_proxy"]
        self.assertEqual(selected.candidate.action_type, "offer")
        self.assertGreaterEqual(selected.candidate.opponent_value_proxy, required - 1e-9)
        self.assertGreater(required, belief.response_policy.rejected_compatibility_max + 0.07)

    def test_v5_7_mixture_updates_utility_softly_from_rejections(self):
        scenario = RLVRScenario(
            item_id="item-v5-7-mixture",
            title="test",
            buyer_budget=100.0,
            seller_cost=80.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $55 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $65 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v5_7").state(3, 6, "learned")
        belief = BroadPriorBeliefStore().get("session", "seller")
        before = belief.preference.reservation_ratio_mean
        LatentProfileMixtureBeliefUpdater().update(belief, state, state.observations)
        self.assertAlmostEqual(sum(belief.latent_profile_weights.values()), 1.0)
        self.assertGreater(belief.preference.reservation_ratio_mean, before)
        below_rejected_mass = sum(
            weight
            for profile_id, weight in belief.latent_profile_weights.items()
            if float(profile_id.split("_")[1]) < 0.65
        )
        self.assertGreater(below_rejected_mass, 0.0)
        self.assertLessEqual(belief.preference.reservation_ratio_low, 0.65)
        self.assertGreater(belief.preference.reservation_ratio_high, 0.65)
        self.assertIsNotNone(belief.latent_profile_entropy_bits)

    def test_v5_8_planner_uses_exact_latent_posterior_cdf(self):
        scenario = RLVRScenario(
            item_id="item-v5-8-cdf",
            title="test",
            buyer_budget=100.0,
            seller_cost=70.0,
            reference_price=110.0,
            codename="item",
        )
        adapter = SimpleEnvAdapter(scenario, [], "v5_8")
        state = adapter.state(1, 6, "learned")
        belief = OpponentBelief("seller")
        belief.preference.reservation_ratio_low = 0.2
        belief.preference.reservation_ratio_mean = 0.5
        belief.preference.reservation_ratio_high = 0.8
        belief.preference.reliability = 1.0
        belief.latent_profile_weights = {"low": 0.5, "high": 0.5}
        belief.latent_profile_reservation_ratios = {"low": 0.2, "high": 0.8}
        ranked = PosteriorIntegratedFrontierPlanner().rank(
            state, belief, adapter.candidates(state, belief)
        )
        middle = min(
            (row for row in ranked if row.candidate.action_type == "offer"),
            key=lambda row: abs(row.candidate.opponent_value_proxy - 0.5),
        )
        self.assertEqual(
            middle.diagnostics["utility_feasibility_source"],
            "latent_profile_posterior_cdf",
        )
        self.assertAlmostEqual(
            middle.diagnostics["utility_feasibility_probability"], 0.5
        )

    def test_v5_9_strategic_readiness_prevents_utility_over_attribution(self):
        scenario = RLVRScenario(
            item_id="item-v5-9-readiness",
            title="test",
            buyer_budget=100.0,
            seller_cost=45.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $70 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $75 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v5_9").state(3, 6, "learned")
        base = BroadPriorBeliefStore().get("base", "seller")
        strategic = BroadPriorBeliefStore().get("strategic", "seller")
        LatentProfileMixtureBeliefUpdater().update(base, state, state.observations)
        StrategicReadinessMixtureBeliefUpdater().update(
            strategic, state, state.observations
        )
        self.assertLess(
            strategic.preference.reservation_ratio_mean,
            base.preference.reservation_ratio_mean,
        )
        self.assertTrue(
            any("readiness" in key for key in strategic.latent_profile_weights)
        )
        self.assertAlmostEqual(sum(strategic.latent_profile_weights.values()), 1.0)

    def test_v6_reservation_aspiration_factorization_handles_hardball_rejections(self):
        scenario = RLVRScenario(
            item_id="item-v6-aspiration",
            title="test",
            buyer_budget=100.0,
            seller_cost=35.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $70 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $78 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 3, "message": "Action: [BUY] $84 (1x item)"},
            {"role": "seller", "round": 3, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v6_0").state(4, 6, "learned")
        readiness = BroadPriorBeliefStore().get("readiness", "seller")
        aspiration = BroadPriorBeliefStore().get("aspiration", "seller")
        StrategicReadinessMixtureBeliefUpdater().update(
            readiness, state, state.observations
        )
        ReservationAspirationMixtureBeliefUpdater().update(
            aspiration, state, state.observations
        )
        self.assertLess(
            aspiration.preference.reservation_ratio_mean,
            readiness.preference.reservation_ratio_mean,
        )
        self.assertGreater(
            aspiration.response_policy.aspiration_ratio_mean,
            aspiration.preference.reservation_ratio_mean,
        )
        self.assertTrue(aspiration.latent_profile_aspiration_margins)
        self.assertAlmostEqual(sum(aspiration.latent_profile_weights.values()), 1.0)

    def test_v6_planner_consumes_aspiration_as_response_not_utility(self):
        scenario = RLVRScenario(
            item_id="item-v6-planner",
            title="test",
            buyer_budget=100.0,
            seller_cost=40.0,
            reference_price=110.0,
            codename="item",
        )
        adapter = SimpleEnvAdapter(scenario, [], "v6_0")
        state = adapter.state(2, 6, "learned")
        belief = OpponentBelief("seller")
        belief.preference.reservation_ratio_low = 0.30
        belief.preference.reservation_ratio_mean = 0.40
        belief.preference.reservation_ratio_high = 0.55
        belief.preference.reliability = 0.90
        belief.response_policy.aspiration_ratio_mean = 0.85
        belief.response_policy.aspiration_reliability = 0.90
        candidate = min(
            (row for row in adapter.candidates(state, belief) if row.action_type == "offer"),
            key=lambda row: abs(row.opponent_value_proxy - 0.70),
        )
        base_p, _ = BehavioralFrontierPlanner()._acceptance_probability(
            belief, candidate
        )
        aspiration_p, diagnostics = AspirationAwareFrontierPlanner()._acceptance_probability(
            belief, candidate
        )
        self.assertLess(aspiration_p, base_p)
        self.assertEqual(diagnostics["aspiration_ratio_mean"], 0.85)
        # The durable reservation estimate remains unchanged by planner use.
        self.assertEqual(belief.preference.reservation_ratio_mean, 0.40)

    def test_v6_1_censors_rejection_and_uses_counter_as_soft_upper_bound(self):
        scenario = RLVRScenario(
            item_id="item-v6-1-censor",
            title="test",
            buyer_budget=100.0,
            seller_cost=35.0,
            reference_price=110.0,
            codename="item",
        )
        history = [
            {"role": "buyer", "round": 1, "message": "Action: [BUY] $70 (1x item)"},
            {"role": "seller", "round": 1, "message": "Action: [SELL] $85 (1x item)"},
            {"role": "buyer", "round": 2, "message": "Action: [BUY] $78 (1x item)"},
            {"role": "seller", "round": 2, "message": "Action: [REJECT]"},
            {"role": "buyer", "round": 3, "message": "Action: [BUY] $82 (1x item)"},
            {"role": "seller", "round": 3, "message": "Action: [REJECT]"},
        ]
        state = SimpleEnvAdapter(scenario, history, "v6_1").state(4, 6, "learned")
        uncensored = BroadPriorBeliefStore().get("uncensored", "seller")
        censored = BroadPriorBeliefStore().get("censored", "seller")
        ReservationAspirationMixtureBeliefUpdater().update(
            uncensored, state, state.observations
        )
        CensoredReservationAspirationBeliefUpdater().update(
            censored, state, state.observations
        )
        self.assertLess(
            censored.preference.reservation_ratio_mean,
            uncensored.preference.reservation_ratio_mean,
        )
        likelihood_rows = [
            row for row in censored.evidence
            if row.get("source") == "censored_reservation_aspiration_likelihood"
        ]
        self.assertTrue(likelihood_rows)
        self.assertEqual(likelihood_rows[0]["formal_ask_upper_bound"], 0.85)

    def test_v6_1_probe_bonus_targets_uncertain_aspiration_boundary(self):
        scenario = RLVRScenario(
            item_id="item-v6-1-probe",
            title="test",
            buyer_budget=100.0,
            seller_cost=40.0,
            reference_price=110.0,
            codename="item",
        )
        adapter = SimpleEnvAdapter(scenario, [], "v6_1")
        state = adapter.state(2, 6, "learned")
        belief = OpponentBelief("seller")
        belief.direct_response_count = 1
        belief.response_policy.aspiration_ratio_mean = 0.70
        belief.response_policy.aspiration_ratio_low = 0.45
        belief.response_policy.aspiration_ratio_high = 0.90
        belief.response_policy.aspiration_reliability = 0.35
        ranked = AspirationProbeFrontierPlanner().rank(
            state, belief, adapter.candidates(state, belief)
        )
        offers = [row for row in ranked if row.candidate.action_type == "offer"]
        most_rewarded = max(offers, key=lambda row: row.diagnostics["aspiration_probe_bonus"])
        closest = min(
            offers,
            key=lambda row: abs(row.candidate.opponent_value_proxy - 0.70),
        )
        self.assertEqual(most_rewarded.candidate.candidate_id, closest.candidate.candidate_id)
        self.assertGreater(most_rewarded.diagnostics["aspiration_probe_bonus"], 0.0)


if __name__ == "__main__":
    unittest.main()
