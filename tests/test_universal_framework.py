from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
AGENTICPAY_ROOT = WORKSPACE / "benchmarks" / "AgenticPay"
# Pytest may pre-populate the parent transfer directory ahead of this clean
# repository. Remove duplicate entries and make the repository under test the
# authoritative source, otherwise a stale sibling ``AgenticPay_Env`` can be
# imported while collection still appears to come from this test file.
for path in [AGENTICPAY_ROOT, WORKSPACE]:
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from AgenticPay_Env.buyer.universal_framework import (
    AgenticPayAdapter,
    UniversalAgenticPayBuyerAgent,
)
from AgenticPay_Env.buyer.issue_classifier import OntologicalIssueClassifier
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
    CandidateAction,
    CanonicalOffer,
    CenteredSemanticBeliefUpdater,
    DeadlineAwareBeliefUsablePlanner,
    DominanceAndOptionPlanner,
    LookaheadBeliefUsablePlanner,
    LatentProfileMixtureBeliefUpdater,
    OpponentBelief,
    PosteriorIntegratedFrontierPlanner,
    ReservationAspirationMixtureBeliefUpdater,
    ShadowSemanticBeliefUpdater,
    ScoredCandidate,
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
    @staticmethod
    def _multi_seller_context():
        def contract_config(v_base, delivery_weight):
            return {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": v_base,
                    "continuous_weights": {"delivery_days": delivery_weight},
                    "discrete_weights": {
                        "warranty": {"full": 5.0, "none": 0.0}
                    },
                },
            }

        return {
            "max_price": 100.0,
            "num_sellers": 2,
            "product_info": {"name": "service"},
            "seller_contract_configs": {
                1: contract_config(100.0, -1.0),
                2: contract_config(110.0, -2.0),
            },
        }

    def test_multiseller_adapter_loads_selected_private_buyer_config(self):
        context = self._multi_seller_context()
        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={
                "current_round": 1,
                "max_rounds": 20,
                "_framework_selected_seller": 2,
                "_framework_available_seller_ids": [1, 2],
                "_framework_contract_config": context["seller_contract_configs"][2],
            },
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="multi-seller",
            counterparty_id="seller2",
            counterparty_count=2,
            action_consistent_renderer=True,
        )
        self.assertEqual(
            adapter.contract_config,
            context["seller_contract_configs"][2],
        )
        decision = UniversalNegotiationEngine(language_client=MockClient()).decide(
            adapter.state("learned"), adapter
        )
        parsed = adapter._offer_from_text(decision.rendered_action)
        self.assertIsNotNone(parsed)
        self.assertEqual(set(parsed.continuous_terms), {"delivery_days"})
        self.assertEqual(set(parsed.discrete_terms), {"warranty"})
        self.assertTrue(decision.validation["contract_tag_ok"])
        self.assertTrue(decision.validation["contract_complete_ok"])
        self.assertTrue(decision.validation["routing_ok"])

    def test_v56_accepts_repeated_public_buyer_ir_contract(self):
        context = self._multi_seller_context()
        seller_offer = {
            "price": 70.0,
            "continuous_terms": {"delivery_days": 5.0},
            "discrete_terms": {"warranty": "full"},
        }
        history = [
            {
                "role": "seller",
                "round": turn,
                "content": "<contract>" + json.dumps(seller_offer) + "</contract>",
            }
            for turn in (3, 4)
        ]
        adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 6, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v56-accept",
            counterparty_id="seller1",
            action_consistent_renderer=True,
            deadline_aware_settlement=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        accept = next(c for c in candidates if c.candidate_id == "accept_exact_seller_offer")
        quit_action = next(c for c in candidates if c.candidate_id == "quit")
        accept_row = ScoredCandidate(accept, 1.0, 0.0, 0.1, 0.0, 0.0, -0.1)
        quit_row = ScoredCandidate(quit_action, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        chosen, diagnostics = adapter.validate_selection(
            state,
            OpponentBelief("seller1"),
            [quit_row, accept_row],
            quit_row,
        )
        self.assertEqual(chosen.candidate.candidate_id, "accept_exact_seller_offer")
        self.assertEqual(diagnostics["reason"], "buyer_ir_public_seller_offer_accepted")
        self.assertTrue(diagnostics["overrode_selection"])

    def test_v56_replaces_stale_low_offer_with_buyer_ir_concession(self):
        context = self._multi_seller_context()
        buyer_offer = {
            "price": 60.0,
            "continuous_terms": {"delivery_days": 1.0},
            "discrete_terms": {"warranty": "full"},
        }
        seller_offer = {
            "price": 95.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = []
        for turn in (1, 2, 3):
            history.extend(
                [
                    {
                        "role": "buyer",
                        "round": turn,
                        "content": "<contract>" + json.dumps(buyer_offer) + "</contract>",
                    },
                    {
                        "role": "seller",
                        "round": turn,
                        "content": "<contract>" + json.dumps(seller_offer) + "</contract>",
                    },
                ]
            )
        adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 5, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v56-concede",
            counterparty_id="seller1",
            action_consistent_renderer=True,
            deadline_aware_settlement=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        offers = [c for c in candidates if c.action_type == "offer"]
        low = min(offers, key=lambda c: float(c.offer.price))
        rows = [
            ScoredCandidate(c, c.base_acceptance, 0.0, 0.0, 0.0, 0.0, 0.0)
            for c in candidates
        ]
        selected = next(row for row in rows if row.candidate is low)
        chosen, diagnostics = adapter.validate_selection(
            state,
            OpponentBelief("seller1"),
            rows,
            selected,
        )
        self.assertGreater(float(chosen.candidate.offer.price), float(low.offer.price))
        self.assertGreaterEqual(chosen.candidate.own_utility, 0.0)
        self.assertEqual(
            diagnostics["reason"],
            "stale_offer_replaced_by_highest_settlement_readiness",
        )

    def test_v60_accepts_strong_first_public_offer_before_it_is_withdrawn(self):
        context = self._multi_seller_context()
        seller_offer = {
            "price": 75.0,
            "continuous_terms": {"delivery_days": 5.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = [{
            "role": "seller",
            "round": 1,
            "content": "<contract>" + json.dumps(seller_offer) + "</contract>",
        }]
        adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v60-early-accept",
            counterparty_id="seller1",
            deadline_aware_settlement=True,
            public_offer_recovery=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        accept = next(c for c in candidates if c.action_type == "accept")
        alternative = max(
            (c for c in candidates if c.action_type == "offer"),
            key=lambda c: c.own_utility,
        )
        accept_row = ScoredCandidate(accept, 1.0, 0.0, 0.1, 0.0, 0.0, -1.0)
        alternative_row = ScoredCandidate(
            alternative, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0
        )
        chosen, diagnostics = adapter.validate_selection(
            state,
            OpponentBelief("seller1"),
            [alternative_row, accept_row],
            alternative_row,
        )
        self.assertEqual(chosen.candidate.action_type, "accept")
        self.assertTrue(diagnostics["strong_exact_accept_available"])
        self.assertEqual(
            diagnostics["reason"], "buyer_ir_public_seller_offer_accepted"
        )

        # The new boundary is V60-only: archived V56 behavior must remain
        # reproducible when public_offer_recovery is disabled.
        frozen_adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v56-frozen-early-offer",
            counterparty_id="seller1",
            deadline_aware_settlement=True,
        )
        frozen_state = frozen_adapter.state("learned")
        frozen_candidates = frozen_adapter.candidates(
            frozen_state, OpponentBelief("seller1")
        )
        frozen_accept = next(c for c in frozen_candidates if c.action_type == "accept")
        frozen_alternative = max(
            (c for c in frozen_candidates if c.action_type == "offer"),
            key=lambda c: c.own_utility,
        )
        frozen_accept_row = ScoredCandidate(
            frozen_accept, 1.0, 0.0, 0.1, 0.0, 0.0, -1.0
        )
        frozen_alternative_row = ScoredCandidate(
            frozen_alternative, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0
        )
        frozen_choice, frozen_diagnostics = frozen_adapter._deadline_aware_settlement_choice(
            frozen_state,
            [frozen_alternative_row, frozen_accept_row],
            frozen_alternative_row,
        )
        self.assertIsNone(frozen_choice)
        self.assertFalse(frozen_diagnostics["strong_exact_accept_available"])

    def test_v60_can_reoffer_withdrawn_public_buyer_ir_contract(self):
        context = self._multi_seller_context()
        good_offer = {
            "price": 75.0,
            "continuous_terms": {"delivery_days": 5.0},
            "discrete_terms": {"warranty": "none"},
        }
        withdrawn_offer = {
            "price": 99.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = [
            {
                "role": "seller",
                "round": 1,
                "content": "<contract>" + json.dumps(good_offer) + "</contract>",
            },
            {"role": "buyer", "round": 2, "content": "I would like better terms."},
            {
                "role": "seller",
                "round": 2,
                "content": "<contract>" + json.dumps(withdrawn_offer) + "</contract>",
            },
        ]
        adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v60-historical-reoffer",
            counterparty_id="seller1",
            deadline_aware_settlement=True,
            public_offer_recovery=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        reoffer = next(
            c for c in candidates
            if c.candidate_id == "best_public_buyer_ir_reoffer"
        )
        self.assertAlmostEqual(reoffer.offer.price, 75.0)
        quit_action = next(c for c in candidates if c.action_type == "quit")
        reoffer_row = ScoredCandidate(reoffer, 0.96, 0.0, 0.1, 0.0, 0.0, -0.1)
        quit_row = ScoredCandidate(quit_action, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        chosen, diagnostics = adapter.validate_selection(
            state,
            OpponentBelief("seller1"),
            [quit_row, reoffer_row],
            quit_row,
        )
        self.assertEqual(
            chosen.candidate.candidate_id, "best_public_buyer_ir_reoffer"
        )
        self.assertTrue(diagnostics["historical_reoffer_available"])

    def test_v60_does_not_early_accept_in_price_only_negotiation(self):
        seller_offer = {"price": 75.0}
        history = [{
            "role": "seller",
            "round": 1,
            "content": "### SELLER_PRICE($75) ###",
        }]
        adapter = AgenticPayAdapter(
            context={"max_price": 100.0},
            history=history,
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v60-price-only-frozen",
            counterparty_id="seller1",
            deadline_aware_settlement=True,
            public_offer_recovery=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        accept = next(c for c in candidates if c.action_type == "accept")
        alternative = max(
            (c for c in candidates if c.action_type == "offer"),
            key=lambda c: c.own_utility,
        )
        accept_row = ScoredCandidate(accept, 1.0, 0.0, 0.1, 0.0, 0.0, -1.0)
        alternative_row = ScoredCandidate(
            alternative, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0
        )
        choice, diagnostics = adapter._deadline_aware_settlement_choice(
            state, [alternative_row, accept_row], alternative_row
        )
        self.assertIsNone(choice)
        self.assertFalse(diagnostics["strong_exact_accept_available"])
        self.assertFalse(diagnostics["historical_reoffer_available"])

    def test_v57_finances_repeated_seller_price_with_minimal_term_repair(self):
        context = {
            "max_price": 10.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 10.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {"warranty": {"full": 2.0, "none": 0.0}},
                },
            },
        }
        seller_offer = {
            "price": 8.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = [
            {
                "role": "seller",
                "round": turn,
                "content": "<contract>" + json.dumps(seller_offer) + "</contract>",
            }
            for turn in (1, 2, 3)
        ]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 4, "max_rounds": 20},
            buyer_max_price=10.0,
            self_id="Buyer",
            session_id="v57-financed-settlement",
            counterparty_id="seller1",
            action_consistent_renderer=True,
            stagnation_trade_ledger_repair=True,
            deadline_aware_settlement=True,
        )
        state = adapter.state("learned")
        candidates = adapter.candidates(state, OpponentBelief("seller1"))
        repair = next(
            c for c in candidates
            if c.candidate_id == "stagnation_trade_ledger_repair"
        )
        quit_action = next(c for c in candidates if c.candidate_id == "quit")
        repair_row = ScoredCandidate(repair, 0.98, 0.0, 0.0, 0.0, 0.0, -0.1)
        quit_row = ScoredCandidate(quit_action, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        chosen, diagnostics = adapter.validate_selection(
            state,
            OpponentBelief("seller1"),
            [quit_row, repair_row],
            quit_row,
        )
        self.assertEqual(
            chosen.candidate.candidate_id,
            "stagnation_trade_ledger_repair",
        )
        self.assertAlmostEqual(chosen.candidate.offer.price, 8.0)
        self.assertGreaterEqual(chosen.candidate.own_utility, 0.5 - 1e-6)
        self.assertTrue(diagnostics["stagnation_repair_available"])

    def test_v59_rejection_extraction_ignores_ordinary_opening_counter(self):
        context = self._multi_seller_context()
        opening = {
            "price": 70.0,
            "continuous_terms": {"delivery_days": 1.0},
            "discrete_terms": {"warranty": "full"},
        }
        seller = {
            "price": 75.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        repaired = {
            "price": 75.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "full"},
        }
        history = [
            {"role": "buyer", "content": "<contract>" + json.dumps(opening) + "</contract>"},
            {"role": "seller", "content": "<contract>" + json.dumps(seller) + "</contract>"},
            {
                "role": "buyer",
                "content": (
                    "Your same complete package has remained unchanged. "
                    "I am matching its stated price and changing only the minimum terms.\n"
                    "<contract>" + json.dumps(repaired) + "</contract>"
                ),
            },
            {"role": "seller", "content": "<contract>" + json.dumps(seller) + "</contract>"},
        ]
        adapter = AgenticPayAdapter(
            context={**context, "contract_config": context["seller_contract_configs"][1]},
            history=history,
            current_state={"current_round": 5, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v59-scoped-rejection",
            counterparty_id="seller1",
        )
        self.assertEqual(
            adapter._stagnation_rejected_issue_fields(),
            {"discrete_terms.warranty"},
        )

    def test_multiseller_buyer_routes_scopes_history_and_emits_complete_contract(self):
        context = self._multi_seller_context()
        agent = UniversalAgenticPayBuyerAgent(
            model=MockClient(),
            buyer_max_price=100.0,
            action_consistent_renderer=True,
            terminal_accept_guard=True,
        )
        agent.initialize(context)

        # Initial tie is deterministic and satisfies the native routing API.
        first = agent.respond(
            [],
            {
                "num_sellers": 2,
                "instruction": "Choose exactly one seller and include selected_seller.",
                "current_round": 1,
                "max_rounds": 20,
                "conversation_history_seller1": [],
                "conversation_history_seller2": [],
            },
        )
        self.assertEqual(agent.last_selected_seller, 1)
        self.assertIn("<contract>", first)
        self.assertTrue(agent.traces[-1]["validation"]["routing_ok"])

        seller1_offer = {
            "price": 80.0,
            "continuous_terms": {"delivery_days": 5.0},
            "discrete_terms": {"warranty": "none"},
        }
        seller1_history = [
            {"role": "buyer", "content": first, "round": 1},
            {
                "role": "seller",
                "content": "<contract>" + json.dumps(seller1_offer) + "</contract>",
                "round": 1,
            },
        ]
        combined = [
            {**item, "thread_label": "Talk with Seller 1"}
            for item in seller1_history
        ]
        second = agent.respond(
            combined,
            {
                "num_sellers": 2,
                "instruction": "Choose exactly one seller and include selected_seller.",
                "current_round": 2,
                "max_rounds": 20,
                "conversation_history_seller1": seller1_history,
                "conversation_history_seller2": [],
                "seller_contract_seller1": seller1_offer,
            },
        )
        # The unobserved seller must be probed before utility-based lock-in.
        self.assertEqual(agent.last_selected_seller, 2)
        parsed = AgenticPayAdapter._offer_from_text(second)
        self.assertIsNotNone(parsed)
        self.assertEqual(set(parsed.continuous_terms), {"delivery_days"})
        self.assertEqual(set(parsed.discrete_terms), {"warranty"})
        self.assertEqual(agent.traces[-1]["counterparty_id"], "seller2")
        self.assertEqual(agent.traces[-1]["belief"]["evidence_count"], 0)

    def test_v60_retries_missing_contract_then_routes_by_public_buyer_utility(self):
        context = self._multi_seller_context()
        agent = UniversalAgenticPayBuyerAgent(
            model=MockClient(),
            buyer_max_price=100.0,
            resilient_multiseller_routing=True,
        )
        agent.initialize(context)
        seller1_offer = {
            "price": 80.0,
            "continuous_terms": {"delivery_days": 5.0},
            "discrete_terms": {"warranty": "none"},
        }
        seller1_history = [
            {"role": "buyer", "content": "first probe"},
            {
                "role": "seller",
                "content": "<contract>" + json.dumps(seller1_offer) + "</contract>",
            },
        ]
        seller2_history = [
            {"role": "buyer", "content": "first probe"},
            {"role": "seller", "content": "I can negotiate, but omitted the contract."},
        ]
        state = {
            "conversation_history_seller1": seller1_history,
            "conversation_history_seller2": seller2_history,
            "seller_contract_seller1": seller1_offer,
        }

        # Prose without a parseable contract gets one bounded retry instead
        # of causing permanent lock-in to Seller 1.
        self.assertEqual(agent._select_seller([1, 2], state), 2)
        self.assertEqual(
            agent.last_routing_diagnostics["policy"],
            "bounded_missing_contract_retry",
        )

        # Once the retry budget is exhausted, a malformed edge no longer
        # consumes the remaining negotiation horizon.
        state["conversation_history_seller2"] = seller2_history + [
            {"role": "buyer", "content": "second probe"},
            {"role": "seller", "content": "Still no structured contract."},
        ]
        self.assertEqual(agent._select_seller([1, 2], state), 1)
        self.assertEqual(
            agent.last_routing_diagnostics["policy"],
            "best_public_contract_by_buyer_utility",
        )

        # If Seller 2 later exposes a better contract, routing uses only the
        # buyer's private utility over that public offer.
        seller2_offer = {
            "price": 72.0,
            "continuous_terms": {"delivery_days": 4.0},
            "discrete_terms": {"warranty": "full"},
        }
        state["seller_contract_seller2"] = seller2_offer
        self.assertEqual(agent._select_seller([1, 2], state), 2)
        self.assertFalse(
            agent.last_routing_diagnostics["uses_seller_private_utility"]
        )

    def test_v60_price_only_router_uses_public_prices_not_contract_retry(self):
        agent = UniversalAgenticPayBuyerAgent(
            model=MockClient(),
            buyer_max_price=100.0,
            resilient_multiseller_routing=True,
        )
        agent.initialize({
            "max_price": 100.0,
            "num_sellers": 2,
            "product_info": {"name": "price-only item"},
        })
        state = {
            "conversation_history_seller1": [
                {"role": "buyer", "content": "offer to seller 1"}
            ],
            "conversation_history_seller2": [
                {"role": "buyer", "content": "first offer to seller 2"},
                {"role": "seller", "content": "counter at 70"},
                {"role": "buyer", "content": "second offer to seller 2"},
            ],
            "seller1_price": 80.0,
            "seller2_price": 70.0,
        }
        self.assertEqual(agent._select_seller([1, 2], state), 2)
        self.assertEqual(
            agent.last_routing_diagnostics["policy"],
            "best_public_contract_by_buyer_utility",
        )

    def test_multiseller_validator_repairs_price_only_wire_output(self):
        context = self._multi_seller_context()
        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={
                "current_round": 1,
                "max_rounds": 20,
                "_framework_selected_seller": 1,
                "_framework_available_seller_ids": [1, 2],
                "_framework_contract_config": context["seller_contract_configs"][1],
            },
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="validator",
            counterparty_id="seller1",
            counterparty_count=2,
            action_consistent_renderer=True,
        )
        state = adapter.state("learned")
        candidate = adapter.candidates(state, OpponentBelief("seller1"))[0]
        selected = ScoredCandidate(candidate, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        rendered, validation = adapter.validate_locked(
            state, selected, "Counteroffer.\n\n### BUYER_PRICE($50) ###"
        )
        self.assertIn("<contract>", rendered)
        self.assertTrue(validation["ok"])
        self.assertTrue(validation["contract_tag_ok"])
        self.assertTrue(validation["contract_complete_ok"])
        self.assertTrue(validation["routing_ok"])

    def test_v50_risk_confirmation_is_selected_at_most_once(self):
        agent = UniversalAgenticPayBuyerAgent.__new__(UniversalAgenticPayBuyerAgent)
        agent.risk_budgeted_semantic_confirmation = True
        agent.one_shot_risk_budgeted_confirmation = True
        agent.traces = []
        self.assertTrue(agent._risk_confirmation_enabled_this_turn())

        agent.traces.append({"selected_candidate_id": "offer_0"})
        self.assertTrue(agent._risk_confirmation_enabled_this_turn())

        agent.traces.append(
            {"selected_candidate_id": "risk_budgeted_semantic_confirmation"}
        )
        self.assertFalse(agent._risk_confirmation_enabled_this_turn())

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

    def test_agenticpay_robust_settlement_validator_replaces_contract_accept(self):
        context = {
            "max_price": 100.0,
            "product_info": {"name": "service"},
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {
                        "warranty": {"full": 5.0, "none": 0.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 70.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        history = [{
            "role": "seller",
            "round": 1,
            "content": "<contract>\n" + json.dumps(seller_contract) + "\n</contract>",
        }]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 2, "max_rounds": 10},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
            action_consistent_renderer=True,
            robust_contract_settlement_validator=True,
        )
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        by_id = {item.candidate_id: item for item in candidates}
        accept = by_id["accept_exact_seller_offer"]
        repair = by_id["validator_contract_repair"]
        # Buyer utility at the public seller contract is 23. The
        # pre-registered 70% repair transfers 16.1 to price and retains 6.9.
        self.assertAlmostEqual(repair.offer.price, 86.1)
        self.assertAlmostEqual(repair.own_utility, 6.9)
        self.assertEqual(
            repair.offer.continuous_terms, accept.offer.continuous_terms
        )
        self.assertEqual(
            repair.offer.discrete_terms, accept.offer.discrete_terms
        )

        def scored(candidate, score):
            return ScoredCandidate(
                candidate=candidate,
                p_accept=1.0,
                p_quit=0.0,
                exploitation_value=score,
                exploration_value=0.0,
                risk_penalty=0.0,
                score=score,
            )

        selected = scored(accept, 1.0)
        repair_row = scored(repair, 0.5)
        validated, diagnostics = adapter.validate_selection(
            state, belief, [selected, repair_row], selected
        )
        self.assertEqual(
            validated.candidate.candidate_id, "validator_contract_repair"
        )
        self.assertTrue(diagnostics["overrode_selection"])
        rendered = adapter.render_locked(state, belief, validated, None)
        rendered, checks = adapter.validate_locked(
            state, validated, rendered
        )
        self.assertTrue(checks["ok"])
        self.assertTrue(checks["action_lock_ok"])
        self.assertTrue(checks["action_semantics_ok"])
        self.assertIn("<contract>", rendered)

    def test_agenticpay_adaptive_settlement_margin_uses_public_staleness(self):
        context = {
            "max_price": 100.0,
            "product_info": {"name": "service"},
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {"warranty": {"full": 5.0, "none": 0.0}},
                },
            },
        }
        seller_contract = {
            "price": 70.0,
            "continuous_terms": {"delivery_days": 7.0},
            "discrete_terms": {"warranty": "none"},
        }
        seller_message = "<contract>\n" + json.dumps(seller_contract) + "\n</contract>"

        def adapter_for(history):
            return AgenticPayAdapter(
                context=context,
                history=history,
                current_state={"current_round": 3, "max_rounds": 10},
                buyer_max_price=100.0,
                self_id="Buyer",
                session_id="session",
                counterparty_id="seller",
                action_consistent_renderer=True,
                adaptive_contract_settlement_validator=True,
            )

        # A newly observed complete seller proposal is inspected, but not
        # charged an unconditional safety margin.
        fresh = adapter_for([{"role": "seller", "content": seller_message}])
        state = fresh.state("learned")
        belief = OpponentBelief("seller")
        fresh_candidates = {item.candidate_id: item for item in fresh.candidates(state, belief)}
        self.assertNotIn("validator_contract_repair", fresh_candidates)
        accept = fresh_candidates["accept_exact_seller_offer"]
        accepted, diagnostics = fresh.validate_selection(
            state,
            belief,
            [ScoredCandidate(accept, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)],
            ScoredCandidate(accept, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(accepted.candidate.action_type, "accept")
        self.assertTrue(diagnostics["considered"])
        self.assertFalse(diagnostics["overrode_selection"])
        self.assertEqual(diagnostics["required_repair_fraction"], 0.0)

        # One unchanged public repetition raises rho to 0.35. Buyer utility at
        # price 70 is 23, so the repair transfers 8.05 and retains 14.95.
        stale = adapter_for([
            {"role": "seller", "round": 1, "content": seller_message},
            {"role": "seller", "round": 2, "content": seller_message},
        ])
        state = stale.state("learned")
        candidates = {item.candidate_id: item for item in stale.candidates(state, belief)}
        repair = candidates["validator_contract_repair"]
        self.assertAlmostEqual(repair.offer.price, 78.05)
        self.assertAlmostEqual(repair.own_utility, 14.95)
        self.assertEqual(repair.metadata["repair_fraction"], 0.35)

    def test_agenticpay_calibrated_opening_locks_validated_terms_and_price(self):
        context = {
            "max_price": 100.0,
            "product_info": {"name": "service"},
            "contract_config": {
                "contrainfo": {"product_request": "reliable service"},
                "field_descriptions": {
                    "continuous_terms.delivery_days": "delivery time",
                    "discrete_terms.warranty": "service warranty",
                },
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {"warranty": {"full": 5.0, "none": 0.0}},
                },
            },
        }

        class OpeningClient(MockClient):
            def generate(self, prompt, **kwargs):
                if "LOW-FRICTION" in prompt:
                    # Price is deliberately ignored by trusted calibration;
                    # the model is responsible only for a legal term profile.
                    return json.dumps({
                        "price": 1.0,
                        "continuous_terms": {"delivery_days": 1},
                        "discrete_terms": {"warranty": "full"},
                    })
                return super().generate(prompt, **kwargs)

        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 10},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
            action_consistent_renderer=True,
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
        )
        state = adapter.state("learned")
        proposal = adapter.prepare_llm_proposal(state, OpeningClient())
        self.assertEqual(proposal["model_proposed_price"], 1.0)
        self.assertEqual(proposal["calibrated_price"], 94.0)
        self.assertTrue(proposal["buyer_ir_valid"])
        belief = OpponentBelief("seller")
        candidates = {item.candidate_id: item for item in adapter.candidates(state, belief)}
        opening = candidates["calibrated_opening"]
        self.assertAlmostEqual(opening.offer.price, 94.0)
        self.assertAlmostEqual(opening.own_utility, 10.0)

        fallback = next(item for key, item in candidates.items() if key.startswith("offer_"))
        fallback_row = ScoredCandidate(fallback, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        opening_row = ScoredCandidate(opening, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [fallback_row, opening_row], fallback_row
        )
        self.assertEqual(selected.candidate.candidate_id, "calibrated_opening")
        self.assertTrue(diagnostics["overrode_selection"])
        rendered = adapter.render_locked(state, belief, selected, None)
        _, checks = adapter.validate_locked(state, selected, rendered)
        self.assertTrue(checks["action_lock_ok"])
        self.assertTrue(checks["buyer_ir_ok"])

    def test_public_counteroffer_validator_aligns_repeated_terms_and_ends_opening(self):
        context = {
            "max_price": 100.0,
            "product_info": {"name": "service"},
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"warranty": ["full", "none"]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {"warranty": {"full": 5.0, "none": 0.0}},
                },
            },
        }
        seller_message = """<contract>
{"price": 80, "continuous_terms": {"delivery_days": 7},
 "discrete_terms": {"warranty": "none"}}
</contract>"""
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {"role": "seller", "round": 1, "content": seller_message},
                {"role": "seller", "round": 1, "content": seller_message},
            ],
            # Upstream may still expose round 1 after a seller response. V18
            # must nevertheless treat this as a counteroffer, not an opening.
            current_state={"current_round": 1, "max_rounds": 10},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
            action_consistent_renderer=True,
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            public_counteroffer_term_validator=True,
        )
        state = adapter.state("learned")
        proposal = adapter.prepare_llm_proposal(state, MockClient())
        self.assertEqual(
            proposal["reason"], "calibrated_opening_first_contract_action_only"
        )
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        selected = next(
            item
            for item in candidates
            if item.action_type == "offer"
            and item.offer.price == 80.0
            and item.metadata.get("term_profile") == "own_best"
        )
        aligned = next(
            item
            for item in candidates
            if item.action_type == "offer"
            and item.offer.price == 80.0
            and item.metadata.get("term_profile") == "seller_latest"
        )
        selected_row = ScoredCandidate(selected, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        aligned_row = ScoredCandidate(aligned, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        validated, diagnostics = adapter.validate_selection(
            state, belief, [selected_row, aligned_row], selected_row
        )
        self.assertEqual(validated.candidate.metadata["term_profile"], "seller_latest")
        self.assertEqual(
            diagnostics["reason"],
            "repeated_public_terms_aligned_at_selected_price",
        )
        self.assertTrue(diagnostics["overrode_selection"])

    def test_minimal_buyer_ir_repair_changes_only_required_seller_term(self):
        context = {
            "max_price": 13.85,
            "product_info": {"name": "sandals"},
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 5}},
                "discrete_options": {
                    "return_policy": ["30_days", "none"],
                    "size_color_confirmation": ["confirmed", "standard"],
                    "user_product_preference": [
                        "strong_match", "partial_match", "mismatch_or_uncertain"
                    ],
                },
                "buyer_preferences": {
                    "v_base": 13.85,
                    "continuous_weights": {"delivery_days": -0.18},
                    "discrete_weights": {
                        "return_policy": {"30_days": 1.4, "none": -1.6},
                        "size_color_confirmation": {"confirmed": 1.6, "standard": -0.8},
                        "user_product_preference": {
                            "strong_match": 0.22,
                            "partial_match": 0.09,
                            "mismatch_or_uncertain": -0.18,
                        },
                    },
                },
            },
        }
        contract = {
            "price": 11.75,
            "continuous_terms": {"delivery_days": 1},
            "discrete_terms": {
                "return_policy": "none",
                "size_color_confirmation": "standard",
                "user_product_preference": "partial_match",
            },
        }
        seller_message = "<contract>\n" + json.dumps(contract) + "\n</contract>"
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {"role": "seller", "round": 1, "content": seller_message},
                {"role": "seller", "round": 2, "content": seller_message},
            ],
            current_state={"current_round": 4, "max_rounds": 20},
            buyer_max_price=13.85,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
            action_consistent_renderer=True,
            adaptive_contract_settlement_validator=True,
            calibrated_opening_candidate=True,
            minimal_buyer_ir_term_repair_validator=True,
        )
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        repair = next(
            item
            for item in candidates
            if item.candidate_id == "minimal_buyer_ir_term_repair"
        )
        self.assertEqual(repair.offer.continuous_terms, contract["continuous_terms"])
        self.assertEqual(repair.offer.discrete_terms["return_policy"], "none")
        self.assertEqual(
            repair.offer.discrete_terms["size_color_confirmation"], "confirmed"
        )
        self.assertEqual(
            repair.offer.discrete_terms["user_product_preference"], "partial_match"
        )
        self.assertEqual(len(repair.metadata["changed_fields"]), 1)
        self.assertGreater(repair.offer.price, contract["price"])
        self.assertGreaterEqual(
            repair.own_utility, repair.metadata["target_buyer_reserve"] - 1e-8
        )

        selected = next(
            item
            for item in candidates
            if item.action_type == "offer"
            and abs(item.offer.price - 11.75) < 1e-8
            and item.metadata.get("term_profile") == "own_best"
        )
        selected_row = ScoredCandidate(selected, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        repair_row = ScoredCandidate(repair, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        validated, diagnostics = adapter.validate_selection(
            state, belief, [selected_row, repair_row], selected_row
        )
        self.assertEqual(
            validated.candidate.candidate_id, "minimal_buyer_ir_term_repair"
        )
        self.assertEqual(
            diagnostics["reason"], "minimal_public_contract_deviation_required"
        )

        text_only = AgenticPayAdapter(
            context=context,
            history=[{"role": "seller", "round": 1, "content": "I need better terms."}],
            current_state={"current_round": 1, "max_rounds": 20},
            buyer_max_price=13.85,
            self_id="Buyer",
            session_id="text-only",
            counterparty_id="seller",
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            minimal_buyer_ir_term_repair_validator=True,
        )
        proposal = text_only.prepare_llm_proposal(text_only.state("learned"), MockClient())
        self.assertEqual(
            proposal["reason"], "calibrated_opening_first_contract_action_only"
        )

    def test_semantic_low_burden_opening_snaps_continuous_term_to_endpoint(self):
        context = {
            "max_price": 20.0,
            "product_info": {"name": "pickup service"},
            "contract_config": {
                "field_descriptions": {
                    "continuous_terms.wait_time_mins":
                        "How long the provider must wait at pickup."
                },
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {"service": ["standard", "extra"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"wait_time_mins": 0.5},
                    "discrete_weights": {"service": {"standard": 0.0, "extra": 1.0}},
                },
            },
        }

        class BurdenClient(MockClient):
            def generate(self, prompt, **kwargs):
                if "EVERY continuous issue" in prompt:
                    return json.dumps({
                        "price": 18.0,
                        # Deliberately interior: trusted code must snap this
                        # provider-wait obligation to the nearest endpoint 0.
                        "continuous_terms": {"wait_time_mins": 10},
                        "discrete_terms": {"service": "standard"},
                    })
                return super().generate(prompt, **kwargs)

        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 10},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="session",
            counterparty_id="seller",
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            seller_burden_endpoint_opening=True,
        )
        diagnostics = adapter.prepare_llm_proposal(
            adapter.state("learned"), BurdenClient()
        )
        self.assertTrue(diagnostics["buyer_ir_valid"])
        self.assertEqual(diagnostics["offer"]["continuous_terms"]["wait_time_mins"], 0.0)
        self.assertEqual(
            diagnostics["endpoint_normalizations"],
            [{"issue": "wait_time_mins", "model_value": 10.0, "endpoint": 0.0}],
        )

        class FirewallClient(MockClient):
            last_prompt = ""

            def generate(self, prompt, **kwargs):
                self.last_prompt = prompt
                if "private utility is withheld" in prompt:
                    return json.dumps({
                        "price": 18.0,
                        "continuous_terms": {"wait_time_mins": 0},
                        "discrete_terms": {"service": "standard"},
                    })
                return super().generate(prompt, **kwargs)

        firewall_client = FirewallClient()
        firewall = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 10},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="firewall",
            counterparty_id="seller",
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            seller_burden_endpoint_opening=True,
            public_only_burden_selector=True,
        )
        firewall_result = firewall.prepare_llm_proposal(
            firewall.state("learned"), firewall_client
        )
        self.assertTrue(firewall_result["buyer_ir_valid"])
        self.assertNotIn('"continuous_weights"', firewall_client.last_prompt)
        self.assertNotIn('"default_best_terms"', firewall_client.last_prompt)
        self.assertIn("withheld_from_semantic_selector", firewall_client.last_prompt)

    def test_classifier_composed_opening_is_low_burden_and_buyer_ir(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "contrainfo": {"product_request": "a taxi ride"},
                "field_descriptions": {
                    "continuous_terms.wait_time_mins": "Minutes the driver must wait.",
                    "discrete_terms.route_preference": "Tunnel or local streets.",
                    "discrete_terms.user_product_preference": "Match to user preference.",
                },
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {
                    "route_preference": ["tunnel", "local_streets"],
                    "user_product_preference": ["strong_match", "partial_match"],
                },
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {
                        "route_preference": {"tunnel": 4.0, "local_streets": -2.0},
                        "user_product_preference": {"strong_match": 0.28, "partial_match": 0.1},
                    },
                },
            },
        }

        class ClassifierClient(MockClient):
            def __init__(self):
                self.outputs = [
                    '{"role":"PROVIDER_REQUIRED_EFFORT_DURATION","confidence":0.95}',
                    '{"role":"PROVIDER_REQUIRED_EFFORT_DURATION","confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":4},{"id":"O1","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":4},{"id":"O1","burden":0}],"confidence":0.95}',
                    # Tie => abstain; composer must record it and use legal own-best.
                    '{"ratings":[{"id":"O0","burden":1},{"id":"O1","burden":1}],"confidence":0.9}',
                    '{"ratings":[{"id":"O0","burden":1},{"id":"O1","burden":1}],"confidence":0.9}',
                ]

            def generate(self, prompt, **kwargs):
                return self.outputs.pop(0)

        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="classifier",
            counterparty_id="seller",
            action_consistent_renderer=True,
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            ontological_issue_classifier_opening=True,
        )
        classifier = OntologicalIssueClassifier(
            ClassifierClient(), min_confidence=0.75, consensus_passes=2
        )
        diagnostics = adapter.prepare_issue_classifier_opening(
            adapter.state("learned"), classifier
        )
        offer = diagnostics["offer"]
        self.assertEqual(offer["continuous_terms"]["wait_time_mins"], 0.0)
        self.assertEqual(offer["discrete_terms"]["route_preference"], "local_streets")
        self.assertEqual(offer["discrete_terms"]["user_product_preference"], "strong_match")
        self.assertEqual(diagnostics["discrete_abstentions"], ["user_product_preference"])
        self.assertTrue(diagnostics["buyer_ir_valid"])
        self.assertLess(offer["price"], 0.94 * 16.65)

        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        opening = next(
            item for item in candidates
            if item.candidate_id == "classifier_calibrated_opening"
        )
        fallback = next(item for item in candidates if item.candidate_id.startswith("offer_"))
        fallback_row = ScoredCandidate(fallback, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        opening_row = ScoredCandidate(opening, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        selected, validation = adapter.validate_selection(
            state, belief, [fallback_row, opening_row], fallback_row
        )
        self.assertEqual(selected.candidate.candidate_id, "classifier_calibrated_opening")
        self.assertEqual(validation["reason"], "validated_classifier_opening_required")

    def test_rejectable_classifier_guard_persists_until_direct_counteroffer(self):
        context = {
            "max_price": 12.15,
            "contract_config": {
                "contrainfo": {"product_request": "delivered food"},
                "field_descriptions": {
                    "discrete_terms.delivery_speed": "Rush, standard, or batched delivery.",
                    "discrete_terms.extra_condiments": "Whether extras are included.",
                },
                "continuous_bounds": {},
                "discrete_options": {
                    "delivery_speed": ["rush", "standard", "batched"],
                    "extra_condiments": [True, False],
                },
                "buyer_preferences": {
                    "v_base": 12.15,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "delivery_speed": {"rush": 3.0, "standard": 0.0, "batched": -2.0},
                        "extra_condiments": {"true": 1.5, "false": 0.0},
                    },
                },
            },
        }

        class ClassifierClient(MockClient):
            def __init__(self):
                self.outputs = [
                    '{"ratings":[{"id":"O0","burden":4},{"id":"O1","burden":2},{"id":"O2","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":4},{"id":"O1","burden":2},{"id":"O2","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":2},{"id":"O1","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":2},{"id":"O1","burden":0}],"confidence":0.95}',
                ]

            def generate(self, prompt, **kwargs):
                return self.outputs.pop(0)

        classifier = OntologicalIssueClassifier(
            ClassifierClient(), min_confidence=0.75, consensus_passes=2
        )
        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 20},
            buyer_max_price=12.15,
            self_id="Buyer",
            session_id="classifier-guard",
            counterparty_id="seller",
            action_consistent_renderer=True,
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            ontological_issue_classifier_opening=True,
            ontological_issue_classifier_guard=True,
        )
        adapter.prepare_issue_classifier_opening(adapter.state("learned"), classifier)
        profiles = adapter._term_profiles()
        self.assertEqual([profile[2] for profile in profiles], ["classifier_guarded"])
        self.assertEqual(profiles[0][1], {"delivery_speed": "batched", "extra_condiments": False})

        seller_contract = {
            "price": 10.5,
            "continuous_terms": {},
            "discrete_terms": {"delivery_speed": "standard", "extra_condiments": False},
        }
        guarded_after_response = AgenticPayAdapter(
            context=context,
            history=[
                {"role": "buyer", "round": 1, "content": "<contract>" + json.dumps(profiles[0][1]) + "</contract>"},
                {"role": "seller", "round": 1, "content": "<contract>" + json.dumps(seller_contract) + "</contract>"},
            ],
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=12.15,
            self_id="Buyer",
            session_id="classifier-guard",
            counterparty_id="seller",
            action_consistent_renderer=True,
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            ontological_issue_classifier_opening=True,
            ontological_issue_classifier_guard=True,
        )
        guarded_after_response.prepare_issue_classifier_opening(
            guarded_after_response.state("learned"), classifier
        )
        profiles = guarded_after_response._term_profiles()
        sources = [profile[2] for profile in profiles]
        self.assertIn("classifier_guarded", sources)
        self.assertIn("seller_latest_direct_evidence", sources)
        self.assertNotIn("own_best", sources)
        self.assertFalse(any(profile[1].get("delivery_speed") == "rush" for profile in profiles))

    def test_classifier_settlement_buffer_is_buyer_ir_capped_and_mandatory(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {"route_preference": ["local_streets", "tunnel"]},
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {
                        "route_preference": {"local_streets": -2.0, "tunnel": 4.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 13.9,
            "continuous_terms": {"wait_time_mins": 0.0},
            "discrete_terms": {"route_preference": "local_streets"},
        }
        history = [
            {
                "role": "seller",
                "round": 2,
                "content": "Counter.\n<contract>\n"
                + json.dumps(seller_contract)
                + "\n</contract>",
            }
        ]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="buffer",
            counterparty_id="seller",
            action_consistent_renderer=True,
            classifier_settlement_buffer_fraction=0.01,
        )
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        exact = next(item for item in candidates if item.candidate_id == "accept_exact_seller_offer")
        buffered = next(item for item in candidates if item.candidate_id == "classifier_settlement_buffer")
        self.assertAlmostEqual(buffered.offer.price, 13.9 + 0.1665, places=6)
        self.assertGreater(buffered.own_utility, 0.0)
        exact_row = ScoredCandidate(exact, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        buffer_row = ScoredCandidate(buffered, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        selected, validation = adapter.validate_selection(
            state, belief, [exact_row, buffer_row], exact_row
        )
        self.assertEqual(selected.candidate.candidate_id, "classifier_settlement_buffer")
        self.assertEqual(validation["reason"], "fresh_public_contract_requires_small_ir_buffer")

    def test_v54_partner_ir_reserve_uses_only_nonabstained_operational_roles(self):
        context = {
            "max_price": 30.0,
            "contract_config": {
                "continuous_bounds": {},
                "discrete_options": {
                    "return_policy": ["none", "30_days"],
                    "user_product_preference": [
                        "mismatch_or_uncertain",
                        "strong_match",
                    ],
                },
                "buyer_preferences": {
                    "v_base": 30.0,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "return_policy": {"none": 0.0, "30_days": 2.0},
                        "user_product_preference": {
                            "mismatch_or_uncertain": 0.0,
                            "strong_match": 2.0,
                        },
                    },
                },
            },
        }
        seller_contract = {
            "price": 10.0,
            "continuous_terms": {},
            "discrete_terms": {
                "return_policy": "30_days",
                "user_product_preference": "strong_match",
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=30.0,
            self_id="Buyer",
            session_id="v54-role-reserve",
            counterparty_id="seller",
            action_consistent_renderer=True,
            classifier_settlement_buffer_fraction=0.01,
            classifier_settlement_semantic_risk_multiplier=0.04,
            public_partner_ir_reserve_multiplier=0.12,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_discrete = {
            "return_policy": "none",
            "user_product_preference": "mismatch_or_uncertain",
        }
        adapter._llm_proposal_diagnostics["issue_role_decisions"] = [
            {
                "issue": "return_policy",
                "label": "OPERATIONAL_OBLIGATION",
                "abstained": False,
            },
            {
                "issue": "user_product_preference",
                "label": "DESCRIPTIVE_EVIDENCE",
                "abstained": False,
            },
            {
                "issue": "unknown",
                "label": "OPERATIONAL_OBLIGATION",
                "abstained": True,
            },
        ]
        self.assertAlmostEqual(adapter._operational_issue_share(), 0.5)
        candidates = adapter.candidates(adapter.state("learned"), OpponentBelief("seller"))
        buffered = next(
            item
            for item in candidates
            if item.candidate_id == "classifier_settlement_buffer"
        )
        # semantic risk=1; effective fraction=.01+.04+.12*(1/2)=.11.
        self.assertAlmostEqual(buffered.offer.price, 13.3, places=6)
        self.assertAlmostEqual(
            buffered.metadata["operational_issue_share"], 0.5
        )

    def test_semantic_risk_buffer_scales_and_excludes_abstentions(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 0, "max": 10}},
                "discrete_options": {"return_policy": ["none", "30_days"]},
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"delivery_days": 1.0},
                    "discrete_weights": {
                        "return_policy": {"none": 0.0, "30_days": 1.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 13.9,
            "continuous_terms": {"delivery_days": 1.0},
            "discrete_terms": {"return_policy": "30_days"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="semantic-risk-buffer",
            counterparty_id="seller",
            action_consistent_renderer=True,
            ontological_issue_classifier_guard=True,
            classifier_settlement_buffer_fraction=0.01,
            classifier_settlement_semantic_risk_multiplier=0.04,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"delivery_days": 10.0}
        adapter._classifier_guard_discrete = {"return_policy": "none"}
        # The discrete classifier abstained, so its visible disagreement must
        # not inflate semantic risk. Only |1-10| / 10 = 0.9 contributes.
        adapter._classifier_guard_abstentions = {"discrete_terms.return_policy"}
        self.assertAlmostEqual(
            adapter._classifier_semantic_departure_risk(adapter.last_seller_offer),
            0.9,
        )
        candidates = adapter.candidates(adapter.state("learned"), OpponentBelief("seller"))
        buffered = next(
            item for item in candidates if item.candidate_id == "classifier_settlement_buffer"
        )
        expected_fraction = 0.01 + 0.04 * 0.9
        self.assertAlmostEqual(
            buffered.metadata["effective_buffer_fraction"], expected_fraction
        )
        self.assertAlmostEqual(
            buffered.metadata["buffer_delta"], expected_fraction * 16.65
        )

    def test_v30_buffer_floor_also_catches_an_ordinary_matching_offer(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {"route_preference": ["local_streets", "tunnel"]},
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {
                        "route_preference": {"local_streets": -2.0, "tunnel": 4.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 13.9,
            "continuous_terms": {"wait_time_mins": 0.0},
            "discrete_terms": {"route_preference": "local_streets"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="v30-offer-floor",
            counterparty_id="seller",
            action_consistent_renderer=True,
            classifier_settlement_buffer_fraction=0.01,
            evidence_gated_settlement_frontier=True,
        )
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        buffered = next(
            item
            for item in adapter.candidates(state, belief)
            if item.candidate_id == "classifier_settlement_buffer"
        )
        # This is deliberately an OFFER, not ACCEPT. V29 only guarded ACCEPT,
        # which allowed a normal planner candidate to bypass the price floor.
        ordinary = CandidateAction(
            candidate_id="ordinary_matching_offer",
            action_type="offer",
            counterparty_id="seller",
            offer=adapter.last_seller_offer,
            own_utility=adapter._buyer_utility(adapter.last_seller_offer),
            own_utility_normalized=0.1,
            opponent_value_proxy=1.0,
            base_acceptance=1.0,
        )
        ordinary_row = ScoredCandidate(ordinary, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        buffer_row = ScoredCandidate(buffered, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, buffer_row], ordinary_row
        )
        self.assertEqual(selected.candidate.candidate_id, "classifier_settlement_buffer")
        self.assertEqual(
            diagnostics["reason"],
            "seller_term_matching_offer_below_semantic_risk_floor",
        )

    def test_v30_public_acceptance_is_positive_and_price_grounded(self):
        offer = CanonicalOffer(
            price=26.5,
            continuous_terms={"delivery_days": 5.0},
            discrete_terms={"return_policy": "30_days"},
        )
        self.assertTrue(
            AgenticPayAdapter._explicit_public_acceptance(
                "I am willing to accept your package at $26.50; let's make it happen.",
                offer,
            )
        )
        self.assertFalse(
            AgenticPayAdapter._explicit_public_acceptance(
                "I cannot accept your package at $26.50.", offer
            )
        )
        self.assertFalse(
            AgenticPayAdapter._explicit_public_acceptance(
                "I accept a different price of $29.00.", offer
            )
        )

    def test_v30_frontier_probe_is_buyer_ir_capped_and_one_shot(self):
        context = {
            "max_price": 20.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 10}},
                "discrete_options": {"return_policy": ["none", "30_days"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {
                        "return_policy": {"none": 0.0, "30_days": 8.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 18.0,
            "continuous_terms": {"delivery_days": 10.0},
            "discrete_terms": {"return_policy": "none"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="v30-frontier",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
        )
        result = adapter._evidence_gated_frontier_probe()
        self.assertIsNotNone(result)
        probe, metadata = result
        self.assertEqual(probe.continuous_terms["delivery_days"], 1.0)
        self.assertEqual(probe.discrete_terms["return_policy"], "30_days")
        self.assertLessEqual(probe.price, 20.0)
        self.assertGreaterEqual(adapter._buyer_utility(probe), 1.0 - 1e-8)
        self.assertTrue(metadata["one_shot"])

        repeated = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                },
                {
                    "role": "buyer",
                    "round": 3,
                    "content": "<contract>" + json.dumps(probe.to_dict()) + "</contract>",
                },
            ],
            current_state={"current_round": 4, "max_rounds": 20},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="v30-frontier-repeat",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
        )
        self.assertIsNone(repeated._evidence_gated_frontier_probe())

    def test_v31_rejects_extreme_buyer_utility_leverage(self):
        context = {
            "max_price": 10.0,
            "contract_config": {
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 100}},
                "discrete_options": {},
                "buyer_preferences": {
                    "v_base": 10.0,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {},
                },
            },
        }
        seller_contract = {
            "price": 9.6,
            "continuous_terms": {"wait_time_mins": 0.0},
            "discrete_terms": {},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=10.0,
            self_id="Buyer",
            session_id="v31-leverage-cap",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"wait_time_mins": 0.0}
        self.assertIsNone(adapter._evidence_gated_frontier_probe())

    def test_v31_uses_midpoint_continuous_probe_and_max_compensation(self):
        context = {
            "max_price": 20.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 10}},
                "discrete_options": {"return_policy": ["none", "30_days"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"delivery_days": -0.2},
                    "discrete_weights": {
                        "return_policy": {"none": 0.0, "30_days": 4.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 18.0,
            "continuous_terms": {"delivery_days": 10.0},
            "discrete_terms": {"return_policy": "none"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="v31-midpoint",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"delivery_days": 10.0}
        adapter._classifier_guard_discrete = {"return_policy": "none"}
        result = adapter._evidence_gated_frontier_probe()
        self.assertIsNotNone(result)
        probe, metadata = result
        self.assertEqual(probe.continuous_terms["delivery_days"], 5.5)
        self.assertEqual(probe.discrete_terms["return_policy"], "30_days")
        self.assertEqual(probe.price, 20.0)
        self.assertLess(metadata["buyer_utility_leverage_ratio"], 0.75)
        self.assertEqual(metadata["continuous_step_fraction"], 0.5)

    def test_v31_ordinary_offer_uses_base_not_semantic_floor(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {"route_preference": ["local_streets", "tunnel"]},
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {
                        "route_preference": {"local_streets": 0.0, "tunnel": 4.0}
                    },
                },
            },
        }
        seller_contract = {
            "price": 12.0,
            "continuous_terms": {"wait_time_mins": 0.0},
            "discrete_terms": {"route_preference": "local_streets"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="v31-base-floor",
            counterparty_id="seller",
            action_consistent_renderer=True,
            ontological_issue_classifier_guard=True,
            classifier_settlement_buffer_fraction=0.01,
            classifier_settlement_semantic_risk_multiplier=0.04,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"wait_time_mins": 30.0}
        adapter._classifier_guard_discrete = {"route_preference": "tunnel"}
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        base = next(
            item
            for item in candidates
            if item.candidate_id == "classifier_base_settlement_floor"
        )
        semantic = next(
            item
            for item in candidates
            if item.candidate_id == "classifier_settlement_buffer"
        )
        self.assertLess(base.offer.price, semantic.offer.price)
        ordinary = CandidateAction(
            candidate_id="ordinary_matching_offer",
            action_type="offer",
            counterparty_id="seller",
            offer=adapter.last_seller_offer,
            own_utility=adapter._buyer_utility(adapter.last_seller_offer),
            own_utility_normalized=0.1,
            opponent_value_proxy=1.0,
            base_acceptance=1.0,
        )
        ordinary_row = ScoredCandidate(ordinary, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        base_row = ScoredCandidate(base, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5)
        semantic_row = ScoredCandidate(semantic, 0.4, 0.0, 0.4, 0.0, 0.0, 0.4)
        selected, diagnostics = adapter.validate_selection(
            state,
            belief,
            [ordinary_row, base_row, semantic_row],
            ordinary_row,
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "classifier_base_settlement_floor",
        )
        self.assertEqual(diagnostics["ordinary_offer_floor_type"], "base_only")

    def test_v32_probe_changes_one_issue_and_cannot_cross_seller_price(self):
        context = {
            "max_price": 12.15,
            "contract_config": {
                "continuous_bounds": {},
                "discrete_options": {
                    "delivery_speed": ["rush", "standard", "batched"],
                    "extra_condiments": [True, False],
                    "user_product_preference": [
                        "strong_match",
                        "partial_match",
                        "mismatch_or_uncertain",
                    ],
                },
                "buyer_preferences": {
                    "v_base": 12.15,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "delivery_speed": {
                            "rush": 3.0,
                            "standard": 0.0,
                            "batched": -2.0,
                        },
                        "extra_condiments": {"true": 1.5, "false": 0.0},
                        "user_product_preference": {
                            "strong_match": 0.3,
                            "partial_match": 0.12,
                            "mismatch_or_uncertain": -0.25,
                        },
                    },
                },
            },
        }
        seller_contract = {
            "price": 10.08,
            "continuous_terms": {},
            "discrete_terms": {
                "delivery_speed": "batched",
                "extra_condiments": False,
                "user_product_preference": "strong_match",
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=12.15,
            self_id="Buyer",
            session_id="v32-noncrossing",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
            noncrossing_single_issue_frontier=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_discrete = {
            "delivery_speed": "batched",
            "extra_condiments": False,
            "user_product_preference": "strong_match",
        }
        result = adapter._evidence_gated_frontier_probe()
        self.assertIsNotNone(result)
        probe, metadata = result
        self.assertEqual(len(metadata["changed_fields"]), 1)
        self.assertEqual(
            metadata["changed_fields"][0]["field"],
            "discrete_terms.delivery_speed",
        )
        self.assertEqual(probe.discrete_terms["delivery_speed"], "rush")
        self.assertFalse(probe.discrete_terms["extra_condiments"])
        self.assertLess(probe.price, seller_contract["price"])
        self.assertGreaterEqual(metadata["noncrossing_margin"], 0.01 - 1e-8)
        self.assertTrue(metadata["single_issue"])
        self.assertEqual(metadata["probe_phase"], "information_only")

        candidate = CandidateAction(
            candidate_id="evidence_gated_frontier_probe",
            action_type="offer",
            counterparty_id="seller",
            offer=probe,
            own_utility=adapter._buyer_utility(probe),
            own_utility_normalized=0.1,
            opponent_value_proxy=0.5,
            base_acceptance=0.5,
        )
        rendered = adapter.render_locked(
            adapter.state("learned"),
            OpponentBelief("seller"),
            ScoredCandidate(candidate, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
            None,
        )
        self.assertIn("single-issue information probe", rendered)

        repeated = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                },
                {"role": "buyer", "round": 3, "content": rendered},
            ],
            current_state={"current_round": 4, "max_rounds": 20},
            buyer_max_price=12.15,
            self_id="Buyer",
            session_id="v32-one-shot",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
            noncrossing_single_issue_frontier=True,
        )
        repeated._classifier_guard_ready = True
        repeated._classifier_guard_discrete = dict(
            adapter._classifier_guard_discrete
        )
        self.assertIsNone(repeated._evidence_gated_frontier_probe())

    def test_v32_matching_offer_restores_full_semantic_settlement_floor(self):
        context = {
            "max_price": 16.65,
            "contract_config": {
                "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
                "discrete_options": {
                    "route_preference": ["local_streets", "tunnel"]
                },
                "buyer_preferences": {
                    "v_base": 16.65,
                    "continuous_weights": {"wait_time_mins": 1.0},
                    "discrete_weights": {
                        "route_preference": {
                            "local_streets": 0.0,
                            "tunnel": 4.0,
                        }
                    },
                },
            },
        }
        seller_contract = {
            "price": 12.0,
            "continuous_terms": {"wait_time_mins": 0.0},
            "discrete_terms": {"route_preference": "local_streets"},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "seller",
                    "round": 2,
                    "content": "<contract>"
                    + json.dumps(seller_contract)
                    + "</contract>",
                }
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=16.65,
            self_id="Buyer",
            session_id="v32-semantic-floor",
            counterparty_id="seller",
            action_consistent_renderer=True,
            ontological_issue_classifier_guard=True,
            classifier_settlement_buffer_fraction=0.01,
            classifier_settlement_semantic_risk_multiplier=0.04,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
            noncrossing_single_issue_frontier=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"wait_time_mins": 30.0}
        adapter._classifier_guard_discrete = {"route_preference": "tunnel"}
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        semantic = next(
            item
            for item in candidates
            if item.candidate_id == "classifier_settlement_buffer"
        )
        ordinary = CandidateAction(
            candidate_id="ordinary_matching_offer",
            action_type="offer",
            counterparty_id="seller",
            offer=adapter.last_seller_offer,
            own_utility=adapter._buyer_utility(adapter.last_seller_offer),
            own_utility_normalized=0.1,
            opponent_value_proxy=1.0,
            base_acceptance=1.0,
        )
        ordinary_row = ScoredCandidate(
            ordinary, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
        )
        semantic_row = ScoredCandidate(
            semantic, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5
        )
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, semantic_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "classifier_settlement_buffer",
        )
        self.assertEqual(
            diagnostics["ordinary_offer_floor_type"], "semantic_risk"
        )

    def test_v33_contracted_positive_acceptance_is_grounded(self):
        offer = CanonicalOffer(
            price=3.855467,
            continuous_terms={},
            discrete_terms={"delivery_speed": "standard"},
        )
        self.assertTrue(
            AgenticPayAdapter._explicit_public_acceptance(
                "I'm happy to accept your price of $3.855467.",
                offer,
                allow_contracted_copula=True,
            )
        )
        self.assertTrue(
            AgenticPayAdapter._explicit_public_acceptance(
                "We're willing to accept this package at $3.855467.",
                offer,
                allow_contracted_copula=True,
            )
        )
        self.assertFalse(
            AgenticPayAdapter._explicit_public_acceptance(
                "I'm happy to accept your price of $3.855467.", offer
            )
        )
        self.assertFalse(
            AgenticPayAdapter._explicit_public_acceptance(
                "I can't accept your price of $3.855467.", offer
            )
        )

    def test_v33_post_probe_public_counter_is_forced_to_settlement(self):
        context = {
            "max_price": 12.4,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 12.4,
                    "continuous_weights": {"delivery_days": -0.18},
                    "discrete_weights": {
                        "return_policy": {"30_days": 0.8, "none": -0.9}
                    },
                },
            },
        }
        probe = {
            "price": 10.14,
            "continuous_terms": {"delivery_days": 7},
            "discrete_terms": {"return_policy": "30_days"},
        }
        counter = {
            "price": 10.15,
            "continuous_terms": {"delivery_days": 7},
            "discrete_terms": {"return_policy": "none"},
        }
        history = [
            {
                "role": "buyer",
                "round": 2,
                "content": (
                    "This is a single-issue information probe.\n<contract>"
                    + json.dumps(probe)
                    + "</contract>"
                ),
            },
            {
                "role": "seller",
                "round": 2,
                "content": "My counter is:\n<contract>"
                + json.dumps(counter)
                + "</contract>",
            },
        ]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=12.4,
            self_id="Buyer",
            session_id="v33-public-counter",
            counterparty_id="seller",
            action_consistent_renderer=True,
            ontological_issue_classifier_guard=True,
            classifier_settlement_buffer_fraction=0.01,
            classifier_settlement_semantic_risk_multiplier=0.04,
            evidence_gated_settlement_frontier=True,
            bounded_compensated_active_frontier=True,
            noncrossing_single_issue_frontier=True,
            public_response_settlement_latch=True,
        )
        adapter._classifier_guard_ready = True
        adapter._classifier_guard_continuous = {"delivery_days": 7.0}
        adapter._classifier_guard_discrete = {"return_policy": "none"}
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        latch = next(
            item
            for item in candidates
            if item.candidate_id == "post_probe_public_counter_latch"
        )
        self.assertEqual(latch.offer.continuous_terms, counter["continuous_terms"])
        self.assertEqual(latch.offer.discrete_terms, counter["discrete_terms"])
        self.assertGreaterEqual(latch.offer.price, counter["price"])
        ordinary = next(
            item for item in candidates if item.action_type == "offer" and item is not latch
        )
        ordinary_row = ScoredCandidate(
            ordinary, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
        )
        latch_row = ScoredCandidate(latch, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, latch_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "post_probe_public_counter_latch",
        )
        self.assertEqual(
            diagnostics["reason"],
            "post_probe_complete_public_counter_latched",
        )

    def test_v34_repairs_non_ir_post_probe_counter_once_at_same_price(self):
        context = {
            "max_price": 12.4,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 12.4,
                    "continuous_weights": {"delivery_days": -0.2},
                    "discrete_weights": {
                        "return_policy": {"30_days": 0.8, "none": -0.9}
                    },
                },
            },
        }
        probe = {
            "price": 10.14,
            "continuous_terms": {"delivery_days": 7},
            "discrete_terms": {"return_policy": "30_days"},
        }
        counter = {
            "price": 10.15,
            "continuous_terms": {"delivery_days": 7},
            "discrete_terms": {"return_policy": "none"},
        }
        history = [
            {
                "role": "buyer",
                "round": 2,
                "content": (
                    "This is a single-issue information probe.\n<contract>"
                    + json.dumps(probe)
                    + "</contract>"
                ),
            },
            {
                "role": "seller",
                "round": 2,
                "content": "My counter is:\n<contract>"
                + json.dumps(counter)
                + "</contract>",
            },
        ]

        def make_adapter(current_history, *, domain_grid=False):
            return AgenticPayAdapter(
                context=context,
                history=current_history,
                current_state={"current_round": 3, "max_rounds": 20},
                buyer_max_price=12.4,
                self_id="Buyer",
                session_id="v34-minimal-repair",
                counterparty_id="seller",
                action_consistent_renderer=True,
                ontological_issue_classifier_guard=True,
                classifier_settlement_buffer_fraction=0.01,
                classifier_settlement_semantic_risk_multiplier=0.04,
                evidence_gated_settlement_frontier=True,
                bounded_compensated_active_frontier=True,
                noncrossing_single_issue_frontier=True,
                public_response_settlement_latch=True,
                post_probe_minimal_buyer_ir_repair=True,
                post_probe_domain_aware_repair_grid=domain_grid,
            )

        adapter = make_adapter(history)
        repair_result = adapter._post_probe_minimal_buyer_ir_repair()
        self.assertIsNotNone(repair_result)
        repair, metadata = repair_result
        self.assertAlmostEqual(repair.price, counter["price"])
        self.assertEqual(len(metadata["changed_fields"]), 1)
        self.assertTrue(metadata["same_public_counter_price"])
        self.assertTrue(metadata["one_field"])
        self.assertTrue(metadata["one_shot"])
        self.assertEqual(repair.discrete_terms, counter["discrete_terms"])
        self.assertLess(repair.continuous_terms["delivery_days"], 7)
        self.assertIsNone(metadata["changed_fields"][0]["semantic_grid"])
        self.assertGreaterEqual(adapter._buyer_utility(repair), 0.01 * 12.4 - 2e-6)

        domain_adapter = make_adapter(history, domain_grid=True)
        domain_result = domain_adapter._post_probe_minimal_buyer_ir_repair()
        self.assertIsNotNone(domain_result)
        domain_repair, domain_metadata = domain_result
        self.assertEqual(domain_repair.continuous_terms["delivery_days"], 6.0)
        self.assertEqual(domain_metadata["changed_fields"][0]["semantic_grid"], 1.0)
        self.assertGreaterEqual(
            domain_adapter._buyer_utility(domain_repair), 0.01 * 12.4 - 2e-6
        )

        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        repair_candidate = next(
            item
            for item in candidates
            if item.candidate_id == "post_probe_minimal_buyer_ir_repair"
        )
        ordinary = next(
            item
            for item in candidates
            if item.action_type == "offer" and item is not repair_candidate
        )
        ordinary_row = ScoredCandidate(
            ordinary, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
        )
        repair_row = ScoredCandidate(
            repair_candidate, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1
        )
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, repair_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "post_probe_minimal_buyer_ir_repair",
        )
        self.assertEqual(
            diagnostics["reason"],
            "non_ir_public_counter_requires_minimal_one_field_repair",
        )

        repeated_history = history + [
            {
                "role": "buyer",
                "round": 3,
                "content": "Minimal repair:\n<contract>"
                + json.dumps(repair.to_dict())
                + "</contract>",
            }
        ]
        self.assertIsNone(
            make_adapter(repeated_history)._post_probe_minimal_buyer_ir_repair()
        )

    def test_v36_compensates_publicly_rejected_repair_with_ir_safe_cents(self):
        context = {
            "max_price": 13.85,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 10}},
                "discrete_options": {
                    "return_policy": ["30_days", "none"],
                    "size_color_confirmation": ["confirmed", "standard"],
                    "user_product_preference": [
                        "strong_match", "partial_match", "mismatch_or_uncertain"
                    ],
                },
                "buyer_preferences": {
                    "v_base": 13.85,
                    "continuous_weights": {"delivery_days": -0.18},
                    "discrete_weights": {
                        "return_policy": {"30_days": 1.4, "none": -1.6},
                        "size_color_confirmation": {
                            "confirmed": 1.6, "standard": -0.8
                        },
                        "user_product_preference": {
                            "strong_match": 0.22,
                            "partial_match": 0.09,
                            "mismatch_or_uncertain": -0.18,
                        },
                    },
                },
            },
        }
        repair = {
            "price": 13.5,
            "continuous_terms": {"delivery_days": 1},
            "discrete_terms": {
                "return_policy": "none",
                "size_color_confirmation": "confirmed",
                "user_product_preference": "partial_match",
            },
        }
        rejection = {
            "price": 13.5,
            "continuous_terms": {"delivery_days": 1},
            "discrete_terms": {
                "return_policy": "none",
                "size_color_confirmation": "standard",
                "user_product_preference": "partial_match",
            },
        }
        history = [
            {
                "role": "buyer",
                "round": 5,
                "content": (
                    "I am making only the smallest one-field adjustment needed "
                    "for a feasible agreement.\n<contract>"
                    + json.dumps(repair)
                    + "</contract>"
                ),
            },
            {
                "role": "seller",
                "round": 5,
                "content": (
                    "Confirmed adds a measurable handling burden. I cannot "
                    "accept it without a corresponding adjustment in price.\n"
                    "<contract>" + json.dumps(rejection) + "</contract>"
                ),
            },
        ]

        def make_adapter(current_history):
            return AgenticPayAdapter(
                context=context,
                history=current_history,
                current_state={"current_round": 6, "max_rounds": 20},
                buyer_max_price=13.85,
                self_id="Buyer",
                session_id="v36-public-compensation",
                counterparty_id="seller",
                action_consistent_renderer=True,
                evidence_gated_settlement_frontier=True,
                post_probe_minimal_buyer_ir_repair=True,
                post_probe_domain_aware_repair_grid=True,
                post_repair_public_compensation=True,
            )

        adapter = make_adapter(history)
        result = adapter._post_repair_public_compensation_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertAlmostEqual(offer.price, 13.62)
        self.assertEqual(offer.continuous_terms, repair["continuous_terms"])
        self.assertEqual(offer.discrete_terms, repair["discrete_terms"])
        self.assertAlmostEqual(metadata["price_increment"], 0.12)
        self.assertTrue(metadata["same_non_price_terms"])
        self.assertTrue(metadata["cent_quantized"])
        self.assertGreaterEqual(
            adapter._buyer_utility(offer), 0.01 * 13.85 - 2e-6
        )

        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidate = next(
            c for c in adapter.candidates(state, belief)
            if c.candidate_id == "post_repair_public_compensation"
        )
        ordinary = next(
            c for c in adapter.candidates(state, belief)
            if c.action_type == "offer" and c is not candidate
        )
        ordinary_row = ScoredCandidate(ordinary, 1, 0, 1, 0, 0, 1)
        compensation_row = ScoredCandidate(candidate, 0.1, 0, 0.1, 0, 0, 0.1)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, compensation_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "post_repair_public_compensation",
        )
        self.assertEqual(
            diagnostics["reason"],
            "seller_publicly_requested_compensation_for_repair",
        )

        repeated = history + [{
            "role": "buyer", "round": 6,
            "content": (
                "I am adding a concrete price adjustment.\n<contract>"
                + json.dumps(offer.to_dict()) + "</contract>"
            ),
        }]
        self.assertIsNone(
            make_adapter(repeated)._post_repair_public_compensation_offer()
        )

    def test_v37_rolls_to_latest_structured_counter_and_bounds_replay(self):
        context = {
            "max_price": 20.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"delivery_days": -0.1},
                    "discrete_weights": {
                        "return_policy": {"30_days": 0.5, "none": 0.0}
                    },
                },
            },
        }
        probe = {
            "price": 10.0,
            "continuous_terms": {"delivery_days": 5},
            "discrete_terms": {"return_policy": "30_days"},
        }
        first_counter = {
            "price": 10.5,
            "continuous_terms": {"delivery_days": 4},
            "discrete_terms": {"return_policy": "30_days"},
        }
        buyer_followup = {
            "price": 10.8,
            "continuous_terms": {"delivery_days": 4},
            "discrete_terms": {"return_policy": "30_days"},
        }
        latest_counter = {
            "price": 10.8,
            "continuous_terms": {"delivery_days": 3},
            "discrete_terms": {"return_policy": "30_days"},
        }
        history = [
            {
                "role": "buyer", "round": 2,
                "content": "This is a single-issue information probe.\n<contract>"
                + json.dumps(probe) + "</contract>",
            },
            {
                "role": "seller", "round": 2,
                "content": "My counter.\n<contract>"
                + json.dumps(first_counter) + "</contract>",
            },
            {
                "role": "buyer", "round": 3,
                "content": "Settlement proposal.\n<contract>"
                + json.dumps(buyer_followup) + "</contract>",
            },
            {
                "role": "seller", "round": 3,
                "content": (
                    "I am happy to accept your price, but delivery is three days.\n"
                    "<contract>" + json.dumps(latest_counter) + "</contract>"
                ),
            },
        ]

        def make_adapter(current_history):
            return AgenticPayAdapter(
                context=context,
                history=current_history,
                current_state={"current_round": 4, "max_rounds": 20},
                buyer_max_price=20.0,
                self_id="Buyer",
                session_id="v37-rolling-state",
                counterparty_id="seller",
                action_consistent_renderer=True,
                evidence_gated_settlement_frontier=True,
                bounded_compensated_active_frontier=True,
                noncrossing_single_issue_frontier=True,
                public_response_settlement_latch=True,
                post_probe_minimal_buyer_ir_repair=True,
                post_probe_domain_aware_repair_grid=True,
                post_repair_public_compensation=True,
                rolling_public_contract_state=True,
                classifier_settlement_buffer_fraction=0.01,
                classifier_settlement_semantic_risk_multiplier=0.04,
            )

        adapter = make_adapter(history)
        latest = adapter._latest_post_probe_public_counter()
        self.assertEqual(latest.price, latest_counter["price"])
        self.assertEqual(latest.continuous_terms, latest_counter["continuous_terms"])
        self.assertEqual(latest.discrete_terms, latest_counter["discrete_terms"])
        self.assertIsNone(adapter._latest_publicly_accepted_buyer_offer())
        candidates = adapter.candidates(adapter.state("learned"), OpponentBelief("seller"))
        latch = next(
            c for c in candidates
            if c.candidate_id == "post_probe_public_counter_latch"
        )
        self.assertEqual(latch.offer.price, latest_counter["price"])
        self.assertEqual(
            latch.offer.continuous_terms, latest_counter["continuous_terms"]
        )
        self.assertEqual(
            latch.offer.discrete_terms, latest_counter["discrete_terms"]
        )
        self.assertEqual(latch.metadata["buffer_delta"], 0.0)
        self.assertTrue(latch.metadata["rolling_followup"])

        # If an abnormal environment failed to terminate after two exact
        # buyer echoes, neither acceptance nor post-probe latch may replay it.
        repeated = history + [
            {
                "role": "buyer", "round": 4,
                "content": "Exact settlement.\n<contract>"
                + json.dumps(latest_counter) + "</contract>",
            },
            {
                "role": "seller", "round": 4,
                "content": "I accept this exact offer.\n<contract>"
                + json.dumps(latest_counter) + "</contract>",
            },
            {
                "role": "buyer", "round": 5,
                "content": "Exact settlement retry.\n<contract>"
                + json.dumps(latest_counter) + "</contract>",
            },
        ]
        repeated_adapter = make_adapter(repeated)
        repeated_ids = {
            c.candidate_id for c in repeated_adapter.candidates(
                repeated_adapter.state("learned"), OpponentBelief("seller")
            )
        }
        self.assertNotIn("public_acceptance_latch", repeated_ids)
        self.assertNotIn("post_probe_public_counter_latch", repeated_ids)

    def test_v38_projects_classifier_profile_to_buyer_utility_budget(self):
        context = {
            "max_price": 10.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 10}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 10.0,
                    "continuous_weights": {"delivery_days": -0.2},
                    "discrete_weights": {
                        "return_policy": {"30_days": 2.0, "none": -2.0}
                    },
                },
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 1, "max_rounds": 20},
            buyer_max_price=10.0,
            self_id="Buyer",
            session_id="v38-opening-budget",
            counterparty_id="seller",
            utility_preserving_multiissue_guard=True,
        )
        continuous, discrete, diagnostics = (
            adapter._project_classifier_profile_to_utility_budget(
                {"delivery_days": 10.0},
                {"return_policy": "none"},
            )
        )
        self.assertEqual(discrete["return_policy"], "30_days")
        self.assertLess(continuous["delivery_days"], 10.0)
        self.assertLessEqual(
            diagnostics["final_profile_utility_loss"],
            diagnostics["total_profile_budget"] + 1e-8,
        )
        self.assertTrue(diagnostics["projected_fields"])

    def test_v38_blocks_low_utility_latch_and_forces_noncrossing_counter(self):
        context = {
            "max_price": 20.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"delivery_days": -1.0},
                    "discrete_weights": {
                        "return_policy": {"30_days": 2.0, "none": -2.0}
                    },
                },
            },
        }
        opening = {
            "price": 10.0,
            "continuous_terms": {"delivery_days": 1},
            "discrete_terms": {"return_policy": "30_days"},
        }
        counter = {
            "price": 10.5,
            "continuous_terms": {"delivery_days": 7},
            "discrete_terms": {"return_policy": "none"},
        }
        history = [
            {
                "role": "buyer",
                "round": 2,
                "content": "This is a single-issue information probe.\n<contract>"
                + json.dumps(opening)
                + "</contract>",
            },
            {
                "role": "seller",
                "round": 2,
                "content": "My complete counter.\n<contract>"
                + json.dumps(counter)
                + "</contract>",
            },
        ]

        def make_adapter(guard):
            return AgenticPayAdapter(
                context=context,
                history=history,
                current_state={"current_round": 4, "max_rounds": 20},
                buyer_max_price=20.0,
                self_id="Buyer",
                session_id="v38-settlement-floor",
                counterparty_id="seller",
                action_consistent_renderer=True,
                evidence_gated_settlement_frontier=True,
                public_response_settlement_latch=True,
                rolling_public_contract_state=True,
                utility_preserving_multiissue_guard=guard,
            )

        v37_adapter = make_adapter(False)
        v37_ids = {
            candidate.candidate_id
            for candidate in v37_adapter.candidates(
                v37_adapter.state("learned"), OpponentBelief("seller")
            )
        }
        self.assertIn("post_probe_public_counter_latch", v37_ids)

        adapter = make_adapter(True)
        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        floor = adapter._settlement_buyer_utility_floor(state)
        candidates = adapter.candidates(state, belief)
        ids = {candidate.candidate_id for candidate in candidates}
        self.assertNotIn("post_probe_public_counter_latch", ids)
        self.assertIn("utility_preserving_public_counter", ids)
        safe = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == "utility_preserving_public_counter"
        )
        self.assertLess(safe.offer.price, counter["price"])
        self.assertGreaterEqual(adapter._buyer_utility(safe.offer), floor - 1e-6)
        self.assertLessEqual(len(safe.metadata["changed_fields"]), 1)

        ordinary = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id != "utility_preserving_public_counter"
        )
        ordinary_row = ScoredCandidate(ordinary, 1, 0, 1, 0, 0, 1)
        safe_row = ScoredCandidate(safe, 0.1, 0, 0.1, 0, 0, 0.1)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, safe_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "utility_preserving_public_counter",
        )
        self.assertEqual(
            diagnostics["reason"],
            "public_counter_below_dynamic_buyer_utility_floor",
        )

    def test_v39_limits_opening_and_compensates_one_burden_field(self):
        context = {
            "max_price": 20.0,
            "contract_config": {
                "continuous_bounds": {"delivery_days": {"min": 1, "max": 10}},
                "discrete_options": {"return_policy": ["30_days", "none"]},
                "buyer_preferences": {
                    "v_base": 20.0,
                    "continuous_weights": {"delivery_days": -0.2},
                    "discrete_weights": {
                        "return_policy": {"30_days": 2.0, "none": -2.0}
                    },
                },
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=20.0,
            self_id="Buyer",
            session_id="v39-confirmed-tradeoff",
            counterparty_id="seller",
            utility_preserving_multiissue_guard=True,
            single_protected_issue_opening=True,
            semantic_departure_terminal_guard=True,
        )
        continuous, discrete, diagnostics = (
            adapter._project_classifier_profile_to_utility_budget(
                {"delivery_days": 10.0},
                {"return_policy": "none"},
            )
        )
        self.assertEqual(discrete["return_policy"], "30_days")
        self.assertEqual(continuous["delivery_days"], 10.0)
        self.assertEqual(
            diagnostics["protected_issue"],
            "discrete_terms.return_policy",
        )

        adapter._classifier_guard_continuous = {"delivery_days": 10.0}
        adapter._classifier_guard_discrete = {"return_policy": "none"}
        adapter._classifier_guard_ready = True
        counter = CanonicalOffer(
            price=15.0,
            continuous_terms={"delivery_days": 1.0},
            discrete_terms={"return_policy": "30_days"},
        )
        result = adapter._burden_reducing_tradeoff_offer(counter, 1.0)
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertLess(offer.price, counter.price)
        self.assertEqual(len(metadata["changed_fields"]), 1)
        self.assertGreater(metadata["semantic_departure_risk"], 0.45)
        self.assertGreaterEqual(adapter._buyer_utility(offer), 1.0)
        self.assertFalse(metadata["seller_private_utility_used"])

    def test_v40_repairs_only_repeated_anchor_with_minimum_ir_fields(self):
        context = {
            "max_price": 100.0,
            "contract_config": {
                "continuous_bounds": {"lease_months": {"min": 1, "max": 10}},
                "discrete_options": {"include_service": [False, True]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"lease_months": -2.0},
                    "discrete_weights": {
                        "include_service": {"false": 0.0, "true": 10.0}
                    },
                },
            },
        }
        anchor = {
            "price": 100.0,
            "continuous_terms": {"lease_months": 10.0},
            "discrete_terms": {"include_service": False},
        }
        history = [
            {
                "role": "seller",
                "round": turn,
                "content": "Still my complete offer.\n<contract>"
                + json.dumps(anchor)
                + "</contract>",
            }
            for turn in range(1, 4)
        ]
        adapter = AgenticPayAdapter(
            context=context,
            history=history,
            current_state={"current_round": 4, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v40-stagnation-ledger",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            stagnation_trade_ledger_repair=True,
        )
        result = adapter._stagnation_trade_ledger_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertEqual(offer.price, anchor["price"])
        self.assertGreaterEqual(adapter._buyer_utility(offer), 5.0 - 1e-6)
        self.assertEqual(metadata["public_counter_exact_repeats"], 3)
        self.assertEqual(metadata["changed_field_count"], 2)
        self.assertFalse(metadata["seller_private_utility_used"])

        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        repair = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == "stagnation_trade_ledger_repair"
        )
        ordinary = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id.startswith("offer_")
        )
        repair_row = ScoredCandidate(repair, 0.1, 0, 0.1, 0, 0, 0.1)
        ordinary_row = ScoredCandidate(ordinary, 1, 0, 1, 0, 0, 1)
        selected, diagnostics = adapter.validate_selection(
            state, belief, [ordinary_row, repair_row], ordinary_row
        )
        self.assertEqual(
            selected.candidate.candidate_id,
            "stagnation_trade_ledger_repair",
        )
        self.assertEqual(
            diagnostics["reason"],
            "repeated_public_contract_deadlock_repaired",
        )

        # Two repeats are evidence, but not enough to trigger an intervention.
        early = AgenticPayAdapter(
            context=context,
            history=history[:2],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v40-before-stagnation",
            counterparty_id="seller",
            stagnation_trade_ledger_repair=True,
        )
        self.assertIsNone(early._stagnation_trade_ledger_offer())
        self.assertFalse(early._complete_offer(None))

    def test_v41_compensates_only_high_conflict_buyer_safe_opening(self):
        context = {
            "max_price": 100.0,
            "contract_config": {
                "contrainfo": {"product_request": "a service contract"},
                "field_descriptions": {
                    "continuous_terms.commitment_months": (
                        "Number of months the customer must remain committed."
                    ),
                    "discrete_terms.include_service": (
                        "Whether the provider includes the extra service."
                    ),
                },
                "continuous_bounds": {
                    "commitment_months": {"min": 1, "max": 10}
                },
                "discrete_options": {"include_service": [True, False]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"commitment_months": -2.0},
                    "discrete_weights": {
                        "include_service": {"true": 20.0, "false": 0.0}
                    },
                },
            },
        }

        class ClassifierClient(MockClient):
            def __init__(self):
                self.outputs = [
                    '{"role":"CUSTOMER_COMMITMENT_DURATION","confidence":0.95}',
                    '{"role":"CUSTOMER_COMMITMENT_DURATION","confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":3},{"id":"O1","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":3},{"id":"O1","burden":0}],"confidence":0.95}',
                ]

            def generate(self, prompt, **kwargs):
                return self.outputs.pop(0)

        def opening(enabled):
            adapter = AgenticPayAdapter(
                context=context,
                history=[],
                current_state={"current_round": 1, "max_rounds": 20},
                buyer_max_price=100.0,
                self_id="Buyer",
                session_id=f"v41-{enabled}",
                counterparty_id="seller",
                action_consistent_renderer=True,
                llm_proposal_candidate=True,
                calibrated_opening_candidate=True,
                ontological_issue_classifier_opening=True,
                ontological_issue_classifier_guard=True,
                compensated_conflict_opening=enabled,
            )
            classifier = OntologicalIssueClassifier(
                ClassifierClient(), min_confidence=0.75, consensus_passes=2
            )
            diagnostics = adapter.prepare_issue_classifier_opening(
                adapter.state("learned"), classifier
            )
            return adapter, diagnostics

        frozen_v37, v37 = opening(False)
        self.assertLess(v37["offer"]["price"], 100.0)
        self.assertEqual(v37["offer"]["continuous_terms"]["commitment_months"], 10.0)
        self.assertFalse(v37["offer"]["discrete_terms"]["include_service"])
        self.assertFalse(
            v37["compensated_conflict_opening"]["activated"]
        )
        self.assertFalse(frozen_v37.stagnation_trade_ledger_repair)

        v41_adapter, v41 = opening(True)
        self.assertTrue(v41["compensated_conflict_opening"]["activated"])
        self.assertEqual(v41["offer"]["price"], 100.0)
        self.assertTrue(v41["offer"]["discrete_terms"]["include_service"])
        self.assertGreaterEqual(v41_adapter._buyer_utility(
            CanonicalOffer(
                price=v41["offer"]["price"],
                continuous_terms=v41["offer"]["continuous_terms"],
                discrete_terms=v41["offer"]["discrete_terms"],
            )
        ), 5.0)
        self.assertFalse(
            v41["compensated_conflict_opening"]["seller_private_utility_used"]
        )

    def test_v42_requires_one_field_confirmation_for_undercompensated_counter(self):
        context = {
            "max_price": 10.0,
            "contract_config": {
                "continuous_bounds": {},
                "discrete_options": {
                    "delivery_speed": ["rush", "batched"],
                    "extra_condiments": [True, False],
                },
                "buyer_preferences": {
                    "v_base": 10.0,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "delivery_speed": {"rush": 3.0, "batched": 0.0},
                        "extra_condiments": {"true": 2.0, "false": 0.0},
                    },
                },
            },
        }
        counter = {
            "price": 8.0,
            "continuous_terms": {},
            "discrete_terms": {
                "delivery_speed": "rush",
                "extra_condiments": True,
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[{
                "role": "seller",
                "round": 1,
                "content": "My counter. <contract>"
                + json.dumps(counter)
                + "</contract>",
            }],
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=10.0,
            self_id="Buyer",
            session_id="v42-confirmation",
            counterparty_id="seller",
            action_consistent_renderer=True,
            evidence_gated_settlement_frontier=True,
            sequential_semantic_confirmation=True,
        )
        adapter._classifier_guard_discrete = {
            "delivery_speed": "batched",
            "extra_condiments": False,
        }
        adapter._classifier_guard_continuous = {}
        adapter._classifier_guard_ready = True
        result = adapter._sequential_semantic_confirmation_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertEqual(offer.price, counter["price"])
        self.assertEqual(len(metadata["changed_fields"]), 1)
        self.assertFalse(offer.discrete_terms["extra_condiments"])
        self.assertEqual(offer.discrete_terms["delivery_speed"], "rush")
        self.assertGreaterEqual(adapter._buyer_utility(offer), 0.5)
        self.assertFalse(metadata["seller_private_utility_used"])

        state = adapter.state("learned")
        belief = OpponentBelief("seller")
        candidates = adapter.candidates(state, belief)
        confirmation = next(
            item
            for item in candidates
            if item.candidate_id == "sequential_semantic_confirmation"
        )
        ordinary = next(item for item in candidates if item.candidate_id.startswith("offer_"))
        forced, diagnostics = adapter.validate_selection(
            state,
            belief,
            [
                ScoredCandidate(ordinary, 1, 0, 1, 0, 0, 1),
                ScoredCandidate(confirmation, 0.1, 0, 0.1, 0, 0, 0.1),
            ],
            ScoredCandidate(ordinary, 1, 0, 1, 0, 0, 1),
        )
        self.assertEqual(
            forced.candidate.candidate_id,
            "sequential_semantic_confirmation",
        )
        self.assertEqual(
            diagnostics["validator_type"],
            "sequential_semantic_confirmation",
        )

        fully_compensated = AgenticPayAdapter(
            context=context,
            history=[{
                "role": "seller",
                "round": 1,
                "content": "<contract>"
                + json.dumps({**counter, "price": 9.5})
                + "</contract>",
            }],
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=10.0,
            self_id="Buyer",
            session_id="v42-preserve",
            counterparty_id="seller",
            sequential_semantic_confirmation=True,
        )
        fully_compensated._classifier_guard_discrete = dict(
            adapter._classifier_guard_discrete
        )
        fully_compensated._classifier_guard_ready = True
        self.assertIsNone(
            fully_compensated._sequential_semantic_confirmation_offer()
        )

    def test_v43_uses_raw_semantic_profile_not_projected_action_profile(self):
        context = {
            "max_price": 100.0,
            "contract_config": {
                "contrainfo": {"product_request": "a service contract"},
                "field_descriptions": {
                    "continuous_terms.commitment_months": (
                        "Number of months the customer must remain committed."
                    ),
                    "discrete_terms.include_service": (
                        "Whether the provider includes the extra service."
                    ),
                },
                "continuous_bounds": {
                    "commitment_months": {"min": 1, "max": 10}
                },
                "discrete_options": {"include_service": [True, False]},
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {"commitment_months": -2.0},
                    "discrete_weights": {
                        "include_service": {"true": 20.0, "false": 0.0}
                    },
                },
            },
        }

        class ClassifierClient(MockClient):
            def __init__(self):
                self.outputs = [
                    '{"role":"CUSTOMER_COMMITMENT_DURATION","confidence":0.95}',
                    '{"role":"CUSTOMER_COMMITMENT_DURATION","confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":3},{"id":"O1","burden":0}],"confidence":0.95}',
                    '{"ratings":[{"id":"O0","burden":3},{"id":"O1","burden":0}],"confidence":0.95}',
                ]

            def generate(self, prompt, **kwargs):
                return self.outputs.pop(0)

        # This counter exactly matches the buyer-safe projected profile, but
        # strongly departs from the raw low-provider-burden profile.
        counter = {
            "price": 70.0,
            "continuous_terms": {"commitment_months": 5.0},
            "discrete_terms": {"include_service": True},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[{
                "role": "seller",
                "round": 1,
                "content": "<contract>" + json.dumps(counter) + "</contract>",
            }],
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v43-raw-profile",
            counterparty_id="seller",
            llm_proposal_candidate=True,
            calibrated_opening_candidate=True,
            ontological_issue_classifier_opening=True,
            ontological_issue_classifier_guard=True,
            compensated_conflict_opening=True,
            sequential_semantic_confirmation=True,
            raw_semantic_confirmation_profile=True,
        )
        classifier = OntologicalIssueClassifier(
            ClassifierClient(), min_confidence=0.75, consensus_passes=2
        )
        adapter.prepare_issue_classifier_opening(adapter.state("learned"), classifier)
        self.assertEqual(
            adapter._classifier_raw_guard_continuous["commitment_months"],
            10.0,
        )
        self.assertFalse(adapter._classifier_raw_guard_discrete["include_service"])
        self.assertNotEqual(
            adapter._classifier_guard_discrete,
            adapter._classifier_raw_guard_discrete,
        )
        result = adapter._sequential_semantic_confirmation_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertEqual(metadata["reference_profile"], "raw_low_burden")
        self.assertEqual(len(metadata["changed_fields"]), 1)
        self.assertEqual(offer.price, 70.0)
        self.assertGreaterEqual(adapter._buyer_utility(offer), 5.0)

        adapter.risk_budgeted_semantic_confirmation = True
        budgeted = adapter._risk_budgeted_semantic_confirmation_offer()
        self.assertIsNotNone(budgeted)
        budgeted_offer, budgeted_metadata = budgeted
        self.assertLess(budgeted_offer.price, counter["price"])
        self.assertEqual(len(budgeted_metadata["changed_fields"]), 2)
        self.assertLessEqual(
            budgeted_metadata["final_semantic_departure_risk"],
            budgeted_metadata["semantic_risk_budget"],
        )
        self.assertTrue(budgeted_metadata["strictly_noncrossing"])
        self.assertGreaterEqual(adapter._buyer_utility(budgeted_offer), 5.0)

    def test_v45_tightens_semantic_budget_after_public_rejection(self):
        context = {
            "max_price": 100.0,
            "contract_config": {
                "contrainfo": {"product_request": "three optional services"},
                "field_descriptions": {
                    "discrete_terms.a": "Whether service A is included.",
                    "discrete_terms.b": "Whether service B is included.",
                    "discrete_terms.c": "Whether service C is included.",
                },
                "continuous_bounds": {},
                "discrete_options": {
                    "a": [True, False],
                    "b": [True, False],
                    "c": [True, False],
                },
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "a": {"true": 5.0, "false": 0.0},
                        "b": {"true": 5.0, "false": 0.0},
                        "c": {"true": 5.0, "false": 0.0},
                    },
                },
            },
        }
        prior_confirmation = {
            "price": 69.99,
            "continuous_terms": {},
            "discrete_terms": {"a": False, "b": False, "c": True},
        }
        restored_counter = {
            "price": 70.0,
            "continuous_terms": {},
            "discrete_terms": {"a": True, "b": False, "c": True},
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "buyer",
                    "round": 1,
                    "content": "<contract>"
                    + json.dumps(prior_confirmation)
                    + "</contract>",
                },
                {
                    "role": "seller",
                    "round": 1,
                    # Positive prose cannot override a different structured
                    # contract: this is still a public counter/rejection.
                    "content": "I accept your offer and can close the deal. <contract>"
                    + json.dumps(restored_counter)
                    + "</contract>",
                },
            ],
            current_state={"current_round": 2, "max_rounds": 20},
            buyer_max_price=100.0,
            self_id="Buyer",
            session_id="v45-rejection-tightening",
            counterparty_id="seller",
            risk_budgeted_semantic_confirmation=True,
            rejection_tightened_semantic_confirmation=True,
        )
        adapter._classifier_raw_guard_continuous = {}
        adapter._classifier_raw_guard_discrete = {
            "a": False,
            "b": False,
            "c": False,
        }
        adapter._classifier_raw_guard_ready = True
        self.assertTrue(adapter._raw_semantic_confirmation_was_rejected())
        result = adapter._risk_budgeted_semantic_confirmation_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertEqual(metadata["semantic_risk_budget"], 0.0)
        self.assertTrue(
            metadata["risk_budget_tightened_after_public_rejection"]
        )
        self.assertEqual(len(metadata["changed_fields"]), 2)
        self.assertEqual(offer.discrete_terms, {"a": False, "b": False, "c": False})
        self.assertLess(offer.price, restored_counter["price"])
        self.assertGreaterEqual(adapter._buyer_utility(offer), 5.0)
        adapter._classifier_discrete_burden_scores = {
            issue: {"true": 1.0, "false": 0.0}
            for issue in ("a", "b", "c")
        }
        adapter.ordinal_frontier_only_when_raw_endpoint_non_ir = True
        self.assertIsNone(adapter._ordinal_burden_frontier_offer())
        adapter.ordinal_frontier_only_when_raw_endpoint_non_ir = False
        adapter.ordinal_intermediate_only = True
        self.assertIsNone(adapter._ordinal_burden_frontier_offer())

    def test_v46_uses_ordinal_intermediate_when_raw_endpoint_breaks_buyer_ir(self):
        context = {
            "max_price": 10.9,
            "contract_config": {
                "contrainfo": {"product_request": "delivered meal"},
                "field_descriptions": {},
                "continuous_bounds": {},
                "discrete_options": {
                    "delivery_speed": ["rush", "standard", "batched"],
                    "extra_condiments": [True, False],
                    "user_product_preference": [
                        "strong_match",
                        "partial_match",
                        "mismatch_or_uncertain",
                    ],
                },
                "buyer_preferences": {
                    "v_base": 10.9,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "delivery_speed": {
                            "rush": 3.0,
                            "standard": 0.0,
                            "batched": -2.0,
                        },
                        "extra_condiments": {"true": 1.5, "false": 0.0},
                        "user_product_preference": {
                            "strong_match": 0.3,
                            "partial_match": 0.12,
                            "mismatch_or_uncertain": -0.25,
                        },
                    },
                },
            },
        }
        prior = {
            "price": 9.49,
            "continuous_terms": {},
            "discrete_terms": {
                "delivery_speed": "rush",
                "extra_condiments": False,
                "user_product_preference": "mismatch_or_uncertain",
            },
        }
        counter = {
            "price": 9.5,
            "continuous_terms": {},
            "discrete_terms": {
                "delivery_speed": "rush",
                "extra_condiments": False,
                "user_product_preference": "partial_match",
            },
        }
        adapter = AgenticPayAdapter(
            context=context,
            history=[
                {
                    "role": "buyer",
                    "round": 2,
                    "content": "<contract>" + json.dumps(prior) + "</contract>",
                },
                {
                    "role": "seller",
                    "round": 2,
                    "content": "I prefer another package. <contract>"
                    + json.dumps(counter)
                    + "</contract>",
                },
            ],
            current_state={"current_round": 3, "max_rounds": 20},
            buyer_max_price=10.9,
            self_id="Buyer",
            session_id="v46-ordinal-frontier",
            counterparty_id="seller",
            risk_budgeted_semantic_confirmation=True,
            rejection_tightened_semantic_confirmation=True,
            ordinal_burden_frontier_confirmation=True,
            ordinal_frontier_only_when_raw_endpoint_non_ir=True,
        )
        adapter._classifier_raw_guard_discrete = {
            "delivery_speed": "batched",
            "extra_condiments": False,
            "user_product_preference": "mismatch_or_uncertain",
        }
        adapter._classifier_raw_guard_ready = True
        adapter._classifier_discrete_burden_scores = {
            "delivery_speed": {
                '"rush"': 1.0,
                '"standard"': 0.5,
                '"batched"': 0.125,
            },
            "extra_condiments": {"true": 0.625, "false": 0.0},
            "user_product_preference": {
                '"strong_match"': 0.5,
                '"partial_match"': 0.25,
                '"mismatch_or_uncertain"': 0.0,
            },
        }
        # The categorical raw endpoint is outside buyer IR at this price.
        raw_offer = CanonicalOffer(
            price=9.49,
            continuous_terms={},
            discrete_terms=dict(adapter._classifier_raw_guard_discrete),
        )
        self.assertLess(adapter._buyer_utility(raw_offer), 0.05 * 10.9)
        result = adapter._ordinal_burden_frontier_offer()
        self.assertIsNotNone(result)
        offer, metadata = result
        self.assertEqual(offer.discrete_terms["delivery_speed"], "standard")
        self.assertFalse(offer.discrete_terms["extra_condiments"])
        self.assertEqual(
            offer.discrete_terms["user_product_preference"],
            "mismatch_or_uncertain",
        )
        self.assertLess(offer.price, counter["price"])
        self.assertGreaterEqual(adapter._buyer_utility(offer), 0.05 * 10.9)
        self.assertLess(
            metadata["selected_ordinal_burden"],
            metadata["counter_ordinal_burden"],
        )
        self.assertTrue(metadata["raw_endpoint_non_ir_gate"])
        self.assertLess(
            metadata["raw_endpoint_buyer_utility"],
            metadata["buyer_utility_reserve"],
        )
        adapter.ordinal_freeze_publicly_rejected_fields = True
        rejection_aware = adapter._ordinal_burden_frontier_offer()
        self.assertIsNotNone(rejection_aware)
        rejection_aware_offer, rejection_aware_metadata = rejection_aware
        self.assertEqual(
            rejection_aware_offer.discrete_terms["delivery_speed"],
            "standard",
        )
        self.assertEqual(
            rejection_aware_offer.discrete_terms["user_product_preference"],
            "partial_match",
        )
        self.assertIn(
            "discrete_terms.user_product_preference",
            rejection_aware_metadata["publicly_rejected_fields_frozen"],
        )
        adapter.ordinal_intermediate_only = True
        intermediate_only = adapter._ordinal_burden_frontier_offer()
        self.assertIsNotNone(intermediate_only)
        intermediate_offer, intermediate_metadata = intermediate_only
        self.assertEqual(
            intermediate_offer.discrete_terms["delivery_speed"],
            "standard",
        )
        self.assertTrue(intermediate_metadata["ordinal_intermediate_only"])

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
