from __future__ import annotations

import json
from pathlib import Path
import unittest

from Simple_Env.buyer.continuous_solver.identifiable import (
    ActiveProbePlanner,
    BehaviorHypothesis,
    ChangePointBehaviorBelief,
    RegimeMixtureBelief,
    SellerBehaviorModel,
    build_profile_suite,
    categorical_js_divergence,
    normalize_public_outcome,
)
from Simple_Env.buyer.continuous_solver.planner.params import PlannerParams


class IdentifiableFrameworkTests(unittest.TestCase):
    def test_profile_suite_passes_oracle_gate(self):
        suite = build_profile_suite(min_js_bits=0.12)
        self.assertGreaterEqual(len(suite), 3)
        self.assertTrue(all(pair.identifiable for pair in suite))
        self.assertTrue(all(pair.oracle_js_bits >= 0.12 for pair in suite))

    def test_behaviorally_identical_profiles_have_zero_divergence(self):
        model = SellerBehaviorModel(65.0, 3.5)
        response = model.response_distribution(60.0)
        self.assertAlmostEqual(categorical_js_divergence(response, response), 0.0)

    def test_active_probe_prefers_behavior_separating_offer(self):
        old = SellerBehaviorModel(55.0, 3.5)
        new = SellerBehaviorModel(80.0, 3.5)
        planner = ActiveProbePlanner(surplus_cost_weight=0.02)
        chosen = planner.choose(
            old=old,
            new=new,
            candidate_offers=range(30, 96, 5),
            buyer_budget=95.0,
        )
        self.assertGreater(chosen.information_gain_bits, 0.5)
        self.assertGreater(chosen.offer, 55.0)
        self.assertLess(chosen.offer, 80.0)

    def test_regime_mixture_detects_repeated_new_regime_responses(self):
        old = SellerBehaviorModel(55.0, 2.5)
        new = SellerBehaviorModel(80.0, 2.5)
        belief = RegimeMixtureBelief(old=old, new=new, hazard=0.02)
        before = belief.change_probability
        for _ in range(4):
            belief.update(offer=67.0, outcome="counter")
        self.assertGreater(belief.change_probability, before)
        self.assertGreater(belief.change_probability, 0.8)

    def test_reject_is_nonterminal_counter_evidence(self):
        self.assertEqual(normalize_public_outcome("reject"), "counter")

    def test_deployable_change_filter_detects_shift_without_oracle_label(self):
        hypotheses = [
            BehaviorHypothesis("low", SellerBehaviorModel(55.0, 2.5)),
            BehaviorHypothesis("high", SellerBehaviorModel(80.0, 2.5)),
        ]
        belief = ChangePointBehaviorBelief(hypotheses, hazard=0.02)
        for _ in range(5):
            belief.update(offer=67.0, outcome="accept")
        pre_change = belief.change_probability
        for _ in range(5):
            belief.update(offer=67.0, outcome="counter")
        self.assertLess(pre_change, 0.25)
        self.assertGreater(belief.change_probability, 0.8)
        self.assertGreater(belief.expected_reservation(), 70.0)

    def test_posterior_probe_uses_hypothesis_library_not_true_profile(self):
        hypotheses = [
            BehaviorHypothesis("low", SellerBehaviorModel(55.0, 3.5)),
            BehaviorHypothesis("high", SellerBehaviorModel(80.0, 3.5)),
        ]
        belief = ChangePointBehaviorBelief(hypotheses, hazard=0.02)
        chosen = ActiveProbePlanner(surplus_cost_weight=0.02).choose_from_belief(
            belief=belief,
            candidate_offers=range(30, 96, 5),
            buyer_budget=95.0,
        )
        self.assertGreater(chosen.information_gain_bits, 0.5)
        self.assertGreater(chosen.offer, 55.0)
        self.assertLess(chosen.offer, 80.0)

    def test_frozen_planner_snapshot_matches_current_defaults(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "buyer/continuous_solver/identifiable/frozen_planner.json"
        )
        frozen = json.loads(path.read_text(encoding="utf-8"))["params"]
        self.assertEqual(frozen, PlannerParams().to_dict())


if __name__ == "__main__":
    unittest.main()
