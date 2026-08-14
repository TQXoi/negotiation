import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_formal_results_cn import (  # noqa: E402
    difference_in_differences,
    exact_sign_permutation_p,
    mechanism_metrics,
)


class FormalAnalysisTests(unittest.TestCase):
    def test_exact_sign_randomization_resolution_for_five_runs(self):
        self.assertEqual(exact_sign_permutation_p([1, 1, 1, 1, 1]), 0.0625)
        self.assertEqual(exact_sign_permutation_p([0, 0, 0, 0, 0]), 1.0)

    def test_switch_difference_in_differences_removes_learning_curve(self):
        control = {}
        switched = {}
        for run in range(1, 6):
            for episode in range(1, 21):
                control_reward = 1.0 if episode <= 10 else 2.0
                switch_reward = 1.0 if episode <= 10 else 4.0
                control[(run, episode)] = {
                    "run": run, "episode": episode,
                    "focal_reward": control_reward, "error": None,
                }
                switched[(run, episode)] = {
                    "run": run, "episode": episode,
                    "focal_reward": switch_reward, "error": None,
                }
        result = difference_in_differences(control, switched, 200, 7)
        controlled = result["controlled_post_minus_pre"]
        self.assertEqual(controlled["paired_runs"], 5)
        self.assertEqual(controlled["mean_delta"], 2.0)
        self.assertEqual(controlled["bootstrap_95_ci"], [2.0, 2.0])

    def test_resource_mechanism_uses_true_opponent_utility(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = (
                root / "resource_first" / "framework_v4" / "model_traces"
                / "run_01" / "Player_RED" / "framework_v4_decisions.jsonl"
            )
            trace.parent.mkdir(parents=True)
            # RED asks BLUE to give one X and returns no resources. BLUE values
            # X=2.5, so its true utility is -2.5 and the proposal is infeasible.
            trace.write_text(json.dumps({
                "episode": 1, "decision": 1,
                "posterior": {"x_weight_mean": 0.8},
                "planner": {"chosen_type": "PROPOSE"},
                "chosen_action": {
                    "type": "PROPOSE", "kind": "exploit",
                    "trade": {"RED": {}, "BLUE": {"X": 1}},
                    "belief_use": {"accept_mean": 0.7, "accept_safeguarded": 0.5},
                },
            }) + "\n")
            result = mechanism_metrics(root, "resource_first", "framework_v4")
            self.assertAlmostEqual(result["posterior_truth"], 5 / 6)
            self.assertEqual(result["structurally_infeasible_proposals"], 1)
            self.assertEqual(result["structurally_infeasible_proposal_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
