from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


WORKSPACE = Path(__file__).resolve().parents[1]
AGENTICPAY_ROOT = WORKSPACE / "benchmarks" / "AgenticPay"
for path in (AGENTICPAY_ROOT, WORKSPACE):
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from experiments.run_agenticpay_single28_framework import (
    calculate_contract_diagnostics,
    instrument_env,
    instrument_module_environment_constructors,
    patch_module,
)
from AgenticPay_Env.environment.all_tasks import (
    _focal_buyer_metrics,
    _record_key,
    _normalize_upstream_summary,
    filter_task_paths,
)


class _NoopModelClient:
    """Constructor-compatible stand-in; these tests never call the model."""

    def generate(self, *_args, **_kwargs):
        return ""


class _FakeContractEnv:
    """Minimal unchanged-env stand-in for testing read-only capture."""

    use_contract_mode = True
    buyer_max_price = 20.0
    seller_min_price = 10.0
    price_tolerance = 0.0
    max_rounds = 5
    gamma = 0.99
    contract_config = {"discrete_options": {"speed": ["slow", "fast"]}}
    current_round = 0

    def __init__(self) -> None:
        self.state = SimpleNamespace(
            buyer_price=None,
            seller_price=None,
            metadata={"agreed_contract": None},
        )

    def _extract_price(self, _text: str):
        return None

    def step(self, buyer_action=None, seller_action=None):
        contract = {
            "price": 14.0,
            "continuous_terms": {},
            "discrete_terms": {"speed": "slow"},
        }
        self.state.buyer_price = 15.0
        self.state.seller_price = 14.0
        self.state.metadata.update(
            {
                "buyer_contract": {**contract, "price": 15.0},
                "seller_contract": contract,
                "agreed_contract": contract,
                "buyer_utility": 6.0,
                "seller_utility": 4.0,
            }
        )
        return {}, 1.0, True, False, {"round": 1, "status": "agreed"}


class AgenticPayRunnerInstrumentationTests(unittest.TestCase):
    def test_focal_only_replaces_exactly_one_buyer(self):
        module = SimpleNamespace()
        client = _NoopModelClient()
        patch_module(
            module,
            client=client,
            buyer_client=client,
            seller_client=client,
            # V1 exercises the same focal-only factory path without requiring
            # the separately trained residual-planner checkpoint used by V60.
            variant="universal_framework_v1",
            seller_variant="native",
            max_tokens=64,
            model_alias="mock",
            focal_buyer_index=1,
        )

        focal = module.BuyerAgent(
            model=client,
            name="Buyer 1",
            buyer_max_price=100.0,
        )
        control = module.BuyerAgent(
            model=client,
            name="Buyer 2",
            buyer_max_price=100.0,
        )
        audit = module._negotiation_buyer_patch_audit

        self.assertEqual(focal.__class__.__name__, "UniversalAgenticPayBuyerAgent")
        self.assertEqual(control.__class__.__name__, "BuyerAgent")
        self.assertEqual(audit["buyer_instances_created"], 2)
        self.assertEqual(audit["variant_buyer_instances"], 1)
        self.assertEqual(audit["native_control_buyer_instances"], 1)

    def test_focal_index_is_part_of_resume_identity(self):
        common = ("v60", "only_multi_buyer/Task1.py", "buyer", "seller", "native")
        self.assertNotEqual(_record_key(*common, None), _record_key(*common, 1))

    def test_focal_metrics_preserve_official_score_and_expose_native_reward(self):
        metrics = _focal_buyer_metrics(
            {
                "selected_buyer": 2,
                "buyer1_reward": -0.2,
                "buyer1_max_price": 150.0,
                "buyer_score": 29.403,
            },
            1,
        )
        self.assertEqual(metrics["focal_buyer_reward"], -0.2)
        self.assertFalse(metrics["focal_buyer_selected"])
        self.assertEqual(metrics["focal_buyer_max_price"], 150.0)
        self.assertNotIn("buyer_score", metrics)

    def test_focal_index_must_be_positive(self):
        with self.assertRaises(ValueError):
            patch_module(
                SimpleNamespace(),
                client=_NoopModelClient(),
                variant="repo_native",
                seller_variant="native",
                max_tokens=64,
                model_alias="mock",
                focal_buyer_index=0,
            )

    def test_direct_example_environment_constructor_is_instrumented(self):
        class DirectEnv(_FakeContractEnv):
            @staticmethod
            def resolve_selected_seller(*_args):
                return 2

        DirectEnv.__module__ = "agenticpay.envs.fake_task"
        module = SimpleNamespace(DirectEnv=DirectEnv)
        capture = {"rounds": [], "env": {}}
        wrapped = instrument_module_environment_constructors(module, capture)
        self.assertTrue(isinstance(module.DirectEnv, type))
        self.assertEqual(module.DirectEnv.resolve_selected_seller("", {}), 2)
        env = module.DirectEnv()
        self.assertIsInstance(env, DirectEnv)
        env.step(buyer_action="buyer", seller_action="seller")
        self.assertEqual(wrapped, 1)
        self.assertEqual(len(capture["rounds"]), 1)

    def test_all_tasks_exact_task_filter_keeps_requested_alias(self):
        paths = [
            WORKSPACE
            / "benchmarks/AgenticPay/agenticpay/examples/only_multi_seller"
            / name
            for name in ("Task1_alpha.py", "Task2_beta.py", "Task10_gamma.py")
        ]
        selected = filter_task_paths(paths, "Task2,Task10_gamma")
        self.assertEqual([path.stem for path in selected], ["Task2_beta", "Task10_gamma"])

    def test_nested_multi_product_summary_gets_explicit_macro_scores(self):
        normalized = _normalize_upstream_summary(
            {
                "success": True,
                "product_results": [
                    {
                        "buyer_score": 70.0,
                        "seller_score": 40.0,
                        "global_score": 60.0,
                        "buyer_reward": 10.0,
                        "seller_reward": 5.0,
                        "rounds": 2,
                    },
                    {
                        "buyer_score": 90.0,
                        "seller_score": 50.0,
                        "global_score": 80.0,
                        "buyer_reward": 20.0,
                        "seller_reward": 7.0,
                        "rounds": 3,
                    },
                ],
            }
        )
        self.assertEqual(normalized["buyer_score"], 80.0)
        self.assertEqual(normalized["seller_score"], 45.0)
        self.assertEqual(normalized["global_score"], 70.0)
        self.assertEqual(normalized["buyer_reward"], 30.0)
        self.assertEqual(normalized["seller_reward"], 12.0)
        self.assertEqual(normalized["total_rounds"], 5.0)
        self.assertEqual(
            normalized["score_aggregation"],
            "macro_mean_over_product_results",
        )

    def test_terminal_contract_state_is_captured_without_changing_step(self):
        capture = {"rounds": [], "env": {}}
        env = instrument_env(_FakeContractEnv(), capture)
        buyer = (
            '<contract>{"price":15,"continuous_terms":{},'
            '"discrete_terms":{"speed":"slow"}}</contract>'
        )
        seller = (
            '<contract>{"price":14,"continuous_terms":{},'
            '"discrete_terms":{"speed":"slow"}}</contract>'
        )
        _observation, reward, terminated, truncated, _info = env.step(
            buyer_action=buyer,
            seller_action=seller,
        )

        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(capture["env"]["agreed_contract"]["price"], 14.0)
        self.assertEqual(capture["env"]["seller_utility"], 4.0)
        self.assertEqual(
            capture["rounds"][0]["buyer_contract_extracted"]["price"],
            15.0,
        )
        self.assertEqual(
            capture["rounds"][0]["seller_contract_extracted"]["price"],
            14.0,
        )

    def test_contract_diagnostics_count_json_serialized_boolean_weights(self):
        """Persisted true/false keys must match native bool contract values."""

        record = {
            "contract_config": {
                "buyer_preferences": {
                    "v_base": 100.0,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "utilities_included": {"true": 20.0, "false": 0.0}
                    },
                },
                "seller_preferences": {
                    "c_base": 70.0,
                    "continuous_weights": {},
                    "discrete_weights": {
                        "utilities_included": {"true": -5.0, "false": 0.0}
                    },
                },
                "continuous_bounds": {},
                "discrete_options": {"utilities_included": [True, False]},
            },
            "agreed_contract": {
                "price": 90.0,
                "continuous_terms": {},
                "discrete_terms": {"utilities_included": True},
            },
        }

        diagnostics = calculate_contract_diagnostics(record)
        self.assertEqual(diagnostics["contract_buyer_utility"], 30.0)
        self.assertEqual(diagnostics["contract_seller_utility"], 15.0)
        self.assertEqual(diagnostics["contract_z_max"], 45.0)
        self.assertTrue(diagnostics["contract_score_feasible"])
        self.assertFalse(diagnostics["contract_ir_violation"])


if __name__ == "__main__":
    unittest.main()
