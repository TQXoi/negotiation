import json
import tempfile
import unittest

from arena_integration.repeated_agents import (
    DualTimescaleBeliefPlannerAgent,
    OpponentSimulationAgent,
    RepeatedLanguageAgent,
    _canonicalize_protocol_response,
    _json_from_text,
    _legacy_protocol_valid,
    _protocol_valid,
)
from arena_integration.decision_calibrated_agent import (
    ActionSpaceAdaptiveCommitmentBeliefPlannerAgent,
    CrossEpisodeInformationPlannerAgent,
    DecisionRelevantInformationPlannerAgent,
    EndogeneityCalibratedInformationPlannerAgent,
    FactorizedPreferencePolicyBeliefPlannerAgent,
    ReliabilityGatedFactorizedBeliefPlannerAgent,
    DecisionCalibratedBeliefPlannerAgent,
    ReciprocalCommitmentBeliefPlannerAgent,
    ScopedCommitmentBeliefPlannerAgent,
    UncertaintySafeguardedBeliefPlannerAgent,
    parse_trade_text,
)
from arena_integration.paper_aligned_opponent_simulation import PaperAlignedOpponentSimulationAgent
from arena_integration.repeated_games import ValueMaximisationGoal
from arena_integration.repeated_games import PaperRepeatedTradingGame
from arena_integration.run_opponent_simulation_setting import (
    focal_index,
    is_transport_failure,
    runs_requiring_restart,
    summarize,
    strategic_brainstorming_prompt,
)
from arena_integration.summarize_repeated_comparison import metrics, paired_run_deltas
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation
from games.buy_sell_game.game import BuySellGameDefaultParser


VALID_RESPONSE = """<proposal count> 1 </proposal count>
<my resources> ZUP: 1000 </my resources>
<my goals> test </my goals>
<reason> test </reason>
<player answer> PROPOSAL </player answer>
<newly proposed trade> Player RED Gives X: 1 | Player BLUE Gives ZUP: 50 </newly proposed trade>
<message> Offer 50. </message>"""

VALID_RESOURCE_RESPONSE = """<my name> Player RED </my name>
<my resources> X: 25, Y: 5 </my resources>
<my goals> maximize value </my goals>
<reason> complementary exchange </reason>
<player answer> NONE </player answer>
<message> Trade five units each. </message>
<newly proposed trade> Player RED Gives X: 5 | Player BLUE Gives Y: 5 </newly proposed trade>"""


class StaticAgent(Agent):
    def __init__(self, name, responses):
        super().__init__(name)
        self.responses = list(responses)
        self.conversation = []
        self.prompt_entity_initializer = "system"

    def init_agent(self, system_prompt, role):
        self.conversation = [{"role": "system", "content": system_prompt + role}]

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": str(message)})

    def chat(self):
        return self.responses.pop(0)

    def get_state(self):
        return {"class": self.__class__.__name__, "agent_name": self.agent_name, "conversation": self.conversation}


class RepeatedIntegrationTests(unittest.TestCase):
    def test_error_run_resume_restarts_whole_repeated_game(self):
        latest = {
            (1, 1): {"run": 1, "episode": 1, "error": None},
            (1, 2): {"run": 1, "episode": 2, "error": "URLError: refused"},
            (2, 1): {"run": 2, "episode": 1, "error": None},
        }
        self.assertEqual(runs_requiring_restart(latest, True), {1})
        self.assertEqual(runs_requiring_restart(latest, False), set())
        self.assertTrue(is_transport_failure(latest[(1, 2)]))
        self.assertFalse(is_transport_failure({"error": "ValueError: bad protocol"}))

    def test_framework_diagnostics_calibration_and_belief_interventions(self):
        agent = CrossEpisodeInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.last_own_action = {
            "type": "PROPOSE",
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 50",
            "response": {"accept": 0.7},
        }
        agent._record_response_calibration({"answer": "ACCEPT"})
        self.assertAlmostEqual(agent.episode_calibration[0]["brier"], 0.09)

        learned_buy = [1.0 / 101.0] * 101
        learned_resource = [1.0 / 21.0] * 21
        agent._set_decision_belief("oracle", learned_buy, learned_resource)
        self.assertEqual(agent._posterior_json()["mean"], 43.0)
        agent._set_decision_belief("wrong_confident", learned_buy, learned_resource)
        self.assertEqual(agent._posterior_json()["mean"], 57.0)
        agent._set_decision_belief("shuffled", learned_buy, learned_resource)
        self.assertAlmostEqual(sum(agent.buy_probs), 1.0)

    def test_summary_reports_calibration_action_flip_and_switch_metrics(self):
        rows = []
        for episode, switched in ((10, False), (11, True)):
            rows.append(
                {
                    "run": 1, "episode": episode, "agreement": True,
                    "focal_reward": 10.0 if not switched else 6.0,
                    "opponent_reward": 10.0, "joint_reward": 20.0, "turns": 2,
                    "focal_model_calls": 1, "opponent_model_calls": 1,
                    "focal_protocol": {}, "opponent_protocol": {}, "price": 53,
                    "opponent_switched": switched, "error": None,
                    "focal_diagnostics": {
                        "calibration": [{
                            "predicted_accept": 0.8, "accepted": 1.0,
                            "brier": 0.04, "nll": -__import__("math").log(0.8),
                        }],
                        "belief_abs_error": 2.0,
                        "belief_q10_q90_covered": True,
                        "action_decisions": 2,
                        "action_flip_opportunities": 1,
                        "action_flip_counts": {
                            "frozen": 1, "wrong_confident": 0,
                            "shuffled": 1, "oracle": 0,
                        },
                    },
                }
            )
        result = summarize(rows)
        self.assertAlmostEqual(result["belief_accept_brier"], 0.04)
        self.assertAlmostEqual(result["action_flip_rate"]["frozen"], 1.0)
        self.assertEqual(result["run_metrics"][0]["pre_switch_reward"], 10.0)
        self.assertEqual(result["run_metrics"][0]["post_switch_reward"], 6.0)

    def test_policy_protocol_failure_is_zero_reward_not_infrastructure_error(self):
        rows = [{
            "run": 1, "episode": 1, "agreement": False,
            "focal_reward": 0.0, "opponent_reward": 0.0, "joint_reward": 0.0,
            "turns": 3, "focal_model_calls": 1, "opponent_model_calls": 2,
            "focal_protocol": {}, "opponent_protocol": {"repair_failures": 1},
            "focal_diagnostics": None, "opponent_switched": False,
            "policy_protocol_failure": {
                "type": "ProtocolFormatError", "actor": "opponent",
            },
            "error": None,
        }]
        result = summarize(rows)
        self.assertEqual(result["valid_episodes"], 1)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["infrastructure_errors"], 0)
        self.assertEqual(result["policy_protocol_failures"], 1)
        self.assertEqual(result["format_success_rate"], 0.0)
        self.assertEqual(result["mean_focal_reward"], 0.0)

    def test_paper_setting_roles_and_released_strategy_prompt(self):
        self.assertEqual(focal_index("buyer"), 1)
        self.assertEqual(focal_index("seller"), 0)
        self.assertEqual(focal_index("resource_first"), 0)
        self.assertEqual(focal_index("resource_second"), 1)
        prompt = strategic_brainstorming_prompt("reward rule")
        self.assertIn("explicitly enumerate 5", prompt)
        self.assertIn("at every iteration/turn", prompt)
        self.assertIn("<strategy declaration>", prompt)

    def test_comparison_metrics_count_errors_as_zero(self):
        rows = {
            (1, 1): {"agreement": True, "focal_reward": 10.0, "joint_reward": 12.0,
                     "turns": 2, "focal_model_calls": 1, "error": None},
            (1, 2): {"agreement": False, "error": "format failure"},
        }
        result = metrics(rows, expected_episodes=4)
        self.assertEqual(result["completion_rate"], 0.5)
        self.assertEqual(result["format_success_rate"], 0.5)
        self.assertEqual(result["all_episode_mean_focal_reward"], 5.0)
        self.assertEqual(result["all_episode_agreement_rate"], 0.5)
        self.assertEqual(paired_run_deltas(rows, rows), [0.0])

    def test_json_extraction(self):
        self.assertEqual(_json_from_text('```json\n{"chosen_id": 2}\n```')["chosen_id"], 2)

    def test_candidate_protocol_guard(self):
        self.assertTrue(_protocol_valid(VALID_RESPONSE, "buyer_seller"))
        self.assertFalse(_protocol_valid(VALID_RESPONSE.replace("PROPOSAL", "MAYBE"), "buyer_seller"))
        self.assertFalse(_protocol_valid(VALID_RESPONSE.replace(" | ", ", "), "buyer_seller"))
        self.assertTrue(_legacy_protocol_valid(VALID_RESPONSE.replace(" | ", ", "), "buyer_seller"))

    def test_buy_sell_protocol_canonicalizer_repairs_delimiter(self):
        malformed = VALID_RESPONSE.replace(" | ", ", ")
        repaired = _canonicalize_protocol_response(malformed, "buyer_seller")
        self.assertIsNotNone(repaired)
        self.assertTrue(_protocol_valid(repaired, "buyer_seller"))
        self.assertIn("Player RED Gives X: 1 | Player BLUE Gives ZUP: 50", repaired)

    def test_buy_sell_protocol_canonicalizer_does_not_guess_missing_price(self):
        malformed = VALID_RESPONSE.replace("ZUP: 50", "ZUP:")
        self.assertIsNone(_canonicalize_protocol_response(malformed, "buyer_seller"))

    def test_resource_protocol_canonicalizer_recovers_missing_answer_from_trade(self):
        malformed = VALID_RESOURCE_RESPONSE.replace(
            "<player answer> NONE </player answer>\n", ""
        )
        repaired = _canonicalize_protocol_response(malformed, "resource_exchange")
        self.assertIsNotNone(repaired)
        self.assertTrue(_protocol_valid(repaired, "resource_exchange"))
        self.assertIn("<player answer> NONE </player answer>", repaired)

    def test_resource_protocol_canonicalizer_reorders_complete_bundle(self):
        reversed_trade = VALID_RESOURCE_RESPONSE.replace(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5",
            "Player BLUE Gives Y: 5 | Player RED Gives X: 5",
        )
        repaired = _canonicalize_protocol_response(reversed_trade, "resource_exchange")
        self.assertIsNotNone(repaired)
        self.assertTrue(_protocol_valid(repaired, "resource_exchange"))

    def test_resource_protocol_canonicalizer_repairs_amount_before_resource(self):
        compact_trade = VALID_RESOURCE_RESPONSE.replace(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5",
            "Player RED Gives 5X | Player BLUE Gives 5Y",
        )
        self.assertFalse(_protocol_valid(compact_trade, "resource_exchange"))
        repaired = _canonicalize_protocol_response(compact_trade, "resource_exchange")
        self.assertIsNotNone(repaired)
        self.assertTrue(_protocol_valid(repaired, "resource_exchange"))
        self.assertIn(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5", repaired
        )
        self.assertIn(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5", repaired
        )

    def test_protocol_validator_rejects_unparseable_private_resources(self):
        malformed = VALID_RESPONSE.replace("ZUP: 1000", "one thousand ZUP")
        self.assertFalse(_protocol_valid(malformed, "buyer_seller"))

    def test_buy_sell_public_message_uses_opponent_facing_tags(self):
        public = BuySellGameDefaultParser().parse(VALID_RESPONSE).message_to_other_player()
        self.assertIn("<other player message> Offer 50.", public)
        self.assertIn("<other player answer> PROPOSAL", public)
        self.assertIn(
            "<other player proposed trade> Player RED Gives X: 1 | "
            "Player BLUE Gives ZUP: 50",
            public,
        )

        buyer = DecisionCalibratedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        buyer.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        buyer.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        response = buyer.step(public)
        self.assertTrue(_protocol_valid(response, "buyer_seller"))
        self.assertEqual(buyer.last_opponent_trade["BLUE"]["ZUP"], 50)
        self.assertEqual(buyer._posterior_json()["evidence_count"], 1)

    def test_protocol_safe_agent_repairs_before_conversation_update(self):
        agent = RepeatedLanguageAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward = 63 - price",
        )
        agent.init_agent("game rules", "You are Player BLUE")
        agent.chat = lambda: VALID_RESPONSE.replace(" | ", ", ")
        repaired = agent.think()
        self.assertTrue(_protocol_valid(repaired, "buyer_seller"))
        self.assertEqual(agent.protocol_invalid_raw_count, 1)
        self.assertEqual(agent.protocol_deterministic_repair_count, 1)
        self.assertEqual(agent.conversation[-1]["content"], repaired)

    def test_repeated_agent_matches_first_mover_message_layout(self):
        starter = RepeatedLanguageAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY",
        )
        starter.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward = 63 - price", starts_episode=True,
        )
        starter.init_agent("game rules", "You are Player BLUE.")
        self.assertEqual([row["role"] for row in starter.conversation], ["system", "user"])

        follower = RepeatedLanguageAgent(
            agent_name="Player RED", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY",
        )
        follower.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="seller reward = price - 43", starts_episode=False,
        )
        follower.init_agent("game rules", "You are Player RED.")
        self.assertEqual([row["role"] for row in follower.conversation], ["system"])

    def test_value_goal_returns_net_utility(self):
        initial = Resources({"X": 25, "Y": 5})
        goal = ValueMaximisationGoal(initial, Valuation({"X": 0.5, "Y": 2.5}))
        self.assertEqual(goal.goal_reached(Resources({"X": 20, "Y": 10})), 10.0)
        json.dumps(goal.json())

    def test_framework_runs_belief_candidates_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = DualTimescaleBeliefPlannerAgent(
                agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
                api_key="EMPTY", trace_dir=tmp, candidate_count=1,
            )
            agent.prepare_episode(
                episode_index=1, total_episodes=20, game_kind="buyer_seller",
                private_objective="buyer reward = 63 - price",
            )
            agent.init_agent("game rules", "You are Player BLUE")
            outputs = {
                "framework_continuous_belief_update": json.dumps(agent._empty_posterior()),
                "framework_posterior_conditioned_candidates": json.dumps(
                    {"candidates": [{"id": 1, "kind": "safe", "rationale": "x", "response": VALID_RESPONSE}]}
                ),
                "framework_belief_usable_planning": json.dumps(
                    {"scores": [{"id": 1, "total": 1}], "chosen_id": 1, "decision_reason": "safe"}
                ),
            }
            calls = []
            agent._call = lambda stage, messages, **kwargs: calls.append(stage) or outputs[stage]
            self.assertEqual(agent.step("seller offered 52"), VALID_RESPONSE)
            self.assertEqual(calls, [
                "framework_continuous_belief_update",
                "framework_posterior_conditioned_candidates",
                "framework_belief_usable_planning",
            ])
            self.assertEqual(len(agent.posterior_history), 1)

    def test_opponent_simulation_uses_three_stages(self):
        agent = OpponentSimulationAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", candidate_count=1,
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward = 63 - price",
        )
        agent.init_agent("game rules", "You are Player BLUE")
        outputs = {
            "oppsim_candidate_generation": json.dumps(
                {"candidates": [{"id": 1, "strategy": "safe", "response": VALID_RESPONSE}]}
            ),
            "oppsim_opponent_model": "seller anchors high and concedes slowly",
            "oppsim_future_rollout_and_selection": json.dumps(
                {"evaluations": [{"id": 1, "predicted_reward": 10}], "chosen_id": 1}
            ),
        }
        calls = []
        agent._call = lambda stage, messages, **kwargs: calls.append(stage) or outputs[stage]
        self.assertEqual(agent.step("seller offered 52"), VALID_RESPONSE)
        self.assertEqual(calls, [
            "oppsim_candidate_generation", "oppsim_opponent_model",
            "oppsim_future_rollout_and_selection",
        ])

    def test_paper_aligned_opponent_simulation_uses_independent_rollouts(self):
        agent = PaperAlignedOpponentSimulationAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", candidate_count=2,
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward = 63 - price",
        )
        agent.init_agent("game rules", "You are Player BLUE")
        second = VALID_RESPONSE.replace("ZUP: 50", "ZUP: 48").replace("Offer 50", "Offer 48")
        outputs = {
            "oppsim_paper_candidate_sample_1": VALID_RESPONSE,
            "oppsim_paper_candidate_sample_2": second,
            "oppsim_paper_independent_rollout_1": json.dumps(
                {"predicted_focal_reward": 9, "agreement_probability": 0.9}
            ),
            "oppsim_paper_independent_rollout_2": json.dumps(
                {"predicted_focal_reward": 12, "agreement_probability": 0.7}
            ),
        }
        calls = []
        agent._call = lambda stage, messages, **kwargs: calls.append(stage) or outputs[stage]
        self.assertEqual(agent.step("seller offered 52"), second)
        self.assertEqual(
            calls,
            [
                "oppsim_paper_candidate_sample_1",
                "oppsim_paper_candidate_sample_2",
                "oppsim_paper_independent_rollout_1",
                "oppsim_paper_independent_rollout_2",
            ],
        )

    def test_v2_trade_parser_is_safe_and_structured(self):
        trade = parse_trade_text("Player RED Gives X: 3 | Player BLUE Gives Y: 2")
        self.assertEqual(trade, {"RED": {"X": 3}, "BLUE": {"Y": 2}})
        self.assertIsNone(parse_trade_text("NONE"))
        self.assertIsNone(parse_trade_text("Player RED Gives X: 3"))

    def test_framework_v2_buy_sell_is_action_locked_and_updates_posterior(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = DecisionCalibratedBeliefPlannerAgent(
                agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
                api_key="EMPTY", trace_dir=tmp, language_realizer=False,
                semantic_belief=False, protocol_mode="normalize",
            )
            agent.prepare_episode(
                episode_index=1, total_episodes=20, game_kind="buyer_seller",
                private_objective="buyer reward is private WTP 63 minus deal price",
            )
            agent.init_agent(
                "<my resources> ZUP: 1000 </my resources>"
                "<my goals> buy X </my goals>",
                "You are Player BLUE.",
            )
            public = (
                "<other player message> I offer 52. </other player message>"
                "<other player answer> PROPOSAL </other player answer>"
                "<other player proposed trade> "
                "Player RED Gives X: 1 | Player BLUE Gives ZUP: 52 "
                "</other player proposed trade>"
            )
            response = agent.step(public)
            self.assertTrue(_protocol_valid(response, "buyer_seller"))
            self.assertEqual(agent._posterior_json()["evidence_count"], 1)
            self.assertEqual(len(agent.posterior_history), 1)
            self.assertIsNotNone(agent.last_own_action)

    def test_framework_v2_resource_uses_same_core_and_exact_utility(self):
        agent = DecisionCalibratedBeliefPlannerAgent(
            agent_name="Player RED", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="resource_exchange",
            private_objective="maximize net resource value with private values X=0.5 and Y=2.5",
        )
        agent.init_agent(
            "<my resources> X: 25, Y: 5 </my resources>"
            "<my goals> maximize value </my goals><my name> Player RED </my name>",
            "You are Player RED.",
        )
        response = agent.step("")
        self.assertTrue(_protocol_valid(response, "resource_exchange"))
        chosen_trade = agent.last_own_action["trade"]
        self.assertGreater(agent._self_utility(chosen_trade), 0)
        self.assertEqual(agent._posterior_json()["kind"], "opponent_relative_resource_value")

    def test_framework_v2_uses_bounded_semantic_evidence_not_llm_posterior(self):
        agent = DecisionCalibratedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=True,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        calls = []
        agent._call = lambda stage, messages, **kwargs: calls.append(stage) or json.dumps(
            {
                "explicit_reservation_claim": 43,
                "concession_signal": 0.25,
                "firmness": 0.8,
                "evidence_quote_or_paraphrase": "My cost is 43.",
                "reason": "explicit claim",
            }
        )
        public = (
            "<other player message> My production cost is 43. </other player message>"
            "<other player answer> PROPOSAL </other player answer>"
            "<other player proposed trade> Player RED Gives X: 1 | "
            "Player BLUE Gives ZUP: 52 </other player proposed trade>"
        )
        response = agent.step(public)
        self.assertTrue(_protocol_valid(response, "buyer_seller"))
        self.assertEqual(calls, ["framework_v2_semantic_belief_evidence"])
        self.assertTrue(agent.semantic_evidence[-1]["applied_claim"])
        self.assertLess(agent.semantic_evidence[-1]["bounded_reliability"], 0.36)

    def test_framework_v2_respects_arena_proposal_limit(self):
        agent = DecisionCalibratedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize", game_turn_limit=10,
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        for count in range(1, 5):
            prior = VALID_RESPONSE.replace(
                "<proposal count> 1", f"<proposal count> {count}"
            )
            agent.conversation.append({"role": "assistant", "content": prior})
        public = (
            "<other player message> Final offer. </other player message>"
            "<other player answer> PROPOSAL </other player answer>"
            "<other player proposed trade> Player RED Gives X: 1 | "
            "Player BLUE Gives ZUP: 62 </other player proposed trade>"
        )
        response = agent.step(public)
        self.assertIn("<player answer> ACCEPT </player answer>", response)
        self.assertIn("<newly proposed trade> NONE </newly proposed trade>", response)

    def test_framework_v3_uncertainty_changes_response_forecast_and_blocks_over_anchor(self):
        agent = UncertaintySafeguardedBeliefPlannerAgent(
            agent_name="Player RED", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="seller reward is deal price minus private production cost 43",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> X: 1 </my resources><my goals> sell X </my goals>",
            "You are Player RED.",
        )
        agent.decision_index = 1
        event = {
            "answer": "PROPOSAL",
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 43}},
            "message": "Opening offer.",
        }
        agent._update_from_event(event)
        forecast = agent._response_distribution(
            {"RED": {"X": 1}, "BLUE": {"ZUP": 70}}
        )
        self.assertLess(forecast["accept"], forecast["accept_mean"])
        candidates = agent._build_candidates()
        self.assertTrue(all(row["safeguard_eligible"] for row in candidates))
        chosen, planner = agent._choose(candidates, event)
        self.assertEqual(chosen["type"], "PROPOSE")
        self.assertLessEqual(chosen["trade"]["BLUE"]["ZUP"], 63)
        self.assertEqual(planner["risk_gate"]["mode"], "posterior_tail_safeguard")

    def test_framework_v3_outside_option_dominates_negative_proposal(self):
        agent = UncertaintySafeguardedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="resource_exchange",
            private_objective="maximize net resource value with private values X=2.5 and Y=0.5",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> X: 5, Y: 25 </my resources><my goals> maximize value </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"Y": 1}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives Y: 1",
            "self_utility": 2.0,
            "response": {"accept": 0.1, "accept_mean": 0.2},
            "safeguarded_contingent_value": -0.1,
            "kind": "exploit",
        }
        chosen, planner = agent._choose([candidate], event=None)
        self.assertEqual(chosen["type"], "WAIT")
        self.assertEqual(chosen["kind"], "outside_option_dominates_proposals")
        self.assertEqual(planner["outside_option"], 0.0)

    def test_framework_v3_agreement_regret_prefers_certain_positive_offer(self):
        agent = UncertaintySafeguardedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 2
        event = {"trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 50}}}
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 45}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 45",
            "self_utility": 18.0,
            "response": {"accept": 0.2, "accept_mean": 0.5},
            "safeguarded_contingent_value": 15.0,
            "kind": "exploit",
        }
        chosen, planner = agent._choose([candidate], event)
        self.assertEqual(chosen["type"], "ACCEPT")
        self.assertEqual(chosen["self_utility"], 13.0)
        self.assertLess(planner["counter_value_after_regret"], 13.0)

    def test_framework_v3_1_commitment_floor_blocks_repeated_deterioration(self):
        agent = ReciprocalCommitmentBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.episode_records = [
            {"agreement": True, "own_reward": reward} for reward in (16, 16, 15)
        ]
        agent.prepare_episode(
            episode_index=4, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        event = {"trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 54}}}
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 48}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 48",
            "self_utility": 15.0,
            "response": {"accept": 0.6, "accept_mean": 0.8},
            "safeguarded_contingent_value": 8.0,
            "kind": "reciprocal",
        }
        chosen, planner = agent._choose([candidate], event)
        self.assertEqual(chosen["type"], "PROPOSE")
        self.assertEqual(planner["historical_commitment_floor"], 15.0)
        self.assertEqual(
            planner["reason"], "reciprocal_commitment_floor_blocks_deteriorating_offer"
        )

    def test_framework_v3_1_opening_exploit_not_removed_by_safe_gate(self):
        agent = ReciprocalCommitmentBeliefPlannerAgent(
            agent_name="Player RED", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="resource_exchange",
            private_objective="maximize net resource value with private values X=0.5 and Y=2.5",
            starts_episode=True,
        )
        agent.init_agent(
            "<my resources> X: 25, Y: 5 </my resources><my goals> maximize value </my goals>",
            "You are Player RED.",
        )
        agent.decision_index = 1
        candidates = agent._build_candidates()
        exploit = next(row for row in candidates if row["kind"] == "exploit")
        self.assertEqual(exploit["trade_text"], "Player RED Gives X: 10 | Player BLUE Gives Y: 10")
        self.assertFalse(exploit["safeguard_eligible"])

    def test_framework_v3_2_ungated_exploit_is_scoped_to_resource_opening(self):
        buyer = ScopedCommitmentBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        buyer.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        buyer.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        buyer.decision_index = 1
        buyer_candidates = buyer._build_candidates()
        buyer_exploit = next(row for row in buyer_candidates if row["kind"] == "exploit")
        self.assertTrue(buyer_exploit["safeguard_eligible"])
        self.assertEqual(buyer_exploit["frontier_scope"], "response_supported")

        resource = ScopedCommitmentBeliefPlannerAgent(
            agent_name="Player RED", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        resource.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="resource_exchange",
            private_objective="maximize net resource value with private values X=0.5 and Y=2.5",
            starts_episode=True,
        )
        resource.init_agent(
            "<my resources> X: 25, Y: 5 </my resources><my goals> maximize value </my goals>",
            "You are Player RED.",
        )
        resource.decision_index = 1
        resource_candidates = resource._build_candidates()
        resource_exploit = next(row for row in resource_candidates if row["kind"] == "exploit")
        self.assertFalse(resource_exploit["safeguard_eligible"])
        self.assertEqual(resource_exploit["frontier_scope"], "resource_opening_full")

    def test_framework_v3_3_frontier_scope_follows_action_space_structure(self):
        buyer = ActionSpaceAdaptiveCommitmentBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        buyer.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        buyer.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        buyer.decision_index = 2
        buyer_exploit = next(row for row in buyer._build_candidates() if row["kind"] == "exploit")
        self.assertTrue(buyer_exploit["safeguard_eligible"])
        self.assertEqual(buyer_exploit["frontier_scope"], "response_supported")

        resource = ActionSpaceAdaptiveCommitmentBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        resource.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="resource_exchange",
            private_objective="maximize net resource value with private values X=2.5 and Y=0.5",
            starts_episode=False,
        )
        resource.init_agent(
            "<my resources> X: 5, Y: 25 </my resources><my goals> maximize value </my goals>",
            "You are Player BLUE.",
        )
        resource.decision_index = 3
        resource_exploit = next(
            row for row in resource._build_candidates() if row["kind"] == "exploit"
        )
        self.assertFalse(resource_exploit["safeguard_eligible"])
        self.assertEqual(
            resource_exploit["frontier_scope"], "combinatorial_action_space_full"
        )

    def test_framework_v4_cross_episode_information_can_replace_early_accept(self):
        agent = CrossEpisodeInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        observed = {"RED": {"X": 1}, "BLUE": {"ZUP": 55}}
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 53}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 53",
            "self_utility": 10.0,
            "response": {"accept": 0.3, "accept_mean": 0.75},
            "decision_value_of_information": 1.7,
            "safeguarded_contingent_value": 3.3,
            "safeguard_eligible": True,
            "kind": "probe",
        }
        chosen, planner = agent._choose([candidate], {"trade": observed})
        self.assertEqual(chosen["type"], "PROPOSE")
        self.assertEqual(planner["reason"], "cross_episode_information_value_justifies_probe")
        self.assertGreater(planner["cross_episode_information_value"], 0.0)

    def test_framework_v4_information_bonus_vanishes_at_final_episode(self):
        agent = CrossEpisodeInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=10, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        observed = {"RED": {"X": 1}, "BLUE": {"ZUP": 55}}
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 53}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 53",
            "self_utility": 10.0,
            "response": {"accept": 0.3, "accept_mean": 0.75},
            "decision_value_of_information": 1.7,
            "safeguarded_contingent_value": 3.3,
            "safeguard_eligible": True,
            "kind": "probe",
        }
        chosen, planner = agent._choose([candidate], {"trade": observed})
        self.assertEqual(chosen["type"], "ACCEPT")
        self.assertEqual(planner["cross_episode_information_value"], 0.0)

    def test_framework_v5_tempers_opponent_initiated_offer_evidence(self):
        def make(cls):
            agent = cls(
                agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
                api_key="EMPTY", language_realizer=False, semantic_belief=False,
                protocol_mode="normalize",
            )
            agent.prepare_episode(
                episode_index=1, total_episodes=10, game_kind="buyer_seller",
                private_objective="buyer reward is private WTP 63 minus deal price",
                starts_episode=False,
            )
            agent.init_agent(
                "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
                "You are Player BLUE.",
            )
            return agent

        baseline = CrossEpisodeInformationPlannerAgent
        raw_agent = make(baseline)
        calibrated = make(EndogeneityCalibratedInformationPlannerAgent)
        event = {
            "answer": "PROPOSAL",
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 55}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 55",
            "message": "Opening offer.",
        }
        raw_agent._update_from_event(event)
        calibrated._update_from_event(event)
        self.assertGreater(
            calibrated._posterior_json()["normalized_entropy"],
            raw_agent._posterior_json()["normalized_entropy"],
        )
        tempering = calibrated.evidence[-1]
        self.assertEqual(tempering["source"], "opponent_initiated_offer")
        self.assertEqual(tempering["effective_strength"], 0.25)

    def test_framework_v5_repeated_evidence_has_diminishing_strength(self):
        agent = EndogeneityCalibratedInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        event = {
            "answer": "PROPOSAL",
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 55}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 55",
            "message": "Repeated offer.",
        }
        agent._update_from_event(event)
        first = agent.evidence[-1]["effective_strength"]
        agent._update_from_event(event)
        second = agent.evidence[-1]["effective_strength"]
        self.assertLess(second, first)

    def test_framework_v6_shadow_response_updates_are_ordered(self):
        agent = DecisionRelevantInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 50}},
        }
        accepted = agent._shadow_outcome_posterior(candidate, "ACCEPT")
        rejected = agent._shadow_outcome_posterior(candidate, "REJECT")
        accept_mean = sum(v * p for v, p in zip(agent.buy_values, accepted))
        reject_mean = sum(v * p for v, p in zip(agent.buy_values, rejected))
        self.assertLess(accept_mean, reject_mean)
        self.assertAlmostEqual(sum(accepted), 1.0)
        self.assertAlmostEqual(sum(rejected), 1.0)

    def test_framework_v6_voi_records_all_outcome_branches(self):
        agent = DecisionRelevantInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        candidate = next(row for row in agent._build_candidates() if row["kind"] == "probe")
        diagnostic = agent._decision_relevant_voi(candidate)
        self.assertEqual(set(diagnostic["outcomes"]), {"ACCEPT", "COUNTER", "REJECT"})
        self.assertAlmostEqual(
            sum(row["probability"] for row in diagnostic["outcomes"].values()),
            1.0,
            places=5,
        )
        self.assertGreaterEqual(diagnostic["identifiable_mass"], 0.0)
        self.assertLessEqual(diagnostic["identifiable_mass"], 1.0)
        self.assertGreaterEqual(diagnostic["action_flip_probability"], 0.0)
        self.assertLessEqual(diagnostic["action_flip_probability"], 1.0)

    def test_framework_v6_blocks_probe_when_bounded_value_cannot_cover_deal(self):
        agent = DecisionRelevantInformationPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=10, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        agent.decision_index = 1
        observed = {"RED": {"X": 1}, "BLUE": {"ZUP": 55}}
        candidate = {
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 53}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 53",
            "self_utility": 10.0,
            "response": {"accept": 0.3, "accept_mean": 0.75},
            "decision_value_of_information": 1.7,
            "safeguarded_contingent_value": 3.3,
            "safeguard_eligible": True,
            "kind": "probe",
        }
        agent._decision_relevant_voi = lambda row: {
            "uncapped_future_information_value": 100.0,
            "posterior_value_range_cap": 100.0,
        }
        chosen, planner = agent._choose([candidate], {"trade": observed})
        self.assertEqual(chosen["type"], "ACCEPT")
        self.assertEqual(planner["probe_information_value"], 4.0)
        self.assertTrue(planner["probe_blocked_by_bounded_opportunity_cost"])

    @staticmethod
    def _make_v7_buyer():
        agent = FactorizedPreferencePolicyBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        return agent

    def test_framework_v7_counter_updates_response_policy_belief(self):
        agent = self._make_v7_buyer()
        agent.decision_index = 1
        trade = {"RED": {"X": 1}, "BLUE": {"ZUP": 50}}
        agent.last_own_action = {"type": "PROPOSE", "trade": trade}
        before = agent._posterior_json()["response_policy"]["counter_propensity"]
        agent._update_from_event(
            {"answer": "PROPOSAL", "trade": trade, "trade_text": "", "message": ""}
        )
        after = agent._posterior_json()["response_policy"]["counter_propensity"]
        self.assertGreater(after, before)
        self.assertEqual(agent.evidence[-2]["source"], "joint_response_to_locked_offer")

    def test_framework_v7_accept_and_reject_move_preference_in_opposite_directions(self):
        def posterior_mean(outcome):
            agent = self._make_v7_buyer()
            agent.decision_index = 1
            trade = {"RED": {"X": 1}, "BLUE": {"ZUP": 50}}
            agent.last_own_action = {"type": "PROPOSE", "trade": trade}
            event = {
                "answer": outcome,
                "trade": None,
                "trade_text": None,
                "message": "",
            }
            agent._update_from_event(event)
            return agent._posterior_json()["mean"]

        # The focal agent is the buyer, so the opponent is a seller. Acceptance
        # of price 50 supports a lower seller reservation than rejection.
        self.assertLess(posterior_mean("ACCEPT"), posterior_mean("REJECT"))

    def test_framework_v7_surprise_reset_is_policy_scoped(self):
        agent = self._make_v7_buyer()
        agent.decision_index = 1
        trade = {"RED": {"X": 1}, "BLUE": {"ZUP": 1}}
        # Concentrate policy mass on a confident accept-biased responder while
        # keeping the preference marginal unchanged, then observe an unlikely accept.
        theta = list(agent.buy_probs)
        policy = [0.0] * len(agent.policy_particles)
        policy[-1] = 1.0
        agent._set_joint_probs(agent._factorized_joint(theta, policy))
        agent.last_own_action = {"type": "PROPOSE", "trade": trade}
        agent._update_from_event(
            {"answer": "ACCEPT", "trade": None, "trade_text": None, "message": ""}
        )
        resets = [row for row in agent.evidence if row.get("event") == "POLICY_SURPRISE_RESET"]
        self.assertTrue(resets)
        self.assertTrue(resets[-1]["preference_marginal_preserved_before_likelihood"])

    def test_framework_v7_point_mass_joint_selects_zero_regret_exploit(self):
        agent = self._make_v7_buyer()
        agent.decision_index = 1
        theta = agent._point_mass(agent.buy_values, 43.0)
        policy = [0.0] * len(agent.policy_particles)
        policy[len(policy) // 2] = 1.0
        agent._set_joint_probs(agent._factorized_joint(theta, policy))
        candidates = agent._build_candidates()
        exploit = next(row for row in candidates if row["kind"] == "exploit")
        self.assertAlmostEqual(
            exploit["belief_usable_planning"]["cvar90_particle_regret"], 0.0, places=5
        )
        self.assertGreater(exploit["safeguarded_contingent_value"], 0.0)

    def test_framework_v7_policy_interventions_preserve_preference_marginal(self):
        agent = self._make_v7_buyer()
        learned_theta = list(agent.buy_probs)
        learned_resource = list(agent.resource_probs)
        agent._set_decision_belief("policy_shuffled", learned_theta, learned_resource)
        self.assertTrue(
            all(abs(a - b) < 1e-12 for a, b in zip(agent.buy_probs, learned_theta))
        )
        self.assertAlmostEqual(sum(agent.policy_probs), 1.0)

    @staticmethod
    def _make_v8_buyer():
        agent = ReliabilityGatedFactorizedBeliefPlannerAgent(
            agent_name="Player BLUE", model="mock", base_url="http://127.0.0.1:1/v1",
            api_key="EMPTY", language_realizer=False, semantic_belief=False,
            protocol_mode="normalize",
        )
        agent.prepare_episode(
            episode_index=1, total_episodes=20, game_kind="buyer_seller",
            private_objective="buyer reward is private WTP 63 minus deal price",
            starts_episode=False,
        )
        agent.init_agent(
            "<my resources> ZUP: 1000 </my resources><my goals> buy X </my goals>",
            "You are Player BLUE.",
        )
        return agent

    def test_framework_v8_wrong_confidence_is_diluted_without_direct_evidence(self):
        agent = self._make_v8_buyer()
        learned_buy = list(agent.buy_probs)
        learned_resource = list(agent.resource_probs)
        agent._set_decision_belief("wrong_confident", learned_buy, learned_resource)
        gate = agent._last_reliability_gate
        self.assertEqual(gate["theta_trust"], 0.0)
        self.assertAlmostEqual(agent._posterior_json()["mean"], 50.0)
        self.assertNotEqual(agent._posterior_json()["mean"], 57.0)

    def test_framework_v8_offer_does_not_increase_policy_trust(self):
        agent = self._make_v8_buyer()
        agent.decision_index = 1
        event = {
            "answer": "PROPOSAL",
            "trade": {"RED": {"X": 1}, "BLUE": {"ZUP": 55}},
            "trade_text": "Player RED Gives X: 1 | Player BLUE Gives ZUP: 55",
            "message": "Opening offer.",
        }
        agent._update_from_event(event)
        agent._set_decision_belief(
            "continuous", list(agent.buy_probs), list(agent.resource_probs)
        )
        gate = agent._last_reliability_gate
        self.assertGreater(gate["theta_trust"], 0.0)
        self.assertEqual(gate["policy_trust"], 0.0)
        self.assertEqual(gate["evidence_channels"]["opponent_offer"], 1)

    def test_framework_v8_direct_response_monotonically_increases_both_trusts(self):
        agent = self._make_v8_buyer()
        agent.decision_index = 1
        learned_buy = list(agent.buy_probs)
        learned_resource = list(agent.resource_probs)
        agent._set_decision_belief("continuous", learned_buy, learned_resource)
        before = dict(agent._last_reliability_gate)

        trade = {"RED": {"X": 1}, "BLUE": {"ZUP": 50}}
        # Restore the learned joint before simulating the response, just as chat
        # does after its action shadows.
        agent._set_joint_probs(list(agent._learned_joint_before_decision))
        agent.last_own_action = {"type": "PROPOSE", "trade": trade}
        agent._update_from_event(
            {"answer": "ACCEPT", "trade": None, "trade_text": None, "message": ""}
        )
        agent._set_decision_belief(
            "continuous", list(agent.buy_probs), list(agent.resource_probs)
        )
        after = agent._last_reliability_gate
        self.assertGreater(after["theta_trust"], before["theta_trust"])
        self.assertGreater(after["policy_trust"], before["policy_trust"])

    def test_framework_v8_oracle_remains_diagnostic_upper_bound(self):
        agent = self._make_v8_buyer()
        agent._set_decision_belief(
            "oracle", list(agent.buy_probs), list(agent.resource_probs)
        )
        self.assertEqual(agent._last_reliability_gate["theta_trust"], 1.0)
        self.assertEqual(agent._posterior_json()["mean"], 43.0)

    def test_resource_exchange_game_has_scalar_symmetric_gain(self):
        offer = """<my name> Player RED </my name>
<my resources> X: 25, Y: 5 </my resources>
<my goals> maximize value </my goals>
<reason> complementary exchange </reason>
<player answer> NONE </player answer>
<message> Trade five units each. </message>
<newly proposed trade> Player RED Gives X: 5 | Player BLUE Gives Y: 5 </newly proposed trade>"""
        accept = """<my name> Player BLUE </my name>
<my resources> X: 5, Y: 25 </my resources>
<my goals> maximize value </my goals>
<reason> positive utility </reason>
<player answer> ACCEPT </player answer>
<message> Agreed. </message>
<newly proposed trade> NONE </newly proposed trade>"""
        initial = [Resources({"X": 25, "Y": 5}), Resources({"X": 5, "Y": 25})]
        values = [Valuation({"X": 0.5, "Y": 2.5}), Valuation({"X": 2.5, "Y": 0.5})]
        with tempfile.TemporaryDirectory() as tmp:
            game = PaperRepeatedTradingGame(
                players=[StaticAgent(AGENT_ONE, [offer]), StaticAgent(AGENT_TWO, [accept])],
                iterations=10,
                resources_support_set=Resources({"X": 0, "Y": 0}),
                player_goals=[ValueMaximisationGoal(initial[i], values[i]) for i in range(2)],
                player_initial_resources=initial,
                player_social_behaviour=["", ""],
                player_roles=["red", "blue"],
                log_dir=tmp,
                log_path=tmp + "/episode",
            )
            game.run()
        self.assertEqual(game.game_state[-1]["summary"]["player_outcome"], [10.0, 10.0])

    def test_resource_accept_cannot_reactivate_stale_own_counteroffer_after_wait(self):
        red_offer = VALID_RESOURCE_RESPONSE
        blue_counter = VALID_RESOURCE_RESPONSE.replace(
            "<my name> Player RED", "<my name> Player BLUE"
        ).replace(
            "<my resources> X: 25, Y: 5", "<my resources> X: 5, Y: 25"
        ).replace(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5",
            "Player RED Gives Y: 5 | Player BLUE Gives X: 1",
        )
        red_wait = VALID_RESOURCE_RESPONSE.replace(
            "Player RED Gives X: 5 | Player BLUE Gives Y: 5", "NONE"
        )
        blue_accept = """<my name> Player BLUE </my name>
<my resources> X: 5, Y: 25 </my resources>
<my goals> maximize value </my goals>
<reason> accept </reason>
<player answer> ACCEPT </player answer>
<message> Accept. </message>
<newly proposed trade> NONE </newly proposed trade>"""
        initial = [Resources({"X": 25, "Y": 5}), Resources({"X": 5, "Y": 25})]
        values = [Valuation({"X": 0.5, "Y": 2.5}), Valuation({"X": 2.5, "Y": 0.5})]
        with tempfile.TemporaryDirectory() as tmp:
            game = PaperRepeatedTradingGame(
                players=[
                    StaticAgent(AGENT_ONE, [red_offer, red_wait]),
                    StaticAgent(AGENT_TWO, [blue_counter, blue_accept]),
                ],
                iterations=4,
                resources_support_set=Resources({"X": 0, "Y": 0}),
                player_goals=[ValueMaximisationGoal(initial[i], values[i]) for i in range(2)],
                player_initial_resources=initial,
                player_social_behaviour=["", ""],
                player_roles=["red", "blue"],
                log_dir=tmp,
                log_path=tmp + "/episode",
            )
            game.run()
        summary = game.game_state[-1]["summary"]
        self.assertNotEqual(summary["final_response"], "ACCEPT")
        self.assertIsNone(summary["proposed_trade"])
        self.assertEqual(summary["player_outcome"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
