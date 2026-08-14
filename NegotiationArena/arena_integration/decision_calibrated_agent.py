"""Decision-calibrated, reciprocal planner for repeated NegotiationArena games.

This module is intentionally separate from ``DualTimescaleBeliefPlannerAgent``.
The old class is the prompt-only V1 variant and its historical runs remain
reproducible.  V2 makes the decision path executable and auditable:

* public offers/acceptances update a small Bayesian particle posterior;
* every structured candidate receives an action-conditional response forecast;
* exact private self utility is used instead of an LLM-authored payoff score;
* a reciprocity ledger penalizes uncompensated concessions;
* the language model can verbalize, but cannot change, the locked action.

The implementation uses the same core for Buyer--Seller and Resource Exchange.
Only the compact domain functions below differ.

中文阅读导引
------------
本文件通过继承链保存 framework 的方法演化，越靠后的 class 只覆盖相对上一版
发生变化的 belief/update/planner 部分：

``V2 structured posterior + action lock``
  -> ``V3 uncertainty safeguard``
  -> ``V3.1 repeated commitment``
  -> ``V3.2/V3.3 action-space frontier``
  -> ``V4 cross-episode information``
  -> ``V5 evidence-endogeneity calibration``
  -> ``V6 decision-relevant VOI``
  -> ``V7 factorized preference theta / response policy phi``
  -> ``V8 evidence-reliability gate``。

正常方法只读取公开对话和本方私有效用。``evaluation_opponent_truth`` 只允许
evaluator diagnostics 及 oracle/wrong intervention 使用，不能进入 normal prompt。
最终 action 由 structured planner 锁定，LLM 只负责把已选动作表述成协议文本。
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from arena_integration.repeated_agents import (
    RepeatedLanguageAgent,
    _json_from_text,
    _tag_content,
)


OTHER_ANSWER = "other player answer"
OTHER_TRADE = "other player proposed trade"
OTHER_MESSAGE = "other player message"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _normalize(weights: list[float]) -> list[float]:
    total = sum(max(0.0, item) for item in weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, item) / total for item in weights]


def _weighted_quantile(values: list[float], probs: list[float], q: float) -> float:
    threshold = _clip(q, 0.0, 1.0)
    cumulative = 0.0
    for value, probability in sorted(zip(values, probs), key=lambda row: row[0]):
        cumulative += probability
        if cumulative >= threshold:
            return value
    return max(values)


def _entropy(probs: Iterable[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)


def parse_trade_text(text: str | None) -> dict[str, dict[str, int]] | None:
    """Parse the public canonical trade without using ``eval`` or arena internals."""
    if not text or str(text).strip().upper() == "NONE":
        return None
    result: dict[str, dict[str, int]] = {}
    for player, resources in re.findall(
        r"Player\s+(RED|BLUE)\s+Gives\s+([^|]+)", str(text), re.I
    ):
        parsed = {
            name.upper(): int(amount)
            for name, amount in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+)", resources)
        }
        result[player.upper()] = parsed
    return result if {"RED", "BLUE"}.issubset(result) else None


def render_trade(trade: dict[str, dict[str, int]]) -> str:
    def resources(player: str) -> str:
        rows = trade.get(player, {})
        return ", ".join(f"{name}: {int(value)}" for name, value in sorted(rows.items()))

    return f"Player RED Gives {resources('RED')} | Player BLUE Gives {resources('BLUE')}"


# V2 是所有后续版本的共同基础：posterior、候选枚举、精确本方 utility、reciprocity
# ledger 与 action lock 都从这里开始。阅读新框架时应先理解本类，再看子类差分。
class DecisionCalibratedBeliefPlannerAgent(RepeatedLanguageAgent):
    """Framework V2: structured posterior, reciprocal planning and action lock."""

    method_name = "decision_calibrated_reciprocal_planner_v2"

    def __init__(
        self,
        *args,
        candidate_count: int = 5,
        belief_mode: str = "continuous",
        language_realizer: bool = True,
        semantic_belief: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.candidate_count = max(2, int(candidate_count))
        self.belief_mode = belief_mode
        self.language_realizer = bool(language_realizer)
        self.semantic_belief = bool(semantic_belief)
        self.meta_buy_probs = [1.0 / 101.0] * 101
        self.meta_resource_probs = [1.0 / 21.0] * 21
        self.buy_values = [float(value) for value in range(101)]
        self.resource_values = [value / 20.0 for value in range(21)]
        self.buy_probs = list(self.meta_buy_probs)
        self.resource_probs = list(self.meta_resource_probs)
        self.posterior_history: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.last_own_action: dict[str, Any] | None = None
        self.last_opponent_trade: dict[str, dict[str, int]] | None = None
        self.last_opponent_offer_utility: float | None = None
        self.processed_user_messages = 0
        self.cumulative_own_concession = 0.0
        self.cumulative_received_gain = 0.0
        self.private_resources = ""
        self.private_goal_text = ""
        self.decision_index = 0
        self.semantic_acceptance_shift = 0.0
        self.semantic_evidence: list[dict[str, Any]] = []
        self.episode_calibration: list[dict[str, Any]] = []
        self.episode_action_flips: list[dict[str, Any]] = []
        # Evaluator-only ground truth. It is never included in prompts or the
        # learned posterior; oracle/wrong interventions and diagnostics may use
        # it. The runner updates it when a controlled private-preference switch
        # changes the opponent's actual utility function.
        self.evaluation_opponent_truth: float | None = None

    def prepare_episode(self, **kwargs):
        super().prepare_episode(**kwargs)
        uniform_buy = 1.0 / len(self.meta_buy_probs)
        uniform_resource = 1.0 / len(self.meta_resource_probs)
        # Meta history is deliberately a weak prior because old outcomes are
        # confounded by the focal agent's earlier policy.
        self.buy_probs = _normalize([0.7 * p + 0.3 * uniform_buy for p in self.meta_buy_probs])
        self.resource_probs = _normalize(
            [0.7 * p + 0.3 * uniform_resource for p in self.meta_resource_probs]
        )
        self.posterior_history = []
        self.evidence = []
        self.last_own_action = None
        self.last_opponent_trade = None
        self.last_opponent_offer_utility = None
        self.processed_user_messages = 0
        self.cumulative_own_concession = 0.0
        self.cumulative_received_gain = 0.0
        self.private_resources = ""
        self.private_goal_text = ""
        self.decision_index = 0
        self.semantic_acceptance_shift = 0.0
        self.semantic_evidence = []
        self.episode_calibration = []
        self.episode_action_flips = []

    def init_agent(self, system_prompt, role):
        super().init_agent(system_prompt, role)
        content = self.conversation[0]["content"]
        self.private_resources = _tag_content(content, "my resources") or self._default_resources()
        self.private_goal_text = _tag_content(content, "my goals") or self.private_objective
        # The starter receives its own role as the initial user message to match
        # NegotiationArena's original ChatGPTAgent layout.  It is an instruction,
        # not opponent evidence.
        self.processed_user_messages = sum(
            row.get("role") == "user" for row in self.conversation
        )

    def get_state(self):
        return {
            **super().get_state(),
            "candidate_count": self.candidate_count,
            "belief_mode": self.belief_mode,
            "language_realizer": self.language_realizer,
            "semantic_belief": self.semantic_belief,
            "posterior": self._posterior_json(),
            "posterior_updates": len(self.posterior_history),
            "reciprocity_ledger": self._ledger_json(),
        }

    def record_episode(self, **kwargs):
        super().record_episode(**kwargs)
        if self.game_kind == "buyer_seller":
            self.meta_buy_probs = list(self.buy_probs)
        elif self.game_kind == "resource_exchange":
            self.meta_resource_probs = list(self.resource_probs)

    def chat(self):
        self.decision_index += 1
        event = self._latest_unprocessed_event()
        pre_update_buy = list(self.buy_probs)
        pre_update_resource = list(self.resource_probs)
        if event:
            self._record_response_calibration(event)
        if event and (self.belief_mode != "frozen" or not self.posterior_history):
            self._update_from_event(event)
            if self.semantic_belief and event.get("message", "").strip():
                self._update_from_semantic_evidence(event)
        learned_buy = list(self.buy_probs)
        learned_resource = list(self.resource_probs)

        actual_mode = self.belief_mode if self.belief_mode in {
            "wrong_confident", "shuffled", "oracle",
            "policy_uniform", "policy_shuffled",
        } else "continuous"
        self._set_decision_belief(actual_mode, learned_buy, learned_resource)
        posterior_before = self._posterior_json()
        candidates = self._build_candidates()
        chosen, planner = self._choose(candidates, event)
        shadow = self._shadow_actions(
            event=event,
            learned_buy=learned_buy,
            learned_resource=learned_resource,
            pre_update_buy=pre_update_buy,
            pre_update_resource=pre_update_resource,
        )
        actual_signature = self._action_signature(chosen)
        flip_record = {
            "episode": self.episode_index,
            "decision": self.decision_index,
            "actual_mode": actual_mode if self.belief_mode != "frozen" else "frozen",
            "actual": actual_signature,
            "update_opportunity": bool(event),
            "counterfactuals": shadow,
            "flips": {
                mode: item["signature"] != actual_signature for mode, item in shadow.items()
            },
        }
        self.episode_action_flips.append(flip_record)
        # Interventions change only the planner's decision belief. Preserve the
        # learned posterior for subsequent evidence updates and meta history.
        self.buy_probs = learned_buy
        self.resource_probs = learned_resource
        response = self._realize_locked_action(chosen, planner)
        self._commit_chosen_action(chosen)
        trace = {
            "episode": self.episode_index,
            "decision": self.decision_index,
            "event": event,
            "evidence": deepcopy(self.evidence[-8:]),
            "semantic_evidence": deepcopy(self.semantic_evidence[-4:]),
            "posterior": posterior_before,
            "learned_posterior": self._posterior_json(),
            "candidates": candidates,
            "reciprocity_ledger_before": self._ledger_json(),
            "planner": planner,
            "action_flip_diagnostics": flip_record,
            "chosen_action": chosen,
            "locked_response": response,
        }
        self.posterior_history.append(posterior_before)
        self._append_decision(trace)
        return response

    def _record_response_calibration(self, event: dict[str, Any]) -> None:
        previous = self.last_own_action
        if not previous or previous.get("type") != "PROPOSE":
            return
        response = previous.get("response") or {}
        prediction = response.get("accept")
        if prediction is None:
            return
        probability = _clip(float(prediction), 1e-6, 1.0 - 1e-6)
        label = 1.0 if str(event.get("answer", "NONE")).upper() == "ACCEPT" else 0.0
        self.episode_calibration.append(
            {
                "episode": self.episode_index,
                "decision": self.decision_index,
                "predicted_accept": probability,
                "accepted": label,
                "brier": (probability - label) ** 2,
                "nll": -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)),
                "offered_trade": previous.get("trade_text"),
                "observed_answer": event.get("answer"),
            }
        )

    def _true_opponent_parameter(self) -> float:
        if self.evaluation_opponent_truth is not None:
            return float(self.evaluation_opponent_truth)
        if self.game_kind == "buyer_seller":
            return 43.0 if self.agent_name.upper().endswith("BLUE") else 63.0
        return 1.0 / 6.0 if self.agent_name.upper().endswith("BLUE") else 5.0 / 6.0

    @staticmethod
    def _point_mass(values: list[float], target: float) -> list[float]:
        index = min(range(len(values)), key=lambda idx: abs(values[idx] - target))
        result = [0.0] * len(values)
        result[index] = 1.0
        return result

    def _set_decision_belief(
        self,
        mode: str,
        learned_buy: list[float],
        learned_resource: list[float],
    ) -> None:
        self.buy_probs = list(learned_buy)
        self.resource_probs = list(learned_resource)
        if mode == "oracle":
            truth = self._true_opponent_parameter()
            if self.game_kind == "buyer_seller":
                self.buy_probs = self._point_mass(self.buy_values, truth)
            else:
                self.resource_probs = self._point_mass(self.resource_values, truth)
        elif mode == "wrong_confident":
            truth = self._true_opponent_parameter()
            wrong = 100.0 - truth if self.game_kind == "buyer_seller" else 1.0 - truth
            if self.game_kind == "buyer_seller":
                self.buy_probs = self._point_mass(self.buy_values, wrong)
            else:
                self.resource_probs = self._point_mass(self.resource_values, wrong)
        elif mode == "shuffled":
            probs = learned_buy if self.game_kind == "buyer_seller" else learned_resource
            size = len(probs)
            multiplier = 37 if size == 101 else 16
            offset = (self.seed + 17 * self.episode_index) % size
            shuffled = [0.0] * size
            for index, probability in enumerate(probs):
                shuffled[(multiplier * index + offset) % size] = probability
            if self.game_kind == "buyer_seller":
                self.buy_probs = shuffled
            else:
                self.resource_probs = shuffled

    @staticmethod
    def _action_signature(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": action.get("type"),
            "trade_text": action.get("trade_text") or (
                render_trade(action["trade"]) if action.get("trade") else None
            ),
        }

    def _shadow_actions(
        self,
        *,
        event: dict[str, Any] | None,
        learned_buy: list[float],
        learned_resource: list[float],
        pre_update_buy: list[float],
        pre_update_resource: list[float],
    ) -> dict[str, dict[str, Any]]:
        shadow: dict[str, dict[str, Any]] = {}
        for mode in ("continuous", "frozen", "wrong_confident", "shuffled", "oracle"):
            source_buy = pre_update_buy if mode == "frozen" else learned_buy
            source_resource = pre_update_resource if mode == "frozen" else learned_resource
            applied = "continuous" if mode == "frozen" else mode
            self._set_decision_belief(applied, source_buy, source_resource)
            alternatives = self._build_candidates()
            alternative, _ = self._choose(alternatives, event)
            shadow[mode] = {
                "signature": self._action_signature(alternative),
                "posterior": self._posterior_json(),
            }
        return shadow

    def episode_diagnostics(self) -> dict[str, Any]:
        truth = self._true_opponent_parameter()
        posterior = self._posterior_json()
        if self.game_kind == "buyer_seller":
            estimate = float(posterior["mean"])
            covered = float(posterior["q10"]) <= truth <= float(posterior["q90"])
        else:
            estimate = float(posterior["x_weight_mean"])
            covered = (
                float(posterior["x_weight_q10"]) <= truth <= float(posterior["x_weight_q90"])
            )
        flip_modes = ("frozen", "wrong_confident", "shuffled", "oracle")
        opportunities = sum(
            bool(row.get("update_opportunity")) for row in self.episode_action_flips
        )
        return {
            "calibration": deepcopy(self.episode_calibration),
            "calibration_count": len(self.episode_calibration),
            "belief_truth": truth,
            "belief_estimate": estimate,
            "belief_abs_error": abs(estimate - truth),
            "belief_q10_q90_covered": covered,
            "action_decisions": len(self.episode_action_flips),
            "action_flip_opportunities": opportunities,
            "action_flip_counts": {
                mode: sum(
                    bool(row.get("update_opportunity")) and bool(row["flips"].get(mode))
                    for row in self.episode_action_flips
                )
                for mode in flip_modes
            },
        }

    # ------------------------------------------------------------------
    # Observation and posterior update
    # ------------------------------------------------------------------
    def _latest_unprocessed_event(self) -> dict[str, Any] | None:
        user_messages = [row["content"] for row in self.conversation if row.get("role") == "user"]
        if len(user_messages) <= self.processed_user_messages:
            return None
        raw = str(user_messages[-1])
        self.processed_user_messages = len(user_messages)
        answer_text = _tag_content(raw, OTHER_ANSWER)
        trade_text = _tag_content(raw, OTHER_TRADE)
        message = _tag_content(raw, OTHER_MESSAGE)
        # Compatibility with standard same-name public tags and with historical
        # traces created before BuySellAgentMessage was repaired.
        answer_text = answer_text or _tag_content(raw, "player answer")
        trade_text = trade_text or _tag_content(raw, "newly proposed trade")
        message = message or _tag_content(raw, "message")
        if answer_text is None:
            legacy = re.search(r"<(ACCEPT|REJECT|PROPOSAL)>\s*player answer\s*</\1>", raw, re.I)
            answer_text = legacy.group(1) if legacy else None
        if trade_text is None:
            legacy = re.search(
                r"<(Player\s+RED\s+Gives\s+.*?\|\s*Player\s+BLUE\s+Gives\s+.*?)>\s*"
                r"newly proposed trade\s*</.*?>",
                raw,
                re.I | re.S,
            )
            trade_text = legacy.group(1).strip() if legacy else None
        answer = (answer_text or "NONE").strip().upper()
        return {
            "answer": answer,
            "trade_text": trade_text,
            "trade": parse_trade_text(trade_text),
            "message": message or "",
        }

    def _update_from_event(self, event: dict[str, Any]) -> None:
        if self.game_kind == "buyer_seller":
            self._update_buy_sell(event)
        else:
            self._update_resource(event)

    def _update_from_semantic_evidence(self, event: dict[str, Any]) -> None:
        """Extract bounded weak evidence from language; code owns the posterior.

        The LLM may identify an explicit reservation claim or a qualitative
        concession signal.  It cannot author posterior probabilities or planner
        scores.  Reliability is capped because negotiation utterances are
        strategic and need not be truthful.
        """
        prompt = f"""
Read only the opponent's latest PUBLIC negotiation message and structured action. Extract
decision-relevant evidence without guessing private facts from its role name. A claimed cost,
budget, floor, ceiling, or value is a CLAIM rather than ground truth. If no explicit numeric
reservation claim appears, use null. Concession signal is -1 for firmer/less willing, 0 for
unknown, and +1 for more willing to agree. Return JSON only:
{{"explicit_reservation_claim":null,"concession_signal":0.0,"firmness":0.5,
"evidence_quote_or_paraphrase":"...","reason":"..."}}
Game family: {self.game_kind}
Opponent public answer: {event.get('answer')}
Opponent structured trade: {event.get('trade_text')}
Opponent public message: {event.get('message')}
"""
        raw = self._call(
            "framework_v2_semantic_belief_evidence",
            [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative evidence extractor. Output observations, "
                        "not a free-form opponent personality or a final action."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        try:
            data = _json_from_text(raw)
            claim_raw = data.get("explicit_reservation_claim")
            claim = float(claim_raw) if claim_raw is not None else None
            if claim is not None and not 0.0 <= claim <= 100.0:
                claim = None
            concession = _clip(float(data.get("concession_signal", 0.0)), -1.0, 1.0)
            firmness = _clip(float(data.get("firmness", 0.5)), 0.0, 1.0)
        except (ValueError, TypeError, AttributeError):
            self.semantic_evidence.append(
                {"parse_error": True, "raw_response": raw[:2000], "applied": False}
            )
            return

        # Language shifts the acceptance logit only mildly.  Numeric actions and
        # actual accept/counter responses remain the dominant evidence.
        reliability = 0.12 + 0.12 * firmness
        signed_signal = concession - (firmness - 0.5)
        self.semantic_acceptance_shift = _clip(
            0.65 * self.semantic_acceptance_shift + reliability * signed_signal,
            -0.35,
            0.35,
        )
        applied_claim = False
        if claim is not None and self.game_kind == "buyer_seller":
            width = 8.0
            likelihood = [
                (1.0 - reliability)
                + reliability * math.exp(-0.5 * ((reservation - claim) / width) ** 2)
                for reservation in self.buy_values
            ]
            self.buy_probs = _normalize(
                [prob * item for prob, item in zip(self.buy_probs, likelihood)]
            )
            applied_claim = True
        item = {
            "explicit_reservation_claim": claim,
            "concession_signal": concession,
            "firmness": firmness,
            "bounded_reliability": round(reliability, 4),
            "semantic_acceptance_shift": round(self.semantic_acceptance_shift, 4),
            "evidence_quote_or_paraphrase": str(
                data.get("evidence_quote_or_paraphrase", "")
            )[:500],
            "applied_claim": applied_claim,
            "parse_error": False,
        }
        self.semantic_evidence.append(item)
        self.evidence.append(
            {
                "event": "PUBLIC_LANGUAGE",
                "source": "bounded_llm_semantic_extractor",
                "reliability": round(reliability, 4),
                "claim": claim,
            }
        )

    def _update_buy_sell(self, event: dict[str, Any]) -> None:
        likelihood = [1.0] * len(self.buy_values)
        answer = event.get("answer", "NONE")
        previous = self.last_own_action
        if previous and previous.get("trade"):
            price = self._price(previous["trade"])
            if price is not None:
                for idx, reservation in enumerate(self.buy_values):
                    p_accept = self._buy_accept_probability(price, reservation)
                    if answer == "ACCEPT":
                        likelihood[idx] *= max(0.02, p_accept)
                    elif answer in {"PROPOSAL", "REJECT", "NONE"}:
                        # A counteroffer is evidence of non-acceptance, but weaker
                        # than a terminal rejection because strategic delay is legal.
                        strength = 0.82 if answer == "REJECT" else 0.68
                        likelihood[idx] *= max(0.05, (1.0 - p_accept) ** strength)
                self.evidence.append(
                    {
                        "event": answer,
                        "source": "response_to_locked_offer",
                        "price": price,
                        "reliability": 0.9 if answer in {"ACCEPT", "REJECT"} else 0.72,
                    }
                )
        offered_price = self._price(event.get("trade"))
        if offered_price is not None:
            for idx, reservation in enumerate(self.buy_values):
                rational = reservation <= offered_price if self._opponent_is_seller() else reservation >= offered_price
                likelihood[idx] *= 0.8 if rational else 0.2
            self.evidence.append(
                {
                    "event": "OPPONENT_OFFER",
                    "source": "structured_trade",
                    "price": offered_price,
                    "reliability": 0.65,
                }
            )
            self._observe_opponent_offer(event["trade"])
        self.buy_probs = _normalize([p * l for p, l in zip(self.buy_probs, likelihood)])

    def _update_resource(self, event: dict[str, Any]) -> None:
        likelihood = [1.0] * len(self.resource_values)
        answer = event.get("answer", "NONE")
        previous = self.last_own_action
        if previous and previous.get("trade"):
            for idx, weight_x in enumerate(self.resource_values):
                utility = self._opponent_resource_utility(previous["trade"], weight_x)
                p_accept = _sigmoid(utility / 1.5)
                if answer == "ACCEPT":
                    likelihood[idx] *= max(0.02, p_accept)
                else:
                    likelihood[idx] *= max(0.05, (1.0 - p_accept) ** 0.68)
            self.evidence.append(
                {
                    "event": answer,
                    "source": "response_to_locked_offer",
                    "reliability": 0.9 if answer == "ACCEPT" else 0.65,
                }
            )
        trade = event.get("trade")
        if trade:
            for idx, weight_x in enumerate(self.resource_values):
                utility = self._opponent_resource_utility(trade, weight_x)
                likelihood[idx] *= 0.15 + 0.85 * _sigmoid(utility / 2.0)
            self.evidence.append(
                {
                    "event": "OPPONENT_OFFER",
                    "source": "structured_trade",
                    "trade": trade,
                    "reliability": 0.7,
                }
            )
            self._observe_opponent_offer(trade)
        self.resource_probs = _normalize([p * l for p, l in zip(self.resource_probs, likelihood)])

    def _observe_opponent_offer(self, trade: dict[str, dict[str, int]]) -> None:
        utility = self._self_utility(trade)
        if self.last_opponent_offer_utility is not None:
            self.cumulative_received_gain += max(0.0, utility - self.last_opponent_offer_utility)
        self.last_opponent_offer_utility = utility
        self.last_opponent_trade = trade

    # ------------------------------------------------------------------
    # Candidate generation and action-conditional response prediction
    # ------------------------------------------------------------------
    def _build_candidates(self) -> list[dict[str, Any]]:
        if self.game_kind == "buyer_seller":
            return self._build_buy_candidates()
        return self._build_resource_candidates()

    def _build_buy_candidates(self) -> list[dict[str, Any]]:
        rows = []
        for price in range(0, 101):
            trade = {"RED": {"X": 1}, "BLUE": {"ZUP": price}}
            self_utility = self._self_utility(trade)
            if self_utility < 0:
                continue
            response = self._response_distribution(trade)
            rows.append(self._score_candidate(trade, self_utility, response))
        return self._select_candidate_families(rows)

    def _build_resource_candidates(self) -> list[dict[str, Any]]:
        rows = []
        # The paper setting has complementary endowments.  Search both exchange
        # directions so the core remains valid under role/order changes.
        for red_resource, blue_resource in (("X", "Y"), ("Y", "X")):
            for red_amount in range(1, 11):
                for blue_amount in range(1, 11):
                    trade = {
                        "RED": {red_resource: red_amount},
                        "BLUE": {blue_resource: blue_amount},
                    }
                    if not self._trade_legal_under_paper_endowments(trade):
                        continue
                    self_utility = self._self_utility(trade)
                    if self_utility <= 0:
                        continue
                    response = self._response_distribution(trade)
                    rows.append(self._score_candidate(trade, self_utility, response))
        return self._select_candidate_families(rows)

    def _score_candidate(
        self,
        trade: dict[str, dict[str, int]],
        self_utility: float,
        response: dict[str, float],
    ) -> dict[str, Any]:
        previous_utility = (
            self._self_utility(self.last_own_action["trade"])
            if self.last_own_action and self.last_own_action.get("trade")
            else self_utility
        )
        concession = max(0.0, previous_utility - self_utility)
        ledger_credit = max(0.0, self.cumulative_received_gain - 0.5 * self.cumulative_own_concession)
        uncompensated = max(0.0, concession - ledger_credit)
        accept = response["accept"]
        remaining = max(0.0, 1.0 - self.decision_index / 5.0)
        continuation = remaining * 0.22 * self_utility
        downside = (1.0 - accept) * 0.10 * self_utility
        # Bernoulli variance is only a proxy for one-step decision value.  It is
        # bounded and vanishes late, unlike the old unbounded entropy bonus.
        dvoi = accept * (1.0 - accept) * remaining * min(self_utility, 10.0)
        total = (
            accept * self_utility
            + response["counter"] * continuation
            - downside
            + 0.08 * dvoi
            - 0.65 * uncompensated
        )
        return {
            "trade": trade,
            "trade_text": render_trade(trade),
            "self_utility": round(self_utility, 4),
            "response": {key: round(value, 6) for key, value in response.items()},
            "decision_value_of_information": round(dvoi, 4),
            "downside_risk": round(downside, 4),
            "concession_cost": round(concession, 4),
            "uncompensated_concession": round(uncompensated, 4),
            "contingent_value": round(total, 4),
        }

    def _select_candidate_families(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        selectors = [
            ("exploit", lambda row: row["contingent_value"]),
            ("safe", lambda row: row["response"]["accept"] * 10.0 + row["self_utility"] * 0.01),
            (
                "probe",
                lambda row: row["decision_value_of_information"] - 0.25 * row["concession_cost"],
            ),
            (
                "reciprocal",
                lambda row: row["contingent_value"] - 1.5 * row["uncompensated_concession"],
            ),
            (
                "fallback",
                lambda row: 2.0 * row["response"]["accept"] + 0.2 * row["self_utility"],
            ),
        ]
        selected: list[dict[str, Any]] = []
        seen = set()
        for family, key in selectors[: self.candidate_count]:
            ranked = sorted(rows, key=lambda row: (key(row), row["self_utility"]), reverse=True)
            choice = next((row for row in ranked if row["trade_text"] not in seen), ranked[0])
            item = deepcopy(choice)
            item["id"] = len(selected) + 1
            item["kind"] = family
            selected.append(item)
            seen.add(item["trade_text"])
        return selected

    def _response_distribution(self, trade: dict[str, dict[str, int]]) -> dict[str, float]:
        if self.game_kind == "buyer_seller":
            price = self._price(trade)
            accept = sum(
                probability * self._buy_accept_probability(float(price), reservation)
                for reservation, probability in zip(self.buy_values, self.buy_probs)
            )
        else:
            accept = sum(
                probability
                * _sigmoid(
                    self._opponent_resource_utility(trade, weight_x) / 1.5
                    + self.semantic_acceptance_shift
                )
                for weight_x, probability in zip(self.resource_values, self.resource_probs)
            )
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        counter = (1.0 - accept) * (0.82 - 0.42 * progress)
        reject = max(0.0, 1.0 - accept - counter)
        return {"accept": accept, "counter": counter, "reject_or_exit": reject}

    # ------------------------------------------------------------------
    # Planner, action locking, and language realization
    # ------------------------------------------------------------------
    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        proposal_limit = max(0, self.game_turn_limit // 2 - 1)
        if self._proposal_count() >= proposal_limit:
            observed_trade = event.get("trade") if event else None
            accept_value = self._self_utility(observed_trade) if observed_trade else None
            if observed_trade and accept_value is not None and accept_value >= 0:
                chosen = {
                    "type": "ACCEPT",
                    "trade": observed_trade,
                    "trade_text": render_trade(observed_trade),
                    "self_utility": round(accept_value, 4),
                    "kind": "deadline_legal_accept",
                }
            else:
                chosen = {
                    "type": "REJECT" if self.game_kind == "buyer_seller" else "WAIT",
                    "trade": None,
                    "kind": "proposal_limit_no_safe_offer",
                }
            return chosen, {
                "reason": "proposal_limit_reached",
                "proposal_limit": proposal_limit,
                "accept_observed_value": accept_value,
                "chosen_type": chosen["type"],
                "chosen_kind": chosen["kind"],
                "risk_gate": self._risk_gate(),
            }
        if not candidates:
            chosen = {"type": "REJECT" if self.game_kind == "buyer_seller" else "WAIT", "trade": None}
            return chosen, {
                "reason": "no_positive_legal_candidate",
                "risk_gate": self._risk_gate(),
            }
        best = max(candidates, key=lambda row: row["contingent_value"])
        chosen = {"type": "PROPOSE", **best}
        accept_value = None
        if event and event.get("trade"):
            accept_value = self._self_utility(event["trade"])
            # Agreement is compared with the counterfactual contingent value,
            # not accepted merely because the opponent offer is non-negative.
            deadline_margin = 0.15 + 0.25 * _clip(self.decision_index / 5.0, 0.0, 1.0)
            if accept_value >= 0 and accept_value + deadline_margin >= best["contingent_value"]:
                chosen = {
                    "type": "ACCEPT",
                    "trade": event["trade"],
                    "trade_text": render_trade(event["trade"]),
                    "self_utility": round(accept_value, 4),
                    "kind": "accept_observed_offer",
                }
        planner = {
            "objective": "expected own utility under action-conditional response posterior",
            "accept_observed_value": accept_value,
            "best_counterfactual_value": best["contingent_value"],
            "chosen_type": chosen["type"],
            "chosen_kind": chosen.get("kind"),
            "risk_gate": self._risk_gate(),
        }
        return chosen, planner

    def _risk_gate(self) -> dict[str, Any]:
        probs = self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        maximum_entropy = math.log(len(probs))
        normalized_entropy = _entropy(probs) / maximum_entropy if maximum_entropy else 0.0
        return {
            "normalized_entropy": round(normalized_entropy, 6),
            "mode": "robust" if normalized_entropy > 0.82 else "posterior_expected_value",
        }

    def _realize_locked_action(self, chosen: dict[str, Any], planner: dict[str, Any]) -> str:
        action_type = chosen["type"]
        fallback_message = self._fallback_message(chosen)
        message = fallback_message
        if self.language_realizer:
            prompt = f"""
Write one concise public negotiation message for the locked structured action below.
Do not state or imply my private reservation, utility weights, posterior, or exact private goal.
Do not change any price, quantity, action type, or resource. Return JSON only:
{{"message":"at most 45 words"}}
Game: {self.game_kind}
Locked action: {json.dumps(chosen, ensure_ascii=False)}
Planner mode: {json.dumps(planner.get('risk_gate', {}), ensure_ascii=False)}
"""
            try:
                raw = self._call(
                    "framework_v2_language_realization",
                    [
                        {"role": "system", "content": "You verbalize an already locked negotiation action."},
                        {"role": "user", "content": prompt},
                    ],
                    # Honour a deterministic evaluation temperature while retaining
                    # the original capped stochastic realizer in paper-style runs.
                    temperature=min(self.temperature, 0.3),
                    max_tokens=180,
                )
                parsed = _json_from_text(raw)
                candidate = str(parsed.get("message", "")).strip()
                if candidate:
                    message = candidate[:500]
            except (ValueError, TypeError, AttributeError):
                message = fallback_message
        return self._render_protocol(action_type, chosen.get("trade"), message, planner)

    def _render_protocol(
        self,
        action_type: str,
        trade: dict[str, dict[str, int]] | None,
        message: str,
        planner: dict[str, Any],
    ) -> str:
        reason = (
            f"Locked {action_type}; optimize exact self utility with calibrated response risk; "
            f"ledger balance {self._ledger_balance():.2f}; mode {planner['risk_gate']['mode']}."
        )
        if self.game_kind == "buyer_seller":
            answer = "PROPOSAL" if action_type == "PROPOSE" else action_type
            trade_text = render_trade(trade) if action_type == "PROPOSE" and trade else "NONE"
            proposal_count = self._proposal_count() + (1 if action_type == "PROPOSE" else 0)
            return "\n".join(
                [
                    f"<proposal count> {proposal_count} </proposal count>",
                    f"<my resources> {self.private_resources} </my resources>",
                    f"<my goals> {self.private_goal_text} </my goals>",
                    f"<reason> {reason} </reason>",
                    f"<player answer> {answer} </player answer>",
                    f"<newly proposed trade> {trade_text} </newly proposed trade>",
                    f"<message> {message} </message>",
                ]
            )
        answer = "ACCEPT" if action_type == "ACCEPT" else "NONE"
        trade_text = render_trade(trade) if action_type == "PROPOSE" and trade else "NONE"
        return "\n".join(
            [
                f"<my name> {self.agent_name} </my name>",
                f"<my resources> {self.private_resources} </my resources>",
                f"<my goals> {self.private_goal_text} </my goals>",
                f"<reason> {reason} </reason>",
                f"<player answer> {answer} </player answer>",
                f"<message> {message} </message>",
                f"<newly proposed trade> {trade_text} </newly proposed trade>",
            ]
        )

    def _commit_chosen_action(self, chosen: dict[str, Any]) -> None:
        if chosen.get("type") == "PROPOSE" and chosen.get("trade"):
            new_utility = self._self_utility(chosen["trade"])
            if self.last_own_action and self.last_own_action.get("trade"):
                previous = self._self_utility(self.last_own_action["trade"])
                self.cumulative_own_concession += max(0.0, previous - new_utility)
            self.last_own_action = deepcopy(chosen)

    def _fallback_message(self, chosen: dict[str, Any]) -> str:
        if chosen["type"] == "ACCEPT":
            return "I accept this agreement."
        if chosen["type"] in {"REJECT", "WAIT"}:
            return "I cannot improve on the available terms, so I will not make a new offer."
        return f"I propose the following concrete exchange: {chosen.get('trade_text', '')}."

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = Path(self.trace_dir) / self.agent_name.replace(" ", "_") / "framework_v2_decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Domain math
    # ------------------------------------------------------------------
    def _opponent_is_seller(self) -> bool:
        return self.agent_name.upper().endswith("BLUE")

    def _price(self, trade: dict[str, dict[str, int]] | None) -> float | None:
        if not trade:
            return None
        value = trade.get("BLUE", {}).get("ZUP")
        return float(value) if value is not None else None

    def _buy_accept_probability(self, price: float, reservation: float) -> float:
        margin = price - reservation if self._opponent_is_seller() else reservation - price
        return _sigmoid(margin / 2.5 + self.semantic_acceptance_shift)

    def _self_utility(self, trade: dict[str, dict[str, int]]) -> float:
        if self.game_kind == "buyer_seller":
            price = self._price(trade)
            if price is None:
                return -1e9
            bound = self._own_buy_sell_bound()
            return bound - price if self.agent_name.upper().endswith("BLUE") else price - bound
        values = self._own_resource_values()
        red_net = {
            resource: trade.get("BLUE", {}).get(resource, 0) - trade.get("RED", {}).get(resource, 0)
            for resource in {"X", "Y"}
        }
        sign = 1.0 if self.agent_name.upper().endswith("RED") else -1.0
        return sign * sum(values[item] * red_net[item] for item in red_net)

    def _opponent_resource_utility(
        self,
        trade: dict[str, dict[str, int]],
        weight_x: float,
    ) -> float:
        # Scale normalized weights to the paper's total value mass 3.0.
        values = {"X": 3.0 * weight_x, "Y": 3.0 * (1.0 - weight_x)}
        red_net = {
            resource: trade.get("BLUE", {}).get(resource, 0) - trade.get("RED", {}).get(resource, 0)
            for resource in {"X", "Y"}
        }
        opponent_is_red = self.agent_name.upper().endswith("BLUE")
        sign = 1.0 if opponent_is_red else -1.0
        return sign * sum(values[item] * red_net[item] for item in red_net)

    def _own_buy_sell_bound(self) -> float:
        numbers = [float(item) for item in re.findall(r"\b\d+(?:\.\d+)?\b", self.private_objective)]
        if numbers:
            return numbers[-1]
        return 63.0 if self.agent_name.upper().endswith("BLUE") else 43.0

    def _own_resource_values(self) -> dict[str, float]:
        matches = dict(
            (name.upper(), float(value))
            for name, value in re.findall(r"\b([XY])\s*=\s*(\d+(?:\.\d+)?)", self.private_objective, re.I)
        )
        if {"X", "Y"}.issubset(matches):
            return {"X": matches["X"], "Y": matches["Y"]}
        return {"X": 0.5, "Y": 2.5} if self.agent_name.upper().endswith("RED") else {"X": 2.5, "Y": 0.5}

    @staticmethod
    def _trade_legal_under_paper_endowments(trade: dict[str, dict[str, int]]) -> bool:
        endowments = {"RED": {"X": 25, "Y": 5}, "BLUE": {"X": 5, "Y": 25}}
        return all(
            0 <= amount <= endowments[player].get(resource, 0)
            for player, resources in trade.items()
            for resource, amount in resources.items()
        )

    def _proposal_count(self) -> int:
        return sum(
            1
            for row in self.conversation
            if row.get("role") == "assistant"
            and (_tag_content(str(row.get("content", "")), "player answer") or "").upper() in {"PROPOSAL", "NONE"}
            and parse_trade_text(_tag_content(str(row.get("content", "")), "newly proposed trade"))
        )

    def _default_resources(self) -> str:
        if self.game_kind == "buyer_seller":
            return "ZUP: 1000" if self.agent_name.upper().endswith("BLUE") else "X: 1"
        return "X: 25, Y: 5" if self.agent_name.upper().endswith("RED") else "X: 5, Y: 25"

    def _ledger_balance(self) -> float:
        return self.cumulative_received_gain - 0.5 * self.cumulative_own_concession

    def _ledger_json(self) -> dict[str, float]:
        return {
            "cumulative_own_concession": round(self.cumulative_own_concession, 6),
            "cumulative_received_gain": round(self.cumulative_received_gain, 6),
            "balance": round(self._ledger_balance(), 6),
        }

    def _posterior_json(self) -> dict[str, Any]:
        if self.game_kind == "buyer_seller":
            mean_value = sum(value * prob for value, prob in zip(self.buy_values, self.buy_probs))
            return {
                "kind": "opponent_reservation",
                "q10": _weighted_quantile(self.buy_values, self.buy_probs, 0.1),
                "mean": round(mean_value, 6),
                "q90": _weighted_quantile(self.buy_values, self.buy_probs, 0.9),
                "normalized_entropy": round(_entropy(self.buy_probs) / math.log(len(self.buy_probs)), 6),
                "evidence_count": len(self.evidence),
            }
        mean_weight = sum(value * prob for value, prob in zip(self.resource_values, self.resource_probs))
        return {
            "kind": "opponent_relative_resource_value",
            "x_weight_q10": _weighted_quantile(self.resource_values, self.resource_probs, 0.1),
            "x_weight_mean": round(mean_weight, 6),
            "x_weight_q90": _weighted_quantile(self.resource_values, self.resource_probs, 0.9),
            "normalized_entropy": round(
                _entropy(self.resource_probs) / math.log(len(self.resource_probs)), 6
            ),
            "evidence_count": len(self.evidence),
        }


# V3 不改 belief updater；它改变 belief 的使用方式，用 posterior adverse tail 和
# agreement opportunity cost 防止高不确定性时的冒险动作。
class UncertaintySafeguardedBeliefPlannerAgent(DecisionCalibratedBeliefPlannerAgent):
    """Framework V3: make posterior uncertainty change the locked action.

    V2 records an entropy-based risk mode but still ranks candidates with the
    posterior-mean acceptance probability.  V3 leaves the updater, candidate
    space, reciprocity ledger and language realizer unchanged.  It changes only
    belief utilization: response probabilities are shrunk toward a posterior
    lower tail, counteroffers pay an explicit agreement-opportunity cost, and a
    proposal cannot be preferred to the zero outside option when its
    safeguarded value is non-positive.
    """

    method_name = "uncertainty_safeguarded_belief_planner_v3"

    def _acceptance_particles(
        self, trade: dict[str, dict[str, int]]
    ) -> tuple[list[float], list[float]]:
        if self.game_kind == "buyer_seller":
            price = self._price(trade)
            values = [
                self._buy_accept_probability(float(price), reservation)
                for reservation in self.buy_values
            ]
            return values, self.buy_probs
        values = [
            _sigmoid(
                self._opponent_resource_utility(trade, weight_x) / 1.5
                + self.semantic_acceptance_shift
            )
            for weight_x in self.resource_values
        ]
        return values, self.resource_probs

    def _response_distribution(self, trade: dict[str, dict[str, int]]) -> dict[str, float]:
        particle_accept, posterior = self._acceptance_particles(trade)
        mean_accept = sum(prob * value for prob, value in zip(posterior, particle_accept))
        tail_accept = _weighted_quantile(particle_accept, posterior, 0.10)
        risk = self._risk_gate()
        # At maximum entropy, 70% of the forecast comes from the adverse tail.
        # Below entropy .55 the response forecast is the posterior expectation.
        uncertainty_weight = 0.70 * _clip(
            (float(risk["normalized_entropy"]) - 0.55) / 0.45, 0.0, 1.0
        )
        safeguarded_accept = (
            (1.0 - uncertainty_weight) * mean_accept + uncertainty_weight * tail_accept
        )
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        counter = (1.0 - safeguarded_accept) * (0.82 - 0.42 * progress)
        reject = max(0.0, 1.0 - safeguarded_accept - counter)
        return {
            "accept": safeguarded_accept,
            "accept_mean": mean_accept,
            "accept_tail_q10": tail_accept,
            "uncertainty_weight": uncertainty_weight,
            "counter": counter,
            "reject_or_exit": reject,
        }

    def _score_candidate(
        self,
        trade: dict[str, dict[str, int]],
        self_utility: float,
        response: dict[str, float],
    ) -> dict[str, Any]:
        row = super()._score_candidate(trade, self_utility, response)
        row["outside_option"] = 0.0
        row["safeguarded_contingent_value"] = row["contingent_value"]
        row["belief_use"] = {
            "accept_mean": round(response["accept_mean"], 6),
            "accept_tail_q10": round(response["accept_tail_q10"], 6),
            "accept_safeguarded": round(response["accept"], 6),
            "uncertainty_weight": round(response["uncertainty_weight"], 6),
        }
        # A high-payoff proposal with negligible response support is not an
        # admissible exploit under uncertainty.  These thresholds operate on
        # action-conditional probabilities, so they do not encode a domain
        # price or resource heuristic.
        row["safeguard_eligible"] = bool(
            response["accept_mean"] >= 0.55 and response["accept"] >= 0.18
        )
        return row

    def _select_candidate_families(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [row for row in rows if row.get("safeguard_eligible")]
        # Retain a legal fallback when the posterior says that no proposal has
        # meaningful response support; outside-option dominance still prevents
        # a negative safeguarded value from being sent.
        return super()._select_candidate_families(eligible or rows)

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        proposal_limit = max(0, self.game_turn_limit // 2 - 1)
        observed_trade = event.get("trade") if event else None
        accept_value = self._self_utility(observed_trade) if observed_trade else None
        risk_gate = self._risk_gate()

        if self._proposal_count() >= proposal_limit:
            if observed_trade and accept_value is not None and accept_value >= 0:
                chosen = {
                    "type": "ACCEPT",
                    "trade": observed_trade,
                    "trade_text": render_trade(observed_trade),
                    "self_utility": round(accept_value, 4),
                    "kind": "deadline_legal_accept",
                }
            else:
                chosen = {
                    "type": "REJECT" if self.game_kind == "buyer_seller" else "WAIT",
                    "trade": None,
                    "kind": "proposal_limit_outside_option",
                }
            return chosen, {
                "objective": "uncertainty-safeguarded own utility",
                "reason": "proposal_limit_reached",
                "proposal_limit": proposal_limit,
                "accept_observed_value": accept_value,
                "outside_option": 0.0,
                "chosen_type": chosen["type"],
                "chosen_kind": chosen["kind"],
                "risk_gate": risk_gate,
            }

        if not candidates:
            chosen = {
                "type": "REJECT" if self.game_kind == "buyer_seller" else "WAIT",
                "trade": None,
                "kind": "no_positive_legal_candidate",
            }
            return chosen, {
                "objective": "uncertainty-safeguarded own utility",
                "reason": "no_positive_legal_candidate",
                "outside_option": 0.0,
                "risk_gate": risk_gate,
            }

        best = max(candidates, key=lambda row: row["safeguarded_contingent_value"])
        best_value = float(best["safeguarded_contingent_value"])
        best_accept = float(best["response"]["accept"])
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        opportunity_cost = (1.0 - best_accept) * max(0.0, accept_value or 0.0)
        regret_weight = 0.55 + 0.45 * progress
        counter_value_after_regret = best_value - regret_weight * opportunity_cost
        certainty_margin = 0.15 + 0.35 * progress

        if best_value <= 0.0:
            if observed_trade and accept_value is not None and accept_value >= 0:
                chosen = {
                    "type": "ACCEPT",
                    "trade": observed_trade,
                    "trade_text": render_trade(observed_trade),
                    "self_utility": round(accept_value, 4),
                    "kind": "outside_option_dominates_proposals_accept",
                }
            else:
                chosen = {
                    "type": "REJECT" if self.game_kind == "buyer_seller" else "WAIT",
                    "trade": None,
                    "kind": "outside_option_dominates_proposals",
                }
        elif (
            observed_trade
            and accept_value is not None
            and accept_value >= 0
            and accept_value + certainty_margin >= counter_value_after_regret
        ):
            chosen = {
                "type": "ACCEPT",
                "trade": observed_trade,
                "trade_text": render_trade(observed_trade),
                "self_utility": round(accept_value, 4),
                "kind": "safeguarded_accept_observed_offer",
            }
        else:
            chosen = {"type": "PROPOSE", **best}

        planner = {
            "objective": "uncertainty-safeguarded own utility",
            "accept_observed_value": accept_value,
            "best_counterfactual_value": best_value,
            "best_counterfactual_accept_mean": best["response"].get("accept_mean"),
            "best_counterfactual_accept_safeguarded": best_accept,
            "counter_failure_opportunity_cost": round(opportunity_cost, 6),
            "counter_value_after_regret": round(counter_value_after_regret, 6),
            "certainty_margin": round(certainty_margin, 6),
            "outside_option": 0.0,
            "chosen_type": chosen["type"],
            "chosen_kind": chosen.get("kind"),
            "risk_gate": risk_gate,
        }
        return chosen, planner

    def _risk_gate(self) -> dict[str, Any]:
        gate = super()._risk_gate()
        entropy = float(gate["normalized_entropy"])
        gate.update(
            {
                "mode": "posterior_tail_safeguard" if entropy > 0.55 else "posterior_expected_value",
                "tail_quantile": 0.10,
                "uncertainty_weight": round(
                    0.70 * _clip((entropy - 0.55) / 0.45, 0.0, 1.0), 6
                ),
            }
        )
        return gate

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v3_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V3.1 从最近已成交 reward 构造 soft commitment floor，避免 repeated opponent
# 通过逐局恶化报价“训练”focal 接受越来越差的 deal。
class ReciprocalCommitmentBeliefPlannerAgent(UncertaintySafeguardedBeliefPlannerAgent):
    """Framework V3.1: preserve earned terms across repeated negotiation.

    A myopic safe planner can teach a repeated opponent to worsen its opening
    offer because accepting a positive deal is locally rational.  V3.1 adds a
    soft, evidence-based commitment floor derived only from this agent's own
    realized rewards.  It also separates opening exploitation from the safe
    response-support frontier, avoiding V3's collapse to low-utility resource
    openings when no agreement opportunity is currently at risk.
    """

    method_name = "reciprocal_commitment_belief_planner_v3_1"

    def _select_candidate_families(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        eligible = [row for row in rows if row.get("safeguard_eligible")]
        protected = eligible or rows
        selectors = [
            ("exploit", rows, lambda row: row["safeguarded_contingent_value"]),
            ("safe", protected, lambda row: row["response"]["accept"] * 10.0 + row["self_utility"] * 0.01),
            (
                "probe", protected,
                lambda row: row["decision_value_of_information"] - 0.25 * row["concession_cost"],
            ),
            (
                "reciprocal", protected,
                lambda row: row["safeguarded_contingent_value"] - 1.5 * row["uncompensated_concession"],
            ),
            (
                "fallback", protected,
                lambda row: 2.0 * row["response"]["accept"] + 0.2 * row["self_utility"],
            ),
        ]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for family, pool, key in selectors[: self.candidate_count]:
            ranked = sorted(pool, key=lambda row: (key(row), row["self_utility"]), reverse=True)
            choice = next((row for row in ranked if row["trade_text"] not in seen), ranked[0])
            item = deepcopy(choice)
            item["id"] = len(selected) + 1
            item["kind"] = family
            selected.append(item)
            seen.add(item["trade_text"])
        return selected

    def _historical_commitment_floor(self) -> float:
        agreed = [
            float(row.get("own_reward", 0.0))
            for row in self.episode_records[-5:]
            if row.get("agreement") and float(row.get("own_reward", 0.0)) > 0.0
        ]
        if not agreed:
            return 0.0
        # Median is robust to one unusually favorable deal.  The one-unit slack
        # permits bounded reciprocity without rewarding systematic deterioration.
        agreed.sort()
        middle = len(agreed) // 2
        median = agreed[middle] if len(agreed) % 2 else 0.5 * (agreed[middle - 1] + agreed[middle])
        return max(0.0, median - 1.0)

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        chosen, planner = super()._choose(candidates, event)
        floor = self._historical_commitment_floor()
        observed_trade = event.get("trade") if event else None
        accept_value = self._self_utility(observed_trade) if observed_trade else None
        planner["historical_commitment_floor"] = round(floor, 6)
        planner["commitment_window"] = min(5, len(self.episode_records))

        if (
            chosen.get("type") == "ACCEPT"
            and observed_trade
            and accept_value is not None
            and accept_value + 1e-9 < floor
            and self._proposal_count() < max(0, self.game_turn_limit // 2 - 1)
        ):
            viable = [
                row
                for row in candidates
                if row["self_utility"] >= floor
                and row["safeguarded_contingent_value"] > 0.0
            ]
            if viable:
                best = max(viable, key=lambda row: row["safeguarded_contingent_value"])
                chosen = {"type": "PROPOSE", **best}
                planner.update(
                    {
                        "reason": "reciprocal_commitment_floor_blocks_deteriorating_offer",
                        "chosen_type": "PROPOSE",
                        "chosen_kind": best.get("kind"),
                        "best_counterfactual_value": best["safeguarded_contingent_value"],
                    }
                )
            else:
                planner["commitment_floor_unenforceable"] = True
        return chosen, planner

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v3_1_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V3.2 只允许 resource-first opening 使用完整 exploitation frontier；其他时刻仍受
# response-support safeguard 约束。
class ScopedCommitmentBeliefPlannerAgent(ReciprocalCommitmentBeliefPlannerAgent):
    """Framework V3.2: scope ungated exploitation to resource openings only."""

    method_name = "scoped_reciprocal_commitment_belief_planner_v3_2"

    def _allow_ungated_exploit(self) -> bool:
        return bool(
            self.game_kind == "resource_exchange"
            and self.starts_episode
            and self.decision_index == 1
        )

    def _full_frontier_label(self) -> str:
        return "resource_opening_full"

    def _select_candidate_families(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        eligible = [row for row in rows if row.get("safeguard_eligible")]
        protected = eligible or rows
        allow_full_frontier = self._allow_ungated_exploit()
        exploit_pool = rows if allow_full_frontier else protected
        selectors = [
            ("exploit", exploit_pool, lambda row: row["safeguarded_contingent_value"]),
            ("safe", protected, lambda row: row["response"]["accept"] * 10.0 + row["self_utility"] * 0.01),
            (
                "probe", protected,
                lambda row: row["decision_value_of_information"] - 0.25 * row["concession_cost"],
            ),
            (
                "reciprocal", protected,
                lambda row: row["safeguarded_contingent_value"] - 1.5 * row["uncompensated_concession"],
            ),
            (
                "fallback", protected,
                lambda row: 2.0 * row["response"]["accept"] + 0.2 * row["self_utility"],
            ),
        ]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for family, pool, key in selectors[: self.candidate_count]:
            ranked = sorted(pool, key=lambda row: (key(row), row["self_utility"]), reverse=True)
            choice = next((row for row in ranked if row["trade_text"] not in seen), ranked[0])
            item = deepcopy(choice)
            item["id"] = len(selected) + 1
            item["kind"] = family
            item["frontier_scope"] = (
                self._full_frontier_label()
                if family == "exploit" and allow_full_frontier
                else "response_supported"
            )
            selected.append(item)
            seen.add(item["trade_text"])
        return selected

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v3_2_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V3.3 按 action space 调整 frontier：连续价格空间较窄，组合资源空间保留更宽的
# exploitation candidates。这是旧正式矩阵中最稳定的版本之一。
class ActionSpaceAdaptiveCommitmentBeliefPlannerAgent(ScopedCommitmentBeliefPlannerAgent):
    """Framework V3.3: adapt proposal breadth to action-space structure.

    Scalar price bargaining keeps every candidate behind posterior response
    support.  Combinatorial resource exchange retains a broad exploitation
    candidate at every proposal decision, because a narrow response-supported
    frontier can prematurely remove complementary bundles.  Acceptance remains
    locked to the immediately observed valid offer by the game evaluator.
    """

    method_name = "action_space_adaptive_commitment_belief_planner_v3_3"

    def _allow_ungated_exploit(self) -> bool:
        return self.game_kind == "resource_exchange"

    def _full_frontier_label(self) -> str:
        return "combinatorial_action_space_full"

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v3_3_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V4 把当前 probe 的信息价值乘到剩余 episodes；它探索“能降 entropy 的动作”，
# 但未保证信息会改变未来最优动作，因此可能产生探索机会成本。
class CrossEpisodeInformationPlannerAgent(ActionSpaceAdaptiveCommitmentBeliefPlannerAgent):
    """Framework V4: price a probe by its value to later episodes.

    The previous planner treated each accept-versus-counter decision as a
    one-shot problem even though its posterior persists across episodes. V4 may
    replace an early acceptance with a response-supported probe when the probe's
    current contingent value plus bounded cross-episode information value beats
    the certain agreement. The bonus vanishes with the remaining horizon and
    posterior entropy, so it cannot justify late or already-calibrated probing.
    """

    method_name = "cross_episode_information_planner_v4"

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        chosen, planner = super()._choose(candidates, event)
        remaining_episodes = max(0, self.total_episodes - self.episode_index)
        entropy = float(self._risk_gate()["normalized_entropy"])
        planner.update(
            {
                "remaining_episodes": remaining_episodes,
                "cross_episode_information_value": 0.0,
            }
        )
        if chosen.get("type") != "ACCEPT" or remaining_episodes <= 0 or entropy <= 0.65:
            return chosen, planner

        accept_value = float(chosen.get("self_utility", 0.0))
        supported_probes = [
            row
            for row in candidates
            if row.get("kind") == "probe"
            and row.get("safeguard_eligible")
            and float(row.get("self_utility", -1e9)) > accept_value
            and float(row.get("safeguarded_contingent_value", 0.0)) > 0.0
        ]
        if not supported_probes:
            return chosen, planner

        probe = max(
            supported_probes,
            key=lambda row: (
                float(row.get("decision_value_of_information", 0.0)),
                float(row.get("safeguarded_contingent_value", 0.0)),
            ),
        )
        cross_episode_value = (
            float(probe.get("decision_value_of_information", 0.0))
            * math.sqrt(float(remaining_episodes))
            * entropy
        )
        probe_total = float(probe["safeguarded_contingent_value"]) + cross_episode_value
        acceptance_margin = 0.15 + 0.35 * _clip(self.decision_index / 5.0, 0.0, 1.0)
        planner.update(
            {
                "cross_episode_information_value": round(cross_episode_value, 6),
                "probe_total_with_future_information": round(probe_total, 6),
                "probe_trade": probe.get("trade_text"),
            }
        )
        if probe_total > accept_value + acceptance_margin:
            chosen = {"type": "PROPOSE", **probe}
            planner.update(
                {
                    "reason": "cross_episode_information_value_justifies_probe",
                    "chosen_type": "PROPOSE",
                    "chosen_kind": "probe",
                }
            )
        return chosen, planner

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v4_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V5 修复 evidence endogeneity：对我方 locked offer 的正式响应权重大，主动 proposal
# 和重复相似线索权重较小。目标是避免把自己诱导出的行为重复当作独立证据。
class EndogeneityCalibratedInformationPlannerAgent(CrossEpisodeInformationPlannerAgent):
    """Framework V5: temper endogenous and repeated negotiation evidence.

    Opponent proposals are strategic actions induced by the current dialogue, not
    independent samples of private utility. V5 preserves strong updates for an
    accept/reject response to our locked offer, tempers unsolicited/counter offers,
    and discounts exact repeated evidence within an episode.
    """

    method_name = "endogeneity_calibrated_information_planner_v5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observation_counts: dict[str, int] = {}

    def prepare_episode(self, **kwargs):
        super().prepare_episode(**kwargs)
        self._observation_counts = {}

    def _update_from_event(self, event: dict[str, Any]) -> None:
        prior = list(self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs)
        response_to_own_offer = bool(self.last_own_action and self.last_own_action.get("trade"))
        answer = str(event.get("answer", "NONE")).upper()
        fingerprint = json.dumps(
            {
                "answer": answer,
                "trade": event.get("trade"),
                "responding_to": (
                    self.last_own_action.get("trade") if response_to_own_offer else None
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        count = self._observation_counts.get(fingerprint, 0) + 1
        self._observation_counts[fingerprint] = count

        super()._update_from_event(event)
        raw_posterior = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        if response_to_own_offer and answer in {"ACCEPT", "REJECT"}:
            source = "response_to_locked_offer_terminal"
            base_strength = 0.90
        elif response_to_own_offer:
            source = "endogenous_counter_to_locked_offer"
            base_strength = 0.55
        else:
            source = "opponent_initiated_offer"
            base_strength = 0.25
        effective_strength = base_strength / math.sqrt(float(count))
        tempered = _normalize(
            [
                math.exp(
                    (1.0 - effective_strength) * math.log(max(old, 1e-12))
                    + effective_strength * math.log(max(new, 1e-12))
                )
                for old, new in zip(prior, raw_posterior)
            ]
        )
        if self.game_kind == "buyer_seller":
            self.buy_probs = tempered
        else:
            self.resource_probs = tempered
        self.evidence.append(
            {
                "event": "ENDOGENEITY_TEMPERING",
                "source": source,
                "base_strength": base_strength,
                "repeat_count": count,
                "effective_strength": round(effective_strength, 6),
            }
        )

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v5_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V6 只有在可能的 response 会改变未来 structured action，且信息价值覆盖当前
# deal 机会成本时才 probe；因此它比 V4 更保守。
class DecisionRelevantInformationPlannerAgent(EndogeneityCalibratedInformationPlannerAgent):
    """Framework V6: buy information only when it can change a later decision.

    V4 rewards generic response uncertainty and can therefore reject a useful
    observed deal merely because many repeated episodes remain.  V6 retains V5's
    endogeneity-tempered posterior, but replaces that heuristic with a bounded,
    auditable value-of-information calculation.  A probe receives credit only
    when its possible structured responses produce different future proposal
    choices and improve their posterior-contingent value.

    The shadow calculation is deterministic and makes no additional model call.
    It deliberately models only the identifiable response to the locked offer;
    free-form language and the content of an unknown counteroffer receive no
    speculative information credit.
    """

    method_name = "decision_relevant_information_planner_v6"
    future_discount = 0.85

    def _outcome_probabilities(self, candidate: dict[str, Any]) -> dict[str, float]:
        response = candidate.get("response") or {}
        accept = _clip(
            float(response.get("accept_mean", response.get("accept", 0.0))),
            0.0,
            1.0,
        )
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        counter = (1.0 - accept) * (0.82 - 0.42 * progress)
        reject = max(0.0, 1.0 - accept - counter)
        return {"ACCEPT": accept, "COUNTER": counter, "REJECT": reject}

    def _shadow_outcome_posterior(
        self,
        candidate: dict[str, Any],
        outcome: str,
    ) -> list[float]:
        """Apply the V5 structured-response update without mutating live state."""
        trade = candidate.get("trade")
        if self.game_kind == "buyer_seller":
            prior = list(self.buy_probs)
            price = self._price(trade)
            if price is None:
                return prior
            likelihood = []
            for reservation in self.buy_values:
                p_accept = self._buy_accept_probability(price, reservation)
                if outcome == "ACCEPT":
                    likelihood.append(max(0.02, p_accept))
                else:
                    strength = 0.82 if outcome == "REJECT" else 0.68
                    likelihood.append(max(0.05, (1.0 - p_accept) ** strength))
        else:
            prior = list(self.resource_probs)
            if not trade:
                return prior
            likelihood = []
            for weight_x in self.resource_values:
                p_accept = _sigmoid(self._opponent_resource_utility(trade, weight_x) / 1.5)
                if outcome == "ACCEPT":
                    likelihood.append(max(0.02, p_accept))
                else:
                    likelihood.append(max(0.05, (1.0 - p_accept) ** 0.68))

        raw = _normalize([prob * item for prob, item in zip(prior, likelihood)])
        # Match V5: a direct terminal response is strong evidence, whereas a
        # counter is endogenous to the offer and therefore receives less weight.
        strength = 0.90 if outcome in {"ACCEPT", "REJECT"} else 0.55
        return _normalize(
            [
                math.exp(
                    (1.0 - strength) * math.log(max(old, 1e-12))
                    + strength * math.log(max(new, 1e-12))
                )
                for old, new in zip(prior, raw)
            ]
        )

    def _future_policy_snapshot(self, posterior: list[float]) -> dict[str, Any]:
        """Evaluate a next-episode opening proposal under one shadow posterior.

        The two compared worlds share the same action space, private objective,
        role and historical commitment ledger.  Episode-local dialogue and
        concessions are reset, matching ``prepare_episode``.  The 70/30 mixture
        mirrors the weak meta-prior used by the live agent.
        """
        names = (
            "buy_probs",
            "resource_probs",
            "decision_index",
            "last_own_action",
            "last_opponent_trade",
            "last_opponent_offer_utility",
            "cumulative_own_concession",
            "cumulative_received_gain",
            "semantic_acceptance_shift",
            "conversation",
        )
        saved = {name: deepcopy(getattr(self, name)) for name in names}
        try:
            if self.game_kind == "buyer_seller":
                uniform = 1.0 / len(posterior)
                self.buy_probs = _normalize([0.7 * p + 0.3 * uniform for p in posterior])
            else:
                uniform = 1.0 / len(posterior)
                self.resource_probs = _normalize([0.7 * p + 0.3 * uniform for p in posterior])
            self.decision_index = 1
            self.last_own_action = None
            self.last_opponent_trade = None
            self.last_opponent_offer_utility = None
            self.cumulative_own_concession = 0.0
            self.cumulative_received_gain = 0.0
            self.semantic_acceptance_shift = 0.0
            self.conversation = [
                row for row in self.conversation if row.get("role") == "system"
            ]
            candidates = self._build_candidates()
            if not candidates:
                return {"value": 0.0, "signature": {"type": "WAIT", "trade_text": None}}
            best = max(
                candidates,
                key=lambda row: float(row.get("safeguarded_contingent_value", 0.0)),
            )
            value = max(0.0, float(best.get("safeguarded_contingent_value", 0.0)))
            signature = self._action_signature(
                {"type": "PROPOSE", **best} if value > 0.0 else {"type": "WAIT"}
            )
            return {
                "value": value,
                "signature": signature,
                "accept_mean": float((best.get("response") or {}).get("accept_mean", 0.0)),
            }
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _identifiable_mass(self, candidate: dict[str, Any]) -> float:
        particle_accept, posterior = self._acceptance_particles(candidate["trade"])
        mean_accept = sum(p * value for p, value in zip(posterior, particle_accept))
        variance = sum(
            p * (value - mean_accept) ** 2
            for p, value in zip(posterior, particle_accept)
        )
        # A Bernoulli probability has maximum variance .25.  This normalization
        # is domain-independent and makes the term interpretable in [0, 1].
        return _clip(variance / 0.25, 0.0, 1.0)

    def _decision_relevant_voi(self, candidate: dict[str, Any]) -> dict[str, Any]:
        current_posterior = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        current = self._future_policy_snapshot(current_posterior)
        probabilities = self._outcome_probabilities(candidate)
        credibility_by_outcome = {"ACCEPT": 1.0, "COUNTER": 0.55, "REJECT": 0.90}
        outcomes: dict[str, dict[str, Any]] = {}
        expected_value = 0.0
        action_flip_probability = 0.0
        credibility = 0.0
        for outcome, probability in probabilities.items():
            updated = self._shadow_outcome_posterior(candidate, outcome)
            future = self._future_policy_snapshot(updated)
            flipped = future["signature"] != current["signature"]
            expected_value += probability * float(future["value"])
            action_flip_probability += probability * float(flipped)
            credibility += probability * credibility_by_outcome[outcome]
            outcomes[outcome] = {
                "probability": round(probability, 6),
                "credibility": credibility_by_outcome[outcome],
                "future_value": round(float(future["value"]), 6),
                "future_signature": future["signature"],
                "action_flip": flipped,
            }
        evi = expected_value - float(current["value"])
        identifiable_mass = self._identifiable_mass(candidate)
        decision_relevant_voi = (
            credibility
            * identifiable_mass
            * action_flip_probability
            * max(0.0, evi)
        )
        remaining = max(0, self.total_episodes - self.episode_index)
        effective_horizon = sum(self.future_discount ** index for index in range(remaining))
        outcome_values = [float(item["future_value"]) for item in outcomes.values()]
        posterior_value_range_cap = (
            max(outcome_values) - min(outcome_values) if outcome_values else 0.0
        )
        return {
            "current_future_value": round(float(current["value"]), 6),
            "current_future_signature": current["signature"],
            "outcomes": outcomes,
            "expected_future_value": round(expected_value, 6),
            "evi": round(evi, 6),
            "credibility": round(credibility, 6),
            "identifiable_mass": round(identifiable_mass, 6),
            "action_flip_probability": round(action_flip_probability, 6),
            "decision_relevant_voi": round(decision_relevant_voi, 6),
            "remaining_episodes": remaining,
            "effective_horizon": round(effective_horizon, 6),
            "uncapped_future_information_value": round(
                effective_horizon * decision_relevant_voi, 6
            ),
            "posterior_value_range_cap": round(posterior_value_range_cap, 6),
        }

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Bypass V4's entropy bonus while retaining the V3.3 action-space and
        # historical-commitment decision rule.
        chosen, planner = ActionSpaceAdaptiveCommitmentBeliefPlannerAgent._choose(
            self, candidates, event
        )
        planner.update(
            {
                "information_policy": "decision_relevant_bounded_voi",
                "decision_relevant_voi_evaluated": False,
                "probe_information_value": 0.0,
            }
        )
        remaining = max(0, self.total_episodes - self.episode_index)
        if chosen.get("type") != "ACCEPT" or remaining <= 0:
            return chosen, planner

        accept_value = max(0.0, float(chosen.get("self_utility", 0.0)))
        supported_probes = [
            row
            for row in candidates
            if row.get("kind") == "probe"
            and row.get("safeguard_eligible")
            and float(row.get("self_utility", -1e9)) > accept_value
            and float(row.get("safeguarded_contingent_value", 0.0)) > 0.0
        ]
        if not supported_probes:
            return chosen, planner

        diagnostics = [
            (row, self._decision_relevant_voi(row)) for row in supported_probes
        ]
        probe, voi = max(
            diagnostics,
            key=lambda item: (
                float(item[1]["uncapped_future_information_value"]),
                float(item[0].get("safeguarded_contingent_value", 0.0)),
            ),
        )
        # The agent will spend at most half the certain observed surplus (with a
        # small unit-scale floor) on information.  The second cap prevents a
        # probe from claiming more value than its outcomes can distinguish.
        opportunity_cost_cap = max(0.5, 0.5 * accept_value)
        probe_bonus = min(
            float(voi["uncapped_future_information_value"]),
            opportunity_cost_cap,
            float(voi["posterior_value_range_cap"]),
        )
        probe_total = float(probe["safeguarded_contingent_value"]) + probe_bonus
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        acceptance_margin = 0.15 + 0.35 * progress
        planner.update(
            {
                "decision_relevant_voi_evaluated": True,
                "probe_trade": probe.get("trade_text"),
                "probe_current_contingent_value": probe.get("safeguarded_contingent_value"),
                "probe_information_value": round(probe_bonus, 6),
                "probe_total_with_future_information": round(probe_total, 6),
                "opportunity_cost_cap": round(opportunity_cost_cap, 6),
                "decision_relevant_voi": voi,
            }
        )
        if probe_total > accept_value + acceptance_margin:
            chosen = {"type": "PROPOSE", **probe}
            planner.update(
                {
                    "reason": "decision_relevant_voi_justifies_probe",
                    "chosen_type": "PROPOSE",
                    "chosen_kind": "probe",
                }
            )
        else:
            planner["probe_blocked_by_bounded_opportunity_cost"] = True
        return chosen, planner

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v6_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V7 将“对手想要什么”(theta) 与“对手如何回应”(phi) 分开建模，并在 joint
# particles 上计算 expected value 与 tail regret，便于 preference/policy 单独干预。
class FactorizedPreferencePolicyBeliefPlannerAgent(DecisionRelevantInformationPlannerAgent):
    """Framework V7: separate what the opponent wants from how it responds.

    A public counteroffer is generated jointly by private preference and a
    strategic response policy.  Treating it as preference-only evidence creates
    endogeneity bias.  V7 maintains an interpretable factorized approximation to
    ``b(preference, response_policy)`` and evaluates proposals with posterior
    action regret.  It adds no model calls and uses no evaluator truth online.
    """

    method_name = "factorized_preference_policy_regret_planner_v7"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_particles = [
            {
                "rationality_scale": scale,
                "acceptance_bias": bias,
                "counter_propensity": counter,
            }
            for scale in (0.60, 1.00, 1.80)
            for bias in (-0.75, 0.0, 0.75)
            for counter in (0.50, 0.72, 0.90)
        ]
        buy_size = len(self.buy_values) * len(self.policy_particles)
        resource_size = len(self.resource_values) * len(self.policy_particles)
        self.meta_buy_joint_probs = [1.0 / buy_size] * buy_size
        self.meta_resource_joint_probs = [1.0 / resource_size] * resource_size
        self.buy_joint_probs = list(self.meta_buy_joint_probs)
        self.resource_joint_probs = list(self.meta_resource_joint_probs)
        self.policy_probs = [1.0 / len(self.policy_particles)] * len(self.policy_particles)
        self._pre_update_joint_probs: list[float] | None = None
        self._pre_update_policy_probs: list[float] | None = None
        self._learned_joint_before_decision: list[float] | None = None
        self._learned_policy_before_decision: list[float] | None = None
        self._in_shadow_intervention = False
        self.policy_surprise_events: list[dict[str, Any]] = []
        self._particle_response_cache: dict[
            tuple[int, float, str], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._current_risk_gate_cache: dict[str, Any] | None = None

    def _theta_values(self) -> list[float]:
        return self.buy_values if self.game_kind == "buyer_seller" else self.resource_values

    def _joint_probs(self) -> list[float]:
        return self.buy_joint_probs if self.game_kind == "buyer_seller" else self.resource_joint_probs

    def _set_joint_probs(self, probs: list[float]) -> None:
        if self.game_kind == "buyer_seller":
            self.buy_joint_probs = _normalize(probs)
        else:
            self.resource_joint_probs = _normalize(probs)
        self._marginalize_joint()
        self._current_risk_gate_cache = None

    def _factorized_joint(
        self, theta_probs: list[float], policy_probs: list[float]
    ) -> list[float]:
        return _normalize(
            [
                theta_prob * policy_prob
                for theta_prob in theta_probs
                for policy_prob in policy_probs
            ]
        )

    def _joint_index(self, theta_index: int, policy_index: int) -> int:
        return theta_index * len(self.policy_particles) + policy_index

    def _marginalize_joint(self) -> None:
        joint = self._joint_probs()
        theta_size = len(self._theta_values())
        policy_size = len(self.policy_particles)
        theta_probs = [
            sum(joint[self._joint_index(theta, policy)] for policy in range(policy_size))
            for theta in range(theta_size)
        ]
        self.policy_probs = [
            sum(joint[self._joint_index(theta, policy)] for theta in range(theta_size))
            for policy in range(policy_size)
        ]
        if self.game_kind == "buyer_seller":
            self.buy_probs = _normalize(theta_probs)
        else:
            self.resource_probs = _normalize(theta_probs)
        self.policy_probs = _normalize(self.policy_probs)

    def prepare_episode(self, **kwargs):
        super().prepare_episode(**kwargs)
        meta = (
            self.meta_buy_joint_probs
            if self.game_kind == "buyer_seller"
            else self.meta_resource_joint_probs
        )
        if self.belief_mode == "no_cross_episode":
            # Clean causal control: retain within-episode inference and the
            # identical planner, but start every episode from the same uniform
            # prior rather than carrying evidence across episodes.
            meta = [1.0 / len(meta)] * len(meta)
        uniform = 1.0 / len(meta)
        self._set_joint_probs([0.7 * prob + 0.3 * uniform for prob in meta])
        self._pre_update_joint_probs = None
        self._pre_update_policy_probs = None
        self._learned_joint_before_decision = None
        self._learned_policy_before_decision = None
        self._in_shadow_intervention = False
        self.policy_surprise_events = []
        self._particle_response_cache = {}
        self._current_risk_gate_cache = None

    def record_episode(self, **kwargs):
        super().record_episode(**kwargs)
        if self.belief_mode == "no_cross_episode":
            return
        if self.game_kind == "buyer_seller":
            self.meta_buy_joint_probs = list(self.buy_joint_probs)
        else:
            self.meta_resource_joint_probs = list(self.resource_joint_probs)

    def _joint_response(
        self,
        trade: dict[str, dict[str, int]],
        theta: float,
        policy: dict[str, float],
    ) -> dict[str, float]:
        if self.game_kind == "buyer_seller":
            price = self._price(trade)
            if price is None:
                return {"accept": 0.0, "counter": 0.0, "reject_or_exit": 1.0}
            margin = price - theta if self._opponent_is_seller() else theta - price
            base_scale = 2.5
        else:
            margin = self._opponent_resource_utility(trade, theta)
            base_scale = 1.5
        accept = _sigmoid(
            margin / (base_scale * float(policy["rationality_scale"]))
            + float(policy["acceptance_bias"])
            + self.semantic_acceptance_shift
        )
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        counter_ratio = float(policy["counter_propensity"]) * (1.0 - 0.45 * progress)
        counter = (1.0 - accept) * _clip(counter_ratio, 0.0, 1.0)
        return {
            "accept": accept,
            "counter": counter,
            "reject_or_exit": max(0.0, 1.0 - accept - counter),
        }

    def _particle_response_vectors(
        self, trade: dict[str, dict[str, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (
            self.decision_index,
            round(self.semantic_acceptance_shift, 9),
            render_trade(trade),
        )
        cached = self._particle_response_cache.get(key)
        if cached is not None:
            return cached
        policy_size = len(self.policy_particles)
        theta_values = np.asarray(self._theta_values(), dtype=np.float64)
        if self.game_kind == "buyer_seller":
            price = self._price(trade)
            if price is None:
                size = len(theta_values) * policy_size
                result = (np.zeros(size), np.zeros(size))
                self._particle_response_cache[key] = result
                return result
            margin_by_theta = (
                float(price) - theta_values
                if self._opponent_is_seller()
                else theta_values - float(price)
            )
            base_scale = 2.5
        else:
            margin_by_theta = np.asarray(
                [self._opponent_resource_utility(trade, theta) for theta in theta_values],
                dtype=np.float64,
            )
            base_scale = 1.5
        margin = np.repeat(margin_by_theta, policy_size)
        scales = np.tile(
            np.asarray(
                [policy["rationality_scale"] for policy in self.policy_particles],
                dtype=np.float64,
            ),
            len(theta_values),
        )
        biases = np.tile(
            np.asarray(
                [policy["acceptance_bias"] for policy in self.policy_particles],
                dtype=np.float64,
            ),
            len(theta_values),
        )
        propensities = np.tile(
            np.asarray(
                [policy["counter_propensity"] for policy in self.policy_particles],
                dtype=np.float64,
            ),
            len(theta_values),
        )
        logits = np.clip(
            margin / (base_scale * scales) + biases + self.semantic_acceptance_shift,
            -60.0,
            60.0,
        )
        accept_values = 1.0 / (1.0 + np.exp(-logits))
        progress = _clip(self.decision_index / 5.0, 0.0, 1.0)
        counter_ratio = np.clip(propensities * (1.0 - 0.45 * progress), 0.0, 1.0)
        counter_values = (1.0 - accept_values) * counter_ratio
        result = (accept_values, counter_values)
        self._particle_response_cache[key] = result
        return result

    @staticmethod
    def _event_outcome(event: dict[str, Any]) -> str:
        answer = str(event.get("answer", "NONE")).upper()
        if answer == "ACCEPT":
            return "ACCEPT"
        if answer == "REJECT":
            return "REJECT"
        if event.get("trade"):
            return "COUNTER"
        return "REJECT"

    def _update_from_event(self, event: dict[str, Any]) -> None:
        joint = list(self._joint_probs())
        self._pre_update_joint_probs = list(joint)
        self._pre_update_policy_probs = list(self.policy_probs)
        theta_values = self._theta_values()
        policy_size = len(self.policy_particles)
        previous = self.last_own_action

        if previous and previous.get("trade"):
            outcome = self._event_outcome(event)
            accept_values, counter_values = self._particle_response_vectors(previous["trade"])
            if outcome == "ACCEPT":
                likelihood_array = accept_values
            elif outcome == "COUNTER":
                likelihood_array = counter_values
            else:
                likelihood_array = 1.0 - accept_values - counter_values
            likelihood = np.maximum(0.02, likelihood_array).tolist()
            predictive = sum(prob * value for prob, value in zip(joint, likelihood))
            reset_mass = 0.30 if predictive < 0.08 else 0.0
            if reset_mass:
                theta_marginal = list(
                    self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
                )
                reset = self._factorized_joint(
                    theta_marginal,
                    [1.0 / policy_size] * policy_size,
                )
                joint = [
                    (1.0 - reset_mass) * old + reset_mass * fresh
                    for old, fresh in zip(joint, reset)
                ]
                event_row = {
                    "event": "POLICY_SURPRISE_RESET",
                    "outcome": outcome,
                    "predictive_probability": round(predictive, 6),
                    "reset_mass": reset_mass,
                    "preference_marginal_preserved_before_likelihood": True,
                }
                self.policy_surprise_events.append(event_row)
                self.evidence.append(event_row)
            strength = 0.90 if outcome in {"ACCEPT", "REJECT"} else 0.55
            joint = _normalize(
                [old * (value ** strength) for old, value in zip(joint, likelihood)]
            )
            self.evidence.append(
                {
                    "event": outcome,
                    "source": "joint_response_to_locked_offer",
                    "predictive_probability": round(predictive, 6),
                    "effective_strength": strength,
                }
            )

        offered_trade = event.get("trade")
        if offered_trade:
            theta_likelihood = []
            if self.game_kind == "buyer_seller":
                offered_price = self._price(offered_trade)
                for theta in theta_values:
                    rational = (
                        theta <= float(offered_price)
                        if self._opponent_is_seller()
                        else theta >= float(offered_price)
                    )
                    theta_likelihood.append(0.8 if rational else 0.2)
            else:
                for theta in theta_values:
                    utility = self._opponent_resource_utility(offered_trade, theta)
                    theta_likelihood.append(0.15 + 0.85 * _sigmoid(utility / 2.0))
            joint = _normalize(
                [
                    probability * theta_likelihood[index // policy_size] ** 0.25
                    for index, probability in enumerate(joint)
                ]
            )
            self.evidence.append(
                {
                    "event": "OPPONENT_OFFER",
                    "source": "weak_preference_only_offer_evidence",
                    "effective_strength": 0.25,
                }
            )
            self._observe_opponent_offer(offered_trade)
        self._set_joint_probs(joint)

    def _update_from_semantic_evidence(self, event: dict[str, Any]) -> None:
        old_theta = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        DecisionCalibratedBeliefPlannerAgent._update_from_semantic_evidence(self, event)
        self._particle_response_cache = {}
        target_theta = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        joint = list(self._joint_probs())
        policy_size = len(self.policy_particles)
        projected = [
            probability
            * target_theta[index // policy_size]
            / max(old_theta[index // policy_size], 1e-12)
            for index, probability in enumerate(joint)
        ]
        self._set_joint_probs(projected)

    def _set_decision_belief(
        self,
        mode: str,
        learned_buy: list[float],
        learned_resource: list[float],
    ) -> None:
        if not self._in_shadow_intervention:
            self._learned_joint_before_decision = list(self._joint_probs())
            self._learned_policy_before_decision = list(self.policy_probs)
        preference_mode = mode if mode in {"wrong_confident", "shuffled", "oracle"} else "continuous"
        DecisionCalibratedBeliefPlannerAgent._set_decision_belief(
            self, preference_mode, learned_buy, learned_resource
        )
        theta_probs = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        policy_probs = list(self.policy_probs)
        if mode == "policy_uniform":
            policy_probs = [1.0 / len(policy_probs)] * len(policy_probs)
        elif mode == "policy_shuffled":
            size = len(policy_probs)
            offset = (self.seed + 11 * self.episode_index) % size
            shuffled = [0.0] * size
            for index, probability in enumerate(policy_probs):
                shuffled[(10 * index + offset) % size] += probability
            policy_probs = _normalize(shuffled)
        self._set_joint_probs(self._factorized_joint(theta_probs, policy_probs))

    def _shadow_actions(
        self,
        *,
        event: dict[str, Any] | None,
        learned_buy: list[float],
        learned_resource: list[float],
        pre_update_buy: list[float],
        pre_update_resource: list[float],
    ) -> dict[str, dict[str, Any]]:
        saved_joint = list(self._joint_probs())
        saved_policy = list(self.policy_probs)
        learned_policy = list(self._learned_policy_before_decision or saved_policy)
        shadow: dict[str, dict[str, Any]] = {}
        try:
            self._in_shadow_intervention = True
            for mode in (
                "continuous", "frozen", "wrong_confident", "shuffled", "oracle",
                "policy_uniform", "policy_shuffled",
            ):
                source_buy = pre_update_buy if mode == "frozen" else learned_buy
                source_resource = pre_update_resource if mode == "frozen" else learned_resource
                structured_update_occurred = (
                    any(abs(a - b) > 1e-12 for a, b in zip(pre_update_buy, learned_buy))
                    or any(
                        abs(a - b) > 1e-12
                        for a, b in zip(pre_update_resource, learned_resource)
                    )
                )
                if (
                    mode == "frozen"
                    and structured_update_occurred
                    and self._pre_update_policy_probs is not None
                ):
                    self.policy_probs = list(self._pre_update_policy_probs)
                else:
                    self.policy_probs = list(learned_policy)
                applied = "continuous" if mode == "frozen" else mode
                self._set_decision_belief(applied, source_buy, source_resource)
                alternatives = self._build_candidates()
                alternative, _ = self._choose(alternatives, event)
                shadow[mode] = {
                    "signature": self._action_signature(alternative),
                    "posterior": self._posterior_json(),
                }
        finally:
            self._in_shadow_intervention = False
            restored_joint = self._learned_joint_before_decision or saved_joint
            if self.game_kind == "buyer_seller":
                self.buy_joint_probs = list(restored_joint)
            else:
                self.resource_joint_probs = list(restored_joint)
            self._marginalize_joint()
            self._current_risk_gate_cache = None
        return shadow

    def _acceptance_particles(
        self, trade: dict[str, dict[str, int]]
    ) -> tuple[list[float], list[float]]:
        values, _ = self._particle_response_vectors(trade)
        return values, self._joint_probs()

    def _response_distribution(self, trade: dict[str, dict[str, int]]) -> dict[str, float]:
        joint = self._joint_probs()
        joint_array = np.asarray(joint, dtype=np.float64)
        accept_values, counter_values = self._particle_response_vectors(trade)
        mean_accept = float(joint_array @ accept_values)
        mean_counter = float(joint_array @ counter_values)
        order = np.argsort(accept_values)
        ordered_accept = accept_values[order]
        cumulative = np.cumsum(joint_array[order])
        tail_index = min(
            int(np.searchsorted(cumulative, 0.10, side="left")),
            len(ordered_accept) - 1,
        )
        tail_accept = float(ordered_accept[tail_index])
        risk = self._risk_gate()
        uncertainty_weight = 0.70 * _clip(
            (float(risk["normalized_entropy"]) - 0.55) / 0.45, 0.0, 1.0
        )
        accept = (1.0 - uncertainty_weight) * mean_accept + uncertainty_weight * tail_accept
        counter_ratio = mean_counter / max(1.0 - mean_accept, 1e-9)
        counter = (1.0 - accept) * _clip(counter_ratio, 0.0, 1.0)
        return {
            "accept": accept,
            "accept_mean": mean_accept,
            "accept_tail_q10": tail_accept,
            "uncertainty_weight": uncertainty_weight,
            "counter": counter,
            "reject_or_exit": max(0.0, 1.0 - accept - counter),
        }

    @staticmethod
    def _weighted_upper_tail_mean(
        values: list[float], probs: list[float], tail_mass: float = 0.10
    ) -> float:
        remaining = _clip(tail_mass, 1e-9, 1.0)
        total = 0.0
        for value, probability in sorted(zip(values, probs), reverse=True):
            take = min(remaining, probability)
            total += take * value
            remaining -= take
            if remaining <= 1e-12:
                break
        return total / tail_mass

    def _select_candidate_families(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        joint = self._joint_probs()
        joint_array = np.asarray(joint, dtype=np.float64)
        particle_values: list[np.ndarray] = []
        remaining_turn = max(0.0, 1.0 - self.decision_index / 5.0)
        for row in rows:
            utility = float(row["self_utility"])
            continuation = remaining_turn * 0.22 * utility
            concession_penalty = 0.65 * float(row["uncompensated_concession"])
            accept_values, counter_values = self._particle_response_vectors(row["trade"])
            dvoi = (
                accept_values
                * (1.0 - accept_values)
                * remaining_turn
                * min(utility, 10.0)
            )
            values = (
                accept_values * utility
                + counter_values * continuation
                - (1.0 - accept_values) * 0.10 * utility
                + 0.08 * dvoi
                - concession_penalty
            )
            particle_values.append(values)
        value_matrix = np.stack(particle_values, axis=0)
        particle_best = np.max(value_matrix, axis=0)
        preference_entropy = _entropy(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        ) / math.log(len(self._theta_values()))
        regret_weight = 0.35 * preference_entropy
        regret_matrix = particle_best[None, :] - value_matrix
        mean_values = value_matrix @ joint_array
        for row_index, (row, values) in enumerate(zip(rows, particle_values)):
            regrets = regret_matrix[row_index]
            mean_value = float(mean_values[row_index])
            order = np.argsort(regrets)[::-1]
            ordered_regret = regrets[order]
            ordered_prob = joint_array[order]
            cumulative = np.cumsum(ordered_prob)
            boundary = int(np.searchsorted(cumulative, 0.10, side="left"))
            full_mass = float(cumulative[boundary - 1]) if boundary > 0 else 0.0
            cvar_total = float(
                ordered_regret[:boundary] @ ordered_prob[:boundary]
            )
            boundary_take = max(0.0, 0.10 - full_mass)
            cvar_total += boundary_take * float(ordered_regret[boundary])
            cvar_regret = cvar_total / 0.10
            # Regret and action value can have different natural scales across
            # price and resource games.  Penalize the *fraction* of unresolved
            # regret and cap the spend at a fraction of the action's own positive
            # posterior value.  This preserves the outside-option semantics and
            # cannot turn every legal opening negative at a diffuse prior.
            normalized_regret = cvar_regret / (
                cvar_regret + abs(mean_value) + 1e-9
            )
            regret_penalty = (
                regret_weight * normalized_regret * max(0.0, mean_value)
            )
            belief_usable_value = mean_value - regret_penalty
            row["belief_usable_planning"] = {
                "posterior_mean_action_value": round(mean_value, 6),
                "cvar90_particle_regret": round(cvar_regret, 6),
                "normalized_particle_regret": round(normalized_regret, 6),
                "regret_weight": round(regret_weight, 6),
                "regret_penalty": round(regret_penalty, 6),
                "belief_usable_value": round(belief_usable_value, 6),
            }
            row["safeguarded_contingent_value"] = round(belief_usable_value, 4)
        return super()._select_candidate_families(rows)

    def _risk_gate(self) -> dict[str, Any]:
        if self._current_risk_gate_cache is not None:
            return dict(self._current_risk_gate_cache)
        joint = self._joint_probs()
        joint_entropy = _entropy(joint) / math.log(len(joint))
        policy_entropy = _entropy(self.policy_probs) / math.log(len(self.policy_probs))
        gate = super()._risk_gate()
        gate.update(
            {
                "normalized_entropy": round(joint_entropy, 6),
                "joint_preference_policy_entropy": round(joint_entropy, 6),
                "policy_entropy": round(policy_entropy, 6),
                "belief_factorization": "preference_x_response_policy",
            }
        )
        self._current_risk_gate_cache = dict(gate)
        return gate

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # V7 studies belief-conditioned exploitation.  It deliberately bypasses
        # V4's generic entropy probe and V6's inactive accept/probe overlay while
        # retaining the V3.3 commitment, outside-option and opportunity-cost rule.
        chosen, planner = ActionSpaceAdaptiveCommitmentBeliefPlannerAgent._choose(
            self, candidates, event
        )
        planner.update(
            {
                "belief_model": "joint_preference_response_policy",
                "proposal_objective": "posterior_mean_value_minus_cvar_particle_regret",
                "generic_information_probe_enabled": False,
            }
        )
        return chosen, planner

    def _posterior_json(self) -> dict[str, Any]:
        base = DecisionCalibratedBeliefPlannerAgent._posterior_json(self)
        means = {
            name: sum(
                probability * float(policy[name])
                for probability, policy in zip(self.policy_probs, self.policy_particles)
            )
            for name in ("rationality_scale", "acceptance_bias", "counter_propensity")
        }
        base["response_policy"] = {
            **{name: round(value, 6) for name, value in means.items()},
            "normalized_entropy": round(
                _entropy(self.policy_probs) / math.log(len(self.policy_probs)), 6
            ),
            "surprise_resets": len(self.policy_surprise_events),
        }
        base["joint_normalized_entropy"] = round(
            _entropy(self._joint_probs()) / math.log(len(self._joint_probs())), 6
        )
        return base

    def episode_diagnostics(self) -> dict[str, Any]:
        result = super().episode_diagnostics()
        result["response_policy_belief"] = self._posterior_json()["response_policy"]
        result["policy_surprise_resets"] = len(self.policy_surprise_events)
        for mode in ("policy_uniform", "policy_shuffled"):
            result["action_flip_counts"][mode] = sum(
                bool(row.get("update_opportunity")) and bool(row["flips"].get(mode))
                for row in self.episode_action_flips
            )
        return result

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v7_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")


# V8 是当前最终主方法：在 V7 上按 evidence provenance 估计 trust，并把不可靠的
# learned posterior 拉回 episode anchor，从而限制 wrong-confident belief 的损害。
class ReliabilityGatedFactorizedBeliefPlannerAgent(
    FactorizedPreferencePolicyBeliefPlannerAgent
):
    """Framework V8: prevent weak or overconfident evidence controlling actions.

    V7 learns a joint preference/response-policy posterior.  V8 keeps that
    learner intact for future updates and meta history, but exposes a bounded
    mixture of the learned and episode-anchor posterior to the planner.  The
    mixing weights depend only on identifiable public evidence channels; they
    never use evaluator truth, role-specific reward thresholds, or test reward.
    """

    method_name = "reliability_gated_factorized_belief_planner_v8"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_anchor_joint_probs: list[float] = []
        self.reliability_gate_history: list[dict[str, Any]] = []
        self._last_reliability_gate: dict[str, Any] | None = None

    def prepare_episode(self, **kwargs):
        super().prepare_episode(**kwargs)
        self.episode_anchor_joint_probs = list(self._joint_probs())
        self.reliability_gate_history = []
        self._last_reliability_gate = None

    def _marginals_from_joint(
        self, joint: list[float]
    ) -> tuple[list[float], list[float]]:
        theta_size = len(self._theta_values())
        policy_size = len(self.policy_particles)
        theta = [
            sum(joint[self._joint_index(i, j)] for j in range(policy_size))
            for i in range(theta_size)
        ]
        policy = [
            sum(joint[self._joint_index(i, j)] for i in range(theta_size))
            for j in range(policy_size)
        ]
        return _normalize(theta), _normalize(policy)

    @staticmethod
    def _total_variation(left: list[float], right: list[float]) -> float:
        return 0.5 * sum(abs(a - b) for a, b in zip(left, right))

    def _evidence_channel_counts(self) -> dict[str, int]:
        direct = sum(
            row.get("source") == "joint_response_to_locked_offer"
            for row in self.evidence
        )
        offers = sum(
            row.get("source") == "weak_preference_only_offer_evidence"
            for row in self.evidence
        )
        semantic = sum(
            not row.get("parse_error", False)
            for row in self.semantic_evidence
        )
        return {
            "direct_response": int(direct),
            "opponent_offer": int(offers),
            "semantic": int(semantic),
        }

    def _reliability_weights(
        self,
        learned_theta: list[float],
        learned_policy: list[float],
        anchor_theta: list[float],
        anchor_policy: list[float],
    ) -> dict[str, Any]:
        counts = self._evidence_channel_counts()
        direct = counts["direct_response"]
        offers = counts["opponent_offer"]
        semantic = counts["semantic"]

        # A response to our locked action is substantially more identifiable
        # than an opponent-initiated offer or strategic language.  The same
        # coefficients are used in every game and role.
        theta_support = 1.0 - math.exp(
            -(0.85 * direct + 0.10 * offers + 0.05 * semantic)
        )
        policy_support = 1.0 - math.exp(-(0.70 * direct))
        theta_drift = self._total_variation(learned_theta, anchor_theta)
        policy_drift = self._total_variation(learned_policy, anchor_policy)

        # Large movement unsupported by identified evidence is precisely the
        # wrong-confidence failure observed in V7.  Discount it smoothly rather
        # than introducing a role/price-specific hard threshold.
        theta_stability = 1.0 - 0.55 * max(0.0, theta_drift - theta_support)
        policy_stability = 1.0 - 0.55 * max(0.0, policy_drift - policy_support)
        theta_trust = _clip(theta_support * theta_stability, 0.0, 1.0)
        policy_trust = _clip(policy_support * policy_stability, 0.0, 1.0)
        return {
            "evidence_channels": counts,
            "theta_support": round(theta_support, 6),
            "policy_support": round(policy_support, 6),
            "theta_total_variation_from_anchor": round(theta_drift, 6),
            "policy_total_variation_from_anchor": round(policy_drift, 6),
            "theta_trust": round(theta_trust, 6),
            "policy_trust": round(policy_trust, 6),
            "role_specific_gate": False,
            "uses_evaluator_truth": False,
        }

    def _set_decision_belief(
        self,
        mode: str,
        learned_buy: list[float],
        learned_resource: list[float],
    ) -> None:
        if mode == "ungated":
            FactorizedPreferencePolicyBeliefPlannerAgent._set_decision_belief(
                self, "continuous", learned_buy, learned_resource
            )
            return

        FactorizedPreferencePolicyBeliefPlannerAgent._set_decision_belief(
            self, mode, learned_buy, learned_resource
        )
        target_theta = list(
            self.buy_probs if self.game_kind == "buyer_seller" else self.resource_probs
        )
        target_policy = list(self.policy_probs)
        anchor_joint = (
            self.episode_anchor_joint_probs
            if self.episode_anchor_joint_probs
            else list(self._joint_probs())
        )
        anchor_theta, anchor_policy = self._marginals_from_joint(anchor_joint)
        gate = self._reliability_weights(
            target_theta, target_policy, anchor_theta, anchor_policy
        )
        theta_trust = float(gate["theta_trust"])
        policy_trust = float(gate["policy_trust"])
        if mode == "oracle":
            # Oracle is an evaluator-only diagnostic upper bound, not an online
            # action available to the learned V8 agent.
            theta_trust = 1.0
            gate["theta_trust"] = 1.0
            gate["oracle_diagnostic_bypass"] = True
        safe_theta = _normalize(
            [
                theta_trust * learned + (1.0 - theta_trust) * anchor
                for learned, anchor in zip(target_theta, anchor_theta)
            ]
        )
        safe_policy = _normalize(
            [
                policy_trust * learned + (1.0 - policy_trust) * anchor
                for learned, anchor in zip(target_policy, anchor_policy)
            ]
        )
        self._set_joint_probs(self._factorized_joint(safe_theta, safe_policy))
        gate.update(
            {
                "requested_mode": mode,
                "safe_theta_entropy": round(
                    _entropy(safe_theta) / math.log(len(safe_theta)), 6
                ),
                "safe_policy_entropy": round(
                    _entropy(safe_policy) / math.log(len(safe_policy)), 6
                ),
            }
        )
        if not self._in_shadow_intervention:
            self._last_reliability_gate = dict(gate)
            self.reliability_gate_history.append(dict(gate))

    def _shadow_actions(
        self,
        *,
        event: dict[str, Any] | None,
        learned_buy: list[float],
        learned_resource: list[float],
        pre_update_buy: list[float],
        pre_update_resource: list[float],
    ) -> dict[str, dict[str, Any]]:
        shadow = super()._shadow_actions(
            event=event,
            learned_buy=learned_buy,
            learned_resource=learned_resource,
            pre_update_buy=pre_update_buy,
            pre_update_resource=pre_update_resource,
        )
        saved_joint = list(self._joint_probs())
        try:
            self._in_shadow_intervention = True
            FactorizedPreferencePolicyBeliefPlannerAgent._set_decision_belief(
                self, "continuous", learned_buy, learned_resource
            )
            alternatives = self._build_candidates()
            alternative, _ = self._choose(alternatives, event)
            shadow["ungated"] = {
                "signature": self._action_signature(alternative),
                "posterior": self._posterior_json(),
            }

            anchor_theta, anchor_policy = self._marginals_from_joint(
                self.episode_anchor_joint_probs
            )
            self._set_joint_probs(self._factorized_joint(anchor_theta, anchor_policy))
            alternatives = self._build_candidates()
            alternative, _ = self._choose(alternatives, event)
            shadow["anchor"] = {
                "signature": self._action_signature(alternative),
                "posterior": self._posterior_json(),
            }
        finally:
            self._in_shadow_intervention = False
            restored = self._learned_joint_before_decision or saved_joint
            if self.game_kind == "buyer_seller":
                self.buy_joint_probs = list(restored)
            else:
                self.resource_joint_probs = list(restored)
            self._marginalize_joint()
            self._current_risk_gate_cache = None
        return shadow

    def _choose(
        self,
        candidates: list[dict[str, Any]],
        event: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        chosen, planner = super()._choose(candidates, event)
        planner.update(
            {
                "belief_mediation": "evidence_reliability_anchor_mixture",
                "reliability_gate": deepcopy(self._last_reliability_gate),
            }
        )
        return chosen, planner

    def _posterior_json(self) -> dict[str, Any]:
        base = super()._posterior_json()
        base["reliability_gate"] = deepcopy(self._last_reliability_gate)
        return base

    def episode_diagnostics(self) -> dict[str, Any]:
        result = super().episode_diagnostics()
        for mode in ("ungated", "anchor"):
            result["action_flip_counts"][mode] = sum(
                bool(row.get("update_opportunity")) and bool(row["flips"].get(mode))
                for row in self.episode_action_flips
            )
        if self.reliability_gate_history:
            result["mean_theta_trust"] = sum(
                float(row["theta_trust"]) for row in self.reliability_gate_history
            ) / len(self.reliability_gate_history)
            result["mean_policy_trust"] = sum(
                float(row["policy_trust"]) for row in self.reliability_gate_history
            ) / len(self.reliability_gate_history)
        else:
            result["mean_theta_trust"] = 0.0
            result["mean_policy_trust"] = 0.0
        result["reliability_gate_decisions"] = len(self.reliability_gate_history)
        return result

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        from pathlib import Path

        path = (
            Path(self.trace_dir)
            / self.agent_name.replace(" ", "_")
            / "framework_v8_decisions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
