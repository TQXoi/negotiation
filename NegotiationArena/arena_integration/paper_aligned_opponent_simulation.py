"""Paper-aligned Opponent Simulation baseline with independent rollouts.

The authors release prompts and example logs but not the full Algorithm-1
runner.  The older local ``OpponentSimulationAgent`` compresses all candidate
rollouts into one request.  This version follows the paper more closely:

1. N independent samples from the strategic-brainstorming actor create N actions;
2. each action is evaluated in an independent opponent-conditioned rollout;
3. the action with the largest simulated focal reward is executed.

It is still labelled a paper-aligned reimplementation, not an exact official
reproduction: each full trajectory is serialized in one request rather than
executed as an externally alternating multi-call simulator.

中文阅读导引：每个真实动作先独立采样 N 个候选，再为每个候选调用一次
opponent-conditioned trajectory evaluator，最后执行预测 focal reward 最大的动作。
因此 N=5 时每个真实决策最多约 10 次模型调用，计算量远高于 Direct 和我们的
structured framework。它是 paper-aligned reimplementation，不应写成官方代码复现。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena_integration.repeated_agents import (
    OpponentSimulationAgent,
    _canonicalize_protocol_response,
    _compact,
    _json_from_text,
    _legacy_protocol_valid,
    _protocol_valid,
)


class PaperAlignedOpponentSimulationAgent(OpponentSimulationAgent):
    """正式直接竞争 baseline：independent BoN candidates + per-candidate rollout。"""
    method_name = "paper_aligned_bon5_opponent_simulation_v2"

    def chat(self):
        # Algorithm 1 samples actions independently from the acting policy.  Do
        # not ask one model call to author all candidates: doing that couples the
        # samples and was a material approximation in the historical local
        # baseline.  The exact strategic-brainstorming instruction already lives
        # in the game system prompt, just as in the released example logs.
        candidates = []
        for candidate_id in range(1, self.candidate_count + 1):
            response = self._call(
                f"oppsim_paper_candidate_sample_{candidate_id}",
                self.conversation,
                temperature=max(self.temperature, 0.7),
                max_tokens=max(self.max_tokens, 1600),
            )
            validator = _protocol_valid if self.protocol_mode != "raw" else _legacy_protocol_valid
            if self.protocol_mode != "raw" and not validator(response, self.game_kind):
                self.protocol_invalid_raw_count += 1
                normalized = _canonicalize_protocol_response(response, self.game_kind)
                if normalized is not None and validator(normalized, self.game_kind):
                    self.protocol_deterministic_repair_count += 1
                    self._append_protocol_event(
                        "oppsim_internal_candidate_deterministic_repair",
                        response,
                        normalized,
                    )
                    response = normalized
            if validator(response, self.game_kind):
                candidates.append(
                    {
                        "id": candidate_id,
                        "strategy": self._strategy_declaration(response),
                        "response": response,
                    }
                )
        if not candidates:
            return self._call("oppsim_paper_direct_fallback", self.conversation)

        evaluations = []
        completed_history = self.episode_records[-self.history_window :]
        for candidate in candidates:
            prompt = f"""
You are the opponent-policy simulation and trajectory evaluator from a repeated
negotiation algorithm. Infer the actual opponent's time-averaged behavior only from
completed public history and the current partial dialogue. First summarize stable anchors,
concession patterns, acceptance behavior and uncertainty. Do not infer private facts from
role names. When uncertain, be optimistic for the focal agent.

Starting with the locked candidate below, simulate a concrete alternating dialogue through
agreement, rejection, or the ten-turn deadline. Follow the focal base policy for its future
moves and the inferred opponent policy for opponent moves. Respect the exact game protocol,
resource conservation and the focal private objective. No agreement has focal reward 0.

Focal agent: {self.agent_name}
Focal private objective: {self.private_objective}
Completed episodes: {_compact(completed_history, 16000)}
Current dialogue: {_compact(self.conversation, 12000)}
Locked first action: {_compact(candidate, 7000)}

Return JSON only:
{{"id":{candidate['id']},"opponent_behavior_summary":"...",
"trajectory":"concrete future turns","terminal":"agreement|rejection|deadline",
"predicted_focal_reward":0.0,"agreement_probability":0.0,
"uncertainty":"..."}}
"""
            raw = self._call(
                f"oppsim_paper_independent_rollout_{candidate['id']}",
                [
                    {
                        "role": "system",
                        "content": (
                            "Authentically role-play an unknown opponent from history, then evaluate "
                            "the completed trajectory. Never change the locked first action."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max(self.max_tokens, 2200),
            )
            evaluations.append(self._parse_rollout(raw, candidate["id"]))

        chosen_eval = max(
            evaluations,
            key=lambda row: (
                float(row.get("predicted_focal_reward", float("-inf"))),
                float(row.get("agreement_probability", 0.0)),
            ),
        )
        chosen = next(
            (row for row in candidates if row["id"] == chosen_eval["id"]),
            candidates[0],
        )
        self._append_paper_decision(
            {
                "episode": self.episode_index,
                "candidates": candidates,
                "evaluations": evaluations,
                "chosen_id": chosen["id"],
                "implementation_level": "paper_aligned_independent_serialized_rollouts",
            }
        )
        return chosen["response"]

    @staticmethod
    def _parse_rollout(text: str, candidate_id: int) -> dict[str, Any]:
        try:
            data = _json_from_text(text)
            reward = float(data.get("predicted_focal_reward"))
            probability = max(0.0, min(1.0, float(data.get("agreement_probability", 0.0))))
            return {
                "id": candidate_id,
                "predicted_focal_reward": reward,
                "agreement_probability": probability,
                "terminal": data.get("terminal"),
                "opponent_behavior_summary": data.get("opponent_behavior_summary"),
                "trajectory": data.get("trajectory"),
                "uncertainty": data.get("uncertainty"),
                "parse_error": False,
            }
        except (ValueError, TypeError, AttributeError):
            return {
                "id": candidate_id,
                # Keep trace JSON standards-compliant (``-Infinity`` is not
                # valid JSON even though Python's encoder permits it).
                "predicted_focal_reward": -1e12,
                "agreement_probability": 0.0,
                "raw_response": text,
                "parse_error": True,
            }

    @staticmethod
    def _strategy_declaration(response: str) -> str:
        import re

        match = re.search(
            r"<(?:self-selected )?strategy declaration>\s*(.*?)\s*</(?:self-selected )?strategy(?: declaration)?>",
            response,
            re.I | re.S,
        )
        return match.group(1).strip() if match else "independent_actor_sample"

    def _append_paper_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        path = Path(self.trace_dir) / self.agent_name.replace(" ", "_") / "oppsim_paper_decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
