"""Repeated natural-language agents for the Opponent Simulation comparison.

This module deliberately lives beside, rather than replacing, the earlier
single-game ``BeliefPlannerAgent``.  The two tracks answer different questions:
the old agent is a deterministic protocol diagnostic; the agents here retain
cross-episode memory and (for the framework variant) update a typed belief on
every observed turn.

中文阅读导引：``RepeatedLanguageAgent`` 是 matched Direct baseline，同时提供所有
方法共享的模型调用、跨局公开记忆、协议规范化与完整 trace。旧的
``OpponentSimulationAgent`` 用较少请求联合生成/评估候选；正式论文对齐版本位于
``paper_aligned_opponent_simulation.py``，会独立采样候选并逐候选 rollout。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from negotiationarena.agents.agents import Agent
from negotiationarena.openai_compat import ChatCompletionsHTTPClient


JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
REWARD_LIST = re.compile(r"<reward list>\s*\[([^]]+)]\s*</reward list>", re.I | re.S)
BUY_SELL_TRADE = re.compile(
    r"Player\s+RED\s+Gives\s+X\s*:\s*([0-9]+)\s*\|\s*"
    r"Player\s+BLUE\s+Gives\s+ZUP\s*:\s*([0-9]+)",
    re.I,
)
RESOURCE_TRADE = re.compile(
    r"Player\s+RED\s+Gives\s+[^|]+\|\s*Player\s+BLUE\s+Gives\s+.+",
    re.I | re.S,
)


def _canonical_resource_trade(text: str | None) -> str | None:
    """Canonicalize a complete two-party integer bundle without changing it."""
    if not text:
        return None
    parsed: dict[str, list[tuple[str, int]]] = {}
    for player, resources in re.findall(
        r"Player\s+(RED|BLUE)\s+Gives\s+([^|]+)", str(text), re.I
    ):
        item_pattern = re.compile(
            r"(?:([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+)|(-?\d+)\s*([A-Za-z][A-Za-z0-9_]*))"
        )
        items = []
        for match in item_pattern.finditer(resources):
            name = match.group(1) or match.group(4)
            amount = match.group(2) or match.group(3)
            items.append((name, amount))
        if not items:
            return None
        # Reject unparsed residue instead of silently dropping a malformed item.
        residue = item_pattern.sub("", resources).replace(",", "").strip()
        if residue:
            return None
        normalized_items = [(name.upper(), int(amount)) for name, amount in items]
        if len({name for name, _ in normalized_items}) != len(normalized_items):
            return None
        parsed[player.upper()] = normalized_items
    if set(parsed) != {"RED", "BLUE"}:
        return None

    def render(player: str) -> str:
        return ", ".join(f"{name}: {amount}" for name, amount in parsed[player])

    return f"Player RED Gives {render('RED')} | Player BLUE Gives {render('BLUE')}"


class ProtocolFormatError(ValueError):
    """Raised before the legacy arena parser sees an invalid model action."""


def _json_from_text(text: str) -> Any:
    """Extract a JSON value without silently evaluating model output."""
    candidates = [match.group(1).strip() for match in JSON_BLOCK.finditer(text)]
    stripped = text.strip()
    candidates.extend([stripped])
    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if starts:
        candidates.append(stripped[min(starts) :])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("model response did not contain valid JSON")


def _compact(value: Any, max_chars: int = 6000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"


def _tag_content(response: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        response,
        re.I | re.S,
    )
    return match.group(1).strip() if match else None


def _protocol_valid(response: str, game_kind: str) -> bool:
    """Validate the exact grammar consumed by NegotiationArena's parser."""
    common = ["my resources", "my goals", "reason", "player answer", "newly proposed trade", "message"]
    required = ["my name", *common] if game_kind == "resource_exchange" else ["proposal count", *common]
    contents = {tag: _tag_content(response, tag) for tag in required}
    if any(value is None for value in contents.values()):
        return False
    # Arena parses private resources after this guard.  Validate them here so a
    # malformed self-report cannot pass protocol validation and crash later in
    # ``Resources.from_string``.
    resource_text = str(contents["my resources"])
    resource_items = [item.strip() for item in resource_text.split(",") if item.strip()]
    if not resource_items or any(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*:\s*-?\d+", item) is None
        for item in resource_items
    ):
        return False
    answer = str(contents["player answer"]).upper()
    trade = str(contents["newly proposed trade"])
    allowed = {"ACCEPT", "NONE"} if game_kind == "resource_exchange" else {"ACCEPT", "REJECT", "PROPOSAL"}
    if answer not in allowed:
        return False
    if answer in {"ACCEPT", "REJECT"}:
        return trade.upper() == "NONE"
    if game_kind == "buyer_seller":
        match = BUY_SELL_TRADE.fullmatch(trade)
        return bool(match and int(match.group(1)) == 1)
    # Resource exchange uses NONE both for waiting and for counter-proposing.
    if trade.upper() == "NONE":
        return True
    canonical_trade = _canonical_resource_trade(trade)
    return canonical_trade is not None and trade.strip() == canonical_trade


def _legacy_protocol_valid(response: str, game_kind: str) -> bool:
    """The pre-V2 candidate guard, retained so ``protocol_mode=raw`` is stable."""
    common = ["my resources", "my goals", "reason", "player answer", "newly proposed trade", "message"]
    required = ["my name", *common] if game_kind == "resource_exchange" else ["proposal count", *common]
    lowered = response.lower()
    if any(f"<{tag}>" not in lowered or f"</{tag}>" not in lowered for tag in required):
        return False
    answer_match = re.search(r"<player answer>\s*([^<]+)\s*</player answer>", response, re.I)
    trade_match = re.search(r"<newly proposed trade>\s*(.*?)\s*</newly proposed trade>", response, re.I | re.S)
    if not answer_match or not trade_match:
        return False
    answer = answer_match.group(1).strip().upper()
    trade = trade_match.group(1).strip()
    allowed = {"ACCEPT", "NONE"} if game_kind == "resource_exchange" else {"ACCEPT", "REJECT", "PROPOSAL"}
    if answer not in allowed:
        return False
    if answer == "ACCEPT" and trade.upper() != "NONE":
        return False
    if answer == "PROPOSAL" and ("Player RED Gives" not in trade or "Player BLUE Gives" not in trade):
        return False
    return True


def _lenient_trade_content(response: str) -> str | None:
    """Read the trade block even when the closing tag was truncated/misspelled."""
    start = re.search(r"<newly proposed trade>\s*", response, re.I)
    if not start:
        return None
    suffix = response[start.end() :]
    end = re.search(r"</newly proposed trade>|<message>", suffix, re.I)
    return suffix[: end.start()].strip() if end else suffix.strip()


def _canonicalize_protocol_response(response: str, game_kind: str) -> str | None:
    """Deterministically render a parser-safe response when its action is recoverable.

    This function never guesses a missing price/bundle and never asks another model.
    It only canonicalizes delimiters/tags around an action already present in the raw
    output.  Missing semantic fields are delegated to the optional format retry.
    """
    order = (
        ["my name", "my resources", "my goals", "reason", "player answer", "message"]
        if game_kind == "resource_exchange"
        else ["proposal count", "my resources", "my goals", "reason", "player answer", "message"]
    )
    values = {tag: _tag_content(response, tag) for tag in order}
    # Missing answer is unambiguously recoverable when an actual proposed trade
    # is already present: Buy--Sell calls it PROPOSAL, Resource calls it NONE.
    raw_trade = _lenient_trade_content(response)
    if values.get("player answer") is None and raw_trade and raw_trade.upper() != "NONE":
        values["player answer"] = "PROPOSAL" if game_kind == "buyer_seller" else "NONE"
    if any(value is None for value in values.values()):
        return None
    answer = str(values["player answer"]).strip().upper()
    if game_kind == "buyer_seller":
        if answer in {"ACCEPT", "REJECT"}:
            trade = "NONE"
        elif answer == "PROPOSAL":
            if raw_trade is None:
                return None
            price_match = re.search(r"ZUP\s*:\s*([0-9]+)", raw_trade, re.I)
            if not price_match:
                return None
            price = int(price_match.group(1))
            trade = f"Player RED Gives X: 1 | Player BLUE Gives ZUP: {price}"
        else:
            return None
        rendered_order = [
            "proposal count", "my resources", "my goals", "reason",
            "player answer", "newly proposed trade", "message",
        ]
    else:
        if answer == "ACCEPT":
            trade = "NONE"
        elif answer == "NONE" and raw_trade is not None:
            if raw_trade.upper() == "NONE":
                trade = "NONE"
            else:
                trade = _canonical_resource_trade(raw_trade)
            if trade is None:
                return None
        else:
            return None
        rendered_order = [
            "my name", "my resources", "my goals", "reason",
            "player answer", "message", "newly proposed trade",
        ]
    values["player answer"] = answer
    values["newly proposed trade"] = trade
    return "\n".join(f"<{tag}> {values[tag]} </{tag}>" for tag in rendered_order)


# Direct baseline：没有结构化 opponent posterior 或 executable planner；模型根据
# 相同 repeated history 与 strategic-brainstorming prompt 直接生成协议动作。
class RepeatedLanguageAgent(Agent):
    """A dependency-free OpenAI-compatible agent with repeated-game memory."""

    method_name = "direct"

    def __init__(
        self,
        *,
        agent_name: str,
        model: str,
        base_url: str | None,
        api_key: str | None,
        temperature: float = 0.7,
        max_tokens: int = 900,
        seed: int = 0,
        trace_dir: str | None = None,
        history_window: int = 20,
        context_char_limit: int = 38000,
        protocol_mode: str = "raw",
        protocol_repair_attempts: int = 1,
        game_turn_limit: int = 10,
    ):
        super().__init__(agent_name)
        if protocol_mode not in {"raw", "normalize", "normalize_retry"}:
            raise ValueError(f"Unknown protocol_mode: {protocol_mode}")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.trace_dir = trace_dir
        self.history_window = history_window
        self.context_char_limit = context_char_limit
        self.protocol_mode = protocol_mode
        self.protocol_repair_attempts = max(0, int(protocol_repair_attempts))
        self.game_turn_limit = max(1, int(game_turn_limit))
        self.protocol_invalid_raw_count = 0
        self.protocol_deterministic_repair_count = 0
        self.protocol_retry_count = 0
        self.protocol_repair_failure_count = 0
        self.prompt_entity_initializer = "system"
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.client = ChatCompletionsHTTPClient(
            api_key=api_key or "EMPTY", base_url=base_url
        )
        self.conversation: list[dict[str, str]] = []
        self.episode_records: list[dict[str, Any]] = []
        self.episode_index = 1
        self.total_episodes = 20
        self.game_kind = "unknown"
        self.private_objective = "maximize my private utility"
        self.call_index = 0
        self.episode_trace: list[dict[str, Any]] = []
        self.starts_episode: bool | None = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, "HTTPClient" if key == "client" else deepcopy(value, memo))
        return result

    def get_state(self):
        """Keep NegotiationArena logs JSON-safe and bounded."""
        return {
            "class": self.__class__.__name__,
            "agent_name": self.agent_name,
            "method": self.method_name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "game_kind": self.game_kind,
            "completed_episodes": len(self.episode_records),
            "conversation": deepcopy(self.conversation),
            "call_count": self.call_index,
            "protocol_mode": self.protocol_mode,
            "game_turn_limit": self.game_turn_limit,
            "protocol_invalid_raw_count": self.protocol_invalid_raw_count,
            "protocol_deterministic_repair_count": self.protocol_deterministic_repair_count,
            "protocol_retry_count": self.protocol_retry_count,
            "protocol_repair_failure_count": self.protocol_repair_failure_count,
        }

    def prepare_episode(
        self,
        *,
        episode_index: int,
        total_episodes: int,
        game_kind: str,
        private_objective: str,
        starts_episode: bool | None = None,
    ) -> None:
        self.episode_index = episode_index
        self.total_episodes = total_episodes
        self.game_kind = game_kind
        self.private_objective = private_objective
        self.starts_episode = starts_episode
        self.episode_trace = []

    def init_agent(self, system_prompt, role):
        memory = self._memory_prompt()
        protocol_stability = ""
        if self.protocol_mode != "raw":
            protocol_stability = (
                "\n\nPROTOCOL RELIABILITY OVERRIDE:\n"
                "Keep <reason> under 100 words. Finish every required closing tag. "
                "For a buyer/seller proposal, use exactly: Player RED Gives X: 1 | "
                "Player BLUE Gives ZUP: INTEGER. Never replace the vertical bar with a comma. "
                "Output no text after the final required tag."
            )
        augmented_system = f"{system_prompt}\n\n{memory}{protocol_stability}"
        if self.starts_episode is None:
            # Preserve NegotiationArena's original RED-start convention when a
            # caller does not explicitly provide the paper's first-mover role.
            self.starts_episode = self.agent_name.upper().endswith("RED")
        if self.starts_episode:
            # Match the original ChatGPTAgent message layout: the starter's
            # identity is the initial user turn, which also triggers generation.
            self.conversation = [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": str(role)},
            ]
        else:
            self.conversation = [
                {"role": "system", "content": f"{augmented_system}{role}"},
            ]

    def _memory_prompt(self) -> str:
        records = self.episode_records[-self.history_window :]
        prices = [row.get("deal") for row in records]
        rewards = [row.get("own_reward") for row in records]
        return (
            f"Now Episode {self.episode_index}/{self.total_episodes} begins. Please start a new "
            "episode of negotiation from scratch.\n"
            "Here are summarized results from all previous episodes:\n"
            f"The historical public deals from each episode sequentially: {prices}.\n"
            f"The reward you received from each episode sequentially: {rewards}.\n"
            "Remember, at every step of decision making, first summarize and then reflect on "
            "the negotiations from previous episodes. Through the reflection, aim to "
            "self-improve your decision-making across episodes. Never assume access to the "
            "opponent's private utility unless they publicly disclosed it."
        )

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": str(message)})

    def chat(self):
        return self._call("direct_action", self.conversation)

    def think(self):
        """Generate an action, then optionally validate/repair before arena parsing."""
        raw_response = self.chat()
        response = raw_response
        if self.protocol_mode != "raw" and not _protocol_valid(response, self.game_kind):
            self.protocol_invalid_raw_count += 1
            normalized = _canonicalize_protocol_response(response, self.game_kind)
            if normalized is not None and _protocol_valid(normalized, self.game_kind):
                response = normalized
                self.protocol_deterministic_repair_count += 1
                self._append_protocol_event("deterministic_repair", raw_response, response)
            elif self.protocol_mode == "normalize_retry":
                response = self._retry_protocol_format(raw_response)
            if not _protocol_valid(response, self.game_kind):
                self.protocol_repair_failure_count += 1
                self._append_protocol_event("repair_failed", raw_response, response)
                raise ProtocolFormatError(
                    "model action is not valid under the NegotiationArena XML/trade grammar"
                )
        self.update_conversation_tracking("assistant", response)
        return response

    def _retry_protocol_format(self, raw_response: str) -> str:
        response = raw_response
        for _ in range(self.protocol_repair_attempts):
            self.protocol_retry_count += 1
            if self.game_kind == "buyer_seller":
                contract = """
Required order:
<proposal count>...</proposal count>
<my resources>...</my resources>
<my goals>...</my goals>
<reason>at most 60 words</reason>
<player answer>ONE ACTION TOKEN</player answer>
<newly proposed trade>Player RED Gives X: 1 | Player BLUE Gives ZUP: INTEGER</newly proposed trade>
<message>...</message>
ONE ACTION TOKEN means choose exactly one of PROPOSAL, ACCEPT, or REJECT. Never copy a list
of choices into the tag. For ACCEPT or REJECT, the trade content must be NONE. For a
non-NONE trade, the answer must be PROPOSAL.
"""
            else:
                contract = """
Required order:
<my name>...</my name>
<my resources>...</my resources>
<my goals>...</my goals>
<reason>at most 60 words</reason>
<player answer>ONE ACTION TOKEN</player answer>
<message>...</message>
<newly proposed trade>NONE or Player RED Gives ... | Player BLUE Gives ...</newly proposed trade>
ONE ACTION TOKEN means choose exactly one of ACCEPT or NONE. Never output ACCEPT|NONE or
copy any list of choices. If a non-NONE trade is present, the answer must be NONE. If the
answer is ACCEPT, the trade must be NONE. Resource quantities must be integers.
"""
            canonical_resources = _tag_content(self.conversation[0]["content"], "my resources")
            canonical_goal = _tag_content(self.conversation[0]["content"], "my goals")
            prompt = f"""
Repair only the serialization of the action below. Preserve the intended action, every
numeric offer/bundle, private facts and public message. Do not improve the strategy and do
not choose a new price. If the intended action is not recoverable, return <repair failure/>.
Return the complete protocol response only, with no markdown or commentary.
Use these canonical private fields rather than copying malformed variants:
my resources = {canonical_resources}
my goals = {canonical_goal}
{contract}
RAW RESPONSE:
{raw_response}
"""
            candidate = self._call(
                "protocol_format_retry",
                [
                    {"role": "system", "content": "You are a deterministic protocol serializer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
            normalized = _canonicalize_protocol_response(candidate, self.game_kind)
            response = normalized if normalized is not None else candidate
            if _protocol_valid(response, self.game_kind):
                self._append_protocol_event("model_format_retry", raw_response, response)
                return response
        return response

    def _append_protocol_event(self, kind: str, raw: str, final: str) -> None:
        if not self.trace_dir:
            return
        path = Path(self.trace_dir) / self.agent_name.replace(" ", "_") / "protocol_repairs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "episode": self.episode_index,
            "kind": kind,
            "game_kind": self.game_kind,
            "raw_response": raw,
            "final_response": final,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "final_sha256": hashlib.sha256(final.encode("utf-8")).hexdigest(),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _context_safe(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if sum(len(str(m.get("content", ""))) for m in messages) <= self.context_char_limit:
            return messages
        # Preserve the private game rules and the most recent interaction.  Repeated
        # history is already represented compactly in the system reminder.
        first = messages[0]
        remaining = self.context_char_limit - len(str(first.get("content", "")))
        kept: list[dict[str, str]] = []
        used = 0
        for message in reversed(messages[1:]):
            size = len(str(message.get("content", "")))
            if used + size > max(remaining, 0):
                break
            kept.append(message)
            used += size
        return [first, *reversed(kept)]

    def _call(
        self,
        stage: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        safe_messages = self._context_safe(messages)
        started = time.time()
        error = None
        response = ""
        try:
            response = self.client.complete(
                model=self.model,
                messages=safe_messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                seed=self.seed + self.call_index,
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.call_index += 1
            event = {
                "call_index": self.call_index,
                "episode": self.episode_index,
                "stage": stage,
                "model": self.model,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "messages": safe_messages,
                "request_sha256": hashlib.sha256(
                    json.dumps(safe_messages, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "response": response,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            self.episode_trace.append(event)
            self._append_call(event)

    def _append_call(self, event: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        path = Path(self.trace_dir) / self.agent_name.replace(" ", "_") / "calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_episode(
        self,
        *,
        transcript: list[dict[str, Any]],
        own_reward: float,
        agreement: bool,
        deal: Any,
    ) -> None:
        public_turns = [
            {
                "turn": row.get("current_iteration"),
                "speaker": row.get("turn"),
                "public": row.get("player_public_answer_string", ""),
            }
            for row in transcript
            if isinstance(row.get("current_iteration"), int)
        ]
        self.episode_records.append(
            {
                "episode": self.episode_index,
                "agreement": agreement,
                "deal": deal,
                "own_reward": own_reward,
                "public_transcript": public_turns,
            }
        )


# 历史本地 Opponent Simulation 近似。它适合工程对照，但正式论文比较优先使用
# ``PaperAlignedOpponentSimulationAgent``，不要混淆二者结果。
class OpponentSimulationAgent(RepeatedLanguageAgent):
    """Paper-compatible BoN opponent-simulation baseline.

    It uses three calls per actual action: five strategic candidates, an
    opponent-policy simulation conditioned on repeated history, and selection
    from completed simulated trajectories.  The simulator remains a black-box
    behavior policy, matching the key abstraction in arXiv:2602.19309.
    """

    method_name = "opponent_simulation_bon5"

    def __init__(self, *args, candidate_count: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_count = candidate_count

    def get_state(self):
        return {**super().get_state(), "candidate_count": self.candidate_count}

    def chat(self):
        candidate_prompt = f"""
At this turn, brainstorm exactly {self.candidate_count} genuinely different high-level
negotiation strategies (for example anchoring, reciprocity, fairness, probing, or a safe
fallback), then instantiate each strategy as a COMPLETE protocol-valid response. Candidate
responses must obey all XML/tag rules in the system prompt. Return JSON only:
{{"candidates":[{{"id":1,"strategy":"...","response":"..."}}, ...]}}
"""
        raw_candidates = self._call(
            "oppsim_candidate_generation",
            [*self.conversation, {"role": "user", "content": candidate_prompt}],
            temperature=max(self.temperature, 0.7),
            max_tokens=max(self.max_tokens, 1800),
        )
        candidates = self._parse_candidates(raw_candidates)
        if not candidates:
            return self._call("oppsim_candidate_fallback", self.conversation)

        repeated_history = self.episode_records[-self.history_window :]
        simulator_prompt = f"""
Learn a time-averaged behavior policy for the opponent from the completed repeated-game
history and the current partial dialogue. Do not invent access to their private payoff.
Explain stable anchors, concession patterns, acceptance behavior, and uncertainty. When
uncertain, use the optimistic assumption from the Opponent Simulation baseline.
Completed history: {_compact(repeated_history, 18000)}
Current dialogue: {_compact(self.conversation)}
Return a concise opponent behavior profile for use in future-trajectory simulation.
"""
        opponent_profile = self._call(
            "oppsim_opponent_model",
            [{"role": "system", "content": self.conversation[0]["content"]},
             {"role": "user", "content": simulator_prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        rollout_prompt = f"""
You are selecting an action for {self.agent_name}. For every candidate below, simulate the
ENTIRE remaining negotiation through agreement or the ten-step deadline. Play both sides in
concrete natural-language turns, condition the opponent on the learned profile, and calculate
my reward using my private objective: {self.private_objective}. No agreement has reward 0 and
negative-payoff agreements should be avoided.
Opponent profile: {opponent_profile}
Candidates: {_compact(candidates, 14000)}
Return JSON only: {{"evaluations":[{{"id":1,"predicted_reward":0.0,
"agreement_probability":0.0,"trajectory":"..."}}],"chosen_id":1}}.
"""
        raw_rollouts = self._call(
            "oppsim_future_rollout_and_selection",
            [{"role": "system", "content": self.conversation[0]["content"]},
             {"role": "user", "content": rollout_prompt}],
            temperature=0.2,
            max_tokens=max(self.max_tokens, 2600),
        )
        chosen_id = self._chosen_id(raw_rollouts, candidates)
        chosen = next((row for row in candidates if row["id"] == chosen_id), candidates[0])
        return chosen["response"]

    def _parse_candidates(self, text: str) -> list[dict[str, Any]]:
        try:
            data = _json_from_text(text)
            rows = data.get("candidates", []) if isinstance(data, dict) else []
        except ValueError:
            rows = []
        parsed = []
        for idx, row in enumerate(rows[: self.candidate_count], 1):
            response = str(row.get("response", ""))
            validator = _protocol_valid if self.protocol_mode != "raw" else _legacy_protocol_valid
            if not validator(response, self.game_kind):
                continue
            parsed.append(
                {"id": int(row.get("id", idx)), "strategy": str(row.get("strategy", "")), "response": response}
            )
        return parsed

    @staticmethod
    def _chosen_id(text: str, candidates: list[dict[str, Any]]) -> int:
        try:
            data = _json_from_text(text)
            choice = int(data.get("chosen_id"))
            if any(row["id"] == choice for row in candidates):
                return choice
            evaluations = data.get("evaluations", [])
            ranked = sorted(
                evaluations,
                key=lambda row: float(row.get("predicted_reward", float("-inf"))),
                reverse=True,
            )
            if ranked:
                return int(ranked[0]["id"])
        except (ValueError, TypeError, KeyError):
            pass
        reward_match = REWARD_LIST.search(text)
        if reward_match:
            try:
                scores = [float(item.strip()) for item in reward_match.group(1).split(",")]
                return candidates[max(range(min(len(scores), len(candidates))), key=scores.__getitem__)]["id"]
            except (ValueError, IndexError):
                pass
        return candidates[0]["id"]


class DualTimescaleBeliefPlannerAgent(RepeatedLanguageAgent):
    """Universal LLM belief updater + belief-usable planner.

    The posterior schema is game-independent.  Cross-episode records define a
    meta-prior, while every new public opponent message triggers a fresh typed
    posterior update before candidate generation and planning.
    """

    method_name = "dual_timescale_belief_planner_v1"

    def __init__(
        self,
        *args,
        candidate_count: int = 5,
        belief_mode: str = "continuous",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.candidate_count = candidate_count
        self.belief_mode = belief_mode
        self.posterior = self._empty_posterior()
        self.posterior_history: list[dict[str, Any]] = []

    @staticmethod
    def _empty_posterior() -> dict[str, Any]:
        return {
            "opponent_type_probabilities": {
                "competitive": 0.25,
                "cooperative": 0.25,
                "fairness_seeking": 0.25,
                "adaptive": 0.25,
            },
            "reservation_or_acceptance_threshold": {"low": None, "mean": None, "high": None},
            "issue_preferences": [],
            "concession_dynamics": {"direction": "unknown", "rate": "unknown"},
            "confidence": 0.0,
            "evidence": [],
            "unresolved_uncertainties": ["No opponent action observed yet"],
        }

    def prepare_episode(self, **kwargs):
        super().prepare_episode(**kwargs)
        self.posterior = self._meta_prior()
        self.posterior_history = []

    def get_state(self):
        return {
            **super().get_state(),
            "belief_mode": self.belief_mode,
            "posterior": deepcopy(self.posterior),
            "posterior_updates": len(self.posterior_history),
        }

    def _meta_prior(self) -> dict[str, Any]:
        prior = self._empty_posterior()
        if self.episode_records:
            prior["confidence"] = min(0.5, 0.1 + 0.02 * len(self.episode_records))
            prior["evidence"] = [
                {
                    "observation": "public repeated-game outcomes are available",
                    "supports": "cross-episode behavioral prior",
                    "reliability": 0.6,
                }
            ]
            prior["unresolved_uncertainties"] = [
                "Past outcomes confound opponent preference with their response to our old policies"
            ]
        return prior

    def chat(self):
        should_update = self.belief_mode == "continuous" or not self.posterior_history
        if should_update:
            self.posterior = self._update_belief()
        self.posterior_history.append(deepcopy(self.posterior))
        candidates = self._generate_candidates()
        if not candidates:
            return self._call("framework_direct_fallback", self.conversation)
        chosen_id, planner = self._plan(candidates)
        chosen = next((row for row in candidates if row["id"] == chosen_id), candidates[0])
        decision = {
            "episode": self.episode_index,
            "actual_turn": len(self.posterior_history),
            "posterior": self.posterior,
            "candidates": candidates,
            "planner": planner,
            "chosen_id": chosen["id"],
        }
        self._append_decision(decision)
        return chosen["response"]

    def _update_belief(self) -> dict[str, Any]:
        prompt = f"""
Maintain a calibrated posterior over the opponent in a natural-language negotiation. Use only
their public actions and messages; do not infer private facts from role names alone. Treat
silence and persuasive language as weak evidence, numerical concessions/acceptances as stronger
evidence, and explicitly preserve uncertainty. Update the previous posterior rather than
rewriting an unconstrained personality story.

Game family: {self.game_kind}
Completed episodes against this opponent: {_compact(self.episode_records[-self.history_window:], 18000)}
Previous within-episode posterior: {_compact(self.posterior)}
Current public dialogue: {_compact(self.conversation)}

Return JSON only with exactly these fields:
{{"opponent_type_probabilities":{{"competitive":0.0,"cooperative":0.0,
"fairness_seeking":0.0,"adaptive":0.0}},
"reservation_or_acceptance_threshold":{{"low":null,"mean":null,"high":null}},
"issue_preferences":[{{"issue":"...","relative_weight":0.0,"confidence":0.0}}],
"concession_dynamics":{{"direction":"...","rate":"..."}},"confidence":0.0,
"evidence":[{{"observation":"...","supports":"...","reliability":0.0}}],
"unresolved_uncertainties":["..."]}}.
Probabilities must sum to one; intervals must be ordered when numeric.
"""
        raw = self._call(
            "framework_continuous_belief_update",
            [{"role": "system", "content": "You are a conservative Bayesian opponent-modeling module."},
             {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
        )
        try:
            return self._validate_posterior(_json_from_text(raw))
        except (ValueError, TypeError):
            fallback = deepcopy(self.posterior)
            fallback["unresolved_uncertainties"] = [
                *fallback.get("unresolved_uncertainties", []),
                "The latest belief update was not parseable; prior retained",
            ]
            return fallback

    def _validate_posterior(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("posterior must be an object")
        result = self._empty_posterior()
        probs = data.get("opponent_type_probabilities", {})
        clean_probs = {key: max(0.0, float(probs.get(key, 0.0))) for key in result["opponent_type_probabilities"]}
        total = sum(clean_probs.values()) or 1.0
        result["opponent_type_probabilities"] = {key: round(value / total, 6) for key, value in clean_probs.items()}
        threshold = data.get("reservation_or_acceptance_threshold", {})
        values = [threshold.get(key) for key in ("low", "mean", "high")]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            numeric = [float(value) for value in values if value is not None]
            if len(numeric) == 3 and numeric != sorted(numeric):
                numeric.sort()
                values = numeric
            result["reservation_or_acceptance_threshold"] = dict(zip(("low", "mean", "high"), values))
        result["issue_preferences"] = list(data.get("issue_preferences", []))[:12]
        result["concession_dynamics"] = dict(data.get("concession_dynamics", result["concession_dynamics"]))
        result["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        result["evidence"] = list(data.get("evidence", []))[:12]
        result["unresolved_uncertainties"] = list(data.get("unresolved_uncertainties", []))[:12]
        return result

    def _generate_candidates(self) -> list[dict[str, Any]]:
        prompt = f"""
Generate exactly {self.candidate_count} diverse, complete, protocol-valid responses for my next
move. Include: (1) posterior exploitation, (2) an information-gathering probe, (3) an
agreement-safe option, (4) a reciprocal option, and (5) a distinct fallback. Do not reveal my
private objective. Every response must obey the XML/tag format in the game system prompt.
My private objective: {self.private_objective}
Opponent posterior: {_compact(self.posterior)}
Return JSON only: {{"candidates":[{{"id":1,"kind":"exploit|probe|safe|reciprocal|fallback",
"rationale":"...","response":"complete response"}}, ...]}}.
"""
        raw = self._call(
            "framework_posterior_conditioned_candidates",
            [*self.conversation, {"role": "user", "content": prompt}],
            temperature=max(self.temperature, 0.7),
            max_tokens=max(self.max_tokens, 2000),
        )
        try:
            data = _json_from_text(raw)
            rows = data.get("candidates", [])
        except (ValueError, AttributeError):
            return []
        candidates = []
        for idx, row in enumerate(rows[: self.candidate_count], 1):
            response = str(row.get("response", ""))
            validator = _protocol_valid if self.protocol_mode != "raw" else _legacy_protocol_valid
            if not validator(response, self.game_kind):
                continue
            candidates.append(
                {
                    "id": int(row.get("id", idx)),
                    "kind": str(row.get("kind", "unspecified")),
                    "rationale": str(row.get("rationale", "")),
                    "response": response,
                }
            )
        return candidates

    def _plan(self, candidates: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        prompt = f"""
Act as a belief-usable negotiation planner. Score every candidate separately on:
expected own payoff, probability of agreement, downside/CVaR risk under posterior uncertainty,
information gain useful later in this episode or repeated game, future option value, and private
information leakage. Do not reward a candidate merely for mentioning the belief. Prefer robust
expected utility, while allowing a bounded information-gain bonus early in negotiation.

Private objective: {self.private_objective}
Posterior: {_compact(self.posterior)}
Candidates: {_compact(candidates, 15000)}
Return JSON only: {{"scores":[{{"id":1,"expected_own_payoff":0.0,
"agreement_probability":0.0,"downside_risk":0.0,"information_gain":0.0,
"future_option_value":0.0,"leakage_cost":0.0,"total":0.0,"explanation":"..."}}],
"chosen_id":1,"decision_reason":"..."}}.
"""
        raw = self._call(
            "framework_belief_usable_planning",
            [{"role": "system", "content": "You are a risk-aware counterfactual negotiation planner."},
             {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1600,
        )
        try:
            data = _json_from_text(raw)
            chosen = int(data.get("chosen_id"))
            if not any(row["id"] == chosen for row in candidates):
                scores = data.get("scores", [])
                chosen = int(max(scores, key=lambda row: float(row.get("total", float("-inf"))))["id"])
            return chosen, data
        except (ValueError, TypeError, KeyError):
            return candidates[0]["id"], {"parse_error": True, "raw_response": raw}

    def _append_decision(self, decision: dict[str, Any]) -> None:
        if not self.trace_dir:
            return
        path = Path(self.trace_dir) / self.agent_name.replace(" ", "_") / "decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
