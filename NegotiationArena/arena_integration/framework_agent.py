from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from arena_integration.belief import ReservationBelief
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO
from negotiationarena.openai_compat import ChatCompletionsHTTPClient


class BeliefPlannerAgent(Agent):
    """Buy/Sell focal agent with continuous belief and action-locked planner.

    The LLM is used only to realize a short public message. Numeric price and
    ACCEPT/PROPOSAL are selected by the planner, preventing generator drift.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        focal_role: str,
        private_value: int,
        money_cap: int,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 120,
        trace_path: str | None = None,
        seed: int = 0,
    ):
        super().__init__(agent_name)
        self.focal_role = focal_role
        self.private_value = private_value
        self.money_cap = money_cap
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.prompt_entity_initializer = "system"
        self.conversation = []
        self.proposal_count = 0
        self.max_proposals = 4
        opponent_role = "buyer" if focal_role == "seller" else "seller"
        self.belief = ReservationBelief.prior(opponent_role, money_cap)
        self.trace_path = trace_path
        self.trace = []
        self.client = ChatCompletionsHTTPClient(
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == "client":
                setattr(result, key, "OpenAI")
            else:
                setattr(result, key, deepcopy(value, memo))
        return result

    def get_state(self):
        """Return a JSON-safe snapshot for NegotiationArena's legacy logger.

        The platform logs agent state before the first action and after every
        turn. Runtime-only HTTP clients and dataclass belief objects therefore
        cannot be placed in the returned mapping.
        """
        return {
            "class": self.__class__.__name__,
            "agent_name": self.agent_name,
            "focal_role": self.focal_role,
            "private_value": self.private_value,
            "money_cap": self.money_cap,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "run_epoch_time_ms": self.run_epoch_time_ms,
            "conversation": deepcopy(self.conversation),
            "proposal_count": self.proposal_count,
            "max_proposals": self.max_proposals,
            "belief_state": self.belief.to_json(),
            "trace_path": self.trace_path,
            "trace": deepcopy(self.trace),
        }

    def init_agent(self, system_prompt, role):
        self.max_proposals = 4
        self.conversation = [{"role": "system", "content": system_prompt + role}]

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})

    def chat(self):  # Agent abstract method; step() owns structured planning.
        raise RuntimeError("BeliefPlannerAgent.chat is not called directly")

    def step(self, message):
        if message:
            self.update_conversation_tracking("user", message)
        opponent_price = self.belief.update(message or "")
        progress = min((self.proposal_count + 1) / max(self.max_proposals, 1), 1.0)
        candidates = self._candidates(progress)
        last = opponent_price
        accept = last is not None and self._acceptable(last, progress)
        if accept:
            action, price = "ACCEPT", last
        else:
            action, price = "PROPOSAL", candidates[0]["price"]
            self.proposal_count += 1
        public_message = self._generate_message(action, price)
        response = self._render(action, price, public_message)
        row = {
            "turn": len(self.trace) + 1,
            "received_price": opponent_price,
            "belief": self.belief.to_json(),
            "candidates": candidates,
            "chosen": {"action": action, "price": price},
            "realized": response,
        }
        self.trace.append(row)
        self._flush_trace()
        self.update_conversation_tracking("assistant", response)
        return response

    def _candidates(self, progress: float) -> list[dict]:
        mean = self.belief.mean
        if self.focal_role == "seller":
            anchors = [
                self.money_cap * (0.85 - 0.25 * progress),
                mean + (1.0 - progress) * max(mean - self.private_value, 0) * 0.35,
                self.private_value + (1.0 - progress) * 8,
            ]
        else:
            anchors = [
                self.money_cap * (0.15 + 0.20 * progress),
                mean - (1.0 - progress) * max(self.private_value - mean, 0) * 0.35,
                self.private_value - (1.0 - progress) * 8,
            ]
        prices = sorted({int(round(clip_price(x, self.money_cap))) for x in anchors})
        rows = []
        for price in prices:
            own = price - self.private_value if self.focal_role == "seller" else self.private_value - price
            accept = self.belief.acceptance_probability(price, self.focal_role, progress)
            info = (1.0 - self.belief.confidence) * (1.0 - progress) * (1.0 - abs(accept - 0.5) * 2.0)
            score = max(own, -20) * accept + 2.5 * info
            rows.append({
                "price": price,
                "own_surplus": round(own, 3),
                "accept_prob": round(accept, 4),
                "information_gain": round(info, 4),
                "score": round(score, 4),
            })
        rows.sort(key=lambda row: (row["score"], row["own_surplus"]), reverse=True)
        return rows

    def _acceptable(self, price: int, progress: float) -> bool:
        surplus = price - self.private_value if self.focal_role == "seller" else self.private_value - price
        threshold = max(0.0, 8.0 * (1.0 - progress))
        return surplus >= threshold

    def _generate_message(self, action: str, price: int) -> str:
        if action == "ACCEPT":
            return f"I accept your offer of {price} ZUP."
        prompt = (
            f"Write one concise, cooperative negotiation sentence. You are the {self.focal_role}. "
            f"Your locked offer is exactly {price} ZUP. Do not mention private values, another price, "
            "XML, or analysis."
        )
        try:
            text = self.client.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed + len(self.trace),
            )
            text = text.strip().replace("\n", " ")
            if not text or any(char in text for char in "<>"):
                raise ValueError("unsafe generator output")
            return text
        except Exception as exc:
            return f"I propose a price of {price} ZUP. ({type(exc).__name__} fallback)"

    def _render(self, action: str, price: int, public_message: str) -> str:
        trade = "NONE" if action == "ACCEPT" else f"Player RED Gives X: 1 | Player BLUE Gives ZUP: {price}"
        resources = "X: 1" if self.focal_role == "seller" else f"ZUP: {self.money_cap}"
        return (
            f"<proposal count> {self.proposal_count} </proposal count>\n"
            f"<my resources> {resources} </my resources>\n"
            f"<my goals> maximize my own surplus </my goals>\n"
            f"<reason> belief-planner action locked </reason>\n"
            f"<player answer> {action} </player answer>\n"
            f"<newly proposed trade> {trade} </newly proposed trade>\n"
            f"<message> {public_message} </message>"
        )

    def _flush_trace(self):
        if not self.trace_path:
            return
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")


def clip_price(value: float, money_cap: int) -> int:
    return int(max(0, min(money_cap, value)))
