"""RL-style wrapper around AgenticPay for typed buyer actions."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agenticpay import make
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.agents.seller_agent import SellerAgent

from .belief import HeuristicBeliefModel
from .naturalizer import NaturalizerContext, TemplateNaturalizer
from .schemas import ACTION_NAMES, BeliefState, TypedAction


PRICE_PATTERN = re.compile(r"(?:BUYER_PRICE|SELLER_PRICE)\(\$([\d,]+\.?\d*)\)", re.IGNORECASE)


TASK_ENV_IDS = {
    "task1": "Task1_basic_price_negotiation-v0",
    "task2": "Task2_close_price_negotiation-v0",
    "task3": "Task3_close_to_market_price_negotiation-v0",
}


TASK_DEFAULT_QUALITY_THRESHOLDS = {
    "task1": 50.0,
    "task2": 25.0,
    "task3": 25.0,
}


SELLER_CHARACTER_SUFFIXES = {
    "fair": (
        "Seller character: fair. Seek a balanced agreement, make reasonable concessions, "
        "and do not go below your confidential minimum acceptable price."
    ),
    "greedy": (
        "Seller character: greedy. Anchor high, concede slowly, and maximize seller profit "
        "while still allowing a deal only when the buyer reaches an acceptable price."
    ),
    "patient": (
        "Seller character: patient. Prefer a better final price over a quick agreement. "
        "Concede only when the buyer shows clear movement."
    ),
    "impatient": (
        "Seller character: impatient. Prefer closing quickly if there is positive surplus, "
        "while still protecting your confidential minimum acceptable price."
    ),
    "warm": (
        "Seller character: warm. Be cooperative and friendly, make moderate concessions, "
        "and emphasize mutual benefit."
    ),
    "dominant": (
        "Seller character: dominant. Be firm, anchor strongly, concede minimally, "
        "and protect seller profit."
    ),
}


def extract_tagged_price(text: str) -> Optional[float]:
    matches = PRICE_PATTERN.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


class ExternalTypedBuyerAgent(BaseAgent):
    """Placeholder buyer used because AgenticPay expects a buyer agent object."""

    def __init__(self):
        super().__init__(
            model=None,
            role_description="External typed-action buyer controlled by an RL wrapper.",
            name="ExternalTypedBuyer",
        )

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        raise RuntimeError("ExternalTypedBuyerAgent is controlled by TypedAgenticPayEnv.step().")


class ScriptedFixedSellerAgent(BaseAgent):
    """Controlled fixed seller with simple character-dependent concessions."""

    def __init__(self, seller_min_price: float, initial_price: float, character: str):
        super().__init__(
            model=None,
            role_description=f"Scripted fixed seller with character={character}.",
            name=f"ScriptedSeller-{character}",
        )
        self.seller_min_price = seller_min_price
        self.initial_price = initial_price
        self.character = character
        self.last_offer: Optional[float] = None

    def initialize(self, context: Dict[str, Any]):
        super().initialize(context)
        self.last_offer = None

    def respond(self, conversation_history: List[Dict[str, Any]], current_state: Dict[str, Any]) -> str:
        round_index = int(current_state.get("current_round", 0))
        buyer_price = self._last_buyer_price(conversation_history)
        last_offer = self.last_offer if self.last_offer is not None else self.initial_price
        price = self._choose_price(buyer_price=buyer_price, last_offer=last_offer, round_index=round_index)
        self.last_offer = price

        if buyer_price is not None and price <= buyer_price:
            return (
                f"I can accept that for the jacket. "
                f"Let's close at ### SELLER_PRICE(${price:.2f}) ###."
            )
        return (
            f"This is a premium product with strong materials and seasonal value. "
            f"My current offer is ### SELLER_PRICE(${price:.2f}) ###."
        )

    def _choose_price(self, buyer_price: Optional[float], last_offer: float, round_index: int) -> float:
        if buyer_price is not None and buyer_price >= self.seller_min_price:
            accept_threshold = {
                "fair": self.seller_min_price + 12.0,
                "greedy": self.seller_min_price + 35.0,
                "patient": self.seller_min_price + 28.0,
                "impatient": self.seller_min_price + 8.0,
                "warm": self.seller_min_price + 15.0,
                "dominant": self.seller_min_price + 32.0,
            }.get(self.character, self.seller_min_price + 15.0)
            if buyer_price >= accept_threshold:
                return buyer_price

        concession = {
            "fair": 12.0,
            "greedy": 5.0,
            "patient": 4.0 if round_index < 3 else 8.0,
            "impatient": 18.0,
            "warm": 14.0,
            "dominant": 3.0,
        }.get(self.character, 10.0)
        floor_buffer = {
            "fair": 5.0,
            "greedy": 30.0,
            "patient": 22.0,
            "impatient": 3.0,
            "warm": 8.0,
            "dominant": 28.0,
        }.get(self.character, 10.0)

        candidate = max(self.seller_min_price + floor_buffer, last_offer - concession)
        if buyer_price is not None:
            candidate = max(candidate, min(last_offer, buyer_price + floor_buffer))
        return round(candidate, 2)

    @staticmethod
    def _last_buyer_price(conversation_history: List[Dict[str, Any]]) -> Optional[float]:
        for message in reversed(conversation_history):
            if message.get("role") == "buyer":
                return extract_tagged_price(message.get("content", ""))
        return None


@dataclass
class Task1Config:
    """Default AgenticPay Task1 price-negotiation config."""

    task_id: str = "task1"
    buyer_max_price: float = 150.0
    seller_min_price: float = 80.0
    initial_seller_price: float = 150.0
    max_rounds: int = 6
    price_tolerance: float = 0.0
    product_name: str = "Premium Winter Jacket"
    product_list_price: float = 180.0
    user_requirement: str = "I need a high-quality winter jacket for cold weather"
    user_profile: str = (
        "User prefers business/professional style and likes to compare prices before making purchases."
    )

    @classmethod
    def from_task_id(cls, task_id: str, max_rounds: int = 6) -> "Task1Config":
        normalized = task_id.lower().strip()
        if normalized in {"task1", "task1_basic_price_negotiation"}:
            return cls(task_id="task1", max_rounds=max_rounds)
        if normalized in {"task2", "task2_close_price_negotiation"}:
            return cls(
                task_id="task2",
                buyer_max_price=130.0,
                seller_min_price=112.0,
                initial_seller_price=150.0,
                max_rounds=max_rounds,
                product_name="Premium Winter Jacket",
                product_list_price=155.0,
                user_requirement="I need a winter jacket but have a tight budget.",
                user_profile="User is price sensitive and will only buy if the seller gets close to their budget.",
            )
        if normalized in {"task3", "task3_close_to_market_price_negotiation"}:
            return cls(
                task_id="task3",
                buyer_max_price=175.0,
                seller_min_price=155.0,
                initial_seller_price=190.0,
                max_rounds=max_rounds,
                product_name="Market-Priced Winter Jacket",
                product_list_price=180.0,
                user_requirement="I need a winter jacket but know the market price is high.",
                user_profile="User compares market prices carefully and wants a defensible discount.",
            )
        raise ValueError(f"Unknown single-buyer price task: {task_id}")

    @property
    def env_id(self) -> str:
        return TASK_ENV_IDS[self.task_id]

    def env_kwargs(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "initial_seller_price": self.initial_seller_price,
            "buyer_max_price": self.buyer_max_price,
            "seller_min_price": self.seller_min_price,
            "environment_info": {
                "temperature": "warm",
                "season": "summer",
                "weather": "sunny",
            },
            "price_tolerance": self.price_tolerance,
            "reward_weights": {
                "buyer_savings": 1.0,
                "seller_profit": 1.0,
                "time_cost": 0.1,
            },
        }

    def reset_kwargs(self) -> Dict[str, Any]:
        return {
            "user_requirement": self.user_requirement,
            "product_info": {
                "name": self.product_name,
                "brand": "Mountain Gear",
                "price": self.product_list_price,
                "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                "condition": "New",
                "material": "Gore-Tex",
            },
            "user_profile": self.user_profile,
        }


class TypedAgenticPayEnv:
    """Buyer-side RL environment with typed actions and fixed seller."""

    def __init__(
        self,
        seller_character: str = "fair",
        config: Optional[Task1Config] = None,
        reward_mode: str = "terminal_buyer_score",
        include_belief: bool = True,
        seller_backend: str = "scripted",
        seller_model: Optional[Any] = None,
        min_quality_buyer_score: Optional[float] = None,
        quality_bonus: float = 20.0,
        low_quality_deal_penalty: float = 15.0,
        no_deal_penalty: float = 25.0,
    ):
        self.config = config or Task1Config()
        self.seller_character = seller_character
        self.reward_mode = reward_mode
        self.include_belief = include_belief
        self.seller_backend = seller_backend
        self.seller_model = seller_model
        self.min_quality_buyer_score = (
            float(min_quality_buyer_score)
            if min_quality_buyer_score is not None
            else TASK_DEFAULT_QUALITY_THRESHOLDS.get(self.config.task_id, 50.0)
        )
        self.quality_bonus = quality_bonus
        self.low_quality_deal_penalty = low_quality_deal_penalty
        self.no_deal_penalty = no_deal_penalty
        self.buyer_agent = ExternalTypedBuyerAgent()
        self.seller_agent = self._make_seller_agent()
        self.belief_model = HeuristicBeliefModel()
        self.naturalizer = TemplateNaturalizer()
        self.env = None
        self.observation: Optional[Dict[str, Any]] = None
        self.last_belief_state: Optional[BeliefState] = None

    @property
    def num_action_types(self) -> int:
        return len(ACTION_NAMES)

    def action_from_ids(self, action_id: int, price_bin: int, num_price_bins: int = 11) -> TypedAction:
        """Map discrete RL outputs into a typed action.

        `action_id` selects the strategic act. `price_bin` selects a target price
        between seller_min_price and buyer_max_price. The method is intentionally
        simple so bandit/PPO policies can start with a purely discrete action
        space before learning continuous prices.
        """
        if not 0 <= action_id < len(ACTION_NAMES):
            raise ValueError(f"action_id must be in [0, {len(ACTION_NAMES) - 1}], got {action_id}")
        if num_price_bins < 2:
            raise ValueError("num_price_bins must be at least 2")
        clipped_bin = max(0, min(num_price_bins - 1, price_bin))
        fraction = clipped_bin / (num_price_bins - 1)
        target_price = self.config.seller_min_price + fraction * (
            self.config.buyer_max_price - self.config.seller_min_price
        )
        return TypedAction(
            act=ACTION_NAMES[action_id],
            target_price=round(target_price, 2),
            rationale=f"discrete action_id={action_id}, price_bin={price_bin}",
        )

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        self.buyer_agent = ExternalTypedBuyerAgent()
        self.seller_agent = self._make_seller_agent()
        self.env = make(
            self.config.env_id,
            buyer_agent=self.buyer_agent,
            seller_agent=self.seller_agent,
            **self.config.env_kwargs(),
        )
        self.observation, _info = self.env.reset(**self.config.reset_kwargs())
        self.last_belief_state = self._predict_belief(target_price=None)
        return self._build_rl_observation()

    def step(self, action: TypedAction) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.env is None or self.observation is None:
            raise RuntimeError("Call reset() before step().")

        belief_state = self._predict_belief(target_price=action.target_price)
        buyer_message = self.naturalizer.naturalize(
            action,
            NaturalizerContext(
                product_name=self.config.product_name,
                buyer_max_price=self.config.buyer_max_price,
                last_seller_price=self.observation.get("seller_price"),
                last_buyer_price=self.observation.get("buyer_price"),
                round_index=int(self.observation.get("current_round", 0)),
            ),
        )
        seller_history = self.observation["conversation_history"].copy()
        seller_history.append(
            {
                "role": "buyer",
                "content": buyer_message,
                "round": self.observation.get("current_round", 0),
            }
        )
        seller_message = self.seller_agent.respond(seller_history, self.observation)
        next_observation, env_reward, terminated, truncated, info = self.env.step(
            buyer_action=buyer_message,
            seller_action=seller_message,
        )
        reward = self._compute_reward(
            env_reward=env_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action=action,
            next_observation=next_observation,
        )
        self.observation = next_observation
        self.last_belief_state = self._predict_belief(target_price=None)
        rl_observation = self._build_rl_observation()
        info = {
            **info,
            "env_reward": env_reward,
            "buyer_message": buyer_message,
            "seller_message": seller_message,
            "typed_action": action.to_dict(),
            "belief_state": belief_state.to_dict(),
            "seller_character": self.seller_character,
            "seller_backend": self.seller_backend,
            "task_id": self.config.task_id,
            "env_id": self.config.env_id,
            "reward_mode": self.reward_mode,
            "min_quality_buyer_score": self.min_quality_buyer_score,
        }
        return rl_observation, reward, terminated, truncated, info

    def _make_seller_agent(self) -> BaseAgent:
        if self.seller_backend == "scripted":
            return ScriptedFixedSellerAgent(
                seller_min_price=self.config.seller_min_price,
                initial_price=self.config.initial_seller_price,
                character=self.seller_character,
            )
        if self.seller_backend == "agenticpay":
            if self.seller_model is None:
                raise ValueError("seller_backend='agenticpay' requires seller_model.")
            suffix = SELLER_CHARACTER_SUFFIXES.get(
                self.seller_character,
                f"Seller character: {self.seller_character}. Stay professional and protect seller profit.",
            )
            return SellerAgent(
                model=self.seller_model,
                seller_min_price=self.config.seller_min_price,
                system_prompt_suffix=suffix,
            )
        raise ValueError(f"Unknown seller_backend: {self.seller_backend}")

    def _compute_reward(
        self,
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        action: TypedAction,
        next_observation: Dict[str, Any],
    ) -> float:
        if self.reward_mode == "env_reward":
            return float(env_reward)
        if self.reward_mode == "step_buyer_reward":
            return float(info.get("step_buyer_reward", 0.0))
        if self.reward_mode == "terminal_buyer_score":
            if terminated or truncated:
                return float(info.get("buyer_score") or 0.0)
            return 0.0
        if self.reward_mode == "terminal_quality_score":
            if terminated or truncated:
                return self._terminal_quality_reward(info)
            return 0.0
        if self.reward_mode == "shaped_quality_score":
            shaped = self._dense_shaping_reward(action=action, next_observation=next_observation)
            if terminated or truncated:
                shaped += self._terminal_quality_reward(info)
            return shaped
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _terminal_quality_reward(self, info: Dict[str, Any]) -> float:
        buyer_score = float(info.get("buyer_score") or 0.0)
        if info.get("termination_reason") == "agreed":
            if buyer_score >= self.min_quality_buyer_score:
                return buyer_score + self.quality_bonus
            return buyer_score - self.low_quality_deal_penalty
        return buyer_score - self.no_deal_penalty

    def _dense_shaping_reward(self, action: TypedAction, next_observation: Dict[str, Any]) -> float:
        reward = -0.1
        if action.target_price is None:
            return reward
        target = float(action.target_price)
        buyer_max = float(self.config.buyer_max_price)
        seller_min = float(self.config.seller_min_price)
        feasible_span = max(1.0, buyer_max - seller_min)
        if target > buyer_max:
            reward -= 5.0 * ((target - buyer_max) / feasible_span)
        seller_price = next_observation.get("seller_price")
        if isinstance(seller_price, (int, float)):
            gap = float(seller_price) - target
            if gap > 0:
                reward += max(0.0, 1.0 - gap / feasible_span)
            else:
                reward += 1.0
        return reward

    def _predict_belief(self, target_price: Optional[float]) -> BeliefState:
        if self.observation is None:
            return BeliefState(uncertainty=1.0)
        return self.belief_model.predict(
            conversation_history=self.observation.get("conversation_history", []),
            last_seller_price=self.observation.get("seller_price"),
            target_price=target_price,
        )

    def _build_rl_observation(self) -> Dict[str, Any]:
        if self.observation is None:
            raise RuntimeError("No active observation. Call reset() first.")
        current_round = int(self.observation.get("current_round", 0))
        buyer_price = self.observation.get("buyer_price")
        seller_price = self.observation.get("seller_price")
        belief = self.last_belief_state or BeliefState(uncertainty=1.0)
        obs = {
            "round_index": current_round,
            "remaining_rounds": max(0, self.config.max_rounds - current_round),
            "buyer_price": buyer_price,
            "seller_price": seller_price,
            "buyer_price_norm": self._norm_price(buyer_price),
            "seller_price_norm": self._norm_price(seller_price),
            "buyer_max_price_norm": self._norm_price(self.config.buyer_max_price),
            "product_list_price_norm": self._norm_price(self.config.product_list_price),
            "seller_character": self.seller_character,
            "task_id": self.config.task_id,
            "env_id": self.config.env_id,
            "seller_backend": self.seller_backend,
            "conversation_len": len(self.observation.get("conversation_history", [])),
        }
        if self.include_belief:
            obs["belief_state"] = belief.to_dict()
            obs["belief_vector"] = [
                self._norm_price(belief.seller_reservation_mean),
                self._norm_price(belief.seller_reservation_std),
                self._none_to_zero(belief.accept_prob_at_target),
                self._none_to_zero(belief.patience),
                self._none_to_zero(belief.warmth),
                self._none_to_zero(belief.dominance),
                self._none_to_zero(belief.uncertainty),
            ]
        return obs

    @staticmethod
    def _norm_price(price: Optional[float]) -> float:
        if price is None:
            return 0.0
        return float(price) / 200.0

    @staticmethod
    def _none_to_zero(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return float(value)
