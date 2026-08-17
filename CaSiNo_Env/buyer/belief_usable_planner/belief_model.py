from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from CaSiNo_Env.environment.casino import ITEMS, complement_allocation, score_allocation


PRIOR_LOGIT = 0.0


@dataclass
class IssueBelief:
    """Posterior over how much P2 values a single issue.

    `logit_high` is an unconstrained score. Softmax over issue logits gives a
    compact posterior that an issue is P2's highest-priority issue.
    """

    logit_high: float = PRIOR_LOGIT
    evidence: List[str] = field(default_factory=list)


@dataclass
class PartnerBeliefState:
    """Compact belief used by the planner.

    issue_beliefs: per-issue posterior logits and evidence.
    flexibility: estimated willingness to accept mutually beneficial trades.
    stubbornness: estimated tendency to reject offers and hold high-value items.
    walkaway_risk: current risk that P2 exits if P1 pushes too hard.
    last_seller_offer_value: estimated P2 value of P2's latest proposed split.
    entropy: uncertainty of the issue-priority posterior.
    """

    issue_beliefs: Dict[str, IssueBelief] = field(default_factory=lambda: {item: IssueBelief() for item in ITEMS})
    flexibility: float = 0.50
    stubbornness: float = 0.35
    walkaway_risk: float = 0.10
    last_seller_offer_value: float | None = None
    entropy: float = 0.0
    evidence: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["issue_priority_posterior"] = self.issue_priority_posterior()
        data["estimated_partner_values"] = self.estimated_partner_values()
        data["entropy"] = round(self.entropy, 4)
        return data

    def issue_priority_posterior(self) -> Dict[str, float]:
        logits = {item: self.issue_beliefs[item].logit_high for item in ITEMS}
        m = max(logits.values())
        exps = {item: math.exp(value - m) for item, value in logits.items()}
        total = sum(exps.values()) or 1.0
        return {item: round(exps[item] / total, 4) for item in ITEMS}

    def estimated_partner_values(self) -> Dict[str, float]:
        posterior = self.issue_priority_posterior()
        # CaSiNo issue values are 3/4/5. Map posterior mass into this range.
        return {item: round(3.2 + 2.4 * float(posterior[item]), 4) for item in ITEMS}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _entropy(probs: Mapping[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-8)) for p in probs.values())


class MathPartnerBeliefModel:
    """Training-free belief updater for CaSiNo-style P2 preference.

    The model deliberately uses transparent math rather than an LLM prompt:
    - language cues shift issue logits,
    - seller offers shift posterior toward issues P2 keeps,
    - rejections increase stubbornness and walkaway risk,
    - concessions increase flexibility and reduce walkaway risk.
    """

    def __init__(self) -> None:
        self.need_words = ("need", "important", "prefer", "want", "must", "priority", "crucial", "like")
        self.weak_words = ("less important", "can give", "you can have", "do not need", "not need", "give up")
        self.self_markers = ("i need", "i want", "i prefer", "i'd like", "i would like", "my ", "for me", "for myself", "we need", "we want", "we prefer", "our ")
        self.other_markers = ("you ", "your ", "for you", "support you", "help you", "you need", "your health")

    def initialize(self) -> PartnerBeliefState:
        state = PartnerBeliefState()
        state.entropy = _entropy(state.issue_priority_posterior())
        return state

    def update(self, scenario: Any, history: List[Dict[str, Any]]) -> PartnerBeliefState:
        state = self.initialize()
        previous_seller_value = None
        for turn in history:
            if turn.get("actor") != "seller":
                continue
            action = turn.get("action", {})
            text = str(action.get("message") or "")
            action_type = str(action.get("type") or "")
            self._update_from_text(state, text)
            allocation = action.get("allocation")
            if action_type == "offer" and allocation:
                partner_values = state.estimated_partner_values()
                seller_value = sum(float(allocation.get(item, 0)) * float(partner_values[item]) for item in ITEMS)
                state.last_seller_offer_value = round(seller_value, 4)
                previous_seller_value = self._update_from_offer(state, allocation, seller_value, previous_seller_value)
            elif action_type == "accept":
                state.flexibility = _clip(state.flexibility + 0.08, 0.0, 1.0)
                state.walkaway_risk = _clip(state.walkaway_risk - 0.05, 0.0, 1.0)
            elif action_type == "walk_away":
                state.walkaway_risk = _clip(state.walkaway_risk + 0.30, 0.0, 1.0)
            elif action_type in {"reject", "ask_preference"}:
                state.stubbornness = _clip(state.stubbornness + 0.04, 0.0, 1.0)
        state.entropy = _entropy(state.issue_priority_posterior())
        return state

    def _update_from_text(self, state: PartnerBeliefState, text: str) -> None:
        low_text = text.lower()
        for item in ITEMS:
            item_low = item.lower()
            if item_low not in low_text:
                continue
            window_score = self._issue_text_score(low_text, item_low)
            count_match = re.search(rf"(\d+)\s+(?:units?\s+of\s+)?{re.escape(item_low)}", low_text)
            if count_match:
                nearby = self._near_item_window(low_text, item_low)
                if self._speaker_is_self(nearby):
                    window_score += 0.10 * int(count_match.group(1))
            if abs(window_score) > 0:
                state.issue_beliefs[item].logit_high += window_score
                ev = f"text:{item}:{window_score:+.2f}:{text[:120]}"
                state.issue_beliefs[item].evidence.append(ev)
                state.evidence.append(ev)

    def _near_item_window(self, low_text: str, item_low: str, radius: int = 70) -> str:
        idx = low_text.find(item_low)
        if idx < 0:
            return low_text[:radius]
        return low_text[max(0, idx - radius) : idx + len(item_low) + radius]

    def _speaker_is_self(self, window: str) -> bool:
        return any(marker in window for marker in self.self_markers)

    def _speaker_is_other(self, window: str) -> bool:
        return any(marker in window for marker in self.other_markers)

    def _issue_text_score(self, low_text: str, item_low: str) -> float:
        score = 0.0
        for match in re.finditer(re.escape(item_low), low_text):
            window = low_text[max(0, match.start() - 75) : match.end() + 75]
            self_ref = self._speaker_is_self(window)
            other_ref = self._speaker_is_other(window)
            item_idx = window.find(item_low)
            left = window[max(0, item_idx - 35) : item_idx] if item_idx >= 0 else window
            right = window[item_idx + len(item_low) : item_idx + len(item_low) + 35] if item_idx >= 0 else window
            local_context = left + " " + right
            if other_ref and not self_ref:
                # Seller is likely talking about P1's need, not P2's preference.
                continue
            if any(phrase in window for phrase in ("support you", "help you", "for you", "you need", "your need", "your health")) and not any(
                phrase in window for phrase in ("for myself", "for me", "my need", "i need", "i prefer", "i want")
            ):
                continue
            local = 0.0
            for word in self.need_words:
                if word in local_context:
                    local += 0.22 if self_ref else 0.08
            for phrase in self.weak_words:
                if phrase in window and item_low in right + left:
                    local -= 0.28 if self_ref else 0.08
            if "keep" in local_context or "take" in local_context:
                local += 0.18 if self_ref else 0.05
            if ("give up all" in left or "give up" in left or "can give" in left) and item_idx >= 0:
                local -= 0.55 if self_ref else 0.10
            if ("keep" in left or "take" in left) and item_idx >= 0:
                local += 0.20 if self_ref else 0.05
            score += local
        return _clip(score, -0.70, 0.90)

    def _update_from_offer(
        self,
        state: PartnerBeliefState,
        seller_allocation: Mapping[str, int],
        seller_value: float,
        previous_seller_value: float | None,
    ) -> float:
        for item in ITEMS:
            kept = int(seller_allocation.get(item, 0))
            # Keeping more of an issue is evidence P2 values it.
            delta = 0.18 * (kept - 1.5)
            state.issue_beliefs[item].logit_high += delta
            ev = f"offer:{item}:seller_keeps_{kept}:delta_{delta:+.2f}"
            state.issue_beliefs[item].evidence.append(ev)
            state.evidence.append(ev)
        if previous_seller_value is not None and seller_value < previous_seller_value:
            state.flexibility = _clip(state.flexibility + 0.08, 0.0, 1.0)
            state.stubbornness = _clip(state.stubbornness - 0.05, 0.0, 1.0)
            state.walkaway_risk = _clip(state.walkaway_risk - 0.04, 0.0, 1.0)
        else:
            state.stubbornness = _clip(state.stubbornness + 0.03, 0.0, 1.0)
            state.walkaway_risk = _clip(state.walkaway_risk + 0.02, 0.0, 1.0)
        return seller_value

    def score_candidate(
        self,
        scenario: Any,
        belief: PartnerBeliefState,
        candidate: Dict[str, Any],
        *,
        round_id: int,
        max_rounds: int,
    ) -> Dict[str, Any]:
        p1_alloc = candidate["allocation"]
        p2_alloc = complement_allocation(p1_alloc, scenario.pool)
        partner_values = belief.estimated_partner_values()
        self_score = score_allocation(p1_alloc, scenario.p1_values)
        opp_score = sum(float(p2_alloc[item]) * float(partner_values[item]) for item in ITEMS)
        self_norm = self_score / max(score_allocation(scenario.pool, scenario.p1_values), 1)
        opp_norm = opp_score / max(sum(float(scenario.pool[item]) * float(partner_values[item]) for item in ITEMS), 1e-6)
        time_ratio = round_id / max(max_rounds, 1)
        unfairness = abs(self_norm - opp_norm)
        accept_logit = (
            -1.10
            + 4.25 * opp_norm
            + 0.70 * belief.flexibility
            + 0.45 * time_ratio
            - 1.35 * unfairness
            - 0.95 * belief.stubbornness
        )
        accept_prob = 1 / (1 + math.exp(-accept_logit))
        walkaway_risk = _clip(belief.walkaway_risk + 0.35 * max(0.0, 0.42 - opp_norm), 0.0, 1.0)
        info_gain = (belief.entropy / math.log(len(ITEMS))) * max(0.0, 0.72 - accept_prob) * (1.0 - time_ratio)
        ev = (
            self_norm * accept_prob
            + 0.12 * opp_norm
            + 0.10 * info_gain
            - 0.28 * walkaway_risk
            - 0.10 * unfairness * time_ratio
        )
        scored = dict(candidate)
        scored.update(
            {
                "partner_gets": p2_alloc,
                "self_score": round(self_score, 4),
                "estimated_partner_score": round(opp_score, 4),
                "self_norm": round(self_norm, 4),
                "estimated_partner_norm": round(opp_norm, 4),
                "accept_prob": round(accept_prob, 4),
                "walkaway_risk": round(walkaway_risk, 4),
                "info_gain": round(info_gain, 4),
                "belief_score": round(ev, 4),
            }
        )
        return scored
