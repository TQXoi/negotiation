from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field


PRICE_RE = re.compile(
    r"Player\s+BLUE\s+Gives\s+ZUP\s*:\s*(\d+)", re.IGNORECASE
)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ReservationBelief:
    """Interpretable interval belief over the opponent's reservation value.

    For a seller focal agent, the opponent is a buyer and the latent value is
    willingness-to-pay. For a buyer focal agent it is the seller's cost.
    Bounds only use revealed offers; they never read the opponent's private goal.
    """

    opponent_role: str
    lower: float
    upper: float
    observations: int = 0
    offers: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @classmethod
    def prior(cls, opponent_role: str, money_cap: int) -> "ReservationBelief":
        return cls(opponent_role=opponent_role, lower=0.0, upper=float(money_cap))

    def update(self, public_message: str) -> int | None:
        match = PRICE_RE.search(public_message or "")
        if not match:
            return None
        price = int(match.group(1))
        self.observations += 1
        self.offers.append(price)
        if self.opponent_role == "buyer":
            # A buyer offering p is direct evidence WTP >= p.
            self.lower = max(self.lower, float(price))
            self.evidence.append(f"buyer_offer:{price}->wtp_lower")
        else:
            # A seller offering p is direct evidence cost <= p.
            self.upper = min(self.upper, float(price))
            self.evidence.append(f"seller_offer:{price}->cost_upper")
        return price

    @property
    def mean(self) -> float:
        # Recent offers carry useful concession information without pretending
        # they are exact private values.
        midpoint = (self.lower + self.upper) / 2.0
        if not self.offers:
            return midpoint
        recent = sum(self.offers[-3:]) / len(self.offers[-3:])
        if self.opponent_role == "buyer":
            return clip(0.65 * midpoint + 0.35 * recent, self.lower, self.upper)
        return clip(0.65 * midpoint + 0.35 * recent, self.lower, self.upper)

    @property
    def confidence(self) -> float:
        return 1.0 - math.exp(-self.observations / 2.5)

    def acceptance_probability(self, price: int, focal_role: str, progress: float) -> float:
        scale = max((self.upper - self.lower) / 5.0, 3.0)
        if focal_role == "seller":
            logit = (self.mean - price) / scale
        else:
            logit = (price - self.mean) / scale
        logit += 0.6 * progress + 0.25 * self.confidence
        return 1.0 / (1.0 + math.exp(-clip(logit, -20.0, 20.0)))

    def to_json(self) -> dict:
        data = asdict(self)
        data.update(mean=round(self.mean, 3), confidence=round(self.confidence, 4))
        return data
