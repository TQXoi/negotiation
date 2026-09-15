"""Opponent belief estimators for negotiation experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schemas import BeliefState


class HeuristicBeliefModel:
    """A lightweight baseline belief model derived from observed seller offers."""

    def predict(
        self,
        conversation_history: List[Dict[str, Any]],
        last_seller_price: Optional[float],
        target_price: Optional[float] = None,
    ) -> BeliefState:
        seller_prices = self._extract_seller_prices(conversation_history)
        if last_seller_price is not None:
            seller_prices.append(float(last_seller_price))

        if seller_prices:
            reservation_mean = min(seller_prices) * 0.9
            reservation_std = max(5.0, abs(seller_prices[0] - seller_prices[-1]) * 0.5)
            patience = self._estimate_patience(seller_prices)
        else:
            reservation_mean = None
            reservation_std = None
            patience = 0.5

        accept_prob = None
        if target_price is not None and reservation_mean is not None:
            margin = target_price - reservation_mean
            accept_prob = max(0.0, min(1.0, 0.5 + margin / 80.0))

        return BeliefState(
            seller_reservation_mean=reservation_mean,
            seller_reservation_std=reservation_std,
            accept_prob_at_target=accept_prob,
            patience=patience,
            warmth=None,
            dominance=None,
            uncertainty=0.5 if seller_prices else 0.9,
        )

    @staticmethod
    def _estimate_patience(seller_prices: List[float]) -> float:
        if len(seller_prices) < 2:
            return 0.5
        total_drop = max(0.0, seller_prices[0] - seller_prices[-1])
        return max(0.0, min(1.0, 1.0 - total_drop / max(1.0, seller_prices[0])))

    @staticmethod
    def _extract_seller_prices(conversation_history: List[Dict[str, Any]]) -> List[float]:
        import re

        prices: List[float] = []
        for message in conversation_history:
            if message.get("role") != "seller":
                continue
            content = message.get("content", "")
            matches = re.findall(r"SELLER_PRICE\(\$([\d,]+\.?\d*)\)", content)
            for match in matches:
                try:
                    prices.append(float(match.replace(",", "")))
                except ValueError:
                    continue
        return prices
