"""Seller variants supported by the modular AgenticPay wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SellerVariantSpec:
    name: str
    source: str
    description: str


SELLER_VARIANTS: Dict[str, SellerVariantSpec] = {
    "native": SellerVariantSpec(
        name="native",
        source="AgenticPay repo native SellerAgent",
        description="Original AgenticPay seller. Best for paper-style compatibility.",
    ),
    "validated": SellerVariantSpec(
        name="validated",
        source="local fixed-seller evaluation wrapper",
        description="Native seller plus seller-side feasibility repair/guardrails.",
    ),
}


def validate_seller_variant(value: str) -> str:
    if value not in SELLER_VARIANTS:
        raise ValueError(f"Unknown AgenticPay seller variant: {value}. Available: {sorted(SELLER_VARIANTS)}")
    return value
