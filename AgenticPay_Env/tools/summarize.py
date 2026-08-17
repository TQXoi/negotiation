#!/usr/bin/env python3
"""Print a compact AgenticPay_Env summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    "tasks",
    "deal_rate",
    "avg_buyer_score_fixed_seller",
    "avg_buyer_score_no_seller_error",
    "avg_buyer_score_deals_only",
    "avg_buyer_score_original",
    "seller_error_attribution_rate",
    "contract_ir_violation_rate",
    "avg_global_score",
    "avg_seller_score",
    "avg_rounds",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary_path = Path(args.run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("by_buyer_variant", {})
    print("\t".join(["variant", *METRICS]))
    for variant, metrics in sorted(rows.items()):
        values = [variant]
        for metric in METRICS:
            value = metrics.get(metric)
            if isinstance(value, float):
                value = round(value, 4)
            values.append("" if value is None else str(value))
        print("\t".join(values))


if __name__ == "__main__":
    main()
