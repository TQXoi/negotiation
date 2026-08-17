#!/usr/bin/env python3
"""Build and audit the Simple Env hidden-utility profile suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simple_Env.buyer.continuous_solver.identifiable import build_profile_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-js-bits", type=float, default=0.12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = build_profile_suite(min_js_bits=args.min_js_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(pair.to_dict(), ensure_ascii=False) for pair in suite) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "profiles": len(suite),
                "min_js_bits": args.min_js_bits,
                "output": str(args.output),
                "pairs": [pair.to_dict() for pair in suite],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
