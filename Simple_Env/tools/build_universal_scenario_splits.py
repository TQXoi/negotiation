#!/usr/bin/env python3
"""Materialize pre-registered development/validation/final scenario files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 128:
        raise SystemExit(f"Expected the frozen 128-scenario suite, got {len(rows)}")
    splits = {
        "development_000_023": rows[:24],
        "development_wide_stress": [
            row for row in rows[:24]
            if float(row["seller_cost"]) / float(row["buyer_budget"]) < 0.40
        ],
        "validation_024_063": rows[24:64],
        "final_category_shift_064_127": rows[64:128],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.input.resolve()), "splits": {}}
    for name, values in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        manifest["splits"][name] = {
            "path": str(path.resolve()),
            "count": len(values),
            "item_ids": [row["item_id"] for row in values],
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
