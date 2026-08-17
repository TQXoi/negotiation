from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy/validate CaSiNo data for CaSiNo_Env.")
    parser.add_argument("--source-dir", required=True, help="Path to a cloned CaSiNo data directory.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "CaSiNo_Env" / "data" / "casino"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [
        (src / "casino.json", out / "casino.json"),
        (src / "split" / "casino_train.json", out / "casino_train.json"),
        (src / "split" / "casino_valid.json", out / "casino_valid.json"),
        (src / "split" / "casino_test.json", out / "casino_test.json"),
    ]
    for source, target in files:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        print(f"{source} -> {target}")


if __name__ == "__main__":
    main()
