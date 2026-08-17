from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CaSiNo_Env.environment.io import read_jsonl, write_json
from CaSiNo_Env.environment.metrics import summarize_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CaSiNo_Env episode jsonl files.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = {}
    for path in sorted(run_dir.glob("*_episodes.jsonl")):
        variant = path.name.replace("_episodes.jsonl", "")
        summary[variant] = summarize_episodes(list(read_jsonl(path)))
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
