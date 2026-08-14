#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
complete = episodes = errors = 0
for path in args.root.glob("*/*/summary.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    count = int(data.get("episodes", 0))
    episodes += count
    errors += int(data.get("errors", 0))
    complete += int(count == 40)
print(json.dumps({
    "complete_cells": complete,
    "total_cells": 12,
    "episodes": episodes,
    "target_episodes": 480,
    "errors": errors,
}))
