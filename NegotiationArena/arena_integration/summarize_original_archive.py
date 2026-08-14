from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="paper worktree buysell_section_one log directory")
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in glob.glob(f"{args.archive}/*/game_state.json"):
        try:
            state = json.load(open(path, encoding="utf-8"))
            settings = state["game_state"][0]["settings"]
            end = state["game_state"][-1]["summary"]
            values = [item["_value"]["X"] for item in settings["player_valuation"]]
            models = [player["model"] for player in state["players"]]
            if values != [40, 60]:
                continue
            grouped[(models[0], models[1])].append(end)
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    output = {}
    for (seller, buyer), rows in sorted(grouped.items()):
        accepted = [row for row in rows if row["final_response"] == "ACCEPT"]
        output[f"{seller}__{buyer}"] = {
            "episodes": len(rows),
            "agreement_rate": len(accepted) / len(rows),
            "mean_buyer_payoff": statistics.mean(row["player_outcome"][1] for row in rows),
            "mean_seller_payoff": statistics.mean(row["player_outcome"][0] for row in rows),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
