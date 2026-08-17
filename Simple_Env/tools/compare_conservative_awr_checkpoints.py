#!/usr/bin/env python3
"""Compare two conservative AWR checkpoints on one frozen context file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simple_Env.tools.train_cross_environment_awr_planner import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.contexts.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    old = evaluate_checkpoint(args.old_checkpoint, cases)
    new = evaluate_checkpoint(args.new_checkpoint, cases)
    real = new["by_domain"].get("agenticpay_real_trajectory", {})
    official = new["by_domain"].get("agenticpay_official_template", {})
    aggregate = new["aggregate"]
    audit = manifest.get("audit", {})
    gate = {
        "positive_aggregate_delta": aggregate["mean_oracle_delta_vs_base"] > 0.0,
        "real_trajectory_nonnegative": real.get("mean_oracle_delta_vs_base", -1.0) >= 0.0,
        "official_template_nonnegative": official.get("mean_oracle_delta_vs_base", -1.0) >= 0.0,
        "aggregate_harmful_override_le_0_10": aggregate["harmful_override_rate"] <= 0.10,
        "real_harmful_override_le_0_12": real.get("harmful_override_rate", 1.0) <= 0.12,
        "override_rate_0_02_to_0_60": 0.02 <= aggregate["override_rate"] <= 0.60,
        "not_below_iteration_023_on_same_dev": (
            aggregate["mean_oracle_delta_vs_base"]
            >= old["aggregate"]["mean_oracle_delta_vs_base"]
        ),
        "scenario_split_audit": bool(audit.get("split_audit_passed")),
        "feature_schema_audit": bool(audit.get("feature_schema_audit_passed")),
        "private_truth_feature_audit": not bool(audit.get("private_truth_serialized_as_feature")),
    }
    gate["passed"] = all(gate.values())
    payload = {
        "comparison_schema": "iteration_024_same_dev_checkpoint_comparison_v1",
        "contexts": len(cases),
        "old_checkpoint": str(args.old_checkpoint.resolve()),
        "new_checkpoint": str(args.new_checkpoint.resolve()),
        "old": old,
        "new": new,
        "promotion_gate": gate,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
