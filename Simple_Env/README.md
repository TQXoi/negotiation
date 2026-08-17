# Simple Env / AmazonHistoryPrice-style bilateral negotiation

This package implements single-buyer, single-seller, single-item bargaining with private
buyer budget and seller cost. Its native buyer reward is approximately
`(budget - price) / abs(budget - cost)`, with zero for no deal and `-1` for an invalid
buyer overshoot.

`buyer/universal_framework.py` is the thin adapter to the shared `framework/` belief
store and candidate planner. `research_splits/universal_v1/` contains reviewed,
lightweight train/dev/validation scenario manifests; raw trajectories and checkpoints
are excluded.

Run `python -m Simple_Env.eval --help`. Use a fixed `--seller-model` and compare buyer
variants on exactly the same scenario/rollout keys.
