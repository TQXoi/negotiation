# NegotiationArena belief-planner integration

This directory contains a cleaned, reproducible subset of the local
NegotiationArena research workspace. Raw trajectories, API call logs, model
outputs, duplicated version snapshots, and local service logs are excluded.

## Read first

- Framework and result summary:
  [`../docs/reports/NEGOTIATIONARENA_FINAL_REPORT_CN.md`](../docs/reports/NEGOTIATIONARENA_FINAL_REPORT_CN.md)
- Original project introduction: [`README_UPSTREAM.md`](README_UPSTREAM.md)
- Upstream attribution: [`ATTRIBUTION.md`](ATTRIBUTION.md)

## Code map

| Component | Location |
|---|---|
| Unified repeated runner | `arena_integration/run_opponent_simulation_setting.py` |
| Direct/repeated baseline | `arena_integration/repeated_agents.py` |
| Paper-aligned Opponent Simulation | `arena_integration/paper_aligned_opponent_simulation.py` |
| Belief-planner V2–V8 | `arena_integration/decision_calibrated_agent.py` |
| Repeated games/evaluator | `arena_integration/repeated_games.py` |
| Original games | `games/` |
| Protocol package | `negotiationarena/` |
| Tests | `tests/` |
| Version research notes | `research_iterations/` |

The V2–V8 classes form an inheritance chain in
`decision_calibrated_agent.py`. Chinese reading comments identify each version's
delta. `wrong`, `shuffled`, `oracle`, `frozen`, and `no_cross_episode` are
belief modes passed to the same planner classes, not separate prompt baselines.

## Installation and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
python -m unittest tests.test_repeated_integration_unittest -v
```

## Run

```bash
export NEGOTIATION_QWEN_API_KEY="your-key"
export NEGOTIATION_QWEN_BASE_URL="http://127.0.0.1:8002/v1"
export NEGOTIATION_QWEN_MODEL="your-served-model-name"

python -m arena_integration.run_opponent_simulation_setting \
  --method framework_v8 \
  --setting resource_second \
  --runs 1 \
  --episodes-per-run 2 \
  --max-turns 10 \
  --candidate-count 5 \
  --model "$NEGOTIATION_QWEN_MODEL" \
  --base-url "$NEGOTIATION_QWEN_BASE_URL" \
  --api-key-env NEGOTIATION_QWEN_API_KEY \
  --protocol-mode normalize_retry \
  --output-dir arena_runs/smoke
```

The ignored `arena_runs/` directory will contain append-only episodes, model
traces, game states, and summaries.
