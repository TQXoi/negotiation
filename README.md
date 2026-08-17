# Universal Negotiation Belief–Planner

This repository contains the cleaned, portable code and curated research history for a
language-model negotiation framework with continuous opponent belief, belief-usable
planning, structured action locking, and environment-specific adapters.

The research objective is deliberately cross-environment: modify only the focal buyer,
keep the seller fixed, and improve the environment's native buyer reward—not merely a
belief-prediction auxiliary score.

## Repository layout

```text
framework/                 Environment-independent belief store and planners
Simple_Env/                Bilateral single-item / AmazonHistoryPrice-style evaluation
CaSiNo_Env/                Multi-issue camping negotiation evaluation (formerly ASTRA_Env)
AgenticPay_Env/            AgenticPay single/multi-buyer, seller, and issue adapters
NegotiationArena/          NegotiationArena and Opponent Simulation comparisons
experiments/               Shared model clients and required benchmark runners
scripts/                   Portable upstream bootstrap and safety checks
tests/                     Cross-environment framework tests
docs/timeline/             Chronological Chinese/English research reports
docs/presentations/        Curated editable PPTX files
docs/results/              Reviewed aggregate NegotiationArena analyses
```

Raw trajectories, API responses, model weights, checkpoints, virtual environments,
third-party repository copies, and superseded code snapshots are intentionally excluded.

## Installation

```bash
git clone https://github.com/TQXoi/negotiation.git
cd negotiation
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

AgenticPay requires its upstream repository. The bootstrap script clones it into the
ignored `benchmarks/AgenticPay/` directory. It can also refresh the CaSiNo data from the
official upstream repository.

```bash
bash scripts/bootstrap_upstreams.sh
```

Credentials are read only from environment variables. For a local OpenAI-compatible
server a non-secret placeholder key is normally sufficient:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
```

Never put real keys in source files, JSON configs, notebooks, reports, or shell history.

## Smoke tests

Run code-level tests without model calls:

```bash
pytest -q tests Simple_Env/tests
python -m compileall -q framework Simple_Env CaSiNo_Env AgenticPay_Env experiments
```

Simple Env with a fixed seller and local OpenAI-compatible endpoints:

```bash
python -m Simple_Env.eval \
  --buyer-model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --seller-model 'openai@http://127.0.0.1:8003/v1:MODEL_NAME' \
  --buyer-variants 'cot_prompt,full_framework,universal_framework_v5_9' \
  --scenarios-jsonl Simple_Env/research_splits/universal_v1/development_000_023.jsonl \
  --require-scenarios-jsonl --num-test-instances 2 --rollouts-per-instance 1 \
  --output-dir runs/simple/smoke
```

CaSiNo:

```bash
python -m CaSiNo_Env.eval \
  --buyer-model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --seller-model 'openai@http://127.0.0.1:8003/v1:MODEL_NAME' \
  --buyer-variants 'direct_prompt,full_framework,belief_usable_planner' \
  --num-instances 2 --rollouts-per-instance 1 \
  --output-dir runs/casino/smoke
```

AgenticPay (after bootstrapping upstream code):

```bash
python -m AgenticPay_Env.eval \
  --suite single28 \
  --model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --buyer-model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --seller-model 'openai@http://127.0.0.1:8003/v1:MODEL_NAME' \
  --buyer-variants 'repo_native,cot_prompt,full_framework,universal_framework_v11_safe_improvement_arbitrator' \
  --limit 1 --output-dir runs/agenticpay/smoke
```

Learned AWR/LCB variants require a locally supplied `--planner-checkpoint`; checkpoints
are excluded from Git by design.

## Research status

The strongest confirmed Simple Env framework before the final held-out rerun is V5.9;
the reward-labeled V6.4/AgenticPay V13 variants are experimental safe-residual planners.
The last 40-scenario × 3-rollout validation was interrupted after roughly 24 scenarios,
so its partial summary is not a final claim. AgenticPay remains the harder transfer case:
action/protocol safety improved, but the best universal method had not yet surpassed the
direct/CoT buyer on native BuyerScore at the time of archiving.

For the full method, exact result caveats, and next plan, start with:

- `docs/timeline/2026-08-17_21_framework_migration_summary_cn.md`
- `docs/timeline/2026-08-17_23_current_framework_primary_results_cn.md`
- `docs/timeline/2026-08-17_24_reward_first_pipeline_ledger_cn.md`
- `docs/presentations/2026-08-17_12_reward_first_framework_results_plan_editable_en.pptx`

## Licensing and attribution

Our additions are released under the repository MIT license. NegotiationArena,
AgenticPay, and CaSiNo retain their upstream licenses. The included CaSiNo data is
CC BY 4.0 and is accompanied by `CaSiNo_Env/DATA_LICENSE_CC_BY_4_0.txt`. See
`THIRD_PARTY.md` for source links and attribution.
