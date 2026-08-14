# Negotiation Belief–Planner Research

This repository studies how language-model negotiation agents can infer an
opponent's hidden preferences, maintain calibrated beliefs across repeated
interactions, and make those beliefs causally usable by an executable planner.

The repository is intentionally organized as a monorepo. At present it
publishes the cleaned NegotiationArena integration only. Other environments,
reports, and editable presentations can be added later without moving the Git
root or importing the whole research workstation.

## Repository layout

```text
negotiation/
├── NegotiationArena/       # Current published environment and framework
├── docs/
│   ├── reports/            # Curated research reports
│   ├── results/            # Aggregate analyses; no raw model trajectories
│   └── presentations/      # Reserved for reviewed editable PPT/PDF assets
├── .gitignore
├── LICENSE
└── README.md
```

Planned future additions can be placed directly at the repository root, for
example `Simple_Env/`, `CaSiNo_Env/`, `LLM_Deliberation/`, and `ANL/`. Keep this
publication repository separate from the broader research workspace, which may
contain models, credentials, raw trajectories, and unrelated work.

## Current scope

The current release contains:

- the original NegotiationArena games and protocol implementation;
- repeated Buyer–Seller and Resource Exchange settings;
- matched Direct and paper-aligned Opponent Simulation baselines;
- the V2–V8 belief-usable planner inheritance chain;
- frozen, wrong, shuffled, oracle, policy, and no-cross interventions;
- parser-safety and framework integration tests;
- research questions, preregistrations, change notes, analysis scripts, and
  curated aggregate results.

Raw API calls, model outputs, episode JSONL, arena logs, checkpoints, local
service logs, `.env` files, and duplicated code snapshots are deliberately not
published.

## Main findings

In the matched 5 runs × 20 episodes matrix, the best framework version for
each setting changes by role:

| Setting | Best framework in matched matrix | Opponent Simulation | Delta |
|---|---:|---:|---:|
| Focal Buyer | V3.3: 13.11 | 12.76 | +0.35 |
| Focal Seller | V3.3: 16.95 | 19.13 | -2.18 |
| Resource-first | V5: 18.45 | 40.57 | -22.12 |
| Resource-second | V3.3: 22.47 | 11.88 | +10.59 |

There is no universal payoff winner. V8 is the final and most auditable method
version, not a claim of universal SOTA: it factorizes preference and response
policy, locks structured actions, gates belief influence by evidence
reliability, and supports same-state causal interventions. Preference belief is
action-relevant in controlled tests, but learned continuous cross-episode
adaptation is not yet consistently better than frozen/no-cross controls.

See [the final Chinese report](docs/reports/NEGOTIATIONARENA_FINAL_REPORT_CN.md)
for the full settings, variants, results, and limitations.

## Quick start

```bash
cd NegotiationArena
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
python -m unittest tests.test_repeated_integration_unittest -v
```

The repeated framework runner uses an OpenAI-compatible chat-completions
endpoint. Configure credentials in your shell; never write real values into a
tracked file:

```bash
export NEGOTIATION_QWEN_API_KEY="your-local-or-provider-key"
export NEGOTIATION_QWEN_BASE_URL="http://127.0.0.1:8002/v1"
export NEGOTIATION_QWEN_MODEL="your-served-model-name"
```

Example smoke run:

```bash
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
  --output-dir arena_runs/smoke/resource_second/framework_v8
```

Generated output is ignored by Git by default.

## Adding another environment later

1. Create a top-level directory such as `Simple_Env/`.
2. Copy only source, tests, lightweight configs, and reviewed documentation.
3. Exclude nested `.git`, `.env*`, raw runs, caches, checkpoints, datasets, and
   model traces.
4. Add an environment-level README with installation and smoke-test commands.
5. Put cross-environment reports in `docs/reports/` and reviewed editable
   presentations in `docs/presentations/`.
6. Run the secret and large-file checks described in [SECURITY.md](SECURITY.md)
   before staging or pushing.

## Upstream and license

The NegotiationArena environment is derived from
[vinid/negotiationarena](https://github.com/vinid/negotiationarena). Our
belief-planner integrations, repeated evaluation harness, interventions, and
research reports are additions. See
[NegotiationArena/ATTRIBUTION.md](NegotiationArena/ATTRIBUTION.md).

Released under the MIT License. See [LICENSE](LICENSE).
