# External Negotiation Environments and Comparison Plan

Date: 2026-06-21

This note follows the works mentioned in `6.17/6.17_final.pptx`: RLVR
Negotiation, Trade / Bilateral Trade, ASTRA, and Preference Estimation. The
goal is to make our benchmark more convincing by running or approximating our
framework in outside negotiation environments, then using the trajectories to
diagnose what our AgenticPay-only results miss.

## Code Availability Check

| Work | Public code status | Local status | What it can test for us |
|---|---|---|---|
| BOND / Distilling Bayesian Belief States into Language Models for Auditable Negotiation | Public GitHub code linked from arXiv | Downloaded to `/work5/qixint/external_negotiation_envs/CaSiNo_negotiation-agent` | Belief calibration and belief-action coupling |
| Preference Estimation via Opponent Modeling in Multi-Agent Negotiation | No direct public code link found on arXiv or web search | Not downloaded | We can reproduce the evaluation idea on AgenticPay logs: hidden preference prediction and calibration |
| Training Language Models for Bilateral Trade with Private Information | No direct public code link found on arXiv or web search | Not downloaded | Structured binding offers, IR, surplus share, deal rate, concession dynamics |
| ASTRA negotiation agent | Public GitHub code linked in the paper HTML | Downloaded to `/work5/qixint/external_negotiation_envs/ASTRA` | Turn-level opponent modeling + LP/value optimizer + acceptance probability |
| RLVR Negotiation | The exact PPT-mentioned code/repo was not identifiable by web search | Not downloaded | Training protocol idea: verifiable reward, GRPO/RLVR, reward curriculum |
| FAIR Deal-or-No-Deal / End-to-End Negotiator | Public GitHub code linked from the paper | Downloaded to `/work5/qixint/external_negotiation_envs/end-to-end-negotiator` | Item-allocation validity, agreement, reward, conservation constraints |
| CoCoA / CraigslistBargain | Public GitHub code linked by Stanford NLP | Downloaded to `/work5/qixint/external_negotiation_envs/cocoa` | Price negotiation, decoupled strategy/generation, margin, human-likeness |
| Original CaSiNo | Public GitHub code/data linked from the dataset paper | Downloaded to `/work5/qixint/external_negotiation_envs/CaSiNo-original` | Human-human preference inference, strategy labels, satisfaction/opponent-likeness |

## Downloaded Repository

### BOND / CaSiNo Belief-State Negotiation

Repository:

```text
https://github.com/kaneis1/CaSiNo_negotiation-agent
```

Local path:

```text
/work5/qixint/external_negotiation_envs/CaSiNo_negotiation-agent
```

Setup from the repository README:

```bash
cd /work5/qixint/external_negotiation_envs/CaSiNo_negotiation-agent
conda env create -f environment.yml
conda activate casino
pip install -e .
python -m unittest discover -s tests -v
```

Main evaluation entry point:

```bash
python -m casino_belief.evaluation.turn_eval_run \
  --data data/casino/casino_test.json \
  --output-dir artifacts/results/protocol3/turn_eval_smoke_uniform \
  --max-dialogues 5 \
  --agent uniform \
  --annotations external/casino_original/data/casino_ann.json
```

Relevant agents:

- `bayesian`: explicit Bayesian teacher with posterior and menu scoring.
- `distilled_student`: emits posterior, intent, selected content, and final utterance.
- `structured_cot_live` / `structured_cot_replay`: CoT baselines.
- `uniform`, `hybrid`, `sft`: intermediate baselines.

## Why This Helps Our Framework

AgenticPay currently tests deal, score, mismatch, and utility feasibility, but
it does not directly evaluate whether the belief model is calibrated. BOND gives
us a clean target:

- belief state is explicit and scoreable;
- opponent preference order has ground truth;
- Brier score and posterior trajectories can separate belief error from policy error;
- posterior-prefix interventions reveal whether the policy actually uses the belief.

This maps directly onto our weakness: our belief model outputs ranges and text
evidence, but we do not yet know whether these beliefs are calibrated or causally
used by the planner.

## Comparison Plan

### Track A: AgenticPay as Outcome Benchmark

Keep AgenticPay as the main environment for multi-issue commercial negotiation.
Use it for:

- deal rate;
- timeout rate;
- buyer/seller/global score;
- score-success mismatch;
- buyer/seller IR violations;
- contract feasible rate;
- non-price term failures;
- multi-buyer/multi-seller routing failures.

Baselines:

- repo native / direct prompt;
- CoT prompt;
- warm prompt;
- dominant prompt;
- rule offer generator + naturalizer;
- ASTRA-like baseline;
- A+C;
- B+C;
- A+B+C;
- A+B+C + buyer validator;
- A+B+C + seller validator.

Trajectory diagnostics to keep:

- product coverage: does the message negotiate all requested products?
- binding action consistency: does the price/contract match the language?
- concession curve: offer sequence and seller response sequence;
- seller feasibility: does buyer push below seller feasible floor?
- routing: do buyers explore multiple sellers or collapse to Seller1?

### Track B: BOND / CaSiNo as Belief Benchmark

Goal: evaluate our belief module separately from AgenticPay scores.

Plan:

1. Run BOND's existing `bayesian`, `uniform`, and available replay/student agents on a small smoke subset.
2. Add an adapter that calls our `PromptOpponentBeliefModel` on CaSiNo dialogue turns.
3. Map our belief output into CaSiNo-style preference posterior if possible:
   - opponent issue priority ordering;
   - likely acceptable offer;
   - confidence / entropy.
4. Report:
   - Brier score over opponent priority posterior;
   - top-1 preference-order accuracy;
   - posterior entropy over turns;
   - belief-action consistency.
5. Use failed cases to revise our belief schema:
   - replace loose text evidence with factorized posterior;
   - add calibration metrics;
   - add posterior-prefix intervention tests.

What to learn:

- whether our belief ranges are merely plausible text or actually predictive;
- whether planner decisions change when belief changes;
- whether belief confidence is meaningful.

### Track C: Bilateral Trade as Structured-Action Benchmark

No public code was found, but the paper describes a clean environment:
structured tool calls separate binding offers from natural-language messages.
This is close to what we need.

Minimal reproduction:

- State: buyer private value `b`, seller private cost `s`, item value tier, round.
- Actions:
  - `make_offer(price)`;
  - `accept(price)`;
  - `reject()`;
  - `send_message(text)`;
  - `walk_away()`.
- Metrics:
  - individual rationality;
  - surplus share;
  - allocative efficiency / deal rate;
  - no-gains-from-trade walk-away accuracy;
  - concession rate;
  - temporal patience;
  - price-tier consistency.

How to use our framework:

- A predicts seller cost posterior.
- B chooses structured action and price.
- C naturalizes the message.
- Validator ensures binding action and message agree.

Key lesson expected:

- AgenticPay parser mismatch disappears when binding action is structured.
- We can test whether our planner actually improves surplus/IR rather than just producing persuasive text.

### Track D: ASTRA Official Runner

ASTRA official code is now downloaded locally:

```text
/work5/qixint/external_negotiation_envs/ASTRA
```

Wrapper code:

```text
experiments/external_comparisons/run_astra_official_comparison.py
```

The wrapper calls the official `agent_agent_simulation.py` rather than
reimplementing ASTRA.

ASTRA-inspired components:

1. interpret counterpart behavior;
2. estimate opponent reservation and concession;
3. solve candidate counteroffers with a linear/value optimizer;
4. choose tactic based on acceptance probability and reciprocity.

Paper-aligned setting in the wrapper:

- `n_exp=100` per case.
- GPT-4o-mini for opponent-modeling/general components.
- GPT-4o for STR / ASTRA decision making.
- Partner engines: GPT-4o, Gemini-2.0-Flash, Claude-3.5-Sonnet.
- Negotiation types: integrative, distributive, mixed.
- Partner personalities: base, greedy, fair.
- PAP/SA weights: `w1=0.35`, `w2=0.65`.
- `top_n=5`, fine-grained OSAD enabled.

Metrics:

- Avg. Score-All, agent vs partner.
- Avg. Score-Agreement, agent vs partner.
- Walk-Away (%).
- Win rate and joint score are also parsed from the official output JSON.

Why useful:

- It is the closest non-RL baseline to our main claim: planning under latent belief.

### Track E: RLVR / GRPO Training Track

The exact RLVR Negotiation code from the slide was not identifiable, but both
the Bilateral Trade paper and BOND repository suggest a feasible training path.

Short-term training plan:

1. Keep C frozen or native.
2. Train only B, the planner, first.
3. Reward:
   - `+ deal`;
   - `+ buyer utility`;
   - `+ seller non-negative utility`;
   - `- mismatch`;
   - `- timeout`;
   - `- buyer max violation`;
   - `- non-price contract violation`.
4. Use group-relative rewards over multiple sampled plans for the same context.
5. Add curriculum:
   - Stage 1: format and IR;
   - Stage 2: deal completion;
   - Stage 3: surplus optimization;
   - Stage 4: opponent diversity.

Why this matters:

- The Bilateral Trade paper reports a tension between SFT and RL: SFT can improve
  surplus but reduce deal rate; RL can recover deal rate while eroding surplus.
  Our reward must explicitly separate rational walk-away, IR compliance, and deal
  completion.

### Track F: FAIR Deal-or-No-Deal as Allocation Validity Benchmark

Official repository:

```text
/work5/qixint/external_negotiation_envs/end-to-end-negotiator
```

Comparison code:

```text
experiments/external_comparisons/run_deal_or_no_deal_static_eval.py
```

Paper-aligned setting:

- Input contains three item counts and each side's private values.
- Output is a six-number allocation: three items for self and three for partner.
- A valid agreement must conserve every item count.
- Main metrics:
  - agreement rate;
  - valid allocation rate;
  - average self reward;
  - average partner reward;
  - total reward / joint score.

How to test our framework:

1. Convert each context to a negotiation state.
2. Let A infer the partner's hidden value vector from dialogue.
3. Let B choose a structured allocation proposal.
4. Let C generate language.
5. Validator enforces item conservation before the final structured output.

Why useful:

- It is a clean sanity check for our mismatch problem: the final deal is discrete
  and exactly checkable.
- It tests whether our planner can handle non-price issue allocation, not only
  scalar price bargaining.

### Track G: CoCoA / CraigslistBargain as Price-Negotiation Benchmark

Official repository:

```text
/work5/qixint/external_negotiation_envs/cocoa
```

Wrapper code:

```text
experiments/external_comparisons/run_cocoa_craigslist_official_wrapper.py
```

Paper-aligned setting:

- Buyer and seller negotiate a listed Craigslist item.
- Categories:
  - furniture;
  - housing;
  - car;
  - phone;
  - bike;
  - electronics.
- Buyer target price is sampled from:
  - 0.9 x listing price;
  - 0.7 x listing price;
  - 0.5 x listing price.
- Modular baseline:
  - parser maps utterance to coarse dialogue act;
  - manager predicts strategy/dialogue act;
  - generator realizes the utterance.
- RL rewards:
  - margin / utility;
  - fairness;
  - length.

Metrics:

- success / agreement rate;
- average margin;
- fairness;
- dialogue length;
- human-likeness if third-party ratings are available;
- trajectory-level price tracking.

Why useful:

- This is the most direct external analog to AgenticPay's price negotiation.
- It gives a strong historical baseline for "decoupling strategy and generation",
  which is close to our B+C setting.
- Our novelty is to replace a dialogue-act manager with planning under latent
  belief, then train the planner with verifiable rewards.

Important limitation:

- The official code requires Python 2.7 and PyTorch 0.4, so the current wrapper
  defaults to dry-run command generation. Actual execution should happen in a
  pinned legacy environment.

### Track H: Original CaSiNo as Human Strategy and Preference Benchmark

Official repository:

```text
/work5/qixint/external_negotiation_envs/CaSiNo-original
```

Evaluator:

```text
experiments/external_comparisons/run_casino_original_static_eval.py
```

Paper-aligned data:

- 1030 campsite negotiation dialogues.
- Private issue priority order over Food, Water, and Firewood.
- Outcome points.
- Satisfaction and opponent-likeness surveys.
- Utterance-level strategy labels for the annotated subset.

Metrics:

- deal rate;
- walk-away rate;
- points per participant;
- joint points;
- satisfaction and opponent-likeness on the paper's 1-to-5 coding;
- strategy-label distribution;
- high/low preference overlap.

How to test our framework:

- A predicts opponent issue priority.
- B chooses a package split and strategic action.
- C naturalizes the proposal.
- Validator checks package conservation and deal consistency.

Why useful:

- It evaluates whether our belief representation is meaningful in a domain with
  explicit hidden preferences and human strategy annotations.
- It can support a separate "belief quality" table even before full RL training.

## Immediate Next Steps

1. Run BOND unit tests and a 5-dialogue smoke evaluation after creating the conda env.
2. Write a CaSiNo adapter for our belief model:
   - input: dialogue history and self priority;
   - output: posterior over opponent priority orderings.
3. Use the new Bilateral Trade simulator in `experiments/external_comparisons/`.
4. Run ASTRA official dry-run / smoke through `run_astra_official_comparison.py`.
5. Run static evaluators for FAIR Deal-or-No-Deal and original CaSiNo.
6. Generate the CoCoA/CraigslistBargain dry-run config, then decide whether to
   build a Python 2.7/PyTorch 0.4 legacy environment.
7. Add a report table with five benchmark columns:
   - AgenticPay outcome feasibility;
   - BOND belief calibration;
   - Bilateral Trade structured-action rationality.
   - Deal-or-No-Deal allocation validity;
   - CraigslistBargain price-negotiation margin and human-likeness.

## Suggested Report Framing

Current concern:

> AgenticPay alone may not be persuasive enough because it mixes language
> parsing, contract feasibility, and strategic quality.

Better framing:

> We evaluate negotiation ability across complementary environments: AgenticPay
> for realistic product/contract negotiation, BOND/CaSiNo for auditable belief
> estimation, and a Bilateral-Trade-style structured-action benchmark for
> individual rationality and surplus under private information.
