# AgenticPay Experiment Summary for 6.17 Report

This file summarizes all currently available local experiments, their settings, result locations, and usable conclusions. Current date: 2026-06-17 UTC.

## Component Naming

- `prompt`: repo-native/direct prompt buyer.

- `A`: belief model, estimating latent seller information and negotiation posture.

- `B`: high-level strategic planner, selecting typed strategic actions and guardrails.

- `C`: low-level language generator/naturalizer, producing AgenticPay-compliant messages.

- `A+C`: `belief_prompt`; `B+C`: `planner_generator`; `A+B+C`: `full_framework`.

- Buyer validator repairs buyer-side price and non-price contract violations. Seller validator repairs seller-side infeasible responses.

## Part I: Game-theoretic and Multi-agent Interaction Insights

The goal is not to replace the LLM with a pure equilibrium solver. The useful framing is: negotiation is planning under uncertainty with hidden opponent preferences, private reservation values, and language-mediated observations.

| Idea | Literature | How it can improve our framework |
|---|---|---|
| Bayesian belief state instead of loose intervals | Preference Estimation via Opponent Modeling; BOND | Upgrade seller reservation/flexibility/patience/dominance estimates into a calibrated posterior with Brier/ECE diagnostics. |
| Latent strategic action space before solving | LSPO; Deep CFR | Keep natural language out of the solver. Cluster or learn strategic acts, then run best-response/CFR-style updates over compact actions. |
| Turn-level best response / Stackelberg-style planning | ASTRA; Stackelberg planning | Given a belief over seller response, optimize a candidate offer/contract under buyer IR, seller-feasibility, and acceptance probability. |
| Separate binding action from language | Bilateral Trade with Private Information; Dialogue Action Tokens | Make price/contract a structured action and let language naturalize it, reducing parser mismatch and invalid contracts. |
| Train planner with verifiable rewards | RLVR negotiation; Bilateral Trade | Use buyer score, deal, mismatch, IR violation, timeout, and feasibility as reward terms for planner training. |
| External critic / validator as part of the game loop | Automated Mediator pipeline; AgenticPay scoring | Treat validator as a strategic safety layer, not a formatting patch: it checks action-language consistency and both-side feasibility. |

Design implication: the next framework should be `posterior belief -> latent strategic act/value planner -> constrained structured action -> naturalizer -> critic`, with component-wise RL later.

## Main Single28 Runs

| Experiment | Variant | N | Seller | Deal | Timeout | Mismatch | Contract feasible | Contract IR | Global | Buyer | Seller | Result path | Conclusion |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `repo_native` | prompt | 28 | native | 67.9% | 32.1% | 17.9% | 0.0% | 100.0% | 23.02 | 24.63 | 24.07 | `runs/single28_qwen14_native_v2_resume/20260610_122733/task_results.jsonl` | Original AgenticPay baseline; lower deal rate and many timeouts, but mismatch rate is not the only problem because contract scoring is weak. |
| `belief_prompt_v4_partial` | A+C | 13 | native | 84.6% | 15.4% | 15.4% | 60.0% | 25.0% | 37.69 | 44.03 | 35.03 | `runs/single28_framework_belief_prompt_v4/20260615_230130/task_results.jsonl` | Partial run only. Early signal is strong on score, but 13/28 is not enough for a final claim. |
| `planner_generator_validator_v2` | B+C + buyer validator | 28 | native | 89.3% | 10.7% | 39.3% | 40.0% | 55.6% | 21.36 | 27.54 | 20.63 | `runs/single28_framework_planner_generator_validator_v2/20260616_020259/task_results.jsonl` | High deal rate and shorter rounds, but seller-side negative utility causes many mismatches. |
| `full_framework_validator_v2` | A+B+C + buyer validator | 28 | native | 89.3% | 10.7% | 28.6% | 45.0% | 47.1% | 22.71 | 28.99 | 28.15 | `runs/single28_framework_full_framework_validator_v2/20260616_020230/task_results.jsonl` | Improves over B+C on mismatch and seller score; still has seller-side infeasible contracts. |
| `planner_generator_buyer_guardrail_v3` | B+C + stronger buyer guardrail | 28 | native | 92.9% | 7.1% | 42.9% | 20.0% | 77.8% | 19.50 | 18.08 | 27.54 | `runs/single28_planner_generator_buyer_guardrail_v3/20260616_202040/task_results.jsonl` | Worse mismatch; planning without belief tends to over-push seller-infeasible terms. |
| `full_framework_buyer_guardrail_v3` | A+B+C + stronger buyer guardrail | 28 | native | 82.1% | 17.9% | 32.1% | 35.0% | 56.2% | 21.06 | 22.77 | 25.67 | `runs/single28_full_framework_buyer_guardrail_v3/20260616_202028/task_results.jsonl` | Complete 28-task run, but buyer-only guardrail does not solve seller utility violations. |
| `full_framework_buyer_guardrail_validated_seller_v1_partial` | A+B+C + buyer guardrail + seller validator | 23 | validated | 65.2% | 34.8% | 0.0% | 53.3% | 0.0% | 29.38 | 31.77 | 30.85 | `runs/single28_full_framework_buyer_guardrail_validated_seller_v1_resume2/20260617_123219/task_results.jsonl` | Latest partial result is 23/28. Mismatch and IR violations remain zero, but timeout rate is still high. |

## Error Pattern Diagnostics

### `full_framework_validator_v2`
- Mismatch tasks: 8.
- Buyer-negative tasks: 1: `[['Task25_s22_rent_house_2.py', -59.7]]`
- Seller-negative tasks: 7: `[['Task4_s1_beauty_product_negotiation.py', -1.0400000000000003], ['Task5_s2_toothpaste_negotiation.py', -0.25000000000000105], ['Task12_s9_beverage_negotiation.py', -1.1580000000000013], ['Task14_s11_taxi_1.py', -5.890000000000001], ['Task15_s12_taxi_2.py', -5.450000000000001], ['Task17_s14_taxi_4.py', -4.630000000000001], ['Task18_s15_taxi_5.py', -19.779999999999998]]`
- Timeout tasks: 3: `['Task7_s4_headphones_negotiation.py', 'Task16_s13_taxi_3.py', 'Task28_s25_rent_house_5.py']`
### `full_framework_buyer_guardrail_v3`
- Mismatch tasks: 9.
- Buyer-negative tasks: 1: `[['Task28_s25_rent_house_5.py', -59.7]]`
- Seller-negative tasks: 8: `[['Task4_s1_beauty_product_negotiation.py', -2.08], ['Task11_s8_jeans_negotiation.py', -2.595], ['Task12_s9_beverage_negotiation.py', -1.5980000000000012], ['Task14_s11_taxi_1.py', -43.75], ['Task15_s12_taxi_2.py', -43.45], ['Task16_s13_taxi_3.py', -13.679999999999994], ['Task17_s14_taxi_4.py', -4.630000000000001], ['Task18_s15_taxi_5.py', -0.2799999999999993]]`
- Timeout tasks: 5: `['Task3_close_to_market_price_negotiation.py', 'Task5_s2_toothpaste_negotiation.py', 'Task13_s10_food_color_negotiation.py', 'Task24_s21_rent_house_1.py', 'Task27_s24_rent_house_4.py']`
### `full_framework_buyer_guardrail_validated_seller_v1_partial`
- Mismatch tasks: 0.
- Buyer-negative tasks: 0: `[]`
- Seller-negative tasks: 0: `[]`
- Timeout tasks: 8: `['Task3_close_to_market_price_negotiation.py', 'Task4_s1_beauty_product_negotiation.py', 'Task5_s2_toothpaste_negotiation.py', 'Task12_s9_beverage_negotiation.py', 'Task14_s11_taxi_1.py', 'Task15_s12_taxi_2.py', 'Task17_s14_taxi_4.py', 'Task18_s15_taxi_5.py']`
### `planner_generator_buyer_guardrail_v3`
- Mismatch tasks: 12.
- Buyer-negative tasks: 2: `[['Task25_s22_rent_house_2.py', -59.7], ['Task27_s24_rent_house_4.py', -59.7]]`
- Seller-negative tasks: 12: `[['Task4_s1_beauty_product_negotiation.py', -0.5499999999999999], ['Task5_s2_toothpaste_negotiation.py', -1.3], ['Task6_s3_riflescope_negotiation.py', -0.3800000000000007], ['Task7_s4_headphones_negotiation.py', -1.1], ['Task8_s5_wall_lantern_negotiation.py', -1.129999999999999], ['Task10_s7_sandals_negotiation.py', -0.5599999999999994], ['Task11_s8_jeans_negotiation.py', -2.255], ['Task13_s10_food_color_negotiation.py', -0.19499999999999967], ['Task14_s11_taxi_1.py', -47.25], ['Task15_s12_taxi_2.py', -5.950000000000001], ['Task17_s14_taxi_4.py', -5.029999999999999], ['Task18_s15_taxi_5.py', -4.029999999999999]]`
- Timeout tasks: 2: `['Task9_s6_bookshelf_negotiation.py', 'Task24_s21_rent_house_1.py']`

## Hard Task Probe

- Result path: `runs/agenticpay_hard_multi_buyer_seller_product_probe4_qwen14/20260616_042050/task_results.jsonl`
- Setting: `multi_buyer_multi_products_multi_seller`, start-index 102, limit 4, repo/native buyer-seller prompt baseline.
- Result: N=4, deal=100.0%, mismatch=-, global=44.28, buyer=30.38, seller=54.42.
- Task2 is the warning case: success=True but Global/Buyer/Seller scores are negative, showing that agreement alone is not reliable.
- Observed trajectory problems: repeated template-like offers, seller-side `MAKE_DEAL` before buyer-side exact agreement, weak routing discipline across multiple buyers/sellers, and final price/contract mismatch risk.
- Latest hard full-framework validated-seller run path: `runs/agenticpay_hard_multi_full_framework_probe4_qwen14_validated_seller/20260617_105639/task_results.jsonl`
- Latest hard full-framework result: N=4, deal=100.0%, timeout=0.0%, global=88.72, buyer=64.62, seller=54.17.
- Compared with hard native probe: global 44.28 -> 88.72, buyer 30.38 -> 64.62, seller 54.42 -> 54.17.
- Mixed buyer probe path: `runs/multi_buyer_seller_product_mixed_buyers_v1/20260617_122911/task_results.jsonl`
- Mixed setting: buyer roster `full_framework,direct_prompt`, seller `validated`; with `repeat-last`, Task2/Task4 use Buyer1=full_framework and Buyer2/3=direct_prompt.
- Trajectory result: N=4, deal=100.0%, timeout=0.0%, avg rounds=1.50. The useful signal is not the score but the conversation pattern.
- Task1: full framework offers `$160` for both jacket+shoes; direct prompt offers `$100` for the jacket, dropping part of the requested bundle and forcing seller repair/rejection.
- Task2: full framework initially plans `$160` but validator rewrites to `$150` because `reservation_guardrail` is treated as a hard upper bound; direct prompt offers `$160/$170` and Buyer3 wins.
- Task3: full framework tracks seller `$165 -> $162` and keeps a feasible `$160` offer until deal; direct prompt goes `$100 -> $130 -> $140`, still below seller min `$150`.
- Task4: both variants cover all three products, but every buyer selects Seller1, so seller competition is not explored.
- Selected buyer variants: `{'full_framework': 2, 'direct_prompt': 2}`. Direct-prompt wins occur in the 3-buyer tasks where Buyer3 has the highest buyer max, so this is a diagnostic run rather than a clean controlled baseline comparison.
- Mixed-agent insight: full framework is better at maintaining bundle scope and seller-feasible overlap, while direct prompt is more aggressive but unstable. The framework also has a real bug: planner/validator misuse of `reservation_guardrail` can override reasonable planned concessions.
- Earlier full-framework hard run at `runs/agenticpay_hard_multi_full_framework_probe4_qwen14_validated_seller/20260617_070626` produced 0 records; do not use it as evidence.

## Concrete Hard Trajectory Case Study: Full Framework Task4

- Run: `runs/agenticpay_hard_multi_full_framework_probe4_qwen14_validated_seller/20260617_105639/logs/003_multi_buyer_multi_products_multi_seller__Task4_sequential_three_buyer_three_seller_three_product_negotiation.log`
- Setting: sequential three buyers, three sellers, three products. Buyer max prices: 260 / 270 / 280. Seller min prices: 220 / 230 / 240.
- Framework belief/planner behavior: all three buyer agents infer seller feasible floor around 240 and likely acceptable range around 250-280; all choose `anchor_low` with target price 250.
- Naturalized buyer messages: all three buyers produce nearly identical language and offer `BUYER_PRICE($250)` for the entire set.
- Seller response path: seller1 responds to each buyer with `SELLER_PRICE($230)`, which is seller-feasible and below every buyer max.
- Environment selected path: buyer3 + seller1, final deal price 230, total rounds 1, GlobalScore 63.80, BuyerScore 85.80, SellerScore 33.00.
- What the trajectory reveals: the framework finds a feasible deal quickly, but the multi-agent search collapses to seller1; buyers do not diversify routing or exploit seller competition. Language is template-like across agents. Seller score is much lower than buyer score, so the deal is feasible but buyer-skewed.

## Baselines to Add for Next Report

| Baseline | Status | Why it matters |
|---|---|---|
| Direct Prompt | implemented, not yet run | controls for native LLM prompting without modular belief/planning. |
| CoT Prompt | implemented, not yet run | tests whether private self-check alone explains gains. |
| Warm / Dominant Prompt | implemented, not yet run | tests prompt-tactic effects observed in negotiation literature. |
| Rule Offer Generator + Naturalizer | implemented, not yet run | controls for deterministic concession scheduling plus native language generation. |
| ASTRA-like Baseline | implemented, not yet run | opponent modeling + heuristic offer optimization; closest non-RL architecture baseline. |
| Validated Seller | partially run | addresses seller-side infeasible agreements, currently the largest mismatch source. |
| Future RL/RLVR Baseline | not implemented | needed to test whether learned policies outperform prompt scaffolds under verifiable rewards. |

## Current Conclusions

1. The full framework is directionally better than planner-only (`B+C`) when both use buyer-side validation: mismatch drops from 39.3% to 28.6%, seller score improves from 20.63 to 28.15, and deal/timeout remain similar.
2. Buyer-only validation is insufficient. The dominant residual failure is seller-side negative utility caused by non-price contract terms and too-aggressive prices.
3. Adding seller validation is the most promising fix: the latest partial run has 0 mismatch and 0 IR violations, but timeout rises.
4. Hard multi-agent tasks reveal routing and commitment problems: the LLM can negotiate with one seller while other agents receive time penalties, and `MAKE_DEAL` can appear before exact bilateral agreement.
5. Next experiments should compare prompt baselines, rule/ASTRA-like baselines, and the full framework under the same validated-seller condition.
