# Negotiation Codebase 总结报告

**报告日期：** 2026-07-27
**代码库根目录：** `/work5/qixint`
**报告范围：** negotiation 相关代码、外部环境、实验脚本、结果目录、阶段性 PPT / MD 材料
**配套阶段报告：** `/work5/qixint/7.27/NEGOTIATION_CONTINUOUS_SOLVER_STAGE_REPORT_20260727.md`

> 本报告是代码库级别的总览台账：它总结目前已经尝试过的环境、实现过的方法与 baseline、已有实验结果、可复用脚本、PPT/README 资产，以及下一步迁移到更 universal benchmark 的计划。

---

## 1. Executive Summary

当前代码库已经从最初的 AgenticPay prompt / framework 实验，扩展为一个较完整的 LLM negotiation research sandbox。核心资产包括：

1. **Simple Env / RLVR Negotiation 复现与重构。**
   已将 RLVR Negotiation paper 的 buyer-seller 单商品砍价环境重构为 `Simple_Env/`，支持 paper-aligned evaluation setting、trajectory 保存、buyer/seller/environment 模块化、trajectory reader、SFT 数据构建、LoRA 训练、CEM planner parameter search。

2. **Terms Env 统一 benchmark 原型。**
   已实现 `Terms_Env/`，用于在同一 multi-issue terms negotiation 环境下比较 direct prompt、CoT、belief prompt、full framework、typed belief、counterfactual belief、ASTRA-style belief、BOND-style posterior、preference estimation 等 baseline。

3. **AgenticPay multi-agent / multi-task 实验。**
   已在 AgenticPay single28 fixed-seller setting 下跑过 Qwen14B 4GPU 对比，并修正了 “seller error 也扣 buyer 分” 的 metric 问题，增加 `fixed_seller`、`no_seller_error`、`deals_only` buyer score。

4. **ASTRA / CasiNo 复现尝试。**
   已下载并接入 ASTRA 代码，增加 our framework / rich belief buyer，对 Table 1 partner setting 做了适配和 batch resume；但 ASTRA 原代码存在 LLM parse / zero-division 等不稳定点，目前结果只能作为探索性结果，尚不应作为最终证据。

5. **Continuous resolving belief + CEM planner RL。**
   在 `Simple_Env/buyer/continuous_solver/` 中实现了 persistent belief state、rule / LLM / hybrid belief update、EV planner、parametric planner 和 generator。CEM 已在 16-scenario smoke/refine set 上找到 promising planner 参数。

总体结论：

> 代码库已经具备“环境复现、baseline 横向比较、trajectory 诊断、prompt/framework 迭代、SFT 数据收集、低成本 RL 参数搜索”的完整闭环。但当前最强结果主要来自 Simple Env，存在环境特化风险；下一步应迁移到更 universal 的 multi-agent benchmark，并用更大规模 held-out evaluation 验证 continuous resolving / CEM / SFT 是否真正泛化。

---

## 2. 顶层目录地图

| 目录 | 内容 | 当前状态 |
|---|---|---|
| `Simple_Env/` | RLVR Negotiation paper-aligned simple bargaining env；buyer/seller/environment/tools/scripts 模块化 | 核心维护中 |
| `Terms_Env/` | multi-issue terms benchmark prototype，统一比较多种 belief / planner baseline | 已实现 all-baseline run |
| `experiments/` | 早期 AgenticPay、外部环境 wrapper、monolithic RLVR / ASTRA / unified terms scripts | 可复用但部分已被新模块取代 |
| `external_negotiation_envs/` | ASTRA、AmazonPriceHistory、CaSiNo、cocoa、end-to-end negotiator 等外部 repo | 已下载/部分接入 |
| `simple_env_runs/` | 新 Simple Env 结果目录 | 主结果目录 |
| `runs/` | 早期与外部实验结果目录 | 很多历史结果，需谨慎区分 |
| `6.26/` | 外部 benchmark 调研、早期实验 PPT、belief novelty idea | 历史调研资产 |
| `7.1/` | unified benchmark 计划、RLVR prompt fix、terms/RLVR 实验脚本 | 中期资产 |
| `7.17/` | continuous resolving / CEM 阶段 PPT 与 README | 当前主要汇报资产 |
| `7.27/` | 当前阶段报告与代码库总览报告 | 当前报告目录 |

---

## 3. 已尝试与已实现的环境

### 3.1 Simple Env: RLVR Negotiation / AmazonHistoryPrice

**位置：**

- 环境代码：`/work5/qixint/Simple_Env`
- 原 monolithic 脚本：`/work5/qixint/experiments/external_comparisons/run_rlvr_negotiation_paper_eval.py`
- 数据：`/work5/qixint/data/rlvr_negotiation/amazonhistoryprice_test128_first_budget0.8.jsonl`
- 外部数据 repo：`/work5/qixint/external_negotiation_envs/AmazonPriceHistory`

**环境形式：**

- 单 buyer、单 seller、单 product、单 issue price bargaining；
- buyer 有 budget，seller 有 cost；
- 最多 6 turns；
- buyer 目标是低价成交，seller 目标是高价成交；
- paper-aligned metrics 包括 reward、deal rate、buyer bargained ratio、price overshoot rate。

**paper-aligned evaluation setting：**

| 参数 | 值 |
|---|---|
| buyer model | evaluated model |
| seller model | `Qwen3-30B-A3B-Instruct-2507` |
| test instances | 128 |
| rollouts per instance | 4 或 8，视实验而定 |
| max turns | 6 |
| buyer temperature | 1.0 |
| seller temperature | 0.7 |
| max tokens | 4000 |

**已实现能力：**

- paper-style direct buyer / seller prompt；
- episode-level result streaming；
- trajectory JSONL 保存；
- `framework_trace` 保存 belief / planner / generator 中间状态；
- trajectory reader 与 side-by-side comparer；
- SFT dataset build / clean / train / merge；
- CEM planner parameter search。

### 3.2 Terms Env: Unified Multi-Issue Benchmark Prototype

**位置：**

- `/work5/qixint/Terms_Env`

**环境形式：**

- 多条 contract terms / clauses 的谈判；
- scripted seller；
- buyer 需要在 agreement / utility / preference / risk calibration 之间权衡；
- 相比 Simple Env 更复杂，适合作为较难 benchmark。

**已实现 baseline：**

| baseline | 文件 |
|---|---|
| direct prompt | `Terms_Env/buyer/direct_prompt.py` |
| CoT prompt | `Terms_Env/buyer/cot_prompt.py` |
| belief prompt | `Terms_Env/buyer/belief_prompt.py` |
| planner + generator | `Terms_Env/buyer/planner_generator.py` |
| full framework | `Terms_Env/buyer/full_framework.py` |
| typed belief | `Terms_Env/buyer/typed_belief.py` |
| counterfactual belief | `Terms_Env/buyer/counterfactual_belief.py` |
| ASTRA-style belief | `Terms_Env/buyer/astra_style_belief.py` |
| BOND-style posterior | `Terms_Env/buyer/bond_style_posterior.py` |
| preference estimation | `Terms_Env/buyer/preference_estimation.py` |

**主要指标：**

- `deal_rate`
- `avg_buyer_utility`
- `avg_seller_utility`
- `avg_joint_utility`
- `avg_bargained_ratio_on_deals`
- `agreement_efficiency`
- `reservation_range_coverage`
- `accept_prob_brier`
- `evidence_grounding_rate`
- `priority_top1_accuracy`
- violation / repair rates

### 3.3 AgenticPay

**位置：**

- 主脚本：`/work5/qixint/experiments/run_agenticpay_single28_framework.py`
- fixed-seller comparison：`/work5/qixint/experiments/run_agenticpay_single28_fixed_seller_comparison.py`
- metric recompute：`/work5/qixint/experiments/recompute_agenticpay_single28_summary.py`
- framework 组件：`/work5/qixint/experiments/agenticpay_framework/`

**环境价值：**

- 更接近 universal negotiation setting；
- 包含多 buyer / seller / product / bargain / contract structure；
- 能暴露 Simple Env 不能暴露的问题，例如 seller error attribution、contract feasibility、IR violation、multi-party scoring mismatch。

**已完成关键修正：**

原始 AgenticPay scoring 会把 seller 失误也反映到 buyer 分数里。为 fixed-seller buyer comparison，已增加：

- `avg_buyer_score_fixed_seller`
- `avg_buyer_score_no_seller_error`
- `avg_buyer_score_deals_only`
- `seller_error_attribution_rate`
- `fixed_seller_eval_reasons`

### 3.4 ASTRA / CasiNo

**位置：**

- 外部代码：`/work5/qixint/external_negotiation_envs/ASTRA`
- wrapper：`/work5/qixint/experiments/external_comparisons/run_astra_table1_our_framework.py`
- ASTRA agent 适配：`/work5/qixint/external_negotiation_envs/ASTRA/agent/our_framework_agent.py`

**环境形式：**

- CasiNo camping neighbor negotiation；
- 多 issue：food、water、firewood；
- partner types 包含 base / Pro-CoT / greedy / fair；
- table metrics 包括 Avg. Score、Avg. Score Agreement、T-statistic Agreement、Walk-Away。

**当前状态：**

- 已实现 reference ASTRA、standard belief our framework、rich belief our framework 三类 run；
- 已支持 batch / resume；
- 已保存 ASTRA logs / utter-level history；
- 但原环境 LLM parser 和 aggregate stats 容易失败，例如 `'Answer'` parse exception、zero-division；
- 当前 ASTRA 结果应标注为 exploratory，不宜作为最终结论。

### 3.5 其他外部环境与 wrapper

已下载或写过 smoke wrapper 的环境包括：

| 环境 | 位置 / 脚本 | 状态 |
|---|---|---|
| bilateral trade | `experiments/external_comparisons/bilateral_trade_env.py` | smoke |
| RLVR bilateral trade wrapper | `run_rlvr_bilateral_trade_wrapper.py` | smoke |
| CaSiNo original static eval | `run_casino_original_static_eval.py` | smoke |
| Deal-or-No-Deal static eval | `run_deal_or_no_deal_static_eval.py` | smoke |
| cocoa Craigslist / DealOrNoDeal | `external_negotiation_envs/cocoa/` | downloaded / wrapper explored |
| end-to-end negotiator | `external_negotiation_envs/end-to-end-negotiator/` | downloaded |

这些环境目前主要作为后续 universal benchmark 迁移候选，不是当前主结果来源。

---

## 4. 已实现的方法与 Baseline

### 4.1 Prompt / Framework Baselines

| 方法 | 核心思想 | 已适配环境 |
|---|---|---|
| direct prompt | 原论文式直接生成 Thought/Talk/Action | Simple Env, Terms Env, AgenticPay |
| CoT prompt | 显式逐步思考后出价 | Simple Env, Terms Env, AgenticPay |
| belief prompt | 每轮 prompt 估计对手，再出价 | Simple Env, Terms Env, AgenticPay |
| planner + generator | planner 决定策略，generator 生成语言 | Simple Env, Terms Env, AgenticPay |
| full framework | belief + planner + generator | Simple Env, Terms Env, AgenticPay, ASTRA |

### 4.2 Belief Variants

| 方法 | 目的 | 状态 |
|---|---|---|
| typed belief | 将 belief 拆成 typed fields，减少自由文本漂移 | 已实现 |
| typed evidence-grounded belief | belief state 附 evidence，便于诊断与约束 | 已实现 |
| rich ASTRA belief | 在 ASTRA fairness / stance 之外加入 reservation / flexibility / patience 等 range | 已实现 |
| ASTRA-style belief | 在 Terms Env 中模拟 ASTRA priority / stance 思路 | 已实现 |
| BOND-style posterior | 在 Terms Env 中模拟 posterior / preference inference baseline | 已实现 |
| preference estimation | 明确估计对手 preference / priority | 已实现 |
| continuous belief state | 不每轮重置，而是维护 posterior + evidence log | 当前重点 |

### 4.3 Counterfactual / Reranker Variants

| 方法 | 核心思想 | 当前观察 |
|---|---|---|
| counterfactual response belief | 枚举 candidate offers，预测 seller 反应 | 初版不稳定 |
| full framework + counterfactual reranker | 在 full framework 基础上 rerank offers | 已实现，仍需 full eval |
| EV planner | 用 belief posterior 估计接受概率和期望收益 | continuous solver 当前主线 |

### 4.4 SFT / RL

| 方向 | 已实现内容 | 当前结论 |
|---|---|---|
| generator SFT | LoRA 数据清洗、训练、merge、vLLM serve、eval script | 中等规模 eval 未超过 base full framework |
| planner SFT | planner SFT 数据构建、训练、merge、eval script | 初步结果未形成稳定提升 |
| CEM planner RL | 对 concession schedule / accept threshold / future value 等小参数做 CEM search | smoke/refine set 上提升明显 |
| 未来 RL | 可将 CEM 从 search 扩展为 learned parameter policy 或 preference/reranker training | 计划中 |

---

## 5. 主要实验结果台账

### 5.1 Simple Env: direct / belief / full framework

来源：`/work5/qixint/simple_env_runs/direct_belief_full_sft_teacher/summary.json`
设置：128 test instances × 8 rollouts = 1024 episodes / variant，Qwen3-30B buyer/seller，max turns 6。

| buyer | n | avg_reward | deal_rate | bargained_ratio | first_offer_ratio | overshoot |
|---|---:|---:|---:|---:|---:|---:|
| direct_prompt | 1024 | 0.0983 | 0.6162 | 0.1595 | 0.8024 | 0.0000 |
| belief_prompt | 1024 | 0.4167 | 0.7949 | 0.5249 | 0.6254 | 0.0000 |
| full_framework | 1024 | 0.4559 | 0.8232 | 0.5538 | 0.6319 | 0.0000 |
| paper trained Qwen3-30B | - | 0.7664 | 0.9199 | 0.8385 | - | 0.0010 |

**解读：**

- prompt-only direct buyer 过早把 budget 当 target，首轮报价偏高，reward 明显低；
- belief prompt 和 full framework 都显著改善；
- full framework 仍与原论文 trained buyer 有明显差距，说明 prompt / modular framework 不能完全替代 RLVR training；
- overshoot 已被 guardrail 控制为 0。

### 5.2 Simple Env: Continuous Solver 与 CEM

来源：

- `/work5/qixint/simple_env_runs/continuous_solver_smoke_v6_belief_rescue/summary.json`
- `/work5/qixint/simple_env_runs/cem_planner_search_rule_ev_refine/best_planner_params.json`

| variant / setting | n | avg_reward | deal_rate | bargained_ratio | seller_quit / note | overshoot |
|---|---:|---:|---:|---:|---:|---:|
| continuous_rule_ev v6 smoke | 16 | 0.5517 | 0.9375 | 0.5885 | - | 0.0000 |
| continuous_hybrid_ev v6 smoke | 16 | 0.4506 | 0.8750 | 0.5149 | - | 0.0000 |
| continuous_rule_ev v7 center-belief smoke | 16 | 0.4056 | 0.8125 | 0.4992 | - | 0.0000 |
| CEM refine best train-smoke | 16 | 0.6536 | 0.9375 | 0.6972 | 0.0625 | 0.0000 |

**解读：**

- continuous belief + EV planner 可以把 first offer 拉低，同时保住 deal rate；
- 手写 schedule 敏感：v6 好于 v7，说明“中心 belief offer”不总是稳；
- CEM 对 planner 参数搜索有明显 signal；
- 但 CEM 数字来自小样本 search/eval set，必须做 paired held-out full test 后才能作为最终结果。

### 5.3 Simple Env: SFT 初步结果

来源：

- `/work5/qixint/simple_env_runs/eval_qwen30b_generator_sft_medium_64x4/summary.json`
- `/work5/qixint/simple_env_runs/eval_qwen30b_planner_sft_base_generator_medium_64x4/summary.json`
- `/work5/qixint/simple_env_runs/eval_qwen30b_planner_sft_final_base_generator_64x4/summary.json`

| model / setting | n | avg_reward | deal_rate | bargained_ratio | overshoot |
|---|---:|---:|---:|---:|---:|
| generator SFT medium | 256 | 0.3791 | 0.7305 | 0.5189 | 0.0000 |
| planner SFT medium | 256 | 0.3997 | 0.7969 | 0.5041 | 0.0000 |
| planner SFT final partial | 126 | 0.3834 | 0.8333 | 0.4600 | 0.0000 |
| base full framework reference | 1024 | 0.4559 | 0.8232 | 0.5538 | 0.0000 |

**解读：**

- 当前 SFT 没有稳定超过 base full framework；
- 可能原因是 teacher trajectories 不够强、数据筛选目标与 reward 不完全一致、generator/planner 单独 SFT 会破坏原模型灵活性；
- 下一步更推荐优先用 CEM elite trajectories 做 planner/reranker preference data，而不是直接训练 generator。

### 5.4 Terms Env all-baseline result

来源：`/work5/qixint/runs/Terms_Env/all_baselines_qwen/summary.json`，每个 variant n=60。

| buyer | deal_rate | avg_buyer_utility | bargain_on_deals | accept_prob_brier |
|---|---:|---:|---:|---:|
| full_framework | 0.7833 | 12.9182 | 0.1429 | 0.1816 |
| counterfactual_belief | 0.7500 | 12.4307 | 0.1415 | 0.1998 |
| planner_generator | 0.7833 | 10.1364 | 0.1142 | 0.2307 |
| belief_prompt | 0.7833 | 9.5550 | 0.0741 | 0.2135 |
| typed_belief | 0.7833 | 9.4097 | 0.0679 | 0.3511 |
| direct_prompt | 0.7833 | 8.5010 | 0.0613 | 0.3398 |
| astra_style_belief | 0.7833 | 8.4843 | 0.0610 | 0.2529 |
| preference_estimation | 0.7833 | 8.4843 | 0.0610 | 0.2532 |
| bond_style_posterior | 0.7667 | 8.2262 | 0.0627 | 0.2550 |

**解读：**

- 在 Terms Env 中 full framework 目前最强，且 belief calibration 的 Brier score 最低；
- counterfactual belief 接近 full framework，但 deal rate 略低；
- typed belief 当前没有体现出优势，可能只是“格式化 belief”而没有实际改善 planner 对 belief 的使用；
- Terms Env 是比 Simple Env 更难的 unified benchmark 候选。

### 5.5 AgenticPay single28 fixed-seller result

来源：`/work5/qixint/runs/agenticpay_single28_qwen14_4gpu_full/summary.json`，Qwen14B，28 tasks / variant。

| buyer | tasks | deal_rate | fixed_seller buyer score | no_seller_error score | deals_only score |
|---|---:|---:|---:|---:|---:|
| belief_prompt | 28 | 0.7143 | 31.9105 | 24.5851 | 30.0484 |
| cot_prompt | 28 | 0.6071 | 26.8610 | 23.0220 | 35.1412 |
| full_framework | 28 | 0.8214 | 23.6146 | 23.7121 | 28.3110 |
| planner_generator | 28 | 0.9286 | 18.8966 | 18.8550 | 19.6853 |

**解读：**

- planner_generator deal rate 最高，但 buyer score 最低，说明它更容易促成交易但不够会“为 buyer 争取 surplus”；
- belief_prompt fixed-seller score 最高，说明这个环境下更强的 belief / caution 可能比复杂 planner 更有用；
- full_framework deal rate 高，但 fixed-seller buyer score 不突出，暴露出 planner/generator 在复杂 contract 中可能让步太多或目标不稳定；
- AgenticPay 很适合作为下一步 universal benchmark 的重要候选，因为它能暴露 Simple Env 中看不到的 multi-party / contract / seller-error 问题。

### 5.6 ASTRA / CasiNo exploratory result

来源：`/work5/qixint/runs/astra_table1_our_framework_with_ref/`。

当前已保存：

- `partner_base/reference_astra.json`
- `partner_base/our_framework_standard_belief.json`
- `partner_base/our_framework_rich_belief.json`
- `partner_procot/*`
- `partner_greedy/*`
- `partner_fair/*`

示例 walk-away rate：

| partner | reference_astra | standard belief | rich belief |
|---|---:|---:|---:|
| base | 0.0659 | 0.2143 | 0.0714 |
| procot | 0.2143 | 0.6000 | 0.4286 |
| greedy | 0.6000 | 0.9333 | 0.4286 |
| fair | 0.3077 | 0.6154 | 0.0769 |

**解读与注意：**

- rich belief 相比 standard belief 在多个 partner 上显著降低 walk-away，说明 richer opponent state 可能有用；
- 但这些 ASTRA runs 经历过 parse exception 和 zero-division，且样本数/完整性需要重新审计；
- ASTRA 目前更适合作为 CasiNo adaptation proof-of-concept，而不是主结果表。

---

## 6. 已生成 PPT / MD 资产索引

### 6.1 6.26: 外部 benchmark 与 belief novelty

| 文件 | 内容 |
|---|---|
| `6.26/external_negotiation_benchmarks_reproducibility_plan_cn.md` | RLVR / ASTRA / Terms-Bench / Preference Estimation / BOND 等可复现性调研 |
| `6.26/opponent_belief_novelty_ideas_cn.md` | opponent belief novelty 初版想法 |
| `6.26/opponent_modeling_expanded_ideas_and_advisor_pitch_cn.md` | 更扩展的 belief / planner / training idea 与导师汇报思路 |
| `6.26/negotiation_experiments_and_belief_novelty_report_en.pptx` | 早期英文汇报 PPT |
| `6.26/agenticpay_traj_full_vs_belief_analysis_cn.md` | AgenticPay trajectory 分析 |
| `6.26/agenticpay_traj_full_vs_belief_analysis_en.pptx` | AgenticPay 英文汇报 PPT |

### 6.2 7.1: Unified benchmark 与 prompt/belief 计划

| 文件 | 内容 |
|---|---|
| `7.1/next_experiment_plan_cn.md` | 下一步实验计划：unified benchmark + belief model 实验 |
| `7.1/rlvr_negotiation_prompt_fix_and_belief_plan_cn.md` | RLVR prompt fix、typed belief、counterfactual response belief、SFT/RL 计划 |
| `7.1/unified_benchmark_experiment_plan_en.pptx` | unified benchmark 英文汇报 |
| `7.1/README_rlvr_terms_experiments.md` | vLLM / RLVR / Terms 实验命令 |

### 6.3 7.17: Continuous resolving / CEM 阶段

| 文件 | 内容 |
|---|---|
| `7.17/continual_resolving_belief_experiment_plan_cn.md` | DeepStack / Libratus 启发下的 continual resolving belief 计划 |
| `7.17/negotiation_benchmark_and_planner_belief_update_en_editable.pptx` | 当前主要结果 PPT；需忽略 trained belief model 数据 |
| `7.17/continuous_solver_cem_planner_update_en_editable.pptx` | CEM planner RL 更新 PPT |
| `7.17/README.md` | vLLM、CEM eval、CEM refine 等命令 |

### 6.4 7.27: 当前报告

| 文件 | 内容 |
|---|---|
| `7.27/NEGOTIATION_CONTINUOUS_SOLVER_STAGE_REPORT_20260727.md` | 当前阶段报告，聚焦 continuous solver / CEM |
| `7.27/NEGOTIATION_CODEBASE_OVERVIEW_REPORT_20260727.md` | 本代码库总览报告 |

---

## 7. 当前代码库中最重要的可复用入口

### 7.1 Simple Env evaluation

```bash
bash /work5/qixint/Simple_Env/scripts/run_rlvr_paper_eval.sh
```

常用环境变量：

```bash
RUN_OUTPUT_DIR=/work5/qixint/simple_env_runs/<run_name>
RUN_BUYER_VARIANTS=direct_prompt,belief_prompt,full_framework
RUN_NUM_TEST_INSTANCES=128
RUN_ROLLOUTS_PER_INSTANCE=4
RUN_CONCURRENCY=1
RUN_RETRY_FAILED=1
```

### 7.2 Simple Env trajectory reader

```bash
/work5/qixint/miniconda3/envs/research/bin/python \
  /work5/qixint/Simple_Env/tools/read_trajectories.py \
  --run-dir /work5/qixint/simple_env_runs/<run_name> \
  --variant continuous_rule_ev \
  --limit 5 \
  --show-belief
```

### 7.3 Trajectory side-by-side comparison

```bash
/work5/qixint/miniconda3/envs/research/bin/python \
  /work5/qixint/Simple_Env/tools/compare_trajectories.py \
  --left-run-dir /work5/qixint/simple_env_runs/<run_a> \
  --left-variant full_framework \
  --right-run-dir /work5/qixint/simple_env_runs/<run_b> \
  --right-variant continuous_rule_ev \
  --limit 5
```

### 7.4 CEM planner training

```bash
bash /work5/qixint/Simple_Env/scripts/train_continuous_rule_ev_cem_planner.sh
```

### 7.5 Terms Env all baselines

```bash
bash /work5/qixint/Terms_Env/scripts/run_terms_all_baselines.sh
```

### 7.6 AgenticPay fixed-seller comparison

```bash
/work5/qixint/miniconda3/envs/research/bin/python \
  /work5/qixint/experiments/run_agenticpay_single28_fixed_seller_comparison.py
```

具体参数以历史 `runs/agenticpay_single28_qwen14_4gpu_full/summary.json` 中保存的 command 为准。

---

## 8. 当前主要经验与失败模式

### 8.1 Simple Env 中已经确认的行为规律

1. direct prompt 常把 budget 当 target，而不是 upper bound；
2. 只靠 prompt 的 buyer 往往 opening price 太高；
3. full framework 能提高 deal rate 和 bargained ratio，但仍低于 RLVR-trained model；
4. belief 一旦过于悲观，会导致 buyer 过早 walk away；
5. planner 若不显式控制 concession schedule，容易在 aggressive 与 conservative 之间摆动；
6. CEM 小参数搜索可以低成本改善 schedule；
7. generator SFT / planner SFT 若 teacher 数据不够强，可能不如 base full framework。

### 8.2 Terms Env 暴露的问题

1. typed belief 只改格式，不一定提升 planner 决策；
2. counterfactual belief 可以提升 buyer utility，但可能牺牲 deal rate；
3. Brier score 显示 belief calibration 本身可作为独立优化目标；
4. multi-issue 环境需要 planner 显式处理 term priority 和 concession across issues。

### 8.3 AgenticPay 暴露的问题

1. buyer score 容易被 seller error 污染；
2. 高 deal rate 不等于高 buyer score；
3. full framework 在复杂 contract 中可能目标混乱；
4. belief_prompt 在 fixed-seller score 上强，说明 robust opponent modeling 可能比复杂 planner 更关键。

### 8.4 ASTRA 暴露的问题

1. 外部 LLM environment 的 parser / response format fragile；
2. batch resume 是必须能力；
3. 原 Table 1 对比需要严格复现 partner prompt、round、weights 和 scoring；
4. 当前 ASTRA 结果需要重新审计完整性。

---

## 9. 当前可靠性分级

| 结果类型 | 可靠性 | 说明 |
|---|---|---|
| Simple Env direct/belief/full 1024 episodes | 高 | 样本数足，paper setting 基本对齐 |
| Terms Env all-baseline n=60 | 中 | 可比较 baseline，但环境是自建 prototype |
| AgenticPay single28 fixed-seller | 中 | multi-agent 更复杂，但样本数小且 timeout 较多 |
| CEM refine best n=16 | 低到中 | signal 很好，但样本太小，需要 held-out |
| SFT generator/planner medium eval | 中 | 有 256 episodes，但数据/teacher 质量需复查 |
| ASTRA Table 1 adaptation | 低到中 | 已接入，但原代码不稳定，需重新审计 |

---

## 10. 下一步计划

### 10.1 Simple Env 内部闭环

1. 用 CEM best 参数做 paired held-out evaluation：
   对比 `full_framework`、`continuous_rule_ev_conservative`、`continuous_rule_ev_current`、`continuous_rule_ev_cem`。

2. 收集 high-quality trajectories：
   只保留高 reward、低 overshoot、deal 成功、seller quit 少的 trajectories。

3. 用 CEM elite trajectories 训练 planner/reranker：
   优先训练 planner / offer reranker，而不是 generator。

4. 对 belief calibration 单独建评估：
   用 seller cost / accepted prices / rejected offers 计算 posterior coverage、Brier score、calibration error。

### 10.2 迁移到更 universal benchmark

下一步需要调研并选择更有说服力的 benchmark，优先级建议：

1. **AgenticPay extended setting**
   优点：已有代码和 fixed-seller metric；包含 multi buyer / seller / product / contract；最贴近“universal”目标。
   任务：把 continuous belief state / CEM planner 迁移进去。

2. **Terms Env 扩展版**
   优点：当前已有统一 baseline；可控、便于调试 belief calibration；适合做多 issue 实验。
   任务：加入 multi seller / multi buyer 或更真实的 preference estimation baseline。

3. **CaSiNo / ASTRA 稳定复现**
   优点：已有论文 Table 1，可直接讲外部对比；多 issue。
   风险：原代码 fragile，需要 parser / retry / batch aggregation 修复。

4. **Craigslist / Deal-or-No-Deal / cocoa**
   优点：经典 negotiation benchmarks，被较多论文引用。
   风险：老代码和现代 LLM wrapper 适配成本较高。

### 10.3 研究 novelty 方向

建议把 novelty 聚焦为：

> Continual opponent belief resolving for LLM negotiation: maintain calibrated opponent belief state across turns, use it to evaluate candidate actions, and improve planner policy through low-cost parameter search / SFT / RL.

可展开为三个模块：

1. **Belief model novelty**
   从一次性 prompt belief 变成 persistent posterior state，支持 evidence-grounded update、calibration metrics、multi-issue / multi-agent 扩展。

2. **Planner coupling novelty**
   planner 不直接问 LLM “下一步怎么办”，而是在 belief 下枚举 candidate actions，估计 EV / risk / concession schedule，再交给 generator。

3. **Low-cost learning novelty**
   不必一开始 RL fine-tune 30B model；先通过 CEM / preference reranking / small planner head 学习低维 strategy parameters，再用 elite trajectories 训练 planner。

---

## 11. 当前版本报告

**版本名：** `negotiation-codebase-overview-v0.1`
**日期：** 2026-07-27
**新增内容：**

- 扫描并整理了当前 negotiation 代码库中的主要环境、方法、baseline、脚本、结果与 PPT/MD；
- 将 Simple Env、Terms Env、AgenticPay、ASTRA 和外部 smoke wrappers 统一放入一个代码库台账；
- 明确标注哪些结果可靠、哪些只是 smoke / exploratory；
- 给出下一阶段从 Simple Env 迁移到 universal benchmark 的路线。

**本报告验证的判断：**

- 当前代码库已经不只是单个实验脚本，而是一个多环境 negotiation research sandbox；
- Simple Env 结果最完整，但存在特化风险；
- Terms Env / AgenticPay 是下一步最重要的泛化验证目标；
- belief model 的 novelty 应从“prompt 改写”转向“persistent calibrated state + planner action evaluation”。

**下一步最短路径：**

1. 完成 CEM best 的 Simple Env paired held-out full evaluation；
2. 收集 CEM elite trajectories，训练 planner / reranker；
3. 在 AgenticPay 或 Terms Env 中实现 continuous belief state + EV planner；
4. 形成一张跨环境表：Simple Env、Terms Env、AgenticPay 上 direct / belief / full / continuous / CEM 的统一对比。
