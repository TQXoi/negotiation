# ANL 2024 × Continuous Belief / Belief-Usable Planner 基础集成与 Smoke Test 报告

> 日期：2026-08-05
> 实现目录：`/work5/qixint/external_negotiation_envs/ANL_2024_2026/`
> 目的：验证 continuous belief 与 belief-usable planner 能否在 ANL 2024 原生 SAO 协议中运行，并建立可测量的 belief→action→outcome 链条。

## 1. 当前状态

已经完成可运行的 ANL 2024 belief/planner framework 基础版本，并未修改 `official/` 上游快照。

实现文件：

| 文件 | 角色 |
|---|---|
| `belief_planner_anl/anl2024/belief.py` | 对手 reservation value 的离散 posterior、continuous update、置信区间和 information gain |
| `belief_planner_anl/anl2024/planner.py` | 候选 outcome 的自身收益、接受概率、风险、information gain 与 continuation value |
| `belief_planner_anl/anl2024/agent.py` | `ANLNegotiator` adapter、action lock、逐轮 framework trace |
| `belief_planner_anl/anl2024/runner.py` | paired scenario runner、resume、结果与版本记录 |
| `scripts/run_anl2024_basic.sh` | 统一运行入口 |
| `tests/test_anl2024_framework.py` | belief、truth isolation 和 paired runner 测试 |

## 2. 为什么没有照搬 ANL 2025 belief

现有 ANL 2025 adapter 的 `OfferBelief` 主要统计各 issue value 的报价频率。ANL 2024 的信息结构不同：对手 utility function 的 shape 已知，唯一隐藏的核心量是 reservation value。因此新 adapter 使用显式 posterior：

\[
p_t(r_{opp}) \propto p_{t-1}(r_{opp})
P(o_t\mid r_{opp}, h_{t-1}).
\]

公共 evidence 包括：

- 对手提出的 outcome 在其已知 utility shape 下的 utility；
- 对我方上一报价的拒绝；
- relative time；
- 对手报价轨迹的 concession shape。

真实 RV 只由 evaluator 保存，不进入 standard agent 的 private info、belief 或 trace。

## 3. Planner 如何使用 belief

每个合法 outcome 均记录：

```text
own_utility
own_surplus_norm
opponent_utility
P(opponent RV <= opponent utility)
P(accept | posterior, time)
expected_self_surplus
expected_information_gain
aspiration_penalty
disagreement_risk
planner_score
```

接受决策比较当前收到报价的自身 utility 与下一计划报价的 belief-conditioned continuation value。结构化 outcome 由 planner 锁定，ANL mechanism 直接执行，不加入自然语言 generator，避免把格式收益混入第一次 belief/planner 验证。

## 4. 当前 variants

| Variant | 含义 |
|---|---|
| `boulware` | 原生 time-based baseline |
| `rv_fitter` | ANL 官方 curve-fitting RV baseline |
| `framework_static_belief` | 固定 uniform RV prior + 同一 planner |
| `framework_continuous_belief` | continuous RV posterior + planner |
| `framework_no_info_gain` | continuous posterior + planner，information-gain 权重为零 |
| `framework_frozen_belief` | 50% deadline 后冻结 posterior |
| `framework_oracle_belief` | evaluator-only oracle RV 上界，不是实际方法 |
| `framework_shuffled_belief` | 使用下一 paired scenario 的 RV posterior，检验 planner 的 belief 因果敏感性 |

默认脚本先运行前五项。oracle/frozen 用于后续 causal ablation。

## 5. 测试

已通过 7 个代码级测试：

1. 低 opponent-utility 报价会降低 RV posterior；
2. rejection 会作为弱 evidence 更新 posterior；
3. static posterior 不移动；
4. 单局 negotiation 可运行，trace 不含真实 RV；
5. paired experiment 正确输出三个 variants 与 action diagnostics。
6. shuffled/wrong-belief 干预可运行，且 framework trace 不泄露 evaluator truth。
7. 历史 `astra_*` variant 与 import alias 可兼容，但输出会规范化为 `framework_*`。

测试命令：

```bash
MPLCONFIGDIR=/tmp/anl_mpl \
PYTHONPYCACHEPREFIX=/tmp/anl_pycache \
PYTHONPATH=/work5/qixint/external_negotiation_envs/ANL_2024_2026/official/negmas/src:/work5/qixint/external_negotiation_envs/ANL_2024_2026/official/anl-platform-current/src:/work5/qixint/external_negotiation_envs/ANL_2024_2026 \
/work5/qixint/miniconda3/envs/research/bin/python -m pytest -q \
/work5/qixint/external_negotiation_envs/ANL_2024_2026/tests/test_anl2024_framework.py
```

结果：`7 passed`。

## 6. Smoke 过程中发现并修复的问题

第一版将早期 rejection 过强地解释为高 RV。Boulware 即使获得 0.97 左右的 opponent utility，也可能因早期 aspiration 接近 1.0 而拒绝；这不意味着其 RV 为 0.97。

第一版结果完整保留于：

```text
runs/anl2024_smoke_20260805/
```

修正包括：

- opponent offers/concession trajectory 成为主要 RV evidence；
- 对常见 Conceder/Linear/Boulware aspiration exponent 做保守边缘化；
- rejection 只做 fractional/tempered Bayesian update；
- 避免由错误 behavioral-family assumption 导致 posterior 过度自信。

修正后结果保留于：

```text
runs/anl2024_smoke_calibrated_20260805/
```

## 7. 修正后 smoke 结果

设置：2 scenarios × 3 opponents × 5 variants，共 30 episodes；每个 variant 有 6 个 paired episodes，20 steps。该样本只用于功能验证。

| Variant | Agreement | Own utility | Own advantage | RV MAE | q10–q90 coverage | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Boulware | 0.667 | **0.6058** | **0.5324** | — | — | 0 |
| RVFitter | **1.000** | 0.5429 | 0.3169 | — | — | 0 |
| Framework static belief | 0.667 | 0.5998 | 0.5160 | 0.0879 | 1.000 | 0 |
| Framework continuous belief | 0.667 | 0.5998 | 0.5160 | **0.0682** | 0.833 | 0 |
| Framework no-IG | 0.667 | 0.5998 | 0.5160 | **0.0682** | 0.833 | 0 |

相对 static 的 paired action diagnostics：

| Variant | Paired episodes | First-action flip | Offer-sequence flip | Agreement-outcome flip |
|---|---:|---:|---:|---:|
| Framework continuous belief | 6 | 0.000 | **0.500** | 0.000 |
| Framework no-IG | 6 | 0.000 | **0.500** | 0.000 |

## 8. 对结果的严格解释

目前可以确认：

- ANL 2024 原生协议、truth masking、continuous belief、planner 和 trace 均已接通；
- 修正后 continuous posterior 的 smoke RV MAE 低于 static prior；
- belief update 已使一半 paired episodes 的后续 offer sequence 发生变化，planner 没有完全忽略 belief；
- 所有 30 局无运行错误。

目前不能确认：

- framework 性能优于 Boulware/RVFitter；
- continuous belief 已显著改善最终 outcome；
- information gain 有独立贡献；
- 6 个 paired episodes 的 MAE/coverage 具有统计意义。

特别是 continuous、static 和 no-IG 在该 smoke 中最终 agreement/utility 相同，agreement-outcome flip 为零。下一轮必须加入 oracle/shuffle/frozen，并扩大 paired scenarios，才能判断瓶颈在 belief 还是 planner sensitivity。

补充的 2-scenario intervention smoke 位于 `runs/anl2024_intervention_smoke_20260805/`。其中 oracle 相对 static 在 1/2 paired scenarios 改变最终 agreement，并将 agreement rate 从 0.5 提高到 1.0；continuous 和 shuffled 在这个极小样本上尚未改变最终结果。这只是链路验证，不是性能结论，但说明 planner 对正确 RV 存在可测量的 outcome sensitivity，扩大 paired pilot 是有意义的。

## 9. 运行命令

快速 smoke：

```bash
RUNS=2 N_OUTCOMES=20 N_STEPS=15 \
OUTPUT_DIR=runs/anl2024_smoke_user \
bash /work5/qixint/external_negotiation_envs/ANL_2024_2026/scripts/run_anl2024_basic.sh
```

建议的第一轮 pilot：

```bash
RUNS=30 N_OUTCOMES=100 N_STEPS=50 \
VARIANTS=rv_fitter,framework_static_belief,framework_continuous_belief,framework_frozen_belief,framework_oracle_belief,framework_shuffled_belief \
OUTPUT_DIR=runs/anl2024_pilot_n30 \
bash /work5/qixint/external_negotiation_envs/ANL_2024_2026/scripts/run_anl2024_basic.sh
```

中断后继续：在相同环境变量和 output directory 后附加 `--resume`。

## 10. 下一步

1. 运行 30-scenario paired pilot；
2. 增加 shuffled/wrong-confidence belief intervention；
3. 同时报告 per-turn MAE/NLL/coverage，不只 final MAE；
4. 检查 oracle 是否改变 action/outcome：若 oracle 也不改善，先调整 planner；
5. 只有 oracle gap 与 shuffled degradation 都成立后，才加入 Qwen LLM chooser；
6. locked test 至少使用 200 unseen scenarios，并加入 official winners。
