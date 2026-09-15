# Negotiation Continuous Solver 阶段性研究报告

**当前框架：** `Simple_Env` continuous resolving + CEM planner RL
**当前实验版本：** `negotiation-continuous-solver-v0.1`
**报告日期：** 2026-07-27
**研究阶段：** Simple Env 中完成初步闭环与 CEM 参数搜索；跨环境泛化尚待验证
**主要参考材料：** `/work5/qixint/7.17/negotiation_benchmark_and_planner_belief_update_en_editable.pptx`
**注意：** 本报告忽略该 PPT 中 `trained belief model` 相关数据；仅总结当前已实际落地与可追溯的环境、agent、实验和 CEM 结果。

---

## 0. 后续报告更新约定

为降低后续协作成本，从本报告开始，每个重要阶段都应更新一份当前版本总结，至少包含：

| 模块 | 需要记录的内容 |
|---|---|
| 当前版本 | 版本名、日期、对应代码/脚本/输出目录 |
| 已实现内容 | 新增环境、buyer/seller、belief/planner/generator、工具脚本 |
| 修改目的 | 该改动要验证什么猜想，解决什么失败模式 |
| 代码改动 | 关键文件路径与接口变化 |
| 实验结果 | 运行设置、样本数、metrics、traj 观察 |
| 是否验证猜想 | 明确写“支持/不支持/证据不足” |
| 风险与限制 | 是否 overfit、是否缺 held-out、是否环境太简单 |
| 下一步计划 | 具体命令、数据收集、要迁移的 benchmark |

当前报告是第一次总括性报告，覆盖此前已实现的大部分结果，并给出下一阶段路线。

---

## 1. 执行摘要

本阶段的核心目标是把原本较松散的 A+B+C negotiation framework 推进到一个更可学习、更可审计的 negotiation agent：

> 不再只依赖一次性 prompt belief，而是维护一个持续更新的 opponent belief state；planner 根据 belief 枚举与评估 candidate offers；generator 再把 selected plan 转换成强势但格式合规的 buyer dialogue。

目前最重要的进展有三点：

1. **完成了模块化 Simple Env。**
   `Simple_Env` 已将 buyer、seller、environment、tools、scripts 拆分，能够复现 RLVR negotiation paper setting，并保存完整 trajectory / framework trace。

2. **实现了 continuous resolving 流程。**
   在 `Simple_Env/buyer/continuous_solver/` 中实现了 persistent belief model、rule/LLM/hybrid update、EV planner、prompt generator，以及可视化 traj reader。

3. **通过 CEM 找到了一组 promising planner 参数。**
   CEM refine best 在 16-scenario train-smoke 上达到：

   | metric | value |
   |---|---:|
   | avg_reward | `0.6536` |
   | deal_rate | `0.9375` |
   | buyer_bargained_ratio | `0.6972` |
   | seller_quit_rate | `0.0625` |
   | overshoot_rate | `0.0` |

但当前最准确的结论仍然是：

> CEM 已在 Simple Env 的训练/搜索集上找到明显优于手写 schedule 的参数，但尚未完成独立 held-out / full paired evaluation。因此它是 promising signal，不是最终确认结果。

下一阶段应优先做两件事：

1. 在 Simple Env 上用 CEM best 进行 paired smoke/full evaluation，并收集 trajectories 用于 planner / belief model training；
2. 调研并迁移到更 universal、更有说服力的 multi-agent negotiation benchmark，避免方法只对 Simple Env 这种单 buyer、单 seller、单 product、单 issue bargaining 过拟合。

---

## 2. 研究问题与当前假设

### 2.1 核心研究问题

当前研究问题可以表述为：

> 在 LLM negotiation 中，将 opponent belief 从一次性 prompt 转为持续维护的 state，并让 planner 在该 belief 下进行 explicit candidate evaluation / learned concession scheduling，能否提升 buyer 的 reward、deal rate 与 bargaining efficiency？

### 2.2 当前可检验假设

| 假设 | 解释 | 当前状态 |
|---|---|---|
| H1: persistent belief 有助于 planner | belief 不随每轮 prompt 重写，而是积累 seller 行为证据 | 已实现，部分支持 |
| H2: 手写 concession schedule 不稳定 | 不是太 aggressive 导致 seller quit，就是太 conservative 导致 buyer surplus 低 | traj 支持 |
| H3: planner 参数适合低成本 RL | 不训练 Qwen，只搜索小参数向量即可改善 offer/accept 策略 | CEM train-smoke 支持 |
| H4: Simple Env 结果可能环境特化 | 单 product 单 price bargaining 太简单，方法可能只学到特定价格曲线 | 尚未验证，需要迁移 |
| H5: 下一步应训练 planner/reranker 或 belief calibration | CEM elite trajectories 可产生 supervised/preference data | 计划中 |

---

## 3. 已实现的工程系统

### 3.1 Simple Env 模块化环境

主要目录：

```text
Simple_Env/
  eval.py
  buyer/
  seller/
  environment/
  tools/
  scripts/
```

关键功能：

| 功能 | 状态 | 文件 |
|---|---|---|
| RLVR paper-aligned evaluation | 已实现 | `Simple_Env/eval.py` |
| buyer / seller / environment 解耦 | 已实现 | `Simple_Env/buyer/`, `Simple_Env/seller/`, `Simple_Env/environment/` |
| 逐 episode 输出 compact result | 已实现 | `Simple_Env/environment/episode.py` |
| trajectory 保存 | 已实现 | `*_episodes.jsonl` |
| framework trace 保存 | 已实现 | episode 中 `framework_trace` |
| belief + seller cost 诊断 reader | 已实现 | `Simple_Env/tools/read_trajectories.py` |
| side-by-side traj 对比 | 已实现 | `Simple_Env/tools/compare_trajectories.py` |

### 3.2 Buyer variants

当前 Simple Env 已支持以下主要 buyer：

| buyer variant | 作用 |
|---|---|
| `direct_prompt` | paper-style direct buyer prompt |
| `cot_prompt` | chain-of-thought prompt baseline |
| `belief_prompt` | belief-only prompt baseline |
| `full_framework` | A+B+C framework baseline |
| `typed_belief` | typed evidence-grounded belief baseline |
| `counterfactual_response_belief` | counterfactual response belief / reranker baseline |
| `continuous_rule_ev` | persistent posterior belief + rule update + EV planner |
| `continuous_hybrid_ev` | rule update + LLM-assisted update |
| `continuous_rule_ev_conservative` | 较保守 schedule baseline |
| `continuous_rule_ev_current` | 当前 hand-written center-belief schedule |
| `continuous_rule_ev_cem` | CEM-tuned planner parameters |

### 3.3 Seller variants

Simple Env 目前主要使用 fixed default seller：

| seller | 作用 |
|---|---|
| `default` | paper-aligned seller system prompt，目标为尽可能高价成交 |
| `seller_persona=neutral/begging/insulting/unyielding` | 预留 adversarial persona 支持 |

当前主实验使用：

```text
seller_model = Qwen3-30B-A3B-Instruct-2507-base
seller_type = default
seller_persona = neutral
```

---

## 4. Continuous Resolving 当前算法结构

### 4.1 总体流程

```text
history + scenario
    ↓
Persistent Belief Model
    ↓
Rule / LLM / Hybrid Belief Update
    ↓
EV Planner / Parametric Planner
    ↓
Prompt Generator
    ↓
Thought / Talk / Action
```

### 4.2 Dialogue template

当前 generator 使用固定的 negotiation style：

- 首轮 aggressive anchor；
- 后续 slow concession；
- 不把 buyer budget 当 target；
- 不超过 buyer budget；
- 不超过 seller 已给出的最低 formal quote；
- 语言上使用强势、简洁、有战术 finality 的表达。

这部分主要由以下文件控制：

```text
Simple_Env/buyer/continuous_solver/generator/prompt_generator.py
```

### 4.3 Belief model

当前 belief model 维护一个 seller reservation posterior：

```text
P_t(r), r ∈ discrete price grid
```

并维护 interaction state：

```text
reservation mean / p10 / p50 / p90 / confidence
quit_risk
finality_prob
patience
seller concession slope
evidence log
```

关键文件：

```text
Simple_Env/buyer/continuous_solver/belief_model/base.py
Simple_Env/buyer/continuous_solver/belief_model/posterior.py
Simple_Env/buyer/continuous_solver/update/rule_based.py
Simple_Env/buyer/continuous_solver/update/llm_assisted.py
Simple_Env/buyer/continuous_solver/update/hybrid.py
```

当前 rule-based update 的核心思想是：

- seller reject buyer offer：posterior 上移；
- seller accept buyer offer：posterior 下移；
- seller counter price：视为 strategic high anchor，不直接等于 cost；
- seller concession：降低 finality / quit risk；
- seller quit：提高 quit risk，并说明上一 buyer offer 大概率过低。

### 4.4 Planner

当前 planner 先枚举 candidate offers，再计算：

```text
EV = p_accept * reward_if_deal
   + continue_prob * future_value
   - p_quit * quit_penalty
   - lowball_penalty
```

其中：

```text
p_accept(x) = Σ_r P(r) * sigmoid((x - r) / τ)
```

关键文件：

```text
Simple_Env/buyer/continuous_solver/planner/ev_planner.py
Simple_Env/buyer/continuous_solver/planner/parametric_ev_planner.py
Simple_Env/buyer/continuous_solver/planner/conservative_ev_planner.py
Simple_Env/buyer/continuous_solver/planner/params.py
```

### 4.5 CEM planner RL

CEM 全称是 `Cross-Entropy Method`。当前不是训练 Qwen，而是学习 planner 参数：

```text
first_anchor_ratio
early_concession_rate
late_concession_rate
p50_low/mid/high multiplier
seller_discount_early/late
accept thresholds
quit penalty
future value weight
```

训练流程：

```text
sample θ_i ~ N(μ, σ)
run episodes under θ_i
rank by shaped reward
select elite θ
update μ, σ
save best_planner_params.json
```

关键文件：

```text
Simple_Env/buyer/continuous_solver/rl/train_cem_planner.py
Simple_Env/scripts/train_continuous_rule_ev_cem_planner.sh
Simple_Env/scripts/run_continuous_rule_ev_cem_comparison.sh
```

---

## 5. 当前实验结果

### 5.1 Simple Env full framework baseline

目录：

```text
/work5/qixint/simple_env_runs/direct_belief_full_sft_teacher
```

主要结果：

| variant | n | reward | deal rate | bargain ratio | overshoot |
|---|---:|---:|---:|---:|---:|
| `direct_prompt` | 1024 | 0.0983 | 0.6162 | 0.1595 | 0.0 |
| `belief_prompt` | 1024 | 0.4167 | 0.7949 | 0.5249 | 0.0 |
| `full_framework` | 1024 | 0.4559 | 0.8232 | 0.5538 | 0.0 |

解释：

- direct prompt 明显较弱；
- belief prompt 与 full framework 大幅改善 direct prompt；
- full framework 比 belief prompt 略强，但仍低于 RLVR paper trained buyer reference；
- 这支持“结构化 belief/planner 有帮助”，但并不足以形成强 novelty。

### 5.2 Continuous solver smoke 结果

| run | variant | n | reward | deal rate | bargain ratio | first offer / budget |
|---|---|---:|---:|---:|---:|---:|
| v4 conservative-ish | `continuous_rule_ev` | 16 | 0.2322 | 0.8125 | 0.2858 | 0.6876 |
| v6 belief rescue | `continuous_rule_ev` | 16 | 0.5517 | 0.9375 | 0.5885 | 0.4500 |
| v7 center belief | `continuous_rule_ev` | 16 | 0.4056 | 0.8125 | 0.4992 | 0.4500 |

主要观察：

- v4 太 conservative，成交价偏高，buyer surplus 低；
- v5/v7 更 aggressive，但容易引发 seller quit；
- v6 在 deal rate 与 bargain 上较均衡，是目前 hand-written continuous solver 中较强版本；
- traj 说明 hand-written schedule 很难稳定拿捏：不是太低导致 seller quit，就是太高损失 surplus。

这直接推动了 CEM planner parameter search。

### 5.3 CEM planner RL 结果

初始 CEM：

```text
/work5/qixint/simple_env_runs/cem_planner_search_rule_ev
```

refine CEM：

```text
/work5/qixint/simple_env_runs/cem_planner_search_rule_ev_refine
```

refine 设置：

```text
init_params_json = cem_planner_search_rule_ev/best_planner_params.json
init_std_scale = 0.5
num_train_instances = 16
rollouts_per_instance = 1
cem_iters = 4
population_size = 16
total episodes = 1024
```

最佳结果：

| metric | value |
|---|---:|
| avg_reward | 0.6536 |
| deal_rate | 0.9375 |
| buyer_bargained_ratio | 0.6972 |
| seller_quit_rate | 0.0625 |
| overshoot_rate | 0.0 |

学到的关键参数：

| parameter | value | interpretation |
|---|---:|---|
| `first_anchor_ratio` | 0.443 | 首轮低开，接近 budget 的 44% |
| `early_concession_rate` | 0.0276 | 前期慢让步 |
| `late_concession_rate` | 0.1583 | 后期快速救成交 |
| `p50_low_mult` | 0.890 | 可在 belief p50 以下试探 |
| `p50_mid_mult` | 0.978 | 主报价接近 belief p50 |
| `p50_high_mult` | 1.024 | 必要时略高于 p50 |
| `seller_discount_early` | 0.754 | 早期只信 seller quote 的约 75% |
| `seller_discount_late` | 0.840 | 后期更相信 seller quote |
| `accept_reward_threshold` | 0.175 | 接受 seller offer 仍需保留 surplus |

是否验证猜想：

| 猜想 | 当前证据 |
|---|---|
| CEM 能学到更合理 concession schedule | 训练集 best 支持 |
| CEM 能超过 hand-written continuous solver | 训练集 best 支持，held-out 未确认 |
| CEM 能泛化到完整 128x4 | 尚未验证 |

重要限制：

- CEM best 是在 16-scenario search set 上选出来的；
- `continuous_rule_ev_cem_refine_full_test` 目前不是有效 full eval：目录中只写入 1 条 conservative episode，summary 中 CEM/current 仍为 `n=0`；
- 因此不能将 CEM refine best 当作最终 test result。

### 5.4 Terms Env 初步统一 benchmark

目录：

```text
/work5/qixint/runs/Terms_Env/all_baselines_qwen
```

已实现结构：

```text
Terms_Env/
  eval.py
  buyer/
  seller/
  environment/
  tools/
  scripts/
```

代表性结果：

| variant | n | buyer utility | deal rate | bargain ratio on deals | Brier |
|---|---:|---:|---:|---:|---:|
| `direct_prompt` | 60 | 8.501 | 0.7833 | 0.0613 | 0.3398 |
| `belief_prompt` | 60 | 9.555 | 0.7833 | 0.0741 | 0.2135 |
| `planner_generator` | 60 | 10.136 | 0.7833 | 0.1142 | 0.2307 |
| `counterfactual_belief` | 60 | 12.431 | 0.7500 | 0.1415 | 0.1998 |
| `full_framework` | 60 | 12.918 | 0.7833 | 0.1429 | 0.1816 |

解释：

- Terms Env 中 full framework / counterfactual belief 对 buyer utility 和 bargain ratio 有明显优势；
- belief calibration 指标 `accept_prob_brier` 上 full framework 最好；
- 这是比 Simple Env 更接近 multi-issue contract 的方向，但目前 seller 是 scripted，仍需更真实的 multi-agent benchmark。

### 5.5 AgenticPay 固定 seller / 28 task 实验

目录：

```text
/work5/qixint/runs/agenticpay_single28_qwen14_4gpu_full
```

代表性结果：

| variant | tasks | deal rate | fixed-seller buyer score | deals-only buyer score |
|---|---:|---:|---:|---:|
| `belief_prompt` | 28 | 0.7143 | 31.91 | 30.05 |
| `cot_prompt` | 28 | 0.6071 | 26.86 | 35.14 |
| `full_framework` | 28 | 0.8214 | 23.61 | 28.31 |
| `planner_generator` | 28 | 0.9286 | 18.90 | 19.69 |

解释：

- AgenticPay 中曾修正 buyer score，使 fixed-seller comparison 不把 seller 错误错误归因给 buyer；
- 该环境暴露出 planner/generator 分解在复杂交易中未必稳定；
- AgenticPay 是下一阶段 universal benchmark 的重要候选，因为它更接近 multi-agent / multi-product / multi-bargain 场景。

---

## 6. 当前证据支持什么

### 6.1 可以成立的结论

1. **Simple Env 中 modular negotiation pipeline 已经可运行。**
   buyer、seller、environment、trajectory reader、summary writer、vLLM scripts 已形成闭环。

2. **只靠 direct prompt 不够。**
   RLVR Simple Env 中 direct prompt reward 低，belief/full framework 明显提升。

3. **持续维护 belief + EV planner 是可行的。**
   continuous solver 可以保存 belief/posterior/candidate/selected action，并能用 seller cost 做 calibration diagnostics。

4. **手写 concession schedule 不稳定。**
   traj 反复显示过激与保守之间存在 tradeoff，很难靠 prompt/rule 手调完全解决。

5. **CEM planner parameter search 是合理的低成本 learning 方向。**
   不新增模型显存，不训练 Qwen，只优化小参数向量；在 search set 上已找到 promising schedule。

6. **Terms Env 与 AgenticPay 说明跨环境需求真实存在。**
   不同环境中最优 baseline 不一致，说明需要更 universal 的 benchmark 来检验方法，而不是只在 Simple Env 上优化。

### 6.2 目前不能成立的结论

当前证据不足以声称：

- CEM planner 已在 held-out/full test 上显著优于 hand-written continuous solver；
- continuous solver 已达到或超过 RLVR paper trained Qwen buyer；
- 当前 belief posterior 已足够准确；
- 当前方法能自然泛化到 multi-issue / multi-seller / multi-buyer；
- AgenticPay 或 Terms Env 已经完整证明我们的方法优于所有 baseline。

原因：

- CEM full paired eval 尚未完成；
- Simple Env 过于单一，容易 schedule overfit；
- Terms Env 目前更像 controlled synthetic multi-issue test；
- AgenticPay 结果复杂，且 planner/generator 在部分 setting 下表现不稳定。

---

## 7. 当前版本 v0.1 记录

### 7.1 新增内容

| 新增内容 | 目的 | 关键代码 |
|---|---|---|
| `Simple_Env` modular RLVR environment | 让 RLVR negotiation 可复现、可保存 traj、可换 buyer/seller | `Simple_Env/eval.py`, `Simple_Env/environment/` |
| continuous solver | 实现 persistent belief + continual resolving | `Simple_Env/buyer/continuous_solver/` |
| posterior belief model | 不再每轮纯 prompt belief，而是维护数学 state | `belief_model/posterior.py` |
| rule / hybrid updater | 从 seller action 更新 reservation 与 interaction state | `update/rule_based.py`, `update/hybrid.py` |
| EV planner | 在 belief 下枚举 candidate offers 并计算 EV | `planner/ev_planner.py` |
| parametric planner | 为 CEM 提供可学习参数向量 | `planner/parametric_ev_planner.py`, `planner/params.py` |
| conservative baseline | 保留上一版更保守 schedule 作为对比 | `planner/conservative_ev_planner.py` |
| CEM training | 低成本 black-box RL 搜索 planner 参数 | `rl/train_cem_planner.py` |
| trajectory diagnostics | 输出 belief vs seller cost calibration | `tools/read_trajectories.py`, `tools/compare_trajectories.py` |
| Terms Env | 初步统一 multi-issue-style benchmark | `Terms_Env/` |

### 7.2 当前 CEM 版本是否验证了猜想

| 猜想 | 结果 | 解释 |
|---|---|---|
| CEM 可学到可解释参数 | 支持 | best params 呈现低开局、慢前期、快后期、折价 seller quote |
| CEM 训练集 reward 高于 hand-written | 支持 | train-smoke best reward 0.6536 |
| CEM held-out smoke/full 有效 | 未验证 | full test 尚未完成 |
| 继续同一 16 scenario refine 有必要 | 暂不支持 | 过拟合风险高，应先 held-out |

---

## 8. 主要问题与失败模式

### 8.1 Simple Env 过拟合风险

Simple Env 是：

```text
single buyer
single seller
single product
single issue: price
fixed max_turns = 6
```

这种环境适合快速验证 bargaining mechanics，但不足以证明 universal negotiation ability。

### 8.2 belief 仍然脆弱

当前 belief update 虽然不再完全相信 seller quote，但仍可能：

- 被 seller high anchor 带偏；
- 对 seller finality 的真假缺乏校准；
- 对 cheap items 与 expensive items 使用相似规则；
- 将 cost-like reservation 与 strategic reservation 混在一起。

### 8.3 planner 需要学习

traj 显示 planner 常见失败：

- 首轮太低，seller quit；
- 后期太保守，成交价过高；
- belief p50 偏移时 planner 不知道如何抗噪；
- accept threshold 与 concession curve 很难手调。

### 8.4 实验运行稳定性

目前 vLLM 断开/timeout 仍会导致长实验中断。需要：

- 更稳定的 tmux/screen vLLM；
- 更明确的 resume/retry；
- 输出目录避免混乱；
- 检查 summary 是否与 episodes 文件一致。

---

## 9. 下一步计划

### 9.1 Simple Env 数据收集与训练

优先级最高：

1. 用 CEM refine best 进行 paired held-out smoke：

```text
continuous_rule_ev_conservative
continuous_rule_ev_current
continuous_rule_ev_cem
```

2. 如果 smoke 支持，再跑 full：

```text
128 scenarios × 4 rollouts
```

3. 收集 decision-level data：

```text
history
belief state
posterior summary
candidate offers
selected action
seller response
final reward
seller quit / deal / overshoot
```

4. 用这些数据训练：

| 训练对象 | 数据 | 目标 |
|---|---|---|
| planner/reranker | CEM elite trajectories | 学习 candidate action ranking |
| belief calibration model | predicted accept vs actual seller response | 校准 p_accept / p_quit |
| update-strength model | seller utterance + action + outcome | 学习 rule update weight |

### 9.2 Universal benchmark 调研与迁移

当前 Simple Env 结果可能过于特化。下一阶段应调研并迁移到更 universal、更多论文可接受的 benchmark。候选方向：

| benchmark / env | 为什么重要 | 计划 |
|---|---|---|
| AgenticPay | 已在本项目中实现 multi buyer/seller/product probe；更接近复杂商业谈判 | 优先迁移 continuous belief + CEM planner |
| Terms-Bench / Terms Env | multi-issue contract，可测 belief calibration / Brier | 保留为中等难度统一 benchmark |
| ASTRA / CasiNo | 多 issue resource allocation，有 partner/persona 和原论文 baseline | 用于和 negotiation literature 对齐 |
| Deal-or-No-Deal / bilateral trade variants | 经典 negotiation benchmark，更易被读者理解 | 作为补充 sanity benchmark |
| 其他 multi-agent bargaining benchmark | 需要文献调研确认使用频率、代码可得性、metrics | 作为下一阶段 literature survey |

### 9.3 推荐下一阶段执行顺序

```text
Step 1: 修复/确认 vLLM 稳定运行
Step 2: CEM best paired smoke in Simple Env
Step 3: CEM best full eval in Simple Env
Step 4: 从 full eval 构建 planner/reranker training data
Step 5: 调研并选择 universal benchmark
Step 6: 将 continuous solver 迁移到 AgenticPay 或更强 multi-agent benchmark
Step 7: 比较 prompt/full framework/continuous/CEM/learned planner
```

---

## 10. 当前最短结论

当前版本已经完成了一个可运行、可解释、可学习的 negotiation pipeline：

```text
fixed anchor+concession template
+ persistent belief posterior
+ EV planner
+ CEM-tuned concession parameters
+ strong-format generator
```

CEM 在 Simple Env search set 上找到了 promising 参数，说明“planner schedule 需要 learning”这一判断是合理的。但它尚未完成 held-out/full validation；而且 Simple Env 本身过于简单，下一阶段必须迁移到更 universal 的 multi-agent benchmark，尤其是 AgenticPay 或其他 multi buyer / multi seller / multi product / multi issue 环境。

因此，当前研究状态不是“方法已经证明有效”，而是：

> 我们已经有了一个比 prompt-only 更结构化、更可学习的 negotiation agent 形态，并获得了 planner learning 的积极信号；下一步需要用 held-out evaluation 与更 universal benchmark 检验它是否真正泛化。
