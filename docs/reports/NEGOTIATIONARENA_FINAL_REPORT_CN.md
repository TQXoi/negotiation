# NegotiationArena 最终汇报总结

更新日期：2026-08-14
模型：`Qwen3-30B-A3B-Instruct-2507-base`
核心协议：Opponent Simulation 论文对齐的 repeated bilateral negotiation
状态：baseline 主矩阵、V7/V8 confirmatory、belief interventions 与 private-preference switch 均已完成

---

## 0. Executive Summary

NegotiationArena 部分已经完成两个自然语言双边博弈、四个 focal role 的正式测试：

1. Buyer--Seller：分别测试 focal buyer 与 focal seller；
2. Resource Exchange：分别测试 focal RED first mover 与 focal BLUE second mover。

最重要的结果不是“一个统一 framework 全面超过所有 baseline”，而是：

- 同批次严格比较中，我们在 **focal Buyer** 上以 V3.3 得到 `13.11`，比
  Opponent Simulation 的 `12.76` 高 `+0.35`；
- 在 **Resource-second** 上以 V3.3 得到 `22.47`，比 Opponent Simulation 的
  `11.88` 高 `+10.59`；
- 在 **focal Seller** 上低 `−2.18`；
- 在 **Resource-first** 上低 `−22.12`，Opponent Simulation 在主动 opening
  场景仍有压倒性优势；
- 不应跨四个 setting 求简单平均，因为价格游戏总 surplus 上限为 20，而资源游戏
  的 utility scale 不同。

因此，没有单一“全局最优 reward framework”：

- **旧矩阵中最稳定的性能版本是 V3.3**，尤其 Buyer 与 Resource-second；
- **Resource-first 的旧矩阵最好版本是 V5**；
- **当前最终方法版本是 V8**，它的主要价值是 factorized belief、evidence
  reliability gate 和 causal evaluation，而不是 universal payoff SOTA。

关于 belief 是否有用，最终结论必须分层：

1. **Preference belief 能因果改变 planner action，且正确 belief 存在 utility
   headroom**：same-state action flip、wrong/shuffled/oracle intervention 均提供证据；
2. **Learned belief 只在部分场景有用**：Resource-first V8 高于 shuffled `+4.245`，
   但 Resource-second learned belief 不胜 shuffled/frozen；
3. **Response-policy belief 尚未证明有用**：policy-only intervention 的在线 action
   flip 接近零；
4. **Continuous cross-episode update 尚未证明优于 frozen/no-cross**：stationary 和
   private-preference switch 的主要区间均无法稳定排除零；
5. **Calibration 不等于 negotiation utility**：V5/V7 的 Brier/ECE 改善并未在所有
   setting 带来 reward 提升。

最准确的论文式结论是：

> 当前工作证明了 opponent preference belief 可以被 executable planner 因果使用，
> 并提出了 wrong/shuffled/oracle/frozen/no-cross、same-state action flip、calibration
> 与真实 preference switch 组成的可审计评估协议；但尚未证明 learned continuous
> belief 在所有谈判角色上带来稳定 payoff advantage。

---

## 1. 测试环境与四个 focal setting

### 1.1 Focal 的含义

`focal agent` 是当前被评估、其策略被替换为 Direct、Opponent Simulation 或我们的
framework 的一方。另一方保持 matched opponent。它不是第三个 agent。

### 1.2 四个 setting

| Setting | Game | Focal | 先后手 | 私有效用/奖励 |
|---|---|---|---|---|
| `buyer` | Buyer--Seller | BLUE buyer | RED seller 先，focal buyer 后 | WTP=63，reward=`63-price` |
| `seller` | Buyer--Seller | RED seller | BLUE buyer 先，focal seller 后 | cost=43，reward=`price-43` |
| `resource_first` | Resource Exchange | RED | focal RED 先 | RED: X=0.5, Y=2.5 |
| `resource_second` | Resource Exchange | BLUE | RED 先，focal BLUE 后 | BLUE: X=2.5, Y=0.5 |

价格游戏中总可分 surplus 为 `63−43=20`；不成交双方 reward 为 0。

### 1.3 Resource Exchange

| 属性 | RED | BLUE |
|---|---:|---:|
| 初始 X | 25 | 5 |
| 初始 Y | 5 | 25 |
| 每单位 X 的价值 | 0.5 | 2.5 |
| 每单位 Y 的价值 | 2.5 | 0.5 |
| 更希望获得 | Y | X |

双方偏好互补。例如 RED 给 4 X、BLUE 给 4 Y：

- RED 净收益：`4×2.5−4×0.5=8`；
- BLUE 净收益：`4×2.5−4×0.5=8`；
- joint reward：16。

资源 reward 是成交后相对初始持有的私有加性价值增量；没有成交为 0。

`resource_first` 与 `resource_second` 不是两套资源环境：底层 utility、初始资源和协议
相同，只是选择不同 focal side。

- `resource_first`：我们的算法控制 RED，RED 在尚未看到本局对手行为时先提 bundle；
- `resource_second`：固定 RED 先提 bundle，我们的算法控制 BLUE 并 accept/counter。

后手 BLUE 已经观察到一个正式 proposal，因此当前 belief-aware planner 更容易发挥。
先手 RED 则主要依赖跨局 prior、opening planning 或主动 probe，Opponent Simulation
可以通过昂贵 rollout 评估多个 opening，因此在这个 setting 特别强。

注意：`resource exchange` 不是 `resource change`。真正的 change 实验是第 11 episode
改变对手 private X/Y utility 的 `opponent preference switch`。

### 1.4 Repeated protocol

- 每个 run 含 20 个连续 episodes；
- 同一 run 内双方保留公开 repeated history；
- framework 可保留跨 episode posterior；
- 每局最多 10 turns；
- 每个正式 cell 为 5 independent runs × 20 episodes；
- 统计推断应以 5 条 run-level trajectory 为单位，不能把 100 episodes 当成 100 个
  独立 i.i.d. 样本。

---

## 2. Baselines 与 Framework Variants

### 2.1 Baselines

| Method | 核心机制 | 每个真实动作的 LLM 计算 | 定位 |
|---|---|---|---|
| Direct | matched repeated prompt 直接生成协议动作 | 常规 actor call | 最低成本基线 |
| Opponent Simulation（旧） | 联合生成候选和压缩模拟 | 约 3 calls | 历史本地近似 |
| Opponent Simulation（paper-aligned） | 5 个独立 action samples + 每个候选独立 future rollout | 最多约 10 calls | 主要直接竞争 baseline |

Paper-aligned 版本是根据论文 Algorithm 1、released prompts 和 logs 的重实现，不是作者
发布的完整官方 runner；每条未来轨迹仍由一次 serialized request 产生。

### 2.2 Framework 演化

| Variant | 主要改动 | 当前定位 |
|---|---|---|
| V2 | structured posterior、exact self utility、candidate planner、reciprocity ledger、action lock | framework 基础 |
| V3 | posterior adverse-tail safeguard、agreement opportunity cost | 风险感知 belief utilization |
| V3.1 | repeated commitment floor | 防止跨局报价逐渐恶化 |
| V3.2 | 只为 resource-first opening 放宽 proposal frontier | scoped opening 修复 |
| V3.3 | 按价格/组合资源 action space 调整 frontier | 旧矩阵最稳定版本 |
| V4 | generic cross-episode information bonus | 会为无用信息支付机会成本 |
| V5 | 降低 endogenous、主动 proposal、重复 evidence 权重 | calibration 改进版本 |
| V6 | bounded decision-relevant VOI | 离线安全，但在线几乎不 probe |
| V7 | factorized `p(theta,phi)` + particle regret | 分开 preference 与 response policy |
| V8 | V7 + evidence-channel reliability gate | 当前最终方法版本 |

V7/V8 中：

- `theta`：opponent preference。价格游戏是 reservation value，资源游戏是 X/Y
  relative utility weight；
- `phi`：opponent response policy，包括 rationality scale、acceptance bias、counter
  propensity；
- planner 在 joint particles 上计算 action value 与 regret；
- LLM 只 verbalize locked structured action，不能改变价格/bundle/ACCEPT 决定。

### 2.3 Belief intervention variants

| Variant suffix | 实际含义 | 回答的问题 |
|---|---|---|
| normal/continuous | 正常在线更新，并允许跨局 carryover | 主方法表现如何？ |
| `frozen` | 限制持续更新；历史实现仍可能带有部分 meta carryover | continuous 是否必要？ |
| `wrong_confident` | 注入故意错误且高置信度的 belief | 错 belief 会否被 planner 放大？ |
| `shuffled` | 打乱 belief 内容、尽量保留分布形态 | 正确 belief 内容是否重要？ |
| `oracle` | evaluator truth 供 planner 使用 | 正确 belief 的可用上界有多大？ |
| `policy_uniform` | 仅把 response-policy posterior 设为 uniform | learned phi 是否有用？ |
| `policy_shuffled` | 仅打乱 response-policy posterior | action 是否依赖 phi 内容？ |
| `no_cross_episode` | 每局从同一 uniform prior 开始，禁止跨局 carryover | 严格 continuous-vs-no-memory 对照 |

`oracle`、`wrong`、`shuffled` 是因果诊断，不是可以与 baseline 并列宣传的部署方法。

---

## 3. 代码在哪里查看

### 3.1 统一实验入口

- `external_negotiation_envs/NegotiationArena/arena_integration/run_opponent_simulation_setting.py`
  - `parse_args`：注册所有 method/setting；
  - `make_agent`：method 名称到 agent class 的映射；
  - `focal_index`：buyer/resource-second 选择 BLUE，其余选择 RED；
  - `resource_valuations`：evaluator-side RED/BLUE private utility；
  - `make_game`：构造 game、focal 与 first/second mover；
  - `summarize`：正式指标汇总。

### 3.2 Baseline

- Direct 与共享 repeated-memory/protocol 基类：
  `arena_integration/repeated_agents.py::RepeatedLanguageAgent`
- 历史 Opponent Simulation：
  `arena_integration/repeated_agents.py::OpponentSimulationAgent`
- paper-aligned Opponent Simulation：
  `arena_integration/paper_aligned_opponent_simulation.py::PaperAlignedOpponentSimulationAgent`

### 3.3 Framework 继承链

主文件：

`external_negotiation_envs/NegotiationArena/arena_integration/decision_calibrated_agent.py`

| 版本 | Class |
|---|---|
| V2 | `DecisionCalibratedBeliefPlannerAgent` |
| V3 | `UncertaintySafeguardedBeliefPlannerAgent` |
| V3.1 | `ReciprocalCommitmentBeliefPlannerAgent` |
| V3.2 | `ScopedCommitmentBeliefPlannerAgent` |
| V3.3 | `ActionSpaceAdaptiveCommitmentBeliefPlannerAgent` |
| V4 | `CrossEpisodeInformationPlannerAgent` |
| V5 | `EndogeneityCalibratedInformationPlannerAgent` |
| V6 | `DecisionRelevantInformationPlannerAgent` |
| V7 | `FactorizedPreferencePolicyBeliefPlannerAgent` |
| V8 | `ReliabilityGatedFactorizedBeliefPlannerAgent` |

当前代码已增加中文阅读导引和每个版本的差分注释。由于 subclass 只覆盖相对上一版变化，
建议先读 V2，再依次跳到 V3.3、V5、V7、V8。

### 3.4 Game 与 evaluator

- repeated paper setting：`arena_integration/repeated_games.py`
- 原始资源交换：`games/trading_game/game.py`
- 原始价格谈判：`games/buy_sell_game/game.py`

### 3.5 版本记录与结果

- V3.3：`research_iterations/iteration_004_v3_3_action_space_adaptive/`
- V4：`research_iterations/iteration_005_v4_cross_episode_information/`
- V5：`research_iterations/iteration_006_v5_endogeneity_calibration/`
- V6：`research_iterations/iteration_007_v6_decision_relevant_voi/`
- V7：`research_iterations/iteration_008_v7_factorized_preference_policy/`
- V8：`research_iterations/iteration_009_v8_reliability_gated_belief/`
- preference switch：`research_iterations/iteration_010_preference_switch/`
- strict identifiable switch：`research_iterations/iteration_011_identifiable_switch/`

具体 cell 保存 `config.json`、`episodes.jsonl`、`summary.json`、每次 framework
decision、所有 model calls、`game_state.json` 和 `interaction.log`。

---

## 4. 与 Direct / Opponent Simulation 的严格同批次结果

旧正式主矩阵所有方法使用 matched game、opponent、model 和 protocol。每个 cell 为
5 runs × 20 episodes，当前错误为 0。表中为：

`focal reward / agreement rate / joint reward`

| Setting | Direct | Opponent Simulation | V3.3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|
| Buyer | 7.77 / .99 / 19.80 | 12.76 / 1.00 / 20.00 | **13.11** / .95 / 19.00 | 10.65 / .99 / 19.80 | 7.95 / 1.00 / 20.00 |
| Seller | 18.08 / 1.00 / 20.00 | **19.13** / .96 / 19.20 | 16.95 / 1.00 / 20.00 | 16.32 / 1.00 / 20.00 | 15.92 / .99 / 19.80 |
| Resource-first | 12.34 / .94 / 17.36 | **40.57** / .96 / 48.08 | 14.55 / .98 / 34.20 | 12.80 / .98 / 31.96 | 18.45 / .99 / 37.32 |
| Resource-second | 9.75 / .93 / 29.76 | 11.88 / .97 / 30.74 | **22.47** / .96 / 47.90 | 21.55 / 1.00 / 42.82 | 21.90 / .98 / 39.36 |

### 4.1 Best-per-setting 与 Opponent Simulation 差值

| Setting | 我们同批次最好 | Opponent Simulation | Focal reward delta | 判断 |
|---|---:|---:|---:|---|
| Buyer | V3.3: 13.11 | 12.76 | **+0.35** | 小幅领先，但 agreement 低 .05 |
| Seller | V3.3: 16.95 | 19.13 | **−2.18** | 未超过 |
| Resource-first | V5: 18.45 | 40.57 | **−22.12** | 显著失败条件 |
| Resource-second | V3.3: 22.47 | 11.88 | **+10.59** | 明显领先 |

不能写“我们的 framework 平均提升 X”，因为四行 reward 不在同一 utility scale，而且
best-per-setting 是事后选择不同版本。可合法陈述为：四个 setting 中 focal reward 在
两个 setting 超过 Opponent Simulation，其中 Resource-second 的优势最大；Seller 与
Resource-first 未超过。

### 4.2 为什么 Resource-first 的 Opponent Simulation 很强

Resource-first 要求 focal RED 在没有本局对手 proposal 的情况下主动选择组合 bundle。
Opponent Simulation 用多次 inference-time rollout 评估多个 opening；我们的 framework
主要从已观察 evidence 更新 posterior，因此 opening 仍容易受到 prior/likelihood
misspecification 影响。

Resource-second 中 focal BLUE 先观察 RED proposal，正式行为 evidence 可以直接进入
preference belief 和 accept/counter planner，这更符合当前方法的强项。

---

## 5. V8 正式因果矩阵

V8 confirmatory 使用同一新 seed 比较 V5、V7、V8 及 V8 belief interventions。
每个 cell 100 episodes、0 errors：

| Setting | V5 | V7 | V8 | V8-frozen | V8-wrong | V8-shuffled | V8-oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buyer | 8.880 | 8.560 | 7.240 | 7.490 | 8.210 | 7.940 | **12.710** |
| Seller | 14.360 | 15.180 | 11.120 | 13.750 | 14.130 | 11.560 | **17.330** |
| Resource-first | 18.455 | 18.460 | 22.265 | **22.500** | 18.660 | 18.020 | 22.445 |
| Resource-second | 24.420 | 21.875 | 21.675 | 25.125 | 20.055 | **25.665** | 25.655 |

### 5.1 V8 的正向结果

- Resource-first：V8 相对 V5 为 `+3.810`；五个 paired runs 全部为正，描述性
  95% t interval `[1.492, 6.128]`；agreement `.99`，joint reward `40.54`；
- Resource-first：V8 比 shuffled 高 `+4.245`，说明正常 belief 内容在这个 setting
  不只是添加一个固定计算模板；
- V8-wrong 的 Resource-first reward 为 `18.660`，而旧 V7-wrong 曾降到约 `7.0`，
  reliability gate 显著限制了错误高置信度 belief 的 downside；
- Buyer/Seller 的 oracle 分别比 V8 高 `+5.470/+6.210`，说明正确 reservation
  information 确实能被当前 planner 使用。

### 5.2 V8 不能支持的主张

- V8 不是 universal payoff improvement：Buyer、Seller 和 Resource-second 都没有
  成为最优 normal learned method；
- stationary opponent 中 continuous 不优于 frozen：Resource-first `22.265 vs
  22.500`，Resource-second `21.675 vs 25.125`；
- Resource-second learned belief 不胜 shuffled：`21.675 vs 25.665`；
- policy-uniform/policy-shuffled 的 same-state action flip 接近零，不能声称 learned
  response policy 已产生稳定 utility advantage；
- oracle 的 reward 是 upper bound，不是正常 framework 成绩。

---

## 6. Belief 到底有用吗？

### 6.1 Belief 会改变动作吗？

会，但主要是 preference belief。

在相同 public state、相同候选动作、相同 planner 下，替换 normal/frozen/wrong/
shuffled/oracle posterior 会造成可记录的 action flip。Oracle/preference intervention
的 flip 明显；policy-only intervention 接近零。

因此可以说：

- `theta → planner action` 因果通道存在；
- `phi → planner action` 通道尚未有效建立。

### 6.2 正确 belief 会提高收益吗？

部分成立：

- Resource-first：normal V8 > shuffled；
- Buyer/Seller：oracle 明显高于 normal；
- wrong-confident 能造成损害，证明 planner 会真实使用 belief，而不是忽略 belief。

但这不代表 learned belief 在所有 setting 正确：Resource-second shuffled/frozen 高于
normal，说明 estimator 或 belief-to-action mapping 仍会出错。

### 6.3 Calibration 改善等于收益改善吗？

不等于。V4→V5 的 Brier/ECE 在四个 setting 全部改善，但 reward 只有
Resource-first 有清楚提高；Buyer 反而从 `10.65` 降到 `7.95`。

这说明 prediction calibration 是必要诊断，但不是 negotiation utility 的充分条件。

### 6.4 Continuous update 有用吗？

目前证据不足。

- stationary setting 中 continuous 未稳定超过 frozen；
- 最终 away-from-prior preference-switch × strict no-cross 实验中：
  - Resource-first continuous−no-cross reward DiD：`+0.48`，区间
    `[−12.60, 13.56]`；
  - Resource-second：`+6.83`，区间 `[−8.85, 22.51]`；
- 两个区间都无法在 5-run 设计下排除零；
- Resource-second 虽有正 payoff signal，但 post-switch preference MAE 反而变差。

因此不能声称当前 continuous updater 已正确跟踪 hidden preference change。

### 6.5 最终 belief 结论

| 问题 | 结论 |
|---|---|
| Preference belief 是否进入 action？ | 是，有 same-state causal evidence |
| 正确 belief 是否存在 planner-usable headroom？ | 是，oracle 明显提高部分角色 reward |
| Learned belief 是否总比 shuffled/frozen 好？ | 否 |
| Response-policy belief 是否已有效？ | 尚未证明 |
| Continuous cross-episode update 是否已有效？ | 尚未证明 |
| Reliability gate 是否有用？ | 有，能限制 wrong-confident downside |

---

## 7. “最优 Framework”应当如何表述

### 7.1 如果指同批次 payoff

- Buyer：V3.3；
- Seller：V3.3，但低于两条 baseline；
- Resource-first：V5，但远低于 Opponent Simulation；
- Resource-second：V3.3。

不存在统一 reward winner。

### 7.2 如果指当前最终方法

V8 是当前最终 scientific framework，因为它包含完整的方法贡献：

1. continuous factorized belief `p(theta,phi)`；
2. exact-utility executable planner 与 action lock；
3. evidence provenance reliability gate；
4. wrong/shuffled/oracle/frozen/no-cross same-state intervention；
5. calibration、action flip 与真实 private-preference switch evaluation。

V8 应被描述为“最完整、最可审计的方法版本”，而不是“所有 setting reward 最高”。

### 7.3 推荐论文主张

不推荐：

> Our framework universally outperforms Opponent Simulation.

推荐：

> We introduce a causally auditable belief-usable negotiation framework that
> factorizes opponent preferences and response policy, locks structured planner
> actions, and bounds unreliable belief influence. It substantially improves
> focal utility over Opponent Simulation when responding to combinatorial offers,
> while controlled interventions reveal when learned beliefs help, fail, or are
> ignored by the planner.

---

## 8. 局限与下一步

### 8.1 当前局限

1. 正式 NegotiationArena 只覆盖 Buyer--Seller 与 Resource Exchange 两个 bilateral
   game family；
2. 每个 cell 只有 5 independent runs，switch 的区间较宽；
3. 只有少量固定 private utility profiles，behavioral identifiability 不充分；
4. first-mover opening 仍显著弱于高计算成本的 Opponent Simulation；
5. response-policy posterior 对动作影响过弱；
6. learned continuous belief 在 change 后未显示正确 signed adaptation。

### 8.2 下一步实验顺序

1. 冻结 planner reward weights，不再基于当前 confirmatory result 调参；
2. 在 Simple Env 构建 oracle-verified behavioral-identifiability utility-profile suite；
3. 让 active probe 最大化 decision-relevant information，而不是通用 entropy；
4. 使用 old/new regime mixture 与 explicit change-point hazard；
5. 同时报告 signed belief error、Brier/NLL/ECE、action flip、decision regret、agreement
   和 reward；
6. 机制通过后迁移回 CaSiNo，检验 multi-issue natural-language preference inference；
7. NegotiationArena 保留为 Opponent Simulation 对比与 controlled causal intervention
   环境。

---

## 9. 最终结论

NegotiationArena 已经完成了一个比“只看最终 reward”更严格的 opponent-belief 研究闭环：

```text
public language/action evidence
    → factorized belief over preference and response policy
    → reliability-gated posterior
    → structured candidate evaluation
    → locked negotiation action
    → same-state intervention / calibration / switch audit
```

实验证明：belief 不是完全无用的 prompt decoration；preference belief 能改变动作，
oracle 有明确 headroom，错误高置信度 belief 会伤害 utility，而 V8 gate 能限制这种损害。
但 learned continuous update、response-policy inference 和 first-mover exploration 尚未达到
可以普遍超过 Opponent Simulation 的程度。

当前最可信的贡献是：**一套具有 executable belief-to-action chain、reliability-bounded
planning 和 causal belief audit 的 negotiation framework，以及对 belief 何时有用、何时
失败的系统实证结论。**
