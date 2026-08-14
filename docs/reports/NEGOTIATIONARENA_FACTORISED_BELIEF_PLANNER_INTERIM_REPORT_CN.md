# NegotiationArena Factorized Belief–Planner 中期报告

更新日期：2026-08-13

代码库：`NegotiationArena`

本文是一份可编辑的中期技术报告。它严格区分三种证据状态：

1. **正式完成**：旧主矩阵 36/36 cells、opponent-switch 矩阵 20/20 cells，均为每个 cell 5 runs × 20 episodes，且 0 current errors；
2. **阶段完成**：V6/V7 的离线回放、代码测试、2×20 smoke 和 resource-first 定向干预；
3. **仍在运行**：V7 新 seed 的 5×20 confirmatory matrix。本文最近一次结果快照为 18/32 cells、1800/3200 episodes、0 current errors；Buyer 与 Seller 已全部完成，resource-first 的 V5/V7 也已完整落盘。

因此，本文中的旧主矩阵结论可以作为正式结果；V7 Buyer 与 Seller 各 8 个 cells 已完整，resource-first 的 V5/V7 对照已完整；其余 resource interventions 与 resource-second 尚未完成，V7 跨四个 setting 的最终结论仍需等待全部矩阵完成。

---

## 一、当前工作的核心目标

我们的目标不只是提高谈判 reward，而是回答一条更严格的因果链：

```text
自然语言交互证据
    ↓
对手 belief 是否被正确更新？
    ↓
belief 是否改变 planner 的 structured action？
    ↓
action change 是否沿正确方向提高 agreement / focal utility / joint utility？
    ↓
对手发生变化时，continuous belief 是否比 frozen belief 更快恢复？
```

现有很多 opponent-modeling 工作只验证预测、生成的对手描述或最终 reward。我们的工作希望把中间环节显式化、可审计化，并用 frozen、wrong、shuffled、oracle 和 action-flip 等干预逐层验证，而不是把所有改进都归因于“模型更理解对手”。

当前最重要的研究判断是：

> **calibrated prediction、belief-sensitive action 和 realized negotiation utility 是三个不同问题。**

正式实验已经显示，acceptance Brier/ECE 下降并不自动带来 reward 提升；continuous update 也不自动优于 frozen。因此，新方法不能只继续训练一个更准的 belief estimator，而必须同时修复 belief 的可辨识性和 planner 使用 belief 的方式。

---

## 二、NegotiationArena 测试 setting

### 2.1 四个角色条件

当前使用 Opponent Simulation 论文对应的 repeated negotiation setting，并统一成四个 focal role：

| Setting | 游戏 | Focal 位置 | 私有效用 | 主要测试内容 |
|---|---|---|---|---|
| `buyer` | 单商品价格谈判 | Buyer，后手 | WTP = 63；reward = 63 − price | 隐藏 seller cost 的 reservation inference、accept/counter |
| `seller` | 单商品价格谈判 | Seller，后手 | cost = 43；reward = price − 43 | 隐藏 buyer WTP 的 reservation inference、价格锚定 |
| `resource_first` | 两资源组合交换 | RED，先手 | RED: X=0.5, Y=2.5 | 大组合 action space 中的 opening/proposal planning |
| `resource_second` | 两资源组合交换 | BLUE，后手 | BLUE: X=2.5, Y=0.5 | 从对手 proposal 学习偏好并作 accept/counter |

资源初始持有为：

- RED：X=25, Y=5；
- BLUE：X=5, Y=25。

资源 reward 是成交后相对初始持有的私有加性价值变化；未成交为 0。价格谈判中 seller cost=43、buyer WTP=63，总 surplus 上限为 20。

### 2.2 repeated protocol

每个 run 包含 20 个连续 episodes。同一 run 内：

- focal agent 保留跨 episode 的 public memory 和 framework belief；
- 对手也保留 repeated interaction memory；
- 每个 episode 最多 10 turns；
- focal 始终是后手（价格游戏），资源游戏分别测试先手和后手；
- 5 个 run 使用不同派生随机种子，run 才是主要统计单位。

需要强调：100 episodes 不是 100 个独立样本，而是 5 条长度为 20 的 repeated trajectories。最终显著性、置信区间和 paired comparison 应以 run-level 为主，不能把 100 episodes 当作独立 i.i.d. 样本夸大显著性。

### 2.3 当前正式配置

V7 confirmatory 使用：

| 配置项 | 值 |
|---|---|
| 模型 | `Qwen3-30B-A3B-Instruct-2507-base` |
| 服务 | `http://127.0.0.1:8002/v1` |
| Runs | 5 |
| Episodes / run | 20 |
| Max turns | 10 |
| Candidate count | 5 |
| Temperature | 0.7 |
| Max completion tokens | 1600 |
| Context character limit | 30000 |
| Opponent policy | released `strategic_brainstorming` prompt |
| Protocol | `normalize_retry`，最多 1 次 repair |
| Confirmatory seed | 20260900 |

所有当前 V7 variants 使用相同游戏、对手、模型、prompt、temperature、seed policy 和 parser-safe protocol。V7 belief/planner 本身没有增加额外的 LLM rollout 或 chooser call。

### 2.4 与原论文复现的边界

`opponent_simulation_paper` 是当前仓库中更接近论文 Algorithm 1 的重实现：

1. 独立采样 5 个 actor actions；
2. 对每个 action 运行独立 opponent-conditioned serialized rollout；
3. 按 predicted focal reward 选择 action。

但论文只发布了 prompt 和示例 log，没有发布完整官方 runner。因此它应标为 **paper-aligned reimplementation**，不是 bit-exact official reproduction。当前实现把每条未来 trajectory 序列化在一次 evaluator request 中，而不是启动完整外部交替模拟器。

---

## 三、当前代码结构

### 3.1 总体结构

```text
NegotiationArena/
├── negotiationarena/                         # 原环境核心：agent/game/parser/logging
├── games/
│   ├── buy_sell_game/                        # 价格谈判协议与 parser
│   └── trading_game/                         # 资源交换协议与 parser
├── arena_integration/
│   ├── repeated_games.py                     # repeated game、效用、first mover、安全终止
│   ├── repeated_agents.py                    # direct agent、legacy OppSim、LLM I/O、protocol repair
│   ├── paper_aligned_opponent_simulation.py  # 论文对齐的 BoN=5 Opponent Simulation
│   ├── belief.py                             # 早期 reservation belief 数据结构
│   ├── framework_agent.py                    # 早期 framework 原型
│   ├── decision_calibrated_agent.py          # V2–V7 belief/planner 主实现
│   └── run_opponent_simulation_setting.py    # 统一四 setting runner、resume、summary
├── scripts/
│   ├── run_formal_belief_matrix_qwen30b.sh
│   ├── run_opponent_switch_matrix_qwen30b.sh
│   ├── summarize_formal_belief_matrix.py
│   └── monitor_formal_experiments.py
├── tests/
│   └── test_repeated_integration_unittest.py # 当前 47 个 integration tests
├── research_iterations/
│   ├── iteration_007_v6_decision_relevant_voi/
│   └── iteration_008_v7_factorized_preference_policy/
└── arena_runs/                               # baseline、正式矩阵与全量轨迹
```

### 3.2 一次 episode 的执行流

```text
run_opponent_simulation_setting.py
    ├── 创建 focal method 和 matched opponent
    ├── prepare_episode（保留同 run 的 repeated state）
    ├── repeated_games.py 构造游戏与私有目标
    ├── alternating game 驱动自然语言交互
    ├── protocol canonicalization / retry
    ├── 环境计算 agreement、deal、focal/opponent/joint reward
    ├── framework 输出 calibration、posterior、action-flip diagnostics
    └── 逐 episode append 到 JSONL 并更新 summary.json
```

### 3.3 framework 内部执行流

V7 当前的核心流程为：

```text
public response / opponent proposal / bounded semantic evidence
    ↓
joint posterior update b_t(theta, phi)
    ├── theta: 对手 preference / reservation / resource weight
    └── phi: rationality scale / acceptance bias / counter propensity
    ↓
枚举合法 structured proposals
    ↓
计算每个 joint particle 下的 action value Q(a; theta, phi)
    ↓
posterior mean value − uncertainty-weighted upper-tail particle regret
    ↓
与 observed ACCEPT、outside option、commitment floor 比较
    ↓
action lock：最终自然语言只能实现 planner 已选的合法 action
```

V7 的 response-policy latent space 为 27 个离散 particles，即三个 rationality scales × 三个 acceptance biases × 三个 counter propensities。Preference particles 在价格游戏中是 reservation-value grid，在资源交换中是可解释的 resource-weight grid。

### 3.4 保存的调用与审计记录

每个 cell 保存：

```text
<setting>/<method>/
├── config.json
├── episodes.jsonl
├── summary.json
├── arena_logs/run_XX/episode_YY/
│   ├── game_state.json
│   └── interaction.log
└── model_traces/run_XX/<player>/
    ├── calls.jsonl
    ├── protocol_repairs.jsonl
    ├── framework_v5_decisions.jsonl
    └── framework_v7_decisions.jsonl
```

`episodes.jsonl` 是 append-only；旧错误不会被删除。监控与最终统计按 `(run, episode)` 取最新记录，因此恢复实验仍保留全部历史调用和修改轨迹。

一个需要修复的工程风险是：出现 error 的 run 会整 run 重跑，belief history 一致；但如果进程在未写 error 的情况下于 run 中途被杀，当前 `--resume` 只重建公开 episode records，不会精确恢复内部 posterior。当前 confirmatory 一直连续运行，结果不受影响；后续恢复器应将任何 incomplete run 也标记为 whole-run restart。

---

## 四、baseline 与 framework 版本关系

| Method | 作用 | LLM 计算特征 | 当前定位 |
|---|---|---|---|
| Direct | matched repeated prompting，无显式 opponent model | 常规 actor call | 最低成本基线 |
| Opponent Simulation | 5 actor candidates + 5 opponent-conditioned rollout evaluations | 显著增加 inference-time calls | 直接竞争 baseline |
| V2 | structured posterior、candidate planner、action lock | belief/planning 主要为结构化计算 | framework 基础 |
| V3 | posterior tail safeguard、agreement opportunity cost | 不增加 rollout | 风险感知 planner |
| V3.1–V3.3 | repeated commitment、按 action-space 调整 proposal frontier | 不增加 rollout | 旧框架中表现最稳定版本 |
| V4 | generic cross-episode information bonus | 不增加 rollout | 探索版本，但会为无用信息付费 |
| V5 | 下调 endogenous/repeated evidence 权重 | 不增加 rollout | calibration 改进版本 |
| V6 | accept/counter/reject shadow update + bounded decision-relevant VOI | deterministic shadow，无额外 LLM | 机制安全，但在线几乎不触发 probe |
| V7 | factorized preference–policy joint belief + particle-regret planner | 无额外 rollout/chooser | 当前确认性实验方法 |

### V3.3

V3.3 根据 action space 调整 proposal frontier：标量价格谈判保留 response-supported candidates；组合资源交换允许更宽的 exploitation frontier，避免只看局部 acceptance 支持而删除互补 bundle。

### V4

V4 用下式为跨 episode 探索付费：

```text
one-step information gain × sqrt(remaining episodes) × posterior entropy
```

问题是它只奖励“不确定性”，不判断信息是否会改变未来最优 action，因而可能放弃确定的正效用 deal。

### V5

V5 修复 evidence endogeneity：

- 对手对我方 locked offer 的 ACCEPT/REJECT 是强 evidence；
- COUNTER 较弱；
- 对手主动 proposal 只是弱 preference evidence；
- 同一 episode 重复相似证据递减。

V5 确实改善 calibration，但没有稳定改善 reward。

### V6

V6 只允许 decision-relevant information exploration：只有可能 response 会改变未来 structured action，且期望价值能覆盖当前 deal 的机会成本，才允许 probe。离线回放阻止了 20/21 个旧 probe，但在线 smoke 中执行 0 个 probe，所以 V6 没有成为有效的新策略版本。

### V7

V7 修复 V5/V6 没有解决的结构问题：把“对手想要什么”与“对手如何战略性回应”分开建模。

```text
b_t(theta, phi) ∝ b_{t-1}(theta, phi) · p(y_t | offer_t, theta, phi)
```

其中 `theta` 是 preference，`phi` 是 response policy。Planner 不生成一段自由文本 opponent profile，而是对每个合法 action 计算 posterior particle value 与 regret：

```text
BUS(a) = E_b[Q(a)] − lambda(b) · normalized-CVaR_0.9(regret(a))
```

这使 belief 的使用可被 action-flip 和 intervention 直接审计。

---

## 五、已经完成的正式 baseline 比较

以下结果来自旧正式主矩阵：每个 cell 5 runs × 20 episodes，36/36 cells 完成、0 current errors。表中为 focal reward / agreement / joint reward。

| Setting | Direct | Opponent Simulation | V3.3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|
| Buyer | 7.77 / .99 / 19.80 | 12.76 / 1.00 / 20.00 | **13.11** / .95 / 19.00 | 10.65 / .99 / 19.80 | 7.95 / 1.00 / 20.00 |
| Seller | 18.08 / 1.00 / 20.00 | **19.13** / .96 / 19.20 | 16.95 / 1.00 / 20.00 | 16.32 / 1.00 / 20.00 | 15.92 / .99 / 19.80 |
| Resource first | 12.34 / .94 / 17.36 | **40.57** / .96 / 48.08 | 14.55 / .98 / 34.20 | 12.80 / .98 / 31.96 | 18.45 / .99 / 37.32 |
| Resource second | 9.75 / .93 / 29.76 | 11.88 / .97 / 30.74 | **22.47** / .96 / 47.90 | 21.55 / 1.00 / 42.82 | 21.90 / .98 / 39.36 |

能够支持的结论：

1. Framework 不是全面优于 baseline 的单一 SOTA planner；结果强烈依赖 role 与 action space。
2. V3.3 在 Buyer 略高于 Opponent Simulation，在 resource-second 明显高于 direct/Opponent Simulation，证明结构化 belief-aware action selection 有潜力。
3. Seller 中 Opponent Simulation 最强；resource-first 中 Opponent Simulation 大幅领先，所以这两个 setting 是 framework 必须正面解释的失败条件。
4. V4 没有稳定优于 V3.3，说明 generic information exploration 会造成机会成本。
5. V5 在 resource-first 改善，但 Buyer 明显下降，说明 calibration 不是 utility 的充分条件。

### 5.1 Calibration 与 reward 的脱钩

| Setting | V4 Brier → V5 Brier | V4 ECE → V5 ECE | Reward V4 → V5 |
|---|---:|---:|---:|
| Buyer | .282 → **.075** | .519 → **.266** | 10.65 → **7.95** |
| Seller | .297 → **.046** | .526 → **.214** | 16.32 → **15.92** |
| Resource first | .093 → **.024** | .255 → **.141** | 12.80 → **18.45** |
| Resource second | .331 → **.039** | .520 → **.187** | 21.55 → **21.90** |

四个 setting 的 Brier/ECE 都改善，但只有 resource-first 有清晰 reward 增益。这是当前论文最可靠的诊断性 finding。

### 5.2 Continuous 并未证明优于 frozen

正式 V4 中，continuous 相对 frozen 的 reward 差为：

- Buyer：10.65 − 14.97 = **−4.32**；
- Seller：16.32 − 17.53 = **−1.21**；
- Resource first：12.80 − 18.03 = **−5.23**；
- Resource second：21.55 − 25.08 = **−3.53**。

因此不能声称 continuous update 本身有效。更准确的说法是：posterior 会更新，也会改变 action，但当前更新和 planner interface 还没有把新增信息稳定转化为 utility。

### 5.3 Wrong / shuffled / oracle 与 action sensitivity

- Buyer：correct V4 10.65，高于 wrong 6.85、shuffled 8.11，方向合理；
- Resource-first：correct 12.80 反而低于 wrong 16.40、shuffled 22.02，说明 reward 可能来自偶然 surplus extraction，而不是 belief quality；
- 四个 setting 都存在 oracle headroom，说明 inference 和 belief utilization 都仍有改进空间；
- action-flip 证明 action 会随 belief 改变，但还没有证明这些 flip 稳定朝正确方向改善 reward。

### 5.4 已完成的 opponent-switch 结果

旧 switch matrix 为 20/20 cells、2000/2000 episodes、0 current errors。episode 11 将 opponent 替换为 fresh instance，并把 opponent prompt policy 从 strategic brainstorming 切换为 direct；用 no-switch-controlled difference-in-differences 评估适应。

| Setting | V4 continuous DiD | Frozen | Wrong | Shuffled |
|---|---:|---:|---:|---:|
| Buyer | +5.98 | +0.78 | **+11.16** | +9.58 |
| Seller | +2.84 | +3.38 | +2.58 | **+6.56** |
| Resource first | +1.59 | −5.77 | −1.99 | +3.29 |
| Resource second | **−8.51** | −2.82 | −2.00 | −3.06 |

这些结果不能证明 continuous adaptation 的独有优势：Buyer/Seller 中 wrong 或 shuffled 反而更高，resource-second 中 continuous shock 最大。故 switch adaptation 只能在 V7 的 preference/policy 因果链先成立后再进入主 story。

---

## 六、V7 已完成的阶段验证

### 6.1 代码和离线 gate

- 47 个 integration tests 通过；
- ACCEPT / COUNTER / REJECT 对 preference 与 response-policy posterior 的更新方向被覆盖；
- policy surprise reset 不直接重置 preference marginal；
- preference/policy interventions 保持目标 marginal 的定义；
- 553 个旧决策完成离线 replay，没有新增 LLM call，没有产生不可行 proposal。

### 6.2 2×20 smoke

| Setting | V5 | V6 | V7 | V7−V5 两个 paired runs |
|---|---:|---:|---:|---|
| Buyer | 8.025 | 7.950 | **10.600** | +2.90, +2.25 |
| Seller | **18.450** | 17.050 | 16.475 | −3.00, −0.95 |
| Resource first | 18.100 | 16.000 | **20.150** | +0.175, +3.925 |
| Resource second | 17.463 | **29.513** | 24.463 | +13.40, +0.60 |

Smoke 支持进入 confirmatory，但 Seller 两个 runs 均为负，必须保留；不能因为其他三个 setting 较好而删除 Seller。

### 6.3 Resource-first 定向干预 smoke

| Variant | Reward | Agreement | Joint | Correct−variant 两个 paired runs |
|---|---:|---:|---:|---|
| Correct V7 | 20.150 | .975 | 39.70 | — |
| Wrong preference | 11.238 | 1.00 | 11.45 | +6.45, +11.375 |
| Shuffled preference | 18.550 | 1.00 | 38.90 | −0.90, +4.10 |
| Oracle preference | 19.813 | .975 | 27.25 | +1.125, −0.45 |
| Uniform policy | 21.538 | 1.00 | 41.35 | −4.125, +1.35 |
| Shuffled policy | 17.013 | 1.00 | 36.95 | +0.075, +6.20 |

Wrong preference 的损害很强且两 run 一致；shuffled policy 也两 run 有害。但 uniform policy 并不差于 learned policy，说明 response-policy inference 仍未建立稳定优势。

---

## 七、V7 5×20 confirmatory 中期结果

### 7.1 当前完成状态

结果快照：2026-08-13 06:02 UTC。此后恢复队列继续运行，实时状态以 `progress.md` 为准。

- 总计：18/32 cells 完成；
- episodes：1800/3200；
- current errors：0；
- Buyer：8/8 cells 完成；
- Seller：8/8 cells 完成；
- Resource first：V5/V7 完成，其余六个 intervention 在恢复队列中；
- Resource second：尚未进入 confirmatory 队列。

原执行进程在 18 个完整 cell 的边界退出。核验时不存在 partial cell，因此没有 posterior 恢复不一致；恢复脚本只排队 14 个从未开始的 cells，避免覆盖已完成 stdout 日志。审计记录见 `RECOVERY_AUDIT_20260813.md`。

### 7.2 Buyer 完整结果

| Variant | Reward | Agreement | Joint | Brier | ECE | Preference MAE |
|---|---:|---:|---:|---:|---:|---:|
| V5 | 9.81 | 1.00 | 20.00 | .0766 | .2664 | 5.62 |
| V7 continuous | 8.67 | 1.00 | 20.00 | **.0361** | **.1899** | 3.87 |
| V7 frozen | 10.77 | .97 | 19.40 | .0338 | .1837 | **2.12** |
| V7 wrong preference | 6.75 | 1.00 | 20.00 | .3493 | .5910 | 2.42 |
| V7 shuffled preference | 9.24 | .99 | 19.80 | .0370 | .1915 | 3.56 |
| V7 oracle preference | **14.10** | 1.00 | 20.00 | .6548 | .8092 | 3.88 |
| V7 uniform policy | 7.76 | 1.00 | 20.00 | .0344 | .1855 | 5.40 |
| V7 shuffled policy | 8.82 | 1.00 | 20.00 | .0363 | .1901 | 3.36 |

Buyer 的中期判断：

1. V7 相比 V5 calibration 更好，但 reward 从 9.81 降到 8.67，再次验证 calibration ≠ utility。
2. V7 continuous 低于 frozen 10.77，而且五个 paired runs 中 continuous 都低于 frozen；continuous update 仍未证明有益。
3. Wrong preference 降至 6.75，说明 preference belief 确实能因果改变 utility；但 shuffled 为 9.24，且 run 间方向高度不稳定，不能只报告平均值。
4. Oracle preference 达到 14.10，并在五个 paired runs 中都高于 correct V7，说明 inference/planner 仍有明确 headroom。
5. Uniform/shuffled policy 与 correct V7 差异较小且 run 间混合，当前 response-policy learning 尚未建立稳定 utility advantage。

V7−V5 的五个 paired run reward 差为：

```text
[-0.95, -0.95, +2.20, -0.30, -5.70]
```

因此 Buyer smoke 的正向结果没有在 5-run confirmatory 中复现。

### 7.3 Seller 完整结果

| Variant | Reward | Agreement | Joint | Brier | ECE | Preference MAE |
|---|---:|---:|---:|---:|---:|---:|
| V5 | **16.51** | 1.00 | 20.00 | .0469 | .2155 | 3.18 |
| V7 continuous | 15.38 | 1.00 | 20.00 | .0384 | .1958 | 2.43 |
| V7 frozen | 15.79 | 1.00 | 20.00 | .0344 | .1855 | **1.90** |
| V7 wrong preference | 14.07 | .93 | 18.60 | .0000* | .0000* | 1.94 |
| V7 shuffled preference | 14.74 | 1.00 | 20.00 | .0073 | .0856 | 2.12 |
| V7 oracle preference | **16.73** | 1.00 | 20.00 | .6377 | .7985 | 2.39 |
| V7 uniform policy | 15.33 | .99 | 19.80 | .0353 | .1877 | 3.04 |
| V7 shuffled policy | 16.08 | 1.00 | 20.00 | .0356 | .1887 | 2.66 |

`*` Wrong variant 的近零 response Brier 不能解释为 preference 正确；它只说明在该 intervention 诱导的少量观测 action 上，行为接受预测与结果碰巧一致。Response calibration 与 preference truth 是不同变量。

Seller 的中期判断：

1. V7 reward 15.38 低于 V5 16.51，且五 run 差异为 `[+0.10, +0.15, −2.70, −3.70, +0.50]`；平均下降主要来自两个困难 run。
2. Continuous 仅略低于 frozen 15.79，仍没有 continuous advantage。
3. Wrong/shuffled preference 都降低 reward；wrong 还将 agreement 降至 .93、joint 降至 18.60，方向支持 preference causality，但 run-level 稳定性仍需最终 paired statistics。
4. Oracle 为 16.73，高于 correct V7，保留 headroom，但提升远小于 Buyer。
5. Learned policy 15.38 与 uniform policy 15.33 基本相同，shuffled policy 反而为 16.08；更重要的是，Seller 的 policy-uniform/policy-shuffled 在线 shadow action-flip 为 0。因此当前不能把不同独立 cell 的 reward 波动归因于 response-policy belief 改变了 action，Seller 明确没有建立 learned-policy utility advantage。

### 7.4 Resource-first V5/V7 完整对照

| Variant | Reward | Agreement | Joint | Brier | ECE | Preference MAE | q10–q90 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| V5 | 15.015 | .98 | 33.30 | .0267 | .1460 | **.426** | .27 |
| V7 continuous | **19.880** | **1.00** | **39.52** | **.0117** | **.0970** | .468 | **.31** |

V7−V5 的五个 paired-run reward 差为：

```text
[+2.475, +1.775, +2.800, +11.075, +6.200]
```

平均提升为 +4.865；以五个 run 为分析单位的描述性 95% t 区间约为 `[+0.062, +9.668]`。五个 run 方向全部为正，同时 agreement、joint reward、Brier 和 ECE 也改善。这是当前 V7 最强的 confirmatory 正向证据。

但该结果仍不能单独证明 continuous update 或 policy factorization 的贡献，因为 frozen/wrong/shuffled/oracle/policy intervention 尚在运行。V7 的 preference MAE 还略高于 V5，说明 aggregate MAE 也不是足够的决策相关指标；最终必须结合 action flip 与 realized intervention reward 判断。

### 7.5 当前不能宣称什么

目前不能宣称：

- V7 全面超过 Direct、Opponent Simulation 或 V5；
- continuous belief 已优于 frozen belief；
- learned response-policy belief 已优于 uniform prior；
- 更低的 Brier/ECE 会提高谈判 utility；
- opponent-switch adaptation 已被证明。

当前可以较稳健地宣称：

- factorized belief 能在不增加 rollout call 的情况下改变 structured action；
- preference truth 对 action 和 reward 存在可测量的因果影响，尤其 Buyer oracle headroom 和 resource-first wrong-preference collapse；
- response-policy belief 的 utility 价值仍未被证实；
- calibration、belief accuracy、action flip 与 utility 必须分开报告。

---

## 八、希望重点探究的问题

### RQ1：Preference 与 response policy 是否可以从自然语言行为中辨识？

同一个 COUNTER 可能表示：

- 对手真的不喜欢当前 offer；
- 对手喜欢但战略性延迟；
- 对手在锚定；
- 对手输出格式或推理不稳定。

V7 的 factorization 是方法 novelty 的核心，但目前 policy posterior 在很多 setting 保持高 entropy，policy intervention 的 action flip 很低。后续需要验证是 observation design 不可辨识，还是 likelihood model 太弱。

### RQ2：Continuous update 为什么常低于 frozen？

候选解释包括：

1. 对手主动 proposal 是 endogenous action，仍被过度当作 preference evidence；
2. 同一语言策略在 repeated episodes 中高度相关，effective sample size 远低于 episode 数；
3. posterior 更新虽方向合理，但 planner regret penalty 放大了局部错误；
4. opponent 在学习 focal policy，使 stationary-belief 假设失效；
5. semantic evidence 与 structured response evidence 发生重复计数。

需要用 posterior drift、event ablation、effective evidence weight 和 per-decision regret 做机制定位，而不是仅调学习率。

### RQ3：什么样的信息才值得探索？

V4 证明 generic entropy/IG 会浪费确定 deal；V6 又过于保守，在线 0 probes。真正需要的是：

```text
information is useful only if it can change a future optimal action
and the expected improvement exceeds the current deal opportunity cost.
```

后续探索应优先设计可辨识 offer pairs，而不是扩大一个 generic information bonus。

### RQ4：Belief 影响的是自己的 utility，还是只是在重分配 surplus？

Resource-first 中 wrong/shuffled 有时提高 focal reward，却可能显著降低 joint reward。因此任何“改进”都必须同时看：

- focal reward；
- agreement；
- joint utility；
- opponent reward；
- Pareto efficiency / surplus capture；
- no-deal probability。

### RQ5：Action-space 与 role 的结构差异是什么？

- 标量价格谈判：belief 主要改变 acceptance threshold 与 counter price；
- 组合资源交换：belief 同时改变 bundle choice、交换数量与 frontier breadth；
- first mover 缺乏对手当局 evidence，可能更依赖主动 exploration；
- second mover 有 observed offer，accept opportunity cost 更重要。

统一原则可以相同，但候选 action representation 和 identifiability 必须按 action-space 结构设计。

### RQ6：Opponent switch 应拆成 preference switch 与 policy switch 吗？

旧 switch 同时更换 fresh opponent instance 并改变 prompt policy，混合了 identity、memory 和 response policy 变化。V7 后续应分别测试：

1. preference 改变、response policy 不变；
2. preference 不变、response policy 改变；
3. 两者都改变；
4. no-switch control。

只有 factorized posterior 对相应 latent 维度选择性变化，且 controlled DiD 优于 frozen/wrong/shuffled，才能声称适应能力。

---

## 九、后续实验计划与 gate

### 阶段 A：完成当前 V7 confirmatory

当前矩阵必须完整达到：

```text
32/32 cells
3200/3200 latest episodes
0 current errors
每个 cell 5 runs × 20 episodes
```

在此之前不根据 Buyer/Seller 中期结果修改 V7 代码，以避免用 confirmatory 数据调参。

### 阶段 B：冻结分析协议

最终报告必须包含：

1. 每个 setting/variant 的 run-level mean 与 95% CI；
2. V7 vs V5、continuous vs frozen、correct vs intervention 的 paired run differences；
3. sign consistency（5 runs 中有几 run 同方向）；
4. agreement、focal、opponent、joint utility；
5. Brier、NLL、ECE、preference MAE、coverage；
6. action-flip rate；
7. model calls、format success、protocol repair；
8. calibration → action flip → realized utility 的分层 mediation table。

主要因果 gate：

- Correct preference 应优于 wrong/shuffled，且不能只靠一个 run；
- Continuous 应至少不再系统性低于 frozen；
- Learned policy 应优于 uniform/shuffled policy；
- Oracle 应显示合理 headroom，但 oracle reward 不能代替 learned-belief 证据；
- 不允许用跨 setting 总平均掩盖 Seller/resource-first 失败。

### 阶段 C：决定是否运行 V7 opponent-switch

只有同时满足以下条件才运行正式 V7 switch：

1. 至少两个 setting 中 correct preference 明显优于 wrong/shuffled；
2. 至少一个 setting 中 learned policy 优于 uniform/shuffled；
3. Continuous 不再在多数 setting 系统性低于 frozen；
4. Action-flip 不为零，且 flip 后 realized utility 方向合理。

若 gate 不满足，不再投入 2000+ episodes 做 switch，而转入下一轮机制改进。

### 阶段 D：下一轮方法改进（候选 V8）

V8 不应基于当前 confirmatory 逐 setting 调 reward threshold。推荐改进为：

#### D1. Evidence-channel factorization

把 observation 明确分为：

- response to locked offer；
- opponent-initiated offer；
- linguistic preference claim；
- strategic language act（anchor、delay、threat、reciprocity）；
- protocol/noise event。

每个 channel 有独立 reliability 和 effective sample-size discount，避免重复计数。

#### D2. Policy identifiability probes

主动 offer 不再最大化 generic entropy reduction，而是在候选 actions 中寻找：

```text
高 preference agreement + 高 policy response disagreement
```

或：

```text
高 policy agreement + 高 preference response disagreement
```

这样可以有针对性地区分 theta 与 phi。

#### D3. Posterior-safe action mediation

在 trace 中记录每次更新的：

- posterior change；
- top action before/after；
- predicted value delta；
- realized next-step regret；
- 是否由 preference 或 policy marginal 驱动。

只有 posterior change 造成有价值 action flip 时才增加 belief utilization 权重。

#### D4. Non-stationarity model

将 response policy 从固定 `phi` 扩展为慢变量或 change-point model：

```text
theta: 相对稳定的 preference
phi_t: 可随 repeated interaction 改变的 response policy
```

surprise reset 只作用于 `phi_t`，并用 explicit hazard rate 控制，而不是用单次低预测概率硬重置。

### 阶段 E：跨 benchmark 验证

NegotiationArena 只覆盖 bilateral repeated bargaining。若 V7/V8 建立因果链，下一步应回到：

- SimpleEnv / AmazonHistoryPrice：验证标量 reservation inference；
- CaSiNo：验证多议题偏好与自然语言 evidence；
- 一个 opponent-switch / non-stationary split：验证 continuous adaptation。

跨 benchmark 应保持同一抽象接口：

```text
environment adapter
→ structured event extractor
→ factorized belief state
→ legal action generator
→ belief-usable planner
→ language realizer
```

不能为每个环境重新发明一套隐藏变量和 reward-specific rule，否则无法支撑 universal framework 的 story。

---

## 十、当前最合理的 novelty story

### 10.1 方法层

> Natural-language negotiation requires separating what an opponent wants from how it strategically responds. We maintain a factorized, continuously updated preference–policy belief and use it through action-level posterior regret rather than free-form opponent descriptions.

中文表述：

> 自然语言谈判中的可观察行为同时受私有偏好与战略响应策略影响。我们维护可解释的 preference–policy 因子化连续 belief，并通过 action-level posterior regret 将 belief 直接接入合法 structured planner，而不是只生成自由文本对手画像。

### 10.2 评估层

> We causally test the full belief-to-utility chain with targeted interventions, action-flip diagnostics and controlled opponent switches.

中文表述：

> 我们不把最终 reward 自动归因于 opponent modeling，而是通过 frozen、wrong、shuffled、oracle、policy intervention、action-flip 和 controlled switch，逐层检验 belief update、action sensitivity 与 realized utility 的因果关系。

### 10.3 根据当前结果应保持的诚实边界

如果最终 V7 仍表现为：

- calibration 改善但 reward 不改善；
- continuous 低于 frozen；
- learned policy 不优于 uniform；
- oracle 有 headroom，但 learned belief 无稳定优势；

则论文贡献应收缩为：

1. factorized opponent belief 的可审计 framework；
2. belief-causality evaluation protocol；
3. calibration/action/utility 脱钩的系统性 negative finding；
4. 对 generic information exploration 和 naive continuous update 的机制诊断。

不应把方法包装成全面 SOTA。相反，这个结果可以形成更可信的研究 story：过去的 negotiation opponent modeling 评价不足以证明模型真的“利用了 belief”。

如果后续 V8 能让 correct > wrong/shuffled、continuous > frozen、learned policy > uniform，并在 switch 中体现选择性适应，则可以把方法贡献升级为通用的 decision-relevant continuous opponent modeling。

---

## 十一、当前结论

1. 代码已经形成 environment adapter、repeated runner、factorized belief、structured planner、protocol-safe language realization、全量 trace 和因果分析工具链。
2. 旧正式矩阵证明 framework 在 Buyer/resource-second 有潜力，但并非全面超过 Opponent Simulation；Seller/resource-first 是明确的失败条件。
3. V5 和当前 V7 都再次证明 prediction calibration 与 negotiation utility 脱钩。
4. V7 Buyer confirmatory 没有复现 smoke 的 reward 提升；continuous 仍低于 frozen，但 wrong preference 的损害和 oracle headroom 表明 preference belief 具有真实因果作用。
5. Seller 当前也没有超过 V5；两个 policy variants 已完成，但 learned policy 不优于 uniform/shuffled，且 policy intervention 没有造成在线 action flip。
6. Resource-first 的 V7 相对 V5 在五个 paired runs 全部提高，平均 +4.865，同时 joint reward 与 calibration 改善；这是当前最强正向结果，但需要剩余 intervention 才能归因。
7. 现阶段最有 novelty、也最稳健的贡献是：**factorized preference–policy belief + belief-usable structured planning + 对 belief→action→utility 全链条的因果评估**。
8. 下一步首先完成当前 32-cell matrix；随后按预注册 gate 决定是否值得做 V7 switch，或者转入 evidence-channel/identifiability/change-point 方向的 V8。

---

## 十二、相关文件

- 正式结果解释：`NegotiationArena/arena_runs/formal_background_20260812/NEGOTIATIONARENA_FORMAL_INTERPRETATION_AND_V6_PLAN_CN.md`
- 正式完整报告：`NegotiationArena/arena_runs/formal_background_20260812/NEGOTIATIONARENA_FORMAL_RESULTS_CN.md`
- V7 预注册：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/preregistration.md`
- V7 变更记录：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/changes.md`
- V7 smoke decision：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/smoke_decision.md`
- V7 confirmatory progress：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/runs/confirmatory_seed20260900/progress.md`
- V7 confirmatory 自动分析：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/analyze_confirmatory.py`
- V7 恢复审计：`NegotiationArena/research_iterations/iteration_008_v7_factorized_preference_policy/RECOVERY_AUDIT_20260813.md`
- 主 framework 代码：`NegotiationArena/arena_integration/decision_calibrated_agent.py`
- 统一 runner：`NegotiationArena/arena_integration/run_opponent_simulation_setting.py`
