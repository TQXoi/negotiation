# Negotiation Belief Model–Planner 项目完整总结与发表路线图

**结果审计日期：2026-09-07；目录迁移说明更新：2026-09-15**
**项目根目录：`/work5/qixint`**
**现权威代码目录：`/work5/qixint/negotiation_final`**（旧路径 `/work5/qixint/negotiation` 为兼容软链接）

> 目录迁移说明：本报告记录的是 2026-09-07 的实验状态，数值未因 2026-09-15 的归档改变。文中的旧日期目录如 `/work5/qixint/8.23/`、`/work5/qixint/9.2/` 已移动至 `/work5/qixint/negotiation_past/historical_documents/8.23/`、`.../9.2/`；精选报告副本位于 `negotiation_final/docs/`。旧 `universal_reward_research`、`external_negotiation_envs`、`Simple_Env` 路径仍通过兼容软链接可访问。新实验必须写入 `negotiation_final/runs/`，并由 Git 忽略。

---

## 0. 执行摘要

本项目研究的是一个面向自然语言谈判的通用 agent 框架：agent 不仅生成回复，还持续维护对手的偏好、reservation value、响应倾向与当前谈判状态，并要求 planner 在可执行候选动作上显式使用这些 belief。当前最重要的研究认识是：

1. **只提高 belief prediction accuracy 并不会自动提高谈判收益。** ANL、NegotiationArena 和 AgenticPay 都出现过 belief 更准但 reward 更差的情况。核心问题是 belief 是否改变了正确的 probe、offer、concession 与 accept/continue 边界。
2. **有效部分不是单一“大而全 prompt”，而是结构化决策链。** 当前较可靠的共同结构是：canonical state、证据约束 belief、合法候选生成、belief-aware scoring、selection validator、action-locked rendering 和协议校验。
3. **Simple/AmazonHistoryPrice 已得到最清楚的正结果。** 在固定 seller、128 个冻结测试场景、每场景 3 rollouts 的正式比较中，V6.3 的平均 buyer reward 为 **0.5130**，高于 CoT 的 **0.4065** 和历史 full framework 的 **0.4464**；相对两者的 paired bootstrap 置信区间均不跨 0。
4. **AgenticPay 已从“明显失败”推进到 multi-seller 子集上的强结果，但全集证据尚未闭合。** V60 在 `only_multi_seller` 29 tasks × 3 seeds 上 BuyerScore 为 **30.7184**，高于 Direct **22.6749**、CoT **16.8316** 和历史 full framework **17.0819**。但 V60 对 Direct 的区间仍跨 0；all-116/all-231 旧运行又包含 25 个已定位的 adapter instrumentation 错误，修复后的全集需要重跑。
5. **CaSiNo 有收益提升，但 belief 本身的证据较弱。** 最佳 chooser 的 P1 score 比 direct prompt 高约 3.6–4.8 分，但 agreement 略低，belief top-1 仅约 0.38–0.40。
6. **NegotiationArena 展示出强烈角色依赖。** framework 在 buyer 和 BLUE second-mover resource exchange 上可以超过 Opponent Simulation，却在 seller 和 RED first mover 上落后；这证明 planner 必须显式处理角色、信息量与先后手，而不是证明已有一个全设置通胜版本。
7. **目前还不能严谨声称“一个完全相同的 universal framework 在所有 benchmark 上均优于 baseline”。** 现状更准确地说是：已有共同接口和共同决策原则，但 CaSiNo、NegotiationArena 与 AgenticPay 仍包含较多环境专用组件，checkpoint 和参数也未完全统一。

因此，最有希望的论文主线不是“更复杂的 opponent model”，而是：

> **在不确定、部分可识别的自然语言谈判中，如何把对手 belief 转换为安全、可执行且真正改变决策边界的行动；并用 causal belief interventions 验证收益来自有效使用 belief，而不是额外 prompt、协议修复或先验泄漏。**

---

## 1. 项目目标与当前研究问题

### 1.1 最终优化目标

当前实验已明确采用 reward-first 原则：

- Simple/AmazonHistoryPrice：固定 seller，只修改 buyer，以 **buyer reward** 为最终目标；
- AgenticPay：固定其他 agent，只修改 focal buyer，以官方 **BuyerScore** 为最终目标；
- CaSiNo：以 focal P1 score、agreement 和 joint outcome 为主要指标；
- NegotiationArena：以 focal reward 为主要指标，同时报告 agreement 与 joint payoff。

belief accuracy、calibration error、action-flip、probe frequency、wrong/shuffled/oracle belief 等指标只用于解释机制，不能替代最终 reward。

### 1.2 需要回答的科学问题

1. 什么信息可以从自然语言历史中被可靠识别为 opponent belief？
2. belief 的不确定性、证据来源与时效性应如何表示？
3. planner 如何使用 belief 改变：
   - 主动 probe；
   - offer/contract 候选；
   - concession frontier；
   - accept/continue/walk-away 边界；
   - 多 seller 路由与多 issue 条款选择？
4. belief 错误时，如何避免比 direct prompt 更差？
5. 同一套方法能否跨 single-product、multi-issue、multi-party 和 multi-seller topology 工作？

---

## 2. 文件夹总览与推荐使用方式

### 2.1 权威代码目录

后续开发应只以以下目录作为权威代码源：

```text
/work5/qixint/negotiation/
├── framework/                  # 跨环境 canonical schema、belief、planner 与 engine
├── Simple_Env/                 # Simple/AmazonHistoryPrice 环境与 buyer variants
├── CaSiNo_Env/                 # CaSiNo 环境、belief-usable planner 与评测
├── AgenticPay_Env/             # AgenticPay adapter、buyer、issue classifier 与 evaluator
├── NegotiationArena/           # NegotiationArena agents、runners 与分析
├── experiments/                # 跨环境运行入口、模型客户端、训练与诊断
├── scripts/                    # 可复现实验脚本
├── tests/                      # shared framework 与集成测试
├── tools/                      # trajectory 查看和数据工具
└── docs/
    ├── timeline/               # 按时间整理的研究报告
    ├── presentations/          # 可编辑 PPT
    ├── reports/                # benchmark 专项总结
    └── results/                # 精简后的关键结果与分析
```

公开仓库地址为：`git@github.com:TQXoi/negotiation.git`。当前本地 HEAD 为：

```text
aa045bc Add portable universal negotiation environments and research archive
```

**重要状态：本地存在尚未提交的有效改动。** 包括 AgenticPay V60 相关 adapter、issue classifier、runner instrumentation、训练脚本和新测试。因此 GitHub 上的 commit 还不能代表当前最优实验代码。在下一次大规模实验前，应先完成 secret scan、提交并打 tag。

### 2.2 原始实验记录

```text
/work5/qixint/universal_reward_research/
```

该目录保存 Iteration 000–063 的：

- 每轮假设与修改记录；
- config/manifest；
- raw result；
- trajectory 摘要；
- bootstrap 分析；
- AgenticPay adapter 修复与 V60 全任务验证记录。

它是审计研究过程的重要证据，但不应整体提交公开仓库，因为体积大、可能包含原始模型输出，并且当前 `.gitignore` 已忽略 `research_iterations`。

### 2.3 外部 benchmark 和上游代码

```text
/work5/qixint/external_negotiation_envs/
├── LLM-Deliberation/
├── NegotiationArena/
├── OpponentSimulation/
└── ANL_2024_2026/
```

用途：

- 保存原论文代码、paper branch 和复现 harness；
- 对照原 prompt、parser 和 metric；
- 保持第三方 license 与本项目修改分离。

公开仓库不应无选择地复制这些第三方源码，推荐使用固定 commit 的 submodule、安装说明或 patch 文件。

### 2.4 历史/只读目录

以下根目录下的旧副本包含早期实验，但不应继续双向修改：

```text
/work5/qixint/ASTRA_Env/
/work5/qixint/Simple_Env/
/work5/qixint/AgenticPay_Env/
/work5/qixint/Terms_Env/
```

它们可用于追溯早期 script、checkpoint 路径与 raw result。后续任何必要代码应迁入 `/work5/qixint/negotiation` 后再维护。

### 2.5 报告与 PPT

主要时间线资料分布于：

```text
/work5/qixint/7.27/
/work5/qixint/8.5/
/work5/qixint/8.7/
/work5/qixint/8.12/
/work5/qixint/8.13/
/work5/qixint/8.14/
/work5/qixint/8.17/
/work5/qixint/8.23/
/work5/qixint/9.2/
/work5/qixint/9.7/
```

推荐优先阅读：

1. `/work5/qixint/9.2/NEGOTIATION_BELIEF_PLANNER_ALL_BENCHMARK_COMPLETE_SUMMARY_CN.md`
2. `/work5/qixint/8.23/` 下 AgenticPay 问题、任务和 trajectory 报告
3. `/work5/qixint/negotiation/docs/presentations/2026-08-17_12_reward_first_framework_results_plan_editable_en.pptx`
4. `/work5/qixint/negotiation/docs/presentations/2026-08-14_11_universal_framework_results_plan_editable_en.pptx`
5. `/work5/qixint/negotiation/docs/presentations/2026-08-07_10_benchmark_novelty_plan_editable_en.pptx`

### 2.6 不应发布或搬迁的目录

- `/work5/qixint/models/`：模型权重；
- `/work5/qixint/venvs/`、conda/cache：机器相关环境；
- 大规模 raw trajectory；
- API key、token、`.env`、endpoint credential；
- 临时日志、服务 PID、cache；
- 已被 clean repo 取代的重复旧代码。

---

## 3. 当前 framework 的实现

### 3.1 跨环境核心对象

核心 schema 位于：

```text
/work5/qixint/negotiation/framework/schemas.py
```

主要对象：

- `IssueSpec`：issue 名称、类型、可选值与约束；
- `CanonicalOffer`：跨环境统一的 offer/contract 表示；
- `NegotiationObservation`：一轮可观察事件；
- `CanonicalState`：当前己方私有目标、公开历史、轮次、候选和约束；
- `ResponsePolicyBelief`：对手对不同动作的响应概率；
- `PreferenceBelief`：对手 issue 权重、reservation/value preference；
- `OpponentBelief`：将偏好、响应、置信度和证据组织起来；
- `CandidateAction` / `ScoredCandidate`：合法动作及 planner 分数；
- `FrameworkDecision`：被锁定的最终结构化决策。

### 3.2 执行链

统一执行入口位于：

```text
/work5/qixint/negotiation/framework/engine.py
```

当前设计可以概括为：

```text
benchmark observation / private utility
    ↓
EnvironmentAdapter
    ↓
CanonicalState + legal CandidateAction set
    ↓
BeliefStore / evidence-constrained updater
    ↓
belief-usable candidate scoring and ranking
    ↓
selection validator / terminal guard / IR guard
    ↓
action-locked renderer
    ↓
protocol validator and minimal repair
    ↓
unchanged benchmark parser and official scorer
```

此结构的重要原则是：LLM 可以辅助抽取和措辞，但不能在 renderer 阶段偷偷修改 planner 已选择的价格、seller、issue 或 accept/reject 动作。

### 3.3 Belief subsystem

主要代码：

```text
/work5/qixint/negotiation/framework/belief.py
/work5/qixint/negotiation/framework/events.py
```

当前已经实现或探索过：

- broad prior / static / continuous / frozen belief；
- reservation-value posterior；
- response-policy belief；
- strategic concession/aspiration belief；
- latent mixture 与 readiness；
- censored evidence update；
- verified event 与 evidence provenance；
- semantic、verified semantic 和 shadow semantic updater；
- wrong/shuffled/oracle belief intervention；
- repeated-opponent 和 change-point 思路。

最重要的经验是：必须区分“对手明确说出的事实”“由拒绝/接受得到的区间证据”“角色名或任务模板先验”“LLM 自由推断”。后两类不能被当作同置信度的 ground truth。

### 3.4 Planner subsystem

主要代码：

```text
/work5/qixint/negotiation/framework/planner.py
/work5/qixint/negotiation/framework/trainable_planner.py
/work5/qixint/negotiation/framework/conservative_awr_planner.py
```

逐步形成的 planner 能力包括：

- terminal-safe accept/continue；
- belief-grounded arbitration；
- deadline-aware concession；
- behavioral frontier；
- information-gain-aware active probe；
- posterior-integrated candidate scoring；
- option-aware/multi-seller choice；
- conservative residual/AWR policy；
- oracle、shuffled、frozen、no-information-gain 对照。

Simple V5.9/V6.3 的有效改进主要来自：行为 frontier、候选级 scoring、保守 accept/continue 边界和 AWR residual，而不是简单增加搜索次数。

### 3.5 当前“通用性”的真实边界

目前已经实现共同 schema 和共同理念，但还不是一个完全相同的跨环境 agent：

| 环境 | 复用 shared core | 仍然专用的部分 |
|---|---|---|
| Simple | 高 | 金额归一化、seller price parser、单商品动作集合 |
| AgenticPay | 中 | contract template、issue-role classifier、seller routing、multi-party settlement state machine |
| CaSiNo | 较低 | 离散 allocation belief、ratio candidate planner、CaSiNo action protocol |
| NegotiationArena | 较低 | buyer/seller/resource 两类专用 agent 与 game parser |

发表前必须决定：

- 要么真正统一 adapter 后的 planner、belief representation 和参数；
- 要么诚实定位为“统一设计原则 + benchmark-specific adapters”，不声称同一 checkpoint 零修改通用。

---

## 4. Benchmark 总表

| Benchmark | 主要交互 | 当前最佳证据 | 相对 baseline | 数据覆盖 | 论文状态 |
|---|---|---:|---:|---|---|
| Simple/AmazonHistoryPrice | 单 buyer、单 seller、单商品、自然语言 | V6.3 reward **0.5130** | +0.1066 vs CoT；+0.0666 vs historical full | 128/128 × 3 rollouts | 当前最强主结果 |
| AgenticPay only_multi_seller | 多 seller、合同/多 issue、自然语言 | V60 BuyerScore **30.7184** | +13.89 vs CoT；+13.64 vs full；+8.04 vs Direct | 29/29 × 3 seeds | 强子集结果，需扩大验证 |
| AgenticPay all-116/all-231 | 多 topology、自然语言 | 旧 V60 raw run 含 25 adapter errors | 尚不能做正式比较 | 已遍历但运行无效 | 必须修复后重跑 |
| CaSiNo | 双边、3 issue allocation、自然语言 | chooser P1 **22.17–22.30** | 比 direct 高约 3.6–4.8 | official test 100/100，单 rollout | 可作主/次主结果，需 CI |
| NegotiationArena | buy-sell/resource exchange、自然语言 | 不同角色不同 best | buyer +0.35、resource-second +10.59 vs OS；另两设置下降 | 大规模 grid，约 13k episodes | 机制/直接竞争基准 |
| LLM-Deliberation | 6-party、5 issue、公开讨论 | V3 N-1 agreement **0.80** | 接近/达到论文现象 | 20 episodes | 存在先验泄漏，建议附录 |
| ANL 2024/2025 | 结构化自动谈判、无自然语言 | static 通常优于 continuous | 显示 oracle headroom | 已完成 pilot/main | planner 机制附录 |

---

## 5. CaSiNo

### 5.1 任务与代码结构

CaSiNo 是 camping resource allocation：Food、Water、Firewood 各 3 个单位，双方拥有私有优先级，通过自然语言协商分配。

```text
/work5/qixint/negotiation/CaSiNo_Env/
├── buyer/
│   ├── factory.py
│   └── belief_usable_planner/
│       ├── belief_model.py
│       ├── planner.py
│       ├── llm_chooser.py
│       ├── generator.py
│       └── buyer.py
├── environment/
└── eval.py
```

历史正式运行脚本：

```text
/work5/qixint/ASTRA_Env/scripts/run_partner_base_belief_usable_planner.sh
```

当前最佳 variant 为 `belief_usable_llm_chooser`：

1. `MathPartnerBeliefModel` 根据历史维护对方 issue priority 分布；
2. `RatioCandidatePlanner` 枚举/过滤可行 allocation；
3. planner 计算己方收益、预测接受概率和联合收益；
4. LLM 只能从候选 ID 中选择；
5. generator 使用 action lock 渲染，防止自然语言修改 allocation。

### 5.2 完成的结果

Partner-Base，P1 focal，12 rounds，official test split 100：

| 运行 | Variant | P1 score | P2 score | Agreement | Walk-away | Belief top-1 |
|---|---|---:|---:|---:|---:|---:|
| 2026-07-29 | Direct prompt | 18.73 | 17.24 | 99 | 1 | — |
| 2026-07-29 | Historical full | 21.62 | 16.03 | 100 | 0 | — |
| 2026-07-29 | Deterministic planner | 22.27 | 14.18 | 96 | 4 | — |
| 2026-07-29 | LLM chooser | **22.30** | 13.55 | 95 | 5 | 0.40 |
| 2026-08-03 | Direct prompt | 17.40 | 16.87 | 94 | 6 | — |
| 2026-08-03 | Historical full | 21.66 | 15.75 | 99 | 1 | 0.48 |
| 2026-08-03 | Deterministic planner | 21.67 | 13.13 | 92 | 8 | — |
| 2026-08-03 | LLM chooser | **22.17** | 13.87 | 95 | 5 | 0.38 |

### 5.3 结论与限制

- framework 显著提高 focal P1 point estimate；
- 收益伴随对手得分下降和少量 agreement 损失，需同时报告社会福利；
- belief top-1 较低，不能把增益全部归因于准确 opponent modeling；
- 当前是 100 个 test scenario 的全覆盖，但每个 scenario 仅一次随机 rollout，尚缺 paired multi-rollout CI；
- 下一步应增加 calibrated posterior、wrong/shuffled/oracle interventions，并验证相同 planner 在 unseen partner style 上是否仍有增益。

---

## 6. NegotiationArena

### 6.1 任务与代码结构

NegotiationArena 包含两类主要自然语言谈判：

- buyer–seller：围绕单一价格成交；
- resource exchange：双方交换多种资源，可从不同先后手和 focal 角色评估。

```text
/work5/qixint/negotiation/NegotiationArena/arena_integration/
├── decision_calibrated_agent.py
├── repeated_agents.py
├── paper_aligned_opponent_simulation.py
├── repeated_games.py
└── run_opponent_simulation_setting.py
```

关键文档：

```text
/work5/qixint/negotiation/docs/reports/NEGOTIATIONARENA_FINAL_REPORT_CN.md
/work5/qixint/negotiation/docs/results/V8_CONFIRMATORY_ANALYSIS.md
```

### 6.2 已完成的主要比较

四个 focal setting：

1. focal buyer，BLUE second mover；
2. focal seller，RED second mover；
3. focal RED，resource first mover；
4. focal BLUE，resource second mover。

旧正式矩阵每个 cell 为 5 runs × 20 episodes：

| Setting | Direct reward / agreement | Opponent Simulation | Framework best reported | 相对 OS |
|---|---:|---:|---:|---:|
| Buyer | 7.77 / 0.99 | 12.76 / 1.00 | V3.3: **13.11 / 0.95** | +0.35 |
| Seller | 18.08 / 1.00 | **19.13 / 0.96** | V3.3: 16.95 / 1.00 | -2.18 |
| Resource RED first | 12.34 / 0.94 | **40.57 / 0.96** | V5: 18.45 / 0.99 | -22.12 |
| Resource BLUE second | 9.75 / 0.93 | 11.88 / 0.97 | V3.3: **22.47 / 0.96** | +10.59 |

V8 confirmatory 中的 focal reward：

| Setting | V5 | V7 | V8 | Frozen | Shuffled | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Buyer | 8.880 | 8.560 | 7.240 | 7.490 | 7.940 | **12.710** |
| Seller | 14.360 | 15.180 | 11.120 | 13.750 | 11.560 | **17.330** |
| Resource first | 18.455 | 18.460 | 22.265 | **22.500** | 18.020 | 22.445 |
| Resource second | 24.420 | 21.875 | 21.675 | 25.125 | **25.665** | 25.655 |

### 6.3 结论与限制

- opponent belief 在不同角色中的价值不对称：second mover 可利用观察到的 offer，first mover 缺乏 evidence；
- oracle headroom 表明更好的 opponent knowledge 有价值，但现有 continuous updater 未稳定逼近 oracle；
- shuffled/frozen 有时不弱于 learned belief，说明 prompt regularization、candidate set 和 role policy 可能是主要增益来源；
- 已完成约 13,000 episodes 的大规模 grid，基础设施已稳定；
- 但环境只有两类 schema 和固定 utility，不能单独支撑“通用 negotiation framework”；
- 不能把不同 setting 各自最优的 V3.3/V5/V8 拼成一个主方法。发表前必须冻结一个统一 variant，并在所有 setting 上报告。

---

## 7. Simple Env / AmazonHistoryPrice

### 7.1 任务、指标与代码

该环境是最基础且通用的单 buyer–单 seller–单商品自然语言讨价还价。seller 固定，只修改 buyer。

buyer reward 近似为：

\[
R_b = \frac{B-P}{|B-C|}
\]

其中 \(B\) 为 buyer budget/value，\(C\) 为 seller cost/reference boundary，\(P\) 为成交价格；no-deal 为 0，非法 overshoot 为 -1。

```text
/work5/qixint/negotiation/Simple_Env/
├── buyer/
│   ├── universal_framework.py
│   ├── factory.py
│   └── direct/cot/full baseline implementations
├── environment/
└── eval.py
```

shared core：

```text
/work5/qixint/negotiation/framework/
```

正式指标记录：

```text
/work5/qixint/universal_reward_research/iteration_000_full_baseline/analysis/PRIMARY_METRICS.md
```

### 7.2 正式全测试结果

测试设计：128 个冻结 validation/test 场景 × 3 rollouts；scenario-clustered paired bootstrap 10,000 次。

| Variant | Mean buyer reward | 相对 CoT | 相对 historical full |
|---|---:|---:|---:|
| Direct prompt | 0.0961 | -0.3104 | -0.3503 |
| CoT prompt | 0.4065 | — | -0.0399 |
| Historical full framework | 0.4464 | +0.0399 | — |
| V5.9 | 0.4981 | +0.0916 | +0.0517 |
| V6.3 conservative AWR | **0.5130** | **+0.1066** | **+0.0666** |

关键置信区间：

- V6.3 − CoT：**+0.1066，95% CI [0.0581, 0.1560]**；
- V6.3 − historical full：**+0.0666，95% CI [0.0203, 0.1118]**；
- V6.3 − V5.9：+0.0149，95% CI [-0.0296, 0.0589]。

### 7.3 结论

- 这是当前最可信的 framework 正结果；
- V5.9 已经说明 behavioral frontier 与 belief-usable candidate planning 相比 CoT/full 有效；
- V6.3 是最高点估计，但相对 V5.9 的 AWR 增量尚不显著，不能声称 training 本身已带来确定提升；
- 当前已覆盖本地冻结 128/128 场景，但仍需外部独立 split、不同 seller policy 和不同 backbone model 以排除对 manifest 或 seller 的过拟合。

---

## 8. AgenticPay

### 8.1 Task topology

本地 AgenticPay 共 231 tasks，覆盖 8 个 family：

| Family | Tasks | 主要难点 |
|---|---:|---|
| single buyer/product/seller | 28 | 单合同价格、条款一致性 |
| only_multi_products | 29 | 多商品组合与 issue binding |
| only_multi_seller | 29 | seller 选择、并行/切换谈判、timeout |
| only_multi_buyer | 29 | focal buyer 是否被选中、竞争行为 |
| multi_products_multi_seller | 29 | 组合合同 + seller routing |
| multi_buyer_multi_seller | 29 | 双侧竞争和 focal attribution |
| multi_buyer_multi_products | 29 | 多 buyer + bundle contract |
| multi_buyer_multi_products_multi_seller | 29 | 完整复杂 topology |

`all-116` 指四个包含 multi-seller 的 family，共 116 tasks。

### 8.2 官方主要指标

AgenticPay 的 **BuyerScore** 是最终目标，而不是本项目内部的 belief score 或 raw buyer reward。其具体计算随 task topology 与官方权重变化，综合：

- 是否达成有效合同；
- buyer 对价格、数量、商品和合同条款的归一化 utility；
- multi-issue 权重；
- allocation/market efficiency；
- 轮次折扣；
- timeout、invalid contract 和 no-deal penalty。

因此，BuyerScore 上升必须来自更多有效成交或更好的合同效用，不能用 classifier accuracy 替代。

### 8.3 当前代码结构

```text
/work5/qixint/negotiation/AgenticPay_Env/
├── buyer/
│   ├── universal_framework.py     # AgenticPayAdapter + UniversalAgent
│   ├── issue_classifier.py        # 可拒绝的 issue-role classifier
│   └── variants.py                # V1–V60 variant registry
├── environment/
│   ├── single28.py
│   ├── multi_agent.py
│   └── all_tasks.py
└── eval.py
```

运行与模型接口：

```text
/work5/qixint/negotiation/experiments/run_agenticpay_single28_framework.py
/work5/qixint/negotiation/experiments/agenticpay_qwen_eval.py
/work5/qixint/negotiation/experiments/model_clients.py
/work5/qixint/negotiation/experiments/train_agenticpay_issue_role_gate.py
/work5/qixint/negotiation/experiments/train_agenticpay_settlement_eligibility_gate.py
/work5/qixint/negotiation/tools/view_agenticpay_trajectory.py
```

### 8.4 版本演化与学到的经验

| 阶段 | 版本 | 主要修改 | 结果/问题 |
|---|---|---|---|
| 初始迁移 | V1–V2 | canonical adapter、continuous belief、AWR | single28 明显低于 Direct/CoT |
| 协议安全 | V3–V4 | action lock、terminal guard | invalid/mismatch 下降，但 reward 仍低 |
| 复杂规划 | V5–V14 | simulation、LLM candidate、offline AWR | 离线指标改善未稳定转成在线 BuyerScore |
| 条款识别 | V15–V31 | settlement validator、opening calibration、issue classifier、probe | 局部改善，仍有 mismatch/timeout |
| IR repair | V32–V54 | public latch、minimal buyer-IR repair、rolling state、classifier variants | V37 成为冻结候选，但未赢 all-231 |
| multi-seller 协议 | V55 | `selected_seller`、per-seller state、完整 contract renderer/validator | all-116 point estimate 大幅提升 |
| timeout/路由 | V56–V60 | concession、acceptance、rejection routing | V60 成为 only_multi_seller 当前最优 |

AgenticPay 中 direct prompt 较少 mismatch、framework 反而曾大量 mismatch 的原因不是 direct 更会推理，而是 framework 早期存在两个 representation：planner 的 canonical issue 和 generator 的自然语言 contract。issue classifier 一旦把 operational obligation 错分为 negotiable product preference，或 renderer 重写 seller/quantity/price，就会产生 parser 可读但语义不一致的合同。后续 action lock、独立可拒绝 classifier 和 full-contract validator 正是为解决这一问题。

### 8.5 single28 初始正式结果

| Variant | BuyerScore |
|---|---:|
| Direct | **32.9531** |
| CoT | 31.7199 |
| Historical full | 21.3108 |
| V1 | 15.7307 |
| V2 | 18.1775 |

V2 比 V1 提升 +2.4468，但 CI 跨 0；初始 universal framework 未超过 direct/CoT。

### 8.6 V37 all-231 失败结果

同一 manifest/seed `20260904`：

| Variant | BuyerScore | Agreement | Timeout |
|---|---:|---:|---:|
| Direct | 27.3621 | 78.79% | 49 |
| CoT | 27.6087 | 82.68% | 40 |
| Historical full | **29.2199** | 86.15% | 32 |
| V37 | 21.6273 | 51.08% | 113 |

该结果明确表明 V37 不能直接处理 multi-seller/multi-party topology；它不应作为当前最终版本。

### 8.7 V55 all-116 point result

V55 对 multi-seller adapter 修复后，在 all-116、seed `20260904`：

- BuyerScore：**40.311**；
- agreement：105/116 = **90.52%**；
- timeout：**9.48%**；
- valid deal：**84.48%**；
- runner errors：0。

同 seed 的独立历史 baseline：

| Variant | BuyerScore | Agreement | Timeout |
|---|---:|---:|---:|
| Direct | 30.281 | 84.48% | 15.52% |
| CoT | 30.710 | 87.93% | 12.07% |
| Historical full | 29.165 | 78.45% | 21.55% |
| V55 | **40.311** | **90.52%** | **9.48%** |

这是有希望的 point estimate，但这些 baseline 并非在当前代码和同一进程中重新 paired 生成，不能作为最终论文表。

### 8.8 V60 only_multi_seller 正式三 seed 结果

29 tasks × 3 seeds = 87 episodes/variant，共 522 rows：

| Variant | BuyerScore | Agreement | Valid deal | Timeout |
|---|---:|---:|---:|---:|
| CoT | 16.8316 | 82.76% | 48.28% | 17.24% |
| Direct | 22.6749 | 96.55% | 59.77% | 3.45% |
| Historical full | 17.0819 | 100.00% | 44.83% | 0.00% |
| V37/V55 | 27.0907 | 79.31% | 65.52% | 20.69% |
| V60 | **30.7184** | 89.66% | **70.11%** | 10.34% |

Paired task-cluster bootstrap：

- V60 − CoT：**+13.8868，95% CI [3.9867, 23.0207]**；
- V60 − historical full：**+13.6365，95% CI [3.1133, 23.5565]**；
- V60 − Direct：+8.0436，95% CI [-1.3824, 17.2570]；
- V60 − V55：+3.6278，95% CI [-1.0097, 8.4892]。

因此目前可以严谨说：V60 在该 multi-seller 子集上显著优于 CoT 和历史 full；高于 Direct 的点估计较大，但现有样本下尚未达到 95% 显著。

### 8.9 all-116/all-231 与 focal multi-buyer 的真实状态

Iteration 063 的旧全任务输出：

- V60 all-116：116 rows，83 agreements、8 timeouts、25 errors；
- V60 all-231：231 rows，184 agreements、22 timeouts、25 errors。

25 个 error 全部集中于 `multi_products_multi_seller/Task5–29`。根因是旧 instrumentation 将环境 class 替换为 function，导致 `resolve_selected_seller` 丢失。当前代码已改用 subclass，并完成：

- 针对性 unit test；
- Task5 online smoke：成功 agreement，BuyerScore **47.704**，无 error。

但修复后 29-task family、all-116 和 all-231 尚未生成一套完整、干净、可发表的新结果。因此旧 all-116/all-231 只能证明 traversal coverage 和 bug 已定位，不能用于最终均值比较。

focal-only multi-buyer 旧结果也不能作为正式结论：

- repo-native/direct 有完整 116 rows；
- CoT 有 114 个 API/500 errors；
- V60 有 116 个 `APIConnectionError`；
- 原因是 vLLM endpoint 中途退出，而不是 agent 行为。

### 8.10 AgenticPay 下一步必须补齐的实验

1. 修复后 clean `multi_products_multi_seller` 29/29 smoke/full；
2. V60、Direct、CoT、historical full 在同一 runner、同一 seller outputs、同一 manifests 下做 paired all-116；
3. 再做 all-231；
4. 重跑 focal-only multi-buyer，报告 focal selected rate、focal BuyerScore、overall market score；
5. 至少 3 seeds 或 3 rollouts/task，按 task 聚类 bootstrap；
6. 锁定 V60 后不再根据 test trajectory 调参，另建 dev family；
7. 报告 token、latency、API failure 与 renderer repair rate。

---

## 9. 辅助 benchmark

### 9.1 LLM-Deliberation

代码：

```text
/work5/qixint/external_negotiation_envs/LLM-Deliberation/
```

V3，20 episodes：

- N−1 agreement rate：0.80；
- unanimous：0.15；
- any success：1.00；
- P1 score：61.05；
- collective score：363.75；
- parse errors：21。

但后续研究指出，角色名称本身泄漏偏好先验，而且任务主要考察 P1 是否提出多数人接受的 deal，single proposer 也可获得相近效果。因此它不适合作为交互式 continuous belief 的主证据，推荐作为 benchmark audit/附录。

### 9.2 ANL 2024

代码：

```text
/work5/qixint/external_negotiation_envs/ANL_2024_2026/belief_planner_anl/anl2024/
```

90 episodes 的代表结果：

| Variant | Mean own utility | Agreement |
|---|---:|---:|
| Continuous V2 | 0.5699 | 0.9778 |
| Static V2 | **0.5808** | 0.9778 |
| Frozen | 0.5755 | — |
| Oracle | 0.5776 | — |
| Shuffled | 0.5581 | — |
| RVFitter | 0.5356 | 0.9889 |

结论：planner point estimate 高于 RVFitter，但 continuous belief 没有超过 static belief。

### 9.3 ANL 2025

代码：

```text
/work5/qixint/external_negotiation_envs/ANL_2024_2026/belief_planner_anl/anl2025/
```

代表结果：

| Variant | Center utility |
|---|---:|
| Boulware | 0.5242 |
| Linear | 0.4831 |
| Conceder | 0.4498 |
| Static global planner | 0.5662 |
| Continuous local | 0.4439 |
| Continuous global | 0.4367 |
| No information gain | 0.4458 |
| Oracle | **0.6110** |
| Shuffled | 0.5292 |

结论：oracle gap 说明 opponent information 有潜在价值，但当前 continuous updater/planner coupling 失败。ANL 无自然语言，更适合做结构化 planner 机制验证，而非论文主要 benchmark。

---

## 10. 跨 benchmark 的主要经验

### 10.1 已经比较可靠的经验

1. **Action lock 和协议 validator 是必要基础设施。** 它们能阻止 LLM 在最终生成时改变 planner 动作。
2. **Belief 应带 evidence provenance 和置信度。** 未经验证的语义推断必须允许 abstain。
3. **Planner 必须在候选级使用 belief。** 仅将 belief JSON 放入 prompt 通常只增加 token 与错误。
4. **Acceptance 与 terminal safety 比更深的搜索重要。** 多数失败来自错误继续、错误 seller、合同不完整和 timeout。
5. **多 seller 必须维护 per-counterparty state。** 把所有 seller 历史压成单一 belief 会造成 attribution error。
6. **Offline training 应只做保守 residual。** Simple V6.3 的 AWR 最多证明小幅额外 point gain；直接让离线 policy 覆盖规则 planner 风险较大。
7. **必须做 causal intervention。** Frozen、shuffled、wrong、oracle belief 和 action-flip 才能区分“belief 真有用”与“prompt/规则更强”。

### 10.2 反复失败的方向

- 为提高 belief accuracy 而训练，但不约束它如何改变决策；
- 单纯增加 CEM/search 轮数；
- 用 final test split trajectory 持续调参；
- 对 factual/operational clause 与可谈判 preference 不做区分；
- 多 seller 共享一个 opponent state；
- 用 best-per-setting variant 汇报成一个 framework；
- 把 parser error 修复带来的增益全部归因于 belief reasoning；
- 用内部 surrogate reward 代替 benchmark 官方 BuyerScore。

### 10.3 当前最值得冻结的内容

- canonical schema 与 adapter interface；
- action-locked renderer；
- full contract validator 和 minimal IR repair；
- evidence provenance、abstention 和 reliability gate；
- per-opponent belief/state；
- Simple V5.9 behavioral frontier；
- V6.3 conservative AWR residual；
- AgenticPay V60 multi-seller routing/acceptance policy。

下一轮优化应只改变少数可解释模块，避免继续增长不可审计的 variant 分支。

---

## 11. 推荐论文 story 与 novelty

### 11.1 不建议的 story

> “我们使用 continuous belief model，因此在所有谈判 benchmark 上更强。”

现有结果无法支持：continuous 有时低于 static/frozen，且 AgenticPay 的大幅增益同时来自 adapter、validator 和 timeout policy。

### 11.2 建议的核心 story

建议定位为：

> **Decision-Calibrated Belief-Usable Planning for Executable Language Negotiation**：在自然语言谈判中，将带证据与不确定性的 opponent belief 映射到合法候选动作，并通过可靠性 gate、主动 probe、terminal safety 和 action-locked rendering，使 belief 只在可识别且能改善期望收益时影响决策。

中文概括：

> 贡献不在于“让 LLM 多写一个 opponent profile”，而在于建立 belief 到 executable action 的可验证闭环，并系统测量 belief 在什么条件下有用、错误 belief 如何伤害决策、planner 如何安全退化到不依赖 belief 的策略。

### 11.3 可主张的四个贡献

1. **Evidence-calibrated factorized belief**
   将 opponent preference、reservation/aspiration、response policy、counterparty identity 和 regime 分开建模；每条 belief 携带来源、时间和置信度，并允许 abstain。

2. **Belief-usable decision interface**
   belief 不直接生成语言，而是影响 probe、candidate frontier、concession 和 accept/continue；最终动作由 validator 锁定。

3. **Safe cross-topology execution**
   使用 canonical offer、per-counterparty state、contract templates 和 protocol validator，把相同决策原则迁移到单商品、多 issue、多 seller 和多 buyer 环境。

4. **Causal evaluation of belief usefulness**
   除 reward 外，使用 frozen/shuffled/wrong/oracle belief、action flip、opponent switch 和 change-point 来验证因果链：belief 改变了什么动作，这些变化是否提高最终收益。

### 11.4 novelty 风险

Opponent Simulation、game-theoretic workflow、K-level reasoning、EPO 和其他 opponent modeling 工作已经覆盖“推断对手并战略规划”的一般想法。因此真正的差异必须通过以下事实体现：

- 不依赖 NegotiationArena 专用 utility simulator；
- 支持自然语言多 issue/multi-seller 合同；
- 明确区分 belief quality 与 belief usefulness；
- 具有错误 belief 的安全退化机制；
- 在多个 topology 使用同一 planner interface 和冻结参数；
- 提供 causal interventions，而不只比较最终分数。

如果这些点未通过实验闭合，论文更适合定位为系统/benchmark audit，而不是新 planning algorithm。

---

## 12. 为完成发表目标必须做什么

### P0：修复、冻结与可复现性

1. 将当前未提交代码做 secret scan 后提交，建立例如：
   - `v6.3-simple-final`；
   - `v60-agenticpay-final`；
   - `paper-experiment-v1`。
2. 保存所有训练 checkpoint；当前 Simple V6.3 checkpoint 仍指向旧目录，且模型文件被 `.gitignore` 排除，需要制作匿名可下载 artifact。
3. 固定 model name、model snapshot、temperature、seed、max tokens、system prompt 和 seller implementation。
4. 每个结果保存 manifest、commit SHA、raw row、summary 和 analysis script。
5. 修复后先跑 AgenticPay `multi_products_multi_seller` 29/29，再启动 all-116/all-231。
6. 保证模型服务失败不会被记成 agent timeout；API failure 必须单独统计并自动 resume。

### P1：主实验矩阵

建议主 benchmark：

- Simple/AmazonHistoryPrice：最干净的 reward 与 paired test；
- CaSiNo：经典自然语言 multi-issue allocation；
- AgenticPay：multi-product/multi-seller/multi-buyer 复杂 topology；
- NegotiationArena：与 Opponent Simulation 的直接竞争实验。

每个环境至少比较：

- Direct prompt；
- CoT prompt；
- historical full framework；
- benchmark/paper native baseline；
- 本文 frozen final framework。

统一要求：

- 只修改 focal buyer/agent，固定对手；
- 相同 private profile、seller outputs、model calls 和随机种子；
- 至少 3 rollouts 或 3 seeds；
- scenario/task clustered paired bootstrap CI；
- 同时报告 reward、agreement、invalid、timeout、token 与 latency；
- 所有环境使用一个预注册 final version，不允许事后挑 variant。

### P2：必要 ablation 与机制实验

最小 ablation 集：

| Ablation | 回答的问题 |
|---|---|
| No belief/static prior | 动态 belief 是否必要？ |
| Frozen belief | 增益来自初始化还是 update？ |
| Shuffled/wrong belief | planner 是否真的依赖 belief，错误时是否受损？ |
| Oracle belief | 当前方法距离可用上限多远？ |
| No active probe | probe 是否提高后续收益？ |
| No reliability gate | 不确定 belief 是否需要安全退化？ |
| No action lock/validator | 收益有多少来自协议工程？ |
| No AWR residual | training 是否有独立贡献？ |
| Shared vs per-opponent state | multi-seller attribution 是否必要？ |

必须报告完整因果链：

```text
belief intervention
→ belief/calibration change
→ candidate rank or action flip
→ deal/price/contract change
→ final reward change
```

### P3：泛化实验

1. 至少两个 backbone：Qwen3-30B 加另一个开源模型或 GPT-4o；
2. unseen seller strategies：hardliner、conceder、behavioral、LLM seller；
3. repeated opponent：检验跨 episode belief 是否积累；
4. opponent switch/change point：检验过期 belief 是否能快速失效；
5. held-out issue ontology/contract template：检验 AgenticPay 是否只是模板记忆；
6. adversarial ambiguity：对方陈述与行为冲突、事实 issue 与 preference issue 混合。

### P4：训练路线

在新的大规模 RL 前，优先进行低风险训练：

1. **Issue-role/settlement eligibility classifier**
   从真实 AgenticPay trajectory 标注 negotiable preference、operational obligation、seller identity、quantity、price、terminal commitment，并允许 abstain。
2. **Belief calibration**
   训练预测接受概率/区间，而不是直接 SFT 含糊的 `flexibility_delta` 文本。标签来自 verified accept/reject、counteroffer 和最终合同。
3. **Conservative offline AWR/residual policy**
   只学习在规则 planner 候选之间重新排序或小幅调整阈值；使用 behavior-cloning/KL 约束和 pessimistic lower bound。
4. **Reward model/critic**
   必须以 Simple reward 和 AgenticPay official BuyerScore 为 target，belief metrics 只能作为 auxiliary loss。
5. **严格 train/dev/test 隔离**
   过去大量迭代使用同一 28/128 task 集观察结果，发表前需重新划分 unseen test，并冻结后一次性评测。

### P5：论文与代码发布

- 一条命令复现实验；
- Docker/conda lockfile；
- 自动下载 benchmark 或固定上游 commit；
- README 说明 endpoint 与 OpenAI-compatible client；
- CI 运行 unit/integration smoke；
- 发布精简、脱敏 trajectory 和 manifest；
- 附录给出完整 prompt、candidate scorer、validator、失败类型和统计方法；
- 明确第三方代码 license，不直接重新授权上游 benchmark。

---

## 13. 推荐论文表格与图

### 主表 1：跨环境最终 reward

行：Direct、CoT、native baseline、Opponent Simulation、Ours。
列：Simple reward、CaSiNo P1/joint/agreement、AgenticPay BuyerScore/timeout、NegotiationArena 四 setting。

只使用冻结的一个 Ours，报告 paired CI 和 compute。

### 主表 2：belief usefulness ablation

行：continuous、static、frozen、shuffled、wrong、oracle、no-gate。
列：belief calibration、action flip、reward、invalid、timeout。

### 图 1：belief-to-action causal pipeline

展示 evidence → calibrated belief → candidate frontier → action lock → official outcome。

### 图 2：identifiability–usefulness phase diagram

横轴为 belief 可识别程度，纵轴为 planner 对 belief 的敏感度，颜色为 reward gain。预期说明：只有 belief 可识别且能改变正确动作时才有收益。

### 图 3：AgenticPay topology scaling

从 single 到 multi-product、multi-seller、multi-buyer、full topology，展示 BuyerScore、agreement、timeout 和 protocol error。

### Case study

各给出一条：

- successful active probe；
- wrong belief 被 reliability gate 拒绝；
- multi-seller 正确切换并签订完整合同；
- failure case：错误等待/过度 concession/未被 focal buyer 选中。

---

## 14. 代码健康与当前验证

在 2026-09-07，对 clean repo 的相关测试执行：

```bash
PYTHONPATH=/work5/qixint/negotiation/NegotiationArena:/work5/qixint/negotiation \
/work5/qixint/miniconda3/envs/research/bin/python -m pytest -q \
  tests Simple_Env/tests NegotiationArena/tests AgenticPay_Env/tests
```

结果：

```text
181 passed in 3.53s
```

如果不设置上述 `PYTHONPATH`，NegotiationArena 的两个测试会因 `arena_integration` 无法导入而在 collection 阶段失败。这是运行入口/packaging 问题，不是测试逻辑失败。发表前应通过 `pyproject.toml` 或 editable install 消除手动 `PYTHONPATH`。

---

## 15. 最终判断

### 已经具备的成果

- 一套清晰的 belief-to-action framework 原型；
- shared canonical schema、planner、belief 和 validator；
- Simple 上统计显著优于 CoT/full 的完整结果；
- AgenticPay multi-seller 上显著优于 CoT/full 的子集结果；
- CaSiNo 上稳定提高 focal score 的全 split point result；
- NegotiationArena 上大量 causal variants 和直接 Opponent Simulation 对比；
- 丰富的负结果，清楚揭示 calibration 不等于 usefulness；
- 181 个通过的自动测试和较完整的研究轨迹。

### 尚不足以投稿时直接声称的内容

- 一个同参数、同代码路径的 universal agent 在所有 benchmark 都超过 baseline；
- continuous belief update 本身必然提高 reward；
- V60 已在 AgenticPay all-231 上有效超过所有 baseline；
- AWR 相比 V5.9 已有统计显著增益；
- NegotiationArena 所有角色统一超过 Opponent Simulation。

### 最短发表路径

1. 冻结和提交当前代码；
2. 完成 AgenticPay 修复后的 paired all-116/all-231 与 focal multi-buyer；
3. 给 CaSiNo 增加多 rollout paired CI；
4. 在 Simple、CaSiNo、AgenticPay、NegotiationArena 上冻结同一方法定义；
5. 完成 frozen/shuffled/wrong/oracle/no-validator/no-AWR ablation；
6. 增加一个第二 backbone 和 unseen opponent；
7. 以“belief usefulness and safe execution”而不是“belief accuracy”组织论文。

如果 AgenticPay clean full-suite 仍不能超过 Direct，建议收缩主张：以 Simple + CaSiNo 为主要 performance benchmark，NegotiationArena 和 AgenticPay 用于展示 topology stress test、失败分析与安全退化；如果 V60 在 all-116/all-231 的 paired CI 也稳定为正，则可以形成较强的跨环境论文故事。
