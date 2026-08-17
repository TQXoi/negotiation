# Universal Buyer Framework：实现与主结果（动态更新）

更新时间：2026-08-17

## 1. 当前 framework 怎样实现

共享执行链只有五步：

1. **Adapter** 把环境历史转为 `CanonicalState / NegotiationObservation`，并生成环境内合法的候选 action。Simple 入口是 `Simple_Env/buyer/universal_framework.py::SimpleEnvAdapter`；AgenticPay 入口是 `AgenticPay_Env/buyer/universal_framework.py::AgenticPayAdapter`。
2. **Belief store + updater** 按 `(session_id, counterparty_id)` 维护对手的 reservation 区间、response policy、拒绝 frontier、issue-option 偏好、旧/新 regime 权重。结构化 accept/counter/quit 是主证据；LLM 语义抽取只有在行为证据验证后才可小幅进入 belief。
3. **Candidate planner** 只在 adapter 提供的合法候选中排序。V5.9 的 frozen base 是 `BehavioralFrontierPlanner`；V6.3 在其上加载 Iter023 的 `ConservativeAWRPlanner`，只有预测 advantage 的 bootstrap LCB 过安全阈值时才覆盖 base action。
4. **Locked renderer** 可调用同一 LLM 生成不含数值/合同的简短谈判措辞，但 price、terms 和 action type 由 planner 锁定。
5. **Validator** 检查最终文本是否等于 locked candidate；失败时用 deterministic renderer 修复。所有 belief、候选分数、selected id 和修复信息写入 trajectory trace。

共享核心接口：

- schema：`negotiation/framework/schemas.py`；
- orchestration：`negotiation/framework/engine.py`；
- belief：`negotiation/framework/belief.py`；
- base planner：`negotiation/framework/planner.py`；
- offline-RL residual：`negotiation/framework/conservative_awr_planner.py`；
- Iter023 checkpoint：`Simple_Env/research_iterations/iteration_023_cross_environment_awr/results/awr_v1_preregistered/cross_environment_awr_checkpoint.json`。

注意：Iter023 当时的 offline gate 是 synthetic context 上的 `mean_oracle_delta_vs_base`，不是本项目现在冻结的两个 online 主指标。因此 V6.3 必须被视为待验证 candidate，不能因为 offline gate passed 就称为最优。

### 当前必须明确的跨环境不一致

Simple V5.9/V6.3 显式使用 `BroadPriorBeliefStore + StrategicReadinessMixtureBeliefUpdater + VerifiedSemanticBeliefUpdater`。AgenticPay V1/V2 当前构造 `UniversalNegotiationEngine` 时没有显式传这三个组件，因此实际落回默认 `BeliefStore + StructuredBeliefUpdater + SemanticBeliefUpdater`。这意味着当前 AgenticPay V2 只是共享 schema/planner/action lock，**还不是与 Simple 完全相同的 belief model**。在 Iteration 000 全集结束前保持不改；它是下一轮单改动的候选，但必须由主 buyer_score 和轨迹支持。

`persistent_opponent_memory` 会在一个 buyer object 内使用稳定 session id；但当前 Simple runner 每个 episode、AgenticPay single28 runner 每个 task 都重建 buyer/engine。因此正式全集中并没有跨 task posterior 污染，反而是 Iter023 checkpoint 中的 repeated-opponent/cross-session 特征基本未被激活。repeated-opponent 能力需要另设“同一 buyer object 连续面对同一 seller”的实验，不应把它当作当前主 reward 的已实现贡献。

## 2. 主结果相比 baseline 如何

唯一主指标：Simple Env `reward`、AgenticPay `buyer_score`。belief calibration、IR、deal、action flip 等只用于归因。

正式 Iteration 000 已完成：

- Simple：128 场景 × 3 rollouts × 5 variants；
- AgenticPay：28 tasks × 3 seeds × 5 variants；
- fixed seller，buyer-only variants；
- 按 scenario/task 聚类计算 paired bootstrap 95% CI。

| 环境 / 主指标 | CoT | Direct | Full framework | 当前候选 | 结论 |
|---|---:|---:|---:|---:|---|
| Simple `reward` | 0.4065 | 0.0961 | 0.4464 | V5.9 0.4981；V6.3 0.5130 | V6.3 对 CoT `+0.1066`，95% CI `[+0.0581,+0.1560]`；对 full `+0.0666`，CI `[+0.0203,+0.1118]`。但对 V5.9 仅 `+0.0149`，CI 跨 0。 |
| AgenticPay `buyer_score` | 31.7199 | **32.9531** | 21.3108 | V1 15.7307；V2 18.1775 | V1/V2 均未超过 prompt baseline；V2 对 V1 `+2.4468`，CI `[-6.4345,+10.6209]`，不能认为提升。 |

因此初始全集给出的可信结论是：Simple 已有一个胜过单个 CoT/full baseline 的
framework anchor；AgenticPay 尚失败，不能用 belief/calibration 分数掩盖主 reward
落后。完整 task/scenario-clustered bootstrap 见
`iteration_000_full_baseline/analysis/PRIMARY_METRICS.md`。

AgenticPay 的可验证轨迹审计进一步发现，V1 的 977 个 locked-offer turn 中有
865 个公开文本却说 “accept/agree”，V2 为 502/591。Iteration 001 因此只新增
`universal_framework_v3_action_consistent` 修复 renderer/validator，不修改 belief、
candidate 或 planner；其 dev 结果必须由 `buyer_score` 决定是否晋级。

## 实验与版本记录

- 预注册协议：`universal_reward_research/iteration_000_full_baseline/PREREGISTERED_PROTOCOL_CN.md`；
- 启动命令：同目录 `run_simple_full.sh`、`run_agenticpay_full.sh`；
- 原始调用/轨迹：同目录 `runs/`；
- 主指标输出：同目录完成后生成的 `analysis/PRIMARY_METRICS.md` 与 `primary_metrics.json`。
- Iteration 001：`iteration_001_action_consistent_renderer/`，其中包含预注册假设、
  before/after source、固定命令、测试和原始轨迹。

后续每个 iteration 只允许一个方法改动，并保存：`HYPOTHESIS_CN.md`、代码 before/after snapshot、完整命令、raw trajectories、`RESULTS_AND_NEXT_STEPS_CN.md`。候选只有在两个主指标均不低于 `max(CoT, full_framework)` 且满足预注册 CI 规则时才能晋级。
