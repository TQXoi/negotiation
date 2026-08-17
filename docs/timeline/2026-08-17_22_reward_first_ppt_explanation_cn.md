# Reward-First Negotiation Belief–Planner PPT 详细中文解读

对应 PPT：`/work5/qixint/8.17/NEGOTIATION_BELIEF_PLANNER_REWARD_FIRST_FRAMEWORK_RESULTS_PLAN_EDITABLE_EN.pptx`
生成脚本：`/work5/qixint/8.17/build_reward_first_framework_update_ppt.py`
PPT 共 18 页；第 1–7 页继承此前 benchmark / related-work 背景，第 8–18 页为 2026-08-17 新增的 reward-first framework、结果和计划。

## 1. 整份 PPT 想讲什么

这份 PPT 的核心并不是“我们已经在所有环境上超过 baseline”，而是把研究问题重新收紧为一个可证伪的命题：

> 在只修改 buyer、固定 seller、使用 benchmark 原生 scorer 的条件下，一个跨环境共享的 opponent belief state，只有在 planner 能把它转化为 probe、concession 和 accept/continue 决策变化，并最终提升 buyer 的原生任务收益时，才算真正有用。

这一定义刻意把三类指标分开：

1. **最终目标指标**：Simple / AmazonHistoryPrice 的 native reward，以及 AgenticPay 的 official buyer score。
2. **机制解释指标**：belief calibration、coverage、action flip、oracle/wrong/shuffled belief、accept availability。
3. **安全和工程指标**：格式正确率、action/render consistency、buyer IR、timeout、runtime failure。

后两类指标可以解释失败原因，但不能替代第一类指标。例如 belief MAE 下降但 reward 不升，不能晋级；成交率上升但成交合同被 official scorer 判为 infeasible，也不能写成性能成功。

截至 PPT 制作时，证据呈现出非常清楚的不对称：

- **Simple Env 已有可信的正向结果**：V6.3 的 mean reward 为 0.5130，显著高于 CoT 和 legacy full framework。
- **AgenticPay 尚未成功**：早期 universal V1/V2 明显低于 direct/CoT；后续 V3/V4 修复了协议和 accept 边界，但 PPT 制作时 V4 只有 smoke 结果。
- 因此现在能声称的是“架构与因果诊断路径已经建立，Simple 有正向 end-to-end 证据”，不能声称“一个冻结 policy 已在多个环境全面领先”。

## 2. 阅读前必须知道的版本口径

### 2.1 第 1–7 页是继承的背景页

这些页面用于解释 benchmark 选择和相关工作冲击，并不是 8 月 17 日重新检索或重新验证的 literature review。页中“154 citations”等数字应理解为制作旧版时记录的背景信息，而不是本报告重新核验后的实时数字。

### 2.2 第 1/5 页的 “ACL benchmark” 实际应为 ANL

上下文描述的是 alternating-offers、hidden reservation value、非自然语言协议，对应的是 **Automated Negotiation League（ANL）**，不是 ACL 会议 benchmark。这是 PPT 文本中的缩写笔误。

### 2.3 第 16 页的实验状态已经过时

PPT 制作时 Iteration 002 只保存了 `6/24` 条 dev records，所以页面谨慎地只报告 Task4 smoke。PPT 制作后，V4 的固定 8-task × 3-seed dev 已完成 `24/24`：

| Variant | AgenticPay dev buyer score |
|---|---:|
| Direct prompt | 50.0790 |
| CoT prompt | 45.9631 |
| Legacy full framework | 34.6891 |
| V3 action-consistent | 33.1317 |
| V4 terminal accept guard | 41.5588 |

因此 Task4 smoke 的因果结论仍然成立，且 V4 aggregate point estimate 已经高于 V3 和 legacy full；但 V4 仍低于 direct/CoT，只有 8 个 task clusters，paired CI 仍跨 0。

## 3. 第 1–7 页：为什么重新选择 benchmark 和研究定位

### Slide 1：Decision-Calibrated Belief-to-Action Planning

第一页是问题背景。它承认 2026 年相关工作已经开始研究 continuous opponent modeling，因此“连续更新 belief”本身已不足以构成 novelty。我们的研究必须回答更强的问题：belief 是否被校准、是否真正改变 action、是否在不同协议中提高效用。

页面同时给出三个 benchmark 判断：

- **LLM-Deliberation**：知名度高，但复现研究指出 agent 名称和静态角色先验已经能解释相当部分结果，交互过程未必是主要贡献来源，因此不适合作为 belief-through-interaction 的主证据。
- **NegotiationArena**：任务简单，但 Opponent Simulation 在这里给出了可直接比较的强 baseline；适合作为兼容性和机制审计环境。
- **ANL**：协议干净、隐藏变量清晰，适合 posterior calibration 和 planner mechanism test；由于主要不是自然语言交互，不适合单独支撑“通用自然语言谈判框架”的主 story。

这一页的作用是说明：不能依赖单一 benchmark，需要用不同环境分别验证语言、belief 可识别性、planner 因果作用和跨环境迁移。

### Slide 2：Opponent Simulation 的方法

这页介绍直接竞争方法 Opponent Simulation。其计算流程是：

1. 用真实 negotiation trajectories 训练一个 auxiliary opponent LLM；
2. 当前 agent 生成多条候选消息；
3. 对每条候选消息，让 opponent simulator 继续 rollout；
4. 用 rollout 的预期 payoff 选择当前 action。

它的优势是 belief 不是只写在 prompt 中，而是通过模拟未来对手响应直接参与 action selection，因此是一个真正的 belief-usable planner baseline。

它对我们造成的冲击在于：如果我们的贡献只被描述成“维护 continuous belief，然后让 planner 使用”，会和该工作高度重叠。我们必须把贡献落到更明确的差异上：factorized calibrated posterior、有限成本、可审计 action channel、irrational/change-point opponent，以及同一 core 跨 price 与 multi-issue contract 的迁移。

### Slide 3：Opponent Simulation 对 novelty 的冲击

这页强调二者的相似与差异：Opponent Simulation 用 auxiliary LLM 表示对手，并通过在线 rollout 持续适配；我们的潜在差异不是再做一个更大的 simulator，而是维护显式、可校准、可干预的 posterior。

PPT 给出的预期差异包括：

- Opponent Simulation 每个 action 需要多次模型调用，推理成本高；我们的 structured planner 主要使用显式状态和候选评分，只允许有限 semantic/naturalization 调用。
- 对手 simulator 容易绑定到特定游戏和固定对手分布；我们的目标是用 canonical state 与 thin adapter 跨环境迁移。
- 我们显式处理不确定性、非理性行为、response policy 与 utility 的混淆，以及 opponent regime change。

但这页也应谨慎阅读：“更 universal”目前是待验证目标，不是已经由 AgenticPay 结果证明的事实。

### Slide 4：其他相关工作

这一页列出三类邻近方法，用于说明我们的工作同时受到 opponent inference、hierarchical planning 和 structured memory 三条线的影响：

- **MIND**：在 multi-agent travel planning 中推断 willingness，说明意愿/偏好推断已进入多人协调任务。
- **HAR**：采用 hierarchical planning 处理 negotiation task，与我们的 planner decomposition 有关联。
- **GENEGO**：在 CaSiNo 中使用 structured memory 和 pairwise preference inference，适合作为 CaSiNo 上新增 baseline。

这些工作意味着不能把“有 memory”“推断 preference”或“分层 planner”单独写成 novelty。更合理的贡献单元是：统一 belief schema、公开事件边界、belief-to-action 因果验证、跨协议迁移和 reward-first promotion。

### Slide 5：Benchmarks Exploration

第五页重新给出 benchmark 组合结论：AmazonHistoryPrice/Simple 与 CaSiNo 本身已经是社区认可的自然语言 negotiation benchmark，但若只在单一环境优化，很容易被认为是环境特化。

各环境在论文中的合理分工应当是：

- **Simple / AmazonHistoryPrice**：单 buyer–seller、单商品价格谈判，适合验证 reward、主动 probe、repeated opponent 和 change-point。
- **CaSiNo**：自然语言 multi-issue 分配，适合检验 preference ranking、Pareto/individual utility 和 multi-issue trade-off。
- **NegotiationArena**：与 Opponent Simulation 做严格 matched comparison，并进行 wrong/shuffled/oracle belief audit。
- **ANL**：用结构化协议验证 reservation posterior、coverage 和 sequential planning，不承担自然语言主结论。
- **AgenticPay**：从单商品扩展到完整 contract、多 seller / 多 buyer，是 universal framework 最困难也最关键的迁移测试。

### Slide 6：NegotiationArena 的定位

标题“simple tasks, essential direct comparison”准确表达了该环境的角色：它很简单，不能单独证明 universal negotiation intelligence，但因为 Opponent Simulation 使用了明确的 repeated-game setting，所以是不可回避的直接 baseline 环境。

页面中的两张原论文表展示了两类游戏：

1. **Sell & Buy**：seller 拥有 1 个商品 X，buyer 拥有 100 ZUPs；seller 最大化价格、buyer 最小化价格；示例 seller/buyer valuation 分别为 40/60；最多 10 轮。
2. **Resource Exchange**：Player 1 初始有 25X/5Y，Player 2 有 5X/25Y；双方交换资源并最大化自己的总效用；最多 8 轮。

这解释了为什么 NegotiationArena 结果容易被 opening policy、先后手和简单 reservation threshold 主导。它适合回答“belief 是否改变 action”，但不适合单独回答复杂自然语言 multi-issue contract 中是否泛化。

### Slide 7：ANL 的定位

这一页把 ANL 2024–2026 分成三种不同诊断任务：

| League | 信息结构 | 最适合验证的机制 |
|---|---|---|
| ANL 2024 | utility shape 已知，reservation value 隐藏 | RV posterior、区间 coverage、oracle regret |
| ANL 2025 | sequential one-to-many、多个 deal 相互依赖 | continuation value 与 global planning |
| ANL 2026 | utility 与 reservation 更未知，隐藏自身信息有价值 | exploration 与 self-information leakage 的权衡 |

ANL 的价值是 evaluator truth 清晰，能严格判断 posterior 和 regret；局限是没有自然语言理解与生成，因此应作为 calibration/mechanism appendix，而不是主 benchmark。

## 4. 第 8–12 页：reward-first framework 到底是什么

### Slide 8：Updated thesis

这是整份 PPT 最重要的立场页。它冻结四项实验条件：只修改 buyer、使用固定 native seller、不改 scorer、按 task/scenario cluster 评估。这样可以避免把 seller 变化、scorer 修改或不同 task mix 误写成 buyer framework 提升。

优化目标只有两个：Simple native reward 和 AgenticPay buyer score。deal、timeout、belief calibration 等只负责解释“为什么”。页面给出的当前 verdict 是：Simple 已经出现可信正向 anchor；AgenticPay 仍落后 direct/CoT；主要 gap 位于 accept/continue 和 multi-issue adaptation。

页面底部的 falsifiable claim 可拆成四个必要条件：

1. 存在统一 canonical opponent state；
2. state 是从公开轨迹更新且有 calibration/reliability；
3. planner 的 probe、concession 或 accept boundary 因该 state 发生变化；
4. 变化最终提高 native buyer utility。

缺少任何一环，都不能称为完整 belief-to-action 成果。

### Slide 9：六个显式接口

当前 executable framework 由六个接口串联：

1. **Adapter**：把环境历史映射到 `CanonicalState`，同时提供本方精确 utility 和环境合法 action schema。
2. **Belief**：维护 utility preference `θ` 与 response policy `φ`，而不是只输出一段自然语言画像。
3. **Candidates**：枚举合法 offer、accept、quit，以及可能的 probe / term trade-off；planner 不直接自由生成最终结构化 action。
4. **Planner**：根据 buyer utility、agreement probability、information value 和 timeout/quit risk 排序。
5. **Renderer**：把 selected candidate 转成自然语言，但 action 被锁定，renderer 不能改价格或合同字段。
6. **Validator**：重新 parse 输出，检查 exact contract、buyer IR、action semantics 和 selected candidate id；失败时 deterministic repair/render。

Persistent state 以 `(session, counterparty)` 为 key，维护拒绝 frontier 与 old/new regime mixture。每一步都会写入 trace：belief before/after、candidate scores、selected id、validation、公开 response 和 reward。部署分支不能读取 seller cost、seller utility weights、最终 outcome 或 benchmark hidden label。

对应核心代码：

- `negotiation/framework/schemas.py`：canonical state、belief、candidate 和 trace schema；
- `negotiation/framework/engine.py`：完整在线 orchestration；
- `Simple_Env/buyer/universal_framework.py`：Simple adapter 与 variants；
- `AgenticPay_Env/buyer/universal_framework.py`：AgenticPay contract adapter 与 buyer agent。

### Slide 10：Belief state

belief 被分解成四块：

- **Preference `θ`**：reservation interval、issue-option scores，描述哪些 outcome 对对手可能有用。
- **Response policy `φ`**：accept/counter/quit、patience、rejected frontier，描述对手在当前策略状态下会怎样回应。
- **Reliability `ρ`**：根据 evidence count 和 source strength 控制 posterior 对 planner 的影响强度。
- **Regime `z`**：old/new mixture 与 change probability，允许 repeated opponent 在偏好或策略切换后保留多假设，而不是粗暴覆盖历史。

“utility”和“behavior”必须分开，因为拒绝一个有利可图的 offer 可能意味着真实 reservation 较高，也可能只是 strategic aspiration 较高。框架首先信任正式的 ACCEPT/REJECT/COUNTER，其次才使用经过边界约束的 semantic extraction。弱 evidence 下，posterior 会向宽而安全的 prior 混合。

frozen、wrong、shuffled、oracle belief 的作用是因果诊断：如果更换 belief 内容并不改变 action，说明 planner 没有真正使用 belief；如果 action 改了但 reward 更差，说明 belief 或其使用方式不可靠。

### Slide 11：Planner V2

页面给出概念评分式：

`Q(a) = P(agreement | a,b) × U_buyer(a) + information value − quit/timeout risk`

其中：

- `a` 是一个环境合法 candidate；
- `b` 是当前 factorized belief；
- `P(agreement | a,b)` 来自结构先验、公开 response evidence 和 reliability gate；
- `U_buyer(a)` 是 adapter 精确计算的本方效用；
- information value 只在剩余 horizon 足够且不牺牲 buyer safety 时奖励 probe；
- risk 项阻止无意义拖延和末轮继续试探。

Planner V2 重点控制三条直接影响 reward 的边界：

1. **Probe boundary**：只有不确定性会改变最优 action 且仍有后续回合时才 probe。
2. **Concession frontier**：一个区域被正式拒绝后，如果存在安全 successor，就不应原样重复报价。
3. **Terminal accept guard**：已有完整、buyer-IR 为正的 seller proposal 时，比较其确定 accept value 与保守 counter EV；在 deadline 或连续失败后可以接受。

Renderer invariant 则保证 `OFFER` 文本必须表达 propose，`ACCEPT` 必须复制 exact outstanding package。PPT 当时明确提出：在拥有真实 reward-backed labels 前，不应仅凭 synthetic oracle delta 宣称训练成功。

### Slide 12：Evaluation protocol

正式评测采用两套主任务：

| Environment | Coverage | Primary metric | Bootstrap cluster |
|---|---:|---|---|
| Simple / AmazonHistoryPrice | 128 scenarios × 3 rollouts | native reward | scenario |
| AgenticPay single28 | 28 tasks × 3 seeds | official buyer score | task |

为什么必须 cluster bootstrap：同一 scenario/task 的三个 rollout/seed 不是三个完全独立任务。正确做法是先在 cluster 内平均，再对 scenario/task cluster bootstrap，从而避免把重复 rollout 当成独立样本夸大显著性。

解释性 audit 包括 deal、timeout、IR、action mismatch、accept availability、calibration 和 action flip。资源控制使用本地 Qwen3-30B，并限定 GPU1/GPU2；seller 与 scorer 不变。每个 iteration 必须保存 hypothesis、修改前后源码、精确命令、原始 trajectory 和结果报告。

#### AgenticPay BuyerScore 的含义与公式

可直接放进 PPT 的英文简写为：

> **BuyerScore:** discounted buyer-side outcome quality. For a feasible deal,
> `BuyerScore = γ^(t−1) × (Db + Wb × r_b + Eb)`, where `r_b` is the buyer's
> normalized utility; for an infeasible or no-deal outcome,
> `BuyerScore = −Fb × (1 − γ^(t−1))`. In the single28 setting,
> `Db=10, Wb=80, Eb=10, Fb=15, γ=0.99`; higher is better.

在纯价格任务中，令 buyer 私有最高价为 `B`、seller 私有最低价为 `C`、成交价为 `P`，则
`r_b=(B-P)/(B-C)`。只有确实达成协议且 `C≤P≤B` 时才按成功公式计分。在 multi-issue
contract 任务中，`U_b = v_base - P + continuous/discrete term utilities`，再用理论最大联合
surplus `Z_max` 归一化为 `r_b=U_b/Z_max`；同时要求合同完整合法，且 evaluator 计算的
`U_b≥0`、`U_s≥0`。因此，一个在对话层面显示 agreed 的合同，如果违反任一方隐藏 IR，仍会按
失败公式计分。`γ^(t−1)` 使更晚成交折价；默认权重下，成功分主要由归一化 buyer utility
决定（80% 权重），deal 和 efficiency 各占 10%。这也是为什么 deal rate、raw buyer reward 或
“seller 口头接受”都不能替代 official BuyerScore。

## 5. 第 13–16 页：结果如何理解

### Slide 13：Simple Env 正式全集结果

这是目前最强的 positive result。相同的 128 scenarios × 3 rollouts 下：

| Buyer variant | Mean reward | 含义 |
|---|---:|---|
| Direct prompt | 0.0961 | 70.1% episode 为零 reward，稳定性差 |
| CoT prompt | 0.4065 | 强 prompt baseline |
| Legacy full framework | 0.4464 | 历史框架的同批次 anchor |
| Universal V5.9 | 0.4981 | 当前最好冻结机制之一 |
| Universal V6.3 AWR | 0.5130 | 最高 point estimate |

paired contrast 的正确解释是：

- V6.3 相对 CoT：`+0.1066 [0.0581, 0.1560]`，CI 完全高于 0，可以声称显著改善。
- V6.3 相对 legacy full：`+0.0666 [0.0203, 0.1118]`，同样有可信正提升。
- V6.3 相对 V5.9：`+0.0149 [-0.0296, 0.0589]`，CI 跨 0，只能说 point estimate 更高，不能声称显著更优。
- V6.3 相对 per-scenario best envelope：`+0.0060 [-0.0390, 0.0501]`，也没有显著差异。这里的 envelope 是每个 scenario 事后选最好已有 policy 的不可部署参考，并不是部署时可获得的真实 oracle policy。

因此 Simple 的严谨结论是：V6.3 已显著超过单个 CoT 和 legacy full baseline；V6.3 与 V5.9 谁更好尚不确定。`0.5130` 是 native reward，不是成交率或 belief score。

### Slide 14：AgenticPay 初始正式结果

AgenticPay full single28 为 28 tasks × 3 seeds。早期结果明确失败：

| Buyer variant | Buyer score | Deal | Timeout |
|---|---:|---:|---:|
| Direct prompt | 32.953 | 90.5% | 9.5% |
| CoT prompt | 31.720 | 90.5% | 9.5% |
| Legacy full | 21.311 | 90.5% | 9.5% |
| Universal V1 | 15.731 | 81.0% | 19.0% |
| Universal V2 AWR | 18.177 | 94.0% | 6.0% |

V2 虽把 deal 提高到 94%、timeout 降到 6%，buyer score 仍远低于 CoT。其相对 CoT 的 paired delta 是 `-13.542 [-25.867, -1.782]`，区间完全低于 0，说明不是偶然小波动，而是明确负迁移。

trajectory audit 找到一个严重 protocol bug：V1 有 `865/977` 个 turn、V2 有 `502/591` 个 turn，structured action 是 OFFER，但 natural-language message 却说成 ACCEPT。这会让 seller 对策略意图产生错误理解。它说明 action lock 仅锁住结构字段仍不够，语言语义也必须和 action type 一致。

更重要的是，V2 的 offline AWR checkpoint 优化的是 synthetic oracle delta，而不是 AgenticPay online buyer score。这个结果不能包装成“训练有效”，只能作为为什么必须转向真实 trajectory reward labels 的反例。

### Slide 15：Iteration 001——action consistency

V3 只改 renderer 与 semantic validator，不改 belief 和 planner。这是严格的单变量实验。

固定 8-task × 3-seed subset 上，V3 buyer score 从 V1 的 19.759 提升到 33.132；action mismatch 从 `158/177` 降到 `0/329`。这证明 protocol correctness 对 end-to-end reward 有实际作用。

但 V3 仍低于 direct 50.079 和 CoT 45.963；deal 只有 62.5%，timeout 达 37.5%。V3 相对 V1 的 `+13.37` 和相对 direct 的 `-16.95` 都具有很宽、跨 0 的 task-clustered CI，所以不能作强统计结论。

真正指导下一步的是行为诊断：轨迹中出现 236 次可以接受的 seller proposal，planner 只选择 accept 12 次，并出现 53 次 late deferral。这说明 renderer 已修好，但 planner 的 accept/continue boundary 仍错误，因此下一版应该只改该边界。

### Slide 16：Iteration 002——terminal accept guard

PPT 展示 Task4 / seed 20260817 的 causal smoke：

- V3：21 轮无协议，score `-2.731`；
- V4：第 14 轮接受 exact seller package，score `52.263`；
- 成交合同没有 IR violation；
- 唯一机制变化是 proposal-backed terminal accept guard。

其因果意义比“换一个更长 prompt 后 reward 上升”更强：belief、candidate set、renderer、seller 和 scorer 都冻结，只有 accept/continue ranking 改变。因此可以把 reward 变化归因到决策边界。

但 PPT 页面制作时只有 `6/24`，所以只写 smoke。现在完整 dev 已证明 V4 aggregate mean 为 41.5588，高于 V3 33.1317 和 legacy full 34.6891；deal 从 62.5% 增至 87.5%，timeout 从 37.5% 降至 12.5%。不过它仍低于 direct/CoT，CI 跨 0，所以应表述为“成功修复一类 failure mode 并得到正向 point estimate”，不能写成 SOTA。

## 6. 第 17–18 页：novelty 与实验计划

### Slide 17：相对 Opponent Simulation 的 novelty

这一页不是声称“我们已经比 Opponent Simulation 更好”，而是规定需要通过实验赚取的 novelty：

| 维度 | Opponent Simulation | 我们的目标 |
|---|---|---|
| Opponent state | auxiliary LLM simulator | 显式 `θ + φ + ρ + regime` |
| Decision | candidate rollout search | legal candidate EV + probe/frontier/accept |
| 推理成本 | 每个 action 多次 LLM rollout | structured core + bounded semantic calls |
| 迁移单元 | game-specific simulator/prompt | canonical state + thin adapter |
| 审计 | 主要看 payoff | reward + calibration + wrong/shuffled/oracle |

真正的新意不应是“我们也建模 opponent”，而应是以下组合：

1. 将 durable utility、strategic response policy、reliability 和 change-point 显式分解；
2. belief 通过受约束 candidate/planner 改变 action，决策路径可以重放和干预；
3. 不依赖大量 future rollout，在有限推理成本下工作；
4. 同一个冻结 core 能通过 thin adapter 跨单价格、multi-issue contract 和多人协议；
5. 不只报告 payoff，还用 wrong/shuffled/oracle 与 action-flip 检查 belief 是否真的有因果作用。

但页面底部也给出严格条件：只有同一冻结 core 在 Simple、AgenticPay 和兼容性测试中都提高 buyer utility，跨环境 novelty 才真正成立。当前 AgenticPay 尚未满足这一条件。

### Slide 18：Reward-first 计划

计划采用单机制迭代：每次只改变一个可解释机制，保存轨迹，然后回到两个完整环境验证。

原 PPT 五阶段为：

1. 完成 V4 的 8-task × 3-seed paired dev；
2. 对齐 behavioral frontier 与 issue-tradeoff posterior；
3. 在 Simple validation40 × 3 检查迁移且不改 seller；
4. 从 AgenticPay single28 扩展到 1 buyer × 2 sellers / 2 buyers × 2 sellers，并检查 contract IR；
5. 冻结最终 policy，运行 Simple128 × 3 与 AgenticPay28 × 3。

Promotion gate 是：mean 必须高于 legacy full，报告 paired CI 和 task coverage，且不能产生 IR regression。只有真实 trajectory reward labels 足够时才启动 conservative offline RL / AWR，并保留 base fallback。最终 baseline 包含 direct、CoT、legacy full、best universal、wrong/shuffled belief，以及适用环境中的 Opponent Simulation。

PPT 制作后的当前进展是：阶段 1 已完成；正在进行的 V5 只新增 verified response frontier，用于修复 V4 在 Task6 中反复提交已拒绝报价的问题。Task6 smoke 已从 21 轮 timeout 改为 6 轮成交，但 fixed seller 提出了违反其自身隐藏 IR 的合同，导致 official score 仍为负。这类 seller-private truth 不能泄漏给 online buyer，因此需要通过完整 task-clustered dev 判断 V5 是否总体提高 buyer score，而不能为单条轨迹加 oracle rule。

## 7. 关键术语与指标解释

### Direct、CoT、legacy full、universal variant

- **Direct prompt**：不显式维护结构化 belief/planner 的直接生成 baseline。
- **CoT prompt**：要求模型显式思考后生成 action 的强 prompting baseline。
- **Legacy full framework**：迁移研究之前已有的完整框架，作为同批次历史 anchor。
- **Universal V1/V2/...**：基于 canonical core 的版本；版本号只表示该环境分支的迭代顺序，不表示所有环境共享完全相同的训练 checkpoint。

### Native reward 与 buyer score

Simple 的 `0.5130` 与 AgenticPay 的 `41.5588` 不在同一量纲，不能横向比较。前者是 Simple native/normalized reward；后者是 AgenticPay official buyer score。只能在同一环境、相同 scenario/task 与 seed 下做 paired comparison。

### IR

Individual Rationality 表示成交至少不差于 outside option。在线 buyer 可以检查自己的 IR；seller private IR 属于隐藏 evaluator truth，不能输入 deployed buyer。若 native seller 自己提出一个违反其私有 IR 的合同，official scorer 可能出现 agreement 与 score 不一致，此时应报告 `score_success_mismatch`，而不是让 buyer 偷看 seller utility 修复。

### Calibration 与 decision value

Calibration 衡量 posterior 是否与真实 hidden parameter/response frequency 一致；decision value 衡量使用该 posterior 后是否提高任务效用。好的 calibration 不必然提高 reward，错误但保守的 belief 有时也可能偶然得到高 reward，因此二者必须分别报告。

### Wrong / shuffled / oracle belief

- **wrong-confident**：把系统性错误 belief 高置信地交给 planner，测试 planner 是否会被错误模型伤害；
- **shuffled**：把其他 trajectory 的 belief 随机配给当前状态，测试 belief 内容是否重要；
- **oracle**：只在 evaluator/diagnostic 中提供真实隐藏参数，测量 planner 可利用的上限；不能作为 deployed result。

## 8. 一条适合汇报时使用的中文讲述主线

可以用以下五句话串起整份 PPT：

1. “连续 opponent modeling 已不是空白，因此我们的研究问题从‘有没有 belief’转向‘belief 是否校准、是否被 planner 因果使用、是否跨环境提高 buyer utility’。”
2. “我们把环境差异放入 adapter，把公共轨迹变成 factorized belief，再让受约束 planner 选择合法 candidate，并通过 action lock 保证语言不改写决策。”
3. “Simple128 × 3 上，V6.3 达到 0.513，显著超过 CoT 和旧 full framework，这是当前最可信的正向 anchor。”
4. “AgenticPay 初始迁移失败暴露了两个真实问题：语言 action 与结构 action 不一致，以及 planner 明明看到可接受合同却继续拖延；V3/V4 分别修复了这两个问题，但仍未超过 direct/CoT。”
5. “因此 novelty 不是某个环境上的规则堆叠，而是一个可审计、低成本、可跨协议迁移的 belief-to-action core；最终是否成立，只由同一冻结设计在 Simple 和 AgenticPay 的 buyer reward 决定。”

## 9. 当前可以与不可以从 PPT 得出的结论

### 可以得出的结论

- framework 已经形成 adapter → canonical state → factorized belief → legal candidates → planner → action lock/validator 的完整可执行链；
- Simple formal full test 已给出相对 CoT 和 legacy full 的可信正 reward delta；
- AgenticPay V1/V2 的迁移明确失败，且轨迹诊断找到了 action semantics 与 accept boundary 两类机制问题；
- V3 的 protocol repair 和 V4 的 terminal guard 都产生了可解释的行为变化；
- reward-first、单变量、paired clustered evaluation 是合理的后续研究协议。

### 不可以得出的结论

- 不能声称一个完全相同、完全冻结的 policy 已同时在 Simple 与 AgenticPay 领先；
- 不能声称 AgenticPay 已超过 direct prompt 或 CoT；
- 不能把 V6.3 相对 V5.9 的小幅 point estimate 写成统计显著提升；
- 不能把 calibration、deal rate、action-flip 或 synthetic oracle delta 当作最终 reward；
- 不能把 NegotiationArena 的简单任务胜利外推为复杂 multi-issue negotiation 的普适能力；
- 不能利用 seller private utility、最终 scorer label 或事后 outcome 调 deployed buyer。

## 10. 对应代码和实验记录

| 内容 | 路径 |
|---|---|
| Canonical schemas | `/work5/qixint/negotiation/framework/schemas.py` |
| Belief updater | `/work5/qixint/negotiation/framework/belief.py` |
| Planner 与 V4/V5 guards | `/work5/qixint/negotiation/framework/planner.py` |
| Universal engine | `/work5/qixint/negotiation/framework/engine.py` |
| Conservative AWR | `/work5/qixint/negotiation/framework/conservative_awr_planner.py` |
| Simple adapter/variants | `/work5/qixint/Simple_Env/buyer/universal_framework.py` |
| AgenticPay adapter/agent | `/work5/qixint/AgenticPay_Env/buyer/universal_framework.py` |
| AgenticPay variant registry | `/work5/qixint/AgenticPay_Env/buyer/variants.py` |
| Reward-first pipeline ledger | `/work5/qixint/universal_reward_research/PIPELINE_LEDGER_CN.md` |
| Iteration 002 完整结果 | `/work5/qixint/universal_reward_research/iteration_002_terminal_accept_guard/RESULTS_AND_NEXT_STEPS_CN.md` |
| Iteration 003 预注册与运行记录 | `/work5/qixint/universal_reward_research/iteration_003_verified_response_frontier/` |

## 11. 总结

这份 PPT 最重要的贡献是把此前容易分散的 belief、planner、benchmark 和训练尝试收束成一个严格的 reward-first 研究程序。它没有掩盖 AgenticPay 的负结果，而是利用负结果定位真实的决策链故障。当前最合理的论文 story 不是“continuous belief 在所有环境都有效”，而是：

> 我们提出一个 factorized、reliability-gated、可审计的 belief-to-action negotiation framework；它在 Simple 环境中已经产生显著 buyer-reward 增益，并通过 AgenticPay 的逐项单变量实验暴露和修复跨环境执行、accept boundary 与 multi-issue response-frontier 问题。最终 universal claim 以同一冻结 core 在两个环境上的 paired buyer utility 为判据，而不是以 belief 指标或环境特化规则替代任务收益。

这既是当前证据能够支持的最强表述，也是后续实验最容易被审稿人检验和证伪的表述。
