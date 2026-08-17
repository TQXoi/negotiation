# Universal Belief–Usable Negotiation Framework：迁移、迭代与下一阶段完整总结

更新时间：2026-08-17
范围：NegotiationArena → Simple Env / AmazonHistoryPrice → AgenticPay
状态口径：只把已有原始结果支持的结论写成“结果”；尚未完成的在线实验明确写成“计划”。

## 1. 执行摘要

本阶段最重要的进展不是在某一个 benchmark 上继续增加特化规则，而是逐步形成了一个可跨环境复用的 negotiation pipeline：

`public interaction → canonical state → factorized continuous belief → feasible candidate set → belief-usable base planner → conservative learned residual → action lock → natural-language realization`

当前应冻结为下一阶段候选的方法是：

- belief：`StrategicReadinessMixtureBeliefUpdater`，把对方 latent utility、response policy 与 strategic readiness 分开维护，并只消费公开可验证事件；
- base planner：`BehavioralFrontierPlanner`，显式建模对方的行为拒绝边界、deadline、accept/continue 与 offer dominance；
- learned planner：`ConservativeAWRPlanner`，只学习相对 base action 的 advantage，并在 bootstrap lower-confidence bound 足够大时 override；
- environment adapter：分别负责 Simple Env 与 AgenticPay 的状态归一化、合法候选枚举、utility/IR 检查和协议渲染；
- execution safety：planner 选择 candidate 后锁定语义，LLM 只负责自然语言表述，validator 检查最终 action 与 candidate 一致。

这个版本在 Iteration 023 的全新离线 audit 上，相对冻结 base planner 的 mean oracle advantage 为 **+0.0573**，paired cluster bootstrap 95% CI 为 **[+0.0324, +0.0823]**，override rate 为 **36.39%**，harmful override rate 为 **6.39%**，first-action flip 为 **0%**。但这仍是离线 counterfactual evaluator 结果，尚不能写成 Simple Env 或 AgenticPay 的最终在线 benchmark 提升。

目前 reward 提升不理想的主要原因也已经定位：Iteration 023 的 AgenticPay multi-issue 数据来自手工生成的六类 synthetic contract templates，而非 AgenticPay 的真实 LLM trajectory；它虽然在该域的离线 oracle delta 达到 +0.0796，但 harmful override 为 11.11%，说明 candidate ranking 仍没有充分学习真实 seller response 与合同分布。Iteration 024 因此转向真实 trajectory-derived anchors 与官方源码 contract templates，而不是继续在旧 audit 上调 margin。

## 2. 为什么从 NegotiationArena 迁移

NegotiationArena 对工程和因果诊断很有价值：它包含 buyer/seller scalar bargaining 与 resource exchange，能低成本运行 frozen、wrong、shuffled、oracle、opponent-switch 和 action-flip 测试。但它也暴露了三个局限：

1. action space 很小，规则化 planner 或 Opponent Simulation 容易利用环境固定结构；
2. 对手行为往往在少数回合内高度可预测，continuous belief 不一定比 frozen prior 更好；
3. 同一种策略不能同时支配 scalar buyer、scalar seller、resource first mover 与 second mover，说明环境特化 heuristic 不是 universal framework。

因此 NegotiationArena 被重新定位为：协议稳定性、belief intervention、repeated-opponent/change-point 的机制测试床，而不是唯一主 benchmark。Simple Env 用于更广的自然语言单商品买卖与 category shift；AgenticPay 用于 multi-issue、multi-product、multi-buyer/multi-seller 迁移。

## 3. NegotiationArena：版本、尝试与结论

### 3.1 工程基线与 V2

首先修复原库 parser、response tag、serialization、context-length 与 action-lock 问题，使 120/120 validation episodes 有效、0 errors、format success 100%。V2 把 belief、candidate、planner 与 naturalizer 拆开，证明模块化框架可以稳定执行，但 10-episode smoke 的结果只具描述性：buyer V2 reward 14.2，resource-first 16.4，resource-second 14.75；seller 17.7，低于 direct 19.6。

### 3.2 V3–V3.3：从通用 safeguard 到 action-space adaptive

| 版本 | 核心改动 | 观察 | 去留 |
|---|---|---|---|
| V3 | posterior mean/q10、response support、deadline regret、dominance 外零概率 | 修复盲目接受与越界报价 | 作为安全骨架 |
| V3.1 | reciprocal commitment；依据近期 own reward 形成 soft floor | 暴露并修复 stale ACCEPT evaluator bug | 机制保留 |
| V3.2 | broad exploit 只用于 Resource/First opening | 减少把 resource heuristic 污染 scalar bargaining | 过渡版 |
| V3.3 | scalar 只保留 response-supported candidates；resource opening 允许一个 broad exploit | 成为第一版 action-space-adaptive performance candidate | 被后续 factorized belief 取代 |

正式 5×20 结果说明 V3.3 没有跨角色统一优势：buyer 13.11、seller 16.95、resource-first 14.55、resource-second 22.47；Opponent Simulation 分别为 12.76、19.13、40.57、11.88。也就是说，V3.3 在 buyer 和 resource-second 有竞争力，但 resource-first 远落后于 Opponent Simulation，seller 也没有提升。

### 3.3 V4–V6：信息价值与 evidence endogeneity

| 版本 | 核心问题 | 实现 | 结论 |
|---|---|---|---|
| V4 | 是否应为未来 episode 主动 probe | `DVOI × sqrt(remaining episodes) × entropy` | 启发式信息增益容易过度 probe |
| V5 | 对手响应不是等强度的 utility evidence | terminal response 0.90、counter 0.55、opponent offer 0.25，并对重复证据降权 | 正确认识 evidence endogeneity，但 reward 未稳定提升 |
| V6 | 信息是否会真正改变决策 | shadow ACCEPT/COUNTER/REJECT；只有可识别、会 action-flip 且 future value 为正才 probe | 离线 21 个 V4 probes 中挡掉 20 个、恢复 20 个 base offers，只保留 1 个 probe；在线 scalar 场景几乎无 probe，证明旧 VOI 多为伪价值 |

### 3.4 V7：factorized preference-policy belief

V7 不再把“拒绝”直接等同于“对方 utility 不可接受”，而是维护 joint latent state `(preference θ, response policy φ)`，其中 φ 包含 rationality、acceptance bias、counter propensity。surprise 可以先触发 policy-only reset，避免一次异常响应抹掉全部 utility belief。

5 runs × 20 episodes 的 confirmatory matrix 给出两个重要结论：

- oracle belief 对 buyer、resource-second 的提升很大：相对 V7 分别 +5.43 与 +7.83，CI 下界为正，证明这些 action spaces 中确实存在可被 planner 利用的 latent preference value；
- learned V7 仍常不如 frozen：buyer frozen 比 V7 +2.10，CI [1.28, 2.92]；resource-first V7 明显优于 wrong belief，但与 frozen/shuffled 接近，说明“belief 可用”与“belief 已学准”是两件事。

### 3.5 V8：reliability-gated belief

V8 把 learned posterior 与安全 anchor 混合：只有直接 response evidence 才提高 policy trust；低证据/OOD 状态回退 anchor。它显著降低 wrong/shuffled belief 造成的灾难性动作，但没有形成全角色 reward 支配。

正式结果中，V8 的 focal reward 为 buyer 7.24、seller 11.12、resource-first 22.27、resource-second 21.68。相对 V7，resource-first 点估计 +3.81，但 buyer -1.32、seller -4.06，resource-second -0.20。V8-oracle 则在 buyer 12.71、seller 17.33、resource-second 25.66，再次说明 planner 有消费好 belief 的能力，瓶颈主要在 learned belief reliability 与环境内数据量。

### 3.6 V10/V11：opponent switch 与 behavioral identifiability

普通 private-preference switch 中，V8 在 resource-second 的 reward DiD 为 -3.15，95% CI [-6.10, -0.20]；frozen 为 -0.82，CI 跨 0。该实验说明“制造 switch”不等于“可识别 switch”：如果 pre/post 行为在公开轨迹中相似，continuous updater 只能产生虚假适应。

V11 改为 away-from-prior、行为上更可分的 switch。resource-second 中 V8 的自身 reward DiD 为 +4.48 [0.07, 8.89]，而 no-cross-episode 为 -2.35；两者差值 +6.83，但 CI [-8.85, 22.51] 仍很宽。正确结论是：在 oracle-verified behavioral identifiability 条件下出现了正向信号，但样本量不足以宣称稳定胜出。

### 3.7 NegotiationArena 留下的可迁移经验

- preference belief 与 response-policy belief 必须分解；
- wrong/shuffled/oracle 与 same-state action flip 是证明 belief 因果作用的必要实验；
- probe 必须通过“可识别 + 会改变决策 + continuation value 为正”的三重门；
- repeated-opponent/change-point 必须先验证公开行为可区分性；
- planner 不应使用 action-space-specific 常数作为论文主要创新。

## 4. Simple Env / AmazonHistoryPrice：完整版本迭代

Simple Env 的迁移目标是把 NegotiationArena 的模块化思路放到自然语言、不同商品类别、不同 reservation/response policy 的单买家单卖家环境中。早期 0–23 场景是 development；24–63 是 validation40；64–127 因类别排序为 electronics，只能称 category-shift test，不能伪装成 IID final。

### 4.1 V1–V4：确认“校准不等于收益”

| 版本 | Belief / Planner | 结果 | 结论 |
|---|---|---|---|
| V1 | structured + bounded semantic；单步 EV | smoke reward 0.129，deal 0.50，重复低价 | 淘汰 |
| V2 | semantic interval 移动 mean；deadline/direct-rejection penalty | held-out pilot seed0 0.527、seed1 0.329 | 性能 reference；跨 seed 不稳定 |
| V3 | semantic shadow，只使用 formal evidence | deterministic learned 0.395 < frozen 0.518 | 淘汰 |
| V4 | monotone behavioral response belief；behavioral one-step EV | Brier 0.116 < frozen 0.148，但 reward 0.230 < 0.380 | 证明局部 calibration 改善不会自动提高长期 reward |

这一步还发现 transcript 曾把 seller 的 `Thought` 和明确 cost 暴露给 buyer。V5 起 belief 只看到公开 `Talk + Action`；旧设置仅保留为 leakage ablation。

### 4.2 V5–V5.6：censoring、lookahead 与 behavioral frontier

| 版本 | 核心改动 | 关键结果 / 失败原因 |
|---|---|---|
| V5 | censored utility + asking policy + verified semantics；two-step lookahead | 把 seller ask 视为战略性上界，不当作真实 cost |
| V5.1 | 修复 verified semantic；belief candidates 强制单调让步 | 修复 semantic evidence 对 candidate 顺序的破坏 |
| V5.2 | terminal dominance + point-mass CDF | 4-scene reward 0.469；first flip 0、sequence flip 1 |
| V5.3 | all-round dominance + uncertain accept option | 成为后续稳定 base planner 骨架 |
| V5.4 | broad prior + strategic censoring；冻结 V5.3 planner | dev24 reward 0.414；比 frozen +0.161，CI [0.023, 0.310]；但 tight deal 1/5，7 个 MI failures 均重复已拒绝末价 |
| V5.5 | behavioral rejection frontier | 重复失败 1→0，但 automotive6 reward 0.546 < V5.4 0.594 |
| V5.6 | deadline-adaptive active probe | reward 0.488 < 0.594；seller 噪声主导，淘汰 |

关键认识是：reject 不是 reservation truth，却是“同一/相似方案在当前 policy 下不会被接受”的直接行为证据。因此 planner 应维护 behavioral rejection frontier，同时仍对 latent utility 保持 censoring。

### 4.3 V5.7–V6.1：可识别性边界

| 版本 | 核心改动 | 结果 | 判断 |
|---|---|---|---|
| V5.7 | latent utility × response-policy mixture | profile300 MAE 0.168→0.114；reward 比 V5.4 +0.0126，CI 跨 0 | belief 保留 |
| V5.8 | exact posterior-integrated planner | reward 0.392 < V5.7 0.404 | 更多成交但 buyer surplus 降，淘汰 |
| V5.7 NL gate | 在自然语言 dev24 验证 | reward 0.507，比 frozen +0.181 且 CI>0；但 MAE 0.242、coverage 0.25，shuffled deal 更高 | calibration gate 失败 |
| V5.9 | utility × policy × strategic-readiness mixture | profile MAE 0.099；NL6 reward 0.691、MAE 0.212、coverage 0.833 | 当前 belief 性能候选，尚非完全校准 |
| V6.0 | reservation × aspiration mixture + aspiration-aware frontier | old profile reward/MAE 0.381/0.120；NL6 0.444/0.217 | response Brier 改善未转化为 utility/reward |
| V6.1 | censored reservation + ask upper bound + aspiration probe | profile reward/MAE/coverage 0.387/0.173/0.88；probe 比 no-probe 更差 | 淘汰 |

这些版本共同证明：短单局中的 counter 序列通常不能唯一分辨“高 reservation”与“低 reservation + 高 aspiration”。因此不能继续靠固定 coupling 或在 6/20 个场景上调常数，必须引入 repeated opponents、change points 与可验证的 response labels。

### 4.4 Iteration 019–020：正式 validation 与 belief calibration

Iteration 019 在 validation40 上运行 6 variants × 40 scenarios × 3 rollouts，共 720 episodes，并按 scenario 聚类 bootstrap。V5.9 learned 相对 frozen 的 reward delta 为 **+0.088**，95% CI **[+0.003, +0.175]**；相对 shuffled 为 **+0.211**，CI **[+0.134, +0.290]**。这是目前 Simple Env 中最直接的 learned belief 正向结果。

Iteration 020 从可验证事件中建立 768 条 calibration data，split 为 480/144/144。校准后 NLL 1.495→0.780，Brier 0.786→0.467，ECE 0.352→0.248；reservation MAE 0.222→0.210，但 CI 跨 0；q10–q90 coverage 0.667→0.854。它证明 response calibration 可训练，但不能单独证明 end-to-end reward 提升。

### 4.5 Iteration 021–022：从 CEM 失败到 Planner V2

Iteration 021 的 CEM 只调八个弱敏感权重，600/600 held-out action sequences 完全不变，reward delta 为 0。该版本被明确拒绝，形成一个重要规范：任何 planner training 都必须先报告 action flip，而不能只报告参数或 loss 变化。

Planner V2 改为 42 个 environment-independent features 的 candidate-Q MLP：

- V2.1 unguarded：reward 0.3535 vs base 0.4267，action flip 100%；过度让步，拒绝；
- V2.2 unguarded：0.4286 vs 0.4267，agreement 0.695 vs 0.575，仍 100% flip，CI 跨 0；
- V2.3 guarded dev：0.4196 vs 0.3709，delta +0.0486 [0.0006, 0.1042]，flip 31.7%；
- V2.3 fresh final profiles：0.3824 vs 0.3664，delta +0.0160 [0, 0.0345]，agreement delta -0.0067，flip 36%。

由于预注册要求 reward CI 下界严格大于 0，V2.3 final 没有晋级。final split 随即冻结，不再用于调 margin。

### 4.6 Iteration 023：跨环境 conservative offline RL / AWR

V6.3 不再学习绝对 Q，而是学习：

`advantage(candidate) = oracle_Q(candidate) − oracle_Q(base_action)`。

三个 bootstrap heads 给出 mean/std；只有 lower confidence bound 超过 action-dependent threshold 时才覆盖 base。OOD、低证据或收益不足时自动 abstain。train/dev/audit 分别为 540/360/360 contexts，并按 utility profile 或 contract template cluster 隔离。

| 域 | audit contexts / clusters | mean delta vs base | cluster bootstrap 95% CI | override | harmful override |
|---|---:|---:|---:|---:|---:|
| Simple static | 60 / 12 | +0.0432 | [+0.0024, +0.0985] | 15.00% | 1.67% |
| Simple repeated | 60 / 12 | +0.0339 | [0, +0.0813] | 16.67% | 3.33% |
| Simple change-point | 60 / 12 | +0.0278 | [0, +0.0742] | 8.33% | 0.00% |
| AgenticPay synthetic multi-issue | 180 / 6 | +0.0796 | [+0.0626, +0.0968] | 59.44% | 11.11% |
| Aggregate | 360 / 18 | **+0.0573** | **[+0.0324, +0.0823]** | 36.39% | 6.39% |

该结果证明 learned residual 非空且在离线 evaluator 中平均安全增益为正；但 AgenticPay 分域只有六个 synthetic templates，且 harmful override 偏高，所以不能将该 checkpoint 宣称为最终跨环境胜出版本。

## 5. AgenticPay：迁移状态、已有结果与数据缺口

### 5.1 已完成的适配

`AgenticPayAdapter` 已把以下内容映射为统一 schema：

- price、continuous terms、discrete terms；
- buyer-side utility、IR 与 contract completeness；
- seller-specific counterparty id，用于多 seller 分离 belief；
- exact seller offer acceptance；
- legal offer candidates、term trade-off candidates 与 quit；
- `<contract>...</contract>` 或 price tag 的 action-locked rendering；
- persistent opponent memory、episode namespace 与 cross-session progress。

AgenticPay V2 variant 已能加载同一个 Iteration 023 AWR checkpoint，说明代码层面实现了跨环境共用 planner schema。dry-run 和 compile 已通过。

### 5.2 历史 single28 真实轨迹

本地 `single28_all_baselines/manual_qwen` 保存了 140 条完成记录、436 个 captured rounds；139/140 agreement，122 个 contract-mode episodes，覆盖 28 tasks 与六类 buyer variants。这些是 Iteration 024 可用的真实 LLM trajectory 来源。

历史 aggregate 按 variant 如下，但**不是严格 paired comparison**：不同 variant 的有效 task 数为 20–27，且 summary 存在 38.6% score/success mismatch、contract IR violation 等旧 evaluator 问题。因此只能用于描述和构造训练数据，不能直接声称 full framework 超越 baseline。

| 历史 variant | n | deal | buyer score | global score | buyer reward | rounds |
|---|---:|---:|---:|---:|---:|---:|
| ASTRA-style prompt | 20 | 1.00 | 35.10 | 24.21 | 10.60 | 3.05 |
| CoT prompt | 20 | 1.00 | 36.55 | 39.13 | 6.28 | 3.25 |
| Direct prompt | 23 | 1.00 | 33.06 | 36.06 | 6.12 | 2.78 |
| Historical full framework | 25 | 1.00 | 24.86 | 23.38 | 11.32 | 2.44 |
| Repo native | 25 | 0.96 | 28.18 | 23.46 | 2.34 | 4.00 |
| Rule offer generator | 27 | 1.00 | 21.09 | 19.01 | 75.62 | 3.15 |

这些指标量纲本身并不一致，例如 rule generator 的 buyer reward 很高但 buyer score 很低，更说明下一轮必须固定 evaluator、任务和随机性做 paired comparison。

另有 26 条 Qwen14B buyer/full-framework vs Qwen14B seller 的 cross-seller records，deal 1.0、buyer score 34.51、global score 37.12、平均 2.12 rounds。它们也可用于行为建模，但没有同 split paired baseline，不能作为最终效果结论。

### 5.3 Multi-agent / all-task 状态

1BMS、MB1S、MBMS runner 和 all-task registry 已实现；历史 multi-product/multi-seller 仅有 2 条有效 smoke（另 1 条 upstream formatting error），其样本量不支持方法结论。其他历史运行多为 API connection/context-length errors。当前必须诚实表述为“runner 已接入，真实 multi-agent online benchmark 尚未完成”。

### 5.4 为什么需要 Iteration 024

Iteration 023 的 AgenticPay labels 由六个程序生成 contract templates 和 oracle response model 构造，缺少：

- 官方 235 个 task 文件中的真实 issue/weight/option 分布；
- seller LLM 对不同合同的真实 accept/counter/quit 倾向；
- logged action 的真实后续 reward anchor；
- scenario-family split，无法证明跨商品/合同模板迁移。

因此继续增加 CEM 轮数或调 deployment margin 都不能解决核心问题；必须更新数据支持，而不是继续微调安全阈值。

## 6. 当前最终 Framework 的精确定义

### 6.1 Environment-independent core

统一数据结构包括：

- `CanonicalState`：角色、回合、deadline、issues、own outside option、公开 observations 与非私密 metadata；
- `OpponentBelief`：reservation interval、policy/readiness mixture、behavior frontier、reliability、old/new regime weights；
- `CandidateAction`：action type、完整 offer、own utility、opponent proxy、adapter acceptance prior、information gain、feasibility margin；
- `ScoredCandidate`：base score、learned residual、uncertainty和 gate 原因。

core 不使用 benchmark 名称 one-hot，也不把 seller private utility、最终 outcome 或 hidden task id 输入 deployed planner。

### 6.2 Belief update

1. adapter 只抽取公开、可验证的 offer/accept/reject/counter/utterance；
2. verified semantic updater 可从自然语言提取承诺和约束，但不能读取 private thought；
3. mixture updater 分离 utility feasibility 与 response policy/readiness；
4. repeated opponent 使用稳定对手 key 保留长期 posterior，同时用 episode namespace 防止事件误去重；
5. change-point 用 old/new regime mixture 和 change probability 表示，不允许直接读取真实 switch time。

### 6.3 Candidate generation 与 planner

adapter 先枚举环境合法候选，保证自己的 IR 和 schema 完整；base planner 提供可解释、保守的 offer/probe/accept/quit 排序。AWR residual 网络看到 base-relative features，在有足够置信度时改变：

- probe 与普通 offer 的选择；
- concession frontier；
- term trade-off；
- accept/continue 边界。

它不允许直接生成自由文本 action，也不会在不确定时全面替换 base policy。

### 6.4 Action lock

selected candidate 是唯一执行意图；LLM naturalizer 只改变措辞。validator 重新解析输出并检查 exact offer、contract bounds、buyer IR 和 selected candidate id；失败时使用 deterministic rendering。这一层把语言 fluency 与战略决策解耦，也是框架跨环境时保持合法性的关键。

### 6.5 Framework 的完整在线决策链

为了避免把 framework 误解成“在 prompt 前面附加一个 belief JSON”，当前实现可以展开为以下六步：

1. **Environment adapter：将不同环境映射到统一状态。** Simple Env 的单商品价格、AgenticPay 的多字段合同，以及 NegotiationArena 的资源/价格 action，首先被转换为 `CanonicalState`。这一层负责读取本方 utility、当前公开 offer、回合与 deadline，并提供环境合法的 action schema；它不允许把 seller private utility 或最终 outcome 泄漏给 buyer。
2. **Verified observation：只记录当时可见、可复核的信息。** 对手的 offer、reject、counter、accept、quit 以及带明确语义的公开 utterance 被标准化为 event。belief updater 只消费这些事件，而不消费模型 private scratchpad。这样 belief 的更新可以逐回合审计。
3. **Persistent factorized belief：维护“偏好”和“行为”两类不确定性。** utility feasibility 描述哪些合同可能满足对手；response policy/readiness 描述即使合同可行，对手现在是否愿意接受、还会让步多少。repeated-opponent memory 跨 episode 保留长期信息，old/new regime mixture 则在出现行为突变时保留旧、新两个 posterior，而不是直接覆盖历史。
4. **Legal candidate generation：先枚举可执行动作，再让 planner 选择。** candidate 包括 probe、不同 concession frontier 上的 offer、接受当前 offer、继续谈判和退出。每个 candidate 都带 own utility、opponent acceptance proxy、information gain、deadline risk 和 feasibility margin。AgenticPay 中 candidate 必须是完整 contract，而不是只优化 price 后再由 LLM 任意补全其他字段。
5. **Belief-usable planner：belief 必须能够改变行为。** planner 不是输出一段分析文本，而是用 posterior 改变 probe、concession frontier、issue trade-off 与 accept/continue 边界。Simple 当前性能分支还叠加了 conservative AWR residual：只有离线数据支持且 bootstrap 下界通过 gate 时，learned residual 才能改写 base planner；否则回退到可解释的 base action。
6. **Action lock 与执行验证：策略选择和语言生成解耦。** planner 选中的 candidate 是唯一决策；renderer/naturalizer 只能将其转成自然语言，不能偷偷更换价格或合同。输出会被重新 parse，并检查 selected candidate id、精确字段、本方 IR 和环境约束；失败则 deterministic rendering。

因此，framework 中真正需要验证的不是“belief JSON 是否看起来合理”，而是以下因果链是否成立：

`公开事件 → belief 改变 → candidate 排序或 accept 边界改变 → 合法 action 改变 → buyer reward 提升`。

calibration、action-flip、oracle/wrong/shuffled belief 都只是这条链的诊断工具；最终 promotion 标准仍然是固定 seller 下 Simple native reward 与 AgenticPay official buyer score。

### 6.6 当前并不是一个完全冻结的跨环境 policy

截至本报告更新时，代码层已经共享 canonical schema、event boundary、belief 表示、planner interface、engine 和 action lock，但部署策略仍有两条尚未完全收敛的分支：

| 分支 | 当前主要实现 | 已验证作用 | 仍存在的问题 |
|---|---|---|---|
| Simple 性能分支 | `StrategicReadinessMixtureBeliefUpdater + BehavioralFrontierPlanner + ConservativeAWRPlanner V6.3` | 在正式 mixed full128 上取得当前最好 reward | AWR 的训练支持主要来自 Simple 与离线构造数据；尚未证明同一 checkpoint 在 AgenticPay 领先 |
| AgenticPay 在线修复分支 | universal V1 → V3 action-consistent → V4 terminal accept guard → V5 verified response frontier（开发中） | 修复 action mismatch、末轮错误 continue 和部分可接受合同拒绝问题 | 最新完成结果仍是 8-task dev；均值仍低于 direct/CoT，V5 尚需完成验证 |

所以准确表述应当是：**我们已经实现了一个 environment-independent 的架构和接口，但尚未得到一个在所有环境都冻结为同一参数、并同时超过强 prompt baseline 的 universal policy。** 当前研究任务正是把两条分支中可泛化的机制合并，而不是把 Simple 的最好数字直接当作 AgenticPay 的结果。

### 6.7 Simple Env：`0.513` 到底代表什么

最新 Simple 正式比较使用同一组 `128 scenarios × 3 rollouts × 5 variants = 1,920 episodes`。128 个场景是 mixed suite，其中 64–127 属于 category-shift，因此它不是只在 20 个容易场景上的小样本结果，也不是纯 IID test。

| Variant | Mean Simple native reward | 相对 CoT |
|---|---:|---:|
| Direct prompt | 0.0961 | -0.3104 |
| CoT prompt | 0.4065 | 0 |
| Historical full framework | 0.4464 | +0.0399 |
| V5.9 | 0.4981 | +0.0917 |
| **V6.3** | **0.5130** | **+0.1066** |

paired cluster bootstrap 的结果为：V6.3 相对 CoT `+0.1066 [0.0581, 0.1560]`，相对 historical full framework `+0.0666 [0.0203, 0.1118]`；两者区间均高于 0。V6.3 相对 V5.9 的点估计为 `+0.0149`，但置信区间跨 0，因此只能说 V6.3 是当前最高均值，不能声称它已经统计显著优于 V5.9。

这里的 `0.5130` 是 Simple Env 自己的 normalized/native buyer reward，不是 belief accuracy，也不是成交率。它提供了目前最强的 end-to-end 证据：在固定 seller、相同 scenarios 与 seeds 的 paired setting 下，belief-aware candidate/planner 分支最终确实提高了 buyer 的任务收益。

### 6.8 AgenticPay：为什么仍赶不上 direct prompt

AgenticPay 的正式 full single28 比较为 `28 tasks × 3 seeds × 5 variants = 420 episodes`：

| Variant | Official buyer score |
|---|---:|
| **Direct prompt** | **32.9531** |
| CoT prompt | 31.7199 |
| Historical full framework | 21.3108 |
| Universal V1 | 15.7307 |
| Universal V2 | 18.1775 |

因此，对“目前 AgenticPay 是否已经超过 direct prompt”的回答是明确的：**没有。** 原始 universal variants 不仅没有超过 direct/CoT，甚至低于 historical full framework。这也是后续停止只优化 belief calibration、转向直接检查 trajectory 中 reward failure 的原因。

之后在固定的 `8 tasks × 3 seeds = 24 episodes` 开发集上进行了在线修复：

| Variant | Buyer score | Deal rate | 关键变化 |
|---|---:|---:|---|
| Direct prompt | **50.0790** | — | 强 prompt baseline |
| CoT prompt | 45.9631 | — | reasoning prompt baseline |
| Historical full framework | 34.6891 | — | 旧框架 |
| V3 action-consistent | 33.1317 | 62.5% | 锁定 planner action，消除文本执行偏移 |
| **V4 terminal accept guard** | **41.5588** | **87.5%** | 在末轮接受合法且优于 outside option 的已提出合同 |

V4 相对 V3 提升 `+8.4271`，相对 historical full framework 提升 `+6.8697`；但样本仅有 24 个 episode，对应置信区间仍跨 0。V4 相对 direct prompt 为 `-8.5202`，相对 CoT 为 `-4.4043`，所以它是“明显修复失败模式并回升的开发版本”，不是“已经超过 baseline 的最终结果”。此外，V4 已把 action mismatch、late deferral 和 buyer IR violation 分别降为 `0/280`、`0` 和 `0`，说明执行正确性已经改善，但执行正确不自动等价于策略最优。

trajectory 显示 AgenticPay 的主要瓶颈已不只是 belief calibration：

- 某些任务中 framework 能生成大量安全候选，却反复发送已被拒绝的同一合同，缺乏利用真实 response frontier 的有效 concession；例如 Task 6 曾生成 145/156/168 个安全候选，但仍重复 133.82 并最终超时。
- “达成 agreement”也不一定产生高 buyer score。Task 3 中 seller 曾声称 floor 为 175，之后却接受 143；环境记录成交，但 official scorer 给出 `-0.4455` 并标记 `score_success_mismatch`。这表明 planner 必须优化官方 buyer utility/contract validity，而不能把 agreement rate 当作最终目标。
- AgenticPay 是 multi-issue contract。过度保守地同时给 price、delivery、warranty 等多个字段让步，会让 buyer score 快速下降；只校准 reservation price 或提高成交率不足以解决问题。

基于这些证据，当前 AgenticPay V5 的方向是 **verified response frontier**：只从公开的 accept/reject/counter 中建立“什么合同坐标已经被拒绝、什么方向尚值得 probe”的 frontier，抑制重复无效 offer，并在 deadline 前选择经过验证的最小必要让步。V5 仍处于验证阶段，不能提前写成正结果。

### 6.9 两个环境的数字为什么不能直接横向比较

Simple 的 `0.5130` 和 AgenticPay 的 `41.5588` 不在同一量纲：前者是 Simple normalized/native reward，后者是 AgenticPay official buyer score；任务数量、utility 函数、outside option、合同维数和失败惩罚也不同。正确比较方式是在各自环境内做 paired comparison：同一 seller、同一 task/scenario、同一 seed 下比较 framework 与 direct、CoT 和 historical full framework，然后报告场景聚类 bootstrap CI。

因此当前总体结论是：

- **Simple Env：已经有可信的正向 end-to-end reward 结果，最好均值约 0.513。**
- **AgenticPay：尚未追平 direct prompt；V4 在开发集上由 33.13 回升到 41.56，但 direct 为 50.08。**
- **Universal architecture 已成立，universal performance 尚未成立。** 下一阶段的关键不是继续堆叠 belief 字段，而是让公开 response evidence 稳定改变 multi-issue candidate frontier，并在完整 single28 上复验，最后再冻结同一 policy 到两个环境做全集测试。

## 7. 目前能与不能声称的结果

### 7.1 可以声称

- NegotiationArena 中 oracle、wrong/shuffled 与 action-flip 实验证明多个 action space 存在 decision-relevant belief，但 learned continuous belief 并未全面优于 frozen；
- Simple validation40 上 V5.9 learned 相对 frozen reward +0.088，cluster CI 下界 +0.003；相对 shuffled +0.211，CI 下界 +0.134；
- verified event calibration 明显改善 NLL/Brier/ECE；
- Planner V2 能改变决策边界，但 unguarded 会过度让步；
- conservative AWR 在全新离线跨环境 audit 中取得 +0.0573 [0.0324, 0.0823] 的 base-relative oracle advantage，且 first action 不被随意改写；
- universal adapter、per-opponent memory、change-point features、multi-issue candidate 和 action lock 已在代码层完成。
- Simple mixed full128 正式 paired test 中，V6.3 reward 为 0.5130，相对 CoT 提升 +0.1066 [0.0581, 0.1560]，相对 historical full framework 提升 +0.0666 [0.0203, 0.1118]；
- AgenticPay 8-task dev 中，terminal accept guard 将 buyer score 从 V3 的 33.1317 提升至 41.5588，并把 deal rate 从 62.5% 提升到 87.5%；这是开发集上的 failure-mode 修复证据，而非最终 baseline superiority。

### 7.2 不能声称

- 不能把 Iteration 023 synthetic audit 写成 AgenticPay online improvement；
- 不能把历史 AgenticPay 不等样本数 summary 当 paired baseline comparison；
- 不能声称 continuous belief 在所有角色/环境都优于 frozen；
- 不能把 calibration improvement 等同于 reward improvement；
- 不能把一个不可识别 switch 上的波动解释为成功 adaptation；
- 不能在看过 Iteration 023 audit 后继续调其 margin，再报告同一 audit。
- 不能把 Simple V6.3 与 AgenticPay V4/V5 描述成同一个已经冻结的 policy checkpoint；当前共享架构，但策略分支仍在合并；
- 不能把 AgenticPay 8-task dev 当作 full single28 结果，也不能声称 V4 已超过 direct/CoT；
- 不能用 agreement rate 代替 AgenticPay official buyer score，尤其在 score-success mismatch 或低效合同成交时。

## 8. 代码地图与推荐阅读顺序

### 8.1 Core

| 文件 | 作用 |
|---|---|
| `/work5/qixint/negotiation/framework/schemas.py` | canonical state、offer、belief、candidate、trace schema |
| `/work5/qixint/negotiation/framework/events.py` | 可验证公开事件类型和事件抽取边界 |
| `/work5/qixint/negotiation/framework/belief.py` | structured/censored/mixture/readiness/aspiration belief updaters |
| `/work5/qixint/negotiation/framework/planner.py` | base planners；当前 base 为 `BehavioralFrontierPlanner` |
| `/work5/qixint/negotiation/framework/trainable_planner.py` | Planner V2 candidate-Q 与通用 feature encoder |
| `/work5/qixint/negotiation/framework/conservative_awr_planner.py` | 当前 V6.3 residual AWR、bootstrap LCB 与 fallback |
| `/work5/qixint/negotiation/framework/engine.py` | belief update → candidates → rank → render → validate 的统一 orchestration |

### 8.2 Simple Env

| 文件 | 作用 |
|---|---|
| `/work5/qixint/Simple_Env/buyer/universal_framework.py` | `SimpleEnvAdapter` 和 V1–V6.3 variants；V6.3 是当前 AWR 接口 |
| `/work5/qixint/Simple_Env/tools/train_verified_belief_calibrator.py` | verified-event belief calibration |
| `/work5/qixint/Simple_Env/tools/train_decision_boundary_planner_v2.py` | Planner V2 finetune |
| `/work5/qixint/Simple_Env/tools/train_cross_environment_awr_planner.py` | 跨环境 synthetic data、AWR training、dev/audit gate；已支持额外真实 contexts |
| `/work5/qixint/Simple_Env/scripts/analyze_universal_framework_runs.py` | reward、deal、belief、paired/action-flip 分析 |
| `/work5/qixint/Simple_Env/research_iterations/UNIVERSAL_FRAMEWORK_ITERATION_LEDGER_CN.md` | append-only 版本总账 |

### 8.3 AgenticPay

| 文件 | 作用 |
|---|---|
| `/work5/qixint/AgenticPay_Env/buyer/universal_framework.py` | AgenticPay adapter、contract candidate、per-seller belief、AWR runtime |
| `/work5/qixint/AgenticPay_Env/environment/single_product.py` | single buyer/product/seller runner |
| `/work5/qixint/AgenticPay_Env/environment/multi_agent.py` | 1BMS、MB1S、MBMS runner |
| `/work5/qixint/AgenticPay_Env/environment/all_tasks.py` | 官方 task registry 与 all-task execution |
| `/work5/qixint/experiments/run_agenticpay_single28_framework.py` | 140 条历史 trajectory 的原始 runner/capture |
| `/work5/qixint/agenticpay_env_runs/single28_all_baselines/manual_qwen/task_results.jsonl` | Iteration 024 的真实 trajectory source |
| `/work5/qixint/benchmarks/AgenticPay/agenticpay/examples/` | 235 个官方 task/contract source files |

推荐阅读顺序是：`schemas.py` → `engine.py` → `belief.py` → `planner.py` → `conservative_awr_planner.py` → 两个 adapter → Iteration 023/024 报告。各 iteration 下的 `code_snapshots` 保留了当时版本，失败版本没有被覆盖。

## 9. Iteration 024：新的 train/dev iteration

预注册文件：`/work5/qixint/Simple_Env/research_iterations/iteration_024_agenticpay_trajectory_awr/PREREGISTERED_PLAN_CN.md`。

### 9.1 数据源

1. `agenticpay_real_trajectory`：140 条历史 completed records、436 rounds；每个 buyer turn 用此前公开 history 重建 belief/candidates；logged action 用真实 seller response 和 terminal buyer utility形成 Monte-Carlo anchor。
2. `agenticpay_official_template`：从官方 Python task 源码安全抽取 contract configs，包括 continuous bounds、discrete options、buyer/seller preference weights；状态和 response 仍是离线生成，必须保持该标签。
3. `synthetic_regularizer`：Iteration 023 的 Simple static/repeated/change-point 与 synthetic multi-issue cases，只以 0.5 权重保留跨环境 regularization。

训练权重固定为 real trajectory 2.0、official template 1.0、synthetic regularizer 0.5。

### 9.2 防泄漏 split

split unit 是 scenario group，而不是随机 turn：dev 固定为 `s4, s9, s14, s19, s24, core_task3`，其余为 train。同一 scenario 在任何 task family、baseline variant 和 trajectory turn 中不能跨 split。Iteration 024 不创建或查看新 final/audit。

### 9.3 Label 设计

- logged candidate：真实 accept/counter/quit 和后续 terminal buyer utility 提供 anchor；
- unexecuted candidates：用全体 train trajectory 拟合的 `P(response | seller utility, progress, contract features)` 产生明确标记的 model-based counterfactual；
- official private utility 只在 evaluator/label side 使用；deployed features 不含 seller truth；
- 每个 context 保存 `data_source`、`label_source`、`training_weight`、scenario group、trajectory id、turn id 与 counterfactual provenance。

### 9.4 固定 promotion gate

- aggregate mean oracle delta > 0；
- real-trajectory 与 official-template 两个域均不为负；
- aggregate harmful override ≤10%，real trajectory ≤12%；
- override rate 2%–60%；
- 新 checkpoint 在同一个 Iteration 024 dev 上不低于冻结的 Iteration 023 checkpoint；
- private-truth feature audit 与 scenario overlap audit 均通过。

## 10. Iteration 024 之后的在线实验计划

### Phase A：数据与离线 gate

1. 安全 AST 解析官方 contract templates，不 import/execute task source；
2. 构造并审计 trajectory-derived train/dev contexts；
3. 输出 response-model calibration（NLL/Brier/ECE）和 logged-action value consistency；
4. 用固定 deployment thresholds 训练 bootstrap AWR；
5. 同 dev 比较 Iteration 023 与 024，不通过 gate 则保留但不升级。

### Phase B：Simple Env paired online

- variants：V5.9 learned、V5.9 frozen、V5.9 shuffled、V6.3 Iter023 AWR、Iteration024 AWR；
- validation40 × 至少 3 independent rollouts；
- scenario-cluster bootstrap reward CI；
- 报告 agreement、buyer reward、joint reward、action sequence flip、first-action flip、override、belief MAE/coverage、response Brier/ECE；
- repeated-opponent 和 identifiable/non-identifiable change-point 分开报告 false alarm、detection delay、post-switch recovery。

### Phase C：AgenticPay paired online

先固定 single28 的相同任务、buyer/seller model snapshot、temperature、max rounds 和 seed，对比：repo native、direct prompt、historical full framework、V1 universal base、Iteration023 AWR、Iteration024 AWR。随后才扩展官方 multi-issue subset与 1BMS/MB1S/MBMS。

AgenticPay 必报：deal、buyer/seller/global score、buyer-side contract utility、IR violation、contract feasibility、rounds、parse/action-lock error、per-seller selection、override/harmful proxy。不能只报 deal rate。

### Phase D：论文主张的判定

只有在相同 checkpoint、相同 feature schema 下，满足以下条件才可形成 stronger novelty story：

1. Simple 与 AgenticPay 的 reward CI 均非负，至少一个严格为正；
2. wrong/shuffled 明显恶化或产生可解释 action flips，oracle 给出可达上界；
3. continuous belief 在 identifiable repeated/switch setting 优于 no-cross/frozen，在 non-identifiable setting 不过度更新；
4. AWR 的增益来自 probe/frontier/accept boundary 的可验证改变，而不是 parser 修复或更多 LLM calls；
5. 同一 framework 只替换 adapter，不为每个 benchmark 单独重写 planner。

## 11. 最终研究故事与 novelty 边界

建议论文故事不是“首次让 LLM 建模对手”，也不是“在某个谈判环境中设计更强 prompt”，而是：

> A negotiation agent should update only behaviorally identifiable opponent beliefs, translate calibrated uncertainty into candidate-level consequences, and learn only conservative decision-boundary improvements over a safe belief-usable planner. The same core policy operates over environment-provided feasible candidates from scalar bargaining to multi-issue contracts.

可形成的 novelty 组合为：

1. **Factorized, evidence-aware continuous belief**：分离 latent utility 与 response policy/readiness，并显式处理 censored/endogenous evidence、repeated opponents 和 regime mixture；
2. **Belief-usability as a causal property**：用 frozen/wrong/shuffled/oracle、same-state action flip、decision regret，而不是只用 belief text 或 reward，验证 belief 是否真正改变了正确决策；
3. **Conservative cross-environment residual planning**：围绕安全 base action 学 advantage，bootstrap LCB 控制 override，OOD/低证据自动 abstain；
4. **Environment-independent decision core**：adapter 提供合法 candidate 与 utility semantics，core 不读取 benchmark identity，action lock 保证语言输出不破坏战略选择；
5. **Identifiability-aware evaluation**：将 identifiable 与 non-identifiable opponent switch 分开，避免把无法从公开行为推断的 private change 当作 belief failure。

当前最关键的未完成证据是：Iteration 024 的真实 AgenticPay labels 是否能降低 multi-issue harmful overrides，并把离线 advantage 转化为严格 paired online reward。下一阶段所有开发都应围绕这条证据链，而不是继续在旧 validation/final split 调常数。

## 12. 2026-08-17 执行更新：Iteration 024 实际结果

Iteration 024 已按本报告计划完成数据构建和训练。安全 AST 从 247 个官方 files 中抽取 395 个去重 contracts；真实 140 records / 436 rounds 构造 418 contexts；extra train/dev 为 953/255，scenario overlap 为 0。

新 checkpoint 在 975 个 combined dev contexts 上为 +0.03758，cluster CI [0.02318,0.05262]，总体 harmful override 4.82%。官方-template 域从旧 checkpoint 的 -0.02808 / harmful 55.70% 改善到 +0.02420 / 1.27%。但是 real-trajectory 域为 -0.00277、harmful 15.46%，且 aggregate delta 低于旧 checkpoint 的 +0.04509，因此预注册 promotion gate 失败。

失败归因为 terminal outcome 对所有早期 logged offers 的 credit smear、过强 logged behavior prior，以及未按 trajectory 归一权重。完整结果与下一轮 per-turn doubly-robust label 计划见：`/work5/qixint/Simple_Env/research_iterations/iteration_024_agenticpay_trajectory_awr/RESULTS_AND_NEXT_STEPS_CN.md`。Iteration 024 checkpoint 作为失败研究产物保留，不替换 Iteration 023。
