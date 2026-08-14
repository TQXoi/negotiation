# NegotiationArena × Belief–Planner–Generator 工作计划

## 1. 基准、论文与代码版本

- 原论文：*How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis*，ICML 2024。
- 论文主页：https://proceedings.mlr.press/v235/bianchi24a.html
- 原始代码：https://github.com/vinid/NegotiationArena
- 本地主分支固定版本：`c447fafd439a20b84cdedeb2f8a85c4fad764745`。
- 原论文实验分支：`origin/paper_experiment_code`，固定版本 `d35a7a3aa0d94c2d49f1d6ac13c5f931851abf12`。

注意：当前 main 是重构后的平台，README 明确指出论文实验代码位于 `paper_experiment_code`。因此必须分成两条轨道：用论文分支做结果复现；用 main 做长期统一 adapter。不要把 main 当前能运行的示例直接称作论文复现。

## 2. 原论文任务与 method

NegotiationArena 是 benchmark/platform，而不是训练算法。其原方法是让不同 LLM 读取角色、私有资源/目标和社会行为 prompt，在交替回合协议中直接生成消息及结构化交易；环境 parser 抽取 offer/accept 并计算最终 payoff。

论文涵盖三类任务：

1. Multi-turn ultimatum/shared-resource：双方协商如何分配资源；
2. Trading/resource exchange：双方具有不同资源、估值或目标，通过交换达到目标；
3. Buy/Sell bargaining：买卖双方围绕价格协商，私有成本/支付意愿形成 reservation value。

实验比较 GPT-4、GPT-3.5、Claude 2/2.1 等模型，并通过 normal、cunning、desperate/desolate、self-interested 等行为提示研究社会策略。原 agent 基本依赖当前 prompt/history，没有显式持久 belief、候选规划器、belief-conditioned action score 或策略校准。

## 3. 原论文设置与主要结果

论文主要报告 agreement/win、双方和总体 payoff，以及不同模型、角色和 behavior prompt 的差异，并分析 anchoring、首轮报价和典型对话行为。

主要结论是：

- GPT-4 总体谈判能力最强，但更强模型有时会迁就较弱对手、提出对自身更差的协议，说明语言能力不等于收益最大化；
- social behavior prompt 会系统性改变收益，扮演 desperate/desolate 等策略相对标准 GPT-4 最多可带来约 20% 的收益提升；
- 模型表现出类似人类的 anchoring，首轮报价会影响最终结果；
- 结果具有明显的模型配对、任务类型和角色依赖，不能只汇总一个平均 win rate。

论文图表和归档 notebook 是精确数值的唯一复现基准。第一阶段应从 `paper_experiment_code` 重新生成表/图，而不是把新版 runner 的输出与论文数字混合。模型 API 已更新时，要把“代码复现”和“历史模型快照不可复现”分别记录。

## 4. 我们的方法如何跨环境实践

### 4.1 防止 env-specific 的原则

建立统一 `NegotiationDomainAdapter`，只暴露：参与者、合法动作、私有 utility、公开 observations、终止条件和 outcome。核心模块只认识统一对象：

- `BeliefState(opponent_type, latent_utility, reservation, acceptance_model, confidence, evidence)`
- `CandidateAction(offer, message_intent, expected_utility, agreement_prob, information_gain, risk)`
- `PlannerDecision(candidate_id, rationale_features)`
- `GeneratedAction(message, structured_offer)`

各任务只提供转换器和 utility evaluator，不为某个任务另写一套 planner。

### 4.2 各任务的 latent belief

| 任务 | 需要推断的 latent variables | 可用证据 |
|---|---|---|
| Buy/Sell | reservation price、成本/WTP、让步速度、风险偏好 | 报价、拒绝、counteroffer、沉默/退出 |
| Trading | 资源边际价值、目标 bundle、互补性、接受阈值 | 提议组合、保留资源、部分让步 |
| Ultimatum | 公平偏好、最低接受份额、拒绝风险、persona | 分配提案、接受/拒绝、公平话语 |

统一 belief 可以使用不同 domain likelihood，但输出相同的 posterior/confidence 接口。先用可解释的区间/粒子后验，再比较 LLM summarizer；禁止只保存自由文本“对方大概想要什么”。

### 4.3 Planner 与 Generator

Planner 生成 self-max、high-agreement、Pareto/Nash、fairness、probe 和 robust 等候选族，按

`E[self payoff] + agreement probability + information gain + future value - breakdown/manipulation risk`

排序。Generator 只负责把锁定 offer 转为自然语言；parser/validator 检查金额、资源守恒、合法性和 offer-message 一致性。该结构能直接回答核心问题：belief 改变是否造成 proposal、接受或退出决策改变。

## 5. 各环节改进

| 环节 | 原平台局限 | 我们的改进 | 评测 |
|---|---|---|---|
| Observation | history 文本、parser 较脆弱 | 统一事件流、schema 校验、repair | parse/invalid rate |
| Belief | 无持久可校准 state | posterior + confidence + evidence ledger | MAE、Brier/ECE、rank accuracy |
| Candidate | LLM 直接报一个价/交易 | 约束枚举、Pareto pruning、多样候选 | oracle recall、coverage |
| Planner | 无显式 belief-conditioned objective | receding horizon + acceptance/value model | regret、action sensitivity |
| Exploration | 让步与探测混合 | 显式 expected information gain | entropy reduction、future payoff |
| Generator | 语言可能与交易不一致 | action locking + validator | action fidelity |
| Evaluation | win/payoff 不足以解释机制 | 加入效率、校准、反事实、角色公平 | Pareto/Nash、role gap |

## 6. 测试设置

### 6.1 两条实验轨道

**Track A：原论文复现。** 从固定的 `paper_experiment_code` 创建单独 worktree，使用其 notebook、场景、模型列表、行为 prompt 和指标生成原表图。不要切换或污染 main 工作树。

**Track B：统一框架。** 在 main 上实现 adapter。先跑当前最稳定的 Buy/Sell，再接 Trading 和 Ultimatum；每接一个任务先做 adapter 等价测试，再启用 belief/planner。

### 6.2 场景与切分

- Buy/Sell：覆盖不同 bargaining zone、无 ZOPA、不同 surplus，买卖角色互换；
- Trading：覆盖 aligned/conflicting valuation、资源稀缺和互补 bundle，双方位置互换；
- Ultimatum：改变总资源、最低接受阈值、公平偏好及 proposer/responder 位置；
- Development 只开放一种任务或一部分参数网格；其他任务和新参数组合作为 frozen OOD test。

最有说服力的通用性实验是：只在 Buy/Sell 调整框架，冻结全部超参数，zero-shot 测 Trading 和 Ultimatum；随后再报告少量 task-specific calibration 的上界。

后续 ICLR 2026 的 AMPO 工作已使用 NegotiationArena 做 OOD 测试，可借鉴更严格的规模：每个任务 200 个 scenario、角色互换、每个位置 4 次重复，即每任务 1600 局；指标为 self profit、total profit、winning rate。该设置是后续论文协议，不是 ICML 2024 原论文设置，应单列为增强评测。

### 6.3 公平控制

- 使用 paired scenario/seed：同一私有价值、首发方、对手、temperature 和 token budget 比较方法；
- 主实验只替换 focal agent，对手固定为原 prompt baseline；另做双方均使用框架的群体实验；
- 每个场景角色互换，分开报告 proposer/responder、buyer/seller；
- 对 API 模型记录完整 model snapshot/date；确定性 planner 与随机 generator 的方差分别报告；
- 先 smoke test，再至少 20 次/条件；主结果使用 50+ paired samples 或上述 200-scenario 协议，并给 bootstrap 95% CI。

### 6.4 Baseline、消融与负对照

1. 原论文 direct/behavior prompts；
2. history-only direct LLM；
3. planner + uniform belief；
4. belief + myopic policy；
5. full Belief–Planner–Generator；
6. oracle opponent utility/reservation；
7. frozen、shuffled、wrong belief；
8. 去掉 information gain、future value、action lock。

除 agreement、win、self/total payoff 外，增加 surplus capture、Pareto efficiency、Nash welfare、轮数、breakdown、invalid action、belief calibration、candidate oracle recall、planner regret、belief-action sensitivity、role asymmetry。

## 7. 实施里程碑

1. **M0 论文归档复现**：独立 worktree 跑 `paper_experiment_code`，登记可复现图表、失效 API 和历史结果。
2. **M1 Buy/Sell adapter**：原 agent 经 adapter 后保持 outcome 等价；补合法性单测。
3. **M2 通用 Belief**：实现区间/粒子更新，在 synthetic/private ground truth 上做校准。
4. **M3 Planner–Generator**：候选、显式打分、action lock、反事实测试。
5. **M4 跨任务**：接入 Trading/Ultimatum，冻结核心模块做 zero-shot OOD。
6. **M5 规模化评测**：paired roles、完整消融、置信区间及成本/延迟报告。

验收要求：不仅 full method 的 payoff 更高，还要展示 belief 误差下降、planner regret 下降；oracle belief 构成合理上界；shuffled/wrong belief 造成显著退化；从 Buy/Sell 到另外两类任务无需改核心 state/action/objective 定义。
