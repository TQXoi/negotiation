# NegotiationArena 与 ANAC/ANL 2025：Belief Framework 基准审计与实验计划 V2

> 日期：2026-08-04
> 目标：判断两个 challenge 是否存在与 LLM-Deliberation 类似的“无需真实交互也能取得高分”问题，并设计能够分别验证 continuous belief model 与 belief-usable planner 的实验。
> 代码位置：`external_negotiation_envs/NegotiationArena/`、`external_negotiation_envs/ANL2025/`

## 1. 执行结论

结论不是“换掉 LLM-Deliberation 后选一个新环境”，而是让两个环境承担不同的证据职责：

| 基准 | 真实私有信息 | 对手是否真实决定接受 | 是否必须利用历史 | 最适合验证 | 优先级 |
|---|---:|---:|---:|---|---:|
| NegotiationArena Buy/Sell、Trading | 是 | 是 | 中等；原始固定实例存在捷径 | 自然语言中的 belief 抽取、策略表达、生成器 | P1 |
| NegotiationArena Ultimatum | 较弱 | 是 | 弱到中等 | 公平阈值/风险类型 belief；不宜作为主结果 | P2 |
| ANL 2025 multi-deal | 是，且随机化 | 是 | 强，尤其是跨 thread 规划 | continuous belief 与 belief-usable planner 的核心因果验证 | **P0** |
| SCML 2025 | 是，动态市场 | 是 | 强 | 长期扩展：并发、库存、履约与对手模型 | P3 |
| LLM-Deliberation 原 cooperative game | 角色语义泄漏严重 | 最终投票受实现影响 | 已证明可能很弱 | legacy sanity check | P3 |

ANL 2025 是当前最适合证明核心算法的环境：对手效用确实私有，对手自主接受/拒绝，中心 agent 的早期协议会不可逆地影响最终组合效用。NegotiationArena 更适合证明同一个 belief/planner 接口能迁移到自然语言交互，并检查 generator 是否准确、稳健地实现 planner 决策。

但是，任何一个环境上的高 utility 都不能单独证明 belief 有效。必须加入 `no-history/time-only`、`frozen`、`shuffled`、`continuous` 与 `oracle` 对照，并同时报告 belief calibration、planner action sensitivity 和最终收益。

---

## 2. 审计标准：什么算“与 LLM-Deliberation 类似的问题”

LLM-Deliberation 暴露的不是一个普通性能问题，而是构念效度问题：模型可能从 stakeholder 名称和常识推断偏好；P1 只需提出一个多数人会接受的 deal；即使没有对话历史，single-agent/open-loop 方法也可能接近完整 multi-agent 方法。

本报告据此检查六种捷径：

1. **身份/语义泄漏**：名称、角色或自然语言场景是否直接暴露偏好；
2. **固定实例记忆**：价格、资源和博弈模板是否长期固定，可被训练语料或手工策略记住；
3. **伪交互**：是否由环境按阈值自动通过，而不是对手基于自己的状态行动；
4. **open-loop 可解**：只看初始 prompt、回合数和自身 utility 是否已足够；
5. **评分遮蔽**：忽略 tie/no-deal，或只看一个角色，是否制造虚假优势；
6. **强 planner 伪装成 belief**：收益提升是否来自自身效用搜索、时间让步或已知分布，而非在线更新对手模型。

对应的最小证据链应为：

```text
真实隐藏状态
  -> 历史使 posterior 更准确（calibration / ranking）
  -> posterior 改变候选排序或接受决策（action sensitivity）
  -> 正确 posterior 优于 shuffled / frozen posterior（causal utility gain）
  -> oracle belief 构成合理上界（headroom）
  -> 在未见 opponent / utility family 上仍成立（OOD generalization）
```

---

## 3. NegotiationArena

### 3.1 原始任务与 setting

NegotiationArena 发表于 ICML 2024，提供多轮、双边、交替行动的自然语言 negotiation framework。agent 输出包含 private reasoning 与公开 message/offer，环境只把公开部分传给对手。原论文比较 GPT-4、GPT-3.5、Claude-2 和 Claude-2.1；每个场景、每个**有序模型对**运行 60 局，并交换先后手。模型温度为 0.7，单次最多生成 400 tokens。[ICML 论文与 PDF](https://proceedings.mlr.press/v235/bianchi24a.html)、[官方代码](https://github.com/vinid/NegotiationArena)

原论文的准确复现应使用仓库的 `paper_experiment_code` branch；本地主分支 README 明确把当前可工作任务集中在 Buy/Sell 与 Simple，Trading/Ultimatum 代码仍在但不能默认等同论文版本。

#### A. Resource Exchange / Trading

- 两方持有不同数量的 X、Y 资源；自己的资源和目标私有；prompt 明确表示不知道对手资源；
- 通过公开消息与 bundle offer 交换资源；论文实验通常为 8 rounds；
- 适合推断：对手稀缺资源、边际价值、可接受 bundle、披露可信度。

#### B. Multi-turn Ultimatum

- 一方初始持有全部金额，双方反复提出分配并可接受/拒绝，论文设置为 8 rounds；
- 总金额和角色结构是公开的，真正隐藏的主要是公平观、风险偏好和接受阈值；
- 适合研究 persona/fairness belief，但标准博弈模板本身提供很强先验。

#### C. Seller–Buyer

- seller 知道私有生产成本，buyer 知道私有 willingness-to-pay；双方通过价格 offer 协商，论文设置为 10 rounds；
- role 暴露偏好的单调方向，但不暴露 reservation value；对手自主接受/反提议；
- 是 NegotiationArena 中最清晰的 continuous belief 测试：估计 reservation、concession curve 和成交概率。

### 3.2 原始 baseline 与结果

原论文的 baseline 本质上是不同闭源 LLM 的 direct structured prompting，而不是显式 opponent modeling 算法：

- Trading 存在显著后手优势。Claude-2.1 先手、GPT-4 后手时 GPT-4 胜率 76%；次序反转时 Claude-2.1 后手胜率 72%。Claude-2.1 作为 P2 的平均 payoff 2.45，高于 GPT-4 的 1.38。
- Ultimatum 中除 GPT-3.5 外 P1 几乎总是获胜；Claude-2.1 作为 P1 对所有对手平均 payoff 均大于 60。Claude 初始 offer 平均比 GPT 低约 10，因此保留了更多自身收益。
- Buy/Sell 的 cost=40、WTP=60 设置中，多数最终价格低于中点 50；GPT-4 作为 buyer 的跨 seller 平均成交价约 41，是最强 buyer。
- social prompt 试验固定 P1 为默认 GPT-4，P2 为 default/cunning/desperate GPT-4，每对每任务 80 局。cunning/desperate 普遍提高 win/payoff；Ultimatum 中 cunning P2 的 win rate 达 82%，但平均 payoff 仍约 49，因为激进策略增加了 no-deal。
- 论文还发现 anchoring、numerosity bias、规则/标签错误和弱模型“babysitting”效应。附录承认 prompt 为减少 Claude 格式错误做过调整，因此存在 model-specific prompt bias。

这些数字应作为历史参照，不应期待 Qwen-30B 在不同服务、prompt 和代码 branch 下逐点相等。

### 3.3 后续可复现 baseline

目前最重要的已接收后续方法是 ICLR 2026 Poster **Adaptive Social Learning via Mode Policy Optimization for Language Agents**。ASL 设计四级 reasoning mode，先行为克隆，再用 AMPO 同时优化 mode-level 与 sample-level advantage，并加入长度奖励。训练发生在 SOTOPIA-π，NegotiationArena 是 zero-shot OOD 测试；代码公开。[OpenReview](https://openreview.net/forum?id=GG7YQnsdhp)、[代码](https://github.com/MozerWang/AMPO)

其 protocol 对每个 Buy/Sell 与 Ultimatum 任务使用 200 scenarios、交换角色、每 scenario 执行 4 次，即每任务 1,600 次评估；对手为 GPT-4o。论文报告：

| Backbone / 方法 | Buy/Sell self / total / win | Ultimatum self / total / win |
|---|---:|---:|
| Qwen2.5-7B vanilla | 11.90 / 13.87 / 38.1% | 14.23 / 32.68 / 8.1% |
| Qwen2.5-7B GRPO | 16.75 / 23.67 / 52.6% | 34.65 / 76.88 / 25.1% |
| Qwen2.5-7B AMPO | **17.94 / 23.96 / 54.6%** | **35.71 / 79.23 / 28.4%** |
| Llama-3.1-8B vanilla | 3.12 / 3.82 / 9.4% | 14.43 / 22.41 / 15.9% |
| Llama-3.1-8B GRPO | 14.99 / 23.58 / 48.7% | 31.64 / 69.68 / 18.4% |
| Llama-3.1-8B AMPO | **15.57 / 24.00 / 50.2%** | **34.91 / 76.42 / 20.5%** |

AMPO 是“自适应 reasoning depth”的强 baseline，但没有显式、可校准的 opponent posterior，也没有证明 planner 对 belief 的因果使用。因此它与我们的贡献互补：可将 AMPO/GRPO 发表数据作为 scale reference；若依赖和 checkpoint 可运行，再做同协议直接复现。不能运行时必须标记为 `reported`，不能与本地 Qwen-30B 结果混写。

### 3.4 是否存在类似 LLM-Deliberation 的问题

**不存在同等严重的问题，但原设置仍不足以单独证明 belief。**

积极方面：

- 玩家通常仅称 RED/BLUE，不靠“市长/环保组织”等语义身份表达效用；
- Buy/Sell reservation 与 Trading resources 确实私有；
- 接受、拒绝和反提议由真实对手 agent 产生，不是环境按 P1 最终答案自动判分；
- 多轮 offer 提供了能改变 posterior 的行为观测。

风险方面：

1. Buy/Sell 的主表大量使用固定 cost=40、WTP=60；中点、固定 reservation prior 或 benchmark contamination 都可能产生高分。论文虽另有从 `{20,40}` 与 `{60,80}` 抽样的行为实验，主比较仍不够多样。
2. buyer/seller 角色直接给出偏好单调方向。这是合理结构先验，但必须把 `prior-only` 与 history posterior 的贡献分开。
3. Ultimatum 公开总金额且是经典博弈；LLM 可能调用训练语料中的 50/50、70/30 先验。它比 Buy/Sell 更容易被 open-loop 策略解决。
4. Trading 的“资源总量/多样性”目标和固定 inventory pattern 可能被资源计数启发式利用；需生成有冲突的非线性 marginal values。
5. 原文没有 no-history、time-only、single-policy 或 shuffled-history 对照；因此“多轮对话存在”不等于“历史被有效利用”。
6. win rate 排除 ties；必须同时报告 no-deal、tie、agreement 与 payoff，尤其 social tactic 会用更高失败风险换 win rate。

所以 NegotiationArena 比 LLM-Deliberation 更有效，但必须先做去固定化与 communication-ablation 才能成为 belief 证据。

### 3.5 我们的 framework 如何接入

统一循环保持 domain-independent：

```text
public history + private self state
  -> environment adapter / observation normalizer
  -> continuous belief state B_t
  -> candidate proposer C_t
  -> belief-usable planner: rank(candidate | self utility, B_t, time, risk)
  -> generator: realize chosen action as valid XML message/offer
  -> opponent action -> B_{t+1}
```

domain adapter 只定义可观测量、合法 action 与 utility，不把固定策略写进核心模型：

- **Buy/Sell belief**：reservation value distribution、接受概率 `P(accept|price,t,history)`、concession slope、walk-away risk；
- **Trading belief**：每种资源的 latent marginal value、bundle ranking、inventory bounds、信息披露可靠性；
- **Ultimatum belief**：fairness/acceptance threshold、risk tolerance、persona response；
- **planner**：至少生成 exploit、agreement-safe、probe 三类 candidates，按期望效用、接受概率、信息价值和失败风险排序；
- **generator**：不得自行改变 selected numeric offer；其职责是公开消息和严格格式化。记录 `candidate -> chosen -> realized -> parsed`，隔离 planner gain 与 generation fidelity。

### 3.6 NegotiationArena 实验矩阵

#### Phase NA-0：复现与基础设施

- checkout/隔离 `paper_experiment_code`，不覆盖当前 main；
- Qwen-30B direct structured prompt，先做 20 局 smoke，再按原论文每个 ordered pair/scenario 60 局；
- 对正式结论采用至少 5 seeds，并报告 bootstrap 95% CI；
- 与 AMPO 比较时采用其 200 scenarios × 2 roles × 4 repeats protocol。

#### Phase NA-1：捷径审计

同一组 latent games 比较：

1. `static_prior_only`：不传任何 opponent history；
2. `time_only`：只传回合进度与上一 offer，不累积 belief；
3. `frozen_first_offer`：首个观察后停止更新；
4. `continuous_belief`：完整在线更新；
5. `shuffled_belief`：使用另一 episode/opponent 的 posterior；
6. `oracle_belief`：读取真实 reservation/utility，仅作上界。

同时做：角色名匿名化、buyer/seller 描述改写、金额缩放、未见 reservation ranges、随机 turn limits、inventory/goal permutation。若 full framework 在 shuffled history 或无 history 下几乎不降，则不能声称提升来自交互 belief。

#### Phase NA-2：核心 ablation

| Variant | Belief | Planner | Generator | 要回答的问题 |
|---|---|---|---|---|
| Direct | 无 | LLM 隐式 | LLM | 原始 prompting 基线 |
| Planner-only | prior/uniform | 有 | 固定 | 强搜索本身能做多少 |
| Belief-only | continuous | 简单规则 | 固定 | calibration 是否自然转化为收益 |
| Full | continuous | belief-aware | 固定/LLM | 核心方法 |
| Full-shuffled | 错配 posterior | 同 Full | 同 Full | planner 是否真正依赖正确 belief |
| Oracle | 真值 | 同 Full | 固定 | belief 改进空间 |

指标：own payoff、social welfare、agreement/no-deal/tie、role/first-mover 分层结果、illegal/parse rate、tokens/latency；belief 使用 reservation MAE/NLL、pairwise ranking、acceptance Brier/ECE；planner 使用 oracle candidate recall、chosen-vs-no-belief action change rate、expected-utility regret；generator 使用 numeric/action fidelity。

---

## 4. ANAC 2025 与 ANL 2025

### 4.1 名称边界

ANAC 2025 是 IJCAI 2025 的竞赛总称，官方总结聚焦两个 challenge：

1. **ANL 2025**：sequential one-to-many multi-deal negotiation；
2. **SCML 2025**：复杂供应链中的 concurrent negotiation，分 OneShot 与 Standard tracks。

本代码库当前接入的是 ANL 2025，而不是整个 ANAC。以下把 ANL 作为 P0，SCML 作为后续外部效度扩展。[ANAC 官网](https://anac.cs.brown.edu/)、[ANAC 2025 官方总结论文](https://arxiv.org/abs/2604.13914)、[ANL 2025 官方结果](https://anac.cs.brown.edu/anl2025)、[官方 agents 代码](https://github.com/autoneg/anl-agents)

### 4.2 ANL 2025 任务与 setting

- 一个 center 与多个 edge agents 依次进行双边 negotiation；参赛 agent 必须同时支持 center 与 edge 角色；
- 每个 thread 使用 alternating-offers protocol：接受、counteroffer 或 walk away；deadline 随机为 10–1000 steps；
- 自身 utility 完全可知，对手 utility 私有；
- center 的最终 utility 定义在所有 thread agreement 的 bundle 上，通常不是各局收益简单相加；早期协议因此改变后续最优 action；
- tournament 在随机 scenarios 上 round-robin。官方总结称 17 支队伍、12 个 finalists，ANL 按 individual utility 与 Nash distance 评估。

代表性 scenario：

- **Target Quantity**：center buyer 希望多个 seller 的成交数量总和接近 target，过多和不足都降低 utility；
- **Job Hunt**：center 与多个 employer 谈工资和 office days，最终取已获 deals 中的最大 utility；
- **Dinners/组合型场景**：需要组合多个 edge 的可用日或选择，单个局部“好 deal”未必形成全局好 bundle；
- 官方框架还支持 general global center ufun、homogeneous accumulation、side-utility max/linear combination 等类型。

这使 planner 面临两个不同的不确定性：当前对手会接受什么，以及剩余 threads 未来能获得什么。

### 4.3 官方 baseline、参赛方法与结果

基础 baseline 包括 time-dependent Boulware、Linear、Conceder、Random 与官方默认策略。2025 年主要方法可分为：

- **Pessimistic continuation**：假设未来不再成交；UFunAt、SAC、default 属于此类；
- **Contingent continuation**：对未来 deals 维护概率分布；ProbaBot、RUFL、RivAgent、WAgent 属于此类；
- **Optimistic continuation**：假设未来按规划实现；WAgent 在部分条件采用；
- 其他技术包括动态 target、sampling、dynamic programming 与 reinforcement learning。

前三名官方结果为：

| 排名 | Agent | Center | Edge | Final |
|---:|---|---:|---:|---:|
| 1 | RUFL | 0.714 | 0.084 | **0.399** |
| 1 | SAC Agent | **0.733** | 0.064 | **0.399** |
| 3 | UFunAtAgent | 0.686 | 0.078 | 0.382 |

这里按官方表保留指标名称和数值；不能把 `Edge` 列误称为简单成交 utility。正式复现还应通过官方 scorer 输出 individual advantage、Nash distance 和 final score，而不是用本地自定义平均值替代。[官方结果 slides](https://anac.cs.brown.edu/files/anl/y2025/anl2025results.pdf)

#### RUFL：Utility Fit Lookahead

- 构造剩余 subnegotiation outcome tree，估计 partial bundle 的 future utility；
- 对 child expected values 做带 temperature 的 softmax，形成未来 outcome 概率；
- 通过截断深度、近似和提前终止控制组合爆炸；
- 当前对手侧结合 time concession 与 opponent offers 的 utility fit。

RUFL 是与我们最直接的 baseline：它已有显式 continuation distribution，但不等于有经 calibration 验证的 per-opponent continuous belief。[RUFL 官方报告](https://anac.cs.brown.edu/files/anl/y2025/reports/20826_Team%20271_RUFL.pdf)

#### SAC Agent

- 使用 Soft Actor-Critic；
- bidding policy 是 time-dependent concession，SAC 模型指导 concession rate；
- 对 future deals 采取 pessimistic assumption，仍与 RUFL 并列第一。

这一结果是重要反例：ANL 高分并不自动意味着 opponent belief 有效，强让步控制与自身 utility planning 可能已经足够强。[SAC 官方报告](https://anac.cs.brown.edu/files/anl/y2025/reports/20805_University%20of%20Tehran_SacAgent.pdf)

#### UFunAt

- 第三名；采用 pessimistic continuation；
- 应作为开源 finalist baseline 保留，但在本地兼容性验证完成前，不把启动失败记为方法失败。

### 4.4 SCML 2025 简述

SCML 的 OneShot 研究每个模拟日内的 repeated concurrent many-to-many negotiations；Standard 加入非易腐库存、delivery date、仓储成本和同时买卖。最终目标为累计利润。

- OneShot：2024 CautiousOneShotAgent 仍取得最高对照分 1.0915；2025 CostAverse 1.0896，Rchan 1.0895，AlmostEqual 1.0892，三者并列冠军；
- Standard：AS0 1.009 冠军，Penguin 0.992；
- 官方总结认为风险控制和领域启发式仍常常优于复杂 opponent modeling。

SCML 很有价值，但它同时引入库存、生产、违约和市场动态，会使 belief/planner 的因果归因更困难。因此应在 ANL 证据稳定后接入，而不是现在替代 ANL。

### 4.5 ANL 是否有与 LLM-Deliberation 类似的问题

**没有相同的结构性缺陷，是两个候选中更强的 belief/planner benchmark。**

理由：

- 通常是结构化 outcome/utility，而非可从角色名称直接猜出的自然语言偏好；
- 对手 ufun 私有且 scenario 随机；
- agreement 需要对手真实接受，center 不能独自输出一个“多数通过”的最终答案；
- 多个协议依次且不可逆地进入 global center utility，不能只优化最后一次输出；
- 真值 ufun 可供 evaluator 离线计算 belief calibration 与 oracle regret，这是 LLM 环境很难提供的优势。

但仍有四类绕行路径：

1. agent 可只优化已知 global center ufun，加 time-based concession，而不建模具体对手；SAC 的结果说明这一路径很强。
2. scenario generator 与 opponent pool 的分布可能形成 tournament prior；模型可能识别常见 ufun/策略族，而不是在线适应当前对手。
3. deadline 可达 1000 steps，简单时间策略能通过大量试探逐步成交；必须按 short/medium/long horizon 分层。
4. ANL 没有自然语言 generator，因此它证明的是 structured preference learning/planning，不证明语义 theory-of-mind 或自然语言策略执行。

这里不存在有意义的“single-agent 投票”版本；正确的对应 baseline 是 `open-loop/time-only negotiator`。还必须隐藏 opponent class/name/NMI 中不必要的实现身份，并在 test 时使用未见策略族和未见 ufun family。

### 4.6 我们的双层 belief 与 planner

当前本地 V0 已有：per-opponent offer frequency belief、continuous update、基于 own utility/opponent proxy/uncertainty 的 candidate 排序。但它仍是**单 thread planner**，没有正式解决 ANL 的核心 multi-deal 问题。

正式版本拆成：

#### Local Opponent Belief `B_local(i,t)`

- 对手 issue/value ranking 或 utility posterior；
- reservation/disagreement threshold；
- concession curve 与 deadline sensitivity；
- `P(accept | offer, t, history)`；
- uncertainty 与 OOD flag。

#### Continuation Belief `B_future(k+1:N)`

- 剩余每个 thread 的 achievable-outcome distribution；
- deal/no-deal probability；
- 未知 opponent strategy/utility family 的分层 posterior；
- 可采用 RUFL-style tree 作为非学习起点，再训练 continuation value model。

#### Belief-usable Planner

对当前候选 `o_k` 计算：

```text
P_accept(o_k | B_local)
  × E[U_center(A_1:k-1, o_k, O_k+1:N) | B_future]
  + information_gain(o_k)
  - no_deal / opportunity / compute risk
```

center 与 edge 应有不同 planner head。edge 主要是 bilateral acceptance/concession；center 必须显式输入 current bundle、remaining threads 与 global ufun。核心接口保持通用，只有 outcome enumeration 和 utility adapter 属于 ANL。

### 4.7 ANL 实验矩阵

#### Phase ANL-0：兼容与官方 scorer

当前本地 smoke 结果：Boulware、Linear、framework 各 2 runs 都得到 mean center utility 0.9、agreement 1.0；样本太小且场景过简单，只证明 pipeline 可运行。RUFL 当前在 not-started future-thread NMI 上与 `anl2025/negmas` 版本不兼容，1 run 为 error；这不是 RUFL 性能结果。

必须先：

- 固定官方推荐的 `anl2025`、`negmas`、`anl-agents` commit 与环境；
- 分别验证 RUFL、SAC、UFunAt 的 center/edge compatibility；
- 接入官方 tournament/scorer；
- 保存每个 session 的 scenarios、ufun hash、role、seed、deadline、offers/actions、runtime/error；
- compatibility failure 与 negotiation failure 分开统计。

#### Phase ANL-1：两阶段 baseline

1. 小型稳定集：TargetQuantity、JobHunt、Dinners，各 100 scenarios × 5 seeds × center/edge role；
2. 正式 tournament：官方 opponent pool 与官方 scenario distribution；
3. 强 baseline：Boulware、Linear、Conceder、Random、default、RUFL、SAC、UFunAt；
4. 训练型方法必须按 scenario seed、opponent strategy family 和 ufun family 划分 train/validation/test，禁止同实例泄漏。

#### Phase ANL-2：belief 因果消融

| Variant | Local belief | Future belief | Planner |
|---|---|---|---|
| Time-only | 无 | pessimistic | time concession |
| Planner-only | uniform prior | RUFL-style / known prior | bundle look-ahead |
| Frozen | first offer 后冻结 | fixed prior | 同 Full |
| Local-only | continuous | pessimistic | belief-aware bilateral |
| Continuation-only | uniform | learned/softmax | bundle look-ahead |
| Full-dual | continuous | continuous/learned | dual-belief planner |
| Shuffled-local | 其他对手 posterior | 同 Full | 同 Full |
| Shuffled-future | 正确 | 其他 scenario posterior | 同 Full |
| Oracle-local | true edge ufun | 同 Full | 上界 |
| Oracle-full | true edge/future outcomes | oracle rollout | 总上界 |

关键不是只检验 `Full > Time-only`，还要求：

- `Full > Planner-only`：在线 local belief 有增益；
- `Full > Local-only`：跨 thread continuation planning 有增益；
- `Full > Shuffled`：正确 posterior 有因果价值；
- `Oracle > Full`：有合理且可解释的改进空间；
- belief calibration 随回合改善，并与 action/utility improvement 相关。

#### Phase ANL-3：反捷径与泛化

- 隐藏/随机化 opponent agent 名称与 Python class identity；
- held-out strategy：训练内置 time strategies，测试 RUFL/SAC/UFunAt 或反向划分；
- held-out ufun family：例如训练 TargetQuantity，测试 max/combination/新 target shape；
- distribution shift：outcome-space size、edge count、deadline、reserved values、thread order；
- cross-thread identity：同一 opponent 多次出现时允许 belief memory；换对手时强制清空，并以错误共享作为负对照；
- history intervention：截断、shuffle、反事实替换 offers，测 action sensitivity。

#### 指标

- 官方：final score、individual advantage、Nash distance；
- outcome：center utility、edge utility、social welfare、agreement/no-deal、role gap；
- belief：utility/ranking correlation、pairwise accuracy、acceptance Brier/ECE、NLL、posterior contraction；
- planner：candidate oracle recall、terminal-utility regret、accepted-deal opportunity cost、action change rate；
- 系统：runtime、timeout、error、memory、每 action compute/tokens。

---

## 5. 两个 challenge 的统一研究设计

### 5.1 共同接口，分离 environment-specific 部分

```text
EnvironmentAdapter
  observe() / legal_actions() / self_utility() / terminal_score()

BeliefModel
  prior(context) / update(observation) / predict(candidate) / uncertainty()

CandidateGenerator
  exploit() / safe_agreement() / probe()

BeliefUsablePlanner
  rank(candidates, belief, self_state, horizon)

ActionRealizer
  structured action (ANL)
  natural-language + XML action (NegotiationArena)
```

训练时不直接把 `buyer=某固定价格`、`TargetQuantity=某固定 target` 写入核心 prompt/model；domain knowledge 只以 schema、合法 action 与 prior feature 进入 adapter。这样才能回到 CaSiNo 时复用相同 `BeliefModel/Planner`。

### 5.2 统一的四个研究问题

1. **RQ1 Calibration**：continuous update 是否比 prior/frozen 更准确？
2. **RQ2 Usability**：正确 belief 是否改变 candidate ranking 与 accept/reject？
3. **RQ3 Utility**：这种变化是否在控制 planner capacity 后提高收益？
4. **RQ4 Transfer**：相同核心模块是否跨 NegotiationArena、ANL、CaSiNo 和未见对手成立？

### 5.3 预注册式成功标准

正式实验前固定阈值，避免 auto-research 只追逐单一 utility：

- 主任务至少 3 个独立 seeds/批次，正式表建议 5 seeds；
- Full 相对 Planner-only 与 Frozen 的 primary score 提升，bootstrap 95% CI 不跨 0；
- Full 相对 Shuffled belief 有显著下降反转，即正确 posterior 确实重要；
- acceptance Brier/ECE 或 ranking accuracy 随 observation 改善；
- oracle 有正 headroom，且 Full 位于 uniform 与 oracle 之间；
- NegotiationArena parse/realization error <1%，ANL compatibility error 单独为 0；
- OOD opponent/ufun 上保留大部分 ID gain；
- 同时报 utility、agreement、fairness/Nash、no-deal 与成本，不允许只挑 win rate。

---

## 6. 实施路线与交付物

### Sprint 0（2–3 天）：冻结复现环境

- NegotiationArena 建立 paper branch worktree/adapter，确认三任务 round/score；
- ANL 修复官方 agents 版本兼容，接入官方 scorer；
- 建立统一 manifest：commit、model endpoint/name、sampling、seed、prompt hash、scenario hash；
- 交付：20-run smoke 仅作工程验收，不作论文结论。

### Sprint 1（1 周）：无训练的 shortcut audit

- NegotiationArena：static/time-only/frozen/continuous/shuffled/oracle；随机 reservation 与金额缩放；
- ANL：time-only/planner-only/frozen/current-V0/shuffled/oracle；
- 输出 calibration curve、action sensitivity、utility/role 分层图；
- 决策门：若 no-history 接近 Full，先修 benchmark split/任务生成，不进入 calibration training。

### Sprint 2（1–2 周）：ANL dual-belief planner

- local acceptance/utility posterior；
- RUFL-style enumerative continuation baseline；
- current bundle + remaining threads 的 terminal utility rollout；
- Full-dual、Local-only、Continuation-only 消融；
- 与 RUFL/SAC/UFunAt 正式比较。

### Sprint 3（1–2 周）：NegotiationArena 自然语言迁移

- 将同一 belief state 和 planner candidate schema 映射到 Buy/Sell、Trading；
- generator fidelity gate，planner 数值 action 不允许被语言生成器改写；
- 对比 direct Qwen-30B、planner-only、Full、AMPO/GRPO reported 或可运行 checkpoint；
- Ultimatum 只作为 fairness/risk 辅助任务。

### Sprint 4：training 与回到 CaSiNo

- 只有 Sprint 1–3 证明 history 有增量价值后，才做 belief calibration training；
- SFT target 应是可验证的 posterior/ranking/acceptance label，而不是把含 `flexibility_delta` 的不确定 JSON 当绝对真值；
- 先训通用 `predict preference/acceptance + calibrated uncertainty`，再训 planner 使用 belief；
- 最后在 CaSiNo 做 zero-shot transfer、少量 domain adaptation 与原 specialized model 对照。

### 建议目录

```text
external_negotiation_envs/
  common_belief_framework/
    belief/ planner/ candidates/ metrics/ schemas/
  NegotiationArena/
    adapters/ variants/ scripts/ runs/
  ANL2025/
    anl_integration/ variants/ scripts/ runs/

8.4/
  NEGOTIATIONARENA_ANAC2025_BELIEF_FRAMEWORK_PLAN_V2_CN.md
  experiment_registry.jsonl        # 后续创建
  result_tables/                   # 后续生成
```

每次 auto-research iteration 仍需保留：parent code commit/patch、完整 config、trajectory 或脱敏摘要、metrics、Codex proposal、实际 diff、测试日志、是否接受该 variant。不得原地覆盖历史 variant。

---

## 7. 最终 benchmark 组合建议

1. **主算法证据：ANL 2025**
   用 structured offers、真实私有效用、官方强 agent 与跨 thread global utility，验证 continuous belief 和 belief-usable planning 的因果价值。

2. **自然语言迁移证据：NegotiationArena Buy/Sell + Trading**
   在随机化、去固定化设置下验证从对话抽取 belief，以及 generator 执行 planner 的能力。

3. **辅助行为分析：NegotiationArena Ultimatum**
   用于 fairness threshold、persona、风险与 social tactic，不作为 belief 主结论。

4. **legacy 与外部效度：LLM-Deliberation、CaSiNo、未来 SCML**
   LLM-Deliberation 说明 benchmark shortcut；CaSiNo 保留已有结果并检验迁移；SCML 在核心因果证据稳定后扩展到并发市场。

用这一组合，我们的论文主张应从“framework 在某环境得分更高”收紧为：**在线观测使对手 posterior 可测地改善；belief-aware planner 对该 posterior 产生正确且可干预的动作变化；这种机制跨结构化 multi-deal 与自然语言 bilateral negotiation 均能提高结果，并能迁移回 CaSiNo。**

## 8. 主要资料

- [NegotiationArena，ICML 2024](https://proceedings.mlr.press/v235/bianchi24a.html)
- [NegotiationArena 官方代码](https://github.com/vinid/NegotiationArena)
- [ASL/AMPO，ICLR 2026](https://openreview.net/forum?id=GG7YQnsdhp)
- [ASL/AMPO 官方代码](https://github.com/MozerWang/AMPO)
- [ANAC 官网](https://anac.cs.brown.edu/)
- [ANAC 2025 challenges and results](https://arxiv.org/abs/2604.13914)
- [ANL 2025 结果页](https://anac.cs.brown.edu/anl2025)
- [ANL 2025 结果 slides](https://anac.cs.brown.edu/files/anl/y2025/anl2025results.pdf)
- [ANL agents 官方代码与报告](https://github.com/autoneg/anl-agents)
- [RUFL agent report](https://anac.cs.brown.edu/files/anl/y2025/reports/20826_Team%20271_RUFL.pdf)
- [SAC agent report](https://anac.cs.brown.edu/files/anl/y2025/reports/20805_University%20of%20Tehran_SacAgent.pdf)
