# 外部谈判 Benchmark 复现与对比计划

本文整理 5 个近期谈判/偏好建模相关工作：

1. **Bilateral Trade with Private Information**：`2604.16472`
2. **RLVR Negotiation**：`2604.09855`
3. **TERMS-Bench**：`2605.13909`
4. **Preference Estimation via Opponent Modeling**：`2604.15687`
5. **BOND**：`2605.04507`

目标是判断：这些工作分别是什么 benchmark / method，论文 table 如何得到，metrics 是什么，我们是否能复现，是否能把我们的 framework 和它们的 baseline 直接或近似对比，以及优先级如何安排。

## 总体结论

| 工作 | 环境类型 | 是否适合测我们的 full framework | 是否能严格复现原表 | 推荐优先级 |
|---|---|---:|---:|---:|
| TERMS-Bench | 固定 simulator bilateral price negotiation | 高 | 中-高，取决于代码/数据可得性 | P0 |
| RLVR Negotiation | buyer vs regulated seller, verifiable reward | 高 | 中，缺 exact ckpt/split | P0 |
| Bilateral Trade | structured tool-call bilateral trade | 高 | 中，缺 exact split/prompt/code | P1 |
| BOND | CaSiNo belief-state distillation / posterior evaluation | 中，适合测 belief | 高，若代码可用 | P1 |
| Preference Estimation | 多方多议题 preference estimation | 中-低，需较多重建 | 低-中 | P2 |

最推荐优先复现的是 **TERMS-Bench** 和 **RLVR Negotiation**。前者能直接检验 negotiation policy 的 surplus / agreement / belief / violation；后者和我们 buyer-side framework、fixed seller 评估最贴近。BOND 很适合补充一个“belief model 是否真的好”的独立实验，但它不是完整谈判收益 benchmark。Preference Estimation 是多方共识场景，概念相关但工程成本较高，和我们当前 buyer-seller setting 距离最大。

## 1. Bilateral Trade with Private Information

链接：[Training Language Models for Bilateral Trade with Private Information](https://arxiv.org/pdf/2604.16472)

### Benchmark 是什么

这是一个结构化 bilateral bargaining 环境。buyer 和 seller 围绕单个不可分割商品做多轮报价。buyer 有私有 reservation price `b`，seller 有私有 reservation cost `s`。如果成交价为 `p`：

- buyer utility: `b - p`
- seller utility: `p - s`
- 如果 buyer 接受 `p > b` 或 seller 接受 `p < s`，就是 individual rationality violation
- 如果 `b >= s`，存在 ZOPA `[s, b]`，是 GFT scenario
- 如果 `b < s`，无 mutually beneficial deal，是 NGFT scenario

环境强调“binding offer”和自然语言消息分离，agent 通过 tool calls 发出正式 offer / accept / reject / message。这样每个 offer、accept、reject 都可机器解析。

### Method / Baseline

论文主要做两类实验：

1. **frontier model round-robin benchmark**
   - 5 个 frontier models：DeepSeek-V3-0324、Gemini-2.5-Flash、Gemini-2.5-Pro、GPT-4.1、o3
   - 每个模型既当 buyer 又当 seller
   - 25 个 buyer-seller pair
   - 每个 pair 跑 600 个 product listings
   - 总计 `5 x 5 x 600 = 15,000` negotiations

2. **open-weight training**
   - Qwen3 8B / 14B
   - SFT + GRPO
   - against fixed frontier opponent
   - 发现 SFT 提升 surplus share 但降低 deal rate，RL 恢复 deal rate 但削弱 surplus gain

### Table / Metrics 如何得到

论文按三个维度评价：

- **Individual rationality**
  - self violation rate
  - opponent violation induced rate
  - GFT / NGFT 分开统计

- **Strategic effectiveness**
  - buyer surplus share
  - seller surplus share
  - total surplus fraction
  - initial aggressiveness
  - concession rate

- **Allocative efficiency**
  - deal rate
  - efficient trade rate
  - NGFT walk-away correctness

此外还做 price-tier decomposition：按 buyer value / seller cost quintile 分解 surplus share 和 deal rate，观察模型是否能在不同价格层级保持稳定策略。

### 可复现性

**可做 paper-aligned reimplementation，但不能严格 claim 复现原表。**

原因：

- 论文没有提供官方代码。
- product catalog 来自 CraigslistBargains + Amazon Price History，但 exact split / seed / prompt / tool schema 未完整公开。
- frontier model snapshot 和 API 行为可能变化。

本地已有近似实现：

- [bilateral_trade_env.py](/work5/qixint/experiments/external_comparisons/bilateral_trade_env.py)
- [run_bilateral_trade_comparison.py](/work5/qixint/experiments/external_comparisons/run_bilateral_trade_comparison.py)

当前实现已经覆盖：

- private buyer value / seller cost
- structured actions
- GFT / NGFT
- deal rate
- IR
- surplus share
- price-tier breakdown
- 600 episodes per pair 的 paper-scale 结构

### 如何和我们的 framework 对比

推荐对比方式：

| Row | Buyer | Seller | 用途 |
|---|---|---|---|
| direct prompt | our direct buyer | fixed scripted / LLM seller | prompt baseline |
| belief-only | our belief buyer | same seller | 测 opponent modeling |
| planner+generator | our B+C buyer | same seller | 测 strategy/action |
| full framework | our A+B+C buyer | same seller | 主结果 |
| paper-style policies | scripted aggressive / fair / concessive | same seller | sanity baseline |

指标对齐论文：

- buyer surplus share
- deal rate in GFT
- walk-away correctness in NGFT
- buyer IR violation / budget violation
- concession rate
- initial aggressiveness
- price-tier stability

### 优先级

**P1。**

这个 benchmark 与 AgenticPay / RLVR 很近，适合展示我们的 framework 在 private-information price bargaining 中的能力。但由于不能严格复现原 table，优先级略低于 TERMS-Bench 和 RLVR。

## 2. RLVR Negotiation

链接：[Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards](https://arxiv.org/pdf/2604.09855)

### Benchmark 是什么

这是一个 buyer-side price negotiation 环境。buyer 有 private budget `B`，seller 有 private cost `C`。seller 是 regulated seller，不会接受低于 `C` 的价格。episode 以 `[DEAL]`、`[QUIT]` 或最大轮数结束。

核心 reward：

```text
R = (B - P_final) / |B - C|
```

然后 clip 到 `[-1, 1]`。

特殊规则：

- MI scenario: `B > C`，存在 mutually beneficial deal
- CI scenario: `B < C`，理性行为是不成交 / quit
- no deal / quit / timeout reward = `0`
- buyer 超预算或格式错误 reward = `-1`

### Method / Baseline

论文训练 Qwen3-30B-A3B-Instruct-2507 buyer：

- 训练 seller：regulated Qwen3-30B-A3B seller
- 数据：AmazonHistoryPrice
- 802 train products + 128 held-out test instances
- 训练方法：RL with verifiable rewards
- Table 1 在 neutral seller persona 上评价
- Table 2 against unseen frontier seller
- Table 3 against adversarial seller personas: begging / insulting / unyielding

### Table / Metrics 如何得到

Table 1：

- 128 held-out test instances
- 每个 instance 评估 4 次
- neutral seller prompt
- buyer models 包括 trained Qwen、untrained Qwen、GPT、Kimi、DeepSeek、Llama、gpt-oss 等

主要 metrics：

- **Reward**
  - 主要优化目标
  - 综合 surplus extraction、deal success、constraint obedience

- **Deal Rate**
  - 达成协议比例

- **Bargained Ratio**
  - 只在 successful deals 上计算
  - 衡量 buyer capture 的 surplus fraction

- **Price Overshoot Rate**
  - buyer 超预算比例

论文 Table 1 里 trained Qwen3-30B 达到 Reward `0.7664`、Deal Rate `91.99%`、Bargained Ratio `0.8385`、Price Overshoot `0.10%`。

### 可复现性

**中等。环境和 metrics 容易复现，原表严格复现较难。**

困难点：

- 没有公开官方代码。
- 没有 trained Qwen3-30B checkpoint。
- AmazonHistoryPrice exact split 未必完整可得。
- seller prompt 和动作解析细节需要重建。

本地已有 wrapper：

- [run_rlvr_bilateral_trade_wrapper.py](/work5/qixint/experiments/external_comparisons/run_rlvr_bilateral_trade_wrapper.py)

当前实现已经覆盖：

- buyer budget `B`
- regulated seller cost `C`
- reward `(B - P_final) / |B - C|`
- no-deal reward
- overshoot penalty
- bargain ratio
- deal ratio
- first-turn offer ratio
- overshoot rate

### 如何和我们的 framework 对比

这是最适合我们当前工作的一类实验。可以固定 seller，替换 buyer：

| Variant | 说明 |
|---|---|
| direct prompt | 直接谈判 |
| CoT prompt | chain-of-thought prompt |
| belief model | 只注入 seller cost / reservation belief |
| planner+generator | 只用 strategy/action + naturalizer |
| full framework | belief + planner + generator |
| gated full framework | price-only 场景关闭或约束 planner |

需要输出与论文 Table 1 同构的表：

```text
Buyer Model | Reward | Deal Rate | Bargained Ratio | Price Overshoot Rate
```

可扩展成三张表：

1. neutral seller
2. unseen frontier seller
3. adversarial personas: begging / insulting / unyielding

### 优先级

**P0。**

它和我们的 buyer-side framework 最贴近，metrics 清晰、verifiable、容易解释，也能直接验证我们之前发现的“planner/generator 可能过度让步”问题。

## 3. TERMS-Bench

链接：[TERMS-Bench project site](https://terms-bench.github.io/)；论文：[TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond Deal Rate](https://arxiv.org/abs/2605.13909)

### Benchmark 是什么

TERMS-Bench 是一个 diagnostic benchmark for LLM negotiation agents。它不是 LLM-vs-LLM，而是让 agent 和一个固定 stochastic simulator counterpart 进行多轮 alternating-offer bilateral price negotiation。

核心设计：

- 单商品 bilateral price negotiation
- counterpart 是固定随机策略，不是另一个 LLM
- 每个 episode 可由 seed 复现
- agent 只看到 counterpart 的报价、sentiment cue、stance cue
- agent 返回 `Offer(price, message)`、`Accept` 或 `Reject`
- 最大 `K = 10` rounds
- 环境强制 price bounds、monotonic concession、turn budget

每方有 private type：

```text
t = (r, kappa, eta)
```

其中包括 reservation price、urgency、stance。surplus 只有在 agreed price 落入 ZOPA `[r_seller, r_buyer]` 时才实现。

### Method / Baseline

论文/网站评测 13 个 LLM agents，覆盖 frontier、open-weight、sub-frontier 模型。benchmark 设计强调“环境自身作为 verifier”，因为 evaluator 知道 counterpart latent type、policy、payoff structure。

TERMS-Bench 不使用 LLM judge，也不把多项指标合成一个总分，而是分开报告 diagnostic axes。

还包含两个实用模式：

1. **TERMS-Commerce**
   - 把 negotiated price 转换为真实 dollar profit
   - Merchant: agent 从 supplier 买入
   - Vendor: agent 卖给 customer

2. **TERMS-Bankroll**
   - stateful procurement chain
   - 每个 agent 跑 4 个 merchant-side sessions
   - 每个 session 初始 `$100` bankroll
   - 每个 session 50 negotiation periods
   - cash balance 跨 period 保留
   - headline 是 terminal balance

### Table / Metrics 如何得到

主 leaderboard metrics：

- **SE+**
  - feasible surplus efficiency
  - 在 feasible episodes 上捕获的 bargaining surplus fraction

- **AGR+**
  - feasible agreement rate
  - feasible episodes 中达成协议的比例

- **CSE+**
  - conditional feasible deal quality
  - 只在达成 agreement 的 feasible episodes 上计算 surplus quality

- **FAGR-**
  - no-deal false agreement rate
  - infeasible episodes 中错误达成协议的比例

- **BE_type**
  - belief error
  - agent 声明的 counterpart reservation / urgency / stance belief 的 aggregate error

- **CritViol%**
  - critical violations
  - price-bound、individual-rationality、invalid-action violation

- **U-bar**
  - raw mean utility

Commerce table 额外包括：

- Total profit
- Avg / episode
- Avg margin
- Negative profit %
- Walk-away %
- Money left
- Regret %
- Episodes

Bankroll table 包括：

- Terminal $
- SEM
- Avg / period
- Survival
- Ruin @
- Max drawdown
- Memory premium
- Sessions

### 可复现性

**如果代码/runner 可获得，是最高价值复现对象；如果只根据网页重建，仍然可做 close reimplementation。**

优点：

- counterpart 固定，不是 LLM-vs-LLM，方差更可控。
- 所有 metrics 都程序化计算，无 LLM judge。
- agent interface 是 JSON-in / JSON-out contract，容易接我们 framework。
- 包含 belief error，能直接评估我们的 belief model。
- data-grounded robustness 使用 AmazonHistoryPrice，可和 RLVR / bilateral trade 连接。

不确定点：

- 当前网页公开了设计和 leaderboard，但需要确认是否有完整 code / scenario seeds / counterpart policy 参数。
- 如果没有 exact seeds 和 policy parameters，只能做 paper-aligned TERMS-like simulator。

### 如何和我们的 framework 对比

非常适合。我们的 framework 可以作为 TERMS agent：

输入：

- counterpart price
- sentiment cue
- stance cue
- round id
- current bounds
- own reservation / role

输出：

- action: offer / accept / reject
- price
- message
- belief state: reservation estimate、urgency estimate、stance estimate

推荐表格：

```text
Agent | SE+ | AGR+ | CSE+ | FAGR- | BE_type | CritViol% | U-bar
```

我们的 ablation：

- direct prompt
- CoT prompt
- belief-only
- planner+generator
- full framework
- gated planner

特别要观察：

- full framework 是否提升 `BE_type`
- planner 是否提升 `SE+` 但降低 `AGR+`
- gated planner 是否降低 `FAGR-` 和 `CritViol%`
- 是否在 hard difficulty bin 中更稳定

### 优先级

**P0，甚至高于 RLVR。**

理由：

- 它是专门为“诊断 LLM negotiation agents”设计的。
- 固定 counterpart 让对比更干净。
- 同时有 surplus、agreement、belief error、violation。
- 能直接呈现我们的 framework 的模块优势。

## 4. Preference Estimation via Opponent Modeling

链接：[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](https://arxiv.org/pdf/2604.15687)

### Benchmark 是什么

这是一个多方、多议题、共识型 negotiation benchmark。场景是 sports facility / Harbour Sport Park negotiation：

- `N = 6` stakeholders
- 包括两个 veto holders
- `M = 5` issues
- 每个 issue 有 3 到 5 个 options
- 每轮由指定 party 提出 deal 和 utterance
- 最多 `T = 24` rounds
- 每个 party 有 private score function 和 reservation threshold

该场景非常稀疏：

- 所有 720 个 possible deals 中，只有 21 个，即 2.9%，能让至少五方且包括 veto holders 满足 reservation thresholds
- 只有 3 个，即 0.4%，能让所有六方 full agreement

### Method / Baseline

论文提出的是 preference estimation method，而不是单纯 negotiation policy。核心是把 LLM 从 utterance 中抽取出的 qualitative cues 转换为 probabilistic signals，并放入 structured Bayesian opponent modeling。

比较方法：

- **Proposed**
  - `p1`: 只有 leader p1 做 preference estimation
  - `all`: 所有 agents 都做 mutual preference estimation

- **Base-LLM**
  - 原始 prompting，不显式做 preference estimation

- **Base-OM**
  - 传统 Bayesian opponent modeling
  - 只用 deal history，不用自然语言信息

- **LLM-PE**
  - LLM 直接预测 opponent score functions
  - 不用 structured Bayesian framework

所有方法使用 GPT-4.1。

### Table / Metrics 如何得到

每个 method 跑 500 independent negotiation trials。

Table 1 指标：

- **FAR**
  - Full Agreement Rate
  - 所有六方达成 consensus 的比例

- **PAR**
  - Partial Agreement Rate
  - final round 中至少五方且包括 veto holders 达成 agreement 的比例

- **LAR**
  - Latent Agreement Rate
  - T-round 过程中至少提出过一个 valid deal 的比例

Table 1 结果：

| Method | FAR | PAR | LAR |
|---|---:|---:|---:|
| Proposed (p1) | 0.46 | 0.78 | 0.96 |
| Proposed (all) | 0.62 | 0.89 | 0.98 |
| Base-LLM | 0.37 | 0.76 | 0.97 |
| Base-OM (p1) | 0.45 | 0.82 | 0.97 |
| Base-OM (all) | 0.56 | 0.92 | 0.99 |
| LLM-PE (p1) | 0.40 | 0.75 | 0.97 |
| LLM-PE (all) | 0.32 | 0.69 | 0.93 |

Table 2 指标：

- p1 对各 opponent score functions 的 MSE
- Proposed avg MSE `159`
- Base-OM avg MSE `189`
- LLM-PE avg MSE `163`

### 可复现性

**理论上可重建，工程成本较高。**

有利点：

- appendix 给出 parties、issues、options、preference profiles、reservation thresholds。
- 说明了 hypothesis space：
  - issue ranking permutations: `5! = 120`
  - evaluation function combinations: `3 x 3 x 4 x 4 x 5 = 720`
  - numerical offer likelihood sigma = `1.0`
  - LLM temperature = `0`
- 使用 open-source environment from Abdelnabi et al. 2024。

困难点：

- 多方 negotiation 环境和我们当前 bilateral buyer-seller pipeline 差距大。
- 需要实现 6-agent protocol、veto holder、multi-issue deal validity。
- 500 trials 成本较高。
- 它的主要 contribution 是 preference estimation，不是 price bargaining policy。

### 如何和我们的 framework 对比

不建议直接把我们的 buyer framework 拿去和它的 table 主结果比较，因为任务形态不同。

更合理的对比：

1. 把我们的 belief model 改造成 multi-party preference estimator。
2. 用相同 setting 输出：
   - FAR / PAR / LAR
   - MSE of estimated score functions
3. 对比：
   - Base-LLM
   - Base-OM
   - LLM-PE
   - Proposed
   - Our belief + planner

我们的优势可以表述为：

- richer belief state
- evidence-backed preference ranges
- planner/action separation

但这会是一个独立工程，不是当前 AgenticPay/ASTRA 代码的小改。

### 优先级

**P2。**

适合在论文中作为“multi-party preference-estimation extension”，但不应放在最先做。它和我们的主线 negotiation buyer framework 有一定距离。

## 5. BOND

链接：[Distilling Bayesian Belief States into Language Models for Auditable Negotiation](https://arxiv.org/pdf/2605.04507)

### Benchmark 是什么

BOND 使用 CaSiNo negotiation dataset，关注 opponent priority ordering 的 belief-state prediction，而不是仅看最终谈判分数。CaSiNo 中有 food / water / firewood 三种物品，对手 priority ordering 有 6 种可能。

主评估：

- 150-dialogue held-out CaSiNo subset
- 从 `mturk_agent_1` perspective 产生 1054 turn-level predictions
- 评估每个 turn 的 posterior over six opponent-priority hypotheses

还做外部 comparison：

- CaSiNo opponent priority-ordering prediction
- k-penalty protocol
- 对比 Chawla et al. 2022 prior rankers 和 70B structured-CoT baseline

### Method / Baseline

BOND 是 Bayesian Opponent-belief Negotiation Distillation。

组成：

1. **Bayesian teacher**
   - Llama-3.1-8B-Instruct + LoRA
   - LLM 作为 likelihood scorer
   - 对六种 opponent priority ordering 打分
   - Bayesian module 更新 posterior
   - menu planner 使用 posterior 做 accept / counteroffer / walk-away / utterance decision

2. **Distilled student**
   - 同样是 8B base family + LoRA
   - 训练目标是输出：
     - normalized posterior
     - intent
     - selected content
     - utterance

3. **70B structured-CoT baseline**
   - 不天然输出 posterior
   - 通过 self-consistency elicitation 得到 posterior

### Table / Metrics 如何得到

核心 belief metrics：

- **Brier score**
  - turn-level posterior calibration
  - uniform six-way posterior reference 是 `5/36 ≈ 0.139`
  - Bayesian teacher Brier = `0.085`
  - distilled student Brier = `0.114`
  - 70B structured-CoT elicited posterior Brier = `0.194`

Table 1 / Table 17 外部 CaSiNo comparison：

| Model | EMA | Top-1 | NDCG@3 |
|---|---:|---:|---:|
| BoW-Ranker | 27.71 | 52.98 | 64.31 |
| BERT CD+CA+DND | 44.22 | 69.21 | 76.03 |
| RoBERTa CD+CA+DND | 48.72 | 70.03 | 77.14 |
| 70B structured-CoT prompted | 37.30 | 62.99 | 70.94 |
| BOND Supervised SFT 8B | 53.84 | 76.21 | 80.88 |

另有 auditability metrics：

- MAP correctness
- posterior confidence
- entropy
- action/menu alignment
- belief-error vs action-error decomposition
- Accept-F1
- bid cosine

### 可复现性

**相对较高，如果代码仓库可用。**

论文声称 code available：

- `https://github.com/kaneis1/CaSiNo_negotiation-agent`

有利点：

- CaSiNo dataset 公开程度较高。
- BOND 明确给出 split、Protocol 3、turn-level records。
- 提供 LoRA training details：
  - rank 16
  - alpha 32
  - dropout 0.05
  - learning rate `1e-4`
  - batch size 4
  - grad accumulation 4
  - max seq len 1536
  - seed 42
- compute cost清楚：
  - Bayesian teacher eval: 1 x H100，约 30 min
  - 8B LoRA distillation: 1 x H100，约 1h55m
  - student eval: 1 x H100，约 1h39m

困难点：

- 如果不训练 student，只做 belief evaluator，成本较低。
- 若要严格复现 SFT 8B，需要下载模型、LoRA、数据 split 和脚本。
- 我们当前 ASTRA/CaSiNo framework 已在运行，但不是 BOND 的 exact Protocol 3 turn-level belief prediction。

### 如何和我们的 framework 对比

BOND 最适合评估我们的 **belief model**，不适合直接评估完整 buyer payoff。

推荐两个实验：

1. **Turn-level belief evaluation**
   - 输入 partial CaSiNo dialogue
   - 输出 six priority ordering posterior
   - metrics:
     - Brier score
     - MAP accuracy
     - Top-1
     - NDCG@3
     - entropy / calibration

2. **Belief-action coupling**
   - 用我们的 posterior 驱动 planner
   - 检查 action 是否和 posterior-induced menu optimum 一致
   - 对比 BOND 的 belief-policy decomposition

我们的 table 可以包含：

| Model | Brier | MAP Acc | Top-1 | NDCG@3 | Action-menu alignment |
|---|---:|---:|---:|---:|---:|
| BOND teacher | paper number |
| BOND student | paper number |
| structured-CoT | paper number |
| our belief prompt | rerun |
| our rich belief model | rerun |
| our full framework belief head | rerun |

### 优先级

**P1。**

它对我们的核心 claim 很重要：我们的 belief model 是否真的可解释、可校准、能驱动 action。但它不是最终 buyer-score benchmark，因此排在 TERMS-Bench / RLVR 后面。

## 推荐复现路线

### P0: 先做 TERMS-Bench-style 和 RLVR-style

这两个最适合支撑主论文：

1. **RLVR-style**
   - 固定 regulated seller
   - buyer variants:
     - direct prompt
     - CoT prompt
     - belief
     - planner+generator
     - full framework
     - gated full framework
   - table:
     - Reward
     - Deal Rate
     - Bargained Ratio
     - Price Overshoot Rate

2. **TERMS-Bench-style**
   - 固定 stochastic counterpart
   - 输出:
     - SE+
     - AGR+
     - CSE+
     - FAGR-
     - BE_type
     - CritViol%
     - utility
   - 重点展示 full framework 是否降低 belief error、提高 surplus、减少 violation。

### P1: 再做 Bilateral Trade 和 BOND

3. **Bilateral Trade**
   - 适合作为更接近 economics / private information 的 benchmark。
   - 输出 surplus share、IR、deal rate、NGFT walk-away、price-tier stability。
   - 可以和 paper 做 “paper-aligned reimplementation” 对比。

4. **BOND**
   - 用于独立证明 belief model。
   - 不一定先训练 student；可以先做 zero-shot / prompted rich belief posterior evaluator。
   - 输出 Brier、MAP、Top-1、NDCG@3。

### P2: 最后考虑 Preference Estimation

5. **Preference Estimation**
   - 多方、多议题、veto-holder 共识场景。
   - 工程量最大。
   - 更适合作为 appendix / future extension。

## 推荐论文写法

这些工作不能全部以“strict reproduction”名义比较。建议分三类表述：

### Strict / near-strict reproduction

适用于：

- BOND，如果代码和 split 能成功使用。
- TERMS-Bench，如果其 code / seeds / scenario configs 可获得。

用语：

```text
We evaluate our framework under the official benchmark protocol.
```

### Paper-aligned reimplementation

适用于：

- Bilateral Trade
- RLVR Negotiation
- TERMS-Bench if code unavailable

用语：

```text
Since official code/checkpoints are not publicly available, we implement a paper-aligned environment matching the task interface, reward, and metrics.
```

### Conceptual transfer / diagnostic extension

适用于：

- Preference Estimation

用语：

```text
We adapt the benchmark's preference-estimation objective to test whether our belief module generalizes to multi-party multi-issue negotiation.
```

## 最终建议

短期最值得做：

1. **RLVR-style buyer eval**：最快，和现有代码最贴近。
2. **TERMS-Bench-style eval**：最有说服力，能同时测 surplus、agreement、belief、violation。
3. **BOND belief eval**：证明我们的 rich belief model 不是 prompt artifact。

中期再补：

4. **Bilateral Trade paper-aligned round-robin / fixed-seller eval**

长期或 appendix：

5. **Preference Estimation multi-party scenario**
