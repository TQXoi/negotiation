# Opponent Belief-Centric Negotiation Agent: Novelty Ideas

## 背景判断

当前的 A+B+C framework：

- A: belief model
- B: planner / strategic action
- C: generator / naturalizer

确实有机会提升 negotiation performance，但如果只是把 belief、planner、generator 串起来，novelty 容易被认为是系统工程组合。更强的论文卖点应该放在：

> 如何把 opponent belief 做得更结构化、更可校准、更可迁移，并让它直接改善 planning 和 negotiation outcome。

因此，下面的 idea 都围绕一个核心目标：

**把 belief model 从“prompted summary”推进到“可评估、可更新、可用于决策的 opponent model”。**

不要求同时覆盖 multi-seller、multi-buyer、multi-product、multi-issue；只要在其中一个维度做出清楚突破，并能在 benchmark 上打出结果，就足够形成论文贡献。

## Idea 1: Belief State as Typed Latent Contract

### 核心想法

把 opponent belief 从自然语言 summary 改成一个 typed latent contract，也就是一组有明确语义、范围、置信度和证据的 latent variables。

例如 buyer 侧对 seller 建模：

```json
{
  "seller_reservation_range": {"low": 40, "high": 55, "confidence": 0.72},
  "likely_acceptable_price_range": {"low": 48, "high": 62, "confidence": 0.68},
  "seller_flexibility": {"distribution": {"low": 0.15, "mid": 0.60, "high": 0.25}},
  "seller_patience": {"distribution": {"short": 0.20, "medium": 0.55, "long": 0.25}},
  "seller_dominance": {"distribution": {"soft": 0.35, "neutral": 0.40, "hard": 0.25}},
  "seller_friendliness": {"distribution": {"cold": 0.20, "neutral": 0.45, "warm": 0.35}},
  "contract_term_preferences": {
    "delivery_time": {"prefers": "longer", "confidence": 0.61},
    "return_policy": {"prefers": "shorter", "confidence": 0.58}
  },
  "evidence": [
    {"turn": 2, "signal": "rejected 45 but countered 60", "supports": "reservation_range"},
    {"turn": 3, "signal": "used final-offer language", "supports": "low_patience"}
  ]
}
```

对 ASTRA/CaSiNo，可以换成：

```json
{
  "priority_posterior": {
    "food>water>firewood": 0.15,
    "food>firewood>water": 0.05,
    "water>food>firewood": 0.30,
    "water>firewood>food": 0.10,
    "firewood>food>water": 0.12,
    "firewood>water>food": 0.28
  },
  "fairness_preference": {"low": 0.10, "medium": 0.55, "high": 0.35},
  "concession_tendency": {"slow": 0.30, "medium": 0.50, "fast": 0.20},
  "walkaway_risk": {"next_2_rounds": 0.22}
}
```

### Novelty

已有工作通常做以下几类：

- ASTRA: fairness / stance / priority prediction + LP
- BOND: posterior over CaSiNo priority permutations
- RLVR: verifiable reward，不重点建模 opponent latent state
- TERMS-Bench: belief error metrics，但 agent 本身未必有 typed belief architecture

我们的 novelty 可以是：

> A typed opponent belief state that unifies economic reservation beliefs, behavioral traits, contract-term preferences, confidence, and evidence grounding, and is explicitly consumed by the planner.

这比简单 “LLM estimates opponent preference” 更具体，因为它要求：

1. 每个 belief variable 有类型和范围。
2. 每个变量有 confidence / distribution。
3. 每个变量必须绑定 evidence。
4. planner 只能基于 typed belief 做 action。

### 可行实验

#### AgenticPay

比较：

- direct prompt
- CoT prompt
- belief summary prompt
- typed belief prompt
- typed belief + planner
- typed belief + gated planner

指标：

- fixed-seller buyer score
- no-seller-error buyer score
- deals-only buyer score
- seller-error rate
- over-budget / invalid contract rate
- belief calibration proxy:
  - predicted acceptable range 是否覆盖最终成交价
  - predicted reservation range 是否低于 seller accepted price

#### ASTRA/CaSiNo

比较：

- ASTRA priority/fairness/stance belief
- typed belief with priority posterior + concession + walkaway risk
- typed belief + planner

指标：

- Avg Score All
- Avg Score Agreement
- Walk-Away
- partner priority prediction
- Brier score over priority posterior, if ground-truth priority available

#### TERMS-style

最适合：

- BE_type
- SE+
- AGR+
- CritViol%

### 预期结果

Typed belief 应该能：

- 降低 invalid / over-budget / poor contract
- 提高 no-seller-error buyer score
- 在 multi-issue / contract tasks 上明显优于 plain belief
- 在 price-only tasks 上需要 gated planner，否则可能不如 simple belief

### 风险

- 如果 typed belief 只是 LLM 自说自话，容易被 reviewer 质疑。
- 必须有 evidence 和 calibration metric。
- 需要保存 per-turn belief JSON，否则不可审计。

## Idea 2: Belief Update by Counterfactual Offer Probing

### 核心想法

不要只根据对话文本推断 opponent；让 planner 生成一组 counterfactual offers，然后 belief model 预测 opponent 对这些 offers 的 response distribution。

形式：

```json
{
  "candidate_offers": [
    {"price": 45, "terms": {...}},
    {"price": 50, "terms": {...}},
    {"price": 55, "terms": {...}}
  ],
  "predicted_response": [
    {"accept": 0.15, "counter": 0.55, "reject": 0.30},
    {"accept": 0.42, "counter": 0.43, "reject": 0.15},
    {"accept": 0.71, "counter": 0.23, "reject": 0.06}
  ],
  "expected_buyer_score": [...]
}
```

然后 planner 不是直接问“下一步该怎么说”，而是：

1. 生成 offer frontier。
2. belief model 对每个 offer 预测 accept/counter/reject。
3. 选 expected utility 最大，同时控制 walkaway risk 的 offer。

### Novelty

这更像把 opponent modeling 变成 **counterfactual response model**。

相比普通 belief：

- 普通 belief: “seller 大概强硬 / 预算是多少”
- counterfactual belief: “如果我出 48 并让步 delivery，seller 接受概率是多少”

相比 RLVR：

- RLVR 直接学 policy。
- 我们显式学习/估计 response model，再用它做 planning。

相比 ASTRA LP：

- ASTRA LP 给候选 offer，但没有明确给每个 offer 的 accept probability。
- 我们用 response distribution 对候选 offer 做 risk-aware selection。

### 可行实验

#### AgenticPay

候选 offer 可由规则或 planner 生成：

- price ladder: low / mid / high
- contract variants: favorable terms / balanced terms / seller-friendly terms

belief module 输出：

- accept probability
- counter probability
- reject / walkaway probability
- expected final price

planner 选择：

```text
argmax E[buyer_score] - lambda * walkaway_risk - invalid_penalty
```

#### RLVR-style

非常适合，因为 reward 可精确计算。

评估：

- Reward
- Bargained Ratio
- Deal Rate
- Price Overshoot
- predicted accept probability calibration

### 预期结果

这个 idea 有机会明显提升：

- price-only tasks
- fixed seller tasks
- RLVR reward

特别能解决我们之前发现的问题：

> full framework 在简单环境中会 target locking。

Counterfactual probing 可以让 planner 看见 `$170` 被拒绝概率高、`$175` 接受概率高，从而避免 21 轮复读。

### 风险

- 如果 LLM response probability 不校准，会误导 planner。
- 需要用实际 outcomes 做 calibration / reranking。
- 可能增加 API cost。

### 可增强方案

用历史 trajectories 做轻量 SFT / regression：

输入：

```text
dialogue history + candidate offer
```

输出：

```text
accept / counter / reject / walkaway
```

这可以成为一个很强的 “belief response model” contribution。

## Idea 3: Multi-Agent Market Belief for Multi-Seller / Multi-Buyer

### 核心想法

在 multi-seller 或 multi-buyer 场景中，对手不是一个 agent，而是一个 market。belief model 不只估计单个 seller reservation，而是估计 market-level latent state：

```json
{
  "seller_clusters": [
    {
      "cluster": "cheap_but_inflexible",
      "members": ["seller_2", "seller_4"],
      "reservation_range": [35, 45],
      "patience": "low",
      "accept_prob_curve": [...]
    },
    {
      "cluster": "expensive_but_flexible",
      "members": ["seller_1"],
      "reservation_range": [50, 65],
      "patience": "high",
      "accept_prob_curve": [...]
    }
  ],
  "outside_option_value": 47,
  "competition_pressure": 0.72,
  "best_counterparty": "seller_2",
  "negotiation_allocation": {
    "seller_2": "push hard",
    "seller_1": "keep warm",
    "seller_4": "final probe"
  }
}
```

对于 multi-buyer：

```json
{
  "buyer_competition_pressure": 0.81,
  "rival_buyer_budget_range": [60, 75],
  "seller_likely_to_wait_for_other_buyers": 0.64
}
```

### Novelty

大多数现有工作是 single buyer / single seller 或 multi-party consensus，但很少把 opponent belief 扩展为 **market belief**。

我们的 novelty：

> A market-level opponent belief model that estimates heterogeneous counterparty clusters, outside-option value, and competition pressure, then allocates negotiation strategy across counterparties.

这可以把我们的工作从单一谈判 agent 推进到 market negotiation agent。

### 可行实验

AgenticPay 已经有 multi-buyer / multi-seller / multi-product examples。可以优先选一个维度：

#### Multi-seller buyer

固定多个 seller，buyer 选择与谁成交。

指标：

- buyer score
- selected seller optimality
- regret vs best feasible seller
- number of rounds spent per seller
- deal rate
- over-budget / invalid rate

Baselines：

- direct prompt
- independent belief per seller
- market belief clustering
- market belief + planner

#### Multi-buyer seller

我们作为 seller，对多个 buyer 建模。

指标：

- seller profit
- selected buyer optimality
- lost deal rate
- underpricing rate

### 预期结果

Market belief 应该在 multi-seller 场景下特别强：

- 更少被第一个 seller 锁住
- 更好利用 outside option 压价
- 更高 selected seller optimality
- 更低 regret

### 风险

- 多方环境更复杂，现有 LLM 容易格式错误。
- 需要固定其他 agents，避免方差太大。
- 需要设计 clean metric，如 regret vs oracle best seller。

## Idea 4: Belief-to-Plan Consistency Training

### 核心想法

不是只训练 agent 生成最终动作，而是训练它生成：

```text
belief -> plan -> action
```

并加入 consistency constraints：

1. plan 必须引用 belief 中的变量。
2. action 必须执行 plan。
3. 如果 outcome 与 belief 预测冲突，下一轮 belief 必须更新。

可以构造监督数据：

```json
{
  "history": ...,
  "belief": ...,
  "plan": ...,
  "action": ...,
  "outcome": ...,
  "belief_update": ...
}
```

### Novelty

现有 SFT/RL 多半优化 final action 或 reward。我们的 novelty 是：

> Train negotiation agents on auditable belief-plan-action traces, using consistency losses between latent beliefs, strategic plans, and realized actions.

这和 BOND 接近，但 BOND 主要 distill Bayesian belief states into LMs；我们进一步要求 belief 驱动 plan/action，并用 negotiation reward 评估。

### 可行训练方式

#### SFT

从 existing trajectories 生成 pseudo labels：

- belief: typed belief JSON
- plan: target price / concession schedule / risk control
- action: final message

训练 Qwen-14B 或更小模型 LoRA。

#### RL / preference optimization

Reward 组合：

```text
R = outcome_reward
  + alpha * belief_calibration_reward
  + beta * plan_action_consistency_reward
  - gamma * violation_penalty
```

其中：

- outcome_reward: buyer score / RLVR reward
- belief_calibration_reward: predicted accept prob vs actual outcome
- consistency_reward: action 是否实现 plan
- violation_penalty: over-budget / invalid format / bad contract

### 可行实验

最适合 RLVR-style / TERMS-style：

- outcome 可验证
- accept / reject 可验证
- violation 可验证
- belief calibration 可验证

比较：

- SFT on final action only
- SFT on belief-plan-action trace
- RL on reward only
- RL with belief-consistency auxiliary reward

### 预期结果

可能提升：

- reward
- lower overshoot
- lower hallucinated belief
- better calibration
- less target locking

### 风险

- 数据生成成本高。
- 如果 pseudo belief labels 质量差，会放大错误。
- 需要先有足够稳定的 typed belief extractor。

## Idea 5: Uncertainty-Aware Negotiation Policy

### 核心想法

让 belief model 明确输出不确定性，并让 planner 根据不确定性改变策略。

例子：

- high confidence seller reservation low -> aggressive offer
- low confidence -> information-gathering probe
- high walkaway uncertainty -> safer concession
- multi-issue uncertainty high -> ask targeted question

Policy 不再只是 “maximize expected score”，而是：

```text
maximize expected utility + value of information - risk penalty
```

### Novelty

很多 negotiation agent 只输出一个 point estimate 或自然语言判断。我们的 novelty 是：

> An uncertainty-aware belief-to-action policy that chooses between probing, conceding, anchoring, and closing based on posterior entropy and value of information.

这在 multi-issue / hidden preference 场景尤其自然。

### 可行实现

每轮计算：

```json
{
  "reservation_entropy": 0.62,
  "priority_entropy": 1.21,
  "term_preference_entropy": 0.74,
  "walkaway_risk_uncertainty": 0.33
}
```

Planner action set：

- ask preference question
- probe low offer
- propose balanced contract
- close gap
- walk away

Action selection：

```text
if entropy high and enough rounds left:
    ask/probe
elif accept_prob high:
    close
elif walkaway risk high:
    concession
else:
    anchor
```

### 可行实验

ASTRA/CaSiNo：

- priority posterior entropy
- question-asking effectiveness
- partner priority prediction
- Avg Score Agreement

TERMS-style：

- BE_type
- SE+
- AGR+

AgenticPay：

- fewer repeated offers
- better deal-only score
- better no-seller-error score

### 预期结果

相比普通 full framework：

- 更少无效追问
- 更少过早 close
- 更少 target locking
- 在 early rounds 更愿意收集信息

### 风险

- LLM 输出的 confidence 可能不校准。
- 需要校准 confidence：
  - temperature scaling
  - empirical calibration
  - self-consistency sampling

## Idea 6: Social-Behavioral Opponent Model

### 核心想法

不仅估计经济变量，还估计对手的 social / behavioral latent traits：

- dominance
- friendliness
- fairness norm
- concession style
- patience
- threat sensitivity
- reciprocity
- preference for explanation / evidence

这些变量不是为了好看，而是直接影响 planner：

```json
{
  "dominance": "high",
  "fairness_norm": "medium",
  "reciprocity": "low",
  "threat_sensitivity": "negative",
  "recommended_tone": "firm but respectful",
  "recommended_strategy": "avoid apology-heavy concession; use market evidence"
}
```

### Novelty

ASTRA 已有 fairness / stance，但比较窄。我们可以把它扩展成：

> A social-behavioral opponent model that conditions concession schedule and language strategy on inferred negotiation persona.

这能连接 BOND 的 belief auditability 和 RLVR 的 adversarial seller personas。

### 可行实验

RLVR adversarial sellers：

- begging
- insulting
- unyielding

比较：

- no social belief
- social belief prompt
- social belief + planner

指标：

- reward
- deal rate
- overshoot rate
- concession-to-pressure rate
- robustness under adversarial persona

AgenticPay：

- seller friendliness / dominance 可从 dialogue 中估计
- 看是否减少被 seller pressure 诱导过早让步

### 预期结果

这个方向可能最容易在 adversarial seller setting 打出故事：

- 普通 buyer 被 begging / insulting 影响
- social belief buyer 识别 persona，并保持 price discipline

### 风险

- behavioral traits 很主观，必须转成可验证 proxy。
- 需要避免变成泛泛的 “tone adaptation”。

## Idea 7: Opponent Model as Differentiable Memory

### 核心想法

把 belief model 从每轮 prompt 估计，改成可持续更新的 memory：

```json
{
  "long_term_counterparty_profile": ...,
  "episode_belief_state": ...,
  "evidence_store": ...,
  "contradictions": ...,
  "belief_revision_history": ...
}
```

每轮做：

1. retrieve relevant evidence
2. update posterior
3. detect contradiction
4. revise belief
5. plan

### Novelty

相比一次性 prompt belief，memory 让 belief 可追踪、可回滚、可审计。

> A persistent evidence-grounded opponent memory that supports belief revision and contradiction handling across turns and counterparties.

### 可行实验

多轮谈判里最有用：

- ASTRA/CaSiNo
- AgenticPay contract tasks
- multi-seller environment

Metrics：

- belief revision accuracy
- contradiction recovery
- repeated-offer reduction
- score / agreement

### 风险

- 工程复杂度中等。
- 如果没有 long horizon，优势不明显。

## 最推荐的三条论文主线

### 主线 A: Counterfactual Opponent Response Model

一句话：

> We model the opponent not only as latent preferences, but as a counterfactual response distribution over candidate offers.

优势：

- novelty 清楚
- 直接帮助 planning
- 可在 RLVR / AgenticPay price tasks 打结果
- 可解决 target locking

推荐作为主贡献。

### 主线 B: Typed Evidence-Grounded Belief State

一句话：

> We introduce a typed, evidence-grounded belief state that unifies reservation ranges, behavioral traits, term preferences, confidence, and posterior distributions.

优势：

- 解释性强
- 可适配 price / contract / CaSiNo
- 和 BOND / TERMS-Bench 对话自然

推荐作为框架核心。

### 主线 C: Market-Level Belief for Multi-Seller Negotiation

一句话：

> We extend opponent modeling from a single counterparty to market-level beliefs over heterogeneous sellers and outside options.

优势：

- novelty 更大
- 和 multi-seller / multi-product 环境贴合
- 结果如果打出来会很亮

风险：

- 工程更复杂
- baseline 设计要谨慎

## 推荐最终方法组合

最稳的论文方法可以命名为：

**CORTEX-Negotiator: Counterfactual Opponent Response and Typed Evidence-grounded eXpectation for Negotiation**

或者更简单：

**COPE: Counterfactual Opponent Preference Estimation for Negotiation**

方法包括：

1. **Typed Belief Extractor**
   - 生成 reservation / preference / behavioral / term belief
   - 每项带 confidence 和 evidence

2. **Counterfactual Response Estimator**
   - 对候选 offers 预测 accept / counter / reject / walkaway
   - 输出 calibration-aware probability

3. **Risk-Aware Planner**
   - 选择 expected utility 最大的 offer
   - 加入 walkaway risk、violation penalty、information gain

4. **Naturalizer**
   - 将 selected action 转成合法 negotiation message
   - 不负责改变策略，只负责表达

5. **Optional SFT/RL**
   - SFT belief-plan-action trace
   - RLVR reward + belief calibration auxiliary reward

## 推荐实验组合

### Experiment 1: AgenticPay Single28 Fixed Seller

目的：

- 延续当前实验
- 验证 typed/counterfactual belief 是否改善 buyer score

Variants：

- direct prompt
- CoT prompt
- belief prompt
- typed belief
- counterfactual belief
- full A+B+C
- COPE / CORTEX

Metrics：

- fixed-seller buyer score
- no-seller-error buyer score
- deals-only buyer score
- deal rate
- invalid / over-budget / seller-error rates

### Experiment 2: RLVR-style Buyer Eval

目的：

- 用 verifiable reward 证明 buyer-side surplus extraction

Metrics：

- Reward
- Deal Rate
- Bargained Ratio
- Price Overshoot
- accept-probability calibration

### Experiment 3: TERMS-style Diagnostic Eval

目的：

- 用 diagnostic benchmark 证明 belief model 真有用

Metrics：

- SE+
- AGR+
- CSE+
- FAGR-
- BE_type
- CritViol%

### Experiment 4: ASTRA/CaSiNo Belief Eval

目的：

- 对接 ASTRA / BOND
- 证明 priority posterior 更准

Metrics：

- Avg Score All
- Avg Score Agreement
- Walk-Away
- priority prediction accuracy
- Brier score over priority posterior

### Experiment 5: Multi-Seller AgenticPay

目的：

- 展示 market-level belief novelty

Metrics：

- buyer score
- selected seller optimality
- regret vs oracle best feasible seller
- time allocation across sellers
- deal rate

## 论文贡献建议写法

可以把 contribution 写成：

1. **Typed and evidence-grounded opponent belief**
   - We propose a typed belief representation that jointly models reservation range, response likelihood, behavioral traits, term preferences, and uncertainty with evidence.

2. **Counterfactual response modeling for negotiation planning**
   - Instead of planning from a single opponent summary, our agent evaluates candidate offers through a learned/prompted response distribution.

3. **Risk-aware belief-to-plan negotiation**
   - The planner optimizes expected buyer utility under accept probability, walkaway risk, and constraint violation risk.

4. **Comprehensive evaluation across price, contract, and resource-allocation negotiation**
   - AgenticPay, RLVR-style, TERMS-style, ASTRA/CaSiNo.

## 最小可行版本

如果时间有限，建议先做：

1. Typed belief JSON + evidence logging
2. Candidate offer generator
3. LLM counterfactual response estimator
4. Risk-aware selection
5. AgenticPay Single28 + RLVR-style eval

这已经足够形成一个比 A+B+C 更有 novelty 的方法：

> Our key contribution is not modular decomposition, but counterfactual, evidence-grounded opponent belief that directly controls negotiation actions.

Problem: belief model 怎么优化，planner 如何只是基于 prompt 的 LLM，都无法充分利用，可能的解决方法：
# Beyond Belief Representation: How Should a Planner Exploit Opponent Beliefs?

## Motivation

随着 Opponent Belief Model 越来越复杂，一个新的问题逐渐出现：

> belief 已经越来越丰富，但 planner 仍然只是一个 Prompt LLM。

即：

```text
Belief
    ↓
Prompt LLM
    ↓
Action
```

这实际上意味着：

* belief 越复杂，LLM 越容易忽略其中的重要信息；
* planner 无法显式利用 uncertainty、posterior 或 evidence；
* belief 的提升无法稳定转化为策略提升。

因此，我认为下一步研究重点不应该只是：

> **How to represent belief?**

而应该进一步研究：

> **How should a planner consume and exploit belief?**

---

# Idea 1: Belief-conditioned Value Network（★★★★★）

## 核心思想

belief 不再直接输入 Prompt，而是进入一个 Value Network。

整体结构：

```text
Belief
    ↓
Value Network
    ↓
Candidate Action Ranking
    ↓
Naturalizer
```

例如：

belief：

```python
seller_floor=[80,90]

patience=low

truthfulness=0.3
```

Planner 先生成多个 candidate：

```python
offer=82

offer=85

offer=90
```

Value Network：

```python
Q(82)=1.2

Q(85)=3.5

Q(90)=2.9
```

最终：

```text
argmax Q
```

选择：

```text
offer=85
```

## Novelty

不是：

> Better Planner

而是：

> **Belief-conditioned Value Function**

belief 真正控制 decision，而不是作为 prompt context。

---

# Idea 2: Belief as an API

## Motivation

目前 belief 一次性输入 planner：

```text
belief
↓

LLM
```

实际上 planner 未必需要全部 belief。

更好的方式：

Planner 主动查询 belief。

例如：

```text
Need:

seller floor

↓

Belief Module

↓

80~90
```

下一步：

```text
Need:

walkaway probability

↓

Belief Module

↓

0.32
```

belief 更像：

```text
Knowledge API
```

而不是：

```text
Prompt Context
```

## Novelty

> Dynamic Belief Query

而不是：

Static Belief Representation。

---

# Idea 3: Belief-guided Search

Planner 本身可以保持不变。

真正改变的是：

Search。

例如：

belief：

```text
seller floor≈90
```

Tree Search：

直接：

```text
Prune

offer=60
```

这一整条分支。

Belief：

控制：

```text
Search Prior
```

而不是：

LLM Prompt。

这一思路与近年来 belief-space planning 中：

> belief 用于 search，而不是直接生成 action

非常一致。

---

# Idea 4: Graph-based Negotiation World Model

belief 不再是：

```text
Opponent
```

而变成：

```text
Seller1

Seller2

Seller3

↓

Products

↓

Relations
```

例如：

```text
Seller1

↓

cheaper

↓

Seller2

↓

more flexible

↓

Seller3
```

Planner：

进行：

```text
Graph Search
```

而不是：

Language Prompt。

这比：

Multi-Seller Belief

更进一步：

belief 已经成为：

```text
Negotiation World Model
```

---

# Idea 5: Opponent Policy Modeling

目前：

belief：

预测：

```text
Current State
```

例如：

* reservation
* patience
* priority

其实更重要的是：

预测：

```text
Future Strategy
```

例如：

```python
{

accept_prob,

counter_prob,

walkaway_prob,

concede_prob

}
```

Planner：

进行：

Lookahead。

Novelty：

belief 不再表示：

Opponent State，

而表示：

Opponent Policy。

这一方向与认知科学中的：

Inverse Planning

具有天然联系。

---

# Idea 6: Prediction-error Belief Revision

belief：

不仅保存状态。

更持续预测：

```text
Next Observation
```

例如：

belief：

```text
seller

↓

will counter at 180
```

实际：

```text
seller

↓

accept
```

Prediction Error：

巨大。

于是：

```text
Belief Revision
```

立即发生。

belief：

成为：

```text
Predictive Model
```

而不是：

Static Memory。

这一方向非常适合结合：

Bayesian Update

或

Learned Belief Revision。

---

# Idea 7: BDI-style Planner

Planner：

不再直接由 LLM 完成。

而采用：

```text
Belief

↓

Desire

↓

Intention

↓

Naturalizer
```

例如：

```text
Belief

↓

Need cheaper seller

↓

Goal:

maximize buyer utility

↓

Plan:

explore seller2
```

LLM：

只负责：

Naturalizer。

这一方向对应经典：

BDI Agent Architecture。

---

# Idea 8: Active Belief / Information Gathering（★★★★★）

这是目前我最喜欢的方向之一。

目前：

belief：

都是：

```text
Passive Estimation
```

即：

根据已有对话推断 opponent。

其实：

Planner：

很多时候应该主动收集信息。

例如：

belief：

```text
不知道：

seller

delivery

是不是 flexible
```

Planner：

不是：

```text
Offer
```

而是：

```text
Question
```

例如：

```text
Would longer delivery help?
```

因为：

Information Gain：

最大。

整个 Planner：

优化：

```text
Expected Utility

+

Information Gain
```

而不是：

只有：

Expected Utility。

这实际上对应：

Belief-space Planning

中的：

Value of Information。

目前 LLM Negotiation 工作几乎没有真正利用这一思想。

---

# 当前最推荐的 Top 5

| Rank  | Idea                                  | Novelty                                     |
| ----- | ------------------------------------- | ------------------------------------------- |
| ⭐⭐⭐⭐⭐ | Belief-conditioned Value Network      | belief 真正控制 decision，而不是 prompt             |
| ⭐⭐⭐⭐⭐ | Belief API / Query Planner            | planner 主动查询 belief，而不是一次性读完                |
| ⭐⭐⭐⭐☆ | Opponent Policy Modeling              | belief 预测 future strategy，而不是 current state |
| ⭐⭐⭐⭐☆ | Prediction-error Belief Revision      | belief 因 prediction error 而持续演化             |
| ⭐⭐⭐⭐☆ | Active Belief / Information Gathering | planner 主动探索 belief，而不是被动推断                 |

---

# 总结

目前真正的问题已经不是：

> belief 不够丰富。

而是：

```text
Belief

↓

Planner

之间

没有真正的 Interface。
```

下一篇论文真正值得回答的问题不是：

> **How to represent belief?**

而是：

> **How should a planner consume and exploit belief?**

相比继续增加更多 latent variables，这个问题更加本质，也更容易形成区别于现有 BOND、ASTRA、Preference Estimation 等工作的核心 novelty。

未来如果 Opponent Belief Model 成为论文主体，那么 Planner 可以退化成：

> 一个 belief consumer。

Naturalizer：

则完全退化成：

> 一个 language realization module。

这样整个论文故事就会从：

```text
Belief

↓

Planner

↓

Generator
```

转变成：

```text
Opponent Belief

↓

Belief Interface

↓

Decision Module

↓

Language
```

真正的创新点也会集中到：

> **如何表示 belief、更新 belief，以及如何让 belief 真正影响决策。**

问题：使用什么的环境，multiple buyer multiple seller multiple product 吗？

我其实认真想了这个问题，而且我的答案和一开始相比已经发生了变化。

如果你的目标只是做一个更好的 negotiation agent，那么 AgenticPay 是最好的环境。

但如果你的目标是发一篇“Opponent Belief Modeling”论文，那么我反而不建议把 AgenticPay 作为唯一主战场。

原因如下。

我认为 reviewer 最关心的问题

Reviewer 不会首先问：

"你的环境复杂吗？"

他首先问的是：

"你的贡献是不是因为环境特殊？"

假设你的论文标题是：

Opponent Belief Modeling for Multi-Seller Negotiation

然后所有实验都在 AgenticPay。

Reviewer 第一反应很可能是：

这是 AgenticPay 特有的问题吗？

我现在更倾向于“三层实验”

而不是：

AgenticPay

↓

结束
第一层
Controlled Environment

例如：

Bilateral Trade
RLVR Negotiation
TERMS-Bench

这些环境：

特点：

变量少

ground truth 清楚

容易分析

例如：

Bilateral Trade：

Buyer

↓

Seller

只有：

price

reservation

这里特别适合：

验证：

belief

↓

planning

是不是有效。

第二层
Structured Negotiation

例如：

ASTRA

或者：

CaSiNo。

特点：

multi issue

但是：

仍然：

single opponent

这里：

验证：

factorized belief

↓

issue preference

是不是：

比：

BOND：

更好。

第三层
Complex Market

最后：

AgenticPay。

特点：

multiple buyer

multiple seller

multiple product

这是：

真正：

证明：

world model

价值的地方。

为什么我现在不建议一开始就在 AgenticPay 做所有东西

其实：

你现在：

已经碰到了。

例如：

trajectory：

里面：

seller1

↓

seller2

↓

product

↓

contract

全部：

耦合。

如果：

belief：

出了问题。

你不知道：

到底：

seller

product

planner

generator

谁的问题。

Reviewer：

也不知道。

所以：

论文：

不好讲。

我更推荐

论文：

应该：

这样：

讲。

第一部分

我们：

提出：

Belief

在：

Controlled：

环境：

证明：

belief

↓

better

例如：

TERMS：

belief calibration。

第二部分

放：

ASTRA。

证明：

belief

↓

planning

有效。

第三部分

最后：

AgenticPay。

证明：

belief

↓

scale

到：

multi seller

multi product

Reviewer：

一下：

就懂：

为什么：

值得。

AgenticPay 的定位

其实：

我现在觉得：

AgenticPay：

不应该：

是：

Main Benchmark。

应该：

是：

Stress Test。

例如：

论文：

最后：

一节：

Generalization

↓

AgenticPay

例如：

Reviewer：

会觉得：

OK

前面：

已经证明：

belief

是真的。


然后：

AgenticPay

↓

证明：

复杂环境

也成立。
如果是我

我甚至：

会：

故意：

这样：

安排：

Figure 1
TERMS

↓

Belief Accuracy
Figure 2
ASTRA

↓

Planner Benefit
Figure 3
AgenticPay

↓

Scaling

整个：

Story：

特别自然。

如果你的目标是 Opponent Belief

我甚至：

会：

弱化：

AgenticPay。

例如：

论文：

Title：

不是：

Negotiation Framework

而是：

Opponent Belief Modeling

实验：

就是：

Can we infer?

↓

Can we calibrate?

↓

Can we revise?

↓

Can we use?

↓

Can we scale?

最后：

AgenticPay

↓

Yes.
所以我目前最推荐的实验路线
环境	目的	为什么
TERMS-Bench	验证 belief	evaluator 有 hidden state，可以直接测 belief calibration，而不是只看 deal rate。
CaSiNo / ASTRA	验证 factorized belief	ground truth issue preference，非常适合 belief 本身的研究。
Training Language Models for Bilateral Trade with Private Information	验证 planner interface	单 buyer/seller，可以干净验证 belief 如何帮助决策。
AgenticPay	验证复杂场景泛化	multi-seller、multi-product，证明方法可以扩展到真实市场，而不是只在简单 setting 生效。
我还有一个新的想法（也是我目前最喜欢的）

其实你的论文可以不把 AgenticPay 当 benchmark，而把它当 "evidence"。

也就是说：

前面的环境回答：

Opponent belief 是否正确？

AgenticPay 回答：

当 negotiation world 足够复杂（多 seller、多 product、多 issue）时，没有好的 opponent/world belief，现有 agent 会出现 seller routing collapse、重复策略、探索不足等现象；而我们的 belief model 能缓解这些问题。

这样 AgenticPay 就从"刷分环境"变成了"证明研究意义的复杂案例"，说服力反而更强。