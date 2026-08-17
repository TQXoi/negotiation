# 项目展望 v0.2：从 Belief Model 到 Belief-Usable Planner

更新时间：2026-07-29

参考：

- `Counterparty Modeling is Not Strategy: Agents Fail in Multi-Issue Negotiations Despite Accurate Forecasting`，arXiv:2605.16575。
- CaSiNo dataset：A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems。
- ASTRA：Adaptive Negotiation Agent。
- AgenticPay 本地代码：`/work5/qixint/benchmarks/AgenticPay`。

## 1. 新项目故事线

我认为你提出的方向是对的：既然 2605.16575 的核心启发是 **multi-issue negotiation 中“知道对方偏好”不等于“会谈判”**，那么我们的项目故事线应该从 AgenticPay-only 改成：

> 我们研究如何让 LLM negotiation agent 的 planner 更好地使用 belief model。核心不是单独提升 opponent preference prediction，而是让 planner 把 belief 转换成多 issue、多对手、多轮对话中的有效 offer、tradeoff、concession 和 persuasive language。

这条故事线可以分成两个 benchmark 层级：

1. **CaSiNo / ASTRA-style multi-issue preference benchmark**
   - 更干净、更标准、更容易和已有 baseline 对比。
   - 双方围绕固定多个 issue 谈判。
   - 每个参与者有私有 preference ordering。
   - 非常适合研究 belief model 是否帮助 planner 做 logrolling / issue tradeoff。

2. **AgenticPay universal commerce benchmark**
   - 更复杂、更贴近真实 commerce。
   - 包含 multi buyer、multi seller、multi product、multi issue contract。
   - 适合展示方法的通用性和扩展性。

换句话说：

```text
Simple Env:
  学习 price concession / planner timing

CaSiNo / ASTRA:
  证明 belief -> planner -> multi-issue tradeoff

AgenticPay:
  扩展到 universal commerce: multi-agent + multi-product + multi-issue
```

## 2. 2605.16575 对我们的核心启发

2605.16575 最重要的启发是：

- LLM agent 可以在一定程度上预测对方偏好。
- 但准确的 counterparty modeling 并不自动带来好策略。
- multi-issue negotiation 的难点是：
  - 哪些 issue 对我重要？
  - 哪些 issue 对对方重要？
  - 哪些 issue 可以让步？
  - 如何用我低成本的让步换取对方高价值让步？
  - 如何把 preference belief 转化成可执行 offer？

这正好对应我们的项目目标：

> belief model 的价值不在于单独输出一个 posterior，而在于让 planner 能提出更高 expected utility 的 candidate offer。

因此，我们的实验不应只报告 belief accuracy，也应该报告：

- belief-conditioned offer quality
- planner candidate selection quality
- Pareto efficiency / joint utility
- own utility
- agreement rate
- walk-away rate
- issue tradeoff quality

## 3. 为什么 CaSiNo / ASTRA 应加入主线

CaSiNo 是一个 campsite negotiation dataset，两名参与者围绕：

```text
Food
Water
Firewood
```

进行分配谈判。每个参与者有自己的私有偏好和需求。公开资料显示，CaSiNo 包含约 1030 个 negotiation dialogues，并记录：

- preference order
- dialogue
- final outcome
- participant points
- satisfaction
- opponent likeness
- strategy annotations

ASTRA 使用 CaSiNo-style 环境做 agent-to-agent simulation。Table metrics 包括：

- Avg. Score (All), P1 vs P2
- Avg. Score (Agreement), P1 vs P2
- T-statistic on agreement scores
- Walk-Away (%)

它还比较了：

- ASTRA / Ours
- Ours w/o ASTRA
- RL Agent
- ICL-AIF
- Pro-CoT
- Partner-Base / Greedy / Fair

这对我们的好处是：

1. **它是天然 multi-issue preference negotiation。**
   - Food/Water/Firewood 都是 issue。
   - 双方偏好不同。
   - planner 必须用 belief 做 issue tradeoff。

2. **它比 AgenticPay 更干净。**
   - issue 数少。
   - utility/score 更容易解释。
   - 轨迹更容易分析。

3. **它能和已有 baseline 对齐。**
   - ASTRA 原文已经给了 Table 1。
   - 我们已经有 ASTRA 代码适配基础。
   - 可以在相同 partner setting 下跑 our framework。

4. **它更适合证明 novelty。**
   - 如果我们的 belief model 真的能推断对方偏好，planner 应该能提出：
     - “我让出 Food，你让出 Water”
     - “我保留高价值 issue，交换低价值 issue”
   - 这比 Simple Env 中“报价高低”更能展示 belief 的作用。

## 4. AgenticPay 中是否有类似 task？

有，但形式更复杂。

AgenticPay 的 contract-mode task 也支持 multi-issue。其 offer 格式是：

```json
{
  "price": 120,
  "continuous_terms": {},
  "discrete_terms": {}
}
```

双方 utility：

```text
U_b = v_base - price
      + Σ buyer_continuous_weight[t] * value[t]
      + Σ buyer_discrete_weight[t][option]

U_s = price - c_base
      + Σ seller_continuous_weight[t] * value[t]
      + Σ seller_discrete_weight[t][option]
```

这说明 AgenticPay 也有：

- price issue
- continuous issue
- discrete issue
- buyer private preference
- seller private preference

但 AgenticPay 的 contract-mode 更接近 realistic commerce：

- task domain 多样。
- issue schema 不固定。
- 可能还叠加 multi buyer / multi seller / multi product。
- parser/validator 更复杂。

因此，AgenticPay 可以作为第二阶段或第三阶段 benchmark，而不是第一个证明 belief-planner 关系的主 benchmark。

## 5. 推荐 Benchmark Story

### Stage 0: Simple Env

目的：

- 快速调 planner。
- 验证 anchor/concession schedule。
- 收集 SFT/RFT data。

主要问题：

- belief 作用小。
- 单 issue price negotiation 不足以证明 opponent modeling novelty。

### Stage 1: CaSiNo / ASTRA

目的：

- 证明 belief model 能帮助 planner 做 multi-issue tradeoff。
- 与 ASTRA、RL Agent、ICL-AIF、Pro-CoT 等 baseline 对比。

核心实验：

```text
Fixed partner / seller side:
  Partner-Base
  Partner-Greedy
  Partner-Fair
  Pro-CoT partner

Buyer / P1 side:
  Direct Prompt
  CoT Prompt
  Belief-only
  Planner-only
  Full Framework
  Belief-scored Candidate Planner
```

指标：

- Avg. Score (All)
- Avg. Score (Agreement)
- Walk-Away (%)
- Agreement rate
- P1-P2 score gap
- T-statistic on agreement score difference
- issue-tradeoff quality

### Stage 2: AgenticPay Contract-Mode

目的：

- 扩展到 commerce-style multi-issue。
- 测试 schema-varied contract negotiation。
- 检验 framework 是否能泛化到 price + continuous/discrete issue。

核心问题：

- belief model 能否推断 seller issue preference？
- planner 能否用低成本 non-price terms 换取价格或高价值 terms？
- candidate scorer 是否能提高 contract utility？

### Stage 3: AgenticPay Multi-Agent Universal

目的：

- 扩展到 multi buyer / multi seller / multi product。
- 证明 framework 不只是 CaSiNo 特化。

核心问题：

- multi seller：如何选择对手、施加比较压力？
- multi buyer：如何在竞争中平衡 surplus 与 win probability？
- multi product：如何选择 product/bundle？
- multi issue：如何做 issue-level logrolling？

## 6. CaSiNo / ASTRA 中 Belief Model 怎么设计

CaSiNo 的 belief state 应该更明确地围绕 issue preference：

```json
{
  "opponent_issue_preference_posterior": {
    "Food": {
      "rank_prob": {"high": 0.2, "medium": 0.5, "low": 0.3},
      "need_strength": 0.6,
      "evidence": []
    },
    "Water": {
      "rank_prob": {"high": 0.7, "medium": 0.2, "low": 0.1},
      "need_strength": 0.8,
      "evidence": []
    },
    "Firewood": {
      "rank_prob": {"high": 0.1, "medium": 0.3, "low": 0.6},
      "need_strength": 0.4,
      "evidence": []
    }
  },
  "opponent_concession_pattern": {
    "Food": "holds/softens/concedes",
    "Water": "holds/softens/concedes",
    "Firewood": "holds/softens/concedes"
  },
  "opponent_flexibility": 0.5,
  "opponent_patience": 0.5,
  "deal_risk": 0.3
}
```

更新 evidence 包括：

- 对方主动索要某 item。
- 对方拒绝让出某 item。
- 对方愿意交换某 item。
- 对方话语中的 need/urgency。
- 对方连续 concession 的方向。

Bayesian / frequency update 可以很自然：

- 如果对方反复坚持 Water，则 `P(Water=high)` 上升。
- 如果对方主动让出 Firewood，则 `P(Firewood=low)` 上升。
- 如果对方用强烈理由描述 Food，则 Food need strength 上升。

## 7. CaSiNo / ASTRA 中 Planner 怎么使用 Belief

Planner 应该生成 candidate allocations，而不是只生成一句话。

例如候选：

```json
[
  {
    "offer": {"Food": 1, "Water": 3, "Firewood": 2},
    "self_utility": 22,
    "estimated_opponent_utility": 18,
    "accept_prob": 0.55,
    "tradeoff": "keep own high-value Water, concede Firewood"
  },
  {
    "offer": {"Food": 2, "Water": 2, "Firewood": 2},
    "self_utility": 20,
    "estimated_opponent_utility": 20,
    "accept_prob": 0.75,
    "tradeoff": "balanced split"
  }
]
```

Belief model 为每个 candidate 打分：

```text
ExpectedValue =
  self_utility
  * accept_prob
  - walkaway_risk_penalty
  + future_negotiation_value
```

Planner 再选择：

- early round：可选低 accept_prob 但高 self utility 的 anchor。
- middle round：选择 logrolling candidate。
- final round：选择高 accept_prob 且保持 self utility 的 offer。


这正好把 2605.16575 的启发落地：

```text
opponent modeling is not strategy
-> belief must be connected to candidate action evaluation and tradeoff planning
```

## 8. AgenticPay 中对应的 Universal Belief

AgenticPay 的 belief state 应是 CaSiNo 的泛化版：

```json
{
  "per_seller": {
    "seller_1": {
      "reservation_price_posterior": {},
      "issue_preference_posterior": {},
      "acceptance_model": {},
      "flexibility": 0.5,
      "patience": 0.5
    }
  },
  "per_buyer_competitor": {
    "buyer_2": {
      "budget_posterior": {},
      "likely_next_offer": {},
      "aggressiveness": 0.5
    }
  },
  "per_product": {
    "product_1": {
      "seller_cost_uncertainty": {},
      "substitutability": {}
    }
  },
  "per_issue": {
    "delivery_time": {
      "seller_weight_posterior": {},
      "buyer_cost_of_concession": {}
    }
  }
}
```

AgenticPay planner 的 candidate action 需要包括：

- selected seller
- selected product / bundle
- price
- continuous terms
- discrete terms
- concession size
- language strategy

belief scorer 需要估计：

- seller accept probability
- seller counteroffer distribution
- competitor win probability
- own utility
- joint utility / global score proxy

## 9. 哪个 Benchmark 更适合现在优先做？

我建议优先级是：

### Priority 1: CaSiNo / ASTRA

原因：

- 更像 2605.16575 的 multi-issue preference negotiation。
- 更容易展示 belief model novelty。
- baseline 更直接：
  - ASTRA
  - Pro-CoT
  - ICL-AIF
  - RL Agent
  - prompt buyer
  - belief-only / planner-only / full framework
- 结果表更容易讲。

### Priority 2: AgenticPay Contract-Mode Subset

原因：

- AgenticPay 中也有 multi-issue。
- 但 schema 更复杂，跑全部成本高。
- 可以选 contract-mode subset 作为“泛化验证”。

### Priority 3: AgenticPay Full All Tasks

原因：

- 证明 universal。
- 但 full all-task 结果噪声大，debug 成本高。
- 更适合在方法稳定后跑。

## 10. 新实验计划

### 10.1 CaSiNo / ASTRA Main Table

实验设置：

```text
Fixed partner:
  Partner-Base
  Partner-Greedy
  Partner-Fair
  Partner-ProCoT

Our side:
  Direct Prompt
  CoT Prompt
  Belief Only
  Planner Only
  Full Framework
  Belief-Scored Candidate Planner
```

核心指标：

```text
Avg Score All
Avg Score Agreement
Walk-Away %
Agreement Rate
P1-P2 score difference
T-statistic
```

新增分析指标：

```text
Preference inference accuracy
High-value issue retention
Low-cost concession usage
Opponent high-value issue satisfaction
Pareto efficiency proxy
```

### 10.2 CaSiNo Belief-Scored Candidate Planner

每轮流程：

```text
dialogue history
  -> update issue preference posterior
  -> generate K candidate allocations
  -> estimate opponent utility / accept prob for each candidate
  -> LLM planner chooses candidate or edits one
  -> generator writes persuasive message
```

### 10.3 AgenticPay Contract-Mode Transfer

选择：

```text
multi_buyer_multi_products
multi_buyer_multi_products_multi_seller
```

先 `RUN_LIMIT=5/10` 做 smoke，然后扩大。

重点看：

- contract parse success
- buyer utility
- seller utility
- global score
- selected seller/buyer correctness
- issue concession pattern

## 11. 与当前代码的关系

我们已经有：

- ASTRA runner:
  - `experiments/external_comparisons/run_astra_table1_our_framework.py`
- AgenticPay all-task runner:
  - `AgenticPay_Env/environment/all_tasks.py`
- AgenticPay multi-agent Task1 runner:
  - `AgenticPay_Env/environment/multi_agent.py`
- Simple Env continuous belief / planner variants:
  - `Simple_Env/buyer/continuous_solver/`

建议新增：

```text
ASTRA_Env/
  buyer/
    belief/
    planner/
    generator/
  environment/
  scripts/
  tools/
```

或者先在现有 ASTRA wrapper 中增加：

```text
our_belief_scored_candidate_planner
our_issue_preference_belief
```

## 12. 新项目贡献点

最终论文可以讲成三个 contribution：

1. **Belief-Usable Planning Problem**
   - 指出 opponent modeling 准确不等于策略有效。
   - 研究 belief 如何被 planner 使用。

2. **Structured Issue-Level Belief + Candidate Scoring**
   - 维护 per-issue preference posterior。
   - 用 belief 为 candidate offer 估计 opponent utility / accept probability。

3. **LLM Planner with Belief-Scored Candidate Actions**
   - LLM planner 不自由 hallucinate offer。
   - 它在 belief-scored candidates 上选择/修正。
   - generator 负责合规自然语言。

## 13. 结论

我建议把 CaSiNo / ASTRA 正式加入项目主线，而且放在 AgenticPay 之前：

```text
Simple Env: train planner schedule
CaSiNo / ASTRA: prove belief-to-planner for multi-issue preference negotiation
AgenticPay: universal commerce transfer
```

这样故事会更顺：

- Simple Env 解释为什么 planner 很重要。
- CaSiNo / ASTRA 解释为什么 belief model 必须进入 planner。
- AgenticPay 证明方法可扩展到 multi buyer / multi seller / multi product / multi issue。

这比直接在 AgenticPay all tasks 上硬跑结果更可控，也更容易对导师解释 novelty。
