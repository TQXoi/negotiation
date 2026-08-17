# Continual Resolving Opponent Belief：实验探索计划

## 1. 背景与核心目标

目前的 negotiation framework 中，belief model 主要还是一种 prompt-time module：每一轮把当前对话历史喂给 LLM，让它输出对 seller 的判断，再交给 planner 使用。这种方法有几个明显问题：

- belief 没有真正的记忆状态，每轮容易被最新一句话过度影响。
- confidence 往往是语言上的 confidence，而不是统计意义上的不确定性。
- planner 容易盲信 belief 的点估计，例如过早认为 seller 已经 final。
- 很难分析 belief 是否真的随对话变得更准。
- 很难接入数学计算，例如 expected value、acceptance curve、Bayesian update。

参考 DeepStack 与 Libratus / Superhuman Poker 的思路，我们希望把 belief model 从“每次重新 prompt”升级为：

```text
persistent opponent belief state
  + rule-based / LLM-assisted update
  + candidate-action value estimation
  + continual resolving planner
```

也就是说，buyer 在每一轮都维护一个关于 opponent 的显式 belief state。初始 belief 不确定性很强，随着 seller 的报价、拒绝、接受、让步、威胁、finality signal 等证据逐步更新，并且把不确定性变小或转移。planner 不再直接问 LLM “下一步怎么办”，而是基于 belief state 枚举候选动作并计算 expected value。

目标不是一次性设计出最优版本，而是系统探索三个问题：

1. 如何维护 belief？
2. 每轮如何更新 belief？
3. planner 如何使用 belief？

最终希望形成一个可以在 Simple Env / RLVR Negotiation 上先验证，再迁移到 Terms Env、AgenticPay 或 ASTRA 的低成本 framework novelty。

---

## 2. 对 DeepStack / Libratus 的方法抽象

### 2.1 DeepStack：continual re-solving + value estimation

DeepStack 的核心不是预先算完整策略，而是在每个新局面下重新求解局部策略。它维护对隐藏状态的范围估计，并用 value network 估计截断搜索后的局面价值。

对谈判的启发：

- seller 的 reservation price / cost / patience / flexibility 是隐藏状态。
- 每一轮 seller 的 response 是新 observation。
- buyer 不应该一开始制定固定策略，而应该每轮根据新 evidence 重新 resolve。
- 对未来几轮不需要完整展开到终局，可以用 value function 估计继续谈判的价值。

谈判版本可以写成：

```text
belief_t = update(belief_{t-1}, seller_response_t, buyer_action_t)
candidate_actions_t = enumerate_actions(belief_t, history_t)
value(action) = immediate_deal_value + expected_future_value - failure_risk
action_t = argmax value(action)
```

### 2.2 Libratus / Superhuman Poker：blueprint + endgame solving + self-improvement

Libratus 的重要思想是：先有一个 blueprint strategy，再在关键局面做 subgame/endgame solving，并且从失败局面中自动修补策略漏洞。

对谈判的启发：

- 当前 full framework 可以作为 blueprint planner。
- 在 early rounds，planner 主要遵守基本策略：问信息、低开价、避免 overshoot。
- 在 late rounds 或高风险状态，需要局部搜索：接受、counter、reject、quit 的价值差异。
- 从失败 traj 中挖掘 patch，例如 premature quit、over-accept、belief overconfidence。

谈判版本可以写成：

```text
default planner = full_framework prompt planner
critical state detector = late round / seller finality high / budget close / seller quit risk high
local resolver = candidate offer search + opponent response model + EV scoring
failure patch = mined from failed trajectories
```

---

## 3. Belief 应该维护什么？

一个好的 negotiation belief state 不应该只是自然语言总结，而应该包含连续变量、离散状态、概率分布、confidence 和 evidence。

### 3.1 最小 belief state

先在 Simple Env 中维护以下字段：

```json
{
  "seller_reservation": {
    "mean": 720.0,
    "std": 95.0,
    "p10": 610.0,
    "p50": 720.0,
    "p90": 850.0
  },
  "acceptance_curve": [
    {"price": 600.0, "p_accept": 0.10},
    {"price": 700.0, "p_accept": 0.45},
    {"price": 800.0, "p_accept": 0.80}
  ],
  "concession": {
    "slope": 0.08,
    "last_seller_offer": 820.0,
    "num_price_drops": 2
  },
  "interaction_state": {
    "finality_prob": 0.35,
    "quit_risk": 0.20,
    "patience": 0.65,
    "friendliness": 0.50,
    "dominance": 0.40
  },
  "confidence": {
    "reservation": 0.45,
    "acceptance_curve": 0.40,
    "finality": 0.55
  },
  "evidence": [
    {
      "round": 2,
      "type": "seller_counter",
      "text": "seller reduced from 900 to 820",
      "effect": "lower reservation mean, lower uncertainty"
    }
  ]
}
```

### 3.2 更完整的 belief state

后续可以扩展：

- `seller_cost_dist`：seller 成本分布。
- `reservation_price_dist`：seller 最低可接受价格分布。
- `acceptance_curve`：不同 buyer offer 下的接受概率。
- `counteroffer_model`：seller 可能 counter 到什么价格。
- `quit_risk_curve`：offer 太低时 seller quit 概率。
- `concession_slope`：seller 每轮降价速度。
- `finality_prob`：seller 的 final offer 话术是否可信。
- `language_reliability`：seller 的语言强硬程度是否和行为一致。
- `evidence_memory`：哪些 utterance / action 支持当前 belief。
- `uncertainty`：哪些 belief 仍然不确定，planner 是否应该继续询问。

### 3.3 Belief 表示方式的候选方案

#### A. 点估计 + confidence

最简单：

```text
reservation_mean = 720
confidence = 0.6
```

优点：实现简单，容易进入 prompt。

缺点：planner 容易盲信点估计，无法自然做 EV。

#### B. 区间估计

```text
seller_reservation_range = [610, 850]
confidence = 0.55
```

优点：比点估计稳健。

缺点：仍然不能很好表达区间内部概率。

#### C. 离散 posterior distribution

在价格网格上维护概率：

```text
grid = [500, 520, 540, ..., 900]
posterior[i] = P(reservation = grid[i])
```

优点：

- 可以做 Bayesian update。
- 可以直接得到 acceptance curve。
- 可以计算不确定性，例如 entropy / variance。
- 更适合写论文方法。

缺点：

- 需要设计 likelihood。
- 初期可能比较 heuristic。

#### D. Particle belief

维护若干个 seller type particle：

```text
particle = {
  "reservation": 720,
  "patience": 0.6,
  "concession_style": "moderate",
  "language_style": "tough_but_flexible"
}
```

每轮根据 observation 重新加权。

优点：可以同时维护 reservation、patience、style。

缺点：实现和调参更复杂。

初期建议：**先做 C：离散 reservation posterior + 少量 scalar attributes**。这是数学性、可解释性、实现成本之间最平衡的版本。

---

## 4. Belief 如何初始化？

在 Simple Env / RLVR Negotiation 中，buyer 可见：

- title
- description
- list price / reference price
- buyer budget
- product category
- limited negotiation turns

buyer 不知道 seller cost。初始 belief 可以来自三种方式。

### 4.1 Rule-based prior

用 reference price 和 buyer budget 构造初始 reservation posterior。

例如假设 seller reservation 通常低于 list price，高于某个隐含成本：

```text
reservation_grid = linspace(0.3 * reference_price, 1.0 * reference_price)
prior_mean = 0.65 * reference_price
prior_std = 0.18 * reference_price
```

如果 buyer budget 是 `0.8 * reference_price`，则初始 mutual interest 不一定 guaranteed，但 reservation 大概率落在 `[0.4, 0.9] * reference_price`。

### 4.2 Dataset-calibrated prior

从训练集或已有 traj 统计：

```text
seller_cost / reference_price
final_deal_price / reference_price
accepted_offer / budget
```

形成 category-level prior：

```text
P(reservation_ratio | category)
```

例如 electronics、automotive、beauty 的 seller cost ratio 可能不同。

### 4.3 LLM-assisted prior

LLM 根据商品描述和价格判断：

```text
This product is expensive durable equipment; seller may have less flexibility.
```

但 LLM 只输出 prior adjustment，不直接决定最终 belief。

推荐初始化实验：

| Prior | 描述 | 目的 |
|---|---|---|
| P0 rule prior | 只用 reference/budget 的高斯或 beta 分布 | 最强可控 baseline |
| P1 dataset prior | 用历史数据按 category calibrate | 测试统计 prior 是否有用 |
| P2 LLM prior | LLM 输出 style/flexibility adjustment | 测试语义商品信息是否有用 |
| P3 hybrid prior | dataset prior + LLM adjustment | 最终候选 |

---

## 5. 每轮如何更新 belief？

belief update 可以分成 rule-based update 与 LLM-assisted update。关键是：LLM 不直接覆盖 belief，而是提供 observation features 或 adjustment proposal，再由程序更新。

### 5.1 基础 observation 类型

每轮记录：

```json
{
  "buyer_action": "BUY",
  "buyer_offer": 650,
  "seller_action": "SELL",
  "seller_offer": 820,
  "seller_talk": "...",
  "round": 2
}
```

seller 行为有几类：

- `DEAL`：seller 接受 buyer 之前 offer。
- `SELL`：seller 提出 counteroffer。
- `REJECT`：seller 拒绝但没有报价。
- `QUIT`：seller 退出。

### 5.2 Rule-based reservation posterior update

定义 seller reservation price `r`。若 buyer offer 为 `x`：

```text
P(accept | r, x) = sigmoid((x - r) / tau_accept)
P(reject | r, x) = 1 - P(accept | r, x)
```

如果 seller DEAL 接受 `x`：

```text
posterior_new(r) ∝ posterior_old(r) * P(accept | r, x)
```

如果 seller REJECT 或 counter at `y`：

```text
posterior_new(r) ∝ posterior_old(r) * P(reject | r, x)
```

如果 seller counter 到 `y`，额外加入：

```text
r <= y likely
y reveals seller anchor
posterior mass near y - concession_margin increases
```

可以定义：

```text
P(counter_y | r, x) ∝ Normal(y | r + markup_t, sigma_counter)
```

其中 `markup_t` 随轮数下降。

### 5.3 Concession update

如果 seller offer 从 `s_{t-1}` 降到 `s_t`：

```text
drop = s_{t-1} - s_t
concession_slope = EMA(concession_slope, drop / s_{t-1})
reservation_mean -= alpha * drop
reservation_std *= uncertainty_decay
```

如果 seller 多轮不降价：

```text
concession_slope decreases
finality_prob increases
quit_risk increases slightly
```

### 5.4 Finality language update

LLM 或规则抽取 finality cues：

- "final offer"
- "cannot go lower"
- "take it or leave it"
- "lowest I can do"
- "last chance"

但不能盲信。需要区分 language finality 与 behavioral finality：

```text
if seller says final but later lowers price:
    language_reliability -= delta
    finality_prob correction downward
```

建议维护：

```text
finality_language_score
behavioral_finality_score
finality_prob = combine(language, behavior, round_pressure)
```

### 5.5 LLM-assisted update

LLM 负责抽取语义证据，不负责最终数值覆盖：

输入：

```text
previous belief summary
new buyer action
new seller talk/action
round number
```

输出：

```json
{
  "evidence": [
    {
      "type": "finality_cue",
      "strength": 0.7,
      "quote": "lowest I can go"
    },
    {
      "type": "flexibility_cue",
      "strength": 0.4,
      "quote": "I can work with you"
    }
  ],
  "suggested_updates": {
    "finality_delta": 0.15,
    "patience_delta": -0.05,
    "friendliness_delta": 0.10
  }
}
```

程序再做 clipped update：

```text
new_value = clip(old_value + weight * suggested_delta, 0, 1)
```

重要原则：**LLM 只能提供 bounded evidence，不直接重写 belief。**

### 5.6 Update 方案 ablation

| Update | 描述 |
|---|---|
| U0 no persistent belief | 当前 prompt belief baseline |
| U1 rule-only posterior | 只根据 actions/prices 更新 |
| U2 LLM evidence-only | LLM 抽 evidence，但不做 posterior |
| U3 rule posterior + LLM semantic scalars | 推荐主方法 |
| U4 particle filter | 多 seller type particles |
| U5 learned acceptance model | 用 traj 训练 P(accept/counter/quit) |

第一阶段建议只做 U1/U2/U3，与 current full framework 对比。

---

## 6. Planner 如何使用 belief？

planner 的目标不是相信某个 reservation mean，而是基于 belief 选择最大 expected value 的动作。

### 6.1 Candidate action enumeration

每轮生成候选动作：

```text
BUY at aggressive price
BUY at posterior p25
BUY at posterior p50
BUY at safe price
ACCEPT seller offer
REJECT and ask for lower price
QUIT
```

其中价格必须满足：

```text
offer <= buyer_budget
```

候选 offer 可以来自：

- posterior quantile：`q25, q50, q75`
- seller last offer minus concession step
- buyer budget 的安全比例
- previous buyer offer + planned concession

### 6.2 Expected value scoring

对候选 offer `x`：

```text
P_accept(x) = sum_r posterior(r) * sigmoid((x - r) / tau)
reward_if_deal(x) = (buyer_budget - x) / buyer_budget
EV_buy(x) = P_accept(x) * reward_if_deal(x)
          + P_counter(x) * V_future
          - P_quit(x) * quit_penalty
          - overshoot_penalty
```

对接受 seller offer `s`：

```text
EV_accept(s) = reward_if_deal(s) if s <= budget else -inf
```

对继续谈判：

```text
EV_continue = expected_future_value - round_pressure_penalty
```

对 quit：

```text
EV_quit = 0
```

但如果 mutual interest 很可能存在，早退应该有 soft penalty：

```text
EV_quit = - premature_quit_penalty
```

### 6.3 Value function 的几种实现

#### V0 heuristic value

用手写公式估计未来价值：

```text
V_future = max_candidate_EV_next_round * remaining_turn_factor
```

最容易实现。

#### V1 rollout-free learned value

从已有 traj 训练一个小模型：

输入：

```text
belief state + round + last offer + candidate action
```

输出：

```text
expected final reward
```

#### V2 LLM value critique

让 LLM 对候选动作给风险解释，但最终分数仍由程序算：

```json
{
  "candidate": "$720",
  "risk_notes": "seller has lowered twice; moderate chance to accept",
  "risk_adjustment": -0.05
}
```

#### V3 counterfactual response model

对每个候选 offer 预测 seller response：

```text
offer x -> P(DEAL), P(SELL counter), P(REJECT), P(QUIT)
```

这个可以先用 prompt，后续训练小模型。

### 6.4 Planner ablation

| Planner | 描述 |
|---|---|
| P0 current prompt planner | 当前 full framework |
| P1 posterior quantile planner | 用 posterior 分位数报价 |
| P2 EV heuristic planner | 显式计算 EV |
| P3 EV + LLM risk adjustment | 数学 EV 加少量 LLM 风险修正 |
| P4 counterfactual response reranker | 枚举 K offer，预测 seller response |
| P5 learned value planner | 训练 value model |

第一阶段建议：P1/P2/P4。

---

## 7. 具体实验矩阵

### 7.1 第一阶段：Simple Env 小规模验证

目标：验证 persistent belief 是否比 prompt belief 稳。

环境：

```text
Simple_Env / RLVR Negotiation
Qwen3-30B-A3B base seller
buyer variants: direct_prompt, full_framework, typed_belief, persistent_belief variants
num_test_instances = 64
rollouts_per_instance = 4
```

实验组：

| Variant | Belief | Update | Planner |
|---|---|---|---|
| full_framework | prompt belief | prompt each turn | prompt planner |
| typed_belief | typed prompt | prompt each turn | prompt planner |
| persistent_rule_belief | posterior | rule update | quantile planner |
| persistent_hybrid_belief | posterior + semantic scalars | rule + LLM evidence | EV planner |
| persistent_cf_belief | posterior + response model | rule + LLM | counterfactual reranker |

主要 metrics：

- reward
- deal rate
- buyer bargained ratio
- price overshoot rate
- walk-away rate
- avg rounds

新增 belief metrics：

- `accept_prob_brier`：对 buyer offers 的 accept probability 校准。
- `reservation_interval_hit`：最终 deal price / seller accept price 是否落在 predicted interval。
- `posterior_entropy_delta`：belief uncertainty 是否随对话下降。
- `belief_overconfidence_rate`：confidence 高但结果错误的比例。

### 7.2 第二阶段：对 update 机制做 ablation

固定 planner 为 EV planner，比较 update：

| Variant | Update |
|---|---|
| rule_only | 只用 price/action 更新 |
| llm_only | LLM 输出 typed belief，不维护 posterior |
| hybrid_light | rule posterior + LLM finality/flexibility |
| hybrid_strong | rule posterior + LLM evidence + counterfactual response |

关键分析：

- rule-only 是否已经足够？
- LLM semantic evidence 是否减少 early quit？
- LLM 是否引入噪声？
- hybrid 是否能提升 bargained ratio 而不牺牲 deal rate？

### 7.3 第三阶段：对 planner 使用方式做 ablation

固定 belief 为 hybrid belief，比较 planner：

| Variant | Planner |
|---|---|
| quantile | 用 posterior 分位数报价 |
| EV_static | 当前轮 EV |
| EV_future | 加 future value |
| EV_cf | 加 counterfactual seller response |
| EV_patch | 加 failure patch library |

关键分析：

- EV 是否比 prompt planner 更稳定？
- future value 是否减少过早 accept？
- counterfactual 是否提升 late-round deal？
- failure patch 是否修正 premature quit？

### 7.4 第四阶段：迁移到 Terms Env / Multi-agent

如果 Simple Env 有提升，再迁移：

| Env | 迁移重点 |
|---|---|
| Terms Env | belief 从 price 扩展到 multi-issue utility / preference |
| AgenticPay | seller/buyer 多方时维护多个 opponent belief |
| ASTRA/CasiNo | item priority belief + fairness/stance + concession belief |

Terms Env 中 belief 可以变成：

```text
opponent issue weights
opponent reservation utility
acceptance probability over contract terms
concession slope over each issue
```

Multi-agent 中 belief 变成：

```text
belief[opponent_id][issue_id]
coalition risk
competition pressure
```

---

## 8. 数据与日志设计

每个 episode 必须保存完整 belief trajectory：

```json
{
  "round": 3,
  "history": "...",
  "previous_belief": {...},
  "observation": {...},
  "rule_update": {...},
  "llm_evidence": {...},
  "new_belief": {...},
  "candidate_actions": [
    {
      "action": "BUY",
      "price": 700,
      "p_accept": 0.42,
      "p_quit": 0.12,
      "reward_if_deal": 0.22,
      "ev": 0.08
    }
  ],
  "selected_action": {...},
  "final_model_output": "Thought/Talk/Action..."
}
```

这样后续可以分析：

- belief 是否随着 seller 降价而更新。
- confidence 是否合理上升。
- planner 是否选了 EV 最大动作。
- 失败是 belief 错、planner 错，还是 generator 格式错。

---

## 9. 实现路线

### 9.1 新增模块结构

建议在 Simple_Env 中新增：

```text
Simple_Env/
  belief/
    state.py
    priors.py
    rule_update.py
    llm_evidence.py
    posterior.py
    metrics.py
  planner/
    candidate_generator.py
    ev_scorer.py
    continual_resolver.py
  buyer/
    persistent_belief.py
    persistent_hybrid_belief.py
    persistent_counterfactual_belief.py
```

### 9.2 最小实现版本

第一版只需要：

1. `ReservationPosterior`
   - price grid
   - prior initialization
   - update_accept / update_reject / update_counter / update_quit
2. `BeliefState`
   - posterior
   - concession_slope
   - finality_prob
   - quit_risk
   - evidence
3. `EVPlanner`
   - enumerate K prices
   - compute p_accept
   - compute EV
   - select action
4. `PersistentBeliefBuyer`
   - 每轮 load/update belief
   - planner 选 action
   - LLM generator 只负责写 Talk/Action

### 9.3 初版不做的事情

为了控制复杂度，第一版暂不做：

- 端到端 RL。
- 复杂 particle filter。
- neural value model。
- multi-agent belief。
- belief model SFT。

先证明显式 posterior + EV planner 是否有用。

---

## 10. 成功标准

### 10.1 性能成功

相对 base full_framework：

- reward 提升。
- deal rate 不下降，或轻微下降但 bargained ratio 明显上升。
- overshoot 仍为 0。
- premature walk-away 下降。

### 10.2 机制成功

即使性能提升不大，也希望看到：

- posterior entropy 随轮数下降。
- seller concession 后 reservation posterior 下移。
- finality cue 不再导致 planner 盲目 accept/quit。
- planner 的 selected action 和 EV 排名一致。
- high-confidence belief 的 calibration 更好。

### 10.3 论文 novelty 成功

能够从 “LLM framework prompting” 升级成：

> A continual-resolving negotiation agent that maintains an explicit opponent belief state, updates it with rule-based and LLM-extracted evidence, and selects offers by expected-value planning under uncertainty.

这个说法比 “belief prompt + planner prompt + generator prompt” 更有方法贡献。

---

## 11. 风险与对应解决

| 风险 | 表现 | 解决 |
|---|---|---|
| posterior update 过于 heuristic | 不稳定或错误自信 | 增加 uncertainty floor、temperature、LLM evidence clipping |
| rule-only 忽略语言 | seller finality 无法识别 | 加 LLM semantic evidence |
| LLM evidence 噪声大 | belief 被语言带偏 | LLM 只输出 bounded delta，不允许覆盖 posterior |
| EV planner 太保守 | deal rate 高但 reward 低 | 加 aggressive candidate 与 late-round schedule |
| EV planner 太激进 | seller quit 增加 | 加 quit risk curve |
| 训练数据偏向 successful traj | planner 学会过早 accept | 加 failed traj 的 contrastive/DPO pairs |

---

## 12. 推荐的近期实验顺序

### Step 1：实现 rule-only persistent belief

目标：建立可运行 skeleton。

```text
persistent_rule_belief = posterior reservation + quantile/EV planner
```

跑 16x2 smoke，再跑 64x4。

### Step 2：加入 LLM semantic evidence

目标：看语言证据是否能改善 finality / patience / friendliness。

```text
persistent_hybrid_belief = posterior + LLM evidence deltas
```

对比 rule-only。

### Step 3：加入 EV planner

目标：验证数学 planner 是否比 prompt planner 稳。

比较：

```text
posterior quantile planner
EV planner
EV + future value
```

### Step 4：加入 counterfactual response model

目标：验证 candidate offer reranking 是否真有帮助。

比较：

```text
EV planner
EV + LLM predicted seller response
EV + learned response model
```

### Step 5：训练小 value / response model

目标：让 value estimation 不再全靠 heuristic。

数据来自已有 traj：

```text
input = belief state + candidate offer + round
target = seller response / final reward
```

先训练小 classifier 或 LoRA，不直接 RL 30B。

---

## 13. 汇报时的叙事

可以这样向导师讲：

1. 目前 A+B+C framework 有效果，但 belief model 仍然是 prompt-level，novelty 不够。
2. DeepStack/Libratus 的启发是：在隐藏信息博弈中，不应只让模型一次性规划，而要维护 belief 并持续 re-solving。
3. 我们把这个思想迁移到 negotiation：
   - opponent reservation / patience / flexibility 是隐藏状态；
   - seller response 是 observation；
   - buyer 每轮更新 posterior；
   - planner 枚举候选 offer 并最大化 expected value。
4. 这使 framework 从 prompt engineering 变成了可解释、可校准、可训练的 opponent belief/value model。
5. 实验先在 RLVR Simple Env 验证，再扩展到 Terms Env 的 multi-issue preference belief 和 AgenticPay 的 multi-agent belief。

一句话版本：

> We move from prompting an LLM to "guess the opponent" toward maintaining an explicit, continually updated opponent belief state and using it for expected-value offer planning.
