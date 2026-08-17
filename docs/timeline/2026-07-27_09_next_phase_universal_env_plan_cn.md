# Next Phase Universal Env 调研与迁移计划

**日期：** 2026-07-27
**目标：** 为下一阶段选择更 universal、更有说服力的 negotiation benchmark，并设计将我们的 continuous belief / planner / generator framework 从 Simple Env 迁移过去的路线。
**重点环境：** TERMS-Bench / Terms Env、AgenticPay。
**配套报告：**

- `/work5/qixint/7.27/NEGOTIATION_CODEBASE_OVERVIEW_REPORT_20260727.md`
- `/work5/qixint/7.27/NEGOTIATION_CONTINUOUS_SOLVER_STAGE_REPORT_20260727.md`

---

## 1. 总体结论

下一阶段建议采用 **双主线 benchmark 设计**：

1. **TERMS-Bench-style / Terms Env 作为主诊断 benchmark。**
   它的优势是固定 simulator counterpart、latent type 可由 evaluator 观测、metrics 可以直接评价 belief calibration、surplus、agreement 和 protocol violation。它最适合展示我们的核心 novelty：persistent opponent belief + belief-aware planner。

2. **AgenticPay 作为 universal commerce stress benchmark。**
   它的优势是 multi buyer / multi seller / multi product / multi-dimensional contract，更接近真实 agentic commerce。它最适合检验我们的 framework 是否从 Simple Env 的单商品价格砍价泛化到更复杂市场结构。

推荐论文/汇报叙事：

> Simple Env 证明 continuous resolving + CEM planner 在可验证单商品砍价中有效；Terms Env 用统一可诊断 benchmark 证明 belief/planner 模块确实改善 opponent modeling 与决策；AgenticPay 用更真实的 multi-agent commerce 环境验证泛化与实用性。

---

## 2. Literature Survey

### 2.1 TERMS-Bench

**原文链接：**

- arXiv: https://arxiv.org/abs/2605.13909
- Project site / leaderboard: https://terms-bench.github.io/

**论文信息：**

`TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond Deal Rate`
Erica Zhang, Fangzhao Zhang, Aneesh Pappu, Batu El, Jose Blanchet, Susan Athey, Jiashuo Liu, James Zou.
arXiv v1 submitted 2026-05-13, v2 revised 2026-06-13.

#### Benchmark 是什么

TERMS-Bench 的核心定位是：

> A diagnostic benchmark for LLM negotiation agents beyond deal rate.

它不是让两个 LLM 黑盒互相谈判，而是构造一个 **Bayesian-game simulator**：

- LLM agent 与一个 fixed, history-reactive counterpart 谈判；
- counterpart 有 evaluator 可见、agent 不可见的 latent type；
- latent type 包括 reservation value、urgency、stance 等；
- environment 知道 counterpart policy 和 payoff structure；
- 因此 evaluator 可以诊断 agent 是 belief inference 错、uncertainty handling 错、control/planning 错，还是 protocol compliance 错。

从 arXiv 摘要看，论文明确批评过去 LLM negotiation evaluation 依赖 LLM-vs-LLM 或 aggregate outcome，例如 deal rate，导致 failure opaque。TERMS-Bench 的解决方案是让 environment 本身成为 verifier：counterpart 的 latent type、policy、payoff 都由环境指定，因此可以自动评估。

Project site 当前描述的 task 是 bilateral price negotiation：

- 单件商品；
- alternating-offer；
- 最多 `K = 10` rounds；
- 双方各自有 private type `t = (r, kappa, eta)`；
- agent 观察 structured JSON，包括 private context、protocol state、constraints、observation、history、side info；
- agent 输出严格 JSON，包括 decision、price、message、belief。

#### Evaluation Protocol

Project site 给出的 agent output schema：

```json
{
  "decision": "Offer" | "Accept" | "Reject",
  "price": <float> | null,
  "message": <string>,
  "belief": {
    "r_hat": <float>,
    "kappa_hat": <float>,
    "stance_probs": {
      "conciliatory": <float>,
      "neutral": <float>,
      "aggressive": <float>
    }
  }
}
```

这点对我们非常重要，因为 TERMS-Bench 本身要求 agent 输出 belief。这使它天然适合评估我们的 belief model，而不是只看最终成交价格。

#### Metrics

TERMS-Bench 的主要 metrics 分成四个正交诊断轴：

| Metric | 含义 | 越高/越低 |
|---|---|---|
| `SE+` | Feasible surplus efficiency；在有可行 ZOPA 的 episode 中捕获的可行 surplus fraction | 高好 |
| `AGR+` | Feasible agreement rate；有可行交易时达成 agreement 的比例 | 高好 |
| `CSE+` | Conditional feasible deal quality；在已成交 feasible episodes 上的 surplus efficiency | 高好 |
| `FAGR-` | No-deal false agreement rate；无可行交易时仍然成交的比例 | 低好 |
| `BE_type` | Belief error；对 counterpart reservation / urgency / stance belief 的综合误差 | 低好 |
| `CritViol%` | Critical violation；价格越界、IR violation、invalid action 等 | 低好 |
| `Ubar` | Mean utility；平均 raw utility | 高好 |

此外 project site 还展示了 stateful procurement chain / TERMS-Bankroll：

- 每个 agent 进行 merchant-side sessions；
- cash balance 在 50 negotiation periods 中持续变化；
- headline 可以是 terminal bankroll / profit；
- 这给了我们一个更 long-horizon 的扩展方向。

#### Baselines / Methods

TERMS-Bench 主要是 benchmark paper，不是训练方法 paper。它评估的是不同 LLM agent：

- frontier closed models；
- open-weight models；
- 统一 JSON-in / JSON-out prompt；
- 尽量用 provider 支持的 maximum reasoning effort；
- counterpart 是 fixed simulator，不是另一个 LLM。

根据 arXiv 摘要，论文评估 13 个 LLM agents，发现 frontier models 在 deal rate 上趋于饱和，但在 surplus extraction、cue use、belief calibration 和 compliance 上仍有明显差异。

#### Reported Results 的可用性

TERMS-Bench 的核心 reported result 不是“某个训练方法提升多少”，而是：

1. deal rate 不能区分强弱 agent；
2. surplus extraction、belief calibration、protocol compliance 更有诊断价值；
3. fixed simulator counterpart 能把 failure attribution 归因到 agent 自身；
4. oracle-reference optimality gap 可以进一步分解能力短板。

Project site 当前 leaderboard 包含更新后的模型结果和 TERMS-Bankroll 可视化；这些结果适合作为 benchmark 背景，但如果论文写作要引用具体数字，建议优先引用 arXiv table 或固定导出的 leaderboard snapshot，避免网页动态更新造成版本漂移。

#### 对我们的意义

TERMS-Bench 与我们的下一阶段目标高度一致：

- 它本身要求输出 belief，正好评估 typed / posterior / continuous belief；
- latent type 可见，可计算 belief error / Brier / coverage；
- fixed counterpart 减少 LLM seller 方差；
- metrics 不只看 deal rate，避免“成交多但让步太大”的误导；
- 可以很自然地把 Simple Env 中的 continuous belief state 扩展成 `r_hat / kappa_hat / stance_probs`。

---

### 2.2 AgenticPay

**原文链接：**

- arXiv: https://arxiv.org/abs/2602.06008
- GitHub: https://github.com/SafeRL-Lab/AgenticPay
- Tutorial: https://agenticpay-tutorial.readthedocs.io/en/latest/
- 本地代码：`/work5/qixint/benchmarks/AgenticPay`
- 本地论文文本：`/work5/qixint/6.3/AgenticPay_2602.06008v1.txt`

**论文信息：**

`AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions`
Xianyang Liu, Shangding Gu, Dawn Song.
arXiv submitted 2026-02-05.

#### Benchmark 是什么

AgenticPay 是面向 buyer-seller transactions 的 multi-agent LLM negotiation benchmark / simulation framework。根据 arXiv 摘要，它建模：

- buyers and sellers with private constraints；
- product-dependent valuations；
- multi-round natural-language negotiation；
- structured action extraction；
- feasibility、efficiency、welfare metrics；
- tasks from bilateral bargaining to many-to-many markets。

本地论文 v1 中写的是：

- 111 negotiation tasks；
- 8 multi-agent configurations；
- 31 basic tasks + 80 realistic tasks；
- 10 business scenarios；
- product values from `$350` to `$120k`；
- max rounds = 20；
- deterministic decoding temperature = 0；
- max generation length = 1024。

当前 GitHub README 版本写的是：

- 160 benchmark tasks；
- 4 scenarios：E-commerce、Food Delivery、Ride-hailing、Apartment Rental；
- 8 market topologies；
- 支持 multimodal product grounding 和 multi-dimensional JSON contracts。

这说明 AgenticPay repo 可能已经比 arXiv v1 更新。后续正式实验需要固定 commit / version，并在报告中说明使用的是 paper v1 setting 还是 current repo setting。

#### Task Categories

论文把 AgenticPay 设计成复杂度 ladder：

| Category | 含义 | 对我们有何价值 |
|---|---|---|
| 1B1S | single buyer, single seller | 从 Simple Env 迁移的第一步 |
| 1BMS | single buyer, multiple sellers | 测 multi-seller belief / seller selection |
| MB1S | multiple buyers, single seller | 测 competition awareness / urgency |
| MBMS | multiple buyers, multiple sellers | 最接近 universal market setting |
| multi-item bargaining | 多商品/组合选择 | 测 product-level belief 与 bundle planning |
| sequential interaction | agent 决定继续/切换/commit | 测 long-horizon planner |
| parallel interaction | 同时处理多个 negotiation threads | 测 memory isolation 与 cross-seller comparison |

#### Scoring / Metrics

AgenticPay 的 scoring 由 Algorithm 1 定义。设：

- final price `p`
- buyer max price `pmax`
- seller min price `pmin`
- bargaining zone `Z = pmax - pmin`
- deal round `t`
- max rounds `T`
- discount factor `gamma`
- deal completion bonus `D`
- deal quality reward `W`
- efficiency bonus `E`
- failure penalty `F`

如果 `pmin <= p <= pmax`：

```text
rb = (pmax - p) / Z       # buyer normalized utility
rs = (p - pmin) / Z       # seller normalized utility
Q  = 4 * rb * rs          # balanced quality term
d  = gamma^(t - 1)

GlobalScore = d * (D + Q  * W + E)
BuyerScore  = d * (D + rb * W + E)
SellerScore = d * (D + rs * W + E)
```

否则：

```text
d = gamma^(T - 1)
GlobalScore = BuyerScore = SellerScore = -F * (1 - d)
```

论文 v1 使用：

| 参数 | 值 |
|---|---:|
| `D` | 30 |
| `W` | 55 |
| `E` | 15 |
| `gamma` | 0.99 |
| `F` | 15 |
| max rounds | 20 |

辅助 metrics：

- deal rate；
- timeout rate；
- overflow rate；
- average rounds；
- multiplicity breakdown；
- scenario category breakdown；
- cross-play buyer/seller asymmetry；
- personality-based negotiation analysis。

#### Reported Results

AgenticPay Table 1 reports overall performance across all 111 tasks:

| Model | GlobalScore | SellerScore | BuyerScore | Deal Rate | Timeout Rate | Overflow Rate | Avg Rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus 4.5 | 86.9 | 76.1 | 63.5 | 100.0% | 0.0% | 0.0% | 3.7 |
| Gemini-3-Flash | 82.2 | 73.3 | 61.1 | 100.0% | 0.0% | 2.7% | 4.8 |
| GPT-5.2 | 81.7 | 81.1 | 58.5 | 100.0% | 0.0% | 0.0% | 3.8 |
| Qwen3-14B | 63.9 | 58.9 | 47.6 | 79.3% | 20.7% | 1.8% | 7.8 |
| Llama-3.1-8B | 32.5 | 26.3 | 25.2 | 51.4% | 48.6% | 10.8% | 15.0 |

Table 2 further decomposes by multiplicity:

- Claude / Gemini / GPT 保持较高 GlobalScore；
- Qwen3-14B 在 1BMS 上明显弱于其他 topologies；
- Llama-3.1-8B 整体较弱；
- multi-agent topology 会改变模型表现，不能只用 bilateral task 代表真实 commerce ability。

论文还指出一个重要现象：

> 所有模型在 seller role 上普遍比 buyer role 更强，buyer-side negotiation 更难。

这对我们特别重要，因为我们的工作就是 buyer-side framework。

#### 本地已有 AgenticPay 结果

我们已经在 Qwen14B single28 fixed-seller setting 下跑过四个 buyer variant：

来源：`/work5/qixint/runs/agenticpay_single28_qwen14_4gpu_full/summary.json`

| buyer | tasks | deal_rate | fixed_seller buyer score | no_seller_error score | deals_only score |
|---|---:|---:|---:|---:|---:|
| belief_prompt | 28 | 0.7143 | 31.9105 | 24.5851 | 30.0484 |
| cot_prompt | 28 | 0.6071 | 26.8610 | 23.0220 | 35.1412 |
| full_framework | 28 | 0.8214 | 23.6146 | 23.7121 | 28.3110 |
| planner_generator | 28 | 0.9286 | 18.8966 | 18.8550 | 19.6853 |

我们额外修正了 fixed-seller buyer scoring，因为原始 AgenticPay 结果会把 seller 失误也归因到 buyer。新增 metrics 包括：

- `avg_buyer_score_fixed_seller`
- `avg_buyer_score_no_seller_error`
- `avg_buyer_score_deals_only`
- `seller_error_attribution_rate`
- `fixed_seller_eval_reasons`

#### 对我们的意义

AgenticPay 是下一阶段最重要的 universal stress benchmark：

- 它覆盖 multi buyer / multi seller / multi product / multi-issue contracts；
- 有论文 table，可作为外部对比背景；
- 有开源代码，工程上更现实；
- 已经有我们的 local wrapper 和 fixed-seller metric；
- 能直接测试 continuous belief 是否能从 price-only 扩展到 product/contract/multi-party opponent modeling。

风险是：

- 原 repo 版本与 arXiv v1 已有差异；
- current repo 有 multimodal / contract changes，和 paper Table 1 不一定完全一致；
- LLM outputs 容易 timeout / parser fail；
- multi-agent experiments 成本高；
- seller error attribution 需要继续严格处理。

---

## 3. Benchmark Selection Recommendation

### 3.1 为什么不是只用 Simple Env

Simple Env 的优点是：

- reward verifiable；
- paper setting 清楚；
- 单商品 price bargaining 很适合 CEM / RL；
- debugging 快。

但缺点也明显：

- 单 buyer / 单 seller；
- 单 product；
- 单 issue；
- seller 固定；
- method 容易学成特定 anchor + concession schedule；
- reviewer 可能认为只是针对 RLVR-style toy bargaining 的优化。

所以 Simple Env 应作为 method development sandbox，而不是最终 universal benchmark。

### 3.2 为什么 Terms Env 适合作为诊断主表

Terms Env / TERMS-Bench-style benchmark 的优势是：

1. fixed simulator counterpart 减少评估方差；
2. latent type 可见，能直接评价 belief；
3. metrics 覆盖 utility、surplus、agreement、violation、belief calibration；
4. 与我们的 persistent belief state / planner coupling 完全对齐；
5. 实验成本低于多 LLM self-play；
6. 适合展示“不是 prompt engineering，而是 belief-state based decision-making”。

建议把 Terms Env 升级为更 paper-aligned 的 TERMS-Bench implementation：

- role: buyer / seller selectable；
- type: `reservation`, `urgency`, `stance`;
- counterpart policy: fixed stochastic + history-reactive；
- regimes: easy / medium / hard 或 GFT / narrow-ZOPA / no-ZOPA；
- behavioral families: conciliatory / neutral / aggressive / deadline / deceptive-cue / noisy-concession；
- output: JSON decision + price + message + belief；
- metrics: `SE+`, `AGR+`, `CSE+`, `FAGR-`, `BE_type`, `CritViol%`, `Ubar`。

### 3.3 为什么 AgenticPay 适合作为泛化主表/补充主表

AgenticPay 的优势是：

1. 多智能体市场结构更 universal；
2. multi-product / multi-contract 更接近真实 commerce；
3. paper 已有 111-task table；
4. current repo 有 160-task extension；
5. 我们已有 wrapper 和 fixed-seller metric；
6. 它能回答导师可能会问的问题：你的 continuous belief 是否只会砍一个价格？

建议 AgenticPay 用作：

- **Phase 1:** 1B1S fixed-seller clean comparison；
- **Phase 2:** 1BMS single buyer multi seller，展示 multi-seller belief；
- **Phase 3:** MBMS selected tasks，展示 generality；
- **Phase 4:** multi-dimensional contract，展示 multi-issue belief。

---

## 4. Our Framework Migration Design

### 4.1 当前 Simple Env Framework

当前 Simple Env continuous solver：

```text
scenario + history
  -> Persistent Belief State
  -> Rule / LLM / Hybrid Update
  -> EV / Parametric Planner
  -> Prompt Generator
  -> Thought / Talk / Action
```

其中 belief state 主要维护：

- seller reservation posterior；
- accept probability curve；
- quit risk；
- finality probability；
- patience；
- concession slope；
- evidence log。

CEM 目前学习的是 planner 低维参数：

- first anchor ratio；
- early / late concession rate；
- belief quantile multipliers；
- seller discount factor；
- accept thresholds；
- lowball / quit / future value weights。

### 4.2 迁移到 TERMS-Bench / Terms Env

#### Belief State 设计

把 Simple Env 的 seller reservation posterior 扩展为 TERMS type posterior：

```python
class TermsBeliefState:
    reservation_dist: Distribution[float]      # counterpart r
    urgency_dist: Distribution[float]          # counterpart kappa
    stance_probs: dict[str, float]             # conciliatory / neutral / aggressive
    concession_slope_dist: Distribution[float]
    accept_curve: list[tuple[price, prob]]
    no_deal_prob: float
    evidence: list[EvidenceItem]
    confidence: float
```

其中：

- `reservation_dist` 对应 TERMS 的 `r_hat`;
- `urgency_dist` 对应 `kappa_hat`;
- `stance_probs` 对应 `stance_probs`;
- `accept_curve` 给 planner 使用；
- `evidence` 用于 grounding 和可解释性；
- `confidence` 防止 planner 过度相信早期 belief。

#### Update 设计

Rule-based update：

| 观察 | posterior 更新 |
|---|---|
| counterpart reject low offer | reservation 上移，aggressive/unyielding 概率上升 |
| counterpart counter with small concession | urgency 下降，stance aggressive 上升 |
| counterpart counter with large concession | urgency 上升，stance conciliatory 上升 |
| deadline approaching but still hold | patience 上升，accept curve 右移 |
| message contains urgency cue | urgency 上升，但 cue weight capped |
| message contains finality cue | finality risk 上升，但不直接当真 |

LLM-assisted update：

- 只解析 linguistic cue；
- 不直接覆盖 numeric posterior；
- 输出 evidence tags 和 weak likelihood multipliers；
- rule model 负责最终 posterior normalization。

#### Planner 设计

Terms planner 应从 single offer EV 扩展到 protocol-aware decision：

```text
candidate_actions = [
  Offer(price_i, message_style_i),
  Accept(current_offer),
  Reject()
]

score(action) =
  expected_utility(action | belief)
  - violation_risk_penalty
  - false_agreement_penalty_in_no_zopa
  - deadline_risk_penalty
  + information_gain_bonus
```

关键变化：

- 在 feasible ZOPA 内追求 `SE+` 和 `CSE+`;
- 在 no-ZOPA 时追求低 `FAGR-`;
- belief error 不只是 logging，而要进入 planner；
- 如果 uncertainty 高，planner 可以选择信息获取型 offer；
- 如果 accept 当前 offer utility 非负但 future EV 明显更高，则继续 negotiate。

#### Generator 设计

Generator 必须输出 TERMS JSON：

```json
{
  "decision": "Offer",
  "price": 80.0,
  "message": "I can move a little, but this needs to stay economical.",
  "belief": {
    "r_hat": 75.0,
    "kappa_hat": 0.42,
    "stance_probs": {
      "conciliatory": 0.20,
      "neutral": 0.55,
      "aggressive": 0.25
    }
  }
}
```

需要 validator：

- JSON schema validation；
- price bounds；
- monotonic offer constraint；
- accept legality；
- stance probabilities sum to 1；
- `r_hat` inside bounds；
- no private info leakage。

### 4.3 迁移到 AgenticPay

AgenticPay 比 TERMS 更难，因为它不仅是 price：

- multi buyer / seller；
- multi product；
- multi-dimensional contract；
- parallel / sequential interaction；
- possible multimodal product grounding；
- seller error attribution。

#### Belief State 设计

需要从 single seller belief 扩展为 entity-indexed belief：

```python
class AgenticPayMarketBelief:
    seller_beliefs: dict[SellerId, SellerBelief]
    buyer_competition_beliefs: dict[BuyerId, BuyerBelief]
    product_beliefs: dict[ProductId, ProductBelief]
    market_state: MarketStateBelief
```

每个 seller belief：

```python
class SellerBelief:
    reservation_price_range: Range
    accept_price_curve: list[tuple[price, prob]]
    concession_slope: float
    patience: float
    dominance: float
    friendliness: float
    contract_term_preferences: dict[TermName, TermPreferenceBelief]
    reliability: float
    evidence: list[EvidenceItem]
```

contract term preferences：

```python
class TermPreferenceBelief:
    preferred_values: list[str | float]
    flexibility: float
    tradeoff_with_price: float
    confidence: float
```

market state belief：

```python
class MarketStateBelief:
    outside_option_value: float
    seller_competition_level: float
    buyer_competition_level: float
    urgency: float
    expected_switching_value: float
```

#### Update 设计

AgenticPay 的 update 要处理更多 evidence：

| Evidence | Belief update |
|---|---|
| seller price quote | reservation / accept curve |
| seller rejects contract term | term preference / flexibility |
| seller accepts non-price concession | tradeoff_with_price |
| seller delays / times out | reliability / patience |
| competing seller lower offer | outside_option_value 上升，target price 下调 |
| competing buyer pressure | buyer_competition_level 上升，deal urgency 上升 |
| product match high | buyer private value effective weight 上升 |

#### Planner 设计

AgenticPay planner 要从 single-offer planner 变成 market-level planner：

```text
for each active seller/product:
    estimate contract frontier
    generate candidate contracts
    estimate seller acceptance and buyer utility
    estimate switching value

choose:
    continue with seller A
    switch to seller B
    accept offer
    propose contract bundle
    walk away
```

score:

```text
EV(contract, seller) =
    P_accept(contract | seller_belief) * BuyerScore(contract)
  + (1 - P_accept) * FutureValue(next_state)
  - timeout_risk
  - seller_error_risk
  - buyer_infeasibility_penalty
```

对于 multi-seller：

```text
ChosenSeller = argmax_s [
    max_contract EV(contract, s)
    + outside_option_bonus
    - communication_cost
]
```

对于 multi-issue contract：

```text
CandidateContract = {
  price,
  delivery_time,
  return_policy,
  quantity,
  warranty,
  packaging,
  service_level,
  ...
}
```

Planner 需要学习/搜索的不再只是 price concession schedule，还包括：

- price vs term concession tradeoff；
- seller switching threshold；
- accept threshold；
- timeout risk tolerance；
- competition aggressiveness；
- contract repair preference。

#### Generator / Validator 设计

AgenticPay 已有 native parser 和 contract JSON validator，我们应该保留：

- `NativeBuyerNaturalizer` 继续复用 AgenticPay 原生输出格式；
- 我们只在 private prompt suffix 注入 belief / plan；
- `ResponseValidator` 做 buyer-side feasibility；
- fixed-seller metric 继续避免 seller error 污染 buyer score。

---

## 5. Proposed Experimental Roadmap

### Phase A: TERMS-Bench-style Diagnostic Benchmark

目标：做一个能让导师/审稿人一眼看出 belief model 贡献的主诊断表。

#### A1. Paper-aligned Terms Env Upgrade

修改 `Terms_Env/`：

- 加入 latent type `r, kappa, stance`;
- counterpart 改成 fixed stochastic policy；
- 输出 schema 改成 TERMS-style JSON；
- 增加 `SE+ / AGR+ / CSE+ / FAGR- / BE_type / CritViol% / Ubar`;
- 增加 easy / narrow-ZOPA / no-ZOPA regimes；
- 增加 counterpart behavioral families。

#### A2. Baseline Matrix

在 upgraded Terms Env 中比较：

| Group | Variants |
|---|---|
| Prompt | direct_prompt, cot_prompt |
| Existing framework | belief_prompt, planner_generator, full_framework |
| External-style belief | astra_style_belief, bond_style_posterior, preference_estimation |
| Our belief | typed_belief, typed_evidence_belief, continuous_rule_ev |
| Our planner | continuous_ev, continuous_cem |

#### A3. Expected Main Table

| buyer | SE+ | AGR+ | CSE+ | FAGR- | BE_type | CritViol% | Ubar |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_prompt | | | | | | | |
| belief_prompt | | | | | | | |
| full_framework | | | | | | | |
| typed_belief | | | | | | | |
| continuous_rule_ev | | | | | | | |
| continuous_cem | | | | | | | |

#### A4. Key Hypotheses

| Hypothesis | Metric evidence |
|---|---|
| persistent belief improves opponent modeling | lower `BE_type`, better calibration |
| planner coupling improves surplus | higher `SE+`, `CSE+`, `Ubar` |
| uncertainty-aware planner avoids false deals | lower `FAGR-` |
| validator keeps protocol safe | lower `CritViol%` |
| CEM improves control | higher `SE+` at similar `AGR+` |

### Phase B: AgenticPay Universal Commerce Migration

目标：证明方法不是 Simple Env / Terms Env 特化。

#### B1. Clean 1B1S fixed-seller

先复现 single buyer single seller：

- fixed seller；
- variants: direct, belief, full, continuous, CEM；
- metrics: GlobalScore, BuyerScore, fixed-seller BuyerScore, deal rate, timeout, overflow, avg rounds；
- 保存 seller error attribution。

#### B2. 1BMS multi-seller

新增 multi-seller belief：

- 每个 seller 独立 posterior；
- outside option value；
- seller switching planner；
- compare whether continuous belief improves seller selection and buyer score。

#### B3. Multi-product / contract

扩展 belief 到 product / term preferences：

- product fit belief；
- seller contract term flexibility；
- price-term tradeoff；
- contract candidate generator；
- term-level acceptance model。

#### B4. Full selected AgenticPay tasks

选择代表性 task subset：

- 1B1S price；
- 1BMS seller competition；
- MB1S buyer competition；
- MBMS market；
- multi-dimensional contract。

不要一开始跑全 160 tasks；先做 balanced subset，确认 parser/timeout/metric 稳定后再 full run。

---

## 6. Concrete Implementation Plan

### 6.1 Terms Env Code Plan

新增或修改：

```text
Terms_Env/
  environment/
    latent_type.py          # r/kappa/stance regime sampler
    counterpart_policy.py   # fixed stochastic history-reactive seller/buyer
    terms_metrics.py        # SE+, AGR+, CSE+, FAGR-, BE_type
  buyer/
    continuous_terms_buyer.py
    continuous_terms_belief.py
    continuous_terms_planner.py
  tools/
    summarize_terms_belief.py
    plot_calibration.py
```

复用：

```text
Simple_Env/buyer/continuous_solver/
```

但抽象为 common module 会更干净：

```text
negotiation_framework/
  belief/
  update/
  planner/
  generator/
  validators/
```

第一版可以先复制/适配，等接口稳定后再抽公共库。

### 6.2 AgenticPay Code Plan

新增：

```text
experiments/agenticpay_framework/
  continuous_belief.py
  market_belief.py
  continuous_planner.py
  contract_candidate_generator.py
  agenticpay_cem_params.py
  agenticpay_cem_train.py
```

修改：

```text
experiments/agenticpay_framework/buyer_agent.py
experiments/agenticpay_framework/components.py
experiments/run_agenticpay_single28_framework.py
experiments/run_agenticpay_single28_fixed_seller_comparison.py
```

新增 variants：

```text
continuous_rule_ev
continuous_hybrid_ev
continuous_cem
multi_seller_continuous_ev
contract_continuous_ev
```

### 6.3 Metrics Alignment

为了跨环境报告，需要定义一张 common metrics table：

| Common metric | Simple Env | Terms Env | AgenticPay |
|---|---|---|---|
| utility | reward | `Ubar` / buyer utility | BuyerScore / fixed-seller BuyerScore |
| deal | deal_rate | `AGR+` / deal_rate | deal_rate |
| surplus quality | bargained_ratio | `SE+`, `CSE+` | normalized buyer utility / GlobalScore |
| violation | overshoot | `CritViol%`, `FAGR-` | overflow, feasibility / IR violation |
| belief | posterior coverage | `BE_type`, Brier | seller reservation / term coverage |
| efficiency | avg_rounds | rounds / efficiency | avg_rounds |

不要把不同环境的 raw score 混成一个 composite score。主表可以横向展示 normalized metrics，但结论必须按环境解释。

---

## 7. Risks and Mitigations

| 风险 | 影响 | 缓解 |
|---|---|---|
| Terms Env 自建版不完全等于 TERMS-Bench | paper comparison 受限 | 明确称为 TERMS-Bench-style；尽量对齐公开 schema/metrics |
| AgenticPay repo version 与 paper v1 不一致 | table 对比困难 | 固定 commit，标注 paper-v1 / current-repo setting |
| LLM self-play 方差大 | 结果噪声大 | fixed seller / fixed simulator / paired seeds |
| parser / timeout 多 | 数据缺失 | resume、retry、structured validator、failure metrics |
| continuous solver 过拟合 price-only | novelty 受质疑 | 加 multi-issue / multi-seller belief |
| CEM 在小样本过拟合 | full eval 掉分 | held-out regimes + paired eval + confidence interval |
| belief 不被 planner 使用 | 只像 logging | explicit EV/risk formula + ablation |

---

## 8. Recommended Next Actions

### Immediate

1. 等当前 CEM eval 结束后，更新 Simple Env result table；
2. 开始 Terms Env paper-aligned upgrade；
3. 定义 `TermsBeliefState` 和 `terms_metrics.py`；
4. 先跑 30-60 episode smoke，确认 JSON / metrics / trajectory 正常。

### Short Term

1. 在 Terms Env 上跑 full baseline matrix；
2. 对 continuous belief 做 calibration plot；
3. 对 CEM planner 做 held-out evaluation；
4. 写一页 method diagram：persistent belief -> candidate action EV -> generator。

### Medium Term

1. 迁移 continuous belief 到 AgenticPay 1B1S；
2. 加 multi-seller belief state；
3. 在 1BMS 上比较 seller selection / buyer score；
4. 用 elite trajectories 做 planner/reranker SFT 或 preference training。

### Long Term

1. AgenticPay selected full topology benchmark；
2. TERMS-Bankroll-style stateful procurement chain；
3. multi-agent / multi-product / multi-issue continuous resolving；
4. paper-ready cross-env table。

---

## 9. One-Slide Message for Advisor

**Problem:** Existing framework improves Simple Env but may be over-specialized; novelty should move from prompt decomposition to explicit opponent modeling.

**Benchmark Choice:** Use TERMS-Bench-style simulator for clean diagnosis, and AgenticPay for universal commerce stress testing.

**Method:** Maintain persistent, evidence-grounded belief over opponent type; update it across turns; let planner evaluate candidate actions under this belief; learn planner parameters via CEM / SFT / RL.

**Expected Contribution:** A belief-centric negotiation agent that improves not just deal rate, but surplus extraction, false-agreement avoidance, calibration, and protocol compliance across single-price, multi-issue, and multi-agent markets.
