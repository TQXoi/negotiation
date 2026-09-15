# Simple Env / AmazonHistoryPrice Universal Belief–Planner 自动研究报告

更新日期：2026-08-14

## 1. 当前结论

20 个场景不足以证明 framework 有效。`5 seeds × 20 scenes` 主要估计生成随机性，
并没有扩大商品类别、cost/budget 难度和对手策略覆盖。当前研究严格区分：

- development 24：机制调试与版本选择，已使用；
- validation 40：方法冻结后的多 rollout 选择，尚未使用；
- final category-shift 64：全部 electronics，只允许最终一次评测，尚未使用。

截至 V6.1，正式保留的两个关键组件是：

1. V5.4 的 broad、strategically censored belief：拒绝主要是 response-policy 证据，
   不是 seller reservation 的硬真值；
2. V5.5 的 behavioral rejection frontier：planner 不重复没有新条件、且未越过已正式
   拒绝行为区域的方案。

V5.9 strategic-readiness mixture 是当前自然语言 performance 候选，但不是已验证的
calibrated belief：NL6 learned MAE=0.212，仍差于 frozen=0.188。V6.0 的
reservation–aspiration 分解改善 response Brier，却降低 reward；V6.1 的 censored update
接近 frozen 且主动 probe 无净收益。当前不能宣称最终 framework 已确定，更不能进入
training、validation 或 final benchmark。

## 2. 数据与统计设计

原始 128 条数据按 category 排序：automotive 7、baby 11、beauty 12、books 13、
electronics 85。因此固定 split 为：

| split | 原始索引 | n | 用途 | 状态 |
|---|---:|---:|---|---|
| development | 0–23 | 24 | trajectory 归因、机制迭代 | 已使用 |
| validation | 24–63 | 40 | 冻结版本 × 至少 3 rollouts | 未使用 |
| category-shift final | 64–127 | 64 | electronics shift × 5 rollouts | 未使用 |

固定清单位于 `Simple_Env/research_splits/universal_v1/`。最终论文还需要一个预先随机化、
类别平衡的外部 test suite；当前 64 条不能冒充 IID test。

主要报告指标：reward、deal、MI-deal、CI rational walk-away、cost/budget strata、
paired bootstrap CI、belief MAE/coverage、response Brier/ECE、action-lock、first/sequence
action flip、wrong/shuffled/oracle decision regret。只报告平均 reward 或 deal 不足以证明
belief 有用。

## 3. 已发现的 trajectory 问题

### 3.1 私有 scratchpad 泄漏

原 seller transcript 的 `Thought` 明确包含 cost。V5 起 belief 只接收公开
`Talk + Action`；完整原 transcript 仍保留给 evaluator，但不进入 agent belief。

### 3.2 标准单 episode 的正样本缺失

seller accept 会立即终局，belief updater 通常只看到 reject/counter，几乎看不到 accept。
因此把 rejection 当硬 cost 下界会系统性过度让步；完全忽略 rejection 又会重复低报价。

### 3.3 calibration 不等于决策收益

V4 response Brier 改善，但 reward 低于 frozen。V5.8 exact posterior CDF 提高 deal，
却降低 buyer reward。belief calibration、planner 使用和长期 continuation value 必须分别
验证。

### 3.4 live LLM self-play 不是严格 paired

即使 temperature=0，并发 vLLM/MoE 仍会让相同 first action 后的 seller trajectory
不同；某些 seller 甚至在 buyer price 高于真实 cost 时退出。因此同 scenario 的 variant
比较不等同于 same-state causal comparison。为此新增 common-random-number、
oracle-identifiable scripted profile evaluator；它只用于机制选择，不能替代自然语言结果。

## 4. Framework 架构

统一决策流：

`public observations → factorized belief → candidate set → belief-usable planner → locked action renderer`

### 4.1 Factorized belief

- latent preference/utility：reservation distribution、multi-issue option scores/directions；
- response policy：accept curve、counter/quit、patience、asking/concession、rejected frontier；
- reliability：formal action > verified public semantics；
- regime diagnostics：surprise 与 old/new mixture 接口；
- semantic LLM：只做 bounded evidence extraction，不直接选 action。

### 4.2 Belief-usable planner

候选由 environment adapter 保证协议可执行和 focal IR。planner 使用规范化 own utility、
opponent-value proxy、accept/quit belief、uncertainty、continuation 与 information value。
最终 action 由 trusted renderer 锁定，语言模型不能改变价格/合同。

V5.5 的关键区分：

- `reject` 不等于 private reservation；
- 但 formal reject 确实说明“当前同一方案不会被接受”，可形成 behavioral frontier。

这一区分可以迁移到价格、合同、多议题或 allocation，只要求 adapter 提供单调、可审计的
opponent-value proxy。

## 5. 版本与结果

| 版本 | 核心修改 | 关键证据 | 判断 |
|---|---|---|---|
| V5.4 | broad prior + strategic censoring | dev24 reward 0.414；比 frozen +0.161，CI [0.023,0.310]；但 7 个 MI failure 全重复末价 | 当前自然语言 performance reference |
| V5.5 | behavioral rejection frontier | 重复失败可降到 0；小样本 reward 不稳定 | 保留机制 |
| V5.6 | deadline-adaptive 大步 probe | automotive6 reward 0.488 < V5.5 0.594 | 淘汰 |
| V5.7 | latent utility × policy mixture | profile300 belief MAE 0.114；reward +0.0126 vs V5.4，CI 跨 0 | 当前 belief 候选 |
| V5.8 | exact posterior CDF planner | deal 0.550，但 reward 0.392 < V5.7 0.404 | 淘汰 |
| V5.9 | latent strategic readiness | profile MAE 0.099；NL6 reward 0.691，但 MAE 0.212 > frozen 0.188 | performance 候选，calibration 未过 |
| V6.0 | reservation–aspiration factorization + response gate | NL6 Brier 0.113，但 reward/MAE 0.444/0.217 | 淘汰 |
| V6.1 | censored utility likelihood + ask upper bound + aspiration probe | profile reward/MAE 0.387/0.173；probe 差于 no-probe | 淘汰；停止小样本调参 |

V5.4 dev24 的分层结果：

| cost/budget | n | V5.4 reward/deal | full framework | V2 |
|---|---:|---:|---:|---:|
| `<0.40` | 6 | **0.502 / 1.00** | 0.146 / 0.667 | 0.208 / 0.667 |
| `[0.40,0.75)` | 10 | 0.591 / 0.70 | 0.587 / 1.00 | **0.607 / 0.90** |
| `[0.75,1)` | 5 | 0.200 / 0.20 | **0.545 / 0.80** | 0.161 / 0.20 |
| `>=1` | 3 | 0 / 0 | 0 / 0 | 0 / 0 |

V5.7 的 natural-language dev24：reward/deal=0.507/0.750；V5.4=0.419/0.542；
frozen=0.327/0.500；shuffled-mixture=0.474/0.792。learned 相对 frozen reward
`+0.181`，95% CI `[0.016,0.359]`，但相对 V5.4 `+0.088` 的 CI 跨 0。

该 performance 结果没有通过 belief gate：V5.7 MAE/coverage=0.242/0.25，差于 V5.4
的 0.220/0.875；shuffled 的 deal 还更高。因此不能宣称 utility mixture 已正确学习。
根因是 likelihood 没有表示“报价已盈利、seller 仍为更高利润战略 counter”的 latent
accept-readiness。V5.9 虽扩展该维度，仍无法在单次短轨迹中识别高 reservation 与高
aspiration。V6.0/V6.1 的失败进一步确认这是数据可识别性问题，不是继续调一个 planner
常数即可解决。

## 6. Oracle-verified mechanism suite

6 个唯一 hidden utility/policy profiles、每 profile 50 common-random-number rollouts。
buyer 不知道 profile 真值，只观察相同 Talk/Action protocol。

- V5.4/V5.5/V5.6：reward 0.392、deal 0.503；
- oracle utility：0.574/0.773，证明 utility inference 仍有较大上界；
- V5.7：0.404/0.543，belief MAE 0.114；
- V5.8：0.392/0.550，说明更高 deal 未转化为 buyer reward。
- V5.9：0.397/0.523，belief MAE 0.099；
- V6.0：0.381/0.523，MAE 0.120；
- V6.1：0.387/0.503，MAE 0.173，probe 相对 no-probe 无净收益。

该 suite 的作用是拒绝坏机制与验证 identifiability；它不是 AmazonHistoryPrice 的最终
自然语言分数。

## 7. 晋级与下一步

1. 保留 V5.9 作为自然语言 performance reference，但不宣称其 belief 已校准；
2. V6.0/V6.1 作为失败证据永久保留，不再根据同一 6/20 场景调 likelihood 或 planner 常数；
3. 在 repeated-profile suite 中开发 hierarchical policy belief、old/new regime mixture
   与主动 IG probe，使用
   change delay、false alarm、post-switch regret 选择方法；
4. 候选冻结后运行 validation40 × 3 rollouts；
5. 同一 core/不同 adapter 迁移至 AgenticPay，验证 multi-buyer/multi-seller/multi-issue；
6. 只有 Simple、AgenticPay 与至少一个外部 benchmark 都通过后才讨论 calibration SFT；
7. final64 × 5 rollouts 只在方法、超参数和主表定义全部冻结后运行一次。

## 8. 可复现记录

- 总账：`Simple_Env/research_iterations/UNIVERSAL_FRAMEWORK_ITERATION_LEDGER_CN.md`
- V5.4：`iteration_009_v5_4_strategic_censoring/`
- V5.5：`iteration_010_v5_5_behavioral_frontier/`
- V5.6：`iteration_011_v5_6_active_frontier/`
- oracle profile evaluator：`iteration_012_oracle_profile_policy_eval/`
- V5.7：`iteration_013_v5_7_latent_profile_mixture/`
- V5.8：`iteration_014_v5_8_posterior_integrated_planner/`
- V5.7 natural-language：`iteration_015_v5_7_natural_language_gate/`
- V5.9：`iteration_016_v5_9_strategic_readiness_mixture/`
- V6.0：`iteration_017_v6_0_reservation_aspiration/`
- V6.1：`iteration_018_v6_1_censored_aspiration_probe/`

每个 iteration 保存 README、运行命令、config、summary、原始 JSONL trajectory、诊断、
代码快照和 SHA256。失败版本不会删除或覆盖。
