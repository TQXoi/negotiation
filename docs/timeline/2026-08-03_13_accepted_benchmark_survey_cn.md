# 已接收、开源且具有后续采用证据的 Negotiation Benchmark 调研

**日期：** 2026-08-03
**目标：** 寻找比 Simple Env / CaSiNo 更不容易产生环境特化、已被主流会议接收、具有原始代码，并且已被其他方法论文采用的 negotiation benchmarks。

## 1. 结论先行

如果目标是验证 `continuous belief -> planner -> generator` 框架，而不是再找一个单一价格或固定三议题环境，建议把 benchmark 主线调整为：

1. **NeurIPS 2024 LLM-Deliberation**：新的主方法 benchmark。
2. **ICML 2024 NegotiationArena**：跨 game-family 泛化 benchmark。
3. **CraigslistBargain + DealOrNoDeal**：社区公认的 legacy controls。
4. **SOTOPIA / SOTOPIA-Hard**：更开放的 social-negotiation OOD stress test。
5. **AgoraBench**：新型 market-regime stress test；目前仍太新，暂不承担“社区广泛采用”的主张。
6. **AmazonHistoryPrice**：保留为 price-bargaining regression，而不是主 benchmark。

其中最值得立刻接入的是：

```text
LLM-Deliberation
  + NegotiationArena Trading / Sell&Buy
  + CraigslistBargain legacy control
```

这三个环境分别检验：

- multi-agent / multi-issue / adversarial belief planning；
- 跨 resource allocation、exchange、price 三种 game family 的迁移；
- 与多年 negotiation-dialogue 方法的可比性。

CaSiNo 仍可保留为机制诊断环境，但不再单独承担 framework generality 的证据。

## 2. 筛选标准

候选 benchmark 必须尽量满足：

- 已被 ACL、EMNLP、NAACL、ICML、ICLR、NeurIPS 等正式接收；
- 有公开代码或官方可执行环境；
- 至少有一篇独立后续论文实际使用，或已经形成多年方法链；
- 对手存在 private information、hidden goal 或动态行为；
- outcome 可程序化计分，而不是完全依赖 LLM judge；
- action space 足以让不同 belief 导致不同策略；
- 能替换其中一个 agent，而不必重写整个 benchmark；
- 能保存 history、belief、candidate、selected action 和 outcome。

“论文引用 benchmark”与“论文实际在 benchmark 上测试 method”是两回事。本报告优先采用后者。

## 3. 推荐矩阵

| Benchmark | 正式 venue | 官方代码 | 独立方法采用 | 任务丰富度 | Verifiable utility | 适合 belief-planner | 推荐级别 |
|---|---|---:|---:|---:|---:|---:|---|
| LLM-Deliberation | NeurIPS 2024 D&B | 是 | 有后续延伸，生态仍较新 | 很高 | 高 | **很高** | **P0** |
| NegotiationArena | ICML 2024 | 是 | ICLR 2026 AMPO 等 | 高，三类 games | 高 | **很高** | **P0** |
| CraigslistBargain | EMNLP 2018 | 是 | 多篇方法论文 | 中，语言丰富但单价格 | 中–高 | 高 | **P1** |
| DealOrNoDeal | EMNLP 2017 | 是 | 多篇 RL/agent 方法 | 中，多物品但小空间 | 高 | 高 | **P1** |
| SOTOPIA | ICLR 2024 | 是 | ACL 2024/2025/2026 多篇 | 很高 | 低–中，常用 LLM judge | 中 | **P1/P2** |
| AgoraBench | ACL 2026 long | 是 | 暂无足够独立采用 | 高，九种 market regimes | 高 | 高 | **P1/P2** |
| AmazonHistoryPrice | Findings ACL 2024 | 是 | RLVR、TERMS 等复用数据/设定 | 低，单价格 | 高 | 中 | **P2 control** |
| NegotiationToM | Findings EMNLP 2024 | 数据/软件可用 | 主要为诊断 | 离线 mental-state tasks | 高 | 适合 belief，不适合完整 planner | 辅助 |

## 4. P0：LLM-Deliberation

### 4.1 学术与开源状态

论文 **Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation** 被 NeurIPS 2024 Datasets and Benchmarks Track 接收。官方代码为：

- https://github.com/S-Abdelnabi/LLM-Deliberation
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/984dd3db213db2d1454a163b65b84d08-Abstract-Datasets_and_Benchmarks_Track.html

它不是只有一个固定 task，而是一套可扩展的 scorable negotiation-game generator。

### 4.2 为什么比 CaSiNo 更适合验证 generality

该 benchmark 明确包含：

- multi-agent；
- multi-issue；
- semantic-rich goals；
- cooperative、competitive 和 malicious agents；
- private role goals；
- arithmetic、inference、exploration 和 planning；
- 可增加 game difficulty 和新 game setup 的生成程序。

CaSiNo 中 belief 通常集中在三个 issue 的 priority permutation；LLM-Deliberation 的 belief 可以自然扩展为：

```text
counterparty role / goal posterior
cooperation willingness
issue-specific utility
coalition / blocking position
maliciousness or manipulation risk
agreement constraints
```

因此如果相同 framework 能在多个 game templates、不同 party counts 和 adversarial roles 上工作，会比只在 CaSiNo 上提高 score 更能支持 universal claim。

### 4.3 推荐实验

先选三种 difficulty strata：

1. two-party cooperative；
2. multi-party mixed-motive；
3. greedy/malicious-player setting。

每层对比：

- direct/ReAct；
- planner-only；
- belief-only；
- full continuous belief planner；
- oracle-role/goal belief；
- shuffled/wrong belief。

主指标：agreement、individual score、group score、role alignment、constraint violation、malicious exploitation rate、oracle regret。

### 4.4 风险

- 多主体会增加 LLM sampling 方差和成本；
- 不同 game 的 utility schema 需要 adapter；
- 不能直接使用一个 CaSiNo-specific issue posterior。

这些恰好是检验 framework universal abstraction 的必要压力，而不是应规避的问题。

## 5. P0：NegotiationArena

### 5.1 学术与采用证据

**How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis** 是 ICML 2024 Poster：

- https://proceedings.mlr.press/v235/bianchi24a.html
- https://arxiv.org/abs/2402.05863

它包含三个 scenario families：

- shared-resource allocation / ultimatum；
- resource aggregation / trading；
- buy/sell price negotiation。

其价值不只是原 benchmark paper。ICLR 2026 接收的 **Adaptive Social Learning via Mode Policy Optimization** 将 NegotiationArena 的 Sell&Buy 和 Ultimatum 用作 OOD evaluation，每个 task 使用 200 scenarios、role swap 和 4 次执行。这说明它已经开始被独立方法论文作为 transfer benchmark 使用：

- https://openreview.net/forum?id=GG7YQnsdhp

### 5.2 对项目的独特价值

NegotiationArena 最适合回答：

> 同一个 belief/planner interface 能否不改核心算法，从 price bargaining 迁移到 resource exchange 和 ultimatum？

这比在同一类 multi-issue allocation 上换数据更强。

建议定义通用接口：

```text
BeliefState:
  opponent_goal_hypotheses
  opponent_resource/value posterior
  acceptance/response model
  strategy/cooperation type
  uncertainty/evidence

CandidateAction:
  structured proposal
  own utility
  expected opponent utility
  acceptance probability
  information value
  feasibility/risk
```

具体 game 只负责实现：

- `enumerate_or_sample_actions(state)`；
- `self_utility(action)`；
- `belief_features(history)`；
- `validate(action)`；
- `score_outcome(result)`。

如果 core planner 无需知道任务叫 Trading、Sell&Buy 还是 Ultimatum，就能降低 environment-specific 风险。

### 5.3 推荐 protocol

- 先复用 ICLR 2026 的 role swap + repeated runs 思路；
- 固定 GPT-4o 或本地 strong model partner，同时加入 scripted partner；
- train/tune 只使用一个 game family；
- 另外两个 family 作为 zero-shot OOD；
- 报告 in-domain 与 cross-family transfer gap。

## 6. P1：CraigslistBargain

### 6.1 为什么它确实是“大众认可”基准

CraigslistBargain 来源于 EMNLP 2018 的 **Decoupling Strategy and Generation in Negotiation Dialogues**：

- https://aclanthology.org/D18-1256/
- 官方 Cocoa repository 已在本地：`external_negotiation_envs/cocoa/`
- Hugging Face dataset：https://huggingface.co/datasets/stanfordnlp/craigslist_bargains

它有 6K+ human-human negotiations，覆盖不同商品、目标价、语言风格和 dialogue acts。后续方法链包括：

- online value look-ahead；
- personality / Theory-of-Mind modeling；
- RL/self-play negotiation；
- policy planner、diffusion planner 和 dialogue-planning methods；
- 多篇 ACL/EMNLP 方法继续以它作为 negotiation task。

因此在“被社区反复用于 method evaluation”这一点上，它明显强于 AgenticPay、TERMS-Bench 和刚发布的 AgoraBench。

### 6.2 适合本项目的地方

- buyer/seller target price 构成 private information；
- 原论文已经采用 strategy/generation decoupling，与本项目结构高度一致；
- 语言比 Simple Env 更贴近 human marketplace dialogue；
- 可以测试 opponent personality/type inference；
- 可以使用原 human corpus做 offline belief/response-model training。

### 6.3 局限

- 仍是单一 price issue；
- corpus 本身不是现代 LLM interactive evaluator；
- target/list-price normalization 与模拟 counterpart 必须统一；
- 原 Cocoa/Python 依赖较旧。

因此它最适合做 **legacy comparability + language robustness**，不适合单独承担 multi-issue novelty。

### 6.4 推荐用法

- 使用 Cocoa data/schema，但用现代 OpenAI-compatible agent wrapper；
- 固定一组 partner policies：time-dependent、behavior-dependent、LLM partner；
- 使用 human dialogues 训练/校准 response model；
- 在 held-out categories、price ranges 和 partner types 上评价；
- 与 OG-Narrator、planner-only、ToM/personality baseline 对比。

## 7. P1：DealOrNoDeal

### 7.1 学术与采用证据

**Deal or No Deal? End-to-End Learning of Negotiation Dialogues** 发表于 EMNLP 2017，公开了代码和约 6K human-human dialogues：

- https://aclanthology.org/D17-1259/
- 本地原代码：`external_negotiation_envs/end-to-end-negotiator/`
- Cocoa wrapper：`external_negotiation_envs/cocoa/dealornodeal/`

随后被 EMNLP 2018 modular negotiation、EMNLP 2023 personality/self-play 等工作继续使用。

### 7.2 适合与不适合

适合：

- 双方 private item values；
- 多物品组合 allocation；
- exact utility 与 agreement；
- 可枚举 action space；
- 有 human language corpus和长期 baseline lineage。

不适合：

- item space 较小；
- 语言和场景相对模板化；
- 与 CaSiNo 一样可能被“精确枚举 + generic fairness”解决。

推荐作为 CaSiNo 的 legacy replication，而不是另一个主要新环境。它的重要性来自社区历史和可比性，不来自复杂度。

## 8. P1/P2：SOTOPIA

### 8.1 为什么值得加入

SOTOPIA 被 ICLR 2024 接收，拥有持续维护的代码、90 个开放社会场景、private social goals、人物关系和多维评价：

- https://sotopia.world/projects/sotopia
- https://github.com/sotopia-lab/sotopia

它已被后续工作广泛使用：ACL 2024 SOTOPIA-π、ACL 2025 SOTOPIA-Ω、ICLR 2026 AMPO 等，NAACL 2025 还发布了易用的 SOTOPIA-S4，并展示 hiring negotiation：

- https://aclanthology.org/2024.acl-long.698/
- https://aclanthology.org/2025.acl-long.1203/
- https://aclanthology.org/2025.naacl-demo.30/

### 8.2 它不是主 economic benchmark

SOTOPIA 的优势是开放社会目标、关系、秘密与语言策略；弱点是很多指标依赖 LLM judge，own/opponent utility 不像经济博弈那样精确。

因此建议只选 negotiation-tagged scenarios 做最终 OOD stress test，回答：

- belief state 能否从数字偏好扩展到 hidden social goals？
- planner 能否兼顾经济目标、关系、秘密和 social rules？
- framework 是否仍能改善 goal completion，而不会增加 norm violations？

不要用它训练或选择主要 planner 超参数，也不要把 LLM-judge improvement 当作唯一主证据。

## 9. P1/P2：AgoraBench

### 9.1 已确认状态

AgoraBench 对应 ACL 2026 long paper **MERIT Feedback Elicits Better Bargaining in LLM Negotiators**，不是未接收 preprint：

- https://aclanthology.org/2026.acl-long.1155/
- https://github.com/ericoh929/Agorabench

它覆盖九种 economically grounded market regimes，包括：

- vanilla；
- deception；
- monopoly；
- installment；
- negative seller perception；
- single/multi-product substitution。

并提供 MERIT：consumer surplus、negotiation power 和 acquisition ratio 的组合。

### 9.2 评价

AgoraBench 的 scenario diversity 与本项目很匹配，尤其适合：

- seller cost belief；
- deception-aware uncertainty；
- market-power belief；
- alternative-product/outside-option planning；
- multi-product candidate generation。

但你的担忧成立：它刚于 ACL 2026 发布，目前几乎没有时间积累独立方法采用。因此应将其定位为 **new-regime transfer benchmark**，而不是“community-standard benchmark”。

## 10. P2：AmazonHistoryPrice

AmazonHistoryPrice 来自 Findings ACL 2024：

- https://aclanthology.org/2024.findings-acl.213/
- https://github.com/cooelf/AmazonPriceHistory

优点：930 个真实商品、18 categories、历史价格、official code、自动 reward。本地已经深度接入 Simple Env。

它也并非完全无人复用：RLVR negotiation 使用其设定，TERMS-Bench 的 data-grounded scenarios 也使用 AmazonHistoryPrice corpus。但这些复用没有改变它是一维价格 bargaining 的本质。

推荐保留为：

- regression test；
- price-scale/category OOD；
- belief about reservation/cost 的 calibration test；
- 与已有 Simple Env 结果连续对照。

不推荐继续作为 universal framework 的主表。

## 11. 辅助诊断：NegotiationToM 与 multifaceted evaluation

### NegotiationToM

Findings EMNLP 2024，提供 data/software：

- https://aclanthology.org/2024.findings-emnlp.244/

它评估 desires、beliefs、intentions 等 mental states，但主要是离线 ToM stress test。适合评价或预训练 belief model，不适合验证 end-to-end planner outcome。

### Are LLMs Effective Negotiators?

Findings EMNLP 2024：

- https://aclanthology.org/2024.findings-emnlp.310/

它把 negotiation 能力分解为 comprehension、partner modeling、strategic reasoning 和 response generation。建议借用其诊断任务作为模块级 evaluation，而不是把它当 interactive benchmark。

这两项可以避免只看最终 utility：当 full framework 失败时，可以区分是 belief 错、planner 错，还是 generator 没有执行 action。

## 12. 推荐的新 benchmark story

建议不再按“简单到复杂的四个环境”讲，而按 **四种外部有效性** 讲：

### A. Mechanistic validity

- CaSiNo / DealOrNoDeal；
- 目标：belief intervention 是否改变 action、减少 regret。

### B. Cross-game validity

- NegotiationArena；
- 目标：同一 planner 是否跨 allocation、trading、price 工作。

### C. Multi-agent and adversarial validity

- LLM-Deliberation；
- 目标：多方、恶意、复杂约束下的 belief、exploration、coalition/risk planning。

### D. Language and market validity

- CraigslistBargain + AgoraBench；
- 目标：human marketplace language 与复杂 market regimes。

### E. Open social validity

- SOTOPIA negotiation subset；
- 目标：经济 utility 之外的关系、秘密、规范和 hidden social goals。

这样论文不依赖任何一个 environment，同时每个 benchmark 都回答不同问题。

## 13. 建议的最小可执行 suite

考虑开发成本，第一版不要一次接入所有环境。建议：

| Suite | 场景 | 作用 |
|---|---|---|
| CaSiNo Partner-Base | 100–500 paired episodes | 保留现有机制结果 |
| NegotiationArena | Trading + Sell&Buy，各 100 paired scenarios | cross-game transfer |
| LLM-Deliberation | 2-party、multi-party、malicious 各一个 game template | multi-agent stress |
| CraigslistBargain | 100 held-out listings × fixed partners | legacy/community control |

核心 variants 统一为：

```text
direct
planner_only
belief_only
full_continuous_belief_planner
oracle_belief
shuffled_belief
```

所有环境统一报告：

- utility / score over all episodes；
- agreement；
- constraint violation；
- belief calibration/accuracy；
- candidate oracle recall；
- action-selection regret；
- belief-action sensitivity；
- token/latency/error rate。

环境特有指标另列，不组成跨环境 composite score。

## 14. Framework 去环境特化的实现要求

接入新 benchmark 前，建议先抽象四个 protocol：

```text
DomainAdapter
  parse_observation(history, private_state)
  hypothesis_space()
  generate_candidates(belief, state)
  validate_action(action)
  self_utility(action_or_outcome)
  oracle_action(hidden_state)      # evaluator only

BeliefUpdater
  update(prior, observation, action, response)

GenericPlanner
  rank(candidates, belief, horizon, risk_budget)

Generator
  realize(structured_action, history, constraints)
```

核心 planner 只读取通用 candidate features，不读取 `Food/Water/Firewood`、`buyer_budget` 或具体 game 名称。DomainAdapter 可以 domain-specific，但 BeliefUpdater interface、planner objective 和 generator constraint protocol 应保持不变。

最强的 generalization test 是：

1. 在一个环境开发/调参；
2. 冻结 generic planner；
3. 新环境只实现 DomainAdapter；
4. zero-shot evaluation；
5. 再允许少量 calibration，报告 adaptation gain。

## 15. 最终优先级

### 立即做

1. clone/审计 LLM-Deliberation；
2. clone/审计 NegotiationArena；
3. 为两者填写 DomainAdapter feasibility 表；
4. 先做原始 benchmark smoke，确认 parser、score、seed 和 reference baseline；
5. 只实现 `direct` 与现有 `full framework`，验证 wrapper；
6. 再补 oracle/shuffled/planner-only causal ablations。

### 随后做

1. 使用本地 Cocoa 接入 CraigslistBargain；
2. DealOrNoDeal 作为 legacy multi-item control；
3. AgoraBench 做 market-regime transfer；
4. SOTOPIA negotiation subset 做最后的开放社会泛化。

### 不建议

- 同时在 CaSiNo、DealOrNoDeal 上进行大量 planner 调参；两者结构过于接近；
- 把刚发布的 AgenticPay、TERMS 或 AgoraBench 描述为“community-standard”；
- 只统计论文引用数而不检查后续论文是否真正运行 benchmark；
- 用 SOTOPIA 的 LLM judge score替代 verifiable economic metrics；
- 为每个环境重写一个 planner，然后声称 framework generalized。

## 16. 最终判断

用户提出的担忧是成立的：Simple Env、AmazonHistoryPrice、CaSiNo 和 DealOrNoDeal 都有较强学术 lineage，但单独使用会使 framework 看起来只适用于低维、可枚举 bargaining。AgenticPay、TERMS-Bench 和 AgoraBench 更复杂，却尚未积累足够多独立 method evaluations。

现阶段最平衡的选择是：

> 以 NeurIPS 2024 LLM-Deliberation 检验 multi-agent/multi-issue/adversarial planning，以 ICML 2024 NegotiationArena 检验 cross-game transfer，以 CraigslistBargain 提供社区历史可比性；CaSiNo 只承担干净的机制消融。

这个组合既避免押注一个新 benchmark，也避免只在经典但简单的环境中优化。
