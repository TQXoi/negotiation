# ASTRA_Env Belief-Usable Planner v0.1

更新时间：2026-07-29

## 1. 目标

按照当前 plan，先在 ASTRA/CaSiNo 环境中实现一个更清晰的 our framework：

```text
belief model
  -> planner generates several candidate actions with different self/opponent utility ratios
  -> belief model scores each candidate
  -> final chooser selects action
  -> generator produces dialogue
```

本版本暂时不加入完整 continuous resolving，也不训练模型。它先把结构写清楚，并保存足够的 trace，方便后续做 SFT/RFT/DPO。

## 2. 代码结构

```text
ASTRA_Env/buyer/belief_usable_planner/
  belief_model.py   # 数学 belief update 和 candidate scoring
  planner.py        # 多 ratio candidate generation 和 final selection
  generator.py      # LLM 只负责 final dialogue
  buyer.py          # 串联完整 pipeline
```

Factory 中新增 variant：

```text
belief_usable_planner
weighted_belief_planner
belief_usable
belief_usable_llm_chooser
```

## 3. Belief Model

当前 belief state 包含：

```json
{
  "issue_beliefs": {
    "Food": {"logit_high": 0.0, "evidence": []},
    "Water": {"logit_high": 0.0, "evidence": []},
    "Firewood": {"logit_high": 0.0, "evidence": []}
  },
  "flexibility": 0.5,
  "stubbornness": 0.35,
  "walkaway_risk": 0.1,
  "last_seller_offer_value": null,
  "entropy": 1.099
}
```

更新方式是 rule/math based：

- P2 文本中出现 `need / important / prefer / want / must` 等词，并靠近某 issue，会提升该 issue 的 high-priority logit。
- P2 在 offer 中保留更多某 issue，会提升该 issue 的 high-priority logit。
- P2 连续让步会提升 flexibility、降低 walkaway risk。
- P2 保持强硬会提高 stubbornness 和 walkaway risk。

posterior 通过 softmax 得到：

```text
P(issue is P2-high) = softmax(issue_logit)
```

## 4. Planner

Planner 不让 LLM 自由生成所有 action，而是先枚举所有可行 allocation，并用不同 self/opponent utility ratio 生成候选：

```text
aggressive_self:    self_weight = 0.92
self_leaning:       self_weight = 0.78
balanced_tradeoff:  self_weight = 0.62
cooperative:        self_weight = 0.48
high_accept:        self_weight = 0.34
```

每个 ratio 选出一个最优 allocation，然后去重。

## 5. Belief Scoring

Belief model 为每个 candidate 计算：

```text
self_score
estimated_partner_score
self_norm
estimated_partner_norm
accept_prob
walkaway_risk
information_gain
belief_score
```

当前打分：

```text
belief_score =
  self_norm * accept_prob
  + 0.12 * opponent_norm
  + 0.10 * information_gain
  - 0.28 * walkaway_risk
  - 0.10 * unfairness * time_ratio
```

早期如果 belief entropy 高，information gain 会鼓励探索性 offer。后期更重视 accept probability。

## 6. Final Chooser

当前 final chooser 是 deterministic：

- early round：优先选择 self_score 高且 walkaway risk 不太高的 candidate。
- middle round：选择 belief_score 最高的 candidate。
- late round：优先选择 accept_prob 较高的 candidate。
- 如果 P2 已给出对 P1 足够好的 offer，后期可以直接 accept。

后续可训练部分正是这个 chooser：

```text
input = scenario + history + belief + candidates
target = selected candidate id / allocation / action type
```

此外新增一个对照 variant：

```text
belief_usable_llm_chooser
```

它和 deterministic 版本共用同一个 belief model、candidate planner 和 belief scorer，但 final chooser 改成 LLM。LLM 只能从 candidate table 中返回 `candidate_id`，不能发明新 offer。这样可以专门比较：

- 前几轮 belief 不准时，LLM 是否比纯 belief_score 更会探索。
- 后几轮 LLM 是否能更合理地在 P1 score 和 accept probability 之间折中。
- LLM chooser 的选择能否作为后续 SFT target。

## 7. Generator

LLM generator 不再决定最终策略，只负责将 selected candidate 写成自然语言和 JSON action。

如果 generator 偏离 selected allocation，代码会强制覆盖回 planner 选择的 allocation，保证 trace 和 action 一致。

## 8. SFT/RFT 入口

每轮 trace 中保存：

```json
{
  "belief": {},
  "candidates": [],
  "chosen_candidate": {},
  "training_hooks": {
    "planner_sft_input": "scenario + history + compact belief + candidate table",
    "planner_sft_target": "chosen candidate id / action type / allocation",
    "belief_sft_target": "posterior calibration from future P2 responses"
  }
}
```

后续可以低成本训练：

1. belief updater calibration
2. final chooser / candidate ranker
3. planner LoRA / pairwise DPO

## 8.1 新增 Belief Accuracy Metric

`summary.json` 现在会自动统计 belief accuracy：

```text
belief_eval_n
belief_top1_accuracy
belief_true_high_prob
belief_entropy
belief_confidence_gap
```

计算方式：

- 从每个 episode 最后一轮 trace 中取 `issue_priority_posterior`。
- 与 scenario 中 P2 真实 `p2_values` 的最高 value issue 对比。
- `belief_top1_accuracy` 表示 posterior top-1 issue 是否命中 P2 true high issue。
- `belief_true_high_prob` 表示 posterior 给 true high issue 的平均概率。

这个 metric 用于辅助分析：如果 score 上升但 belief accuracy 不高，说明提升可能主要来自 planner/candidate structure，而不是 belief 本身。

## 8.2 Parser Fix

v0.2 parser 修正了一个重要问题：seller 经常复述 buyer 的需求，例如：

```text
I understand your health needs and can support you with food and water.
```

旧 parser 容易把这误认为 P2 自己重视 Food/Water。现在规则会尽量区分：

- `my / for myself / I need / I prefer`：作为 P2 self-need evidence。
- `your / for you / support you / your health`：多数情况下不作为 P2 preference evidence。
- offer allocation evidence 权重仍然保留，因为 P2 保留什么资源通常比语言更可靠。

## 9. 运行命令

Smoke：

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1

RUN_BUYER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
RUN_SELLER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
RUN_NUM_INSTANCES=3 \
bash /work5/qixint/ASTRA_Env/scripts/run_partner_base_belief_usable_planner.sh
```

Parser-fix smoke：

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1

RUN_BUYER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
RUN_SELLER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
bash /work5/qixint/ASTRA_Env/scripts/run_partner_base_belief_usable_planner_parserfix_smoke.sh
```

完整 Partner-Base：

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1

RUN_BUYER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
RUN_SELLER_MODEL=openai:Qwen3-30B-A3B-Instruct-2507-base \
RUN_NUM_INSTANCES=100 \
RUN_N_ROUND=12 \
bash /work5/qixint/ASTRA_Env/scripts/run_partner_base_belief_usable_planner.sh
```

默认会同时跑：

```text
direct_prompt
full_framework
belief_usable_planner
belief_usable_llm_chooser
```
