# AgenticPay：全部方法尝试、最终结果、问题与 Trajectory 查看指南

更新时间：2026-08-23
实验范围：只修改 buyer；native seller、AgenticPay task、协议解析器和官方 scorer 固定
主要优化目标：AgenticPay BuyerScore；belief calibration、classifier accuracy、action flip 等仅用于解释

## 0. 结论先行

目前还不能声称我们的 universal framework 在 AgenticPay 上稳定超过 direct prompt 或 CoT。

最可信的完整 28-task 对比（Iteration 014，28 tasks × 3 CLI seeds）为：

| Buyer | Mean BuyerScore | DealRate | TimeoutRate | score-success mismatch |
|---|---:|---:|---:|---:|
| Direct prompt | 31.87 | 82.14% | 17.86% | 21.43% |
| CoT prompt | **32.16** | 85.71% | 14.29% | 25.00% |
| Historical full framework | 27.23 | **92.86%** | 7.14% | 39.29% |
| Universal V4 | 15.59 | 82.14% | 17.86% | 50.00% |
| Universal V13 (offline AWR/LCB) | 17.98 | 85.71% | 14.29% | 57.14% |

后续 V27 在另一次 all-28 greedy run 中得到 35.18，CoT 为 32.26，表面提高
`+2.92`；但 task-cluster bootstrap 95% CI 为 `[-11.55, +17.49]`，而且三个 CLI
seed 的完整 trajectory 完全相同，实际只有 28 个独立 task clusters，不能将 84 行当成
84 个独立样本。更重要的是：

- 25 个 contract/multi-issue tasks 上，V27 相对 CoT 是 `-1.88`，CI
  `[-15.82, +11.03]`；
- 3 个 price-only tasks 上是 `+42.94`，说明 aggregate 增益主要来自少数简单任务；
- 因而 V27 不是已经验证的 multi-issue universal improvement。

最新 V31 在 6 个困难 task 的定向 gate 中，从 V29 的 34.30 提高到 37.59，DealRate 从
66.67% 提高到 100%，timeout 从 2 降为 0；但 Task20 仍有 1 个 mismatch，所以预注册 gate
判定失败，all-28 confirmation 没有运行。这个结果说明 V31 改善了“谈不成”，尚未解决
“语言上谈成、benchmark scorer 却不能确认有效合同”的核心问题。

因此，当前最准确的总结是：

1. action locking、terminal accept guard、calibrated opening、独立 issue classifier 和
   settlement frontier 分别修复过真实局部错误；
2. 这些机制尚未组合成一个在不同 multi-issue utility profiles 上稳定安全的 planner；
3. 当前最好结果仍是候选信号，不是超过 CoT 的最终结论；
4. 下一步不应继续只在 2–6 个失败 task 上手调阈值，而应首先让 active probe 保证
   non-crossing、一次只改变一个 issue，并在 terminal settlement 恢复语义风险补偿；随后用
   task-held-out、非退化采样做全 28-task 验证。

## 1. AgenticPay task、记录和指标究竟是什么

### 1.1 每个 task 中有什么

single28 suite 包含 28 个 buyer–seller task。每个 task 至少包含：

- public scenario、商品/服务信息和用户需求；
- contract schema：`price`、可选 `continuous_terms`、可选 `discrete_terms`；
- buyer 私有 utility：最大价值、连续项权重、离散选项权重；
- seller 私有 utility：最低成本和各条款履约成本；
- 最多 20 轮自然语言交互；
- 解析出的 contract、成交状态、round discount 和 BuyerScore。

buyer runtime 可以使用自己的 utility 和公开历史，但不得读取 seller private utility。报告和
trajectory viewer 显示 seller utility 只用于 post-hoc failure audit。

### 1.2 BuyerScore

在简单 price-only 可行成交中，BuyerScore 可近似理解为：

`discount × [deal bonus + surplus weight × normalized buyer surplus + efficiency bonus]`

其中 normalized buyer surplus 近似为 `(buyer_max - agreed_price) / (buyer_max - seller_min)`。
multi-issue task 则使用完整合同效用，不只由价格决定；更快交付、退货、保证、服务质量等条款
可能同时提高 buyer utility、降低 seller utility。因此“价格在双方区间内”不等于合同可行。

无有效成交时有 failure/discount penalty。代码中的默认 single28 权重审计为
`Db=10, Wb=80, Eb=10, Fb=15, gamma=0.99`。应以 benchmark 返回的 BuyerScore 为准，
不能用框架内部 `own_utility` 代替它。

### 1.3 native 与 fixed-seller-corrected score

JSONL 同时可能含：

- `buyer_score_original` / `buyer_score`：原生 rollout 的结果；
- `buyer_score_fixed_seller`：固定 seller 审计下，若原生 seller 错过了 buyer 已经提出的可行
  proposal，则使用最佳公开可行 buyer offer 做 correction；
- `score_success_mismatch`：环境状态写为 `success/agreed`，但 contract scorer 判定没有有效合同
  或合同违反隐藏 IR。

本 reward-first 流水线的聚合规则是“有明确 fixed-seller correction 时使用它，否则使用 native
BuyerScore”，同时必须分别报告 DealRate、TimeoutRate 和 mismatch。不能用 correction 掩盖
协议错误。

### 1.4 seed 的统计限制

AgenticPay 原生 buyer/seller 在当前设置以 `temperature=0.0` greedy decoding 调用模型。
runner 的 `--seed` 只设置 Python/NumPy/Torch RNG，没有进入模型采样。因此 Iteration 027 的
三个 seed 逐轮文本完全相同。可信的统计单位是 task cluster，而不是 JSONL 行数；真正多 rollout
实验需要显式设置非零 temperature / API seed，并保存 sampling config。

## 2. 当前 framework 如何实现

统一执行链是：

`environment adapter -> canonical state -> continuous belief update -> candidate generation -> belief-usable planner -> selection validator -> action-locked renderer -> protocol validator`

每一步维护的内容如下：

| 环节 | 维护/输出 | 主要代码 |
|---|---|---|
| Canonical state | session、counterparty、round、公开 observation、issue schema、己方 utility | `negotiation/framework/schemas.py` |
| Belief store/update | reservation interval、issue option preference、response policy、拒绝 frontier、可靠度与 direct evidence count | `negotiation/framework/belief.py` |
| Candidate generation | OFFER/ACCEPT/CONTINUE、完整 contract、own utility、belief 下的 acceptance proxy | `negotiation/AgenticPay_Env/buyer/universal_framework.py` 的 `AgenticPayAdapter` |
| Planner | 对候选的 own utility、agreement、risk、deadline、belief value 进行排序；可叠加 AWR residual | `negotiation/framework/planner.py`、`conservative_awr_planner.py` |
| Selection validator | terminal accept、opening、issue guard、settlement buffer/frontier 等硬边界 | `AgenticPay_Env/buyer/universal_framework.py` 的 `validate_selection` 路径 |
| Locked rendering | 先锁定结构化 action，再让语言层只表达该 action；最后检查语义/contract 一致性 | `negotiation/framework/engine.py` 与 AgenticPay adapter |
| Variant registry | 每个 V1–V31 开关组合和说明 | `negotiation/AgenticPay_Env/buyer/variants.py` |
| 环境执行/落盘 | single28 调用、resume、`task_results.jsonl` | `negotiation/AgenticPay_Env/eval.py`、`environment/single28.py` |
| issue classifier | 对 continuous direction / discrete option burden 分类，可 abstain，可被直接 seller evidence 拒绝 | `negotiation/AgenticPay_Env/buyer/issue_classifier.py` |

当前 V31 具体是在 V30 上：

1. 对 active frontier 加 buyer utility leverage 上限，防止为了猜测 seller 喜好而牺牲过多己方效用；
2. continuous issue 使用 midpoint probe，避免直接跳到极端 endpoint；
3. 普通 OFFER 只受 1% base floor，完整 semantic-risk buffer 只用于 terminal settlement；
4. 保留 5% buyer utility reserve；
5. renderer 明确说明“价格让步与条款让步是一个 compensated package”；
6. 仍只依赖公开 seller 行为，不读取 seller private weights。

上述思路合理，但 Task20 暴露了“探索 offer 意外直接成交”：buyer 将公开 profile 从
`batched/no-extra` 同时改为 `rush/extra`，并把价格提高到 12.15；seller 回出相同非价格条款、
价格 10.08。按原生环境规则，两份合同 terms 相同且 `seller_price <= buyer_price`，因此正常
settle 在 10.08；但该完整合同的 post-hoc seller utility 是 `-4.58`，官方 BuyerScore 因此为
`-0.2985`。问题不是合同 identity，而是一个本应收集证据的 multi-issue probe 在同轮跨价，
尚未来得及经过 terminal semantic-risk floor 就触发成交。

原始 example summary 没有保存 env 内部 `agreed_contract`，旧 wrapper 因而把 diagnostics 写成
`missing_agreed_contract`。2026-08-23 源码与轨迹联合审计已确认这是记录缺口；runner 现在只读
采集终止态 contract，viewer 对旧数据明确标注 reconstructed audit。该修复不改变 seller、环境、
scorer 或历史 BuyerScore。

## 3. 全部 AgenticPay 尝试：Iteration 000–031

以下按“实际试图解决的问题”分组。小样本 smoke 只说明机制是否可达，不等价于 benchmark
最终结果。

### 3.1 初始全集、协议正确性与 terminal boundary（000–004）

| Iteration / Variant | 唯一主要改动 | 结果 | 决策 |
|---|---|---|---|
| 000 / V1,V2 | 初始 universal belief/planner；保守 AWR | all28×3：direct 32.95，V1 15.73，V2 18.18 | universal 明显不合格 |
| 001 / V3 | action-consistent renderer | 8-task×3：V3 33.13 vs V1 19.76；action mismatch 158/177 → 0/329；但 direct 50.08、CoT 45.96 | 保留为 protocol-correct base |
| 002 / V4 | proposal-backed terminal ACCEPT/CONTINUE guard | 8-task×3：41.56 vs V3 33.13；DealRate 87.5%；但低于 CoT 45.96 | 晋级 frozen reference，仍非最终方法 |
| 003 / V5 | public verified response frontier | 39.03 vs V4 41.56；11/24 seller-private infeasible agreements | 拒绝 |
| 004 / V6 | stable public seller proposal feasibility guard | 5-task 18.43；guard override 0；mismatch 4/5 | 机制不可达，拒绝 |

关键经验：先锁 action 和 terminal semantics 是必要的；仅从“seller 说过/重复过这个 proposal”
推断隐藏可行性并不可靠，因为 seller LLM 自己也会接受或提出违反其 benchmark utility 的合同。

### 3.2 public feasibility、语言自然化和 LLM 候选/仲裁（005–010）

| Iteration / Variant | 方法 | 结果 | 决策 |
|---|---|---|---|
| 005 | public feasibility identifiability audit | overall AUC 0.683、balanced acc 0.669；price-only negative recall 0.179 | 跨域 gate 失败，不接 planner |
| 006 / V7 | action-locked strategic naturalization | 5-task 42.26；但战略 preamble 启用 54% < 70%，CI 很宽 | 拒绝 |
| 007 / V8 | 受限 rhetorical-act label selector | 8-task×3：27.74 vs V4 41.56；`FIRM_BOUNDARY` 150/156，mismatch 50% | selector collapse，拒绝 |
| 008 / V9 | public-only LLM structured proposal candidate | 5-task 28.18 vs V4 42.94；49 个 distinct candidates 注入，planner 选择 0 次 | candidate/planner 不匹配，拒绝 |
| 009 / V10 | stalled-state belief-grounded candidate-ID arbitrator | -2.15 vs fresh V4 28.18；大量 override 导致 timeout/mismatch | 灾难退化，拒绝 |
| 010 / V11 | frozen-reference safe-improvement trust region | 与 V4 完全相同 28.18；消除 V10 灾难但增益 0 | 保留 safety shell，拒绝为改进 |

关键经验：LLM 输出合法的 candidate ID，不代表它学会了 calibrated planning；belief 或 prompt
仲裁器一旦高频覆盖 frozen planner，会把 offline 看似合理的候选变成 online timeout。

### 3.3 reward-labelled offline AWR/LCB（011–014）

| Iteration / Variant | 方法 | 结果 | 决策 |
|---|---|---|---|
| 011 / V12 | 524 条真实 trajectory，按 scenario split 的 conservative AWR residual | offline aggregate `+0.0319`，AgenticPay `+0.00838`；但 exact safe deployment 后约 `+0.00042` | offline signal 存在，V12 不上线 |
| 012 / V13 | 去掉与 LCB 冲突的旧 trust region，保留 OOD/LCB abstention | 5-task smoke 39.82 vs V4 28.18，但 Task2 退化、Task3 单点巨升 | 进入全集确认，不先宣布成功 |
| 013 / V14 | 无直接对手证据时 residual abstain | 解决 train/deploy 首轮 override shift 的候选机制 | 未形成超 baseline 的 AgenticPay 结论 |
| 014 / V13 all28 | 完整确认 | V13 17.98，direct 31.87，CoT 32.16，full 27.23；V13 mismatch 57.14% | 明确失败 |

V13 相对 direct 为 `-13.89`，task-cluster CI `[-29.67,+2.65]`；相对 CoT 为
`-14.19`，CI `[-31.55,+4.10]`。这证明 offline AWR surrogate improvement 没有转化为
AgenticPay online BuyerScore。

### 3.4 settlement validator 与 calibrated opening（015–021）

| Iteration / Variant | 方法 | 结果 | 决策 |
|---|---|---|---|
| 015 / V15 | 固定 70% contract settlement repair | 5-task 15.01 vs V13 16.89；Task7 修复，Task5 从 88.35 降到 40.09 | 过度让步，拒绝 |
| 016 / V16 | risk-adaptive settlement repair | 5-task 29.11；8-task×3 为 17.99 vs V13 10.36，但 direct 40.66、CoT 50.96；mismatch 75% | 局部改进，仍远低于 baseline |
| 017 / V17 | validated calibrated opening | 8-task 39.48 vs V16 17.99，CoT 42.49；mismatch 2/8 | 大幅接近 CoT，但 opening 被重复调用，gate 失败 |
| 018 / V18 | public counteroffer term validator | 3-task 11.20 vs V17 11.62；action flip 0 | 无 buyer-IR-aligned successor，拒绝 |
| 019 / V19 | minimal buyer-IR term repair | Task10 51.85 vs V17 35.74，timeout → agreement；但 validator flip 0 | 增益来自 opening bug fix，因果 gate 失败 |
| 020 / V20 | semantic low-burden endpoint opening | Task15/20 -0.30 vs V19 10.51；mismatch 2/2 | endpoint 方向猜错，拒绝 |
| 021 / V21 | public-only burden selector / information firewall | Task15/20 -0.37 vs V20 20.51；mismatch 2/2 | public wording不足以定方向，拒绝 |

关键经验：V17 说明“先生成完整且校验过的 opening contract”比单纯提高 belief/planner 复杂度
更有效；但从描述文本直接推断 provider burden 容易把 continuous direction 反过来。

### 3.5 独立、可拒绝的 issue classifier（022–028）

| Iteration / Variant | 方法 | 结果 | 决策 |
|---|---|---|---|
| 022 | 第一版 abstaining classifier | continuous coverage/accuracy 50%/50%；discrete 13.85%/0% | offline 拒绝 |
| 023 | evidence-decomposed classifier | continuous conditional acc 25%；discrete conditional acc 82.14% | 只部分可用 |
| 024 | ontological continuous direction + anonymous option burden | continuous 100%/100%；discrete coverage 87.69%、conditional acc 80.70% | offline gate 通过 |
| 025 / V25 | classifier-composed opening | Task15/20 -0.59，mismatch 2/2；CoT 26.70 | offline accuracy未转化为成交可行性 |
| 026 / V26 | cross-turn rejectable guard，seller 完整 counter 可覆盖 prior | Task15/20 9.64，mismatch 1/2；修复 Task20，Task15 仍失败 | 有真实局部作用，未过零 mismatch gate |
| 027 / V27 | V26 + 1% settlement uncertainty buffer | 8-task 28.88 vs V26 26.69；all28 35.18 vs CoT 32.26 | aggregate 正信号但 CI 跨 0、多 issue 为负 |
| 028 / V28 | 2% buffer sweep | 2-task stage 25.65，低于 V27 41.39 | 自动选择 V27 |

独立 classifier 的贡献是减少 unsupported term inversion，并给 planner 一个可拒绝 prior；它
没有识别 seller 的真实 utility magnitude。尤其当 seller 接受一个自身 benchmark utility 为负的
package 时，classifier 无法仅凭公开一句“接受”判断这是 LLM 行为错误还是有效证据。

### 3.6 semantic risk 与 active settlement frontier（029–031）

定向集合固定为 Task5/9/10/15/20/23，均是此前的 timeout、term inversion 或 mismatch
高风险任务。

| Variant | 6-task BuyerScore | DealRate | Timeout | Mismatch | Gate |
|---|---:|---:|---:|---:|---|
| CoT（V29 run） | 41.64 | 100% | 0 | 0 | reference |
| V27 | 27.92 | 83.33% | 1 | 1 | reference |
| V29 semantic-risk settlement | 34.30 | 66.67% | 2 | 1 | fail |
| V30 evidence-gated frontier | 28.18 | 100% | 0 | 1 | fail |
| V31 bounded compensated frontier | **37.59** | **100%** | **0** | 1 | fail（要求 mismatch=0） |

- V29 修复 Task10（相对 V27 `+79.98`），却破坏 Task23，并新增两个 timeout。
- V30 消除 timeout，Task20/23 改善，但在 Task15 选择了 buyer utility leverage 2.16× 的
  极端 package；hidden seller utility 仍为负，且 Task5/9 因过度 settlement 得分下降。
- V31 修复 Task15，修复 V29 的 Task23 mismatch，并把 Task9 timeout 变成 64.16 分成交；但
  Task20 的 active probe 同时改变多个高负担 issue 且价格跨过 seller offer，形成 hidden
  seller-IR violation；Task10 相对 V29 下降 14.69。

V31 相对 V29 的逐 task delta：

| Task | Delta | 解释 |
|---|---:|---|
| Task5 | -1.15 | timeout 变成交，但价格/条款让步抵消收益 |
| Task9 | +26.79 | timeout 变成可行 7-round deal |
| Task10 | -14.69 | 更保守的 frontier 降低原本高分 |
| Task15 | 0.00 | leverage cap 成功避免 V30 的极端错误，回到 frozen outcome |
| Task20 | -23.97 | active probe 同时切到 rush/extra 并跨价；合同兼容但 hidden seller utility = -4.58 |
| Task23 | +32.71 | 修复 V29 mismatch |

## 4. 失败模式总结

### 4.1 success 不等于 scorer-feasible agreement

最严重的问题是 protocol-compatible 不等于 bilateral-IR-feasible：环境根据公开合同字段和价格
判断是否成交，而 scorer 还会使用双方隐藏 utility 判断合同是否真正可行。Task20 V31 的最后一轮：

- buyer 提出 price 12.15、rush、extra condiments、strong match；
- seller 回复 price 10.08、**相同非价格条款**；
- 原生环境按 `_contracts_compatible` 在 seller price 10.08 成交；
- 重建官方合同效用：buyer `+6.87`、seller `-4.58`，所以 BuyerScore `-0.2985`。

旧 JSONL 的 `missing_agreed_contract` 来自 Task20 example summary 漏字段，并不是上述失败的
真实机制。新 instrumentation 和 viewer 已修复诊断。direct/CoT 有时 mismatch 更少的一个原因是
它们较少生成结构化的主动多 issue 跳变；framework 的 active probe 会同时改变多个条款，且若
价格在同轮跨过 seller offer，就没有机会再观察对方对各 issue 的可归因响应。

### 4.2 seller 自然语言不能被当作 hidden utility oracle

seller LLM 会同意 benchmark utility 为负的合同。public acceptance 是重要 evidence，但不是
seller IR 的真值标签。V5、V15、V16、V20、V25、V30 都出现过“seller 说可以，scorer 说不可行”。

### 4.3 multi-issue concession 没有共同货币

价格、continuous term 和 discrete term 的让步目前通过启发式 utility proxy 合并。classifier
能给方向，却不能估计 seller 对各 term 的成本 magnitude；active frontier 因而可能：

- 在一个 issue 上给太多，降低 BuyerScore；
- 在多个 buyer-favorable issues 上同时索取，仍不足以补偿 seller；
- 用价格补偿了错误方向的 term；
- 在本应探索的 turn 直接价格交叉，提前形成 hidden-IR-infeasible package。

### 4.4 frozen planner、validator 与新机制互相覆盖

多个版本不是单一 planner 的平滑升级，而是 candidate、residual planner、opening override、term
repair、settlement validator 依次叠加。一个局部修复可能重新覆盖另一个已验证边界，例如：

- V17 opening 重复执行；
- V27 classifier profile 被 legacy minimal term repair 改写；
- V29 修复 Task10，却在 Task23 重新产生 mismatch；
- V31 ordinary offer 与 terminal floor 分开后，active probe 可能在进入 terminal floor 前自动成交。

### 4.5 offline 可识别性与 online reward 有明显 gap

V24 classifier 的 offline accuracy 通过，V11 AWR 的 offline LCB 也为正，但 V25/V13 online
分别失败。这不是“训练完全无效”，而是训练 label 没有覆盖真正的 terminal failure：
seller LLM/scorer disagreement、探索 offer 的同轮价格交叉、deadline 和 multi-issue
package-level feasibility。

### 4.6 小子集过拟合风险

Task15/20 与 6-task targeted set 被重复用于提出、筛选和验收规则。V31 的 37.59 只能视为
development evidence；没有 all-28 confirmation 不能作为最终 benchmark 数字。

## 5. Sample trajectories

已导出三份完整可编辑 Markdown：

1. `SAMPLE_TRAJECTORY_TASK20_V31_MISMATCH.md`：最新失败；保留 raw
   `missing_agreed_contract`，并新增按未修改上游规则重建的合同与 utility audit，明确显示
   buyer utility `+6.87`、seller utility `-4.58`。
2. `SAMPLE_TRAJECTORY_TASK9_V31_SUCCESS.md`：V31 把 V29 timeout 修复为 7-round 有效成交，
   BuyerScore 64.16。
3. `SAMPLE_TRAJECTORY_TASK7_V16_REPAIR_SUCCESS.md`：risk-adaptive settlement validator 将
   V13 的 mismatch 修复为 BuyerScore 60.61，用于观察“局部 validator 确实有用”的正例。

这些文件中的 seller private utility 标注只用于事后分析，不是 deployed buyer 输入。

## 6. 自己查看任意 task 的完整 trajectory

新工具：

`/work5/qixint/negotiation/tools/view_agenticpay_trajectory.py`

它是只读、纯 Python 标准库工具，不调用模型，不重算或修改 benchmark score。

### 6.1 列出一个 run/iteration 中所有 episode

```bash
python /work5/qixint/negotiation/tools/view_agenticpay_trajectory.py \
  --input /work5/qixint/universal_reward_research/iteration_031_bounded_compensated_active_frontier \
  --list
```

列表包含 task、variant、BuyerScore、deal、mismatch、termination、rounds 和原始 JSONL 行号。

### 6.2 查看某个 task 的完整内容

```bash
python /work5/qixint/negotiation/tools/view_agenticpay_trajectory.py \
  --input /work5/qixint/universal_reward_research/iteration_031_bounded_compensated_active_frontier \
  --task Task20 --variant v31 --include-framework-trace
```

输出包括：

- 环境信息、商品信息、用户 requirement；
- 完整 public contract construct/schema；
- buyer/seller utility ground truth（明确标为 post-hoc）；
- 每一轮 buyer 与 seller 的原始文本；
- 每轮抽取价格、format valid、step reward 与 termination；
- agreed contract、final price、BuyerScore、DealRate 对应状态；
- contract feasibility、IR/mismatch reason；
- planner/validator 选择摘要。

### 6.3 导出为 Markdown

```bash
python /work5/qixint/negotiation/tools/view_agenticpay_trajectory.py \
  --input /path/to/run_or_iteration \
  --task Task9 --variant v31 --include-framework-trace \
  --output task9_v31.md
```

可选参数：

- `--hide-private-utility`：分享给外部时隐藏双方 benchmark utility truth；
- `--full-framework-trace`：输出所有 ranked candidates，文件可能很大；
- `--index N`：同一过滤条件有多个 episode 时选择第 N 条。

库内原有的轻量控制台 reader 仍在
`negotiation/AgenticPay_Env/tools/read_trajectories.py`；新工具更适合生成完整、可分享的 Markdown。

## 7. 下一步建议：先修不变量，再优化策略

### 7.1 必须先加入 non-crossing active-probe validator

下一版只修改一个探索/结算边界：

1. active probe 一次只改变一个 continuous/discrete issue，其余条款与公开 reference 保持不变；
2. probe price 必须严格低于最新 seller price（留一个可审计 epsilon），防止 probe 当轮自动成交；
3. probe 只收集 seller accept/counter/reject evidence，不得作为 terminal settlement candidate；
4. 下一轮若进入结算，恢复完整 semantic-risk settlement buffer，并保留 buyer IR reserve；
5. trace 写入 changed issue、non-crossing margin、seller response 与 terminal/probe phase。

这比继续调 CEM/AWR 轮数更优先，因为 Task20 的 -0.2985 是 probe/settlement phase 混淆；
belief 尚未获得可归因 evidence，planner 就已经成交。

### 7.2 将 issue classifier 限定为 prior，不直接决定 terminal contract

- classifier 只产生低负担候选和 uncertainty；
- seller 完整 counter 是 direct behavioral evidence，但不是 hidden IR oracle；
- 没有 repeated/contrastive response 时，不允许 classifier 同时翻转多个 issue；
- active probe 每次只改变一个 issue，并保持价格不跨越 seller offer，以获得可归因 evidence；
- terminal acceptance 只复制已验证 proposal，不再执行 probe。

### 7.3 重新构造训练数据

训练 label 应从“episode 总分粗分配给所有 turn”升级为 proposal-level labels：

- exact contract accepted/rejected/ignored；
- contract hash 是否一致；
- 单 issue change 后 seller response；
- timeout 前的可行 buyer proposal；
- scorer mismatch 类型；
- native 与 fixed-seller correction 的差异。

优先训练三个可拒绝模块：`issue response model`、`accept/continue boundary`、`concession
frontier residual`。训练目标仍是 BuyerScore，但对 probe-phase price crossing 设置硬 mask，
而不是用 reward 软惩罚。

### 7.4 正式验证协议

1. 冻结 native seller、模型 snapshot、prompts、scorer 和 task files；
2. development 使用 task-held-out split，不能反复只看 Task15/20；
3. 先跑 protocol gate：0 error、0 action mismatch、0 unexplained score-success mismatch，且所有
   active probe 满足 one-issue/non-crossing 不变量；
4. 再跑 all 28 tasks；显式启用可复现的非零 temperature/API seed，确认 rollout 真有差异；
5. 每个 task 作为 cluster，报告 mean paired delta、95% cluster bootstrap CI、win rate、DealRate、
   TimeoutRate 和 mismatch；
6. 同时报告 price-only 3 与 contract 25 子集，防止简单任务掩盖 multi-issue 退化；
7. 只有相对 direct 和 CoT 的 all-28 下界不低于 0、且 contract25 不退化，才能声称稳定提升。

## 8. 可追溯源文件

- 不可覆盖流水线账本：`/work5/qixint/universal_reward_research/PIPELINE_LEDGER_CN.md`
- Iteration 014 全集：`iteration_014_agenticpay_v13_full_confirmatory/FULL28_R3_METRICS.json`
- Iteration 016 8-task：`iteration_016_risk_adaptive_settlement_validator/metrics/STAGE2_METRICS.json`
- Iteration 027 all28：`iteration_027_028_settlement_buffer_sweep/metrics/STAGE3_ALL28_3SEED.json`
- Iteration 030 failure audit：`iteration_030_evidence_gated_settlement_frontier/V30_TARGETED_FAILURE_ANALYSIS_CN.md`
- Iteration 031 最新 gate：`iteration_031_bounded_compensated_active_frontier/metrics/TARGETED_GATE.json`
- 每版方法假设、代码 snapshot、原始 runs 和 gate 均保存在对应 `iteration_NNN_*` 目录；失败版本没有覆盖。

## 9. 当前最终判断

AgenticPay 上做过的尝试已经足够说明，问题不是“再加一个更复杂的 belief prompt”即可解决。
框架最可靠的进展是建立了环境无关的 belief → candidate → planner → locked action 分层，并逐步
发现、隔离了 action semantics、terminal accept、opening、issue direction 与 settlement risk。
但最终 BuyerScore 仍受 seller-behavior/scorer disagreement 和 multi-issue causal identifiability 限制。

在解决 active probe 意外成交与 multi-issue evidence attribution 以前，V31 的 37.59/6-task 与
V27 的 35.18/all28 都只能作为下一版候选证据，不能作为“已经超过 CoT”的论文结论。

## 10. 2026-08-23 后续执行状态（不计入已完成结果）

依据第7节已实现并预注册 V32
`universal_framework_v32_noncrossing_single_issue_frontier`：one-shot、one-issue、严格
non-crossing probe，probe 后结算恢复完整 semantic-risk floor。53条相关回归测试通过，代码和
协议快照位于
`/work5/qixint/universal_reward_research/iteration_032_noncrossing_single_issue_frontier/`。

V32 尚无 online 模型结果，因此没有加入第3节的已完成实验表，也不能据此更新最终效果判断。
当前8002模型服务无响应；低负载 watcher 已等待服务恢复，先执行 fresh V31/V32 targeted gate，
通过后才会执行 all-28 多 seed 确认。
