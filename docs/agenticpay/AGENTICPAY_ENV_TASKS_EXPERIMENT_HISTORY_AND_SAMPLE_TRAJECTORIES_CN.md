# AgenticPay Single28：环境、任务、历次实验结果与 Sample Trajectories

更新时间：2026-08-25
实验边界：只修改 buyer；固定 Qwen3-30B native seller；不修改 AgenticPay task、环境状态机、
contract parser 或官方 scorer。
主要指标：官方 AgenticPay `BuyerScore`；DealRate、TimeoutRate、score-success mismatch 用于解释。

当前运行状态（2026-08-25）：Qwen3-30B-A3B-Instruct-2507-base 已在 GPU 0/1 以 TP=2、
port 8002 重启并验证；Iteration 032、033 targeted gates 均已完成，模型服务继续保留供下一轮使用。

## 0. 结论与“是否都在 all-28 上测试”

不是所有实验都在 all-28 上进行。实际采用三层协议：

| 层级 | 典型规模 | 用途 | 能否作为最终 benchmark 结论 |
|---|---:|---|---|
| Smoke / mechanism reachability | 2–5 tasks，通常1 seed | 检查代码、protocol、validator 是否真的触发 | 不能 |
| Targeted development gate | 6–8 个固定困难 tasks | 分析 timeout、mismatch、term inversion 等已知失败 | 不能，只是开发证据 |
| All-28 confirmation | 28 tasks，名义上3 CLI seeds | 与 Direct/CoT 做完整 task-cluster paired 比较 | 可以，但必须审计 seed 是否真有独立轨迹 |

已完成的主要 all-28 实验包括：

1. Iteration 000：Direct、CoT、旧 full framework、Universal V1、V2；
2. Iteration 014：Direct、CoT、旧 full framework、V4、V13；
3. Iteration 027：CoT 与 V27；
4. Iteration 032：fresh targeted V31/V32 gate 已完成；V32 未通过，因此按预注册协议跳过 all-28。

因此，V29、V30、V31 的 6-task 数字不能与 all-28 均值直接横向比较；V31 的 `37.59` 不是
全28结果。当前最准确的结论仍是：V27 在 all-28 上表面达到 `35.18`，高于同次 CoT 的
`32.26`，但 task-cluster bootstrap CI 跨0，且增益来自3个 price-only tasks；25个真正
multi-issue tasks 上相对 CoT 仍为负。

## 1. 环境是什么

### 1.1 交互设置

Single28 是单 buyer、单 seller、单商品/服务的多轮自然语言谈判：

- 双方轮流生成自然语言消息；
- price-only task 用价格标签表达报价；
- contract task 必须在 `<contract>...</contract>` 中提交完整 JSON 合同；
- 默认最多20轮；越晚成交，`gamma=0.99` 的时间折扣越大；
- buyer 看见自己的最大价值和条款效用，但看不到 seller 私有成本/条款权重；
- seller 看见自己的成本和条款效用，但看不到 buyer 私有权重；
- 双方都能看到 public product、user requirement、合同字段、连续边界与离散候选值。

当前 framework 只替换 buyer 的决策路径；seller 保持仓库原生 `SellerAgent` 及原 prompt。

### 1.2 Price-only 与 contract/multi-issue

28个任务中：

- 3个 price-only：只谈一个价格；
- 25个 contract/multi-issue：同时谈价格和2–4个非价格 issue。

在 contract mode 中，buyer 和 seller 的原始效用为：

```text
U_b = v_base - price + Σ(w_b,continuous × value) + Σw_b,discrete(option)
U_s = price - c_base + Σ(w_s,continuous × value) + Σw_s,discrete(option)
```

某些表面上“对 buyer 好”的条款会给 seller 带来很大隐藏履约成本，所以价格在
`[seller_min, buyer_max]` 内并不足以保证合同可行。

### 1.3 合同何时在环境中成交

buyer 与 seller 的两份合同必须：

1. `continuous_terms` 完全相同；
2. `discrete_terms` 完全相同；
3. 价格差在 tolerance 内，或 `seller_price <= buyer_price`。

第3条意味着 buyer 的探索报价如果跨过最新 seller price，可能在同轮立即成交。V31 Task20
正是因此把“probe”错误变成了 terminal settlement。

### 1.4 BuyerScore

令 `discount = gamma^(t-1)`，默认 `Db=10, Wb=80, Eb=10, Fb=15`。

Price-only 有效成交：

```text
u_b = (buyer_max_price - final_price) /
      (buyer_max_price - seller_min_price)
BuyerScore = discount × (Db + Wb × u_b + Eb)
```

Contract 有效成交：

```text
r_b = U_b / Z_max
BuyerScore = discount × (Db + Wb × r_b + Eb)
```

其中完整合同必须 schema 合法、`U_b >= 0` 且 `U_s >= 0`。否则即使语言状态写着 agreed，
scorer 仍返回失败分：

```text
BuyerScore = -Fb × (1 - discount)
```

这就是 `score-success mismatch`：环境对话/状态看似成交，但官方 contract utility 判定合同不可行。

### 1.5 为什么某些 timeout 仍有正 BuyerScore

研究 runner 同时保存 native score 和 `buyer_score_fixed_seller`。后者只用于如下明确审计：native
seller 在公开历史中已经收到可行 buyer proposal，却没有按协议完成接受。聚合时有该 correction
才使用，否则使用 native BuyerScore；同时必须单独报告 DealRate、TimeoutRate 和 mismatch，
不能把 correction 当成环境原生成交。

## 2. 28个任务

Task14 的源文件历史上误命名为 `Task4_s1_taxi_ride_negotiation.py`，但 runner 中的顺序和本报告
编号仍为第14项。

| # | Task | 场景 | 模式 | 非价格 issues | B / C |
|---:|---|---|---|---|---:|
| 1 | Basic price | 冬季夹克基础议价 | price-only | — | 150 / 80 |
| 2 | Close price | 极窄价格区间 | price-only | — | 85 / 80 |
| 3 | Close-to-market | 接近市场价议价 | price-only | — | 200 / 175 |
| 4 | Beauty product | Maybelline 眼影 | contract | delivery days, return, packaging, match | 6.24 / 5.11 |
| 5 | Toothpaste | ARM & HAMMER 牙膏 | contract | delivery days, return, seal, match | 12.40 / 10.15 |
| 6 | Riflescope | Crimson Trace 瞄准镜 | contract | delivery days, return, accessory, match | 171 / 142 |
| 7 | Headphones | 儿童无线耳机 | contract | delivery days, return, volume limit, match | 11.63 / 9.57 |
| 8 | Wall lantern | 户外壁灯 | contract | delivery days, return, glass finish, match | 47.35 / 39.40 |
| 9 | Bookshelf | 四层书架 | contract | delivery days, return, hardware, match | 28.78 / 23.75 |
| 10 | Sandals | 男士人字拖 | contract | delivery days, return, size/color, match | 13.85 / 11.45 |
| 11 | Jeans | 女士牛仔裤 | contract | delivery days, return, size/color flexibility, match | 17.75 / 14.75 |
| 12 | Beverage | Elderflower Rose 饮料 | contract | delivery days, return, freshness packing, match | 24.70 / 20.35 |
| 13 | Food color | AmeriColor 喷雾色素 | contract | delivery days, return, color seal, match | 4.88 / 4.02 |
| 14 | Taxi 1 | Gramercy → Murray Hill | contract | wait time, route, match | 10.36 / 8.67 |
| 15 | Taxi 2 | Union Sq → Lenox Hill West | contract | wait time, route, match | 16.65 / 13.88 |
| 16 | Taxi 3 | LaGuardia → East Chelsea | contract | wait time, route, match | 54.45 / 45.55 |
| 17 | Taxi 4 | Seaport → Battery Park City | contract | wait time, route, match | 17.90 / 14.95 |
| 18 | Taxi 5 | West Village → Sutton Place | contract | wait time, route, match | 25.75 / 21.45 |
| 19 | Food delivery 1 | Izakaya Karaage Chicken | contract | speed, condiments, match | 10.35 / 8.54 |
| 20 | Food delivery 2 | Sticky's sliders & fries | contract | speed, condiments, match | 12.15 / 10.08 |
| 21 | Food delivery 3 | BunSlut fries | contract | speed, condiments, match | 10.90 / 9.06 |
| 22 | Food delivery 4 | Dripped Birria nachos | contract | speed, condiments, match | 10.61 / 8.80 |
| 23 | Food delivery 5 | Sprite canned drink | contract | speed, condiments, match | 4.52 / 3.75 |
| 24 | Rent 1 | NYC East Village studio | contract | lease months, utilities, match | 1092 / 900 |
| 25 | Rent 2 | Barcelona private room | contract | lease months, utilities, match | 1245 / 1030 |
| 26 | Rent 3 | Rio private suite | contract | lease months, utilities, match | 1540 / 1272 |
| 27 | Rent 4 | Bondi 2BR apartment | contract | lease months, utilities, match | 2095 / 1750 |
| 28 | Rent 5 | Newport beach house | contract | lease months, utilities, match | 7510 / 6250 |

`B/C` 是任务中的 buyer base value / seller base cost；contract 真实 reservation 还受条款权重影响。

## 3. 我们如何运行与记录

统一路径：

```text
public history + buyer private utility
  -> canonical state
  -> continuous opponent belief update
  -> complete contract candidate generation
  -> belief-usable planner
  -> settlement/probe validator
  -> action-locked natural-language renderer
  -> unchanged AgenticPay parser/environment/scorer
```

主要代码：

| 内容 | 路径 |
|---|---|
| Universal AgenticPay buyer/adapter | `negotiation/AgenticPay_Env/buyer/universal_framework.py` |
| 独立、可拒绝 issue classifier | `negotiation/AgenticPay_Env/buyer/issue_classifier.py` |
| Variant registry | `negotiation/AgenticPay_Env/buyer/variants.py` |
| Single28 runner及完整落盘 | `negotiation/experiments/run_agenticpay_single28_framework.py` |
| 原生环境与 scorer | `negotiation/benchmarks/AgenticPay/agenticpay/envs/single_buyer_product_seller/Task1_basic_price_negotiation.py` |
| 完整 trajectory viewer | `negotiation/tools/view_agenticpay_trajectory.py` |

每个 episode 保存环境 construct、双方逐轮原文、抽取合同/价格、framework trace、最终合同、官方
BuyerScore、DealRate、termination、mismatch 和 post-hoc utility audit。seller private utility 只在
实验结束后用于诊断，从不提供给 deployed buyer。

## 4. 历次实验结果

### 4.1 第一次完整 all-28 基线：Iteration 000

28 tasks × 名义3 seeds；按 task cluster 统计：

| Buyer | Mean BuyerScore | 相对 CoT | 判断 |
|---|---:|---:|---|
| Direct prompt | **32.95** | +1.23 | 强基线 |
| CoT prompt | 31.72 | — | 强基线 |
| Historical full framework | 21.31 | -10.41 | 失败 |
| Universal V1 | 15.73 | -15.99，CI `[-26.96,-5.83]` | 明确失败 |
| V2 conservative AWR | 18.18 | -13.54，CI `[-25.87,-1.78]` | 明确失败 |

早期 framework 的问题不是 belief 没有变化，而是 action rendering、terminal acceptance 和完整
contract semantics 没有锁住，造成大量“planner 选A、文本说B”和无效合同。

### 4.2 协议、候选和 planner 修复：V3–V14

| Iteration / Variant | 测试范围 | 主要变化 | 结果与决定 |
|---|---|---|---|
| 001 / V3 | 8 tasks×3 | action-consistent renderer | action mismatch 158/177 → 0/329，但 BuyerScore 33.13 < Direct 50.08 |
| 002 / V4 | 8 tasks×3 | proposal-backed terminal guard | 41.56，接近 CoT 45.96；保留 frozen reference |
| 003–006 / V5–V7 | 3–5 tasks | public feasibility/frontier、自然化 | 局部有效但不稳定，拒绝 |
| 007 / V8 | 8 tasks×3 | 修辞标签 selector | 27.74；150/156 塌缩为同一标签，拒绝 |
| 008 / V9 | 5 tasks | LLM structured proposal candidate | 49个候选注入但 planner 选择0次，拒绝 |
| 009 / V10 | 5 tasks | belief-grounded candidate arbitrator | -2.15，大量 timeout/mismatch，灾难退化 |
| 010 / V11 | 5 tasks | frozen-reference trust region | 与 V4 完全相同，安全但增益0 |
| 011–013 / V12–V14 | offline + 5-task smoke | reward-labelled AWR/LCB、evidence gate | offline有小正信号；online高度异质 |

Iteration 014 对 V13 做了完整 all-28：

| Buyer | Mean BuyerScore | DealRate | TimeoutRate | Mismatch |
|---|---:|---:|---:|---:|
| Direct | 31.87 | 82.14% | 17.86% | 21.43% |
| CoT | **32.16** | 85.71% | 14.29% | 25.00% |
| Historical full | 27.23 | **92.86%** | **7.14%** | 39.29% |
| V4 | 15.59 | 82.14% | 17.86% | 50.00% |
| V13 AWR/LCB | 17.98 | 85.71% | 14.29% | 57.14% |

V13 相对 CoT 为 `-14.19`，task-cluster CI `[-31.55,+4.10]`。offline advantage 并未转化为
online BuyerScore，主要因为训练标签把 episode reward 粗略分配给多个 turn，而且部署时首轮
override 分布与训练不同。

### 4.3 Settlement、opening 与 issue classifier：V15–V28

| Variant | 范围 | 尝试 | 主要结果 |
|---|---|---|---|
| V15 | 5 tasks | 固定70% settlement repair | 15.01，过度让步 |
| V16 | 8 tasks×3 | risk-adaptive repair | 17.99，低于 Direct/CoT，mismatch 75% |
| V17 | 8 tasks | validated calibrated opening | 39.48，接近 CoT 42.49；但 opening 会重复触发 |
| V18–V21 | 2–3 tasks | public counter validator、minimal repair、semantic endpoint | 连续方向常推反，未晋级 |
| V22–V24 | offline 28 task schemas | 可拒绝 issue classifier | V24 continuous 100%，discrete conditional accuracy 80.70% |
| V25–V26 | Task15/20 | classifier opening/guard | V26 修复一个 task，但仍有 mismatch |
| V27 | 8-task gate → all-28 | cross-turn guard + 1% buffer | all-28 35.18 vs CoT 32.26；进入统计审计 |
| V28 | 2 tasks | 2% buffer sweep | 低于 V27，拒绝 |

V27 all-28 的完整结果：

| Buyer | Mean BuyerScore | DealRate | Mismatch |
|---|---:|---:|---:|
| CoT | 32.26 | **89.29%** | 28.57% |
| V27 | **35.18** | 85.71% | **17.86%** |

表面增益为 `+2.92`，但：

- task-cluster bootstrap CI `[-11.55,+17.49]`，不能排除无提升；
- 三个CLI seed的完整轨迹完全相同，实际只有28个 task clusters，不是84个独立样本；
- price-only 3：V27-CoT `+42.94`；
- contract 25：V27-CoT `-1.88`，CI `[-15.82,+11.03]`。

所以 V27 证明了 framework 在简单价格谈判上可能很强，但没有证明 multi-issue universal gain。

### 4.4 困难任务 targeted frontier：V29–V31

固定 Task5/9/10/15/20/23，只用于开发 gate：

| Variant | BuyerScore | DealRate | Timeout | Mismatch | Gate |
|---|---:|---:|---:|---:|---|
| CoT reference | 41.64 | 100% | 0 | 0 | reference |
| V27 | 27.92 | 83.33% | 1 | 1 | reference |
| V29 semantic-risk settlement | 34.30 | 66.67% | 2 | 1 | fail |
| V30 active frontier | 28.18 | 100% | 0 | 1 | fail |
| V31 bounded compensated frontier | **37.59** | **100%** | **0** | 1 | fail |

V31 的正面作用是把 Task9 timeout 变成 BuyerScore `64.16` 的7轮有效成交，并修复 Task23；
负面作用是 Task20 的 active probe 同时改变 rush/extra-condiments 并跨过 seller price，环境立即
settle。最终 buyer utility `+6.87`、seller utility `-4.58`，官方 BuyerScore `-0.2985`。

### 4.5 V32 fresh paired 验证结果

V32 只针对上述可验证失败做一项机制升级：

- probe 一次只改变一个 issue；
- continuous 只走 midpoint；
- probe price 严格低于最新 seller price，防止探索轮自动成交；
- 每局最多一个 probe；
- 下一轮结算恢复完整 semantic-risk floor。

2026-08-25 在 GPU 0/1、同一 Qwen3-30B native seller、同一 CLI seed 下 fresh 运行
V31/V32 × 6 targeted tasks，结果如下：

| Variant | Mean BuyerScore | DealRate | Timeout | Mismatch | Error |
|---|---:|---:|---:|---:|---:|
| Fresh V31 | 36.22 | **83.33%** | **1** | 1 | 0 |
| V32 | **40.23** | 50.00% | 3 | 1 | 0 |

V32 三次实际 probe 均满足 `one changed field`、`information_only`、price 严格低于 seller
anchor 和 margin≥0.01，说明机制实现正确；Task20 的 V31 mismatch 也被消除。但 V32 仅3/6
实际成交，Task5、Task10、Task20 timeout，Task23 又产生 mismatch。因此 gate 明确失败，流水线
以 exit 32 停止，并按预注册协议没有运行 all-28。

逐 task 配对结果揭示了平均分掩盖的问题：V32 在 Task5、Task20 timeout 时因 fixed-seller
post-hoc correction 仍分别计 69.90、30.15；这使平均 BuyerScore 高于 V31，却不能替代真实
DealRate。核心 failure 是 **post-probe deadlock**：probe 成功获得 seller 的完整公开反报价后，
planner 没有复制/接受该 buyer-IR 合同，而是在 seller anchor 以下一分钱或在不兼容条款间循环。

### 4.6 V33 public-response settlement latch

V33 修复 contraction acceptance parser，并在 probe 后强制接管完整、buyer-IR 的 seller public
counter。Fresh V32/V33 六任务结果：

| Variant | Native BuyerScore | 含 fixed-seller correction | DealRate | Timeout | Mismatch |
|---|---:|---:|---:|---:|---:|
| V32 | 22.83 | 47.53 | 50% | 3 | 0 |
| V33 | **24.53** | **54.66** | 50% | 3 | 0 |

Task20 latch 实际触发并有效成交；Task5 seller counter 不满足 buyer IR，因此安全 abstain；Task23
本次没有 grounded explicit acceptance。V33 仍有3 timeout，gate失败，未运行 all-28。该结果也
促使后续统一以官方 native BuyerScore 为主，fixed-seller correction 只作诊断。

## 5. Sample trajectories

### 5.1 all-28 中 V27 的 price-only 成功

Task3：

| Variant | Outcome | BuyerScore | 说明 |
|---|---|---:|---|
| CoT | timeout | -2.73 | 在窄区间持续谈判，未完成成交 |
| V27 | agreed @ 177 | **91.74** | 快速找到接近 seller cost 的可行价 |

完整轨迹：

- `8.23/SAMPLE_TRAJECTORY_TASK3_COT_ALL28.md`
- `8.23/SAMPLE_TRAJECTORY_TASK3_V27_ALL28.md`

这是 V27 aggregate 高于 CoT 的主要类型，也说明不能仅凭总均值证明 multi-issue belief 有效。

### 5.2 all-28 中 V27 的 multi-issue 失败

Task10 sandals：

| Variant | Outcome | BuyerScore | 说明 |
|---|---|---:|---|
| CoT | valid agreement @ 13.75 | **86.89** | 完整合同对双方可行 |
| V27 | agreed @ 13.03 | -1.97 | 表面价格更低，但条款导致隐藏 seller IR 失败 |

完整轨迹：

- `8.23/SAMPLE_TRAJECTORY_TASK10_COT_ALL28.md`
- `8.23/SAMPLE_TRAJECTORY_TASK10_V27_ALL28.md`

该例直接说明 AgenticPay 的难点不是“越低价越好”，而是多 issue 合同要同时满足双方隐藏效用。

### 5.3 V31 的 targeted 正例与反例

- `SAMPLE_TRAJECTORY_TASK9_V31_SUCCESS.md`：从 timeout 修复为7轮有效成交，BuyerScore 64.16；
- `SAMPLE_TRAJECTORY_TASK20_V31_MISMATCH.md`：probe 意外跨价成交，seller utility -4.58；
- `SAMPLE_TRAJECTORY_TASK7_V16_REPAIR_SUCCESS.md`：早期 validator 将 mismatch 修成 BuyerScore 60.61。

### 5.4 V32 的 one-issue probe 与新失败

- `SAMPLE_TRAJECTORY_TASK5_V32_NONCROSSING_DEADLOCK.md`：只改变 return policy、价格10.14低于
  seller 10.15，probe 不变量全部成立；随后未接管 seller 反报价，最终 timeout。
- `SAMPLE_TRAJECTORY_TASK20_V32_PROBE_DEADLOCK.md`：只改变 delivery speed 的 probe 消除了
  V31 的意外成交/mismatch，但 buyer 反复报10.49 + rush，seller 坚持10.50 + batched，timeout。
- `SAMPLE_TRAJECTORY_TASK23_V32_MISMATCH.md`：实际成交但 hidden seller IR 失败，说明单议题 probe
  并没有自动解决 terminal settlement safety。

### 5.5 V33 的 latch 正例与边界

- `SAMPLE_TRAJECTORY_TASK20_V33_PUBLIC_COUNTER_LATCH_SUCCESS.md`：public counter latch 实际触发；
- `SAMPLE_TRAJECTORY_TASK5_V33_IR_BLOCKED_LATCH.md`：counter 对 buyer 非IR，validator安全拒绝；
- `SAMPLE_TRAJECTORY_TASK23_V33_NO_ACCEPTANCE_TIMEOUT.md`：没有 grounded acceptance，不能误触发。

## 6. 自己查看任意完整 trajectory

列出一个 run 的所有 task：

```bash
python /work5/qixint/negotiation/tools/view_agenticpay_trajectory.py \
  --input /work5/qixint/universal_reward_research/iteration_027_028_settlement_buffer_sweep/runs/stage3_all28_seed_20260828 \
  --list
```

查看 Task10 V27，包括环境、construct、双方逐轮原文、最终合同、price、score 和 framework trace：

```bash
python /work5/qixint/negotiation/tools/view_agenticpay_trajectory.py \
  --input /work5/qixint/universal_reward_research/iteration_027_028_settlement_buffer_sweep/runs/stage3_all28_seed_20260828 \
  --task Task10 --variant v27 --include-framework-trace
```

分享给外部时可增加 `--hide-private-utility`。viewer 是只读工具，不会重算或覆盖官方分数。

## 7. 当前主要问题

1. **Natural-language seller 与 hidden utility scorer 不一致。** seller 可能语言上接受一个自身
   benchmark utility 为负的合同；buyer 不能把一句“可以”当作 hidden IR oracle。
2. **Issue identifiability 不足。** 同时改变价格和多个 issue 后，只看到 accept/reject，无法判断
   哪个条款导致响应。
3. **Probe/settlement phase 混淆。** 探索报价一旦跨价，环境会直接成交，terminal validator
   没有第二次介入机会。
4. **不同 issue 缺少可靠共同货币。** buyer utility 已知，但 seller burden 只有方向和弱证据，
   无法安全决定“加多少钱补偿条款变化”。
5. **Offline-online gap。** classifier conditional accuracy 或 AWR offline LCB 为正，并不保证
   online BuyerScore 提升。
6. **小子集过拟合。** Task15/20 和6-task targeted set 已被反复使用，只能作为 development，
   不能替代 all-28。
7. **Seed 独立性。** 当前 native buyer/seller 调用 temperature=0；CLI seed 重复可能完全相同。
   正式统计单位必须是 task cluster，除非显式引入并记录非零 sampling。

## 8. 结果文件与可追溯性

- 首次 all-28：`universal_reward_research/iteration_000_full_baseline/analysis/primary_metrics.json`
- V13 all-28：`iteration_014_agenticpay_v13_full_confirmatory/FULL28_R3_METRICS.json`
- V27 all-28：`iteration_027_028_settlement_buffer_sweep/metrics/STAGE3_ALL28_3SEED.json`
- V29 gate：`iteration_029_semantic_risk_settlement/metrics/TARGETED_GATE.json`
- V30 gate：`iteration_030_evidence_gated_settlement_frontier/metrics/TARGETED_GATE.json`
- V31 gate：`iteration_031_bounded_compensated_active_frontier/metrics/TARGETED_GATE.json`
- V32协议/代码SHA：`iteration_032_noncrossing_single_issue_frontier/`
- V32 fresh gate：`iteration_032_noncrossing_single_issue_frontier/metrics/TARGETED_GATE.json`
- V33 fresh gate：`iteration_033_public_response_settlement_latch/metrics/TARGETED_GATE.json`
- 不可覆盖流水线账本：`universal_reward_research/PIPELINE_LEDGER_CN.md`

## 9. 当前判断

AgenticPay 对本研究有价值，因为它包含自然语言、己方可见 utility、对手隐藏 multi-issue utility、
完整合同和明确 reward；它比只谈单一价格的环境更能暴露 belief-to-action 的失败。

但现有结果还不足以声称 universal framework 胜过 prompt baseline。最有希望的证据是 V27
all-28 `+2.92` aggregate，以及 V32 targeted BuyerScore 40.23 高于 fresh V31 36.22；最关键的
反证是 V27 contract-25 `-1.88`、V32仅50% DealRate，以及多次 hidden seller-IR mismatch。
下一版不能继续放大 probe，而应加入仅依赖公开证据的 post-probe settlement latch。论文可成立的
最低证据应是：后续版本在
all-28 上相对 CoT 的 task-cluster CI 下界不低于0，contract-25 不退化，且 mismatch、DealRate、
TimeoutRate 同时不差。
