# Negotiation Belief–Planner：四个 Benchmark 的代码、结果与全集覆盖审计

更新时间：2026-09-04
范围：CaSiNo、NegotiationArena、Simple/AmazonHistoryPrice、AgenticPay
统一实验原则：只替换 focal buyer/agent；对手、环境、parser 和官方 scorer 保持不变；环境原生 reward/BuyerScore 是主结果，belief accuracy、calibration、action flip、mismatch 等只作为解释与因果诊断。当前 AgenticPay multi-buyer runner 是已明确标注的例外：它会替换 task 内所有 buyer instances，后续必须补 focal-only 版本。

---

## 0. 一页结论

### 0.1 当前最可信的结论

1. **CaSiNo**：在官方 100 条 test split 上，belief-usable candidate planner 的 P1 分数稳定高于 Direct；两次 100 局运行中，LLM chooser 的 `AvgScoreAll(P1)` 为 `22.30/22.17`，Direct 为 `18.73/17.40`。但 belief top-1 只有 `0.40/0.38`，因此结果更能证明“结构化候选与 action lock 有用”，还不能证明高质量的 continuous belief 是增益主因。
2. **NegotiationArena**：没有一个 framework 在四个 focal setting 全胜。旧主矩阵中 V3.3 在 Buyer 和 Resource-second 超过 Opponent Simulation，分别 `+0.35` 和 `+10.59`；Seller 和 Resource-first 分别落后 `−2.18`、`−22.12`。V8 最有研究价值的是 factorized belief、reliability gate 和因果干预协议，而不是 universal payoff SOTA。
3. **Simple/AmazonHistoryPrice**：这是目前最扎实的正结果。完整 128-scenario test manifest、每场景 3 rollouts 上，V6.3 reward `0.5130`，高于 CoT `0.4065`（`+0.1066`，95% CI `[+0.0581,+0.1560]`）和 historical full framework `0.4464`（`+0.0666`，CI `[+0.0203,+0.1118]`）。但 V6.3 相对 V5.9 的 `+0.0149` CI 跨 0，所以应把 V6.3 称为最高 point estimate，把 V5.9/V6.3 共同视为稳定候选。
4. **AgenticPay multi-seller protocol migration**：旧 V37 的 all-231 结果暴露了系统性协议错误；V55 在不改变 belief/planner 的前提下修复 seller routing、per-seller adapter、完整合同 renderer 与 validator。V55 已覆盖四个含 multi-seller 的 family，共 `116/116` tasks、0 runner error，单 seed BuyerScore `40.311`、valid-deal `84.48%`、timeout `9.48%`。同 seed 历史 Direct/CoT/full 在同 116 tasks 上分别为 `30.281/30.710/29.165`。
5. **AgenticPay 最新确认结果**：V60 在最难的 `only_multi_seller` 29 tasks 上完成 3 seeds，即 `87` 个 paired episodes/variant、共 `522` 条结果。V60 BuyerScore `30.718`，高于 CoT `16.832`（`+13.887`，task-cluster 95% CI `[+3.987,+23.021]`）和 historical full `17.082`（`+13.637`，CI `[+3.113,+23.556]`）；相对 Direct `22.675` 为 `+8.044`，但 CI `[-1.382,+17.257]` 仍跨 0。V60 是当前 `only_multi_seller` champion；尚未完成 V60 的 all-116/all-231 扩展，因此不能把它写成整个 AgenticPay 的 universal winner。

### 0.2 是否已经形成“一个 universal framework”

还没有完全形成。当前代码有两层：

- **共享核心**：`schema -> belief update -> candidate ranking -> action lock -> validator`，Simple 与 AgenticPay 都使用 `negotiation/framework/`；
- **环境适配/历史实现**：CaSiNo 和 NegotiationArena 仍有各自的 belief/planner；AgenticPay V60 又包含较多合同状态机和环境特定 safety guard。

因此目前最准确的 story 是：我们已经得到一个可审计的通用接口和一组跨环境机制，但尚未证明同一套 learned belief/planner 参数在四个 benchmark 上同时带来 payoff gain。

---

## 1. 统一代码结构

### 1.1 共享执行链

```text
environment history + focal private utility
  -> Environment Adapter: CanonicalState / legal candidates
  -> Belief Store: opponent preference + response policy + evidence provenance
  -> Belief Updater: structured behavior first, verified semantics second
  -> Belief-Usable Planner: score/rank only legal executable candidates
  -> Selection Validator: reject protocol-valid but strategically unsafe action
  -> Locked Renderer: LLM may verbalize but cannot change price/terms/action type
  -> Final Validator / deterministic repair
  -> unchanged benchmark parser, transition and native scorer
```

### 1.2 共享核心代码

| 模块 | 代码 | 作用 |
|---|---|---|
| Canonical schema | `negotiation/framework/schemas.py` | `CanonicalState`、`OpponentBelief`、`CandidateAction`、`FrameworkDecision` |
| Orchestration | `negotiation/framework/engine.py` | belief update、候选排序、selection validator、locked rendering、trace |
| Belief | `negotiation/framework/belief.py` | reservation interval、response policy、latent regime mixture、verified semantic evidence |
| Rule planner | `negotiation/framework/planner.py` | deadline/frontier/lookahead/active-probe 等可执行 planner |
| Conservative offline RL | `negotiation/framework/conservative_awr_planner.py` | AWR residual 只在 bootstrap LCB 过阈值时覆盖 frozen base |
| Trainable planner | `negotiation/framework/trainable_planner.py` | 可训练 accept/continue、probe、concession decision boundary |
| Verified events | `negotiation/framework/events.py` | 区分直接行为证据、语义证据与不可验证推断 |

### 1.3 当前限制

- CaSiNo 的最优版本仍使用 `MathPartnerBeliefModel + RatioCandidatePlanner`，没有直接调用共享 `framework/engine.py`。
- NegotiationArena V8 使用自己的 particle/factorized posterior，尚未迁入共享 core。
- AgenticPay 的合同 adapter 已超过 5,000 行，包含 issue-role、public contract state、post-probe repair、seller routing 等环境语义；这些 safety components 合理，但不能全部作为“通用 belief model”宣传。
- Simple V6.3 和 AgenticPay V60 共享接口，不共享同一个最终 planner class/checkpoint。跨环境参数共享仍是未完成目标。

---

## 2. CaSiNo

### 2.1 Setting 与实现

- camping multi-issue negotiation：Food、Water、Firewood，每项 3 units；
- 双方有私有 issue priority；P1 先行动，Partner-Base P2 固定；
- 12 rounds；主指标为 P1/P2 allocation score、agreement/walk-away；
- benchmark test split 共 100 条。

当前表现最好的是 `belief_usable_llm_chooser`：

1. `MathPartnerBeliefModel` 从 P2 的自我陈述、保留资源和让步行为更新 issue posterior、flexibility、stubbornness、walk-away risk；
2. `RatioCandidatePlanner` 枚举合法 allocation，以 aggressive/self-leaning/balanced/cooperative/high-accept 五类 ratio 构造候选；
3. belief model 为候选估计 partner score、accept probability、information gain 和 risk；
4. LLM chooser 只能返回 candidate id；
5. generator 只写对话，若偏离 allocation，action lock 强制恢复 planner 的结构化动作。

### 2.2 代码怎么查看

| 内容 | 当前整理库路径 | 原始工作路径 |
|---|---|---|
| Belief | `negotiation/CaSiNo_Env/buyer/belief_usable_planner/belief_model.py` | `ASTRA_Env/buyer/belief_usable_planner/belief_model.py` |
| Candidate planner | `negotiation/CaSiNo_Env/buyer/belief_usable_planner/planner.py` | `ASTRA_Env/buyer/belief_usable_planner/planner.py` |
| LLM chooser | `negotiation/CaSiNo_Env/buyer/belief_usable_planner/llm_chooser.py` | 同名路径 |
| Generator/action lock | `negotiation/CaSiNo_Env/buyer/belief_usable_planner/generator.py` | 同名路径 |
| Pipeline | `negotiation/CaSiNo_Env/buyer/belief_usable_planner/buyer.py` | 同名路径 |
| Variant factory | `negotiation/CaSiNo_Env/buyer/factory.py` | `ASTRA_Env/buyer/factory.py` |
| 正式脚本 | — | `ASTRA_Env/scripts/run_partner_base_belief_usable_planner.sh` |
| 算法说明 | `negotiation/docs/timeline/2026-08-03_11_casino_belief_usable_planner_cn.md` | `ASTRA_Env/BELIEF_USABLE_PLANNER_ALGORITHM_V01_CN.md` |

### 2.3 结果

两次均为相同 100 条 test split、1 rollout/scenario：

| Run | Variant | P1 AvgScoreAll | P2 AvgScoreAll | Agreement | Walk-away | Belief top-1 |
|---|---|---:|---:|---:|---:|---:|
| 2026-07-29 | Direct | 18.73 | 17.24 | 99% | 1% | — |
| 2026-07-29 | Historical full | 21.62 | 16.03 | 100% | 0% | 0.38 |
| 2026-07-29 | Deterministic belief-usable | 22.27 | 14.18 | 96% | 4% | 0.36 |
| 2026-07-29 | **LLM chooser** | **22.30** | 13.55 | 95% | 5% | 0.40 |
| 2026-08-03 | Direct | 17.40 | 16.87 | 94% | 6% | — |
| 2026-08-03 | Historical full | 21.66 | 15.75 | 99% | 1% | **0.48** |
| 2026-08-03 | Deterministic belief-usable | 21.67 | 13.13 | 92% | 8% | 0.29 |
| 2026-08-03 | **LLM chooser** | **22.17** | 13.87 | 95% | 5% | 0.38 |

相对 Direct，LLM chooser 的 P1 point gain 为 `+3.57/+4.77`。代价是第一轮中 agreement 下降 4 个百分点；第二轮则上升 1 个百分点。

结果文件：

- `astra_env_runs/v0.1/20260729_partner_base_belief_usable_planner/summary.json`
- `astra_env_runs/v0.1/20260803_partner_base_belief_usable_planner/summary.json`

### 2.4 全集判断

**已覆盖本项目冻结的官方 test split 100/100。** 但每次只有 1 stochastic rollout，现有 `t_statistic` 是成交局内 P1−P2，不是 method-vs-baseline 的 paired CI。论文级结论仍建议用 common seeds 再做至少 3 rollouts/scenario，并按 scenario 聚类 bootstrap。

---

## 3. NegotiationArena

### 3.1 Setting

正式评估包含四个 focal settings：

| Setting | Focal role | 当前 agent 观察 |
|---|---|---|
| Buyer | BLUE buyer，后手 | seller 的第一个 price offer |
| Seller | RED seller，后手 | buyer 的第一个 price offer |
| Resource-first | RED，先手 | 本局尚无对手正式动作 |
| Resource-second | BLUE，后手 | RED 的第一个 bundle proposal |

每个 run 有 20 个连续 episodes；同一 run 可跨 episode 保留公开历史/posterior；每局最多 10 turns。正式 cell 为 5 runs × 20 episodes，推断单位是 run，而不是把 100 episodes 当作独立样本。

### 3.2 方法与代码

最终研究版本 V8：

- preference `theta`：reservation value 或 X/Y relative utility；
- response policy `phi`：rationality、acceptance bias、counter propensity；
- joint particles 上计算 action value/regret；
- evidence-channel reliability gate 限制错误/内生 evidence 的影响；
- LLM 只 verbalize locked price/bundle/ACCEPT。

| 内容 | 路径 |
|---|---|
| V2–V8 agent 继承链 | `negotiation/NegotiationArena/arena_integration/decision_calibrated_agent.py` |
| Direct/repeated baseline | `negotiation/NegotiationArena/arena_integration/repeated_agents.py` |
| Paper-aligned Opponent Simulation | `negotiation/NegotiationArena/arena_integration/paper_aligned_opponent_simulation.py` |
| Repeated games | `negotiation/NegotiationArena/arena_integration/repeated_games.py` |
| 正式 runner | `negotiation/NegotiationArena/arena_integration/run_opponent_simulation_setting.py` |
| 完整报告 | `negotiation/docs/reports/NEGOTIATIONARENA_FINAL_REPORT_CN.md` |
| V8 analysis | `negotiation/docs/results/V8_CONFIRMATORY_ANALYSIS.md` |
| 原始 calls/trajectories | `external_negotiation_envs/NegotiationArena/research_iterations/iteration_001_*` 至 `iteration_011_*` |

### 3.3 主 baseline 对比

表中为 focal reward / agreement rate / joint reward；不同 game 的 utility scale 不同，不能跨行简单平均。

| Setting | Direct | Opponent Simulation | 我们旧矩阵最好 | 对 OS delta |
|---|---:|---:|---:|---:|
| Buyer | 7.77 / .99 / 19.80 | 12.76 / 1.00 / 20.00 | **V3.3 13.11 / .95 / 19.00** | **+0.35** |
| Seller | 18.08 / 1.00 / 20.00 | **19.13 / .96 / 19.20** | V3.3 16.95 / 1.00 / 20.00 | **−2.18** |
| Resource-first | 12.34 / .94 / 17.36 | **40.57 / .96 / 48.08** | V5 18.45 / .99 / 37.32 | **−22.12** |
| Resource-second | 9.75 / .93 / 29.76 | 11.88 / .97 / 30.74 | **V3.3 22.47 / .96 / 47.90** | **+10.59** |

V8 confirmatory 的关键结果：

| Setting | V5 | V7 | V8 | V8 frozen | V8 shuffled | V8 oracle |
|---|---:|---:|---:|---:|---:|---:|
| Buyer | 8.880 | 8.560 | 7.240 | 7.490 | 7.940 | **12.710** |
| Seller | 14.360 | 15.180 | 11.120 | 13.750 | 11.560 | **17.330** |
| Resource-first | 18.455 | 18.460 | 22.265 | **22.500** | 18.020 | 22.445 |
| Resource-second | 24.420 | 21.875 | 21.675 | 25.125 | **25.665** | 25.655 |

解释：V8 在 Resource-first 相对 V5 `+3.810`，5-run 描述性 95% t interval `[1.492,6.128]`，且正常 belief 高于 shuffled；但 Resource-second learned belief 低于 frozen/shuffled，stationary opponent 中 continuous 没有稳定胜 frozen。

### 3.4 全集判断

NegotiationArena 不是静态 dataset，没有“遍历所有样本”的含义。按本项目预注册的 benchmark grid，已经完成：旧正式 matrix 3,600 episodes、opponent switch 2,000、V7 3,200、V8 2,800、两组 identifiable switch/control 1,400，总计约 13,000 episodes，当前记录为 0 infrastructure error。这个 grid 是完整的；但它只有两个 game schema 和固定 utility parameterization，不能代表开放域 negotiation 全集。

---

## 4. Simple Env / AmazonHistoryPrice

### 4.1 Setting 与 reward

- single buyer、single seller、single item；固定 native/neutral seller，只改 buyer；
- buyer budget `B`、seller cost `C`、成交价 `P`；
- buyer reward 约为 `(B-P)/|B-C|`；no-deal 为 0，invalid overshoot 为 −1；
- 冻结 test manifest：`data/rlvr_negotiation/amazonhistoryprice_test128_first_budget0.8.jsonl`，128 scenarios。

### 4.2 当前最优结构

V5.9 是 frozen behavioral base；V6.3 在它上面加 conservative AWR residual：

1. `SimpleEnvAdapter` 把价格/history 转成 canonical state，并生成 price/probe/accept/continue legal candidates；
2. `BroadPriorBeliefStore` 维护 reservation interval；
3. `StrategicReadinessMixtureBeliefUpdater` 用 accept/counter/reject 更新 reservation、response readiness 与 latent profile mixture；
4. `VerifiedSemanticBeliefUpdater` 只允许可验证语义以小权重进入 belief；
5. V5.9 `BehavioralFrontierPlanner` 控制 anchor、concession frontier、deadline accept/continue；
6. V6.3 `ConservativeAWRPlanner` 预测 residual action advantage，只有 ensemble bootstrap lower-confidence bound 过安全阈值才覆盖 V5.9；
7. locked renderer 与 validator 保证最终 BUY/LEAVE 和价格不偏离 selected candidate。

### 4.3 代码怎么查看

| 内容 | 路径 |
|---|---|
| Simple adapter 与 V1–V6.5 variants | `negotiation/Simple_Env/buyer/universal_framework.py` |
| Variant factory | `negotiation/Simple_Env/buyer/factory.py` |
| Direct/CoT/full baselines | `negotiation/Simple_Env/buyer/direct_prompt.py`、`cot_prompt.py`、`full_framework.py` |
| Eval | `negotiation/Simple_Env/eval.py` |
| Shared belief/planner/engine | `negotiation/framework/` |
| V6.3 checkpoint | `Simple_Env/research_iterations/iteration_023_cross_environment_awr/results/awr_v1_preregistered/cross_environment_awr_checkpoint.json` |
| 预注册协议 | `universal_reward_research/iteration_000_full_baseline/PREREGISTERED_PROTOCOL_CN.md` |
| 主统计 | `universal_reward_research/iteration_000_full_baseline/analysis/PRIMARY_METRICS.md` |

### 4.4 完整主结果

128 scenarios × 3 rollouts × 5 variants，共 1,920 episodes；使用 scenario-cluster paired bootstrap 10,000 次：

| Variant | Mean reward | vs CoT | vs historical full |
|---|---:|---:|---:|
| Direct | 0.0961 | −0.3104 | −0.3503 |
| CoT | 0.4065 | — | −0.0399 `[-0.0779,-0.0024]` |
| Historical full framework | 0.4464 | +0.0399 `[+0.0024,+0.0779]` | — |
| **V5.9** | **0.4981** | +0.0916 `[+0.0472,+0.1373]` | +0.0517 `[+0.0109,+0.0924]` |
| **V6.3 conservative AWR** | **0.5130** | **+0.1066 `[+0.0581,+0.1560]`** | **+0.0666 `[+0.0203,+0.1118]`** |

V6.3−V5.9 为 `+0.0149 [-0.0296,+0.0589]`，所以 AWR residual 本身没有被显著证明；可信增益主要来自 V5.9 的 behavioral frontier/continuous belief 架构，V6.3 只提供最高 point estimate。

### 4.5 全集判断

**已完整覆盖本项目冻结的 128/128 test manifest，并有 3 rollouts/scenario。** 这不是“Amazon 全部商品历史记录”，而是 AmazonHistoryPrice-derived benchmark 的冻结 128 条 evaluation manifest。对当前论文实验，它可称 full test split；对原始数据源，不能称全 AmazonHistoryPrice corpus。

---

## 5. AgenticPay

### 5.1 本地 task inventory：不是一个单一的 “All-28”

当前 checkout 中共有 8 个 market-topology families、231 个可执行 task scripts：

| Family | 参与者/商品结构 | Task 数 | 主要决策 |
|---|---|---:|---|
| `single_buyer_product_seller` | 1 buyer × 1 product × 1 seller | 28 | 单边合同/价格协商 |
| `only_multi_products` | 1 buyer × multiple products × 1 seller | 29 | 多商品逐项、bundle 或选子集 |
| `only_multi_seller` | 1 buyer × 1 product × 2/3 sellers | 29 | seller routing + offer/contract |
| `only_multi_buyer` | 2/3 buyers × 1 product × 1 seller | 29 | buyers 竞争；seller 选成交 buyer |
| `multi_products_multi_seller` | 1 buyer × multiple products × 2/3 sellers | 29 | 商品与 seller 联合选择 |
| `multi_buyer_multi_seller` | 2/3 buyers × 1 product × 2/3 sellers | 29 | buyer–seller edge matching |
| `multi_buyer_multi_products` | 2/3 buyers × multiple products × 1 seller | 29 | buyer competition + bundle/selection |
| `multi_buyer_multi_products_multi_seller` | multiple buyers × products × sellers | 29 | routing、matching、合同联合决策 |
| **总计** | — | **231** | — |

按当前源码而不是 README 的概括计数：`single` family 有 3 个 legacy price-only tasks，其余 7 个 family 各有 4 个 legacy topology tasks，共 31 个；每个 family 另有 25 个 scenario contract tasks，共 200 个。这 25 个 scenario tasks 通常包括 10 个电商、5 个 ride-hailing、5 个 food-delivery、5 个 apartment-rental 实例。因而：

- “single-28”只覆盖最简单 topology；
- “all-28/all-29”必须同时写明是哪一个 family；
- “all-116”指四个含 multi-seller 的 families，`4×29=116`；
- “all-231”才是当前本地 task-script inventory 全覆盖。

### 5.2 一个 AgenticPay episode 到底在优化什么

price-only task 的动作含 `### BUYER_PRICE($p) ###`；contract task 的每次有效动作必须提交完整结构，而不是只谈当前修改的一项：

```json
{
  "price": 925,
  "continuous_terms": {"lease_months": 24},
  "discrete_terms": {
    "include_utilities": false,
    "user_product_preference": "strong_match"
  }
}
```

合同效用是 multi-attribute utility：

\[
U_b=v_{base}-p+\sum_i w^b_i x_i+\sum_j w^b_j(o_j),\qquad
U_s=p-c_{base}+\sum_i w^s_i x_i+\sum_j w^s_j(o_j).
\]

双方合同的所有 continuous/discrete terms 必须完全一致，并且价格相交或落在 tolerance 内，环境才结算。对 contract mode，`Z_max` 是在合法条款空间内可达到的理论最大 joint surplus；multi-seller 时使用市场中理论最优 seller 的 `Z_market` 做归一化。令 `r_b=U_b/Z_market`、`d=gamma^(t-1)`，官方 BuyerScore 的共同形式是：

\[
\text{BuyerScore}=
\begin{cases}
d(D_b+W_b r_b+E_b), & \text{feasible valid deal}\\
-F_b(1-d), & \text{timeout/invalid/non-IR deal}.
\end{cases}
\]

各 topology 的权重可能不同，例如单 buyer/seller 默认 `(D_b,W_b,E_b)=(30,55,15)`，multi-seller 环境默认 `(10,80,10)`。因此 BuyerScore 同时奖励：达成有效交易、buyer 自身效用、尽早完成；它不是 `buyer_reward` 的别名。`DealRate` 只检查环境是否标为 agreed，而 `valid-deal rate` 还要求 BuyerScore `>0`。本报告以官方 mean BuyerScore 为主指标，raw agreement、timeout、mismatch、belief calibration 仅用于解释。

### 5.3 Multi-seller：一个 buyer 如何在多个 seller 之间切换

AgenticPay 同时提供 parallel 与 sequential 协议：

- legacy Task1/2 是并行 2/3 sellers：同一轮可并列比较多个 seller response；
- legacy Task3/4 是顺序 2/3 sellers；
- 当前 Task5–29 的 scenario contracts 主要复用 sequential two-seller 环境。buyer 每轮先输出 `<selected_seller>N</selected_seller>`，随后只在该 buyer–seller edge 上发一份完整合同；未被选择 seller 的历史仍保留，之后可以切回。

以 `only_multi_seller/Task5_s1_beauty_product_negotiation.py` 为例：buyer 的 base value 为 `$6.18`，Seller 1/2 的私有 base cost 分别为 `$5.54/$4.95`；除价格外还谈 `delivery_days∈[1,7]`、`return_policy`、`packaging` 和 `user_product_preference`。两个 seller 对快递、退货和包装的成本不同，所以“选报价较低者”不等于“选合同效用较高者”。

当前 framework 的处理是：

1. 为每个 `counterparty_id=seller1/seller2/...` 维护独立公开历史与独立 belief，禁止把 Seller 1 的 concession 更新到 Seller 2；
2. 从当前所选 seller 的 buyer-visible schema 构造合同，运行时不读取 seller-private weights/floor；
3. 首轮确定性探索未访问的 edge，之后只用公开 seller contracts 和 buyer 自己的 utility 路由；
4. renderer 同时锁定 `<selected_seller>` 与完整 `<contract>`；
5. validator 检查 `routing_ok / contract_tag_ok / contract_complete_ok / action_lock_ok / action_semantics_ok / buyer_ir_ok`；
6. V60 对缺失结构化合同的 seller 最多重试一次；对首次优质完整报价允许早接受；若 seller 后来撤回，可重提该 seller 曾公开的最佳 buyer-IR 合同；deadline 时优先可结算而非重复低价或 `quit`。

最新实际轨迹例子：V60 在 price-only Task1 第 2 round 选 Seller 1、以 `$96.5` 成交，BuyerScore `79.53`；在 rent Task25 第 6 round 选 Seller 2、以 `$925`、15-month lease、utilities included、strong-match 成交，BuyerScore `38.98`。Beauty Task5 与 food-delivery Task20 仍会跑到 21 rounds timeout，说明协议 bug 已修复，但 hard negative-IR frontier 尚未解决。

### 5.4 Multi-buyer：不是 buyer 自己选择“赢得交易”

以 legacy `only_multi_buyer/Task1_parallel_two_buyer_negotiation.py` 为例：Buyer 1/2 的私有 max price 为 `$150/$160`，seller floor 为 `$80`。每轮两个 buyers 分别报价，native seller 分别响应；若两条 edge 同时相交，环境选择 effective price 更高的 buyer 成交。Sequential contract tasks（例如 beauty Task5）则由 native seller 输出 `<selected_buyer>`，每轮只推进一个 seller–buyer edge。

对我们的代码有两个重要含义：

1. 每个 buyer instance 都有自己的 private utility、session 和对 seller 的 belief；某 buyer 看不到另一 buyer 的 private budget，只能从其被选择/未被选择等公开结果间接推断竞争压力。
2. 当前 `patch_module()` 会把 task 中创建的所有 `BuyerAgent` 都替换成同一个被测 variant，而 seller 始终保持 native/fixed。也就是说，现有 multi-buyer 数字衡量的是“同一 buyer policy 控制市场中的所有 buyers 后，最终 selected buyer 的官方 BuyerScore”，还不是严格的“只替换一个 focal buyer、其他 buyers 固定”的博弈实验。

因此 multi-buyer family 可用于协议迁移和系统能力测试，但论文中的策略归因还需要增加 focal-buyer mode：只替换 Buyer 1，固定 Buyer 2/3 为 repo-native 或预注册 baseline，并分别报告 focal payoff、selection probability 与 market BuyerScore。

### 5.5 Multi-issue：为什么比多轮谈价格难得多

multi-issue 不只是把一串字段塞进 JSON；每个字段的方向、离散语义和可验证性不同：

| 场景例子 | Continuous issues | Discrete issues | 典型冲突 |
|---|---|---|---|
| Beauty/e-commerce | `delivery_days` | `return_policy`、`packaging`、`user_product_preference` | buyer 要快递/可退/保护包装，seller 承担履约成本 |
| Taxi | `wait_time_mins` | `route_preference`、preference match | buyer 可能愿意等但想走 tunnel；driver 可能偏好 local streets |
| Food delivery | 可为空 | `delivery_speed`、`extra_condiments`、preference match | rush 与 condiment 提升 buyer utility、增加 seller cost |
| Apartment rental | `lease_months` | `include_utilities`、preference match | buyer 不愿长租，landlord 以长租降低 vacancy risk |

当前 framework 的处理链是：

1. **Schema adapter**：读取 bounds/options 和 buyer 自己的 MAUT weights；每个 candidate 都是字段齐全、可被环境 parser 执行的合同。
2. **Issue classifier**：将 issue 区分为 operational obligation、可议 price/term、描述性事实（如 `user_product_preference`）；证据不足可 abstain，避免把事实标签当作可随意交换的承诺。
3. **Continuous belief**：从 seller 的 structured counter/reject/accept 更新 reservation/frontier、response policy 和 issue-option evidence；自然语言语义只能以较低可靠度补充。
4. **Active probe**：一次只改变一个可辨识 issue，并用价格/其他条款补偿，观察 seller 是否恢复旧值；probe 不能跨过对方公开价格导致意外成交。
5. **Post-probe repair**：复制最新公开 seller contract，只修改维持 buyer individual rationality 所需的最少字段，避免 LLM 在“接受”时又悄悄改动两三项。
6. **Locked rendering + validator**：planner 选择的 seller、price、continuous terms、discrete terms 必须逐字段出现在最终 wire action；缺字段或语义矛盾则 deterministic rebuild。

在 combined task（例如 two buyers × two sellers × two-product bundle）中，上述机制嵌套运行：每个 buyer 先选择 seller edge，再提交针对整套 bundle 的合同；环境在可行 edges 中决定最终 buyer–seller pair。当前 V55 已能无 runner error 地覆盖这类任务，但还没有证明 cross-buyer competition belief 或 bundle-level planner 本身带来增益。

### 5.6 从 V37 到 V60：修了什么，什么仍是环境适配

| 版本 | 核心变化 | 结论 |
|---|---|---|
| V37 `rolling_public_contract_state` | non-crossing probe、public counter latch、minimal buyer-IR repair、旧 anchor 失效 | single-28 有正向单-seed结果；迁移 multi-seller 时协议失败 |
| V38–V54 | trade ledger、semantic confirmation、rejectable issue-role/factual routing、partner-IR reserve | 多数增加 timeout/mismatch；未通过 safety/CI gate |
| V55 `multiseller_protocol_safe` | 只修 seller routing、per-seller adapter、完整合同 renderer、六项 validator；belief/planner 继承 V37 | 消除 Task5–29 的系统性 missing-contract timeout，可覆盖 116 tasks |
| V56–V59 | deadline settlement、stagnation-financed repair、rejection-aware/scoped repair | 提供失败诊断，但未成为最终 champion |
| **V60 `resilient_multiseller_settlement`** | bounded missing-contract retry、公开合同 utility routing、early strong-offer recovery、historical public-offer replay | `only_multi_seller` 三种子最高 BuyerScore；当前冻结 champion |

其中 V55 的 adapter/renderer/validator 是必须的 execution layer，不应包装成 belief novelty；V60 的公开 evidence routing、opponent-indexed belief 与 settlement decision 才属于 planner 可以泛化和消融的部分。

### 5.7 最新结果 A：V55 已完成四个 multi-seller families 的 all-116

固定 Qwen3-30B、native sellers、seed `20260904`，V55 完成 116/116、0 runner error：

| Family | n | BuyerScore | Agreement | Valid deal | Timeout |
|---|---:|---:|---:|---:|---:|
| only multi-seller | 29 | 28.963 | 68.97% | 62.07% | 31.03% |
| multi-products + multi-seller | 29 | 34.778 | 96.55% | 79.31% | 3.45% |
| multi-buyer + multi-seller | 29 | 47.546 | 96.55% | 96.55% | 3.45% |
| multi-buyer + products + sellers | 29 | **49.957** | **100%** | **100%** | **0%** |
| **Overall** | **116** | **40.311** | **90.52%** | **84.48%** | **9.48%** |

同一个 seed 和 task inventory 的历史结果为：Direct `30.281`、CoT `30.710`、historical full `29.165`、旧 V37 `8.375`。V55 point gain 分别是 `+10.030/+9.601/+11.146/+31.936`。这是 matched-task、单-seed、不同时间执行的对比；可说明协议修复是大幅跃迁，但不能替代 fresh multi-seed control matrix。分 family 看，V55 在最后一个完整 topology 上仍低于 CoT（`49.957 < 54.811`），因此也不能声称每个 family 都胜 baseline。

原始结果：

- `universal_reward_research/iteration_056_multiseller_protocol_safe/runs/full116_clean_seed_20260904/task_results.jsonl`
- `universal_reward_research/iteration_056_multiseller_protocol_safe/runs/full116_clean_seed_20260904/summary.json`
- `universal_reward_research/iteration_056_multiseller_protocol_safe/INTERIM_PROTOCOL_FIX_AND_SMOKE_RESULTS_CN.md`

### 5.8 最新结果 B：V60 的 only-multi-seller 三种子正式矩阵

矩阵为 29 tasks × 3 seeds × 6 variants，即每个 variant 87 episodes、总计 522 条记录；所有 seed 均为 174/174，0 runner error。

| Variant | BuyerScore | Agreement | Valid deal | Timeout |
|---|---:|---:|---:|---:|
| CoT | 16.832 | 82.76% | 48.28% | 17.24% |
| Direct | 22.675 | **96.55%** | 59.77% | **3.45%** |
| Historical full | 17.082 | **100%** | 44.83% | **0%** |
| V37 | 27.091 | 79.31% | 65.52% | 20.69% |
| V55 | 27.091 | 79.31% | 65.52% | 20.69% |
| **V60** | **30.718** | 89.66% | **70.11%** | 10.34% |

按 task 聚类 bootstrap 10,000 次，V60 的 paired BuyerScore delta 为：

| Comparison | Delta | 95% CI | 当前判断 |
|---|---:|---:|---|
| V60 − CoT | **+13.887** | **[+3.987,+23.021]** | 明确为正 |
| V60 − historical full | **+13.637** | **[+3.113,+23.556]** | 明确为正 |
| V60 − Direct | +8.044 | [−1.382,+17.257] | point gain，统计未定 |
| V60 − V55 | +3.628 | [−1.010,+8.489] | point gain，统计未定 |

分 task domain 后可见增益不均匀：

| Domain | CoT | Direct | Full | V60 |
|---|---:|---:|---:|---:|
| Legacy topology/price-only（4 tasks） | 33.709 | 33.161 | 32.203 | **79.345** |
| E-commerce contracts（10） | 8.157 | **27.479** | 4.279 | 23.671 |
| Ride-hailing contracts（5） | −1.268 | −0.327 | 4.537 | **16.513** |
| Food-delivery contracts（5） | **41.833** | 25.599 | 21.271 | 17.646 |
| Apartment-rental contracts（5） | 13.777 | 24.756 | **38.947** | 33.190 |

所以 V60 的总体提升是真实的 paired mean gain，但一部分来自 4 个 price-only topology tasks；它在 food delivery 明显弱于 CoT/Direct，在 e-commerce 弱于 Direct，在 rental 弱于 historical full。这正是下一轮应优化的 heterogeneous contract boundary，而不是继续堆更多全局 heuristic。

主要失败轨迹：

- Beauty Task5：3/3 timeout，后期连续选择 `quit`，但上游协议并不会把该 action 当作立即 no-deal，导致继续循环；
- Beauty Task6：一次 timeout、两次 nonpositive agreement；
- Food Task20：3/3 timeout，曾观察到可行公开合同但继续 counter，之后 seller 撤回；
- Food Task24：BuyerScore 大幅低于 CoT；
- Taxi Task17：3/3 agreement 但 BuyerScore 为轻微负值，暴露 planner buyer-IR 与官方 scorer 口径仍有偏差；
- Rent Task25/28/29：能稳定成交，但部分 baseline 的条款组合效用更高。

正式分析与原始结果：

- `universal_reward_research/iteration_062_agenticpay_score_autoloop/analysis/upstream_formal/matrix_analysis.json`
- `universal_reward_research/iteration_061_resilient_multiseller_settlement/formal_paired_analysis.tsv`
- `universal_reward_research/iteration_061_resilient_multiseller_settlement/runs/formal_seed_20260904/`
- 同级 `formal_seed_20260905/`、`formal_seed_20260906/`

### 5.9 当前流水线状态

截至 2026-09-04：

- 三种子 V60 matrix 已完整结束并写入 `COMPLETE`；当前没有 AgenticPay model experiment 在后台运行；
- 自动研究器正确选出 V60 并写入 `PRELIMINARY_FREEZE.json` 与 `FINAL_STATE.json`；
- 两次 rule-based V61 和一次 low-cost training V61 均未真正产生代码或实验。失败原因是 launcher 硬编码了 Snap shim `/snap/bin/codex`；该入口在 detached/sandbox 环境中无法通过 `snap-confine` 权限检查/解析实际 Codex 命令，三个调用均立即失败（日志为 `No such file or directory`）。这不是“训练效果不好”，而是训练没有开始；
- 当前冻结版本应保持 V60。修正 Codex executable discovery 后，才能重启 held-out-safe V61 流程；在此之前报告中不得出现“V61 已训练/已验证”。

### 5.10 代码怎么查看

| 内容 | 路径 |
|---|---|
| 上游 8-family task scripts | `benchmarks/AgenticPay/agenticpay/examples/` |
| 上游 environment/parser/scorer | `benchmarks/AgenticPay/agenticpay/envs/` |
| V1–V60 adapter、candidate、planner、renderer、validator | `negotiation/AgenticPay_Env/buyer/universal_framework.py` |
| 可拒绝 issue-role/factual classifier | `negotiation/AgenticPay_Env/buyer/issue_classifier.py` |
| Variant registry 与说明 | `negotiation/AgenticPay_Env/buyer/variants.py` |
| Agent replacement 与 trajectory capture | `negotiation/experiments/run_agenticpay_single28_framework.py` |
| 231-task discovery、filter、metric normalization | `negotiation/AgenticPay_Env/environment/all_tasks.py` |
| Unified eval CLI | `negotiation/AgenticPay_Env/eval.py` |
| Trajectory viewer | `negotiation/tools/view_agenticpay_trajectory.py` |
| V55 协议修复报告 | `universal_reward_research/iteration_056_multiseller_protocol_safe/INTERIM_PROTOCOL_FIX_AND_SMOKE_RESULTS_CN.md` |
| V60 计划与结果 | `universal_reward_research/iteration_061_resilient_multiseller_settlement/EXPERIMENT_PLAN_CN.md` |
| 自动分析/研究器 | `universal_reward_research/iteration_062_agenticpay_score_autoloop/` |
| 历史失败轨迹 | `8.23/AGENTICPAY_ENV_TASKS_EXPERIMENT_HISTORY_AND_SAMPLE_TRAJECTORIES_CN.md` |

### 5.11 全集判断

- 旧 V37/Direct/CoT/historical full：均完成 all-231 × 1 seed，但 V37 的 100 个 contract multi-seller tasks 有已知协议 bug，不能作为 belief/planner 结论；
- V55：已完成全部含 multi-seller 的 all-116 × 1 seed，证明修复后的协议可广覆盖；
- V60：已完成 `only_multi_seller` all-29 × 3 seeds，统计最可信，但范围最窄；
- 尚缺：V60 all-116、V60 all-231、fresh 3-seed all-116 controls，以及 focal-only multi-buyer protocol。

因此最新 AgenticPay 结论应写成：“V60 在 one-buyer/multi-seller family 上超过 CoT 与 historical full，V55 已证明执行层可迁移到四个 multi-seller topologies；整个 231-task benchmark 的 universal reward gain 尚未完成确认。”

---

## 6. 全集覆盖总表

| Benchmark | 本项目 evaluation universe | Framework 覆盖 | Baseline 覆盖 | 当前状态 |
|---|---:|---:|---:|---|
| CaSiNo | official test 100 | 100/100 × 2 runs | Direct/full 同范围 | **完整 test split**；rollout 数仍偏少 |
| NegotiationArena | 预注册 4 settings + interventions | 全部 cell | Direct/Opponent Simulation 同主矩阵 | **完整实验 grid**；不是静态 dataset |
| Simple/AmazonHistory | frozen test manifest 128 | 128 × 3 | 128 × 3 | **完整本地 test manifest** |
| AgenticPay single-28 | 28 tasks | 多轮 All-28；V37 有正式单 seed | 初始 5 variants 有 3 rollouts/task；V37 同次只有 CoT | **历史 task 完整，replication 不完全** |
| AgenticPay multi-seller-116 | 4 families × 29 | V55 116/116 × 1 seed | 历史 Direct/CoT/full 同 116 tasks | **修复后 topology 覆盖完整**；fresh multi-seed controls 尚缺 |
| AgenticPay only-multi-seller-29 | 29 tasks | V60 29/29 × 3 seeds | 5 controls 同范围同 seeds | **当前最可信 AgenticPay 矩阵**；V60 胜 CoT/full，vs Direct CI 跨 0 |
| AgenticPay all-231 | 231 tasks / 8 families | 旧 V37 231/231；V60 尚未跑 | Direct/CoT/full 均 231/231 × 1 seed | **旧单-seed matrix 完整但含已知 V37 协议 bug**；post-fix 全集待补 |

“全集”应始终同时说明 universe 和 repetition。遍历 100/128/231 个 task 不等于统计确定；CaSiNo 和 All-231 仍需要 independent rollouts/seeds 才能给出稳定 CI。

---

## 7. 应如何使用这些结果讲 story

### 7.1 可以支持的主张

- 在多个自然语言 negotiation protocols 中实现了同一可审计接口：continuous belief、executable candidate planner、action lock、validator 与完整 trace。
- Simple 上有正式 clustered-CI 的 buyer reward 增益。
- CaSiNo 上候选化 belief-usable planning 有可重复 point gain。
- NegotiationArena 提供 same-state wrong/shuffled/oracle/frozen、action-flip 和 preference-switch 因果诊断，揭示“reward gain 是否真的来自正确 belief”。
- AgenticPay 已把旧 V37 的协议 failure boundary 分离为两部分：V55 证明 adapter/renderer/validator 修复可覆盖 116 个 multi-seller tasks；V60 又在 one-buyer/multi-seller 三种子矩阵上取得 BuyerScore 增益。跨 231 tasks 的 universal gain 仍未确认。

### 7.2 不能支持的主张

- 不能说同一个 learned belief model 已经在四个 benchmark 全部 SOTA；
- 不能把 oracle、shuffled/wrong 或 belief calibration 当成 reward baseline；
- 不能把 AgenticPay single-28、all-116 与 all-231 的数字直接拼接；task universe、seed 数和协议版本必须同时对齐；
- 不能说 continuous update 普遍优于 frozen；NegotiationArena 中存在明确反例；
- 不能把 AgenticPay 的 post-hoc fixed-seller correction 当作原生成交。

### 7.3 当前最合理的方法定位

> 一个把 opponent belief 约束为可验证、按 counterparty 分离、可干预状态，并要求 planner 只能在环境合法候选中使用该 belief 的 negotiation architecture；其贡献不仅是 payoff，还包括 belief-to-action 的因果可审计性和错误 belief 的 bounded downside。当前正收益已在 Simple/CaSiNo 出现，NegotiationArena 提供机制证据；AgenticPay 则证明 execution-safe framework 可以跨价格、完整合同和多 seller routing 工作，并在 one-buyer/multi-seller 上取得显著于 CoT/full 的 BuyerScore 增益。跨 multi-buyer/all-231 的相同结论仍待确认。

---

## 8. 下一步优先级

1. 冻结已经验证的 V55 protocol layer 与 V60 action validator；不要再同时修改 adapter 和 reward policy。
2. 先修复自动研究器的 Codex executable discovery，再恢复 development/held-out 隔离的 V61；当前三次调用均未开始，不能视为负实验。
3. 针对现有 failure atlas，只训练/调整 `quit-or-continue`、`first viable public contract recovery`、`probe/concession frontier` 三个边界；重点处理 Beauty 5/6、Food 20/24、Taxi 17，避免新增 task-ID 规则。
4. 训练标签按完整 episode official BuyerScore 做 conservative offline RL/AWR，并保留 V60 deterministic safety fallback；不能把 terminal reward 平均复制给所有 turn。
5. 先做 fresh 3-seed all-116 controls；通过后把 V60 扩至 all-231。另增 focal-only multi-buyer mode，避免“所有 buyers 同时被替换”造成归因歧义。
6. 最终统一运行：CaSiNo 100×3、Simple 128×3、NegotiationArena 固定 confirmatory grid、AgenticPay 231×至少3 seeds。只有同一 candidate 在四个环境均不低于各自强 baseline，才能称为 universal best。

---

## 9. 最短阅读路径

1. `negotiation/README.md`：整理后的公共代码总览；
2. `negotiation/framework/engine.py`：理解通用执行链；
3. `negotiation/framework/belief.py` 与 `planner.py`：理解 belief/planner；
4. 对应 benchmark adapter：
   - CaSiNo：`negotiation/CaSiNo_Env/buyer/belief_usable_planner/`；
   - NegotiationArena：`negotiation/NegotiationArena/arena_integration/decision_calibrated_agent.py`；
   - Simple：`negotiation/Simple_Env/buyer/universal_framework.py`；
   - AgenticPay：`negotiation/AgenticPay_Env/buyer/universal_framework.py`；
5. 结果：本报告第 2–5 节列出的原始 summary/analysis 文件。
