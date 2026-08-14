# NegotiationArena：V8 正式结果与真实偏好切换实验计划

更新日期：2026-08-13
状态：V8 stationary confirmatory 已完成；真实 private-preference switch 已实现并通过测试。

## 一、当前代码结构

核心 runner 是 `arena_integration/run_opponent_simulation_setting.py`。它负责四种角色 setting、重复 episode、模型调用、协议修复、完整轨迹落盘、resume 和汇总。核心方法位于 `arena_integration/decision_calibrated_agent.py`：

- V5：受控的 continuous belief + executable planner；
- V7：将 opponent preference `theta` 与 response policy `phi` 因子化；
- V8：在 V7 上加入 evidence-channel reliability gate，将 learned posterior 与 episode anchor 混合；
- frozen、wrong-confident、shuffled、oracle：作用在同一 planner 上的 belief intervention，不是不同 prompt baseline；
- same-state shadow action：在不重新调用模型的条件下，用 frozen/wrong/shuffled/oracle/ungated/anchor posterior 重算当前动作，记录 action flip。

原始调用、修复记录、公开对话、每次 framework decision、episode JSONL、run-level 与 cell-level summary 均保留在版本目录中。正式分析器为 `research_iterations/iteration_009_v8_reliability_gated_belief/analyze_confirmatory.py`。

## 二、正式测试 setting

模型为 `Qwen3-30B-A3B-Instruct-2507-base`，OpenAI-compatible endpoint 为 `127.0.0.1:8002/v1`。四种 setting 为：

1. Buyer：focal buyer，seller cost=43，buyer WTP=63；
2. Seller：focal seller，同一 buy-sell game；
3. Resource-first：focal RED first mover，RED private value X=0.5/Y=2.5，BLUE X=2.5/Y=0.5；
4. Resource-second：focal BLUE second mover，同一 resource game。

每个 cell 为 5 independent runs × 20 repeated episodes；每局最多 10 turns；候选数 5；temperature 0.7；`normalize_retry` 仅做协议安全修复。正式种子 `20261100`。28 cells 共 2800 episodes，全部完成，当前错误为 0。

比较方法为 V5、V7、V8、V8-frozen、V8-wrong、V8-shuffled、V8-oracle。旧正式矩阵还保留 direct prompting、Opponent Simulation、V3.3/V4/V5；因此不会用 V8 开发过程覆盖原 baseline。

## 三、希望回答的问题

本轮不是只问“reward 是否提高”，而是拆成五个可证伪问题：

1. learned belief 是否比 frozen/shuffled belief 更有用？
2. 正确 oracle belief 是否存在 planner-usable upper bound？
3. wrong belief 会否导致 catastrophic policy degradation？
4. reliability gate 是否能限制错误 belief 的 downside？
5. continuous update 在 stationary opponent 中无优势时，能否在真实 preference change 后更快恢复？

## 四、V8 正式结果

| Setting | V5 | V7 | V8 | V8-frozen | V8-wrong | V8-shuffled | V8-oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buyer focal reward | 8.880 | 8.560 | 7.240 | 7.490 | 8.210 | 7.940 | 12.710 |
| Seller focal reward | 14.360 | 15.180 | 11.120 | 13.750 | 14.130 | 11.560 | 17.330 |
| Resource-first focal reward | 18.455 | 18.460 | 22.265 | 22.500 | 18.660 | 18.020 | 22.445 |
| Resource-second focal reward | 24.420 | 21.875 | 21.675 | 25.125 | 20.055 | 25.665 | 25.655 |

所有 cell 为 100 episodes、0 errors。完整 agreement、joint reward、Brier、ECE、preference MAE、coverage、trust、action-flip 和五 run paired interval 见同目录 `analysis_final.md/json`。

### 可以成立的结论

- Resource-first 中，V8 相对 V5 提升 `+3.810`，五个 paired runs 全部为正，描述性 95% t interval 为 `[1.492, 6.128]`；agreement 0.99，joint reward 40.54。
- V8-wrong 在 Resource-first 为 18.66，而旧 V7-wrong 为 7.0。门控显著缩小 wrong belief 的灾难性损失。
- Buyer/Seller oracle 分别比 V8 高 5.47/6.21，证明 reservation belief 存在 planner-usable headroom。
- Resource-first normal belief 胜过 shuffled 4.245；belief 内容确实能改变并改善行为，而不只是添加计算模板。
- same-state intervention 确认 belief 会改变动作；oracle flip 很高，而 policy-only intervention 接近零，说明当前主要决策通道是 preference marginal，不是 response-policy marginal。

### 不能成立的结论

- V8 不是 universal payoff improvement：Buyer 与 Seller 均低于 V7；Resource-second focal reward 也没有提高。
- stationary opponent 下不能证明 continuous update 优于 frozen。Resource-first V8 与 frozen 为 22.265 vs 22.500；Resource-second 为 21.675 vs 25.125。
- Resource-second learned belief 不胜 shuffled（21.675 vs 25.665），因此不能声称 posterior accuracy 或 learned belief utility 已普遍解决。
- preference MAE 与 coverage 并未随 reward 一致改善；acceptance Brier/ECE 也不能替代 latent-preference calibration。
- oracle intervention 的 Brier/ECE 是 learned response head 的记录，不是 oracle head calibration；不可混为一谈。

## 五、当前最合适的 novelty story

不再讲“belief prompt 能提高 negotiation reward”，而讲：

> A belief-usable negotiation agent must expose a causal chain from public language evidence, through factorized and uncertainty-aware opponent beliefs, to executable action selection. Because learned beliefs are unreliable, the planner should gate decision influence by evidence provenance and must be evaluated with same-state interventions, wrong/shuffled/oracle beliefs, calibration, and actual opponent preference shifts—not only stationary self-play payoff.

方法贡献可分三层：

1. **Representation**：continuous、factorized `p(theta, phi)`，区分对方的私有偏好与语言/响应策略；
2. **Decision**：候选动作由 exact self utility、预测接受、reciprocity、risk/VOI 评分，LLM 只 verbalize locked action；
3. **Reliability and causal evaluation**：按 evidence channel 估计 trust，将不可靠 posterior 拉回 anchor，并提供 frozen/wrong/shuffled/oracle/ungated same-state intervention。

目前最强实证贡献是第三层：它揭示 belief 模块何时真正影响动作、何时错误 belief 会伤害 planner，以及门控能否限制 downside。第二层有 oracle headroom；第一层的 continuous adaptation 尚待真实 change test。

## 六、下一步：真实 private-preference switch

旧 `--opponent-switch-episode` 只替换 opponent agent instance/提示策略，不能证明 belief tracking。新增的 `--opponent-preference-switch-episode` 会在第 11 episode 起只改变对手的真实私有 utility，focal utility 保持不变：

- Resource-first：BLUE 的 X:Y 从 2.5:0.5（theta=5/6）改为 2:1（theta=2/3）；
- Resource-second：RED 的 X:Y 从 0.5:2.5（theta=1/6）改为 1:2（theta=1/3）。

偏好排序保持不变，仍有互补交易空间，避免把“偏好变化”混成“博弈从互补变为完全冲突”。私有目标仅给对应 opponent；focal 不获得 switch 时间、方向或新 utility。ground truth 只用于 evaluator diagnostics 与 oracle/wrong intervention，不进入 normal V7/V8 prompt 或 learned posterior。

### 预注册比较

只测 Resource-first/second × V7/V8/V8-frozen，共 6 个 switch cells；其 no-switch control 直接使用同种子 `20261100` 已完成的正式 cells，避免额外模型噪声。每个 cell 仍为 5×20，第 11 episode 切换。

主要 estimand：

- `post - pre` reward 的 switch-control difference-in-differences；
- episode 11–15 与 16–20 的 recovery curve；
- belief estimate 对新 truth 的 MAE、coverage 与 update latency；
- V8 vs V8-frozen 的 post-switch reward/agreement/joint reward；
- switch 前不应出现 anticipatory change。

成功标准不是要求 V8 在所有 reward 上最大，而是：continuous V8 相对 frozen 在 post-switch adaptation 上有一致改善，并伴随 belief 朝新 truth 移动和 belief-caused action flip。若这些条件不成立，应结论为当前 continuous updater 尚未学到可用的 change adaptation，不继续用 stationary reward 包装该主张。

## 七、后续顺序

1. 完成 6-cell preference-switch matrix 和 paired DiD；
2. 若 change detection 失败，下一版本只改 forgetting/change-point mechanism，不重新调 planner reward weights；
3. 在 NegotiationArena 的更多 private utility profiles 与 opponent models 上做 held-out generalization；
4. 再把经过因果验证的 belief updater/planner 转移回 CaSiNo，检验 multi-issue natural-language preference ranking 与社会效用；
5. ANL 仅作为 structured control，不能替代 natural-language 主 benchmark。

## 八、首轮 toward-prior switch 的结果与设计修正

首轮 6 cells 已完成（600 episodes，0 errors）。以 stationary matrix 为 control 的 reward DiD：

- Resource-first：V7 `+1.45`，V8 `−2.95`，原 frozen `+2.66`；
- Resource-second：V7 `+1.86`，V8 `−3.15`，原 frozen `−0.82`；其中 V8 的描述性 95% t interval 为 `[−6.10, −0.20]`。

这不能支持 continuous V8 的 adaptation claim。进一步审计发现两项识别问题：

1. truth 从 `5/6→2/3`、`1/6→1/3` 是向 prior 0.5 移动，即使 posterior 不变，MAE 也会机械下降；
2. 历史 `frozen` 的语义是单局内只吸收第一次 event，但仍将该 belief 写入跨局 meta prior，因此它不是严格的 no-memory control。

这些结果与缺陷均保留，不删除、不重命名。新增 `framework_v8_no_cross_episode`：每局从同一个 uniform prior 开始，保留相同的局内 updater、planner 和 language realizer，但禁止跨局 belief carryover。最终 identifiable test 改为偏好远离 prior：BLUE `5/6→0.95`，RED `1/6→0.05`，并为 continuous/no-cross 同时运行 switch 与 no-switch controls。该版本才用于检验 continuous cross-episode adaptation。

## 九、最终 away-from-prior × strict no-cross 结果

最终 8 cells 全部完成：switch/control × Resource-first/second × V8/no-cross，800 episodes，0 errors。完整结果在 `research_iterations/iteration_011_identifiable_switch/runs/seed20261300/analysis_final.md/json`。

| Setting | Method | Switch pre→post reward | Reward DiD | Estimate pre→post | MAE pre→post | Post agreement | Post joint |
|---|---|---:|---:|---:|---:|---:|---:|
| Resource-first | V8 continuous | 16.04→17.03 | +2.47 [−3.18, 8.12] | .388→.364 | .445→.586 | .96 | 42.35 |
| Resource-first | strict no-cross | 16.90→18.35 | +1.99 [−7.45, 11.43] | .404→.447 | .430→.503 | .98 | 44.89 |
| Resource-second | V8 continuous | 20.67→25.03 | +4.48 [.07, 8.89] | .553→.529 | .386→.479 | 1.00 | 69.89 |
| Resource-second | strict no-cross | 20.47→19.19 | −2.35 [−15.99, 11.29] | .521→.538 | .354→.488 | .90 | 63.50 |

Continuous − no-cross 的 reward DiD 在 Resource-first 为 `+0.48 [−12.60, 13.56]`，Resource-second 为 `+6.83 [−8.85, 22.51]`。均不能在 5-run 设计下排除零。

### 最终判断

1. Resource-second 的 continuous V8 有正的自身 DiD，并在 post-switch reward/agreement/joint 上优于 no-cross；这是值得扩大样本验证的信号，但目前 continuous-minus-control 区间很宽。
2. 两个 role 的 preference MAE 都在 switch 后上升；Resource-first estimate 甚至朝错误方向移动。因此 reward 信号没有被正确 belief tracking 的机制证据支持。
3. away-from-prior switch 虽消除了“MAE 机械下降”，却保留原 issue ranking，很多合理交易仍然可接受，公开行为变化不足。该 profile 对 latent preference 的可辨识性仍有限。
4. 因此当前论文不能主张“continuous belief 已在 opponent preference change 后正确恢复”；可以主张我们提供了会发现这一失败的 causal evaluation protocol。

## 十、建议冻结的论文主张与下一版方法

### 可以写入主文的主张

- **Belief usability，而非 belief narration**：belief 进入 executable planner，动作被锁定，LLM 只负责语言实现；
- **Factorized causal audit**：preference 与 response policy 分解，same-state action flip、wrong/shuffled/oracle/no-cross 干预直接测 belief→action；
- **Reliability-bounded planning**：Resource-first 的 V8 相对 V5 有稳定增益，同时将 V7-wrong 的 catastrophic loss 显著压缩；
- **Negative but important finding**：stationary payoff、acceptance calibration 和漂亮的 belief JSON 均不能证明 opponent modeling；必须同时要求 latent calibration、intervention sensitivity 与 change recovery。

### 不能写入主文的主张

- V8 在所有 role 普遍提高 self reward；
- continuous belief 普遍优于 frozen/no-cross；
- 当前 preference posterior 已被校准；
- reward 增益由正确 preference tracking 导致。

### 下一版 V9 不应先改 planner 权重

V9 应先改实验与 observation model：

1. 构建**行为可辨识、仍有交易空间**的多 profile resource suite，而不是只在一个 ratio 上切换；profile 要预先用 oracle policy 验证会产生不同最优 offer/accept boundary；
2. 使用主动 probe，使两个候选 preference hypotheses 对同一报价给出不同预测响应；以 expected information gain 选择 probe；
3. 只在连续多个低 predictive-probability response 出现时触发 preference change-point posterior，不能仅 reset 到 uniform；应维护 `old regime/new regime` mixture；
4. 以 held-out utility profiles 测 recovery latency、signed estimate movement 和 decision regret；planner reward 作为次指标；
5. 成功 gate：continuous 相对 no-cross 在至少两个 held-out profiles 上降低 post-switch decision regret，并同时改善 signed belief error。未过 gate 就不进入大规模训练。

这使当前 story 从“我们已经解决 belief learning”收敛为更可信的贡献：**我们提出 belief-usable、reliability-aware 的 negotiation architecture，并给出一套能区分 belief narration、belief influence、belief correctness 和 non-stationary adaptation 的因果评测方法；现有 LLM opponent belief 在最后两项上仍有明确缺口。**
